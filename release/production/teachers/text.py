"""文本决策教师：把"末位那串数字里的候选字母"折成一份可蒸馏的选项分布。

【做什么】
    输入是渲染层已经拼好的一段请求文字与候选字母清单，输出是每个字母的概率（和为 1）。
    背后跑的是一个本地预训练 decoder（本项目教师 = StartLux-Decision-4B）：请求走一次
    前向，取"最后一个真位置"的下一词分数，只留候选字母那几列，按题型倍数缩放后再折成
    份额。全程一个字也不生成——与学生读出层同一套口径，伪标才可迁移。
    配套三件：`TeacherStats` 记账（前向/在线/命中次数，"零前向"断言的数据源）、
    `load_decision_config` 直读权重自带的 decision_config.json、`check_letters` 校接缝。

【怎么做】
    三层。最外是 `TextTeacher`：先问缓存（键 = 模型名+请求+候选，见 cache.py），命中
    直接返回、一次都不算；未命中才叫后端算，算完回写缓存。中间是后端
    `HfCausalLMBackend`：把文字变成 id、跑一次前向、按每行真实长度取回末位分数、查字母
    列，交出逐字母分数；右补空洞时取的是 `长度-1` 那一格，绝不当成最后一列（与
    `sys1/decision/readout.py` 同一约定）。最里是 `load_decision_config` 读来的
    `temperature_by_type`：分数同除一个正数再折份额——同除正数不改变谁排第一，
    只改变份额的胖瘦，这个单调性是校准叙事的地基。
    兜底路径 `score_options_generative`：走 answer 协议让教师真生成十几个符号，
    抽出的确定答案折成近 one-hot 分布；抽不出来就重试一轮，两轮都失败返回 None。

【为什么】
    判别式读出对教师与学生"看同样的东西"是硬要求：生成出来的措辞会把教师自己的模板
    偏好混进伪标，学生学到的就不是判断而是复述。被否方案一：让教师自由作答再用文本
    解析器认字母——长句里的 "A" 满天飞，解析器只能靠猜，且每次多花几十倍符号预算；
    保留生成式兜底只为拿不到 logits 的场合与协议自检。
    被否方案二：把 logits 整块转成 float32 再 softmax——二十多万词表下 (B,T,V) 一块
    就是吉字节级，白算 vocab 倍；本层先按行取末位、再只在 (B,V) 上折份额。
    被否方案三：左补空洞以便只取最后一列——softmax 注意力有掩码护着，但线性注意力/
    因果卷积会把前面的空洞算进状态，读数被污染；右补 + 逐行取位是唯一稳妥的写法。
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .cache import DistCache
from .protocol import extract_answer, hard_distribution, retry_prompt

__all__ = [
    "HfCausalLMBackend",
    "TeacherStats",
    "TextTeacher",
    "check_letters",
    "load_decision_config",
    "softmax_over",
]

# 候选字母表（与 P1 渲染层同序：字母位置 i ↔ order[i]），最多 26 个候选。
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# decision_config.json 缺席时的落点：1.0 = 不缩放，并如实标未标定（与 P1 temperature 同口径）。
UNCALIBRATED_TEMPERATURE = 1.0
# StartLux/Qwen 系模板的思考关闭后缀：读点位必须落在它之后（jevfmt 同源常量）。
THINK_OFF = "<think>\n\n</think>\n\n"
# 缓存行里"这行是谁算的"标记：本地后端 / 生成式兜底 / 外部产标。
SOURCE_LOCAL = "local_model"
SOURCE_GENERATIVE = "generative_answer"
# 渲染层与教师共用的那句规矩（模板要求 system 行时用它）
SYSTEM_LINE = "Apply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only."


@dataclass
class TeacherStats:
    """教师用量账本：前向了几次、在线接口打了几次、缓存中了几次。

    白话：给教师装一个计价器。训练和测试只看这几个数，就知道"重复请求有没有真的
    少走推理"，不用猜。
    """

    forward_calls: int = 0
    api_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    generations: int = 0

    def as_dict(self) -> dict[str, int]:
        """摊成普通字典（写进 run 的 metrics 行；dataclass 直接进 json 会带类名）。"""
        return asdict(self)

    def reset(self) -> None:
        """清零全部计数——单测靠它把"这一段没有前向"钉死。"""
        for name in self.__dataclass_fields__:
            setattr(self, name, 0)


@dataclass
class DecisionSettings:
    """权重目录里 decision_config.json 的读法：题型倍数 + 字母 token 号 + 原文。"""

    temperature_by_type: dict[str, float] = field(default_factory=dict)
    letter_token_ids: tuple[int, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)
    path: str = ""

    def temperature_for(self, qtype: str | None) -> tuple[float, bool]:
        """按题型取倍数，返回 `(倍数, 是否真标定过)`——查不到就 1.0 并明说是没标定。

        白话：教师随权重发布了一份"每类题该除多少"的小抄。小抄里有就照用，没有就
        按原样用（除 1），并把"这类没标定"这件事写在结果里，不假装调过。
        """
        if qtype is None or qtype not in self.temperature_by_type:
            return UNCALIBRATED_TEMPERATURE, False
        value = float(self.temperature_by_type[qtype])
        if not math.isfinite(value) or value <= 0.0:
            return UNCALIBRATED_TEMPERATURE, False
        return value, True


def load_decision_config(model_path: str | Path, *, filename: str = "decision_config.json") -> DecisionSettings:
    """读教师权重目录里的 decision_config.json；文件缺失返回"全未标定"的空设置。

    白话：教师包里自带一张小抄（每类题的缩放倍数、26 个字母的 token 号）。这里把它
    取出来；万一没有也不报错——照常打分，只是把所有题都当"没标定"处理。
    """
    path = Path(model_path)
    candidate = path / filename
    if not candidate.is_file():
        return DecisionSettings(path=str(path))
    raw = json.loads(candidate.read_text(encoding="utf-8"))
    temps = {str(k): float(v) for k, v in (raw.get("temperature_by_type") or {}).items()}
    letters = tuple(int(v) for v in (raw.get("letter_token_ids") or []))
    return DecisionSettings(temperature_by_type=temps, letter_token_ids=letters, raw=raw, path=str(candidate))


def softmax_over(scores: dict[str, float], *, temperature: float = 1.0) -> dict[str, float]:
    """把"逐候选的分数"折成份额；倍数必须是正数，同除一个正数不改变谁排第一。

    白话：给每个候选的底气做指数放大，再摊成总和为 1 的份额。倍数越大份额越平、
    越小越陡；但无论多大，第一名还是第一名——这是"缩放只调信心、不改答案"的硬保证。
    """
    if not scores:
        raise ValueError("scores 不能为空")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature 必须是有限正数，实得 {temperature!r}")
    keys = list(scores)
    values = [float(scores[k]) / temperature for k in keys]
    top = max(values)
    weights = [math.exp(v - top) for v in values]  # 先减去最大值：防指数上溢，折出来的份额不变
    total = sum(weights)
    return {k: w / total for k, w in zip(keys, weights)}


def check_letters(tokenizer: Any, *, expected: Sequence[int] | None = None) -> list[int]:
    """26 个字母必须"独占一格且编号互不相同"，返回字母 token 号表；不合格直说哪个字母。

    白话：读答案的位子只有字母这一个字符，所以先要确认每个字母在它的文字切分规则里独占一格。
    26 个都合格才给出对照表；教师小抄里写了号的话，还要两相对上才算接缝没问题。
    """
    ids: list[int] = []
    for letter in LETTERS:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"字母 {letter!r} 不是单 token（实得 {encoded}），末位读出不可用")
        ids.append(int(encoded[0]))
    if len(set(ids)) != len(ids):
        raise ValueError(f"字母 token 号有重复：{ids}")
    if expected is not None and list(expected) != ids:
        raise ValueError(f"decision_config 的字母 token 号与分词器实算不符：期望 {list(expected)}，实得 {ids}")
    return ids


class HfCausalLMBackend:
    """transformers 因果模型后端：交出一段请求，收回逐候选字母的末位分数。

    白话：这一层只干"算分"，不管缓存也不管记账。它把请求送进模型，从写完最后一步
    留下的那串数字里挑出字母对应的那几列，交回给教师去折份额。
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_id: str,
        *,
        device: str = "cpu",
        apply_chat_template: bool = True,
    ) -> None:
        if tokenizer is None or model is None:
            raise ValueError("model 与 tokenizer 都必须给（本后端不负责去找权重）")
        # 约定右补空洞：空洞排在真符号之后，才不会污染要读的那一位（与 P1 读出同口径）
        side = getattr(tokenizer, "padding_side", "right")
        if side != "right":
            raise ValueError(f"padding_side 必须是 right（左补会让线性注意力读到空洞），实得 {side!r}")
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.device = device
        self.apply_chat_template = apply_chat_template
        self.letter_ids = check_letters(tokenizer)  # 装配即校接缝：字母读不出就别想往下算
        self._token_cache: dict[str, int] = {}

    # ── 装配 ────────────────────────────────────────────────────────────
    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        model_id: str | None = None,
        device: str = "cpu",
        dtype: str = "bfloat16",
    ) -> "HfCausalLMBackend":
        """从本地权重目录装载教师（默认 CPU：MPS 常被一阶段长跑占着，不抢）。

        白话：给个装着权重的文件夹就装好教师；名字不给就用文件夹名兜底，事后能对上是
        谁算的哪一行。想用显卡就传 device="mps"，但那属于抢资源，得自己先确认空闲。
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = str(model_path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token  # 补空洞必须有符号，不能拿真符号凑
        model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype), device_map={"": device})
        model.eval()
        return cls(model, tokenizer, model_id or _default_model_id(path), device=device)

    # ── 读分 ────────────────────────────────────────────────────────────
    def last_letter_logits(self, prompts: Sequence[str], option_keys: Sequence[str]) -> list[dict[str, float]]:
        """一批请求一次前向，返回每行"候选字母的末位下一词分数"（未折份额的原值）。

        白话：把几道题一起喂进去，每道题在写完最后一步时都留下一串可学的数字；这里
        只把候选字母对应的那几列挑出来交出去。哪道题补了空洞，就按它真写了几个字去
        取那一格，绝不取到空洞上。
        """
        import torch

        items = [self._prepare(p) for p in prompts]
        if not items:
            return []
        tokens = [self._key_token(k) for k in option_keys]
        encoded = self.tokenizer(items, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        input_ids = encoded["input_ids"]
        mask = encoded["attention_mask"]
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=mask, use_cache=False)
            logits = out.logits                    # (B,T,V) 保持原精度：整块转 float32 会白占一吉
            lengths = mask.sum(dim=1)              # 右补空洞时：真末位 = 长度 - 1
            rows = torch.arange(input_ids.shape[0], device=logits.device)
            last = logits[rows, lengths - 1, :]    # (B,V) 只留要读的那一行
            picked = last[:, tokens].float()       # (B,k) 候选字母列
        return [{k: float(v) for k, v in zip(option_keys, row)} for row in picked]

    def letter_logprobs(self, prompt: str, option_keys: Sequence[str]) -> dict[str, float]:
        """单条请求的末位候选对数分数（log-softmax 之后）——教师主路的直接输入。

        白话：一次问一道题，拿回"每个候选字母有多可能被写下来"的对数分数。
        折份额的活儿不在这里做，教师那边还要按题型倍数缩放再 softmax。
        """
        import torch

        scores = self.last_letter_logits([prompt], option_keys)[0]
        values = torch.tensor([scores[k] for k in option_keys], dtype=torch.float32)
        logp = torch.log_softmax(values, dim=-1)
        return {k: float(v) for k, v in zip(option_keys, logp)}

    def generate_text(self, prompt: str, *, max_new_tokens: int = 12, temperature: float = 0.0) -> str:
        """少量生成一段回复（只服务 answer 协议兜底路径，主路一个字都不生成）。

        白话：非要教师"开口说一句话"时才用这里，且把长度压到十几个符号——我们的目的
        只是拿到尖括号里那个值，多说一个字都是白花预算。
        """
        import torch

        text = self._prepare(prompt)
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
        with torch.no_grad():
            generated = self.model.generate(**encoded, **kwargs)
        new_ids = generated[0][encoded["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    # ── 私有：请求成形与候选 token ──────────────────────────────────────
    def _prepare(self, prompt: str) -> str:
        """把渲染层的扁平文字套进对话模板并留好思考关闭后缀（无模板时原样加后缀）。"""
        if THINK_OFF in prompt:
            return prompt  # 调用方已自行成形，不再套第二遍
        template = getattr(self.tokenizer, "chat_template", None)
        if not (self.apply_chat_template and template):
            return prompt + THINK_OFF
        messages = _as_messages(prompt)
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except (TypeError, ValueError):
            # 模板不认 enable_thinking（非 StartLux/Qwen 系模板）：退回不带该参数的写法
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _key_token(self, key: str) -> int:
        """候选字母 → 单 token 号；多 token 直接拒绝（读的是末位一列，拆开就不是一回事）。"""
        cached = self._token_cache.get(key)
        if cached is not None:
            return cached
        encoded = self.tokenizer.encode(str(key), add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"候选 {key!r} 在教师词表里不是单 token（{encoded}）——"
                "末位读出只能读独占一格的代号；请检查渲染层的字母表或改用字母键"
            )
        token = int(encoded[0])
        self._token_cache[str(key)] = token
        return token


class TextTeacher:
    """文本教师门面：缓存 → 记账 → 打分，接口就是 `score_options(prompt, option_keys)`。

    白话：训练和评测那边只认这一个入口。它先翻账本看算过没有，没算过才请后端去算，
    算完把结果抄进账本；用得越多，被重复计算的东西就越少。
    """

    def __init__(
        self,
        backend: Any,
        *,
        cache: DistCache | None = None,
        stats: TeacherStats | None = None,
        settings: DecisionSettings | None = None,
        default_temperature: float = UNCALIBRATED_TEMPERATURE,
    ) -> None:
        self.backend = backend
        self.model_id = getattr(backend, "model_id", "") or "unknown-text-teacher"
        self.cache = cache
        self.stats = stats if stats is not None else TeacherStats()
        self.settings = settings if settings is not None else DecisionSettings()
        self.default_temperature = default_temperature

    # ── 主路 ────────────────────────────────────────────────────────────
    def score_options(
        self,
        prompt: str,
        option_keys: Sequence[str],
        *,
        qtype: str | None = None,
        temperature: float | None = None,
        use_cache: bool = True,
    ) -> dict[str, float]:
        """请求 + 候选字母 → 份额分布（和为 1）；命中缓存时一次前向都不发。

        白话：问"这几个候选里教师觉得哪个最可能"。先在算过的结果里找，找到就原样交出
        （不消耗任何算力）；找不到才真去跑一次模型，然后把这份答案记进账本备查。
        """
        keys = [str(k) for k in option_keys]
        if not keys:
            raise ValueError("option_keys 不能为空")
        cached = self._cached(prompt, keys, use_cache)
        if cached is not None:
            return cached
        dist = self._forward(prompt, keys, qtype=qtype, temperature=temperature)
        self._store(prompt, keys, dist, source=SOURCE_LOCAL)
        return dist

    def score_messages(
        self,
        messages: Sequence[dict[str, str]],
        option_keys: Sequence[str],
        **kwargs: Any,
    ) -> dict[str, float]:
        """吃 messages（system + user）的写法：压成扁平文字后与 `score_options` 同路。

        白话：有的渲染层交出来的是两段对话而不是整段文字，这里首尾接成一整段再走同
        一条打分路，保证两种写法算出的份额完全一样。
        """
        return self.score_options(_flatten(messages), option_keys, **kwargs)

    def score_batch(
        self,
        items: Sequence[tuple[str, Sequence[str]]],
        *,
        qtype: str | None = None,
        temperature: float | None = None,
        use_cache: bool = True,
    ) -> list[dict[str, float]]:
        """一次调用打一批（候选集必须一致）：先挑出没算过的，剩余的一趟前向补齐。

        白话：把一堆题打包问，教师一趟就都答了；已经算过的题直接从账本里抄，
        不占这一趟的位子。
        """
        if not items:
            return []
        key_lists = [tuple(sorted(str(k) for k in keys)) for _, keys in items]
        if len(set(key_lists)) != 1:
            raise ValueError("批量打分要求所有请求的候选集一致（不同候选集请分组分别调用）")
        keys = [str(k) for k in items[0][1]]
        results: list[dict[str, float] | None] = [None] * len(items)
        todo: list[int] = []
        for i, (prompt, _) in enumerate(items):
            hit = self._cached(prompt, keys, use_cache) if use_cache else None
            if hit is not None:
                results[i] = hit
            else:
                todo.append(i)
        if todo:
            import torch

            rows = self.backend.last_letter_logits([items[i][0] for i in todo], keys)
            temp = self._temperature_for(qtype, temperature)
            for i, row in zip(todo, rows):
                values = torch.tensor([row[k] for k in keys], dtype=torch.float32)
                logp = torch.log_softmax(values, dim=-1)
                dist = softmax_over({k: float(v) for k, v in zip(keys, logp)}, temperature=temp)
                self.stats.forward_calls += 1  # 一次前向打了多行：按行记，与逐条调用口径可比
                results[i] = dist
                self._store(items[i][0], keys, dist, source=SOURCE_LOCAL)
        return [r for r in results if r is not None]

    # ── 兜底：answer 协议生成式 ─────────────────────────────────────────
    def score_options_generative(
        self,
        prompt: str,
        option_keys: Sequence[str],
        *,
        qtype: str = "choice",
        max_new_tokens: int = 12,
        mass: float = 0.9,
        use_cache: bool = True,
    ) -> dict[str, float] | None:
        """让教师按 answer 协议真生成一个值再折分布；抽不出来重试一轮，仍败返回 None。

        白话：主路读不出分数的场合（例如只能拿到文字的接口）才走这里。第一次没给出
        能认的答案，就补一句更硬的格式提醒再问一次；两次都不成就算了，交回 None，
        调用方按缺失处理——绝不编一个份额出来。
        """
        keys = [str(k) for k in option_keys]
        cached = self._cached(prompt, keys, use_cache, model_id=f"{self.model_id}#gen")
        if cached is not None:
            return cached
        for attempt in (0, 1):
            ask = prompt if attempt == 0 else retry_prompt(prompt, keys, qtype=qtype)
            text = self.backend.generate_text(ask, max_new_tokens=max_new_tokens)
            self.stats.generations += 1
            answer = extract_answer(text, qtype=qtype, option_keys=keys)
            if answer is None:
                continue  # 不可解析：第二轮补一句更硬的格式提醒（只加话术，不改题目）
            try:
                dist = hard_distribution(answer, keys, mass=mass)
            except ValueError:
                continue  # 抽出的值不在候选集里：同样算没答上，交给重试/None
            self._store(prompt, keys, dist, source=SOURCE_GENERATIVE, model_id=f"{self.model_id}#gen")
            return dist
        return None

    # ── 私有：缓存三面与真算 ────────────────────────────────────────────
    def _cached(self, prompt: str, keys: Sequence[str], use_cache: bool, *, model_id: str | None = None) -> dict[str, float] | None:
        """查缓存：命中记一笔 hit 并原样交出，未命中记一笔 miss（计数是零前向断言的源）。"""
        if self.cache is None or not use_cache:
            return None
        hit = self.cache.lookup(model_id or self.model_id, prompt, keys)
        if hit is None:
            self.stats.cache_misses += 1
            return None
        self.stats.cache_hits += 1
        return {k: float(hit[k]) for k in keys if k in hit} or hit

    def _store(self, prompt: str, keys: Sequence[str], dist: dict[str, float], *, source: str, model_id: str | None = None) -> None:
        """回写缓存（没挂缓存就什么都不做——教师本身不假设一定要有账本）。"""
        if self.cache is None:
            return
        self.cache.record(
            model_id=model_id or self.model_id, prompt=prompt, option_keys=keys, dist=dist, source=source
        )

    def _forward(self, prompt: str, keys: list[str], *, qtype: str | None, temperature: float | None) -> dict[str, float]:
        """真跑一次前向并折份额（缓存未命中的唯一出口，计数在这里加）。"""
        self.stats.forward_calls += 1
        logprobs = self.backend.letter_logprobs(prompt, keys)
        temp = self._temperature_for(qtype, temperature)
        return softmax_over({k: float(logprobs[k]) for k in keys}, temperature=temp)

    def _temperature_for(self, qtype: str | None, explicit: float | None) -> float:
        """倍数优先级：调用方明说 > 教师小抄（按题型） > 构造默认值。"""
        if explicit is not None:
            return float(explicit)
        if qtype is not None and self.settings.temperature_by_type:
            value, calibrated = self.settings.temperature_for(qtype)
            if calibrated:
                return value
        return self.default_temperature


def _flatten(messages: Sequence[dict[str, str]]) -> str:
    """把 messages 首尾接成一段文字（system 在前、正文紧随，中间只隔一个换行）。"""
    return "\n".join(str(m.get("content", "")) for m in messages)


def _as_messages(prompt: str) -> list[dict[str, str]]:
    """扁平文字还原成 messages：认得出渲染结构就补 system 行，否则整段只当 user。

    渲染层交出的是一整段文字，对话模板却要吃 role/content 两段。固定 system 行是
    渲染层与教师共用的那句规矩；文字里已经有它，模板拼出来才与训练时逐字一致。
    """
    head = prompt.split("\n", 1)[0]
    rendered = head.startswith(("Evidence:", "Question:", "Options:")) or "\nQuestion:" in prompt
    if rendered:
        return [{"role": "system", "content": SYSTEM_LINE}, {"role": "user", "content": prompt}]
    return [{"role": "user", "content": prompt}]


def _default_model_id(path: str | Path) -> str:
    """没给 model_id 时用"目录名@分片规模指纹"兜底，保证换权重必然换键。"""
    p = Path(path)
    shards = sorted(p.glob("*.safetensors"))
    stamp = f"{len(shards)}-{shards[0].stat().st_size}" if shards else "no-weights"
    return f"{p.name}@{stamp}"
