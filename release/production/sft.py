"""p2-05 SFT 栈：载重 backbone + LoRA 读出位 CE + 教师伪标 soft 混合（本地开发/CPU 冒烟档）。

【做什么】
    把三件现成的东西接成一条训练路：数据从 p2-03 的 registry 口取（`sys1.eval.run.load_axis_records`），
    教师分布只从 p2-02 的 parquet 缓存口取（`DistCache`，训练循环一次教师前向都不发），模型从 p2-01 的
    载重接缝取（`production.assets.load_backbone`）。产出的不是"会写字的模型"而是"会报份额的模型"：
    损失只在读出位（每行真末位）对 26 枚字母里的候选列计交叉熵，非读出位一律屏蔽。对外一个 CLI：
    训练、预热伪标缓存（`--prefill-pseudo`）、把适配器按模型目录形态落盘。

【怎么做】
    ① 装配：一行 registry 信封 → `decision/render.from_systemone` 排版 → 同一份 row 既产"教师请求文字"
       （缓存键的原料）又产"学生编号串"（对话模板 + 思考关闭），字母序 `order[i] ↔ LETTERS[i]` 两侧共用，
       于是"教师答的候选"与"学生读的列"必然同序。hard 标签走 p2-02 的 `hard_distribution`（不自己造 one-hot），
       soft 走 `DistCache.lookup`；两者按 `mix_soft` 配比线性混合，查不到伪标的行退成 hard-only 并计入
       `soft_coverage`（覆盖率是事实，不当成 0.5）。
    ② 损失：只做一件事——调 `sys1.decision.readout.option_scores` 取末位×字母行，再对 (B,k) 做
       soft 交叉熵；批内候选数不齐时把未上榜列填 -inf 参与归一、再在乘权重前把该列归零（否则 0×-inf=nan）。
       "取哪一格、读哪几列"这两条口径全部留在 decision/ 里，本文件不重抄一遍（tests -k loss_reuse 有 grep 门）。
    ③ 可训部分：本域自带的 `LoRALinear` 把 qkvo 四枚投影包成"冻结底座 + 低秩旁路"，A 用 LoRA 论文的
       kaiming(√5) 初值、B 全零（起点等于底座），scaling=α/r=2.0 与 StartLux 超参照抄；旁路有两条实现——
       `torch`（普通乘加，自动求导图自己算梯度）与 `kernel`（走 p2-13 的 `ascend.kernels.lora_kernel`，
       前向 `apply`、反向 `backward`，由 `ascend_target` 决定 cpu/ascend），两条路的数与梯度都对得上尺子。
       适配器落盘用 peft 的键名（`base_model.model.<路径>.lora_A.default.weight`），下游换回 peft 也能读。
    ④ 步长与调度：AdamW(lr 1e-4, wd 0) + 5% warmup + cosine + 梯度裁剪 1.0 + accum 微批累加，
       全部照抄 StartLux `finetune/finetune_lora.py`（同构可比）；批规模按"补齐后的总字数"分桶，
       与它 `batches(max_tokens)` 的口径一致，910B 折算只改这两个数。

【为什么】
    被否方案一：整段交给 trl/peft 的 SFTTrainer——那是"逐符号自回归交叉熵"，与本项目"只读末位字母份额"
        的读点完全不同，用它等于另立一套损失口径，P1 的同构证据（同一份读出代码）当场作废。
    被否方案二：训练循环里现算教师分布——伪标必须全走缓存（父设计 D9/本域 design），在线补算会把
        4B 教师的耗时算进训练预算，且同一条请求两次跑出不同份额时无法归因。
    被否方案三：LoRA 只做 peft 转发、不给 kernel 开关——p2-13 交付的 `lora_asc` 就没有训练侧消费者，
        云端换自研栈时要临时补一层，反而更容易和 torch 路漂移；两条路同处一文件、同一份参数张量，
        对拍才有意义。
    被否方案四：把 backbone 前向也换成 p2-13 内核——本地既无昇腾卡，p2-13 的 GDN 反向还挂在 partial
        例外条上（见其 spec），硬换会得到一条"本地跑不通、云端也没法对拍"的死路；故 kernel 开关
        目前只接管 SFT 自己新增的 LoRA 旁路，backbone 前向仍是 HF 实现，正式档在 C5 按底座决策整栈切换。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# ── 上游现成接口（本域一律"用"不"造"：registry 口 / 教师缓存口 / 载重接缝 / 决策读出 / 方言件）──
from ascend.kernels import ascend_env, lora_kernel                     # p2-13 算子口
from production.assets import load_backbone                            # p2-01 载重接缝
from production.teachers.cache import DistCache, make_key              # p2-02 教师缓存口
from production.teachers.protocol import hard_distribution             # p2-02 硬标签折法
from sys1.decision.readout import option_scores                        # P1 读出（冻结件）
from sys1.decision.render import (                                     # P1 渲染（冻结件）
    LETTERS,
    MAX_OPTIONS,
    RENDER_VERSION,
    from_systemone,
    option_order,
    prompt_text,
    render as render_messages,
)
from sys1.eval import run as eval_run                                  # p2-03 registry 口
from sys1.runs import new_run                                          # run 记录本

__all__ = [
    "ADAPTER_CONFIG_NAME",
    "ADAPTER_FILE",
    "AsmStats",
    "BACKENDS",
    "BACKEND_KERNEL",
    "BACKEND_TORCH",
    "BASELINE_LORA",
    "Batch",
    "CONFIG_FILE",
    "DECISION_CONFIG_FILE",
    "DEFAULT_HARD_MASS",
    "DEFAULT_TEACHER_MODEL_ID",
    "DTYPES",
    "LOADERS",
    "LOADER_MINIMAL",
    "LOADER_SEAM",
    "LORA_TARGET_LEAVES",
    "LoRALinear",
    "README_NAME",
    "SCAFFOLD_SUFFIX",
    "SFT_VERSION",
    "Sample",
    "Student",
    "TRAINABLE_NOTE",
    "assemble",
    "bucket_batches",
    "build_parser",
    "build_student",
    "collate",
    "cosine_lambda",
    "decision_config_payload",
    "default_cfg",
    "enable_gradient_checkpointing",
    "encode_record",
    "freeze_all_but_lora",
    "inject_lora",
    "load_adapter",
    "load_config",
    "load_quality_records",
    "lora_delta_kernel",
    "lora_delta_torch",
    "main",
    "merge_lora_weights",
    "mix_targets",
    "prefill_pseudo",
    "prompt_index",
    "readout_ce_loss",
    "save_adapter",
    "scaffold_teacher",
    "set_backend_all",
    "soft_lookup",
    "student_forward",
    "train",
    "write_model_dir",
]

#: 本次交付的 SFT 版本号（进 decision_config.json，与 P1 s2 的 `dmlaya_s2_sft_v1` 分家）
SFT_VERSION = "dmlaya_p2_sft_v1"

#: 旁路实现的两档取值；`kernel` 只可能经 p2-13 的入口件，`torch` 是参照与回退
BACKEND_TORCH = "torch"
BACKEND_KERNEL = "kernel"
BACKENDS = (BACKEND_TORCH, BACKEND_KERNEL)

#: LoRA 基线超参——照抄 StartLux-Decision/finetune/finetune_lora.py（仅超参参考，D11：其权重不进训练）
BASELINE_LORA: dict[str, Any] = {
    "r": 16, "alpha": 32, "dropout": 0.05,
    "lr": 1e-4, "warmup_ratio": 0.05, "weight_decay": 0.0, "max_grad_norm": 1.0,
    "epochs": 2.0, "max_tokens": 16384, "accum": 4, "max_length": 8192,
}

#: 只挂 language 侧这四枚投影（Qwen3.5 的整注意力层才有；线性注意力层与视觉塔一律不动）
LORA_TARGET_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj")

#: 教师默认身份（C3 产标用的真教师号）；本地冒烟显式换成带 `#scaffold-cpu` 的袖珍号，键天然分家
DEFAULT_TEACHER_MODEL_ID = "StartLuxAI/StartLux-Decision-4B"
SCAFFOLD_SUFFIX = "#scaffold-cpu"

#: hard/soft 混合默认配比（spec：默认 50/50 可配）——`mix_soft` 是 soft 分量的权重
DEFAULT_MIX_SOFT = 0.5
#: 人工硬标签的押注质量（复用 p2-02 的折法，0.9 即"绝大部分押它、余下一点均摊"）
DEFAULT_HARD_MASS = 0.9

#: 模型目录形态（与 P1 冻结的 sys1/model.py 布局对齐，下游 predict 不感知差异）
ADAPTER_FILE = "adapter.safetensors"
ADAPTER_CONFIG_NAME = "adapter_config.json"
CONFIG_FILE = "config.json"
DECISION_CONFIG_FILE = "decision_config.json"
README_NAME = "README.md"

#: 写进 run notes 的口径声明——本地半场永远是"替身 + 冒烟"，正式档在云端 C5
TRAINABLE_NOTE = "CPU 替身口径（本地开发档）；910B 正式档（gate08b/scaling 全量）待 C5"


# ================================================================ 工作项 A：数据装配
@dataclass(frozen=True)
class Sample:
    """一条已装配好的训练样本：编号串 + 字母序 + hard/soft/mixed 三份分布 + 教师请求文字。

    白话：把一道题按学生要读的样子和教师被问的样子各备一份，再把"该押几成"的三个版本
    （人工的、教师算过的、两者掺起来的）一并带上；进批次时只带这三个数和编号，
    文字留着是为了下次还能对上同一道题目。
    """

    sample_id: str
    ids: tuple[int, ...]
    letters: tuple[str, ...]          # 与字母列同序的代号（A/B/C…），长度 = k
    order: tuple[str, ...]            # letters[i] 代表的候选代号（continue/human_review/…）
    hard: tuple[float, ...]           # 人工硬标签折出的分布（和为 1）
    soft: tuple[float, ...] | None    # 教师伪标分布；缓存没查到就是 None
    target: tuple[float, ...]         # 混合后的训练目标（和为 1）
    qtype: str
    prompt: str                       # 教师请求文字（缓存键的原料，不含对话模板）
    source: str                       # 来自哪个评测集 id（registry 登记名）


@dataclass
class AsmStats:
    """装配台账：编了多少、丢了多少、伪标覆盖到多少、教师缓存命中几回。

    白话：这一趟到底装出多少能用的题、多少题其实没有教师份额可掺，都得当场报数——
    事后看曲线时，"覆盖率两成"和"覆盖率十成"是两件完全不同的事。
    """

    seen: int = 0
    encoded: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    @property
    def soft_coverage(self) -> float:
        """伪标覆盖率 = 命中的行数 / 编成的行数（没有行时按 0 算，不当成满覆盖）。"""
        return self.hits / self.encoded if self.encoded else 0.0

    def as_dict(self) -> dict[str, Any]:
        """摊成能进 run 记录的普通字典（dataclass 原名进 json 会带类名）。

        白话：把这本小账抄成一张能塞进记录本的普通表格——记的东西要能被别人原样
        读回去，留着类名就成了一句"谁知道这数是从哪个本子来的"。
        """
        return {
            "seen": self.seen, "encoded": self.encoded, "skipped": dict(self.skipped),
            "teacher_hits": self.hits, "teacher_misses": self.misses,
            "soft_coverage": round(self.soft_coverage, 6),
        }


def load_quality_records(axis: str = "quality", *, data_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """取训练题：只走 p2-03 的 registry 口，本域不自己找题、不自己造题。

    白话：题目从已经登过账的那份副本里原样读出来。读不到就交空表，绝不为了"凑够一屏"
    补几条假题——哪来的题、钉在哪个版本，registry 那本账已经写死了。

    :param axis: 评测轴名（默认质量轴 = typed-decisions + intern-decision 两集）。
    :param data_dir: 数据根目录覆盖（默认 registry.DATA_DIR）。
    :returns: 统一信封行列表（一题一行）。
    """
    return eval_run.load_axis_records(axis, data_dir=data_dir)


def encode_record(
    record: dict[str, Any],
    tokenizer: Any,
    *,
    cache: DistCache | None = None,
    teacher_model_id: str = DEFAULT_TEACHER_MODEL_ID,
    mix_soft: float = DEFAULT_MIX_SOFT,
    max_length: int = int(BASELINE_LORA["max_length"]),
    hard_mass: float = DEFAULT_HARD_MASS,
) -> Sample | None:
    """一行信封 → 一份样本：渲染、编号、折 hard、查 soft、按比例掺成 target。

    白话：先照 P1 那套版式把题面排出来（学生和教师都读这一份版式，只是包装不同：
    教师读扁平文字、学生读套了壳的编号）。再看人工给了哪一档最多，按 p2-02 的现成折法
    压成"绝大部分押它"的分布；然后拿这道题的扁平文字去教师账本里问一次有没有算过——
    问到就掺，没问到就先用人工那份，并把"这题没教师份额"如实记一笔。

    :param record: registry 的统一信封行（含 `sample.state/questions/targets`）。
    :param tokenizer: 学生侧编号器（backbone 自带）。
    :param cache: 教师分布缓存（None 表示这批不用伪标，全部退成 hard-only）。
    :param teacher_model_id: 缓存键里的教师身份（换人必须换键，不许串味）。
    :param mix_soft: soft 分量的权重（0=纯人工，1=纯教师）；越界当场报错。
    :param max_length: 单条编号串的长度上限，超出直接丢（不截成读不到末位的半截题）。
    :param hard_mass: 人工硬标签的押注质量。
    :returns: `Sample`，或 None（该题不适合这条读点：候选超 26、没有人工份额、过长）。
    """
    if not 0.0 <= mix_soft <= 1.0:
        raise ValueError(f"mix_soft 需落在 [0,1]，实得 {mix_soft!r}")
    sample = record.get("sample") or {}
    questions = sample.get("questions") or {}
    if not questions:
        return None
    targets_all = sample.get("targets") or {}
    for qid, spec in questions.items():
        if not isinstance(spec, dict):
            continue
        gold = targets_all.get(qid)
        if not isinstance(gold, dict) or not gold:
            continue                                  # 没有人工份额的题面只能当语料，不能当训练目标
        row = from_systemone(sample.get("state"), spec, qid=qid)
        order = option_order(row)
        if not order or len(order) > MAX_OPTIONS:
            return None                               # 超 26 候选要走 wide 分组，本域先不吞
        letters = tuple(LETTERS[: len(order)])
        vec = [float(gold.get(k, 0.0) or 0.0) for k in order]
        total = math.fsum(vec)
        if total <= 0:
            return None                               # 人工份额没落在在场候选上：与 StartLux 同口径当场弃
        best_letter = letters[max(range(len(order)), key=lambda i: vec[i])]
        hard_map = hard_distribution(best_letter, list(letters), mass=hard_mass)
        hard = tuple(hard_map[k] for k in letters)
        prompt = prompt_text(row, order)
        soft_map = soft_lookup(cache, teacher_model_id, prompt, letters) if cache is not None else None
        soft = None if soft_map is None else tuple(soft_map.get(k, 0.0) for k in letters)
        target = mix_targets(hard, soft, mix_soft)
        ids = _tokenize_prompt(tokenizer, row, order, max_length)
        if ids is None:
            return None
        return Sample(
            sample_id=f"{record.get('id', '?')}#{qid}", ids=ids, letters=letters, order=tuple(order),
            hard=hard, soft=soft, target=tuple(target), qtype=str(row.get("type") or record.get("qtype") or "choice"),
            prompt=prompt, source=str(record.get("task") or ""),
        )
    return None


def soft_lookup(
    cache: DistCache | None, teacher_model_id: str, prompt: str, letters: Sequence[str]
) -> dict[str, float] | None:
    """只查缓存、绝不用在线教师补算：问到就交份额，问不到就交 None。

    白话：拿着"哪个教师、问了什么、给了哪几档"去账本上对门牌号。对上了原样抄回来，
    对不上就明说没查到——训练循环里最忌讳的就是"顺手替教师算一遍"，那笔算力与
    这份份额的出处都会说不清。
    """
    if cache is None:
        return None
    return cache.lookup(teacher_model_id, prompt, list(letters))


def mix_targets(hard: Sequence[float], soft: Sequence[float] | None, mix_soft: float) -> list[float]:
    """按比例掺人工与教师两份份额（soft 缺席就原样交人工那份），返回和为 1 的新列。

    白话：一个说"我押这个"，一个说"我看这几档各占几成"。各听几分由配置说了算，
    掺完再归一成"总和还是一锅汤"，免得掺着掺着总份额变成 1.7 或者 0.4。

    :raises ValueError: 两列长度不等、或配比越界时抛出。
    """
    if not 0.0 <= float(mix_soft) <= 1.0:
        raise ValueError(f"mix_soft 需落在 [0,1]，实得 {mix_soft!r}")
    h = [float(v) for v in hard]
    if soft is None:
        total = math.fsum(h) or 1.0
        return [v / total for v in h]
    if len(soft) != len(h):
        raise ValueError(f"hard/soft 长度须一致，实得 {len(h)} vs {len(soft)}")
    w = float(mix_soft)
    mixed = [(1.0 - w) * a + w * float(b) for a, b in zip(h, soft)]
    total = math.fsum(mixed)
    if total <= 0:
        raise ValueError("掺完的份额总和为 0，无法归一（检查 hard/soft 是否全零）")
    return [v / total for v in mixed]


def _tokenize_prompt(tokenizer: Any, row: dict[str, Any], order: Sequence[str], max_length: int) -> tuple[int, ...] | None:
    """渲染 → 套壳关思考 → 编号；超长就丢（截断会把读点位切掉，比少一条更糟）。"""
    messages, _ = render_messages(row, list(order))
    text = _apply_chat_template(tokenizer, messages)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        return None
    if len(ids) > max_length:
        return None
    return tuple(int(i) for i in ids)


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """套对话模板并显式关思考；模板不认这个开关时退回不带开关的写法（口径与 p2-02 后端一致）。"""
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def assemble(
    records: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    cache: DistCache | None = None,
    teacher_model_id: str = DEFAULT_TEACHER_MODEL_ID,
    mix_soft: float = DEFAULT_MIX_SOFT,
    max_length: int = int(BASELINE_LORA["max_length"]),
    limit: int = 0,
) -> tuple[list[Sample], AsmStats]:
    """把一批信封编成样本表，并把丢弃原因与伪标覆盖率如实记进台账。

    白话：一道一道往里编，编成的进表，编不成的记下是为什么（候选太多、没有人工份额、
    太长、编号器不认）。同时数一遍"多少题真掺到了教师份额"——这个数决定这条曲线
    到底是"师生合训"还是"只有人工标签"，事后必须一句话能答。

    :param records: registry 交出的信封行。
    :param tokenizer: 学生侧编号器。
    :param cache: 教师分布缓存（只读）。
    :param teacher_model_id: 教师身份（缓存键的一部分）。
    :param mix_soft: soft 分量权重。
    :param max_length: 单条长度上限。
    :param limit: 编成条数的上限（0=不限），给 tiny 冒烟封顶用。
    :returns: `(samples, AsmStats)`。
    """
    stats = AsmStats()
    samples: list[Sample] = []
    for record in records:
        stats.seen += 1
        before = (cache.stats["hits"], cache.stats["misses"]) if cache is not None else (0, 0)
        one = encode_record(record, tokenizer, cache=cache, teacher_model_id=teacher_model_id,
                            mix_soft=mix_soft, max_length=max_length)
        if one is None:
            stats.skipped["rejected"] = stats.skipped.get("rejected", 0) + 1
            continue
        if cache is not None:
            after = (cache.stats["hits"], cache.stats["misses"])
            stats.hits += after[0] - before[0]
            stats.misses += after[1] - before[1]
        samples.append(one)
        if limit and len(samples) >= limit:
            break
    stats.encoded = len(samples)
    return samples, stats


@dataclass
class Batch:
    """一次前向要吃的一微批：编号、可见位、目标份额、在场候选列、每行真长度。

    白话：几道题排齐了送进模型——短的那几行尾巴上补了空洞，所以另外带着"每行真写了
    几个字"这本小账，读答案时按它取格；候选少的行，没上榜的列在算分时直接封掉。
    """

    ids: torch.Tensor               # (B,T)
    mask: torch.Tensor              # (B,T) 0/1 可见位
    target: torch.Tensor            # (B,k) 每行和为 1
    keep: torch.Tensor              # (B,k) bool：该行真正在场的候选列
    letter_ids: list[int]           # 前 k 枚字母的编号（与列同序）
    lengths: list[int]              # 每行真符号数（右补空洞约定）
    k: int

    def to_(self, device: str | torch.device) -> "Batch":
        """把五张表搬到指定设备（就地返回自己，方便链式；CPU 冒烟时是空操作）。

        白话：题目、可见位、份额、在场列这几张表得跟模型待在同一个地方，
        不然一开口就"你在东我在西"；搬完把自己交回去，写起来能连着点下去。
        """
        self.ids = self.ids.to(device)
        self.mask = self.mask.to(device)
        self.target = self.target.to(device)
        self.keep = self.keep.to(device)
        return self

    @property
    def tokens(self) -> int:
        """补齐后的字数（吞吐口径的分母，与 StartLux `max_tokens` 同义）。"""
        return int(self.ids.numel())


def bucket_batches(samples: Sequence[Sample], *, max_tokens: int, shuffle: bool, seed: int) -> list[list[int]]:
    """按长度分桶成微批：桶内按最长行算字数，超过 `max_tokens` 就另起一桶。

    白话：把差不多长的题凑成一桌，短桌多塞几个、长桌少塞几个，桌子的总座位数有上限；
    这样补出来的空洞最少，一批真正算的字数也稳住。分完桌子再把桌子顺序打乱，
    免得每轮都从同一桌开始。

    :param samples: 样本表（只用长度信息）。
    :param max_tokens: 一微批补齐后的字数上限。
    :param shuffle: 是否打乱桶序。
    :param seed: 打乱用的种子（同种子必得同一套桶序，冒烟才可复跑）。
    :returns: 桶列表，每桶是样本下标。
    """
    order = sorted(range(len(samples)), key=lambda i: -len(samples[i].ids))
    out: list[list[int]] = []
    cur: list[int] = []
    longest = 0
    for i in order:
        n = len(samples[i].ids)
        if cur and max(longest, n) * (len(cur) + 1) > max_tokens:
            out.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, n)
    if cur:
        out.append(cur)
    if shuffle:
        random.Random(seed).shuffle(out)
    return out


def collate(samples: Sequence[Sample], idx: Sequence[int], *, pad_id: int,
            letter_map: Sequence[int]) -> Batch:
    """把若干样本下标拼成一个微批张量表（右补空洞、目标列按字母序对齐）。

    白话：按最长的那行开格子，每题把自己那串号写在前面、尾巴留空并标成"看不见"；
    份额表按候选数补齐，没到场的列标 false，交给损失那一层封掉。字母编号取前 k 枚，
    顺序与渲染层交回的候选列一模一样。

    :param samples: 样本表。
    :param idx: 进批的下标。
    :param pad_id: 空洞用的编号（用分词器自己的 pad，没有就退回句尾符）。
    :param letter_map: 26 枚字母的编号表（A 在第一个）。
    :returns: `Batch`。
    """
    rows = [samples[i] for i in idx]
    width = max(len(r.ids) for r in rows)
    k = max(len(r.letters) for r in rows)
    ids = torch.full((len(rows), width), int(pad_id), dtype=torch.long)
    mask = torch.zeros_like(ids)
    target = torch.zeros((len(rows), k), dtype=torch.float32)
    keep = torch.zeros((len(rows), k), dtype=torch.bool)
    for j, r in enumerate(rows):
        n = len(r.ids)
        ids[j, :n] = torch.tensor(list(r.ids), dtype=torch.long)
        mask[j, :n] = 1
        c = len(r.letters)
        target[j, :c] = torch.tensor(list(r.target), dtype=torch.float32)
        keep[j, :c] = True
    lengths = [int(mask[j].sum()) for j in range(len(rows))]
    return Batch(ids=ids, mask=mask, target=target, keep=keep,
                 letter_ids=[int(v) for v in letter_map[:k]], lengths=lengths, k=k)


# ================================================================ 工作项 B：读出位损失
def readout_ce_loss(
    last_hidden: torch.Tensor,
    head_weight: torch.Tensor,
    letter_ids: Sequence[int],
    targets: torch.Tensor,
    *,
    keep: torch.Tensor | None = None,
    lengths: Sequence[int] | None = None,
) -> torch.Tensor:
    """读出位软交叉熵：只从每行真末位取候选分数，再按目标份额加权罚分。

    白话：一道题只有写到最后一个字那一刻的心气算数，前面每格都不计分；目标说该押几成，
    押得越准罚得越轻。"取哪一格、读哪几列"这两条口径全交给 decision/ 那把现成的尺子，
    本文件绝不另摆一把，免得两条路各说各话。

    :param last_hidden: `(B,T,d)` 逐位置输出（带图，来自 `student_forward`）。
    :param head_weight: `(vocab,d)` 输出层权重（冻结，调用侧先 detach）。
    :param letter_ids: 前 k 枚字母编号，顺序 = 渲染层字母序。
    :param targets: `(B,k)` 每行和为 1 的目标份额。
    :param keep: `(B,k)` bool；False 的列不参与算分（批内候选数不齐时用）。
    :param lengths: 每行真符号数；None 表示按整行最右一格读。
    :returns: 标量损失（对行取均值）。
    """
    if last_hidden.dim() != 3:
        raise ValueError(f"last_hidden 需为 (B,T,d)，实得 {tuple(last_hidden.shape)}")
    if targets.shape[1] != len(letter_ids):
        raise ValueError(f"目标列数 {targets.shape[1]} 与字母数 {len(letter_ids)} 不符")
    scores = option_scores(last_hidden, head_weight, letter_ids, lengths)      # (B,k)：本域唯一读点
    if keep is not None:
        scores = scores.masked_fill(~keep, float("-inf"))
    logp = torch.log_softmax(scores.float(), dim=-1)
    if keep is not None:
        # 封掉的列必须在相乘前归零：留着 -inf 与 target 的 0 相乘会得 nan（冒烟第一版就栽在这）
        logp = logp.masked_fill(~keep, 0.0)
    return -(targets.float() * logp).sum(dim=-1).mean()


# ================================================================ 工作项 B：LoRA 旁路（双后端）
def lora_delta_torch(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scaling: float) -> torch.Tensor:
    """`torch` 档旁路增量：`scaling · (x @ Aᵀ) @ Bᵀ`，逐位可微，同时充当 kernel 档的尺子。

    白话：把这批行先折进一小片薄空间、再展回来，展回来时顺手乘上下笔力度。
    这是最朴素的写法，快慢与显存都不占优，留着的用处是"另一条路算得对不对，跟它比"。

    :param x: `(..., k)` 行块（末轴须等于 A 的列数）。
    :param a: `(r, k)` 降维垫片。
    :param b: `(n, r)` 升维垫片。
    :param scaling: 力度系数（α/r）。
    :returns: `(..., n)` 增量（与 x 的前置轴同形）。
    """
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    if not flat.is_contiguous():
        flat = flat.contiguous()
    delta = (flat @ a.t() @ b.t()) * float(scaling)
    return delta.view(shape[:-1] + (b.size(0),))


class _KernelLoRADelta(torch.autograd.Function):
    """`kernel` 档旁路：前向转交 `lora_kernel.apply`，反向转交 `lora_kernel.backward`。

    白话：同一件事换个执行的人做——薄垫片还是那两片、力度还是那个数，只是折展这两步
    改请 p2-13 交付的算子件出手。自动求导不认识外人的记账方式，所以要在这里手工把
    "倒着走一遍"的三条链接回去，接错了曲线就会跟 torch 档悄悄分家。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scaling: float, target: str):
        """算增量并留下反向要用的三样原料（行块、A、B）。

        白话：交作业的时候把草稿也一并搁下，回头改分时不用再算一遍——草稿就是那批行
        和两片垫片，改分全靠它们。
        """
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        if not flat.is_contiguous():
            flat = flat.contiguous()
        out = lora_kernel.apply(flat, a, b, float(scaling), None, torch.float32, target)
        ctx.save_for_backward(flat, a, b)
        ctx.scaling = float(scaling)
        ctx.target = target
        ctx.in_shape = shape
        return out.view(shape[:-1] + (b.size(0),))

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        """把上游改分要求转成交件人的三条链：dx（回给行块）、dA、dB。

        白话：倒着走同一条薄空间路，先问"这批行该挪多少"，再分别问两片垫片各该挪多少；
        问出来的数按原样摊回各自的位置。scaling 与 target 不是数，不必回话。
        """
        flat, a, b = ctx.saved_tensors
        if not any(ctx.needs_input_grad[:3]):
            return None, None, None, None, None
        dy2 = dy.reshape(flat.size(0), -1)
        if not dy2.is_contiguous():
            dy2 = dy2.contiguous()
        dx, da, db = lora_kernel.backward(dy2.contiguous(), flat, a, b, ctx.scaling, ctx.target)
        return dx.view(ctx.in_shape), da, db, None, None


def lora_delta_kernel(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scaling: float,
                      *, target: str | None = None) -> torch.Tensor:
    """把 `kernel` 档包成普通函数：先定档（cpu/ascend），再走那段手工接的链。

    白话：调用的人只管说"用算子件做"，具体这机器上该请哪一路由 ascend 那本档决定；
    本地没有卡就落在 cpu 档，同一句开关搬到云端自然改指 ascend 档。
    """
    name = ascend_env.normalize_target(target)
    return _KernelLoRADelta.apply(x, a, b, float(scaling), name)


class LoRALinear(torch.nn.Module):
    """一枚冻结线性层 + 两片可训低秩垫片；旁路可选 `torch` 或 `kernel` 两条实现。

    白话：原来那排数字原地锁死，只在旁边垫两小片薄的。垫片起初是零厚度，所以刚挂上时
    整个模型的行为跟没挂分毫不差；学着学着垫片厚起来，差别才一点点显出来。
    """

    def __init__(
        self,
        base: torch.nn.Module,
        *,
        r: int = int(BASELINE_LORA["r"]),
        alpha: int = int(BASELINE_LORA["alpha"]),
        dropout: float = float(BASELINE_LORA["dropout"]),
        backend: str = BACKEND_TORCH,
        ascend_target: str | None = None,
    ) -> None:
        """按底座形状起两片垫片（A 用 LoRA 论文的 kaiming(√5) 初值、B 全零）。

        :raises ValueError: backend 不在 `BACKENDS` 里，或底座不是带 in/out 尺寸的线性层。
        """
        super().__init__()
        if backend not in BACKENDS:
            raise ValueError(f"backend 只能是 {BACKENDS}，实得 {backend!r}")
        if not hasattr(base, "in_features") or not hasattr(base, "out_features"):
            raise ValueError(f"底座须是线性层，实得 {type(base).__name__}")
        for p in base.parameters(recurse=True):
            p.requires_grad_(False)                        # 底座冻结：只有垫片可训
        self.base = base
        self.r = int(r)
        self.alpha = int(alpha)
        self.dropout_p = float(dropout)
        self.backend = backend
        self.ascend_target = ascend_target
        self.scaling = self.alpha / self.r                 # α/r = 2.0，与 StartLux 超参照抄
        k, n = int(base.in_features), int(base.out_features)
        a = torch.empty(self.r, k, dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5))  # 初值口径与 peft 逐元素一致（探针实测）
        self.lora_a = torch.nn.Parameter(a)                # (r,k)
        self.lora_b = torch.nn.Parameter(torch.zeros(n, self.r, dtype=torch.float32))  # (n,r) 零起点

    # ── 后端开关 ────────────────────────────────────────────────────────
    def set_backend(self, backend: str, *, ascend_target: str | None = None) -> None:
        """就地换档（不重建模块，垫片里的数原样留着）。

        白话：换人不换活儿——两片垫片还是那两片，只是下一步折展交给谁办改了，
        所以可以在同一次训练里中途切档做对拍，不会因为重建而丢了已学到的数。
        """
        if backend not in BACKENDS:
            raise ValueError(f"backend 只能是 {BACKENDS}，实得 {backend!r}")
        self.backend = backend
        if ascend_target is not None:
            self.ascend_target = ascend_target

    # ── 前向 ────────────────────────────────────────────────────────────
    def delta(self, x: torch.Tensor) -> torch.Tensor:
        """只算旁路增量（推理期合并、单测对拍都走这个口，不碰底座）。

        白话：把"垫片贡献了多少"单独拎出来给外面看；训练要的合并结果在 forward 里加，
        这里刻意不加，免得想比一比的时候手上没有干净的数。
        """
        if self.training and self.dropout_p > 0:
            x = F.dropout(x, p=self.dropout_p, training=True)   # 与 peft 同位：垫片入口前掉
        if self.backend == BACKEND_KERNEL:
            return lora_delta_kernel(x, self.lora_a, self.lora_b, self.scaling,
                                     target=self.ascend_target)
        return lora_delta_torch(x, self.lora_a, self.lora_b, self.scaling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """底座结果 + 旁路增量（增量按底座位宽对齐，混精度时不炸类型）。"""
        out = self.base(x)
        return out + self.delta(x).to(out.dtype)

    # ── 推理期 ──────────────────────────────────────────────────────────
    def merged_delta(self, *, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        """把两片垫片合成一块和底座一样大的补丁 `scaling · B @ A`（推理期一次贴上去）。

        白话：训练时分着摆的两小片，上线就不必分着了——这里预先把它们揉成一整块，
        形状跟底座严丝合缝，贴上去之后走的就是普通一层，推理那边不用知道旁路的存在。
        """
        dtype = out_dtype or self.base.weight.dtype
        if self.backend == BACKEND_KERNEL:
            return lora_kernel.merge(self.lora_a, self.lora_b, self.scaling,
                                     out_dtype=dtype, target=self.ascend_target)
        return (self.lora_b.detach().float() @ self.lora_a.detach().float()
                * self.scaling).to(dtype)


def _module_at(root: torch.nn.Module, dotted: str) -> tuple[torch.nn.Module, str]:
    """按点分路径找到父模块与末级属性名（`setattr` 要的是父，不是自己）。"""
    parts = dotted.split(".")
    node = root
    for p in parts[:-1]:
        node = getattr(node, p)
    return node, parts[-1]


def inject_lora(
    root: torch.nn.Module,
    *,
    leaves: Sequence[str] = LORA_TARGET_LEAVES,
    r: int = int(BASELINE_LORA["r"]),
    alpha: int = int(BASELINE_LORA["alpha"]),
    dropout: float = float(BASELINE_LORA["dropout"]),
    backend: str = BACKEND_TORCH,
    ascend_target: str | None = None,
    skip_markers: Sequence[str] = ("visual", "vision"),
) -> dict[str, LoRALinear]:
    """把 language 侧命中的 qkvo 投影原地包成 `LoRALinear`，回名表（顺序按模块名）。

    白话：只在文字那一侧的四扇门上装垫片，看图的那半边和别处一概不动——装多了不只是
    多占地方，还会把"只学怎么报份额"这条边界搅浑。装完给一份"哪几扇装了"的名表，
    后面存盘、换档、报参数量都从这张表出发。

    :param root: 要注入的模块树（一般是 backbone 的 body）。
    :param leaves: 末级属性名白名单（默认 q/k/v/o 四枚）。
    :param skip_markers: 路径含这些字样的一律跳过（视觉塔）。
    :returns: `{模块路径: LoRALinear}`。
    """
    hits = tuple(leaves)
    found = [(n, m) for n, m in root.named_modules()
             if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in hits
             and not any(k in n.lower() for k in skip_markers)]
    if not found:
        raise ValueError(
            f"没找到任何可注入的 {hits}（跳过标记 {tuple(skip_markers)}）；"
            "请确认载的是 language 侧模块树，或改用 build_student 的 body"
        )
    loras: dict[str, LoRALinear] = {}
    for name, mod in found:
        parent, attr = _module_at(root, name)
        wrapped = LoRALinear(mod, r=r, alpha=alpha, dropout=dropout,
                             backend=backend, ascend_target=ascend_target)
        setattr(parent, attr, wrapped)
        loras[name] = wrapped
    return loras


def freeze_all_but_lora(model: torch.nn.Module, loras: dict[str, LoRALinear]) -> dict[str, int]:
    """底座整体冻结、垫片整体放开，回一份"可训/冻结各多少数"的清点。

    白话：先把所有数字都锁上，再只把刚装的两片垫片松开——这样"这次到底动了什么"
    在报告里是一句能核的话，而不是"大概只训了 LoRA 吧"。
    """
    for p in model.parameters(recurse=True):
        p.requires_grad_(False)
    for mod in loras.values():
        mod.lora_a.requires_grad_(True)
        mod.lora_b.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {"trainable": trainable, "frozen": frozen, "lora_modules": len(loras)}


def set_backend_all(loras: dict[str, LoRALinear], backend: str, *, ascend_target: str | None = None) -> None:
    """整张名表一起换档（`torch` ↔ `kernel`）：对拍与云端切换都靠这一个动作。

    白话：要比较两条路就整排一起换，不能这张换那张不换；垫片里的数不动，
    换完再各算一遍，两条路的出入就是算子件自己的出入。
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend 只能是 {BACKENDS}，实得 {backend!r}")
    for mod in loras.values():
        mod.set_backend(backend, ascend_target=ascend_target)


# ================================================================ 工作项 B：适配器落盘（peft 键名）
def _adapter_key(path: str, which: str) -> str:
    """垫片路径 → peft 的存储键名（下游 `PeftModel.from_pretrained` 认这个写法）。"""
    return f"base_model.model.{path}.lora_{which}.default.weight"


def save_adapter(
    loras: dict[str, LoRALinear],
    out_dir: str | Path,
    *,
    base_model: str = "",
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把 A/B 两片垫片存成 `adapter.safetensors` + `adapter_config.json`（peft 键名）。

    白话：交出去的不是整台模型，只交那两小片垫片和一张说明条：说明条上写清楚是谁家的
    底座、几片薄、多大力度、装在哪些门上。换个人接手，凭这张条就能把垫片原样贴回去。

    :returns: 落盘清单（文件名、张量数、参数量），直接进 run 记录。
    """
    from safetensors.torch import save_file

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    for name, mod in loras.items():
        tensors[_adapter_key(name, "A")] = mod.lora_a.detach().cpu().clone()
        tensors[_adapter_key(name, "B")] = mod.lora_b.detach().cpu().clone()
    save_file(tensors, str(directory / ADAPTER_FILE))
    params = sum(int(t.numel()) for t in tensors.values())
    cfg: dict[str, Any] = {
        "peft_type": "LORA",
        "r": next(iter(loras.values())).r if loras else 0,
        "lora_alpha": next(iter(loras.values())).alpha if loras else 0,
        "lora_dropout": next(iter(loras.values())).dropout_p if loras else 0.0,
        "target_modules": list(LORA_TARGET_LEAVES),
        "modules_to_save": [],
        "base_model_name_or_path": base_model,
        "fan_in_fan_out": False,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "sft_version": SFT_VERSION,
        "params": params,
    }
    (directory / ADAPTER_CONFIG_NAME).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"file": ADAPTER_FILE, "tensors": len(tensors), "params": params,
            "config": ADAPTER_CONFIG_NAME, "meta": dict(meta or {})}


def load_adapter(loras: dict[str, LoRALinear], path: str | Path) -> int:
    """从 `adapter.safetensors` 把垫片读回来（键名对不上当场报错，不静默跳过）。

    白话：贴回去的活儿要"逐条对上"，少一条或名字打错都得当场喊停——静默跳过只会
    造出一个"看着像、其实没训过"的模型，那比失败更坏。

    :returns: 读回的张量数。
    """
    from safetensors.torch import load_file

    file = Path(path)
    if file.is_dir():
        file = file / ADAPTER_FILE
    state = load_file(str(file))
    count = 0
    for name, mod in loras.items():
        for which, param in (("A", mod.lora_a), ("B", mod.lora_b)):
            key = _adapter_key(name, which)
            if key not in state:
                raise KeyError(f"适配器缺键 {key!r}（垫片 {name} 的 {which}）——存读两侧口径不一致")
            value = state[key].to(dtype=param.dtype)
            if tuple(value.shape) != tuple(param.shape):
                raise ValueError(f"{key} 形状不符：存的是 {tuple(value.shape)}，本机需要 {tuple(param.shape)}")
            with torch.no_grad():
                param.copy_(value)
            count += 1
    return count


def merge_lora_weights(loras: dict[str, LoRALinear], *, out_dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
    """批量出补丁表（推理期用）：`{垫片路径: (n,k) 等效权重增量}`。

    白话：把每扇门的两小片各揉成一整块补丁，攒成一叠交给推理那边；本函数只算不贴，
    贴不贴由调用方决定，这样同一叠补丁可以反复用在不只一台底座上。
    """
    return {name: mod.merged_delta(out_dtype=out_dtype) for name, mod in loras.items()}


# ================================================================ 工作项 B：学生装载（p2-01 接缝 / 最小 HF 兜底）
#: 两种载入口：`seam` = 走 p2-01 的四道接缝（design 主路）；`minimal` = 自封的最小 HF 载入口（替身兜底）
LOADER_SEAM = "seam"
LOADER_MINIMAL = "minimal"
LOADERS = (LOADER_SEAM, LOADER_MINIMAL)

#: 位宽写法到 torch 类型的对照（yaml/CLI 里只认这三个名字，写别的当场报错）
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass
class Student:
    """被训的那台模型：整模 + 只走语言侧的前向壳 + 编号器 + 26 枚字母号 + 出处。

    `body` 与 `model` 分开存是刻意的：训练只吃语言侧逐位置输出（`model` 留着取输出层与存盘），
    视觉塔压根不在图里——这也是"注入只挂 language 侧"这条边界能被单测钉住的前提。
    """

    name: str
    model: Any
    body: torch.nn.Module
    tokenizer: Any
    head_weight: Any
    letter_ids: tuple[int, ...]
    pad_id: int
    loader: str
    dtype: torch.dtype = torch.float32
    provenance: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """一句话身份（进 run 配置与 README 的来源链）。

        白话：一句话说清"这台的型号、从哪道门载进来的、肚子里几个数、字母表多长"，
        报告和 README 都抄这一段，事后不必回头翻代码再认一遍。
        """
        return {"backbone": self.name, "loader": self.loader, "dtype": str(self.dtype),
                "params_m": round(sum(p.numel() for p in self.model.parameters()) / 1e6, 1),
                "letters": len(self.letter_ids), "pad_id": self.pad_id, **self.provenance}


def _pick_body(model: Any) -> torch.nn.Module:
    """从整模里挑"只走语言侧"的前向壳：Qwen3.5 藏在 `language_model`，老结构就在 `model.model`。"""
    inner = getattr(model, "model", model)
    return getattr(inner, "language_model", inner)


def _pad_id_of(tokenizer: Any) -> int:
    """空洞编号：有 pad 用 pad，没有就句尾符顶上（右补约定由调用侧保证）。"""
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    if pad is None:
        raise ValueError("编号器既没有 pad 也没有 eos，无法给空洞编号")
    return int(pad)


def build_student(
    name: str,
    *,
    loader: str = LOADER_SEAM,
    device: str = "cpu",
    dtype: torch.dtype | str = "float32",
    cache_dir: str | Path | None = None,
) -> Student:
    """把学生载起来：默认走 p2-01 接缝，`minimal` 是替身兜底路（两路都当场校 26 枚字母）。

    白话：正式门（接缝）会连带把"多模态编号、字母独占一格"这些规矩一并查一遍，
    0.8B 走这条。手头那台 0.6B 替身没有看图的本钱，硬走正式门会被"图上没有编号"
    这条规矩拦下，所以留一条侧门：照样从同一个仓里取东西、照样逐枚验字母，只是不跑
    那几道与文字无关的体检——这条侧门是替身专用的权宜，正式档必须回到正式门。

    :param name: backbone 名（`production.assets` 的注册表认得 `qwen3-0.6b`/`qwen3.5-0.8b`）。
    :param loader: `seam` 或 `minimal`。
    :param device: 装载设备（本地全 CPU）。
    :param dtype: 位宽（字符串或 torch 类型）。
    :param cache_dir: 权重快照根目录（默认 `production.assets.DEFAULT_CACHE_DIR`）。
    :raises ValueError: loader 名不认识，或字母读点校验不过。
    """
    if loader not in LOADERS:
        raise ValueError(f"loader 只能是 {LOADERS}，实得 {loader!r}")
    want = dtype if isinstance(dtype, torch.dtype) else DTYPES[str(dtype)]
    if loader == LOADER_SEAM:
        bb = load_backbone(name, device=device, dtype=want, cache_dir=cache_dir)
        head = bb.model.get_output_embeddings().weight
        return Student(
            name=bb.snapshot.repo, model=bb.model, body=bb.body, tokenizer=bb.tokenizer,
            head_weight=head.detach(), letter_ids=tuple(int(v) for v in bb.letter_ids),
            pad_id=_pad_id_of(bb.tokenizer), loader=LOADER_SEAM, dtype=want,
            provenance={"snapshot_path": str(bb.snapshot.path), "source": bb.snapshot.source,
                         "revision": bb.snapshot.revision},
        )
    # ── minimal：自封的最小 HF 载入口（依赖边界见模块头与战报；只用于本地替身冒烟）──
    import transformers

    from production.assets import fetch_snapshot
    from production.teachers.text import check_letters

    snap = fetch_snapshot(name, cache_dir=cache_dir)
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(snap.path))
    if getattr(tokenizer, "padding_side", "right") != "right":
        raise ValueError("空洞必须补在尾巴之后（左补会让读点读到空洞）")
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = transformers.AutoConfig.from_pretrained(str(snap.path))
    cls = getattr(transformers, (config.architectures or ["AutoModelForCausalLM"])[0])
    model = cls.from_pretrained(str(snap.path), dtype=want)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)                        # 进门先全锁，装完垫片再单独松
    letters = tuple(check_letters(tokenizer))           # 唯一读点的前提：26 枚各占一格
    return Student(
        name=snap.repo, model=model, body=_pick_body(model), tokenizer=tokenizer,
        head_weight=model.get_output_embeddings().weight.detach(), letter_ids=letters,
        pad_id=_pad_id_of(tokenizer), loader=LOADER_MINIMAL, dtype=want,
        provenance={"snapshot_path": str(snap.path), "source": snap.source, "revision": snap.revision},
    )


def enable_gradient_checkpointing(student: Student, flag: bool = True) -> bool:
    """按需开分段记账（省内存换时间）；底座不支持就如实交回 False，不假装开上了。

    白话：正常是一路记到底、回头照着账倒着改；分段记账是中间只留几个检查点，
    回头时把那段重算一遍。省下的是存放开销，多花的是重算的工夫——大模型上架时常常值，
    小机器上跑冒烟则不必。
    """
    body = student.body
    if not flag:
        return False
    if not hasattr(body, "gradient_checkpointing_enable"):
        return False
    body.gradient_checkpointing_enable()
    if hasattr(body, "enable_input_require_grads"):
        body.enable_input_require_grads()               # 首层输入要留钩子，否则记账链会断
    return True


def student_forward(student: Student, batch: Batch) -> torch.Tensor:
    """带图前向：交回 `(B,T,d)` 逐位置输出（`Backbone.forward` 在 no_grad 下，训练不能用它）。

    白话：把一整排题送进去，只收"每格读完留下的那串数"，不收写好的字；这条链必须
    带着来路记录，因为后面只从真末位那一格往回改数。空洞照样送进去但被掩码挡着，
    每行按它真写了几个字去取格。
    """
    out = student.body(input_ids=batch.ids, attention_mask=batch.mask, use_cache=False, return_dict=True)
    return out.last_hidden_state


# ================================================================ 工作项 B：步长调度
def cosine_lambda(total_steps: int, warmup_ratio: float = float(BASELINE_LORA["warmup_ratio"])):
    """5% 线性升温 + 余弦退火到零（照抄 StartLux `finetune_lora.py` 的调度写法，同构可比）。

    白话：开头几步先把下笔的力度从零线性加满，免得一起手就迈太大把刚装的垫片冲歪；
    之后按余弦一路放缓到接近零——越到后面挪得越小，收尾稳。
    """
    total = max(1, int(total_steps))
    warm = max(1, int(total * float(warmup_ratio)))

    def scale(step: int) -> float:
        progress = min(1.0, step / total)
        return min(1.0, (step + 1) / warm) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return scale


# ================================================================ 工作项 B：配置
def default_cfg() -> dict[str, Any]:
    """默认配置 = LoRA 基线超参 + 本地冒烟档（CPU/替身/百步以内）。

    白话：默认值走"最省时间但不撒谎"的那一档：题量与长度都封顶、一步只算一桌、
    位数用最稳的 float32。正式档不是把这里改小改大，而是另开配置表（configs/ 下两份），
    两份的差别只许落在"用哪台底座、一桌坐几个、走多少步"上。
    """
    return {
        # ── 资源与身份 ──
        "backbone": "qwen3-0.6b", "loader": LOADER_MINIMAL, "device": "cpu", "dtype": "float32",
        "threads": 4, "seed": 20261005, "cache_dir": None,
        # ── 数据（p2-03 registry 口）──
        "axis": "train", "data_dir": None, "limit": 512, "max_length": 512,
        # ── 教师（p2-02 缓存口；训练循环零教师前向）──
        "teacher_model_id": DEFAULT_TEACHER_MODEL_ID,
        "teacher_cache": "bench/teacher_cache/p2_05_pseudo",
        "mix_soft": DEFAULT_MIX_SOFT, "hard_mass": DEFAULT_HARD_MASS,
        # ── 旁路与算子档（p2-13）──
        "kernel_backend": BACKEND_KERNEL, "ascend_target": "cpu",
        "r": int(BASELINE_LORA["r"]), "alpha": int(BASELINE_LORA["alpha"]),
        "dropout": float(BASELINE_LORA["dropout"]), "gradient_checkpointing": False,
        # ── 步长与调度（StartLux 基线）──
        "lr": float(BASELINE_LORA["lr"]), "warmup_ratio": float(BASELINE_LORA["warmup_ratio"]),
        "weight_decay": float(BASELINE_LORA["weight_decay"]),
        "max_grad_norm": float(BASELINE_LORA["max_grad_norm"]),
        "epochs": float(BASELINE_LORA["epochs"]), "max_tokens": int(BASELINE_LORA["max_tokens"]),
        "accum": int(BASELINE_LORA["accum"]), "max_steps": 100,
        # ── 产物与记录 ──
        "run_prefix": "p2-05-dev", "out_dir": None, "prefill_chunk": 16,
    }


def load_config(path: str | Path | None, *, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """三层合流：默认档 ← yaml ← 显式覆盖；认不出的键当场报错（拼错一个字母不该静默失效）。

    白话：先摆一份谁都一样的底单，再照配置表改，最后听命令行。任何一层冒出个底单上没有
    的名字，就把它点出来——"写了没生效"比"报错"难查十倍。

    :param path: yaml 路径（None 表示只用默认档 + 覆盖）。
    :param overrides: 显式覆盖（None 值会被忽略，方便 CLI 只传用户真写了的项）。
    :raises ValueError: 出现未知键时抛出（消息含可用键清单）。
    """
    cfg = default_cfg()
    known = set(cfg)
    if path:
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"配置表须是键值表，实得 {type(data).__name__}：{path}")
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"未知配置键 {unknown}；可用：{sorted(known)}")
        cfg.update(data)
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key not in known:
            raise ValueError(f"未知配置键 {key!r}；可用：{sorted(known)}")
        cfg[key] = value
    return cfg


def _axis_facts(axis: str) -> tuple[str, list[dict[str, Any]]]:
    """这条轴的钉版事实：数据版本串 + 各副本的 revision/split/条数（台账读不动就交占位）。

    记不清"题从哪一版来"就等于没记：run 的 `data_revision` 与 `axis_splits` 都从这一处出，
    两处各读各的迟早对不上。台账本身不在时不硬崩——冒烟照样要跑，但如实写 UNPINNED。
    """
    try:
        from sys1.eval import registry

        sets = (registry.load_manifest() or {}).get("sets") or {}
        rows = [{"set": pid, "revision": str(sets.get(pid, {}).get("revision") or "?"),
                 "split": str(sets.get(pid, {}).get("split") or "?"),
                 "samples": sets.get(pid, {}).get("samples")} for pid in eval_run.axis_ids(axis)]
        revision = "+".join(f"{r['set']}@{str(r['revision'])[:7]}" for r in rows) or "UNPINNED"
        return revision, rows
    except Exception:                                       # noqa: BLE001 — 台账读不动别挡住训练
        return "UNPINNED", []


# ================================================================ 工作项 B：模型目录（产物布局）
def decision_config_payload(student: Student, cfg: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """拼 decision_config.json：P1 的读点/版号槽 + p2-02 读法要的 temperature/字母号两槽。

    白话：这张小抄是"怎么读答案"的说明书，读的人有两拨——一阶段那套照 P1 的格子读，
    p2-02 的教师那套另外找两个名字读。两边要的字段都写齐，谁拿到这本都不用猜；
    温度一个都不填，因为这次根本没做标定，编个数进去就是撒谎。
    """
    return {
        "version": SFT_VERSION,
        "owner": "p2-05-prod-sft",
        "render_version": RENDER_VERSION,
        "readout": {"position": "last_real_token", "letters": list(LETTERS), "max_options": MAX_OPTIONS},
        "letter_token_ids": list(student.letter_ids),
        "temperature_by_type": {},
        "temperatures": {},
        "calibration": {"status": "pending"},
        "sft": meta,
        "teacher": {"model_id": cfg["teacher_model_id"], "cache": str(cfg["teacher_cache"]),
                    "mix_soft": float(cfg["mix_soft"])},
    }


def write_model_dir(
    out_dir: str | Path,
    *,
    student: Student,
    loras: dict[str, LoRALinear],
    cfg: dict[str, Any],
    asm: AsmStats,
    result: dict[str, Any],
) -> dict[str, Any]:
    """按 design 的产物布局落一盘：适配器 + config.json + decision_config.json + README(来源链)。

    白话：交出去的是一个能直接被人拿去跑的目录，而不是一堆散文件。README 那份来源链
    写清"题从哪本账来、教师是谁家的份额、底座是哪一版、代码跑到第几步"，
    接手的人不必回来问就能知道这盘东西的来历。
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": SFT_VERSION,
        "backend": cfg["kernel_backend"], "ascend_target": cfg["ascend_target"],
        "r": int(cfg["r"]), "alpha": int(cfg["alpha"]), "dropout": float(cfg["dropout"]),
        "target_leaves": list(LORA_TARGET_LEAVES), "trainable": result.get("trainable"),
        "steps": result.get("steps"), "first_loss": result.get("first_loss"),
        "final_loss": result.get("final_loss"), "run_id": result.get("run_id"),
        "data": {"axis": cfg["axis"], "revision": result.get("data_revision"),
                 "records_seen": asm.seen, "encoded": asm.encoded, "skipped": asm.skipped},
        "teacher": {"model_id": cfg["teacher_model_id"], "soft_coverage": round(asm.soft_coverage, 6),
                    "hits": asm.hits, "misses": asm.misses},
        "note": TRAINABLE_NOTE,
    }
    listing = save_adapter(loras, directory, base_model=student.provenance.get("snapshot_path", ""), meta=meta)
    (directory / CONFIG_FILE).write_text(
        json.dumps({"backbone": student.summary(), "train_cfg": {k: cfg[k] for k in sorted(cfg)}},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (directory / DECISION_CONFIG_FILE).write_text(
        json.dumps(decision_config_payload(student, cfg, meta), ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# p2-05 SFT 产物（{result.get('run_id')}）", "",
        "【来源链】",
        f"- 题目：registry 轴 `{cfg['axis']}`，数据版本 `{result.get('data_revision')}`，"
        f"副本 {json.dumps(result.get('axis_splits'), ensure_ascii=False)}",
        f"- 编成 {asm.encoded} 条（过目 {asm.seen} 条，丢弃 "
        f"{json.dumps(asm.skipped, ensure_ascii=False) if asm.skipped else '无'}），"
        f"教师伪标覆盖 {asm.soft_coverage:.1%}（命中 {asm.hits}/落空 {asm.misses}）",
        f"- 教师：`{cfg['teacher_model_id']}`，只读缓存 `{cfg['teacher_cache']}`"
        f"（训练循环零教师前向；D11 红线——只吃它的分布缓存，教师权重一律不进训练）",
        f"- 底座：`{student.name}`（载入口 {student.loader}）{json.dumps(student.provenance, ensure_ascii=False)}",
        f"- 旁路：LoRA r{cfg['r']}/α{cfg['alpha']}/dropout {cfg['dropout']}，挂 {', '.join(LORA_TARGET_LEAVES)}；"
        f"实现档 `{cfg['kernel_backend']}`（ascend_target={cfg['ascend_target']}）",
        f"- 训练：{result.get('steps')} 步，loss {result.get('first_loss')} → {result.get('final_loss')}",
        "", f"> {TRAINABLE_NOTE}", "",
    ]
    (directory / README_NAME).write_text("\n".join(lines), encoding="utf-8")
    return {"dir": str(directory), "adapter": listing,
            "files": [ADAPTER_FILE, ADAPTER_CONFIG_NAME, CONFIG_FILE, DECISION_CONFIG_FILE, README_NAME]}


# ================================================================ 工作项 B：训练循环
def train(cfg: dict[str, Any]) -> dict[str, Any]:
    """跑一条 tiny SFT：装数据 → 装垫片 → 逐步只从真末位罚分 → 落产物 + run 四件套。

    白话：一桌一桌地做题。每桌只在"每个字写完"那一刻看它押的份额准不准，前面对错多少
    字都不罚；罚完把这份改动攒几桌再一起下笔（几桌合一次叫 accum），下笔前先按调度调好
    力度。跑满就交三份东西：能贴回去的两片垫片、一张怎么读答案的小抄、一本流水账。

    :param cfg: `default_cfg()`/`load_config()` 出来的配置表。
    :returns: 本次训练的事实表（run_id、步数、首末损失、吞吐、可训参数量……）。
    :raises RuntimeError: 一条样本都编不出来（数据口空/题面全不合格）时抛出。
    """
    torch.set_num_threads(int(cfg["threads"]))
    random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))

    student = build_student(cfg["backbone"], loader=cfg["loader"], device=cfg["device"],
                            dtype=cfg["dtype"], cache_dir=cfg["cache_dir"])
    cache = None
    root = cfg.get("teacher_cache")
    if root and Path(root).is_dir():
        cache = DistCache(root=root, read_only=True).load()     # 只读：训练侧绝不往教师账本里写
    records = load_quality_records(cfg["axis"], data_dir=cfg.get("data_dir"))
    samples, asm = assemble(records, student.tokenizer, cache=cache,
                            teacher_model_id=cfg["teacher_model_id"], mix_soft=float(cfg["mix_soft"]),
                            max_length=int(cfg["max_length"]), limit=int(cfg["limit"]))
    if not samples:
        raise RuntimeError(f"一条样本都没编出来（见 {asm.seen} 条，丢弃 {asm.skipped}）——检查数据口与 max_length")

    axis_revision, axis_splits = _axis_facts(cfg["axis"])
    loras = inject_lora(student.body, r=int(cfg["r"]), alpha=int(cfg["alpha"]),
                        dropout=float(cfg["dropout"]), backend=cfg["kernel_backend"],
                        ascend_target=cfg["ascend_target"])
    counts = freeze_all_but_lora(student.model, loras)
    ckpt_on = enable_gradient_checkpointing(student, bool(cfg["gradient_checkpointing"]))

    params = [p for p in student.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    buckets = bucket_batches(samples, max_tokens=int(cfg["max_tokens"]), shuffle=True, seed=int(cfg["seed"]))
    epochs = max(1, math.ceil(float(cfg["epochs"])))
    total_steps = int(cfg["max_steps"]) or max(1, int(len(buckets) * float(cfg["epochs"]) // max(1, int(cfg["accum"]))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lambda(total_steps, float(cfg["warmup_ratio"])))

    run = new_run(f"{cfg['run_prefix']}-sft-{cfg['backbone']}-{cfg['kernel_backend']}", {
        "stage": f"p2-05-prod-sft/{cfg['run_prefix']}", "sft_version": SFT_VERSION,
        "data_revision": axis_revision, "axis_splits": axis_splits,
        "config": {k: cfg[k] for k in sorted(cfg)}, "assemble": asm.as_dict(),
        "student": student.summary(), "lora_modules": sorted(loras), "counts": counts,
        "gradient_checkpointing": ckpt_on,
        "kernels": {"backend": cfg["kernel_backend"], "ascend_target": cfg["ascend_target"],
                    # 开局清点：此刻还没下笔，编译表为空是事实不是遗漏（终局数见 metrics 的 kernel_compiled）
                    "compiled_at_start": list(ascend_env.compiled_keys())},
    })

    # 装配事实属开局账：现在就记（step 0 行落在曲线最前），别等跑完再补——
    # 否则"step 0"会带着终局墙钟排在第 100 步之后，读曲线的人无法判断该信哪条时间线。
    run.log_metrics(0, asm_encoded=asm.encoded, asm_seen=asm.seen, teacher_hits=asm.hits,
                    teacher_misses=asm.misses, soft_coverage=round(asm.soft_coverage, 6))
    accum = max(1, int(cfg["accum"]))
    step, micro, rows = 0, 0, []
    running = 0.0
    tokens_all = 0
    seconds_all = 0.0
    losses: list[float] = []
    batch: Batch | None = None
    started = time.time()
    student.model.train()
    for epoch in range(epochs):
        for idx in buckets:
            if step >= total_steps:
                break
            batch = collate(samples, idx, pad_id=student.pad_id, letter_map=student.letter_ids).to_(cfg["device"])
            hidden = student_forward(student, batch)
            loss = readout_ce_loss(hidden, student.head_weight, batch.letter_ids, batch.target,
                                   keep=batch.keep, lengths=batch.lengths) / accum
            loss.backward()
            micro += 1
            running += float(loss.detach()) * accum
            tokens_all += batch.tokens
            if micro % accum == 0:
                gnorm = float(torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"])))
                opt.step()
                lr = float(sched.get_last_lr()[0])
                sched.step()
                opt.zero_grad(set_to_none=True)
                spent = time.time() - started
                step_seconds = spent - seconds_all
                seconds_all = spent
                step += 1
                value = running / accum
                losses.append(value)
                rows.append(value)
                run.log_metrics(step, loss=round(value, 6), lr=lr, micro=micro,
                                step_seconds=round(step_seconds, 3),
                                tokens_per_second=round(batch.tokens / max(step_seconds, 1e-6), 1),
                                tokens=tokens_all, batch_tokens=batch.tokens, batch_rows=len(idx),
                                grad_norm=round(gnorm, 4),
                                soft_coverage=round(asm.soft_coverage, 6))
                running = 0.0
                print(f"[{step}/{total_steps}] loss {value:.4f} lr {lr:.2e} "
                      f"{step_seconds:.2f}s/{step} tokens {tokens_all}", flush=True)
            if step >= total_steps:
                break
        if step >= total_steps:
            break
    if micro % accum:
        opt.step()                                        # 收尾那半桌也下笔，别让攒下的改动白丢
        opt.zero_grad(set_to_none=True)

    wall = time.time() - started
    win = max(1, math.ceil(len(rows) * 0.1)) if rows else 0      # 首尾各取一成步数求均值
    first_loss = round(sum(rows[:win]) / win, 6) if win else 0.0
    final_loss = round(sum(rows[-win:]) / win, 6) if win else 0.0
    tok_per_sec = tokens_all / max(wall, 1e-6)
    result = {
        "run_id": run.run_id, "run_dir": str(run.path), "steps": step, "micro": micro,
        "samples": len(samples), "batches": len(buckets), "first_loss": first_loss,
        "final_loss": final_loss, "loss_window": win,
        "loss_decrease_pct": round(100.0 * (first_loss - final_loss) / max(first_loss, 1e-9), 2) if win else 0.0,
        "tokens": tokens_all, "seconds": round(wall, 2), "tokens_per_second": round(tok_per_sec, 1),
        "seconds_per_step": round(wall / max(1, step), 3), "trainable": counts["trainable"],
        "frozen": counts["frozen"], "lora_modules": counts["lora_modules"],
        "soft_coverage": round(asm.soft_coverage, 6), "teacher_hits": asm.hits, "teacher_misses": asm.misses,
        "data_revision": axis_revision, "axis_splits": axis_splits,
        "kernel_backend": cfg["kernel_backend"], "ascend_target": cfg["ascend_target"],
        "compiled_keys": list(ascend_env.compiled_keys()), "losses": rows,
    }
    model_dir = Path(cfg["out_dir"]) if cfg.get("out_dir") else run.path / "model"
    listing = write_model_dir(model_dir, student=student, loras=loras, cfg=cfg, asm=asm, result=result)
    result["model_dir"] = listing["dir"]
    result["adapter_files"] = listing["files"]
    # 自研件终局清点挂在最后一步（真编了几枚；0 = 这条 run 没走 kernel 路）：
    # 编译发生在下笔之后，只有这个时点的数才是事实，开局快照见 config 的 compiled_at_start。
    run.log_metrics(step, kernel_compiled=len(result["compiled_keys"]))
    run.conclude(
        f"tiny SFT 跑通 {step} 步（{cfg['backbone']} 替身 / {cfg['device']} {cfg['dtype']} / "
        f"旁路 {cfg['kernel_backend']}@{cfg['ascend_target']}）：loss {first_loss} → {final_loss}"
        f"（首尾各 {win} 步均值，降 {result['loss_decrease_pct']}%），吞吐 {result['tokens_per_second']} tok/s"
        f"（{result['seconds_per_step']} s/step，{tokens_all} 字 / {result['seconds']} s），"
        f"可训 {result['trainable']} 参数 / {result['lora_modules']} 枚垫片，"
        f"伪标覆盖 {result['soft_coverage']:.1%}（{cfg['teacher_model_id']} 缓存口，训练循环零教师前向）；"
        f"{TRAINABLE_NOTE}。"
    )
    run.finish()
    result["notes"] = str(run.notes_file)
    result["metrics"] = str(run.metrics_file)
    return result


# ================================================================ 工作项 A/B：本地伪标预热（教师产标口）
def scaffold_teacher(cache_root: str | Path):
    """搭一台袖珍教师（真结构 + 随机权重）配一本可写账本，只用于本地产伪标。

    白话：正版教师那 8.7G 的机器本地放不下（p2-02 已如实记过这笔账），所以照它的图纸
    搭个袖珍版先把"产标 → 落账 → 训练只读账"这条路走通。它的份额没有业务意义，
    身份号上明写 `#scaffold-cpu`，与正版教师的账本天然分家，绝不会串进真伪标里。

    :returns: `(teacher, cache)`。
    """
    from transformers import AutoConfig, AutoTokenizer

    from production.teachers.text import HfCausalLMBackend, TeacherStats, TextTeacher, load_decision_config
    from production.teachers.verify_b3 import SNAPSHOT, SCAFFOLD_MODEL_ID, scaffold_model

    config = AutoConfig.from_pretrained(str(SNAPSHOT))
    tokenizer = AutoTokenizer.from_pretrained(str(SNAPSHOT), trust_remote_code=True)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = scaffold_model(config)
    cache = DistCache(root=cache_root).load()                      # 可写：这一趟就是来产标的
    backend = HfCausalLMBackend(model, tokenizer, SCAFFOLD_MODEL_ID, device="cpu")
    teacher = TextTeacher(backend, cache=cache, stats=TeacherStats(),
                          settings=load_decision_config(SNAPSHOT))
    return teacher, cache


def prefill_pseudo(cfg: dict[str, Any], samples: Sequence[Sample] | None = None) -> dict[str, Any]:
    """给 tiny 冒烟预热伪标：按候选集分组批量打分 → 落 parquet → 报"发了几次算力"。

    白话：先把这批题一次性问完教师、把份额抄进账本，之后训练只翻账本、一次都不再问。
    候选一样的题凑成一桌一起问（教师那边要求同桌候选集一致），问完把总数、耗时、
    这桌发了几回算力都摆在面上——日后换成正版教师的伪标包，只要换个身份号与目录，
    这套动作一步都不用改。

    :param cfg: 配置表（用其中的数据口、装配参数、`teacher_cache`、`prefill_chunk`）。
    :param samples: 已编好的题（None 表示现场只渲染题面、不编号——产标用不着编号串）。
    :returns: 预热事实表（题数、前向次数、命中、落盘条目、耗时、教师身份）。
    """
    from collections import defaultdict

    torch.set_num_threads(int(cfg["threads"]))
    teacher, cache = scaffold_teacher(cfg["teacher_cache"])
    model_id = teacher.model_id
    if samples is None:
        items = prompt_index(cfg["axis"], data_dir=cfg.get("data_dir"), limit=int(cfg["limit"]))
    else:
        items = [(one.prompt, tuple(one.letters)) for one in samples]
    groups: dict[tuple[str, ...], list[tuple[str, tuple[str, ...]]]] = defaultdict(list)
    for prompt, letters in items:
        groups[letters].append((prompt, letters))
    chunk = max(1, int(cfg["prefill_chunk"]))
    started = time.time()
    asked = 0
    for letters, bucket in groups.items():
        for i in range(0, len(bucket), chunk):
            teacher.score_batch([(p, list(k)) for p, k in bucket[i:i + chunk]], qtype="choice")
            asked += len(bucket[i:i + chunk])
    flushed = cache.flush()
    stats = teacher.stats.as_dict()
    out = {"teacher_model_id": model_id, "prompts": len(items), "groups": len(groups),
           "asked": asked, "forward_calls": stats.get("forward_calls"),
           "cache_hits": stats.get("cache_hits"), "cache_misses": stats.get("cache_misses"),
           "flushed_rows": flushed, "entries": len(cache.index), "seconds": round(time.time() - started, 2),
           "cache_root": str(cfg["teacher_cache"]),
           "note": f"伪标由同架构袖珍教师产出（随机权重，仅通路证据）；真教师伪标待 C3。{TRAINABLE_NOTE}"}
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


# ================================================================ 工作项 A：产标备料（只渲染，不编号）
def prompt_index(axis: str = "quality", *, data_dir: str | Path | None = None,
                 limit: int = 0) -> list[tuple[str, tuple[str, ...]]]:
    """从 registry 口取题面扁平文字 + 字母序：给产标那一步备料（它用不着编号串）。

    白话：教师那边吃的是"排好的那段话加几个字母"，学生那边才需要编号。所以这一步
    只把话排出来、把字母发下去，不碰编号器——省掉一趟白费，也保证产标用的题目
    与训练用的题目是同一条渲染路出来的同一份文字（同序同字）。
    """
    out: list[tuple[str, tuple[str, ...]]] = []
    for record in load_quality_records(axis, data_dir=data_dir):
        sample = record.get("sample") or {}
        for qid, spec in (sample.get("questions") or {}).items():
            if not isinstance(spec, dict):
                continue
            row = from_systemone(sample.get("state"), spec, qid=qid)
            order = option_order(row)
            if not order or len(order) > MAX_OPTIONS:
                continue
            out.append((prompt_text(row, order), tuple(LETTERS[: len(order)])))
            if limit and len(out) >= limit:
                return out
    return out


# ================================================================ 命令行
#: CLI 开关 ↔ 配置键对照（(键, 类型, 说明)）；只覆盖用户真写出来的项
_CLI_FLAGS: tuple[tuple[str, type, str], ...] = (
    ("backbone", str, "backbone 名（qwen3-0.6b / qwen3.5-0.8b）"),
    ("loader", str, f"载入口：{LOADERS}（seam 走 p2-01 接缝，minimal 是替身兜底）"),
    ("device", str, "装载设备（本地全 CPU）"),
    ("dtype", str, f"位宽：{sorted(DTYPES)}"),
    ("threads", int, "CPU 线程数"),
    ("seed", int, "随机种子（同种子必得同一套桌序）"),
    ("axis", str, "registry 评测轴（训练题源）"),
    ("data_dir", str, "数据根目录（默认 registry.DATA_DIR）"),
    ("limit", int, "装配封顶条数（tiny 冒烟用）"),
    ("max_length", int, "单条长度上限（超过直接丢，不截半截题）"),
    ("teacher_model_id", str, "教师身份号（缓存键的一部分）"),
    ("teacher_cache", str, "教师分布缓存目录（训练侧只读）"),
    ("mix_soft", float, "soft 分量权重（0=纯人工，1=纯教师）"),
    ("hard_mass", float, "人工硬标签的押注质量"),
    ("kernel_backend", str, f"旁路实现档：{BACKENDS}（kernel 走 p2-13 算子件）"),
    ("ascend_target", str, "算子件档位（cpu / ascend）"),
    ("r", int, "低秩维度"),
    ("alpha", int, "缩放分子（力度 = α/r）"),
    ("dropout", float, "垫片入口的丢弃率"),
    ("gradient_checkpointing", bool, "分段记账开关"),
    ("lr", float, "学习率"),
    ("warmup_ratio", float, "升温段占比"),
    ("max_grad_norm", float, "改动幅度上限"),
    ("epochs", int, "过几遍题"),
    ("max_tokens", int, "一桌补齐后的字数上限（批规模）"),
    ("accum", int, "攒几桌下一次笔"),
    ("max_steps", int, "下笔次数上限（0=按桌数与遍数推）"),
    ("run_prefix", str, "run 名前缀（本地半场固定 p2-05-dev）"),
    ("out_dir", str, "产物目录（默认落在 run 目录里的 model/）"),
    ("prefill_chunk", int, "产标一桌问几题"),
)


def build_parser() -> argparse.ArgumentParser:
    """造命令行：默认训练，`--prefill-pseudo` 先产标，`--print-config` 只看三层合流结果。

    白话：所有开关都对着配置表上的名字，写哪个改哪个；没写的就用底单。
    这样同一份脚本能同时服务"本地百步冒烟"和"云端正式档"，差别只在配置表。
    """
    ap = argparse.ArgumentParser(prog="python -m production.sft",
                                 description="p2-05 读出位 SFT（LoRA + 教师伪标 soft 混合）")
    ap.add_argument("--config", help="yaml 配置表（默认档 ← 本表 ← 命令行）")
    ap.add_argument("--print-config", action="store_true", help="只打印合流后的配置并退出")
    ap.add_argument("--prefill-pseudo", dest="prefill_pseudo", action="store_true",
                    help="用袖珍教师给 tiny 冒烟预热伪标缓存（产标后退出，不进训练）")
    for key, kind, help_text in _CLI_FLAGS:
        if kind is bool:
            ap.add_argument(f"--{key.replace('_', '-')}", dest=key,
                            action=argparse.BooleanOptionalAction, default=None, help=help_text)
        else:
            ap.add_argument(f"--{key.replace('_', '-')}", dest=key, type=kind, default=None, help=help_text)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：合流三层配置 → 按需产标或直接训练 → 把事实表打到标准输出。

    白话：先看用户点了什么开关，缺的用底单补；要产标就一趟产完写进账本；
    要训练就训练，跑完把 run 门牌号、首末损失、吞吐和产物目录念一遍——
    念出来的每个数都能在 run 那本流水里指到具体行。
    """
    args = build_parser().parse_args(argv)
    overrides = {key: getattr(args, key) for key, _, _ in _CLI_FLAGS}
    cfg = load_config(args.config, overrides=overrides)
    if args.print_config:
        print(json.dumps(cfg, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.prefill_pseudo:
        prefill_pseudo(cfg)
        return 0
    result = train(cfg)
    print(json.dumps({k: v for k, v in result.items() if k != "losses"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
