"""p2-06 OPD 栈：学生自采样 top-k + 教师同选项 reverse-KL/JSD（本地开发/CPU 半场档）。

【做什么】
    三栈之二（on-policy distillation）。与前一行 `production/sft.py` 的差别只有一处、但要害：
    蒸馏目标不再钉在"人工给定的那档"上，而是**由学生自己决定该在哪些候选上较真**——
    学生先对决策状态做一次末位读出，取自己份额最高的 k 个候选（默认 8），再拿教师
    在**同一组候选**上的归一化分布当老师，逼学生把这份份额学像。配套三件：
    ① `sample_topk/topk_probs`：采样器，输入"模型 + 状态"交出 top-k 的 (opt, p)——
       这是 RL 栈（p2-11）要直接 import 复用的公共契约，签名与形状是稳定面，实现可换；
    ② `reverse_kl/jsd/opd_loss`：稠密散度损失（方向锁定 `KL(p_teacher ‖ p_student)`，可切 JSD）；
    ③ `train/ablation` CLI：train 续训 p2-05 的 SFT 产物，ablation 出 on/off 同 seed 两 run
       并如实记 Δ（零增益、负增益都入库，这是本域交付物而不是失败）。
    教师分布全程只从 p2-02 的 parquet 缓存口取（`DistCache`，只读），训练循环一次教师前向都不发。

【怎么做】
    四步一条链，全部复用现成件、一处读点口径都不重抄：
    ① 排版与采样：一行 registry 信封 → p2-03 `load_axis_records` → P1 冻结件
       `from_systemone/option_order/prompt_text` 排版 → 学生前向（`student_forward` 同款带图路径）
       → **唯一读点** `sys1.decision.readout.option_scores` 取末位×字母行的 (B,C) 分数
       → `topk_probs` 按分数降序取前 k 枚并把份额在这 k 列内重归一（和为 1，spec 场景钉这条）。
       同分按列序破平，采样器因此与随机数无关：同一状态两次必得同一组候选。
    ② 教师同选项分布：`teacher_on_subset` 拿**在场全候选**的键去查缓存（键 = 模型+请求+候选集），
       再 `restrict_normalize` 限制到那 k 列并重归一。这里有条数学等价支撑设计里"缓存键含选项集"
       那句话：教师 softmax 限制到子集后重归一 ≡ 直接对子集做 softmax（分子分母同除 Z 相消），
       温度缩放同样可交换；于是"同 k 选项的归一化分布"不必为每个子集另发一次教师前向，
       一次全键缓存就够——既守住"教师分布 MUST 走离线缓存"，又不为 top-k 组合爆炸产标。
    ③ 损失：log-softmax 一律在 fp32 域算完再回投（fp16 直算不稳，见 design 风险条）；
       批内候选数不齐与"只在这 k 列上较真"都靠 `keep` 布尔列——先把未选列封 -inf 参与归一，
       再在任何乘法之前把该列归零（留着 -inf 与 0 相乘得 nan，SFT 冒烟踩过一次）；
       教师零份额列按 0·log0=0 的约定整项跳过，学生侧 log 概率地板 clamp 防 -inf。
       目标可按 `mix_gt`（默认 0.2）掺人工真值，掺后重新归一，仍是同一个 KL 入口。
    ④ 训练与消融：`train` 装数据 → 采样 top-k → 查教师 → 逐微批散度下降（LoRA 垫片仍由
       p2-05 的 `inject_lora/freeze_all_but_lora` 提供，本域不另造旁路）；`ablation` 用同一份
       cfg 跑 `on/off` 两档，Δ 用 p2-03 的 typed-decisions acc 与 ECE 口径出数并写进 notes。

【为什么】
    被否方案一：把教师分布直接摊在"全部在场候选"上做稠密 KD（= soft-KD）。判别式读出下学生
        本来就看得见全部候选，与 soft-KD 的唯一差别只剩"权重"，OPD 名义的 on-policy 采样成了
        装饰——手册已声明这条教学假设，故本域交付物是**消融 run 与 Δ 结论**本身，而不是假装增益。
    被否方案二：为 top-k 子集单独产教师标。k=8、在场 20 候选 → 每个状态最多 C(20,8) 种子集，
        产标量爆炸且缓存键无法穷举；用"限制+重归一"的等价式换掉，一次前向的缓存覆盖全部子集。
    被否方案三：损失方向取 KL(p_student ‖ p_teacher)。那是 mode-seeking：学生只盯教师最高的
        那一峰，教师的多峰（几个候选并列合理）被抹平，与 D9 的业界口径（Agarwal/OPD-survey 的
        逐 token reverse 口径 = 以教师为参考测度）相反。方向不是风格问题，单测用不对称数值例
        把它钉死：同一对分布，两个方向的散度值差一个数量级。
    被否方案四：本地就用 0.6B 替身跑完 C1/C2 消融。替身 SFT 只训了 100 步、伪标来自袖珍教师，
        拿它出的 Δ 会被误读成正式档结论；故 C1/C2 一律待 C5 正式 SFT 产物，本地只交接口自测。
    被否方案五：采样器直接吃 HF 整模、内部自己找读点位。RL 栈的读点与训练态不同（要 replay
        旧策略分数），把"给分数→给 top-k"这层纯函数拆出来（`topk_probs`），RL 只依赖形状契约，
        不被本域的 backbone 装配拖累。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# ── 上游现成接口（本域一律"用"不"造"：数据口 p2-03 / 教师缓存口 p2-02 / 读点冻结件 / 旁路与学生）──
from production.sft import (                               # p2-05：学生壳、LoRA 旁路、装配与批件
    BACKENDS,
    BACKEND_TORCH,
    BASELINE_LORA,
    DTYPES,
    LOADERS,
    AsmStats,
    DEFAULT_HARD_MASS,
    DEFAULT_TEACHER_MODEL_ID,
    LOADER_MINIMAL,
    SCAFFOLD_SUFFIX,
    Student,
    bucket_batches,
    build_student,
    cosine_lambda,
    enable_gradient_checkpointing,
    freeze_all_but_lora,
    inject_lora,
    load_adapter,
    load_config as _sft_load_config,
    load_quality_records,
    mix_targets,
    student_forward,
    write_model_dir,
)
from production.teachers.cache import DistCache, make_key   # p2-02 教师缓存口（训练侧只读）
from production.teachers.protocol import hard_distribution  # p2-02 硬标签折法（真值掺入用）
from production.teachers.text import TeacherStats           # p2-02 账本："零前向"断言的数据源
from sys1.decision.readout import option_scores            # P1 读出（冻结件，唯一读点）
from sys1.decision.render import (                          # P1 渲染（冻结件）
    LETTERS,
    MAX_OPTIONS,
    RENDER_VERSION,
    from_systemone,
    option_order,
    prompt_text,
)
from sys1.runs import new_run                               # run 记录本

__all__ = [
    "ABLATION_ONSETS",
    "DEFAULT_JSD_BETA",
    "DEFAULT_MIX_GT",
    "DEFAULT_TOP_K",
    "JSD",
    "LOSS_JSD",
    "LOSS_REVERSE_KL",
    "LOSSES",
    "OPD_VERSION",
    "REVERSE_KL",
    "SCAFFOLD_MODEL_ID",
    "OpdBatch",
    "OpdSample",
    "TopKSample",
    "ablation",
    "ablation_plan",
    "build_parser",
    "collect_onpolicy",
    "collate_opd",
    "default_cfg",
    "delta_line",
    "divergence",
    "evaluate_dir",
    "jsd",
    "load_config",
    "main",
    "mixed_target",
    "opd_loss",
    "opd_step",
    "restrict_normalize",
    "reverse_kl",
    "sample_topk",
    "selected_logp",
    "student_topk_scores",
    "teacher_full_dist",
    "teacher_on_subset",
    "topk_probs",
    "train",
]

#: 消融两档的档名（on = 按学生自采样截断；off = 摊全表，即退化 soft-KD）
ABLATION_ONSETS = ("on", "off")

#: 本次交付的 OPD 版本号（进 run 配置与 decision_config，与 p2-05 的 `dmlaya_p2_sft_v1` 分家）
OPD_VERSION = "dmlaya_p2_opd_v1"

#: on-policy 采样的候选宽度（spec：k 可配，默认 8；RL 栈共用这个默认值）
DEFAULT_TOP_K = 8
#: 教师份额里掺入人工真值的比例（design：默认 0.2 = 两成听真值、八成听教师）
DEFAULT_MIX_GT = 0.2
#: JSD 的混合权重（β=0.5 即对称 Jensen-Shannon 散度，design 指定默认）
DEFAULT_JSD_BETA = 0.5

#: 散度两档取值；默认 reverse-KL（方向以教师为参考测度，mode-covering）
LOSS_REVERSE_KL = "reverse_kl"
LOSS_JSD = "jsd"
LOSSES = (LOSS_REVERSE_KL, LOSS_JSD)
REVERSE_KL, JSD = LOSS_REVERSE_KL, LOSS_JSD          # 便于 `opd.REVERSE_KL` 这类短引用

#: 袖珍教师的身份号（本地接口自测专用）：与真教师缓存键天然分家，随机分数绝不当伪标
SCAFFOLD_MODEL_ID = f"{DEFAULT_TEACHER_MODEL_ID}{SCAFFOLD_SUFFIX}"

#: 学生 log 概率的地板：极端列（fp16 投回来可能刚好是 -inf）罚到 -30 封顶，防整批 nan
_LOG_FLOOR = -30.0
#: 教师份额的"视为零"阈值：低于它就不计入散度（0·log 0 的约定），也防 log(1e-40) 抖飞
_TARGET_FLOOR = 0.0


# ================================================================ 工作项 A：on-policy 采样器
@dataclass(frozen=True)
class TopKSample:
    """一次 on-policy 采样的出口：top-k 的 (opt, p) + 复现所需的上下文（全候选序、原始分数）。

    白话：学生给一道题报"我最看好这几档、各占几成"，这里把那几档连同名字一并交出去，
    另外留住两件事后要用的东西：这道题问教师时用的那段话（`prompt`，缓存键的原料）、
    以及这 k 档在原池子里占的总份额（`mass`，看截断截掉了多少信息）。

    形状契约（RL 栈复用面，改它要走父变更评审，不许本域私改）：
    - `letters`/`options`/`probs`/`scores` 四列**等长同序**，长度为 `min(k, 候选数)`；
    - `probs` 在这几列内重归一，和为 1（spec 场景"返回 ≤8 个 (opt, p) 且概率和=1"）；
    - `options[i]` 是候选代号（continue/human_review/…），`letters[i]` 是它的字母列（A/B/…），
      且 `options[i] == order[LETTERS.index(letters[i])]`（列↔代号严格同序）。
    - 按 `probs` 降序排列，同分按字母序破平 → 同一状态两次采样必得同一组候选。
    """

    qid: str
    prompt: str                          # 教师请求文字（扁平，缓存键的原料）
    order: tuple[str, ...]               # 在场全部候选的字母序代号（长度 = 列数 C）
    letters: tuple[str, ...]             # 被选中的字母列（长度 = min(k, C)）
    options: tuple[str, ...]             # letters[i] 对应的候选代号
    probs: tuple[float, ...]             # top-k 内重归一的份额（和为 1）
    scores: tuple[float, ...]            # 末位原始分数（与 probs 同序）
    mass: float                          # top-k 占全候选的原始份额（截断信息）
    sample_id: str = ""                  # f"{record_id}#{qid}"（装配台账与 run 追溯用）

    def to_dict(self) -> dict[str, Any]:
        """摊成普通字典（写 jsonl / run 配置用；frozen dataclass 直接进 json 会带类名）。

        白话：落账要的是"字段名对上号"的白纸黑字，不是一串读不懂的类名；份额与分数
        在这里各留各的精度（写账舍到 8 位与 6 位），够事后追溯，又不至于把账本撑肥。

        :returns: 本条采样的字典形式（键与出口契约一一对应）。
        """
        return {
            "sample_id": self.sample_id, "qid": self.qid, "prompt": self.prompt,
            "order": list(self.order), "letters": list(self.letters), "options": list(self.options),
            "probs": [round(v, 8) for v in self.probs], "scores": [round(v, 6) for v in self.scores],
            "mass": round(self.mass, 8), "k": len(self.letters),
        }


def topk_probs(scores: Sequence[float], *, k: int = DEFAULT_TOP_K, temperature: float = 1.0) -> tuple[list[int], list[float]]:
    """纯函数：一列候选分数 → 前 k 名的下标与重归一份额（RL 栈可直接复用，不碰任何模型）。

    白话：给一组"每档多有可能"的分数，挑出最有底气的前几名；把这几名自己的份额重新摊成一锅汤
    （和为 1），前面的名次也照大小排好。名次同分就按谁排在候选表前面算谁赢，所以两次一样。

    :param scores: 一列分数（logits 或任意单调同向的量）。
    :param k: 取前几名；超过候选数就全取（spec 场景"≤8 个"的"≤"由此来）。
    :param temperature: 折份额前的同除正数（只调陡缓、不改名次，与 p2-02 同一单调性保证）。
    :returns: `(下标列表, 份额列表)`，两者等长；份额和为 1。
    :raises ValueError: 分数为空、k<1、倍数非正时抛出。
    """
    values = [float(v) for v in scores]
    if not values:
        raise ValueError("scores 为空：没有候选就没有 top-k")
    if int(k) < 1:
        raise ValueError(f"k 必须是正整数（取前几名），实得 {k!r}")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature 必须是有限正数（同除正数不改名次），实得 {temperature!r}")
    # 同分破平：按 (-分数, 下标) 排序，下标小的赢——即渲染层的字母序，保证可复现
    ranked = sorted(range(len(values)), key=lambda i: (-values[i], i))[: int(k)]
    scaled = [values[i] / float(temperature) for i in ranked]
    top = max(scaled)                                   # 先减最大值防指数上溢，折出的份额不变
    weights = [math.exp(v - top) for v in scaled]
    total = math.fsum(weights)
    return ranked, [w / total for w in weights]


#: 一道题的规格里会出现的键；出现任何一个就按"单题"读，否则按 `{qid: 规格}` 的多题表读
_QUESTION_KEYS = ("type", "criteria", "options", "instructions", "question")


def _rows_of(state: Any, spec: Any) -> list[tuple[str, dict[str, Any]]]:
    """把"状态 + 题目规格"归一成 `[(qid, 已渲染行)]`：单题、多题、已渲染行三种写法都收。

    白话：有人一次只给一道题，有人把三道题挂在同一个状态上，也有人题目早就在前台排好了版式。
    这里统一收成"一题一行"的队列——排版那活儿仍归 P1 的冻结件管，本函数只管形状，
    并且分岔必须先认形状再排版：把单题规格误读成"多题表"，qid 就会变成 `criteria` 这种
    字段名，教师请求文字跟着错位，缓存键也就对不上当年产的那一行。

    :param state: 决策状态（字符串/dict），或本身就是已渲染行。
    :param spec: 单题规格 dict、`{qid: 规格}` 多题表、或 None（配合已渲染行）。
    :returns: `[(qid, 已渲染行)]`，排版与样本契约校验已由 `from_systemone` 走完。
    :raises ValueError: 既给不出题目规格、state 又不是已渲染行时抛出。
    """
    if spec is None:
        if isinstance(state, Mapping) and "options" in state:       # 直接给已渲染行（RL replay 常见）
            return [(str(state.get("id") or "q"), dict(state))]
        raise ValueError("spec 为 None 时 state 必须是已渲染行（含 `options`），实得不可排版")
    if isinstance(spec, str) or not isinstance(spec, Mapping):      # 只给一串候选代号
        return [("q", from_systemone(state, {"type": "choice", "criteria": dict.fromkeys(spec, "")}, qid="q"))]
    if any(key in spec for key in _QUESTION_KEYS):                  # 单题规格
        return [("q", from_systemone(state, dict(spec), qid="q"))]
    return [(str(qid), from_systemone(state, dict(item), qid=str(qid)))
            for qid, item in spec.items() if isinstance(item, Mapping)]


def student_topk_scores(student: Student, prompts: Sequence[str], cols: int, *, lengths: Sequence[int] | None = None) -> torch.Tensor:
    """一批"学生请求文字"→ `(B, cols)` 末位候选分数（带图；唯一读点走 P1 的 `option_scores`）。

    白话：把几道题一起送进学生模型，只收每行写完最后一字那刻留下的那串数，再跟字母行配钥匙。
    "取哪一格、读哪几列"两条口径全留在 decision/ 那把尺子里，本文件不另摆一把。

    :param student: p2-05 的 `Student`（或鸭子同型的对象：`body/tokenizer/head_weight/letter_ids/pad_id`）。
    :param prompts: 学生侧请求文字序列（渲染层扁平文字，套壳在本函数内完成）。
    :param cols: 要读的字母列数（= 在场候选数，≤26）。
    :param lengths: 每行真符号数；None 时按掩码自算（右补空洞约定）。
    :returns: `(B, cols)` float32 分数（保留计算图，训练路直接往下接）。
    """
    if cols < 1 or cols > MAX_OPTIONS:
        raise ValueError(f"cols 需落在 [1, {MAX_OPTIONS}]，实得 {cols}")
    if not prompts:
        raise ValueError("prompts 为空：没有题就没有分数")
    encoded = _encode_prompts(student, prompts)
    hidden = student_forward(student, encoded)                       # (B,T,d) 带图
    rows = lengths if lengths is not None else [int(n) for n in encoded.mask.sum(dim=1)]
    return option_scores(hidden, student.head_weight, list(student.letter_ids[:cols]), rows)


def sample_topk(
    model: Student,
    state: Any,
    k: int = DEFAULT_TOP_K,
    *,
    spec: Any = None,
    temperature: float = 1.0,
    sample_id: str = "",
    use_current: bool = True,
) -> list[TopKSample]:
    """on-policy 采样器：学生自采样 top-k 选项分布（k 默认 8），一次调用一题一条出口。

    白话：把这道题按学生自己的样子问它一次，它报完"每档几成"后，这里只留它自己最有把握的
    前几名，并把这几名的份额重新摊成"和为一整锅"。这批候选随后就是蒸馏的靶子——
    学生在哪儿下注，教师就在哪儿给它看答案。

    :param model: 学生（p2-05 `Student` 或鸭子同型件）。
    :param state: 决策状态（字符串状态文本，或已渲染行 dict）。
    :param k: top-k 宽度（默认 8）。
    :param spec: 题目规格：单题 dict / `{qid: 规格}` 多题 / None（state 已是渲染行）。
    :param temperature: 折份额的同除倍数（只调陡缓，不改名次）。
    :param sample_id: 记录 id 前缀（进出口台账，事后能追到是哪道题）。
    :param use_current: False 时在 no_grad 下采样（评测/replay 用，不占显存里的来路记录）。
    :returns: `list[TopKSample]`，每题一条（形状契约见 `TopKSample` docstring）。
    """
    out: list[TopKSample] = []
    rendered = _rows_of(state, spec)
    cols = max(len(option_order(row)) for _, row in rendered)
    prompts = [_student_prompt(model, row) for _, row in rendered]
    if use_current:
        scores = student_topk_scores(model, prompts, cols)
    else:
        with torch.no_grad():
            scores = student_topk_scores(model, prompts, cols)
    for j, (qid, row) in enumerate(rendered):
        order = option_order(row)
        vec = [float(v) for i, v in enumerate(scores[j].tolist()) if i < len(order)]
        idx, probs = topk_probs(vec, k=k, temperature=temperature)
        out.append(TopKSample(
            qid=qid, prompt=prompt_text(row, order), order=tuple(order),
            letters=tuple(LETTERS[i] for i in idx), options=tuple(order[i] for i in idx),
            probs=tuple(probs), scores=tuple(vec[i] for i in idx),
            mass=_subset_mass(vec, idx), sample_id=f"{sample_id}#{qid}" if sample_id else qid,
        ))
    return out


def _subset_mass(vec: Sequence[float], idx: Sequence[int]) -> float:
    """top-k 在全候选池里的原始份额（softmax 域）——"截断截掉了多少信息"的度量。"""
    top = max(vec)
    weights = [math.exp(v - top) for v in vec]
    total = math.fsum(weights) or 1.0
    return math.fsum(weights[i] / total for i in idx)


def _student_prompt(student: Student, row: dict[str, Any]) -> str:
    """把已渲染行套成学生要读的扁平文字（对话模板由编号器负责，本层只交原始题面）。"""
    return prompt_text(row, option_order(row))


@dataclass
class _Encoded:
    """一批题面补好空洞后的最小批件：只带 `ids/mask/lengths`（右补空洞，与 p2-05 同口径）。

    白话：把几道题排齐送进模型，只需要"号"和"哪些格子看得见"这两张表，外加每行真写了
    几个字好去取末位；这里不掺份额表——采样阶段还没有目标，硬造一张假表只会误导读代码的人。
    """

    ids: torch.Tensor               # (B,T)
    mask: torch.Tensor              # (B,T) 0/1 可见位
    lengths: list[int]              # 每行真符号数

    @property
    def tokens(self) -> int:
        """补齐后的字数（吞吐口径分母，与 p2-05 `Batch.tokens` 同义）。"""
        return int(self.ids.numel())


def _encode_prompts(student: Student, prompts: Sequence[str]) -> _Encoded:
    """一批题面 → 编号批件：逐条套模板编号，再右补空洞并记下每行真长度。"""
    rows = [_tokenize(student, p) for p in prompts]
    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), int(student.pad_id), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for j, r in enumerate(rows):
        ids[j, : len(r)] = torch.tensor(r, dtype=torch.long)
        mask[j, : len(r)] = 1
    return _Encoded(ids=ids, mask=mask, lengths=[len(r) for r in rows])


def _tokenize(student: Student, text: str) -> list[int]:
    """题面 → 编号串：套对话模板并显式关思考（模板不认这个开关就退回不带，口径与 p2-02 一致）。"""
    messages = [{"role": "system", "content": SYSTEM_LINE}, {"role": "user", "content": text}]
    tok = student.tokenizer
    try:
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except (TypeError, ValueError):
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return [int(i) for i in tok(rendered, add_special_tokens=False)["input_ids"]]


SYSTEM_LINE = (
    "Apply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only."
)


# ================================================================ 工作项 B：教师同选项分布
def teacher_full_dist(
    cache: DistCache | None, teacher_model_id: str, prompt: str, letters: Sequence[str]
) -> dict[str, float] | None:
    """只查缓存取教师在**全部在场候选**上的份额；查不到就交 None，绝不顺手替教师算一遍。

    白话：拿着"哪个教师、问了什么、给了哪几档"去账本上对门牌号。对上了原样抄回来，
    对不上就明说没查到——训练循环里最忌讳的就是自作主张补算，那笔算力与这份份额的
    出处都会说不清（design：教师分布 MUST 走离线缓存）。
    """
    if cache is None:
        return None
    return cache.lookup(teacher_model_id, prompt, [str(v) for v in letters])


def restrict_normalize(dist: Mapping[str, float], keys: Sequence[str]) -> dict[str, float]:
    """把一份份额表限制到指定列并重归一（和仍为 1）；列不在表里按 0 计。

    白话：教师原本对二十档各报了几成，现在只看其中的八档。把这八档的份额挑出来、
    再摊回一整锅就是。这里有条能站住的等价式：softmax 先在全池上折、再挑子集重摊，
    等于直接对子集折它（分子分母同除的那个总分相消），温度缩放同样可交换
    ——于是"学生自采样挑中的那几档上的份额"不必为每一种组合另产一份教师标。

    :param dist: `{候选代号: 份额}`（不必和为 1，本函数负责归一）。
    :param keys: 要保留的候选代号（顺序即出口顺序）。
    :returns: `{key: 归一后份额}`，`keys` 上的和为 1。
    :raises ValueError: 保留列上的总份额为 0（没有可归一的东西）时抛出。
    """
    wanted = [str(v) for v in keys]
    vals = [float(dist.get(k, 0.0) or 0.0) for k in wanted]
    total = math.fsum(vals)
    if total <= 0.0:
        raise ValueError(f"选中的列上总份额为 0，无法归一：{wanted} ← 源表 {sorted(dist)}")
    return {k: v / total for k, v in zip(wanted, vals)}


def teacher_on_subset(
    cache: DistCache | None,
    teacher_model_id: str,
    prompt: str,
    letters: Sequence[str],
    subset: Sequence[str],
    *,
    teacher: Any = None,
    allow_online_teacher: bool = False,
) -> dict[str, float] | None:
    """教师在"学生自采样出来的那 k 档"上的归一化分布（缓存先行；在线仅作云端产标档）。

    白话：先按整道题的全部候选去账本里问教师，问到再把份额摊到学生挑的那几档上
    （等价于直接问那几档，理由见 `restrict_normalize`）。只有明确开了在线档、又给了
    教师本尊，才允许真去跑一次前向——那是 C3 产标的活儿，训练循环里不许走。

    :param cache: p2-02 缓存句柄（只读即可）。
    :param teacher_model_id: 缓存键里的教师身份。
    :param prompt: 教师请求文字（扁平，不含对话模板）。
    :param letters: 全部在场候选的字母列（缓存键的候选集）。
    :param subset: 学生 top-k 选中的字母列。
    :param teacher: `TextTeacher` 本尊（仅 `allow_online_teacher=True` 时才碰）。
    :param allow_online_teacher: 缓存未命中时是否允许在线补算（默认关：MUST 走缓存）。
    :returns: `{subset 字母: 归一份额}`，或 None（查不到且未开在线档）。
    """
    if allow_online_teacher and teacher is not None:
        full = teacher.score_options(prompt, list(letters))   # 教师门面自带"缓存先行"：命中同样零前向
    else:
        full = teacher_full_dist(cache, teacher_model_id, prompt, letters)  # 训练档只查账本，绝不补算
    if full is None:
        return None
    wanted = [str(v) for v in subset]
    # 缓存行的键按 p2-02 契约就是字母列（产标时给的是 A/B/C…）；一个都没对上就当未命中，
    # 绝不猜"大概是按代号存的"然后拿零份额去归一——那会当场抛错，把数据污染说成数值问题。
    if not any(k in full for k in wanted):
        return None
    return restrict_normalize(full, wanted)


# ================================================================ 工作项 B：稠密散度损失
def _as_tensor(value: Any, name: str) -> torch.Tensor:
    """把份额/log 概率一律收成 float32 张量（数值单测可直接传普通 list，省得每处造张量）。"""
    if isinstance(value, torch.Tensor):
        return value.float()
    return torch.tensor(value, dtype=torch.float32)


def _keep_of(keep: Any, shape_like: torch.Tensor) -> torch.Tensor:
    """`keep` 缺席时补一张全真的布尔表（形状与目标一致）。"""
    if keep is None:
        return torch.ones_like(shape_like, dtype=torch.bool)
    k = keep if isinstance(keep, torch.Tensor) else torch.tensor(keep, dtype=torch.bool)
    if k.shape != shape_like.shape:
        raise ValueError(f"keep 形状 {tuple(k.shape)} 与目标 {tuple(shape_like.shape)} 不符")
    return k.to(shape_like.device).bool()


def selected_logp(scores: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
    """`(B,C)` 末位分数 → log 概率：位宽提到 fp32 算，`keep=False` 列先封 -inf 参归一、再归零。

    白话：只有"被选中要较真的那几档"参与摊份额，其余档当场封掉，摊完再把封掉的格子
    写成零——不写零的话，零份额乘上一个负无穷会得出不明所以的 nan（SFT 冒烟真踩过）。
    位宽之所以要先抬到 fp32，是因为半精度的指数减法在长尾列上会抖出 nan，design 风险条
    就是钉这一条的。

    :param scores: `(B,C)` 末位候选分数（可带图）。
    :param keep: `(B,C)` bool；False 的列不参与归一。
    :returns: `(B,C)` fp32 log 概率；`keep=False` 的列取 0.0。
    """
    if not isinstance(scores, torch.Tensor):
        raise TypeError(f"scores 需为张量 (B,C)，实得 {type(scores).__name__}")
    if scores.dim() != 2:
        raise ValueError(f"scores 需为 (B,C) 两维，实得 {tuple(scores.shape)}")
    x = scores.float()
    if keep is not None:
        x = x.masked_fill(~keep.bool(), float("-inf"))
    logp = torch.log_softmax(x, dim=-1)
    if keep is not None:
        logp = logp.masked_fill(~keep.bool(), 0.0)
    return logp


def reverse_kl(
    target: Any, logp: Any, *, keep: Any = None, reduction: str = "mean"
) -> torch.Tensor:
    """散度方向锁定：`KL(p_teacher ‖ p_student) = Σ_j p_t[j]·(log p_t[j] − log p_s[j])`。

    白话：以**教师**为参照量学生差多远——教师说哪一档有份额，那档就必须被学生摊到份额，
    摊不到就按教师的信心罚。于是教师同时看好几档时学生不敢只押一档（这叫 mode-covering，
    教师的多峰不会被抹平）。反过来的写法学生只需押中教师最高的那一峰就能少挨罚，
    多峰会被抹平，本域明确不要它（design D9 的业界口径，单测用不对称数值例钉死方向）。

    :param target: `(B,C)` 或 `(C,)` 教师份额（在参与归一的列内和为 1）。
    :param logp: 同形状的**学生** log 概率（`selected_logp` 的出口，带图）。
    :param keep: `(B,C)` bool；False 列整项跳过。
    :param reduction: `mean`（行均值，训练用）或 `none`（逐行散度，诊断/评测用）。
    :returns: 标量张量（mean）或 `(B,)` 张量（none）；非负。
    """
    p = _as_tensor(target, "target")
    lq = _as_tensor(logp, "logp")
    if p.shape != lq.shape:
        raise ValueError(f"target 与 logp 形状不符：{tuple(p.shape)} vs {tuple(lq.shape)}")
    mask = _keep_of(keep, p) & (p > _TARGET_FLOOR)        # 0·log 0 约定：教师零份额列不计
    lq_safe = torch.clamp(lq, min=_LOG_FLOOR)            # 学生侧极端列封地板，防 -inf 传开
    lp = torch.log(torch.clamp(p, min=torch.finfo(torch.float32).tiny))
    term = torch.where(mask, p * (lp - lq_safe), torch.zeros_like(p))
    rows = term.sum(dim=-1)
    return rows if reduction == "none" else rows.mean()


def jsd(target: Any, logp: Any, *, beta: float = DEFAULT_JSD_BETA, keep: Any = None,
        reduction: str = "mean") -> torch.Tensor:
    """JSD(β)：`β·KL(p‖m) + (1−β)·KL(q‖m)`，`m = β·p + (1−β)·q`（可切档，design 指定 β=0.5）。

    白话：不直接量"学生离教师多远"，而是先看两人的平均意见，各自再与这个平均比一比。
    这样两边都留了面子：教师错的地方不会像 reverse-KL 那样被无条件押死，学生的把握
    也不会被无脑摊平。β 决定这份面子偏向哪头，0.5 就是完全对称。

    :param target: `(B,C)`/`(C,)` 教师份额。
    :param logp: 同形状学生 log 概率。
    :param beta: 混合权重（落在 [0,1]）。
    :param keep: `(B,C)` bool；False 列不参与。
    :param reduction: `mean` 或 `none`。
    :returns: 标量或 `(B,)` 张量；非负。
    :raises ValueError: β 越界时抛出。
    """
    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError(f"beta 需落在 [0,1]，实得 {beta!r}")
    p = _as_tensor(target, "target")
    lq = _as_tensor(logp, "logp")
    if p.shape != lq.shape:
        raise ValueError(f"target 与 logp 形状不符：{tuple(p.shape)} vs {tuple(lq.shape)}")
    w = _keep_of(keep, p)
    q = torch.where(w, torch.exp(torch.clamp(lq, min=_LOG_FLOOR)), torch.zeros_like(lq))
    w = w & ((p > _TARGET_FLOOR) | (q > _TARGET_FLOOR))     # 双方都零的列：0·log 0 约定不计
    m = float(beta) * p + (1.0 - float(beta)) * q
    lm = torch.log(torch.clamp(m, min=torch.finfo(torch.float32).tiny))
    left = torch.where(w & (p > _TARGET_FLOOR), p * (torch.log(torch.clamp(p, min=torch.finfo(torch.float32).tiny)) - lm),
                       torch.zeros_like(p))
    right = torch.where(w & (q > _TARGET_FLOOR), q * (torch.log(torch.clamp(q, min=torch.finfo(torch.float32).tiny)) - lm),
                        torch.zeros_like(q))
    rows = (float(beta) * left + (1.0 - float(beta)) * right).sum(dim=-1)
    return rows if reduction == "none" else rows.mean()


def divergence(target: Any, logp: Any, *, loss: str = LOSS_REVERSE_KL, beta: float = DEFAULT_JSD_BETA,
               keep: Any = None, reduction: str = "mean") -> torch.Tensor:
    """两档散度的总入口：训练与单测都从这里进，避免"哪个函数才是官方口径"的分叉。

    白话：默认走 reverse-KL，想要对称那份就切 JSD。档名写错当场报错，不静默退回默认档
    ——消融实验里"以为切了其实没切"是最贵的错。
    """
    if loss == LOSS_REVERSE_KL:
        return reverse_kl(target, logp, keep=keep, reduction=reduction)
    if loss == LOSS_JSD:
        return jsd(target, logp, beta=beta, keep=keep, reduction=reduction)
    raise ValueError(f"未知散度档 {loss!r}（合法：{LOSSES}）")


def opd_loss(
    last_hidden: torch.Tensor,
    head_weight: torch.Tensor,
    letter_ids: Sequence[int],
    target: Any,
    *,
    keep: Any = None,
    lengths: Sequence[int] | None = None,
    loss: str = LOSS_REVERSE_KL,
    beta: float = DEFAULT_JSD_BETA,
) -> torch.Tensor:
    """OPD 稠密蒸馏损失：末位读出 → 只在自采样那 k 列上摊份额 → 与教师份额算散度。

    白话：一道题只有写到最后一个字那一刻的心气算数；而"哪些格子要算"是学生自己挑的
    top-k。挑中之后的格子里，把学生的份额与教师的份额摆一起，看差多少。
    "取哪一格、读哪几列"两条口径全部交给 decision/ 那把现成的尺子（`option_scores`），
    本函数绝不另摆一把，免得两条路各说各话。

    :param last_hidden: `(B,T,d)` 逐位置输出（带图，来自 `student_forward`）。
    :param head_weight: `(vocab,d)` 输出层权重（冻结，调用侧先 detach）。
    :param letter_ids: 在场候选的字母编号（列序 = 渲染层字母序）。
    :param target: `(B,C)` 教师（或掺完真值后）份额；每行在 `keep` 列内和为 1。
    :param keep: `(B,C)` bool：学生 top-k 选中的列；None 表示全列都较真（消融 off 档）。
    :param lengths: 每行真符号数；None 表示按整行最右一格读。
    :param loss: 散度档（`reverse_kl` / `jsd`）。
    :param beta: JSD 的混合权重。
    :returns: 标量损失（行均值）。
    """
    tgt = _as_tensor(target, "target")
    if tgt.dim() != 2:
        raise ValueError(f"target 需为 (B,C) 两维，实得 {tuple(tgt.shape)}")
    if tgt.shape[1] != len(letter_ids):
        raise ValueError(f"目标列数 {tgt.shape[1]} 与字母数 {len(letter_ids)} 不符")
    kmask = _keep_of(keep, tgt)
    scores = option_scores(last_hidden, head_weight, letter_ids, lengths)      # (B,C)：唯一读点
    logp = selected_logp(scores, kmask)
    return divergence(tgt, logp, loss=loss, beta=beta, keep=kmask)


def mixed_target(teacher_sub: Mapping[str, float], gt_sub: Mapping[str, float] | None,
                 mix_gt: float = DEFAULT_MIX_GT) -> list[float]:
    """把人工真值按比例 `mix_gt` 掺进教师份额（design：默认 0.2），出口按教师键序。

    白话：教师说这几档各占几成，人工答案说"就该这一档"。听谁的多，写在配置里；
    掺完重新归一，免得掺着掺着一锅汤变成了 1.7 成。

    :param teacher_sub: 教师 top-k 份额（键序 = 目标列序）。
    :param gt_sub: 人工真值折出的同列份额；None 表示这题没人工答案，原样交教师那份。
    :param mix_gt: 真值的权重（落在 [0,1]）。
    :returns: 与 `teacher_sub` 键序一致的目标列（和为 1）。
    """
    keys = list(teacher_sub)
    t = [float(teacher_sub[k]) for k in keys]
    if gt_sub is None:
        return t
    g = [float(gt_sub.get(k, 0.0) or 0.0) for k in keys]
    return mix_targets(t, g, float(mix_gt))


# ================================================================ 工作项 B2：装配（采样→教师→目标）
@dataclass
class OpdSample:
    """一条 on-policy 装配好的样本：编号串 + 列序 + 选中列 + 教师份额 + 掺完的目标。

    白话：把"学生会怎么读这道题、教师在那几档上各说几成、最后要往哪份份额上逼"三件事
    一次性钉在同一行里；进批次时只带数字，教师请求文字留着是为了下次还能对上同一道题。
    """

    sample_id: str
    ids: tuple[int, ...]                    # 学生侧编号串
    letters: tuple[str, ...]                # 在场全候选的字母列（列序 = 字母序）
    subset: tuple[str, ...]                 # 学生 top-k 选中的字母列
    teacher: tuple[float, ...]              # 教师在 subset 内归一、其余列为 0（全列长）
    target: tuple[float, ...]               # 掺完真值的目标（全列长，subset 内和为 1）
    prompt: str                             # 教师请求文字（缓存键原料）
    qtype: str
    source: str
    cache_key: str = ""                     # 命中的缓存门牌号（追溯是哪一行伪标教的）
    mass: float = 1.0                       # top-k 占全候选的原始份额
    has_teacher: bool = True

    @property
    def keep(self) -> tuple[bool, ...]:
        """全列上的选中位图（`collate_opd` 与损失层的 `keep` 输入）。"""
        return tuple(v in set(self.subset) for v in self.letters)


@dataclass
class OpdBatch:
    """一次 OPD 前向要吃的一微批：编号、可见位、目标份额、选中列、每行真长度。

    白话：几道题排齐了送进模型。短的那几行尾巴上补空洞，所以另带"每行真写了几个字"；
    每题选中的档各不相同，于是列格子按最多候选的那行开，选中与否写进 `keep` 表。
    """

    ids: torch.Tensor               # (B,T)
    mask: torch.Tensor              # (B,T) 0/1 可见位
    target: torch.Tensor            # (B,C) 每行在 keep 列内和为 1
    keep: torch.Tensor              # (B,C) bool：学生自采样选中的列
    letter_ids: list[int]           # 与列同序的字母编号
    lengths: list[int]              # 每行真符号数（右补空洞约定）
    cols: int

    def to_(self, device: str | torch.device) -> "OpdBatch":
        """把四张表搬到指定设备（就地返回自己，方便链式；CPU 冒烟时是空操作）。

        白话：题面、可见位、目标份额、选中列这四份东西必须跟着模型走，否则一半在加速卡上、
        一半还留在内存里，算出来的就是错桌的账。搬完把自己交回去，调用方一行接得上下一步。

        :param device: 目标设备（如 `cpu`，或加速卡编号）。
        :returns: 搬完的自己（就地改，不复制第二份）。
        """
        self.ids = self.ids.to(device)
        self.mask = self.mask.to(device)
        self.target = self.target.to(device)
        self.keep = self.keep.to(device)
        return self

    @property
    def tokens(self) -> int:
        """补齐后的字数（吞吐口径分母，与 p2-05 `Batch.tokens` 同义）。"""
        return int(self.ids.numel())


def collate_opd(samples: Sequence[OpdSample], idx: Sequence[int], *, pad_id: int,
                letter_map: Sequence[int]) -> OpdBatch:
    """把若干样本下标拼成微批张量表（右补空洞；列格子 = 在场候选最多那行）。

    白话：一桌题字数不一、候选档数也不一，拼批就是把短的那些尾巴上补齐、把列格子开到
    最宽那题的宽度，再另带两张便条：哪几格是学生自己挑中的、每题真写了几个字。读点只认
    "真写完"那一刻，所以尾巴上的空洞绝不能被当成最后一个字。

    :param samples: 样本表。
    :param idx: 进批的下标。
    :param pad_id: 空洞用的编号。
    :param letter_map: 26 枚字母的编号表（A 在第一个）。
    :returns: `OpdBatch`。
    """
    rows = [samples[i] for i in idx]
    width = max(len(r.ids) for r in rows)
    cols = max(len(r.letters) for r in rows)
    ids = torch.full((len(rows), width), int(pad_id), dtype=torch.long)
    mask = torch.zeros_like(ids)
    target = torch.zeros((len(rows), cols), dtype=torch.float32)
    keep = torch.zeros((len(rows), cols), dtype=torch.bool)
    for j, r in enumerate(rows):
        n = len(r.ids)
        ids[j, :n] = torch.tensor(list(r.ids), dtype=torch.long)
        mask[j, :n] = 1
        c = len(r.letters)
        target[j, :c] = torch.tensor(list(r.target)[:c], dtype=torch.float32)
        keep[j, :c] = torch.tensor(list(r.keep)[:c], dtype=torch.bool)
    lengths = [int(mask[j].sum()) for j in range(len(rows))]
    return OpdBatch(ids=ids, mask=mask, target=target, keep=keep, cols=cols,
                    letter_ids=[int(v) for v in letter_map[:cols]], lengths=lengths)


def _gt_over_subset(record: dict[str, Any], qid: str, order: Sequence[str],
                    subset_letters: Sequence[str], *, mode: str, hard_mass: float) -> dict[str, float] | None:
    """把人工份额折成"与 top-k 同列"的份额：`hard` 档取最押的那档再按 p2-02 折法摊，`dist` 档直摊。

    白话：真值有两种用法。一是当作答——只认占得最多那档，其余按 p2-02 的现成折法摊一点
    （与 p2-05 同口径，消融才可比）；二是照原样信这份份额。没人工答案就交 None，
    那一题就只有教师说话。
    """
    gold = ((record.get("sample") or {}).get("targets") or {}).get(qid)
    if not isinstance(gold, dict) or not gold:
        return None
    keys = list(order)
    vals = [float(gold.get(k, 0.0) or 0.0) for k in keys]
    if math.fsum(vals) <= 0:
        return None
    if mode == "hard":
        pos = max(range(len(keys)), key=lambda i: vals[i])        # 最押那一档的候选序位
        spread = hard_distribution(LETTERS[pos], list(LETTERS[: len(keys)]), mass=hard_mass)
        picked = {ch: float(spread.get(ch, 0.0)) for ch in subset_letters}      # ch 本身就是字母列
        return restrict_normalize(picked, list(subset_letters))
    if mode == "dist":
        picked = {}
        for ch in subset_letters:
            pos = LETTERS.index(ch)                       # 字母列 → 候选序位
            picked[ch] = float(gold.get(order[pos], 0.0) or 0.0) if pos < len(order) else 0.0
        if math.fsum(picked.values()) <= 0:
            return None
        return restrict_normalize(picked, list(subset_letters))
    raise ValueError(f"gt_mode 只能是 hard/dist，实得 {mode!r}")


def collect_onpolicy(
    records: Sequence[dict[str, Any]],
    student: Student,
    *,
    cache: DistCache | None = None,
    teacher_model_id: str = DEFAULT_TEACHER_MODEL_ID,
    k: int = DEFAULT_TOP_K,
    mix_gt: float = DEFAULT_MIX_GT,
    gt_mode: str = "hard",
    hard_mass: float = DEFAULT_HARD_MASS,
    max_length: int = int(BASELINE_LORA["max_length"]),
    chunk: int = 8,
    limit: int = 0,
    teacher: Any = None,
    allow_online_teacher: bool = False,
) -> tuple[list[OpdSample], AsmStats, TeacherStats | None]:
    """装配 on-policy 训练料：学生自采样 top-k → 教师同选项份额 → 掺真值成目标。

    白话：这道题先让学生自己报"我最看好哪几档"（k 档，默认八档），再拿教师在**同样这几档**
    上的份额当答案，人工答案按配置掺一点进去，最后要做的就是把学生往那份份额上逼。
    整趟只在缓存上查教师——默认一次教师前向都不发（本函数返回的账本会自证这一点）。

    :param records: p2-03 registry 交出的信封行。
    :param student: 学生（`Student` 或鸭子同型件）。
    :param cache: 教师分布缓存（只读）。
    :param teacher_model_id: 教师身份号（缓存键的一部分）。
    :param k: top-k 宽度；`0` 表示不截断（消融 off 档 = 退化为全候选 soft-KD）。
    :param mix_gt: 人工真值权重。
    :param gt_mode: 真值折法（`hard` 与 p2-05 同口径 / `dist` 直接用人工份额）。
    :param hard_mass: `hard` 档的押注质量。
    :param max_length: 单条编号串上限，超出丢弃（不截半截题）。
    :param chunk: 采样器一趟送几道题（一次前向打多行，与 p2-02 批量口径一致）。
    :param limit: 编成条数上限（0=不限）。
    :param teacher: `TextTeacher` 本尊，**只用来记账**（`stats.forward_calls` 是零前向断言的源）。
    :param allow_online_teacher: 缓存未命中时允许在线补算（仅云端产标档；训练档必须关）。
    :returns: `(samples, 装配台账, teacher.stats)`。
    """
    stats = AsmStats()
    out: list[OpdSample] = []
    pending: list[tuple[dict[str, Any], str, dict[str, Any], list[str]]] = []
    for record in records:
        stats.seen += 1
        sample = (record.get("sample") or {})
        questions = sample.get("questions") or {}
        for qid, spec in questions.items():
            if not isinstance(spec, dict):
                continue
            row = from_systemone(sample.get("state"), spec, qid=str(qid))
            order = option_order(row)
            if not order or len(order) > MAX_OPTIONS:
                stats.skipped["too_many_options"] = stats.skipped.get("too_many_options", 0) + 1
                continue
            pending.append((record, str(qid), row, order))
            break                                        # 一信封只取第一题（与 p2-05 同口径）
        if limit and len(pending) >= limit:
            break

    prompts = [_student_prompt(student, row) for _, _, row, _ in pending]
    flat: list[tuple[int, list[int], list[float], list[float]]] = []   # (队列位, 选中下标, k 份额, 全池分数)
    cols = max((len(order) for _, _, _, order in pending), default=0)
    for start in range(0, len(pending), max(1, int(chunk))):
        part = prompts[start: start + max(1, int(chunk))]
        scores = student_topk_scores(student, part, cols)
        for offset, row_scores in enumerate(scores.tolist()):
            j = start + offset
            n = len(pending[j][3])
            vec = [float(v) for v in row_scores[:n]]
            width = n if int(k) <= 0 else min(int(k), n)
            idx, probs = topk_probs(vec, k=width)
            flat.append((j, idx, probs, vec))

    for j, idx, probs, vec in flat:
        record, qid, row, order = pending[j]
        subset = [LETTERS[i] for i in idx]
        letters = LETTERS[: len(order)]
        prompt = prompt_text(row, order)
        tsub = teacher_on_subset(cache, teacher_model_id, prompt, letters, subset,
                                 teacher=teacher, allow_online_teacher=allow_online_teacher)
        if tsub is None:
            stats.misses += 1
            continue
        stats.hits += 1
        gt = _gt_over_subset(record, qid, order, subset, mode=gt_mode, hard_mass=hard_mass)
        mixed = mixed_target(tsub, gt, mix_gt)
        full_target = [0.0] * len(letters)
        for pos, ch in enumerate(subset):
            full_target[LETTERS.index(ch)] = mixed[pos]
        ids = _tokenize(student, prompt)
        if not ids or len(ids) > max_length:
            stats.skipped["too_long"] = stats.skipped.get("too_long", 0) + 1
            continue
        out.append(OpdSample(
            sample_id=f"{record.get('id', '?')}#{qid}", ids=tuple(ids), letters=tuple(letters),
            subset=tuple(subset), teacher=tuple(float(tsub.get(ch, 0.0)) for ch in letters), target=tuple(full_target),
            prompt=prompt, qtype=str(row.get("type") or record.get("qtype") or "choice"),
            source=str(record.get("task") or ""), cache_key=make_key(teacher_model_id, prompt, list(letters)),
            mass=_subset_mass(vec, idx),
        ))
        if limit and len(out) >= limit:
            break
    stats.encoded = len(out)
    return out, stats, (getattr(teacher, "stats", None) if teacher is not None else None)


# ================================================================ 工作项 C：训练与消融
def opd_step(
    student: Student,
    batch: OpdBatch,
    *,
    loss: str = LOSS_REVERSE_KL,
    beta: float = DEFAULT_JSD_BETA,
) -> torch.Tensor:
    """一微批的 OPD 前向：带图读出 → 只在自采样列上与教师算散度，回标量损失（反向由调用方发起）。

    白话：把这一桌题送进学生，收每行末位那串数，只在它自己挑中的那几档上跟教师比差距。
    这里不 `backward`，为的是让单测能把"一步"当纯函数使——给同样的数就该得同样的损失。
    """
    hidden = student_forward(student, batch)
    return opd_loss(hidden, student.head_weight, batch.letter_ids, batch.target,
                    keep=batch.keep, lengths=batch.lengths, loss=loss, beta=beta)


def default_cfg() -> dict[str, Any]:
    """默认配置 = p2-05 基线超参 + OPD 专属三件（k/mix_gt/散度档）+ 本地 CPU 冒烟档。

    白话：底单照抄 SFT 那一档（消融才可比），只多三件事：学生自己挑几档、真值掺几分、
    用哪一种散度。`mix_soft` 一栏是给 p2-05 的产物小抄留的兼容位，语义等于"教师说话的分量"
    （= 1 − mix_gt），别让下游读到一个不存在的名字。
    """
    return {
        # ── 资源与身份 ──
        "backbone": "qwen3-0.6b", "loader": LOADER_MINIMAL, "device": "cpu", "dtype": "float32",
        "threads": 4, "seed": 20261006, "cache_dir": None,
        # ── 数据（p2-03 registry 口）──
        "axis": "quality", "data_dir": None, "limit": 256, "max_length": 512,
        # ── 教师（p2-02 缓存口；训练循环零教师前向）──
        "teacher_model_id": DEFAULT_TEACHER_MODEL_ID,
        "teacher_cache": "bench/teacher_cache/p2_05_pseudo",
        "allow_online_teacher": False,
        # ── OPD 专属 ──
        "top_k": DEFAULT_TOP_K, "mix_gt": DEFAULT_MIX_GT, "gt_mode": "hard",
        "hard_mass": DEFAULT_HARD_MASS, "loss": LOSS_REVERSE_KL, "jsd_beta": DEFAULT_JSD_BETA,
        "sample_chunk": 8,
        # ── 起点（p2-05 SFT 产物：续训的来路必须能被追溯）──
        "parent_run_id": "", "parent_model": "", "opd_on": True,
        # ── 旁路与算子档（p2-13；垫片仍由 p2-05 提供）──
        "kernel_backend": BACKEND_TORCH, "ascend_target": "cpu",
        "r": int(BASELINE_LORA["r"]), "alpha": int(BASELINE_LORA["alpha"]),
        "dropout": float(BASELINE_LORA["dropout"]), "gradient_checkpointing": False,
        # ── 步长与调度（StartLux 基线，与 p2-05 同）──
        "lr": float(BASELINE_LORA["lr"]), "warmup_ratio": float(BASELINE_LORA["warmup_ratio"]),
        "weight_decay": float(BASELINE_LORA["weight_decay"]),
        "max_grad_norm": float(BASELINE_LORA["max_grad_norm"]),
        "epochs": 1.0, "max_tokens": 2048, "accum": 1, "max_steps": 50,
        # ── 产物与记录 ──
        "run_prefix": "p2-06-dev", "out_dir": None, "runs_root": None,
        "eval_axis": "quality", "eval_limit": 64,
        # p2-05 产物小抄的兼容位（= 教师分量）
        "mix_soft": 1.0 - DEFAULT_MIX_GT,
    }


def load_config(path: str | Path | None, *, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """三层合流（默认档 ← yaml ← 显式覆盖），未知键当场报错——拼错一个字母不该静默失效。

    白话：底单是本域自己的，但 yaml 那层的读法直接借 p2-05 的合流器（同一份"未知键即报错"
    的规矩），免得两处各写一遍、将来一边严格一边宽松。
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
    if cfg["loss"] not in LOSSES:
        raise ValueError(f"未知散度档 {cfg['loss']!r}（合法：{LOSSES}）")
    if cfg["gt_mode"] not in ("hard", "dist"):
        raise ValueError(f"gt_mode 只能是 hard/dist，实得 {cfg['gt_mode']!r}")
    cfg["mix_soft"] = 1.0 - float(cfg["mix_gt"])                  # 产物小抄的教师分量
    return cfg


def train(
    cfg: dict[str, Any],
    *,
    student: Student | None = None,
    cache: DistCache | None = None,
    teacher: Any = None,
) -> dict[str, Any]:
    """跑一条 OPD 续训：装起点 → 自采样装配 → 逐微批散度下降 → 落产物 + run 四件套。

    白话：先把 p2-05 训好的那副垫片贴回学生身上（`parent_model`），再让学生自己挑题面上
    最有把握的几档、拿教师在那几档上的份额当答案，一桌一桌地把差距压小。跑完交三样：
    能贴回去的垫片、一张怎么读答案的小抄、一本流水账——账里明写着起点是哪一次 SFT run。

    :param cfg: `default_cfg()/load_config()` 出来的配置表。
    :param student: 注入的学生（本地单测/接口自测用；None 则按 cfg 载真件）。
    :param cache: 注入的教师缓存（None 则按 `teacher_cache` 目录只读打开）。
    :param teacher: 注入的 `TextTeacher`，只用于记账（零前向断言的账本来源）。
    :returns: 事实表（run_id、parent_run_id、步数、首末损失、教师前向数、产物目录……）。
    :raises RuntimeError: 一条样本都装配不出来（缓存没覆盖到这批题）时抛出。
    """
    torch.set_num_threads(int(cfg["threads"]))
    random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))

    if student is None:
        student = build_student(cfg["backbone"], loader=cfg["loader"], device=cfg["device"],
                                dtype=cfg["dtype"], cache_dir=cfg["cache_dir"])
    loras = inject_lora(student.body, r=int(cfg["r"]), alpha=int(cfg["alpha"]),
                        dropout=float(cfg["dropout"]), backend=cfg["kernel_backend"],
                        ascend_target=cfg["ascend_target"])
    counts = freeze_all_but_lora(student.model, loras)
    loaded = load_adapter(loras, cfg["parent_model"]) if cfg.get("parent_model") else 0
    enable_gradient_checkpointing(student, bool(cfg["gradient_checkpointing"]))

    if cache is None and cfg.get("teacher_cache") and Path(str(cfg["teacher_cache"])).is_dir():
        cache = DistCache(root=cfg["teacher_cache"], read_only=True).load()   # 只读：绝不改教师账本
    records = load_quality_records(cfg["axis"], data_dir=cfg.get("data_dir"))
    k = int(cfg["top_k"]) if cfg["opd_on"] else 0                 # off 档 = 不截断（退化 soft-KD）
    samples, asm, tstats = collect_onpolicy(
        records, student, cache=cache, teacher_model_id=cfg["teacher_model_id"], k=k,
        mix_gt=float(cfg["mix_gt"]), gt_mode=cfg["gt_mode"], hard_mass=float(cfg["hard_mass"]),
        max_length=int(cfg["max_length"]), chunk=int(cfg["sample_chunk"]), limit=int(cfg["limit"]),
        teacher=teacher, allow_online_teacher=bool(cfg["allow_online_teacher"]),
    )
    if not samples:
        raise RuntimeError(
            f"一条 OPD 样本都没装配出来（见 {asm.seen} 条，丢弃 {asm.skipped}，教师命中 {asm.hits}/"
            f"未命中 {asm.misses}）——检查 {cfg['teacher_cache']!r} 是否覆盖这批题，或放大 --limit"
        )

    params = [p for p in student.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    buckets = bucket_batches(samples, max_tokens=int(cfg["max_tokens"]), shuffle=True, seed=int(cfg["seed"]))
    total_steps = int(cfg["max_steps"]) or max(1, int(len(buckets) * float(cfg["epochs"]) // max(1, int(cfg["accum"]))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lambda(total_steps, float(cfg["warmup_ratio"])))

    run = new_run(f"{cfg['run_prefix']}-opd-{cfg['backbone']}-{cfg['kernel_backend']}", {
        "stage": f"p2-06-opd/{cfg['run_prefix']}", "opd_version": OPD_VERSION,
        "parent_run_id": cfg["parent_run_id"], "parent_model": cfg["parent_model"],
        "opd_on": bool(cfg["opd_on"]), "ablation_of": cfg.get("ablation_of", ""),
        "config": {key: cfg[key] for key in sorted(cfg)}, "assemble": asm.as_dict(),
        "student": student.summary() if hasattr(student, "summary") else {"backbone": cfg["backbone"]},
        "lora_modules": sorted(loras), "counts": counts, "adapter_tensors_loaded": loaded,
        "onpolicy": {"top_k": k, "mix_gt": float(cfg["mix_gt"]), "gt_mode": cfg["gt_mode"],
                     "loss": cfg["loss"], "jsd_beta": float(cfg["jsd_beta"]),
                     "mean_mass": round(sum(s.mass for s in samples) / len(samples), 6)},
        "kernels": {"backend": cfg["kernel_backend"], "ascend_target": cfg["ascend_target"]},
    }, **({"root": cfg["runs_root"]} if cfg.get("runs_root") else {}))

    forward_before = int(getattr(tstats, "forward_calls", 0) or 0)
    accum, step, micro = max(1, int(cfg["accum"])), 0, 0
    losses: list[float] = []
    running, tokens_all, seconds_all = 0.0, 0, 0.0
    started = time.time()
    student.model.train()
    for epoch in range(max(1, math.ceil(float(cfg["epochs"])))):
        for idx in buckets:
            if step >= total_steps:
                break
            batch = collate_opd(samples, idx, pad_id=student.pad_id, letter_map=student.letter_ids).to_(cfg["device"])
            loss = opd_step(student, batch, loss=cfg["loss"], beta=float(cfg["jsd_beta"])) / accum
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
                run.log_metrics(step, loss=round(value, 6), lr=lr, micro=micro,
                                step_seconds=round(step_seconds, 3),
                                tokens_per_second=round(batch.tokens / max(step_seconds, 1e-6), 1),
                                tokens=tokens_all, grad_norm=round(gnorm, 4),
                                teacher_forward_calls=int(getattr(tstats, "forward_calls", 0) or 0) - forward_before)
                running = 0.0
                print(f"[{step}/{total_steps}] loss {value:.4f} lr {lr:.2e} "
                      f"{step_seconds:.2f}s/{step} tokens {tokens_all}", flush=True)
            if step >= total_steps:
                break

    teacher_forward = int(getattr(tstats, "forward_calls", 0) or 0) - forward_before
    result: dict[str, Any] = {
        "run_id": run.run_id, "parent_run_id": cfg["parent_run_id"], "opd_on": bool(cfg["opd_on"]),
        "steps": step, "samples": len(samples), "losses": losses,
        "first_loss": round(sum(losses[:max(1, min(10, len(losses)))]) / max(1, min(10, len(losses))), 6),
        "final_loss": round(sum(losses[-max(1, min(10, len(losses))):]) / max(1, min(10, len(losses))), 6),
        "teacher_forward_calls": teacher_forward, "teacher_cache": (cache.stats if cache is not None else None),
        "trainable": counts.get("trainable"), "seconds": round(time.time() - started, 2),
        "assemble": asm.as_dict(),
    }
    out_dir = cfg.get("out_dir") or (Path(run.path) / "model")
    listing = write_model_dir(out_dir, student=student, loras=loras, cfg=cfg, asm=asm, result=result)
    result["model_dir"] = listing["dir"]                    # 交目录本身（与 p2-05 同口径，评出口要拿它当路径）
    result["model_files"] = listing["files"]
    run.log_metrics(step, **{"final_" + key: value for key, value in result.items()
                             if key not in ("losses", "assemble") and isinstance(value, (int, float, str, bool))})
    _write_opd_notes(run, cfg, result)
    result["run_dir"] = str(run.path)
    return result


def _write_opd_notes(run: Any, cfg: dict[str, Any], result: dict[str, Any]) -> None:
    """把笔记填成事实：假设、观察（可核对的数）、与 GKD 的差异点、结论（含零/负增益）。

    白话：run 的门槛不在"跑完了"而在"说清了"。这里写进账的每个数都能在那本流水里
    指到具体行；教师缓存没覆盖、步数不够、增益为零，都照原样写，不粉饰。design 还要求
    记下"跟 TRL 的 GKD 差在哪"，接手的人不必回头翻设计稿就知道这盘为什么这么写。
    """
    asm = result.get("assemble") or {}
    lines = [
        "# run: " + str(run.run_id),
        "",
        f"- 假设：载 p2-05 SFT 产物（`{cfg['parent_run_id'] or '未给起点'}）续训 OPD，"
        f"在学生自采样的 top-{cfg['top_k']} 档上以 {cfg['loss']} 逼教师份额，"
        f"{result['samples']} 题 / {result['steps']} 步应看到散度下降；"
        f"消融 off 档（不截断=soft-KD）与之的 Δ 才说明 on-policy 这步值不值。",
        f"- 观察：{result['steps']} 步跑完，loss {result['first_loss']} → {result['final_loss']}；"
        f"装配 {asm.get('seen')} 见 / {result['samples']} 成，教师缓存命中 {asm.get('teacher_hits')}、"
        f"未命中 {asm.get('teacher_misses')}，训练循环教师前向 {result['teacher_forward_calls']} 次；"
        f"耗时 {result['seconds']} s；可训 {result.get('trainable')} 参数。",
        "- 对照 GKD（design 要求记的差异点）：TRL 的 `GKDTrainer` 逐 token 自回归蒸馏、用 λ 混"
        "on/off-policy 采样、可切广义 JSD(β)；本域是判别式末位一次读出的选项份额表，"
        "“token”在这里退化成“一档”，于是没有 λ——顶上它的是 top-k 截断（on 档）与全表摊开"
        "（off 档，退化成 soft-KD）；教师份额只从离线缓存取（D11 红线，训练循环里不现算），"
        f"本跑截断到 {cfg['top_k'] if cfg['opd_on'] else '全表'}、散度档 {cfg['loss']}"
        f"（β={cfg['jsd_beta']}）、另有 {cfg['mix_gt']} 比例由人工答案说话——最后这条 GKD 没有。",
        "",
        f"结论：OPD({'on' if cfg['opd_on'] else 'off'}) 档本地半场跑通 {result['steps']} 步"
        f"（{cfg['backbone']} / {cfg['device']} {cfg['dtype']} / 旁路 {cfg['kernel_backend']}@{cfg['ascend_target']}）："
        f"散度 {result['first_loss']} → {result['final_loss']}，零教师前向={result['teacher_forward_calls'] == 0}；"
        f"起点 run `{cfg['parent_run_id'] or '—'}`。正式档（910B + C5 SFT 产物）与消融 Δ 待 C5。",
        "",
    ]
    Path(run.path, "notes.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate_dir(model_dir: str | Path, *, axis: str = "quality", limit: int = 64,
                 data_dir: str | Path | None = None, records: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    """用 p2-03 现成口径给一盘产物出 acc 与 ECE（按 n 加权成一个数，分桶表也一并带回）。

    白话：判分的算法早就定在评测侧，这里只把答卷递过去、把结果接回来，然后按"每题一份权重"
    把各桶的数合成一个可比的标量——合成规则是评测侧自己声明的做法（分桶不合并，读者按 n 加权）。

    :returns: `{"acc": float|None, "ece": float|None, "buckets": [...], "n": int}`。
    """
    from sys1.eval import calibration, predict, quality
    from sys1.eval import run as eval_run

    rows = list(records) if records is not None else eval_run.load_axis_records(axis, data_dir=data_dir, limit=limit)
    if not rows:
        return {"acc": None, "ece": None, "buckets": [], "n": 0, "reason": "评测口没有题（数据未装配）"}
    predictor = predict.LocalPredictor(model_dir)
    preds = predictor.predict(rows)
    if not preds:
        return {"acc": None, "ece": None, "buckets": [], "n": 0, "reason": "预测出口没出行"}
    q = quality.score_quality(preds)
    buckets = q.get("buckets") or []
    n = sum(int(b["n"]) for b in buckets) or 0
    acc = (sum(float(b["accuracy"]) * int(b["n"]) for b in buckets) / n) if n else None
    c = calibration.score_calibration(preds)
    by_type = c.get("by_type") or []
    cn = sum(int(b["n"]) for b in by_type) or 0
    ece_after = [b.get("ece_after") for b in by_type if b.get("ece_after") is not None]
    ece = (sum(float(v) * int(b["n"]) for b, v in zip(
        [x for x in by_type if x.get("ece_after") is not None], ece_after)) / cn) if cn and ece_after else None
    return {"acc": (round(float(acc), 6) if acc is not None else None),
            "ece": (round(float(ece), 6) if ece is not None else None),
            "buckets": buckets, "n": n}


def ablation_plan(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """出消融对：同 seed、同数据、同步数，**只**差"是否按学生自采样截断"这一件事。

    白话：要比就得只比一处。两档唯一的区别是：on 档学生只在自己挑的 top-k 档上较真，
    off 档摊全表（等于退化成 soft-KD）。种子、题、步数、垫片超参逐字相同，
    否则 Δ 说不清是谁带来的。

    :returns: `[on_cfg, off_cfg]`（两份独立字典，改一份不脏另一份）。
    """
    on = dict(cfg)
    on.update({"opd_on": True, "ablation_of": ABLATION_ONSETS[0]})
    off = dict(cfg)
    off.update({"opd_on": False, "top_k": 0, "ablation_of": ABLATION_ONSETS[1]})
    for name, one in (("on", on), ("off", off)):
        one["run_prefix"] = f"{cfg['run_prefix']}-{name}"
    return [on, off]


def delta_line(on: dict[str, Any], off: dict[str, Any], *, on_run: str = "", off_run: str = "",
               seed: Any = "") -> str:
    """把两档的 acc/ECE 折成一行结论（零增益、负增益都原样写，不挑好看的口径）。

    白话：Δ 就是"on 减 off"。命中率涨、把握误差降才算好消息；只要有一头是零或者反向，
    这行就照实写反向——本域的交付物是这条结论，而不是一个漂亮的数。
    """
    acc_d = _minus(on.get("acc"), off.get("acc"))
    ece_d = _minus(on.get("ece"), off.get("ece"))
    return (
        f"OPD 增益 Δ=acc {_pp(acc_d)} / ECE {_signed(ece_d)}"
        f"（on {on.get('n', 0)} 题 vs off {off.get('n', 0)} 题；"
        f"runs {on_run or on.get('run_id', '?')}/{off_run or off.get('run_id', '?')}；seed {seed}）"
    )


def _minus(a: Any, b: Any) -> float | None:
    """两个可空的数相减；任一侧没出数就交 None（绝不当成 0 混进结论）。"""
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _signed(value: float | None) -> str:
    """带符号的六位小数（None 写成 `n/a`，读的人一眼知道是没数而不是零）。"""
    return "n/a" if value is None else f"{value:+.6f}"


def _pp(value: float | None) -> str:
    """百分点口径的 Δ（命中率差按 pp 报，与 p2-03 的 `delta_pp` 同一把尺）。"""
    return "n/a" if value is None else f"{value * 100.0:+.2f}pp"


def ablation(cfg: dict[str, Any], *, train_fn: Any = None, eval_fn: Any = None) -> dict[str, Any]:
    """跑消融对（on/off 两 run）并把 Δ 结论行写进两本 notes；两档共用同一 seed 与题源。

    白话：一次调用出两盘东西，各自的流水账里都有一行"OPD 增益 Δ=…"。为什么值得单独立一档：
    手册已声明判别式读出下 on-policy 可能与 soft-KD 差不多——那就要用数据把这句话钉死，
    零增益、负增益都是要入库的结论，不是要掩盖的失败。

    :param cfg: 基线配置（两份派生配置都由它拷出）。
    :param train_fn: 训练入口（默认 `train`；单测注入假件验编排与结论行）。
    :param eval_fn: 评出口（默认 `evaluate_dir`；None 时 Δ 记 n/a 并如实说明没评成）。
    :returns: `{"on":…, "off":…, "delta":…, "note":结论行}`。
    """
    train_fn = train_fn or train
    plans = ablation_plan(cfg)
    results: dict[str, dict[str, Any]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for one in plans:
        res = train_fn(one)
        which = one["ablation_of"]
        results[which] = res
        metrics[which] = (eval_fn(res["model_dir"], axis=one["eval_axis"], limit=int(one["eval_limit"]))
                          if eval_fn else {"acc": None, "ece": None, "n": 0, "reason": "未接评出口"})
    note = delta_line(metrics["on"], metrics["off"], on_run=results["on"]["run_id"],
                      off_run=results["off"]["run_id"], seed=cfg["seed"])
    for which in ("on", "off"):
        run_dir = Path(results[which]["run_dir"])
        (run_dir / "notes.md").write_text(
            (run_dir / "notes.md").read_text(encoding="utf-8").rstrip("\n") + f"\n- {note}\n", encoding="utf-8")
    return {"on": results["on"], "off": results["off"],
            "metrics": metrics, "note": note,
            "delta": {"acc": _minus(metrics["on"].get("acc"), metrics["off"].get("acc")),
                      "ece": _minus(metrics["on"].get("ece"), metrics["off"].get("ece"))}}


# ================================================================ 命令行
#: CLI 开关 ↔ 配置键对照（(键, 类型, 说明)）；只覆盖用户真写出来的项
_CLI_FLAGS: tuple[tuple[str, type, str], ...] = (
    ("backbone", str, "backbone 名（qwen3-0.6b / qwen3.5-0.8b）"),
    ("loader", str, f"载入口：{LOADERS}（seam 走 p2-01 接缝，minimal 是替身兜底）"),
    ("device", str, "装载设备（本地全 CPU）"),
    ("dtype", str, f"位宽：{sorted(DTYPES)}"),
    ("threads", int, "CPU 线程数"),
    ("seed", int, "随机种子（消融两档必须同种子）"),
    ("axis", str, "registry 评测轴（题源）"),
    ("data_dir", str, "数据根目录（默认 registry.DATA_DIR）"),
    ("limit", int, "装配封顶条数（tiny 冒烟用）"),
    ("max_length", int, "单条长度上限（超过直接丢）"),
    ("teacher_model_id", str, "教师身份号（缓存键的一部分）"),
    ("teacher_cache", str, "教师分布缓存目录（训练侧只读）"),
    ("allow_online_teacher", bool, "允许缓存未命中时在线补算（仅云端产标档；训练档必须关）"),
    ("top_k", int, "自采样宽度（0=不截断，即消融 off 档）"),
    ("mix_gt", float, "人工真值掺入比例（0=纯教师）"),
    ("gt_mode", str, "真值折法：hard（与 p2-05 同口径）/ dist"),
    ("hard_mass", float, "hard 档的押注质量"),
    ("loss", str, f"散度档：{LOSSES}（默认 reverse-KL，方向以教师为参考）"),
    ("jsd_beta", float, "JSD 的混合权重"),
    ("sample_chunk", int, "采样器一趟送几道题"),
    ("opd_on", bool, "消融档开关：True=按学生自采样截断 top-k，False=摊全表（off 档）"),
    ("parent_run_id", str, "起点 SFT 的 run_id（进 config，可追溯）"),
    ("parent_model", str, "起点适配器目录或 adapter.safetensors 路径"),
    ("kernel_backend", str, f"旁路实现档：{BACKENDS}（垫片仍由 p2-05 提供）"),
    ("ascend_target", str, "算子件档位（cpu / ascend）"),
    ("r", int, "低秩维度"),
    ("alpha", int, "缩放分子（力度 = α/r）"),
    ("dropout", float, "垫片入口的丢弃率"),
    ("gradient_checkpointing", bool, "分段记账开关"),
    ("lr", float, "学习率"),
    ("warmup_ratio", float, "升温段占比"),
    ("max_grad_norm", float, "改动幅度上限"),
    ("epochs", float, "过几遍题"),
    ("max_tokens", int, "一桌补齐后的字数上限"),
    ("accum", int, "攒几桌下一次笔"),
    ("max_steps", int, "下笔次数上限（0=按桌数与遍数推）"),
    ("run_prefix", str, "run 名前缀（本地半场固定 p2-06-dev）"),
    ("out_dir", str, "产物目录（默认落在 run 目录里的 model/）"),
    ("runs_root", str, "记录本根目录（测试/临时实验指到别处，别污染真 runs/）"),
    ("eval_axis", str, "消融两档出数用的轴"),
    ("eval_limit", int, "消融评测条数上限"),
)


def build_parser() -> argparse.ArgumentParser:
    """造命令行：默认单档训练，`--ablation` 出 on/off 两 run，`--print-config` 只看合流结果。

    白话：开关都对着配置表的名字，写哪个改哪个，没写的用底单。同一份脚本因此能同时
    服务"本地接口自测"和"云端正式档"，差别只在配置表与是否走消融档。
    """
    ap = argparse.ArgumentParser(prog="python -m production.opd",
                                 description="p2-06 OPD（on-policy 自采样 + reverse-KL/JSD 稠密蒸馏）")
    ap.add_argument("--config", help="yaml 配置表（默认档 ← 本表 ← 命令行）")
    ap.add_argument("--print-config", dest="print_config", action="store_true",
                    help="只打印合流后的配置并退出")
    ap.add_argument("--ablation", dest="ablation", action="store_true",
                    help="跑 on/off 消融对（同 seed 同数据），Δ 结论写进两本 notes")
    for key, kind, help_text in _CLI_FLAGS:
        if kind is bool:
            ap.add_argument(f"--{key.replace('_', '-')}", dest=key,
                            action=argparse.BooleanOptionalAction, default=None, help=help_text)
        else:
            ap.add_argument(f"--{key.replace('_', '-')}", dest=key, type=kind, default=None, help=help_text)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：合流三层配置 → 单档训练或消融对 → 把事实表打到标准输出。

    白话：先看用户点了什么开关，缺的用底单补；要消融就出两盘并念一行 Δ 结论；
    要单档就训练。念出来的每个数都能在 run 那本流水里指到具体行。
    """
    args = build_parser().parse_args(argv)
    overrides = {key: getattr(args, key) for key, _, _ in _CLI_FLAGS}
    cfg = load_config(args.config, overrides=overrides)
    if args.print_config:
        print(json.dumps(cfg, ensure_ascii=False, indent=2, default=str))
        return 0
    result = ablation(cfg) if args.ablation else train(cfg)
    if args.ablation:
        print(result["note"])
        payload = {which: {k: v for k, v in result[which].items() if k != "losses"} for which in ("on", "off")}
        payload["delta"] = result["delta"]
    else:
        payload = {k: v for k, v in result.items() if k != "losses"}
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
