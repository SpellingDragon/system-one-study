"""OPD 栈（p2-06）单测：采样器契约 / 散度方向与梯度 / 缓存零前向 / 消融编排。

分层说明（为什么全是假件）：
  默认档**不载任何真权重**——用 `_FakeTokenizer` + `_FakeBody` 走真实的
  `production.opd` 代码路径（渲染→编号→末位读出→top-k→缓存查→散度），但读点向量由
  测试写死，于是期望值能手工算出来（本文件多处直接给闭式解，不拿被测函数自己对自己）。
  0.6B 替身（p2-05 dev 档产物）只进 `integration` 档：本地半场要验"接口在真学生上通不通"，
  但 C1 tiny OPD run 与 C2 消融对**必须等 C5 正式 SFT 产物**，所以真权重这条路由
  `SYS1_TEACHER_INTEGRATION=1` 双重门控，默认不跑、也不把它算作已完成。

运行方式（务必在 release/ 下用 -m，保证 production 包可导入）：
    cd release && .venv/bin/python -m pytest tests/test_opd.py -q
    ... -k sampler                      # A1 采样器（spec 场景：top-k 采样）
    ... -k kl                           # B1 稠密蒸馏损失（spec 场景：reverse-KL 方向）
    ... -k cache                        # B2 教师缓存（spec 场景：缓存命中训练）
    ... -k ablation or -k cli           # C 档的编排契约（run 本体待 C5）
真学生接口自测（替身口径，非 C1）：
    SYS1_TEACHER_INTEGRATION=1 .venv/bin/python -m pytest tests/test_opd.py -m integration -q
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sys1.decision.render import from_systemone

from production.opd import (
    DEFAULT_MIX_GT,
    DEFAULT_JSD_BETA,
    DEFAULT_TEACHER_MODEL_ID,
    DEFAULT_TOP_K,
    JSD,
    LOSS_JSD,
    LOSS_REVERSE_KL,
    LOSSES,
    OPD_VERSION,
    REVERSE_KL,
    SCAFFOLD_MODEL_ID,
    OpdSample,
    TopKSample,
    ablation,
    ablation_plan,
    collect_onpolicy,
    collate_opd,
    default_cfg,
    delta_line,
    divergence,
    jsd,
    load_config,
    main,
    mixed_target,
    opd_loss,
    opd_step,
    restrict_normalize,
    reverse_kl,
    sample_topk,
    selected_logp,
    teacher_full_dist,
    teacher_on_subset,
    topk_probs,
    train,
)
from production.teachers.cache import DistCache, make_key
from production.teachers.text import TeacherStats, TextTeacher
from sys1.decision.render import LETTERS, option_order, prompt_text

REPO_RELEASE = Path(__file__).resolve().parents[1]
OPD_FILE = REPO_RELEASE / "production" / "opd.py"
PSEUDO_CACHE = REPO_RELEASE / "bench" / "teacher_cache" / "p2_05_pseudo"
DEV_SFT_MODEL = REPO_RELEASE / "runs" / "1005-p2-05-dev-sft-qwen3-0-6b-torch-7389" / "model"

VOCAB = 64
PAD_ID = 0
OTHER_ID = 40
D = 26                                      # 读点宽度：head_weight[字母] = 基向量 ⇒ 分数=profile 本身
LETTER_AT = {ch: i + 1 for i, ch in enumerate(LETTERS)}     # A→1 … Z→26（与 test_teachers 同口径）
NARROW = {"continue": "c", "human_review": "h", "observe": "o", "stop": "s"}


# ── 假件：最小的"能骗过真实代码路径"的分词器与前向壳 ────────────────────────
class _FakeTokenizer:
    """字符级假分词器：26 枚字母各占一格，其余字符同用一个号，右补空洞。

    `allow_thinking_kw=False` 时故意不认 `enable_thinking` 这个开关，用来验采样器
    套模板的退回写法（口径与 p2-02 后端一致）。
    """

    padding_side = "right"
    pad_token_id = PAD_ID
    eos_token = "</s>"

    def __init__(self, *, allow_thinking_kw: bool = True) -> None:
        self.allow_thinking_kw = allow_thinking_kw
        self.template_calls = 0

    @staticmethod
    def _id(ch: str) -> int:
        return LETTER_AT.get(ch, OTHER_ID)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:  # noqa: FBT002
        return [self._id(ch) for ch in text]

    def __call__(self, text: str, add_special_tokens: bool = False, **_: object) -> dict:  # noqa: FBT002
        return {"input_ids": self.encode(text, add_special_tokens)}

    def apply_chat_template(self, messages, tokenize: bool = False,  # noqa: FBT001, ARG002
                            add_generation_prompt: bool = True, **kwargs) -> str:  # noqa: ARG002
        self.template_calls += 1
        if "enable_thinking" in kwargs and not self.allow_thinking_kw:
            raise TypeError("这个假模板不认 enable_thinking")
        return "\n".join(str(m.get("content", "")) for m in messages)


class _FakeBody(torch.nn.Module):
    """假前向壳：每行、每格都交出同一条 profile 向量（读点位此因此与长度无关）。

    白话：真模型在末位留下一串"可学的数字"，这里把那串数字直接写成测试给定的值——
    于是"学生该押几成"完全由测试说了算，期望概率能用闭式手算，不必反过来信被测函数。
    """

    def __init__(self, dim: int = D) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("_profile", torch.zeros(1, dim))
        self.calls = 0

    def set_profile(self, rows) -> None:
        """写入逐行读点向量（`(B, dim)`；`torch.nn.Parameter` 也收，用来验梯度通路）。"""
        if isinstance(rows, torch.nn.Parameter):
            self._profile = rows                       # 直接挂参数：backward 就能落在这里
        else:
            self._profile = torch.as_tensor(rows, dtype=torch.float32).reshape(-1, self.dim)

    def forward(self, input_ids=None, attention_mask=None, use_cache: bool = True,  # noqa: ARG002
                return_dict: bool = True, **_: object) -> SimpleNamespace:
        self.calls += 1
        batch, width = input_ids.shape
        prof = self._profile
        if prof.shape[0] == 1 and batch > 1:
            prof = prof.expand(batch, -1)
        held = prof[:, None, :].expand(batch, width, self.dim)
        return SimpleNamespace(last_hidden_state=held)


def _head_weight() -> torch.Tensor:
    """`(vocab, D)` 输出层权重：第 j 枚字母的行写成基向量 e_j ⇒ 读出分数 = profile[:, j]。"""
    weight = torch.zeros(VOCAB, D)
    for j, ch in enumerate(LETTERS):
        weight[LETTER_AT[ch], j] = 1.0
    return weight


@dataclass
class _FakeStudent:
    """鸭子型学生：只带采样器真用到的五件（body/tokenizer/head_weight/letter_ids/pad_id）。"""

    body: _FakeBody
    tokenizer: _FakeTokenizer
    head_weight: torch.Tensor
    letter_ids: tuple[int, ...]
    pad_id: int
    model: object = None
    name: str = "fake-student"
    loader: str = "minimal"
    dtype: torch.dtype = torch.float32
    provenance: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """与 p2-05 `Student.summary()` 同形状（run 台账要读这个）。"""
        return {"backbone": self.name, "loader": self.loader, "dtype": str(self.dtype),
                "params_m": 0.0, "letters": len(self.letter_ids), "pad_id": self.pad_id}


def _student(profile_rows, *, thinking_kw: bool = True) -> _FakeStudent:
    """造一个"读点向量由测试写死"的假学生。"""
    body = _FakeBody()
    body.set_profile(profile_rows)
    tok = _FakeTokenizer(allow_thinking_kw=thinking_kw)
    return _FakeStudent(body=body, tokenizer=tok, head_weight=_head_weight(),
                        letter_ids=tuple(LETTER_AT[c] for c in LETTERS), pad_id=PAD_ID, model=body)


def _record(rid: str = "rec-1", codes=None, gold: dict | None = None, *, state: str = "evidence: ship it",
            qid: str = "q") -> dict:
    """造一行 registry 统一信封（与 p2-03 交出的形状一致：`{id, task, qtype, sample}`）。"""
    codes = list(codes or NARROW)
    sample: dict = {
        "state": state,
        "questions": {qid: {"type": "choice", "instructions": "Which action?",
                            "criteria": {c: f"desc {c}" for c in codes}}},
    }
    if gold is not None:
        sample["targets"] = {qid: dict(gold)}
    return {"id": rid, "task": "synthetic-set", "qtype": "choice", "sample": sample}


def _twenty_codes() -> list[str]:
    """20 候选的代号表（spec 场景就是 20 选项）。"""
    return [f"opt{i:02d}" for i in range(20)]


def _softmax(values: list[float]) -> list[float]:
    """测试侧独立实现的 softmax（闭式手算用，绝不 import 被测代码）。"""
    top = max(values)
    weights = [math.exp(v - top) for v in values]
    total = sum(weights)
    return [w / total for w in weights]


# ============================================================ 工作项 A1：on-policy 采样器
def test_sampler_top_k_20_options_returns_at_most_8_and_sums_to_one():
    """spec 场景「top-k 采样」：20 选项状态 → ≤8 个 (opt, p) 且概率和=1。

    期望值全部手工闭式给出：profile 写成 0,1,…,19 ⇒ top-8 是第 19…12 列（T,S,R,Q,P,O,N,M），
    重归一后的份额就是 softmax(7,6,…,0)，截断质量是 softmax(0..19) 上那 8 项之和。
    """
    profile = [[float(i) for i in range(D)]]
    student = _student(profile)
    record = _record(codes=_twenty_codes())
    sample = record["sample"]

    got = sample_topk(student, sample["state"], DEFAULT_TOP_K, spec=sample["questions"]["q"], sample_id="rec-1")

    assert len(got) == 1
    one = got[0]
    assert len(one.letters) == 8 and len(one.options) == 8          # spec：≤8，20 候选下正好 8
    assert one.letters == tuple(LETTERS[i] for i in (19, 18, 17, 16, 15, 14, 13, 12))
    assert one.options == tuple(f"opt{i:02d}" for i in (19, 18, 17, 16, 15, 14, 13, 12))
    assert pytest.approx(sum(one.probs), rel=1e-12) == 1.0          # spec：概率和=1
    expected = _softmax([float(v) for v in range(8)])               # 相对最大值 7..0 闭式重摊
    assert one.probs == pytest.approx(list(reversed(expected)), rel=1e-9)
    assert list(one.probs) == sorted(one.probs, reverse=True)          # 份额降序（元组转列表再比）
    assert one.scores == tuple(float(i) for i in (19, 18, 17, 16, 15, 14, 13, 12))
    full = _softmax([float(i) for i in range(20)])
    assert pytest.approx(one.mass, rel=1e-9) == sum(full[12:20])    # 截断掉的份额被如实记下单
    assert one.mass < 1.0
    # 教师请求文字必须是渲染层的扁平文字（缓存键的原料，与 p2-05 产标时逐字一致）
    row = from_systemone(sample["state"], sample["questions"]["q"], qid="q")
    assert one.order == tuple(option_order(row))
    assert one.prompt == prompt_text(row, option_order(row))


def test_sampler_shape_contract_is_stable_for_rl_reuse():
    """RL（p2-11）复用面：四列等长同序、概率和为 1、出口可摊平——签名不许被本域私改。"""
    student = _student([[3.0, 1.0, 4.0, 0.0, 5.0, 2.0, 6.0, 1.5] + [0.0] * (D - 8)])
    record = _record(codes=_twenty_codes()[:8])
    got = sample_topk(student, record["sample"]["state"], 4, spec=record["sample"]["questions"]["q"])
    one = got[0]

    assert len(one.letters) == len(one.options) == len(one.probs) == len(one.scores) == 4
    assert all(ch in one.order for ch in one.options)               # 选项必须是真在场的候选
    assert list(one.letters) == [LETTERS[one.order.index(c)] for c in one.options]  # 列↔代号严格同序
    assert pytest.approx(sum(one.probs), rel=1e-12) == 1.0
    payload = one.to_dict()
    assert set(payload) == {"sample_id", "qid", "prompt", "order", "letters", "options",
                            "probs", "scores", "mass", "k"}
    assert payload["k"] == 4 and payload["qid"] == "q"
    assert isinstance(one, TopKSample)


def test_sampler_k_larger_than_candidates_takes_all_and_mass_is_one():
    """候选不足 k 时全取（spec 的"≤8"里的"≤"），此时没有截断、质量应为 1。"""
    student = _student([[0.0, 1.0, 2.0, 3.0] + [0.0] * (D - 4)])
    record = _record(codes=["continue", "human_review", "observe", "stop"])
    got = sample_topk(student, record["sample"]["state"], 8, spec=record["sample"]["questions"]["q"])

    one = got[0]
    assert len(one.letters) == 4
    assert pytest.approx(sum(one.probs), rel=1e-12) == 1.0
    assert pytest.approx(one.mass, rel=1e-9) == 1.0
    assert one.letters == ("D", "C", "B", "A")                      # stop>observe>human_review>continue


def test_sampler_ties_break_by_letter_order_and_are_deterministic():
    """同分按字母序破平：采样器因此与随机数无关（可复现是接口契约的一部分）。"""
    student = _student([[0.0] * D])
    record = _record(codes=_twenty_codes())
    first = sample_topk(student, record["sample"]["state"], 8, spec=record["sample"]["questions"]["q"])
    second = sample_topk(student, record["sample"]["state"], 8, spec=record["sample"]["questions"]["q"])

    assert [s.letters for s in first] == [s.letters for s in second]
    assert first[0].letters == tuple(LETTERS[:8])                   # A,B,C,…,H：全同分取最左八列
    assert pytest.approx(sum(first[0].probs), rel=1e-12) == 1.0
    assert first[0].probs == pytest.approx([0.125] * 8, rel=1e-9)   # 均匀摊成 1/8


def test_sampler_temperature_keeps_ranking_only_changes_sharpness():
    """温度只改陡缓、不改名次（与 p2-02 softmax_over 同一单调性保证）。"""
    student = _student([[float(i) for i in range(D)]])
    record = _record(codes=_twenty_codes())
    cold = sample_topk(student, record["sample"]["state"], 4, spec=record["sample"]["questions"]["q"])
    warm = sample_topk(student, record["sample"]["state"], 4, spec=record["sample"]["questions"]["q"],
                       temperature=4.0)

    assert cold[0].letters == warm[0].letters
    assert cold[0].probs[0] > warm[0].probs[0]                      # 高温把份额摊平
    assert pytest.approx(sum(warm[0].probs), rel=1e-12) == 1.0
    assert pytest.approx(warm[0].probs[0], rel=1e-9) == _softmax([19 / 4, 18 / 4, 17 / 4, 16 / 4])[0]


def test_sampler_multi_question_states_yield_one_row_per_question():
    """一个状态挂多题：一次前向打多行，每题各出一条采样（形状与题数一致）。"""
    profile = [[0.0, 2.0, 1.0] + [0.0] * (D - 3), [3.0, 0.0, 1.0] + [0.0] * (D - 3)]
    student = _student(profile)
    state = "evidence: ship it"
    spec = {
        "q1": {"type": "choice", "instructions": "A?", "criteria": NARROW},
        "q2": {"type": "choice", "instructions": "B?", "criteria": {"x": "1", "y": "2", "z": "3"}},
    }

    got = sample_topk(student, state, 2, spec=spec)

    assert len(got) == 2 and student.body.calls == 1                # 两题共用一次前向
    assert {s.qid for s in got} == {"q1", "q2"}
    assert all(len(s.letters) == 2 for s in got)
    assert all(pytest.approx(sum(s.probs), rel=1e-12) == 1.0 for s in got)


def test_sampler_use_current_false_leaves_no_graph():
    """评测/replay 档采样不该拖计算图（`use_current=False` → no_grad）。"""
    body = _FakeBody()
    body.set_profile(torch.nn.Parameter(torch.tensor([[1.0, 5.0, 2.0] + [0.0] * (D - 3)])))
    student = _student(body._profile)                                # noqa: SLF001 — 假件内部直读
    student.body = body
    record = _record(codes=["a", "b", "c"])
    spec = record["sample"]["questions"]["q"]

    live = sample_topk(student, record["sample"]["state"], 2, spec=spec)
    assert body._profile.grad_fn is None or not body._profile.requires_grad  # Parameter 本就无图
    assert len(live) == 1 and live[0].probs[0] > live[0].probs[1]

    with torch.no_grad():
        cold = sample_topk(student, record["sample"]["state"], 2, spec=spec, use_current=False)
    assert [s.letters for s in cold] == [s.letters for s in live]    # 关掉图不改变结果，只省显存


def test_sampler_falls_back_when_template_rejects_thinking_kw():
    """模板不认 `enable_thinking` 时退回不带开关的写法（口径与 p2-02 后端一致）。"""
    student = _student([[0.0, 1.0, 2.0] + [0.0] * (D - 3)], thinking_kw=False)
    record = _record(codes=["a", "b", "c"])

    got = sample_topk(student, record["sample"]["state"], 2, spec=record["sample"]["questions"]["q"])

    assert student.tokenizer.template_calls == 2                 # 带开关被拒一次 + 退回不带开关一次
    assert len(got) == 1 and got[0].letters == ("C", "B")


def test_sampler_accepts_prerendered_row_for_replay():
    """RL replay 常直接给已渲染行（无 spec）：接口须原样收下，不再走排版。"""
    from sys1.decision.render import from_systemone

    row = from_systemone("evidence: replay", {"type": "choice", "instructions": "A?", "criteria": NARROW}, qid="q")
    student = _student([[0.0, 1.0, 2.0, 3.0] + [0.0] * (D - 4)])

    got = sample_topk(student, row, 2)                               # spec=None，state 即已渲染行

    assert len(got) == 1 and got[0].order == tuple(option_order(row))
    assert got[0].letters == ("D", "C")
    assert pytest.approx(sum(got[0].probs), rel=1e-12) == 1.0


def test_sampler_topk_probs_pure_function_no_model_needed():
    """RL 可直接复用的纯函数：给分数就给 top-k，手算闭式期望 + 入参体检。"""
    idx, probs = topk_probs([0.5, 2.0, -1.0, 2.0], k=3)

    assert idx == [1, 3, 0]                                         # 分数降序，同分取下标小的
    assert pytest.approx(sum(probs), rel=1e-12) == 1.0
    expect = _softmax([2.0, 2.0, 0.5])
    assert probs == pytest.approx(expect, rel=1e-12)

    all_idx, all_probs = topk_probs([1.0, 1.0], k=8)                # k 超过候选数 → 全取
    assert all_idx == [0, 1] and all_probs == pytest.approx([0.5, 0.5], rel=1e-12)

    with pytest.raises(ValueError):
        topk_probs([], k=3)
    with pytest.raises(ValueError):
        topk_probs([1.0], k=0)
    with pytest.raises(ValueError):
        topk_probs([1.0, 2.0], k=2, temperature=0.0)


def test_sampler_rejects_unusable_inputs():
    """采样器对空/越界入参当场报错，不静默给出"看起来能训"的假数据。"""
    student = _student([[0.0] * D])
    with pytest.raises(ValueError):
        sample_topk(student, "state only", 8, spec=None)             # 既非渲染行又没 spec
    with pytest.raises(ValueError):
        sample_topk(student, "state only", 8, spec={"type": "choice", "criteria": {}})
    with pytest.raises(ValueError):
        from production.opd import student_topk_scores
        student_topk_scores(student, ["x"], 0)
# ============================================================ 工作项 B1：稠密蒸馏损失（散度方向与梯度）
def _probs_of(z: torch.Tensor) -> torch.Tensor:
    """测试侧独立实现的 softmax（可导；用来验被测损失梯度，不 import 被测代码）。"""
    exps = torch.exp(z - z.max(dim=-1, keepdim=True).values)
    return exps / exps.sum(dim=-1, keepdim=True)


def _logp_of(z: torch.Tensor) -> torch.Tensor:
    """测试侧独立实现的 log-softmax（可导）——`reverse_kl` 的 logp 入参由此造。"""
    shifted = z - z.max(dim=-1, keepdim=True).values
    return shifted - torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))


def _kl(p, q) -> float:
    """测试侧闭式 KL(p‖q)=Σ p·(log p − log q)，`p=0` 的列按 0·log0=0 约定跳过。"""
    return sum(float(a) * (math.log(float(a)) - math.log(float(b)))
               for a, b in zip(p, q) if float(a) > 0.0)


TEACHER_3 = [0.60, 0.35, 0.05]
STUDENT_3 = [0.20, 0.30, 0.50]


def test_kl_direction_is_teacher_referenced_not_student_referenced():
    """spec 场景「reverse-KL 方向」：数值钉死 `KL(p_teacher‖p_student)`，反向写法当场露馅。

    两侧期望都由测试自己按定义算：本档取到的值必须等于"以教师为测度"的那个，
    且与"以学生为测度"的那个明显不等（两个数都是有限正数，比较才有意义）。
    """
    logp = torch.tensor([math.log(v) for v in STUDENT_3])
    got = float(reverse_kl(TEACHER_3, logp))

    assert got == pytest.approx(_kl(TEACHER_3, STUDENT_3), rel=1e-6)      # 方向：教师为参照
    forward = _kl(STUDENT_3, TEACHER_3)
    assert got != pytest.approx(forward, rel=1e-3)                        # 反向写法对不上
    assert got > 0.0 and forward > 0.0


def test_kl_monotone_decreases_as_student_approaches_teacher():
    """p_s→p_t 时 KL 单调降且在终点归零（design 数值方向单测的第一条）。"""
    lams = [i / 10 for i in range(11)]
    values = []
    for lam in lams:
        q = [(1 - lam) * a + lam * b for a, b in zip(STUDENT_3, TEACHER_3)]
        values.append(float(reverse_kl(TEACHER_3, torch.tensor([math.log(v) for v in q]))))

    assert values == sorted(values, reverse=True)                         # 全程严格单调降
    assert values[0] == pytest.approx(_kl(TEACHER_3, STUDENT_3), rel=1e-6)
    assert values[-1] < 1e-6                                              # 终点 KL(p‖p)=0
    assert all(v >= -1e-9 for v in values)                             # 散度不为负（终点允许浮点毛刺）


def test_kl_gradient_is_student_share_minus_teacher_share():
    """梯度方向手工验证：∂KL(p‖q)/∂z_i = q_i − p_i（q=softmax(z)），autograd 必落到这条闭式上。"""
    z = torch.tensor([[1.2, -0.4, 0.7, 2.1]], requires_grad=True)
    p = torch.tensor([[0.45, 0.05, 0.10, 0.40]])

    loss = reverse_kl(p, _logp_of(z))
    loss.backward()
    expected = _probs_of(z.detach()) - p

    assert torch.allclose(z.grad, expected, atol=1e-6)
    # 逐列读法：教师押得多而学生押得少的列，梯度为负（下笔要把这列抬上去）
    assert (z.grad < 0).sum().item() == 2


def test_kl_batched_reduction_none_gives_per_row_divergence():
    """`reduction="none"` 交逐行散度（诊断/评测要按题看，不能被行均值抹平）。"""
    p = torch.tensor([TEACHER_3, [0.2, 0.2, 0.6]])
    q = torch.tensor([STUDENT_3, [0.1, 0.5, 0.4]])
    logp = torch.log(q)

    rows = reverse_kl(p, logp, reduction="none")
    mean = reverse_kl(p, logp, reduction="mean")

    assert rows.shape == (2,)
    assert rows[0].item() == pytest.approx(_kl(TEACHER_3, STUDENT_3), rel=1e-6)
    assert rows[1].item() == pytest.approx(_kl([0.2, 0.2, 0.6], [0.1, 0.5, 0.4]), rel=1e-6)
    assert float(mean) == pytest.approx(float(rows.mean()), rel=1e-6)


def test_kl_selected_logp_masks_unkept_columns_and_leaves_no_nan():
    """`keep` 列纪律：未选中列先封 -inf 参归一、再归零；选中列内 exp 和为 1（0×-inf 的坑不复现）。"""
    scores = torch.tensor([[0.5, 2.0, -1.0, 0.25]])
    keep = torch.tensor([[True, True, False, True]])

    logp = selected_logp(scores, keep)
    kept = [logp[0, i].item() for i in range(4) if keep[0, i]]

    assert logp[0, 2].item() == 0.0                                       # 未选中列写成 0，不是 -inf
    assert pytest.approx(math.fsum(math.exp(v) for v in kept), rel=1e-6) == 1.0
    assert torch.isfinite(logp).all()
    # 未选中列的分数再怎么离谱都不进归一（等价于"这档不在场"）
    moved = selected_logp(torch.tensor([[0.5, 2.0, 99.0, 0.25]]), keep)
    assert torch.allclose(moved, logp, atol=1e-6)


def test_kl_target_zero_mass_column_is_skipped():
    """教师零份额列按 0·log0=0 不计：不然一份"教师没说话"的列会把罚分算成无穷。"""
    p = [0.5, 0.5, 0.0]
    logq = torch.tensor([math.log(0.9), math.log(0.1), -50.0])

    got = float(reverse_kl(p, logq))

    assert math.isfinite(got)
    assert got == pytest.approx(_kl(p, [0.9, 0.1, math.exp(-50.0)]), rel=1e-6)


def test_kl_extreme_and_half_precision_scores_stay_finite():
    """design 风险条：极端分数与半精度位宽不得产 nan——log-softmax 在 fp32 域算完再回投。"""
    scores = torch.tensor([[1e4, -1e4, 20.0, -0.5]]).half()               # fp16 极端读数
    logp = selected_logp(scores, None).float()
    assert torch.isfinite(logp).all()
    assert logp[0, 1].item() < -1e3                     # 极低的列是有限负数（不是 nan，也不是 -inf）

    loss = float(reverse_kl(torch.tensor([[0.25, 0.25, 0.25, 0.25]]), logp))
    assert math.isfinite(loss) and loss > 0.0


def test_kl_divergence_switch_is_jsd_and_symmetric_at_half():
    """可切档：JSD(β=0.5) 对称、同分布归零；档名写错当场报错，不静默退回默认档。"""
    logq = torch.tensor([math.log(v) for v in STUDENT_3])
    forward_val = float(divergence(TEACHER_3, logq, loss=LOSS_JSD))
    backward_val = float(jsd(STUDENT_3, torch.tensor([math.log(v) for v in TEACHER_3])))

    assert forward_val == pytest.approx(backward_val, rel=1e-6)           # β=0.5 完全对称
    assert forward_val > 0.0
    assert float(jsd(TEACHER_3, torch.tensor([math.log(v) for v in TEACHER_3]))) < 1e-7
    assert float(divergence(TEACHER_3, logq, loss=REVERSE_KL)) == pytest.approx(
        float(reverse_kl(TEACHER_3, logq)), rel=1e-9)
    assert set(LOSSES) == {LOSS_REVERSE_KL, LOSS_JSD}

    with pytest.raises(ValueError):
        divergence(TEACHER_3, logq, loss="forward_kl")                    # 未开过的档不许当成默认
    with pytest.raises(ValueError):
        jsd(TEACHER_3, logq, beta=1.5)


def test_kl_keep_mask_excludes_unselected_columns_from_loss():
    """损失侧同样只较真选中列：整项跳过，未选中列既不罚分也不参与归一。"""
    scores = torch.tensor([[3.0, -3.0, 0.0]])
    keep = torch.tensor([[True, False, True]])
    logp = selected_logp(scores, keep)
    target = torch.tensor([[0.4, 0.0, 0.6]])

    kept_only = float(reverse_kl(target, logp, keep=keep))
    same_no_zero_cols = float(reverse_kl(torch.tensor([[0.4, 0.6]]), logp[:, [0, 2]]))

    assert kept_only == pytest.approx(same_no_zero_cols, rel=1e-6)        # 等价于把这列摘掉


def test_kl_shape_mismatch_is_rejected():
    """形状不合当场报错：目标与 logp 错位会算出"看着像收敛"的假损失。"""
    with pytest.raises(ValueError):
        reverse_kl([0.5, 0.5], torch.tensor([math.log(0.5)]))
    with pytest.raises(ValueError):
        reverse_kl([[0.5, 0.5]], [[-0.7, -0.7]], keep=torch.tensor([True]))
    with pytest.raises(TypeError):
        selected_logp([[0.5, 0.5]], None)


def test_kl_mixed_target_blends_ground_truth_by_weight():
    """目标掺真值：mix_gt=0 全听教师、=1 全听真值、=0.2 按八成教师+两成真值再归一。"""
    teacher = {"A": 0.5, "D": 0.5}
    gt = {"A": 1.0, "D": 0.0}

    assert mixed_target(teacher, gt, 0.0) == pytest.approx([0.5, 0.5], rel=1e-9)
    assert mixed_target(teacher, gt, 1.0) == pytest.approx([1.0, 0.0], rel=1e-9)
    blended = mixed_target(teacher, gt, DEFAULT_MIX_GT)
    assert blended == pytest.approx([0.6, 0.4], rel=1e-9)               # 0.8×0.5+0.2×1=0.6，余下 0.4
    assert pytest.approx(sum(blended), rel=1e-12) == 1.0
    assert mixed_target(teacher, None, DEFAULT_MIX_GT) == pytest.approx([0.5, 0.5], rel=1e-9)
    assert mixed_target({"A": 0.25, "B": 0.75}, {"A": 0.0, "B": 1.0}, 0.5) == pytest.approx(
        [0.125, 0.875], rel=1e-9)                                          # 两份都已归一，掺完和仍为 1


def _opd_sample(sid="s-1", letters=4, subset=("A", "D"), shares=(0.6, 0.4), ids=(1, 2, 3)):
    """手造一条已装配好的样本（不碰缓存：损失与梯度用例只要"列格子 + 目标"这两样）。"""
    cols = LETTERS[:letters]
    full = [0.0] * letters
    for ch, value in zip(subset, shares):
        full[LETTERS.index(ch)] = float(value)
    return OpdSample(sample_id=sid, ids=tuple(ids), letters=tuple(cols), subset=tuple(subset),
                     teacher=tuple(full), target=tuple(full), prompt=f"prompt {sid}",
                     qtype="choice", source="unit-test")


def test_kl_opd_step_backprops_with_teacher_referenced_sign():
    """一步损失的梯度方向手工验证：教师押得多而学生押得少的列必须被抬起来（q−p 的符号）。"""
    param = torch.nn.Parameter(torch.tensor([[0.0, 1.0, 2.0, 3.0] + [0.0] * (D - 4)]))
    student = _student(param)                                   # 假件直接收参数：backward 落在读点向量上
    sample = _opd_sample(subset=("A", "D"), shares=(0.6, 0.4))             # 学生此刻在 D 上过重
    batch = collate_opd([sample], [0], pad_id=PAD_ID, letter_map=student.letter_ids)

    first = float(opd_step(student, batch).detach())
    param.grad = None
    opd_step(student, batch).backward()
    grad = param.grad.detach()[0]

    assert grad[0].item() < 0.0 < grad[3].item()                          # A 要抬、D 要压
    assert grad[1].item() == 0.0 == grad[2].item()                        # 未选中列不吃罚（keep 归零）

    for _ in range(60):                                                   # 手工下笔：损失应单调降
        param.grad = None
        opd_step(student, batch).backward()
        with torch.no_grad():
            param -= 0.4 * param.grad
        param.grad = None
        now = float(opd_step(student, batch).detach())
        assert now <= first + 1e-6
        first = now
    assert first < 0.02                                                   # 已逼近 KL(p‖p)=0


def test_kl_opd_loss_reads_the_frozen_readout_only():
    """静态门（复用面）：取格与读列只有 decision/ 那一把尺子，本域不另摆平行实现。"""
    src = OPD_FILE.read_text(encoding="utf-8")

    assert src.count("option_scores(") == 2                               # 采样路 + 损失路
    assert src.count("log_softmax") == 1                                  # 只在 selected_logp 一处
    assert "from sys1.decision.readout import option_scores" in src
    for banned in (".generate(", "index_select", "[:, -1]", "torch.softmax", "logits[0, -1"):
        assert banned not in src, f"读点/采样口径不许在 opd 里重抄一遍：{banned}"


# ============================================================ 工作项 B2：教师缓存命中（零前向）
class _StrictBackend:
    """"发前向就当场炸"的教师后端：用来证明训练路真的一个前向都没发。"""

    def __init__(self, model_id: str, *, rows: dict | None = None) -> None:
        self.model_id = model_id
        self.rows = rows or {}
        self.forwards = 0

    def letter_logprobs(self, prompt: str, option_keys):                  # TextTeacher._forward 的唯一出口
        self.forwards += 1
        if prompt not in self.rows:
            raise AssertionError(f"训练循环不许发教师前向：{prompt[:40]!r}")
        return self.rows[prompt]

    def last_letter_logits(self, prompts, option_keys):                   # 批量口同样炸（产标档才用）
        raise AssertionError("训练循环不许发批量教师前向")


def _flat_rows(records, codes):
    """按渲染层口径把"教师请求文字 + 字母列"算出来（产缓存与查缓存必须同源，否则键对不上）。"""
    out = []
    for rec in records:
        spec = rec["sample"]["questions"]["q"]
        row = from_systemone(rec["sample"]["state"], spec, qid="q")
        order = option_order(row)
        assert len(order) == len(codes)
        out.append((prompt_text(row, order), [LETTERS[i] for i in range(len(order))]))
    return out


def _teacher_rows(prompts, cols, weights):
    """给每道题产一份"全候选上的教师份额"，并按 p2-02 的键写进缓存目录。"""
    return [
        (prompt, colset, {ch: w / sum(weights[:len(colset)]) for ch, w in zip(colset, weights)})
        for prompt, colset in zip(prompts, cols)
    ]


def _seeded_records(count=3, codes=None):
    codes = list(codes or ["continue", "human_review", "observe", "stop"])
    return [_record(rid=f"rec-{i}", codes=codes, gold={codes[-1]: 1.0},
                    state=f"evidence {i}: ship it or hold") for i in range(count)]


def _tmp_cache(tmp_path, model_id, rows):
    """写一个分片再按 `read_only=True` 重开（训练侧就该这样只读打开）。"""
    root = tmp_path / "cache"
    cache = DistCache(root=root)
    for prompt, cols, dist in rows:
        cache.record(model_id=model_id, prompt=prompt, option_keys=cols, dist=dist, source="opd-unit")
    assert cache.flush() == len(rows)
    return DistCache(root=root, read_only=True).load()


def test_cache_hit_training_sends_zero_teacher_forwards(tmp_path):
    """spec 场景「缓存命中训练」：装配 + 两步损失跑完，教师前向计数增量为 0，且命中数 >0。

    断言有牙的三点：① 后端一旦被叫到就 `AssertionError`（不是靠计数为 0 蒙对）；
    ② 账本来自 `TeacherStats`（`collect_onpolicy` 把它原样交回）；③ 缓存以只读方式打开。
    """
    codes = ["continue", "human_review", "observe", "stop"]
    records = _seeded_records(3, codes)
    student = _student([[float(i) for i in range(D)]])
    flat = _flat_rows(records, codes)
    rows = _teacher_rows([p for p, _ in flat], [c for _, c in flat], [1.0, 2.0, 3.0, 4.0])
    cache = _tmp_cache(tmp_path, SCAFFOLD_MODEL_ID, rows)
    assert cache.read_only is True

    teacher = TextTeacher(_StrictBackend(SCAFFOLD_MODEL_ID), cache=cache, stats=TeacherStats())
    teacher.stats.reset()
    samples, asm, tstats = collect_onpolicy(
        records, student, cache=cache, teacher_model_id=SCAFFOLD_MODEL_ID, k=2, teacher=teacher
    )
    before = tstats.forward_calls

    assert len(samples) == 3 and asm.hits == 3 and asm.misses == 0        # 每题都问到了缓存
    batches = [collate_opd(samples, [0, 1], pad_id=student.pad_id, letter_map=student.letter_ids),
               collate_opd(samples, [2], pad_id=student.pad_id, letter_map=student.letter_ids)]
    losses = [float(opd_step(student, b).detach()) for b in batches]
    assert all(math.isfinite(v) for v in losses)

    assert tstats.forward_calls == before == 0                            # 训练循环零教师前向
    assert tstats.api_calls == 0 and teacher.backend.forwards == 0
    assert cache.stats["hits"] >= 3 and cache.stats["pending"] == 0       # 只读：没往教师账本写一行
    assert tstats is teacher.stats                                        # 交回的就是那份账本


def test_cache_miss_without_online_door_never_calls_teacher(tmp_path):
    """未命中且没开在线档：明说查不到并计入 misses，绝不"顺手替教师算一遍"。"""
    codes = ["continue", "human_review", "observe", "stop"]
    records = _seeded_records(1, codes)
    student = _student([[float(i) for i in range(D)]])
    cache = DistCache(root=tmp_path / "empty", read_only=True).load()      # 空账本：一条伪标都没有

    teacher = TextTeacher(_StrictBackend(SCAFFOLD_MODEL_ID), cache=cache, stats=TeacherStats())
    samples, asm, tstats = collect_onpolicy(records, student, cache=cache, teacher_model_id=SCAFFOLD_MODEL_ID, k=2, teacher=teacher)

    assert samples == [] and asm.hits == 0 and asm.misses == 1
    assert tstats.forward_calls == 0 and teacher.backend.forwards == 0


def test_cache_forward_counter_is_live_when_online_door_open(tmp_path):
    """对照用例：把在线档打开，同一个 `forward_calls` 就会加——上一条的 0 不是恒等式。"""
    codes = ["continue", "human_review", "observe", "stop"]
    records = _seeded_records(2, codes)
    student = _student([[float(i) for i in range(D)]])
    flat = _flat_rows(records, codes)
    cache = DistCache(root=tmp_path / "writable").load()                  # 产标档要能回写
    rows = {p: {ch: math.log(w) for ch, w in zip(cols, [4.0, 3.0, 2.0, 1.0])} for p, cols in flat}
    backend = _StrictBackend(SCAFFOLD_MODEL_ID, rows=rows)
    teacher = TextTeacher(backend, cache=cache, stats=TeacherStats())

    samples, asm, tstats = collect_onpolicy(
        records, student, cache=cache, teacher_model_id=SCAFFOLD_MODEL_ID, k=2,
        teacher=teacher, allow_online_teacher=True
    )

    assert len(samples) == 2 and tstats.forward_calls == 2 and backend.forwards == 2
    assert asm.hits == 2 and asm.misses == 0                              # 份额是现算来的，不是查来的
    assert cache.read_only is False and cache.stats["misses"] == 2 and cache.stats["pending"] == 2


def test_cache_scaffold_model_id_is_keyed_apart_from_real_teacher():
    """`#scaffold-cpu` 与真教师键分家：同一题面换教师身份必须 miss（袖珍分数绝不当伪标）。"""
    cache = DistCache(root=PSEUDO_CACHE, read_only=True).load()
    if len(cache) == 0:   # skip-when-missing（R-P1-4 统一口径）：scaffold 伪标包是 bench/ 本地副本，
                         # gitignored 按设计不入库；空账本测不出"键分家"，缺件属环境事实而非回归。
        pytest.skip(f"scaffold 伪标包不在盘上：{PSEUDO_CACHE}（bench/ 按设计不入库；host p2-05 跑后回归）")
    row = next(iter(cache.index.values()))
    model_id = str(row["model_id"])
    import json as _json
    keys = _json.loads(row["option_keys"])
    prompt = str(row["prompt"])

    assert model_id == SCAFFOLD_MODEL_ID and model_id.endswith("#scaffold-cpu")
    assert teacher_full_dist(cache, model_id, prompt, keys) is not None
    assert cache.stats["hits"] >= 1
    before = cache.stats["hits"]
    assert teacher_full_dist(cache, DEFAULT_TEACHER_MODEL_ID, prompt, keys) is None
    assert cache.stats["misses"] >= 1 and cache.stats["hits"] == before   # 换人即 miss，没串味


def test_cache_restrict_normalize_equals_direct_subset_softmax():
    """等价式（本域架构支柱）：全池 softmax 限制到子集再重归一 ≡ 直接对子集 softmax。

    这条式子成立，才允许"一次全键产标覆盖所有 top-k 子集"，而不必为 C(20,8) 组合另产教师标。
    """
    logits = [float(i) for i in range(20)]
    full = _softmax(logits)
    subset = [3, 7, 11, 19]
    dist = {LETTERS[i]: full[i] for i in range(20)}

    restricted = restrict_normalize(dist, [LETTERS[i] for i in subset])

    assert pytest.approx(sum(restricted.values()), rel=1e-12) == 1.0
    for pos, i in enumerate(subset):
        assert restricted[LETTERS[i]] == pytest.approx(_softmax([logits[j] for j in subset])[pos], rel=1e-9)

    hot = _softmax([v / 3.0 for v in logits])                              # 温度侧同样可交换
    warm = restrict_normalize({LETTERS[i]: hot[i] for i in range(20)}, [LETTERS[i] for i in subset])
    assert list(warm.values()) == pytest.approx(_softmax([logits[j] / 3.0 for j in subset]), rel=1e-9)

    with pytest.raises(ValueError):
        restrict_normalize({"A": 1.0, "B": 0.0}, ["B"])                    # 选中列全零无法归一
    assert restrict_normalize({"A": 2.0, "B": 3.0}, ["A", "B"]) == {"A": 0.4, "B": 0.6}


def test_cache_teacher_on_subset_returns_none_when_letters_do_not_align(tmp_path):
    """缓存行里的键对不上学生选的列：交 None，绝不猜"大概按代号存的"再拿零份额归一。"""
    row = _opd_sample(subset=("A", "D"), shares=(0.5, 0.5))
    cache = _tmp_cache(tmp_path, SCAFFOLD_MODEL_ID, [("some prompt", ["W", "X"], {"W": 0.5, "X": 0.5})])

    assert teacher_on_subset(cache, SCAFFOLD_MODEL_ID, "some prompt", ["W", "X"], list(row.letters)) is None
    assert teacher_on_subset(None, SCAFFOLD_MODEL_ID, "any", ["A"], ["A"]) is None


def test_cache_collected_sample_target_is_normalized_within_subset(tmp_path):
    """装配出口的列纪律：目标在选中列内和为 1、未选中列为 0，教师列与 `restrict_normalize` 同值。"""
    codes = ["continue", "human_review", "observe", "stop"]
    records = _seeded_records(2, codes)
    student = _student([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] + [0.0] * (D - 8)])
    flat = _flat_rows(records, codes)
    weights = [1.0, 2.0, 3.0, 4.0]
    rows = _teacher_rows([p for p, _ in flat], [c for _, c in flat], weights)
    cache = _tmp_cache(tmp_path, SCAFFOLD_MODEL_ID, rows)

    samples, asm, tstats = collect_onpolicy(records, student, cache=cache,
                                            teacher_model_id=SCAFFOLD_MODEL_ID, k=2, mix_gt=0.5)
    assert len(samples) == 2 and cache.stats["hits"] >= 2 and cache.stats["pending"] == 0

    for one, (prompt, cols) in zip(samples, flat):
        assert one.letters == tuple(cols) and len(one.subset) == 2
        assert sum(one.keep) == 2 and one.keep[0] is False                # A 分数最低，不会被选中
        kept = [v for v, on in zip(one.target, one.keep) if on]
        assert pytest.approx(sum(kept), rel=1e-9) == 1.0                  # 选中列内和为 1
        assert [v for v, on in zip(one.target, one.keep) if not on] == [0.0, 0.0]
        expect_t = restrict_normalize(dict(zip(cols, [w / 10.0 for w in weights])), list(one.subset))
        assert [one.teacher[LETTERS.index(ch)] for ch in one.subset] == pytest.approx(
            list(expect_t.values()), rel=1e-9)                            # 教师那份就是重归一的结果
        assert one.cache_key == make_key(SCAFFOLD_MODEL_ID, prompt, list(cols))
        assert 0.0 < one.mass < 1.0
# ====================================================== 工作项 C：训练与消融的编排契约（run 本体待 C5）
class _ToyAttn(torch.nn.Module):
    """真会算的注意力壳：qkvo 四枚 `Linear`，好让 p2-05 的 `inject_lora` 有地方装垫片。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(dim, dim, bias=False)
        self.k_proj = torch.nn.Linear(dim, dim, bias=False)
        self.v_proj = torch.nn.Linear(dim, dim, bias=False)
        self.o_proj = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(torch.tanh(self.v_proj(x)) * torch.sigmoid(self.q_proj(x)))


class _ToyTrainableBody(torch.nn.Module):
    """能真训的假学生壳：编号嵌入 → 两层 qkvo → 逐位置输出（26 维，与假读出尺同宽）。

    为什么要有它：`train()` 的分桶、调度、梯度裁剪、run 四件套与笔记是本域交付物，
    这些只在"有可训参数、有计算图"的对象上才跑得起来；C1 正式档必须等 C5 的 SFT 产物，
    本地半场就用这副小壳把**管路**验通——里面的数不代表任何真模型，也不当成任何评测结论。
    """

    def __init__(self, dim: int = D, vocab: int = VOCAB, layers: int = 2) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.layers = torch.nn.ModuleList([_ToyAttn(dim) for _ in range(layers)])
        self.calls = 0

    def forward(self, input_ids=None, attention_mask=None, use_cache: bool = True,  # noqa: ARG002
                return_dict: bool = True, **_: object) -> SimpleNamespace:
        self.calls += 1
        x = self.emb(input_ids)
        for layer in self.layers:
            x = layer(x)
        return SimpleNamespace(last_hidden_state=x)


def _trainable_student(seed: int = 7) -> _FakeStudent:
    """造一副"有 LoRA 可装、有图可反传"的假学生（train/消融管路专用，数值不作数）。"""
    torch.manual_seed(seed)
    body = _ToyTrainableBody()
    return _FakeStudent(body=body, tokenizer=_FakeTokenizer(), head_weight=_head_weight(),
                        letter_ids=tuple(LETTER_AT[c] for c in LETTERS), pad_id=PAD_ID, model=body,
                        provenance={"snapshot_path": "unit", "source": "unit-test"})


def _local_half_cfg(tmp_path, cache, **extra):
    """本地半场的 train 配置：题源与缓存都由测试注入，记录本指到 tmp，绝不碰真 runs/。"""
    cfg = load_config(None, overrides={
        "axis": "quality", "limit": 4, "top_k": 2, "max_tokens": 128, "epochs": 1.0, "max_steps": 2,
        "accum": 1, "r": 2, "alpha": 4, "dropout": 0.0, "threads": 1,
        "teacher_model_id": SCAFFOLD_MODEL_ID, "teacher_cache": str(cache.root),
        "parent_run_id": "unit-parent-sft", "run_prefix": "p2-06-unit", "runs_root": str(tmp_path / "runs"),
    })
    cfg.update(extra)
    return cfg


def test_train_local_half_writes_run_and_notes_with_zero_teacher_forwards(tmp_path, monkeypatch):
    """C 档管路自测（**不是 C1**）：装配→两步散度下降→落产物与 notes，教师前向恒为 0。

    C1 的门槛是"载 C5 正式 SFT 产物在正式档上跑 tiny OPD run"，本波拿不到那份产物，
    所以这里只钉两件事：① `parent_run_id` 进了 run 配置（起点可追溯）；
    ② notes 的三行与"零教师前向"结论由真实数字填出。用的学生是 26 维假壳，数值不作数。
    """
    from production import opd as opd_module                       # 就地换题源（train 走模块全局）

    codes = ["continue", "human_review", "observe", "stop"]
    records = _seeded_records(4, codes)
    monkeypatch.setattr(opd_module, "load_quality_records", lambda axis, data_dir=None: list(records))

    student = _trainable_student()
    flat = _flat_rows(records, codes)
    cache = _tmp_cache(tmp_path, SCAFFOLD_MODEL_ID,
                       _teacher_rows([p for p, _ in flat], [c for _, c in flat], [1.0, 2.0, 3.0, 4.0]))
    teacher = TextTeacher(_StrictBackend(SCAFFOLD_MODEL_ID), cache=cache, stats=TeacherStats())

    result = train(_local_half_cfg(tmp_path, cache), student=student, cache=cache, teacher=teacher)

    assert result["samples"] == 4 and result["steps"] == 2 and result["parent_run_id"] == "unit-parent-sft"
    assert result["teacher_forward_calls"] == 0 and teacher.backend.forwards == 0
    assert result["assemble"]["teacher_hits"] == 4 and result["assemble"]["teacher_misses"] == 0
    assert all(math.isfinite(v) and v >= 0.0 for v in result["losses"])

    run_dir = Path(result["run_dir"])
    notes = (run_dir / "notes.md").read_text(encoding="utf-8")
    assert "零教师前向=True" in notes and "假设" in notes and "观察" in notes and "结论" in notes
    assert "对照 GKD" in notes                                           # design 要求记的差异点也落了账
    cfg_text = (run_dir / "config.yaml").read_text(encoding="utf-8")
    assert "unit-parent-sft" in cfg_text and OPD_VERSION in cfg_text
    assert (Path(result["model_dir"]) / "adapter.safetensors").exists()
    assert (Path(result["model_dir"]) / "decision_config.json").exists()
    assert student.body.calls > 0                                   # 真跑过学生前向（不是空转的假绿）


def test_ablation_plan_pairs_differ_only_by_truncation():
    """消融对是单变量对：两档只差"是否按学生自采样截断"，种子/题源/步数逐字相同。"""
    cfg = load_config(None, overrides={"top_k": 6, "seed": 123, "limit": 32})
    on, off = ablation_plan(cfg)

    diffs = {key for key in set(on) | set(off) if on.get(key) != off.get(key)}
    assert diffs == {"opd_on", "top_k", "run_prefix", "ablation_of"}
    assert on["opd_on"] is True and on["top_k"] == 6
    assert off["opd_on"] is False and off["top_k"] == 0             # off = 不截断 = 退化 soft-KD
    assert on["seed"] == off["seed"] == 123 and on["limit"] == off["limit"] == 32
    assert (on["ablation_of"], off["ablation_of"]) == ("on", "off")
    assert on["run_prefix"].endswith("-on") and off["run_prefix"].endswith("-off")


def test_ablation_off_arm_untruncates_the_whole_candidate_table(tmp_path):
    """同一批题、同一个只读缓存：on 档只较真 top-2，off 档摊开全部候选（截断就是唯一变量）。"""
    codes = ["continue", "human_review", "observe", "stop"]
    records = _seeded_records(2, codes)
    flat = _flat_rows(records, codes)
    cache = _tmp_cache(tmp_path, SCAFFOLD_MODEL_ID,
                       _teacher_rows([p for p, _ in flat], [c for _, c in flat], [1.0, 2.0, 3.0, 4.0]))

    on_cfg, off_cfg = ablation_plan(load_config(None, overrides={"top_k": 2}))
    assert (on_cfg["top_k"], off_cfg["top_k"]) == (2, 0)

    outs = {}
    for name, one in (("on", on_cfg), ("off", off_cfg)):
        student = _student([[float(i) for i in range(D)]])
        samples, asm, _ = collect_onpolicy(records, student, cache=cache,
                                          teacher_model_id=SCAFFOLD_MODEL_ID, k=one["top_k"])
        assert len(samples) == 2 and asm.hits == 2
        outs[name] = samples
        for s in samples:
            assert sum(s.keep) == (2 if name == "on" else len(s.letters))
            assert pytest.approx(sum(v for v, on in zip(s.target, s.keep) if on), rel=1e-9) == 1.0
    assert [s.subset for s in outs["on"]] == [("D", "C"), ("D", "C")]   # 同一题同一份截断（可复现）
    assert outs["off"][0].subset == ("D", "C", "B", "A")               # 全候选摊开


def test_ablation_delta_line_keeps_zero_negative_and_missing_honest():
    """结论行：零增益写 +0.00pp、负增益原样带负号、没评出来就 n/a（绝不当成 0 混进结论）。"""
    zero = delta_line({"acc": 0.5, "ece": 0.1, "n": 10}, {"acc": 0.5, "ece": 0.1, "n": 10},
                      on_run="R-on", off_run="R-off", seed=1)
    assert zero.startswith("OPD 增益 Δ=acc +0.00pp / ECE +0.000000")
    assert "R-on/R-off" in zero and "seed 1" in zero

    worse = delta_line({"acc": 0.4, "ece": 0.2, "n": 5}, {"acc": 0.5, "ece": 0.1, "n": 5})
    assert "-10.00pp" in worse and "+0.100000" in worse               # 负增益照写

    blind = delta_line({"acc": None, "ece": None, "n": 0}, {"acc": 0.5, "ece": 0.1, "n": 4})
    assert "n/a" in blind and "0 题" in blind                          # 没数就说没数


def test_ablation_writes_the_delta_line_into_both_notes(tmp_path):
    """消融档把同一行 Δ 结论写进两本 notes（train/eval 都注入假件，只验编排与落账）。"""
    def fake_train(one_cfg):
        where = tmp_path / one_cfg["ablation_of"]
        where.mkdir(parents=True, exist_ok=True)
        (where / "notes.md").write_text("# run\n\n结论：占位\n", encoding="utf-8")
        return {"run_id": f"run-{one_cfg['ablation_of']}", "run_dir": str(where),
                "model_dir": str(where), "steps": 1}

    seen: list[str] = []
    calls = {"n": 0}

    def counting_train(one_cfg):
        calls["n"] += 1
        seen.append(one_cfg["run_prefix"])
        return fake_train(one_cfg)

    out = ablation(load_config(None, overrides={"seed": 9, "top_k": 4}), train_fn=counting_train,
                   eval_fn=lambda path, **kw: {"acc": 0.62, "ece": 0.11, "n": 20})

    assert calls["n"] == 2 and seen == ["p2-06-dev-on", "p2-06-dev-off"]
    assert out["note"].startswith("OPD 增益 Δ=acc +0.00pp / ECE +0.000000")
    assert out["delta"] == {"acc": 0.0, "ece": 0.0}
    assert out["on"]["run_id"] == "run-on" and out["off"]["run_id"] == "run-off"
    for which in ("on", "off"):
        text = (tmp_path / which / "notes.md").read_text(encoding="utf-8")
        assert "结论：占位" in text and "OPD 增益 Δ=" in text          # 追加而非覆盖原有结论


def test_ablation_reports_eval_absence_instead_of_inventing_numbers(tmp_path):
    """没接评出口时 Δ 记 n/a，并在 metrics 里写明原因——不拿 0 冒充"零增益"。"""
    out = ablation(load_config(None, overrides={"seed": 9}), train_fn=lambda one_cfg: {
        "run_id": f"run-{one_cfg['ablation_of']}",
        "run_dir": str(_mk_note(tmp_path, one_cfg["ablation_of"])),
        "model_dir": "unused", "steps": 1}, eval_fn=None)

    assert out["metrics"]["on"]["reason"] == "未接评出口"
    assert "n/a" in out["note"] and out["delta"] == {"acc": None, "ece": None}


def _mk_note(tmp_path, which: str) -> Path:
    """给消融档补一间带 notes.md 的假 run 屋（只服务结论行的落账）。"""
    where = tmp_path / which
    where.mkdir(parents=True, exist_ok=True)
    (where / "notes.md").write_text("# run\n", encoding="utf-8")
    return where


def test_cli_print_config_merges_defaults_yaml_and_flags(capsys, tmp_path):
    """命令行 `--print-config` 只看三层合流结果：底单 ← yaml ← 显式覆盖，且 mix_soft 跟随 mix_gt。"""
    import json

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("top_k: 5\nloss: jsd\n", encoding="utf-8")

    code = main(["--config", str(yaml_path), "--print-config", "--mix-gt", "0.25", "--opd-on"])
    payload = json.loads(captured(capsys))

    assert code == 0
    assert payload["top_k"] == 5 and payload["loss"] == "jsd"          # yaml 层生效
    assert payload["mix_gt"] == 0.25 and payload["mix_soft"] == pytest.approx(0.75)
    assert payload["opd_on"] is True                                   # 显式开关生效
    assert payload["device"] == "cpu" and payload["allow_online_teacher"] is False
    assert payload["top_k"] != DEFAULT_TOP_K                           # 确实被覆盖过，不是底单原样


def captured(capsys) -> str:
    """取 stdout 里最后一段 JSON（`--print-config` 只打一张表，但为防别处有噪声只吃尾块）。"""
    text = capsys.readouterr().out
    return text[text.index("{"):]


def test_cli_and_config_reject_bad_switches(capsys):
    """拼错一个键、写错一个档名都当场炸：静默失效的开关比报错贵得多。"""
    with pytest.raises(SystemExit):
        main(["--not-a-key", "1"])
    assert "unrecognized" in capsys.readouterr().err

    with pytest.raises(ValueError):
        load_config(None, overrides={"nope": 1})
    with pytest.raises(ValueError):
        load_config(None, overrides={"loss": "forward_kl"})
    with pytest.raises(ValueError):
        load_config(None, overrides={"gt_mode": "soft"})

    cfg = load_config(None, overrides={"top_k": None, "seed": 5})      # None 视为"没写"，不覆盖
    assert cfg["top_k"] == DEFAULT_TOP_K and cfg["seed"] == 5


def test_default_cfg_is_cpu_cache_only_and_parent_traceable():
    """底单契约（C1/C5 接手时靠它）：CPU 默认、只读缓存、在线教师默认关、起点字段留空可追。"""
    cfg = default_cfg()

    assert cfg["device"] == "cpu" and cfg["dtype"] == "float32"
    assert cfg["allow_online_teacher"] is False and cfg["opd_on"] is True
    assert cfg["top_k"] == DEFAULT_TOP_K and cfg["mix_gt"] == DEFAULT_MIX_GT
    assert cfg["loss"] == REVERSE_KL and cfg["jsd_beta"] == DEFAULT_JSD_BETA
    assert cfg["teacher_cache"] == "bench/teacher_cache/p2_05_pseudo"
    assert cfg["parent_run_id"] == "" and cfg["parent_model"] == ""
    assert cfg["mix_soft"] == pytest.approx(1.0 - DEFAULT_MIX_GT)
    assert cfg["kernel_backend"] == "torch" and cfg["ascend_target"] == "cpu"


# ============================================================ 替身接口自测（integration；C1 正式档待 C5）
@pytest.mark.integration
def test_integration_real_student_interface_with_pseudo_teacher():
    """接口自测（**替身口径**，非 C1）：p2-05 dev 档的 0.6B+垫片 + 袖珍教师缓存走通 OPD 一步。

    默认不跑（要 `SYS1_TEACHER_INTEGRATION=1`），因为它要载 0.6B 权重。这里只验接口：
    真分词器、真末位读出、真 LoRA 垫片在 `sample_topk / collect_onpolicy / opd_step` 上
    形状对得上、教师零前向守得住；损失数值不代表正式档结论（正式档在 C5 之后按 910B 跑）。
    """
    if not os.environ.get("SYS1_TEACHER_INTEGRATION"):
        pytest.skip("载真权重：需 SYS1_TEACHER_INTEGRATION=1")
    from production.sft import BASELINE_LORA, build_student, inject_lora, load_adapter
    from sys1.eval.run import load_axis_records

    student = build_student("qwen3-0.6b", loader="minimal", device="cpu", dtype="float32")
    loras = inject_lora(student.body, r=int(BASELINE_LORA["r"]), alpha=int(BASELINE_LORA["alpha"]),
                        dropout=float(BASELINE_LORA["dropout"]), backend="torch", ascend_target="cpu")
    loaded = load_adapter(loras, DEV_SFT_MODEL)                        # dev 档产物当替身起点
    assert loaded > 0

    cache = DistCache(root=PSEUDO_CACHE, read_only=True).load()
    records = load_axis_records("quality", limit=16)
    teacher = TextTeacher(_StrictBackend(SCAFFOLD_MODEL_ID), cache=cache, stats=TeacherStats())

    samples, asm, tstats = collect_onpolicy(
        records, student, cache=cache, teacher_model_id=SCAFFOLD_MODEL_ID,
        k=DEFAULT_TOP_K, limit=4, teacher=teacher,   # 只借它的账本，不开在线档
    )
    assert samples, f"袖珍缓存没覆盖这批题：{asm.as_dict()}"
    assert tstats.forward_calls == 0 and asm.hits > 0                  # 只吃 parquet，一个新键都不发


    picked = samples[0]
    batch = collate_opd(samples, list(range(len(samples))), pad_id=student.pad_id,
                        letter_map=student.letter_ids).to_("cpu")
    assert batch.ids.shape[0] == len(samples)
    assert batch.cols == max(len(s.letters) for s in samples)
    assert len(picked.subset) <= DEFAULT_TOP_K
    assert pytest.approx(sum(v for v, on in zip(picked.target, picked.keep) if on), rel=1e-6) == 1.0

    loss = opd_step(student, batch)
    loss.backward()
    grads = [mod.lora_b.grad for mod in loras.values() if mod.lora_b.grad is not None]

    assert torch.isfinite(loss) and float(loss.detach()) >= 0.0
    assert grads and all(torch.isfinite(g).all() for g in grads)       # 真垫片确实吃到了梯度
    assert teacher.backend.forwards == 0                               # 全程一个新教师请求都不发
