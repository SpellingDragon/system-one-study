"""教师适配层单测：接口契约 / 数值一致性 / 缓存零前向 / 分域抽取 / 视觉回放与产标。

分层说明（为什么要两套假件）：
  本文件默认全离线、全 mock——用 `_FakeTokenizer` + `_FakeLogitsModel` 走真实的
  `HfCausalLMBackend` 代码路径（含右补空洞取位、log_softmax、温度缩放、缓存读写），
  但权重是随机结构、字母分数由假件写死，于是能"精确复算"教师分布该是多少。
  真实权重与真实接口不混进默认单测：它们由 `integration` 标记 + 环境变量
  `SYS1_TEACHER_INTEGRATION=1` 双重门控（见 test_integration_*），跑不跑得动都由
  机器资源决定，不由本文件假装通过。

运行方式（务必在 release/ 下用 -m，保证 production 包可导入）：
    cd release && .venv/bin/python -m pytest tests/test_teachers.py -q
    ... -k "text|numeric|cache|vision"  # 对应各孙任务的验证命令
真实路径（B3）：
    SYS1_TEACHER_INTEGRATION=1 .venv/bin/python -m pytest tests/test_teachers.py -m integration -q
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from production.teachers import protocol as proto
from production.teachers.cache import CacheMissError, DistCache, image_digest, make_key
from production.teachers.text import (
    HfCausalLMBackend,
    TeacherStats,
    TextTeacher,
    check_letters,
    load_decision_config,
    softmax_over,
)
from production.teachers.vision import (
    GlmApiBackend,
    MissingApiKeyError,
    TeacherUnreachableError,
    VisionTeacher,
    confidence_distribution,
    encode_image,
    pack_rows,
    parse_confidences,
)

REPO_RELEASE = Path(__file__).resolve().parents[1]
STARTLUX_SNAPSHOT = REPO_RELEASE / "bench" / "ms_models" / "models" / \
    "StartLuxAI--StartLux-Decision-4B" / "snapshots" / "master"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# 真实积分测试用的教师原始分数（写死，好让期望值可手算）
RAW_SCORES = {"A": 2.0, "B": -1.0, "C": 0.5, "D": 0.0}
KEYS = ["A", "B", "C"]
PAD_ID = 0
UNKNOWN_ID = 60
VOCAB = 64


# ── 假件：最小的"能骗过真实代码路径"的分词器与模型 ──────────────────────────
class _FakeTokenizer:
    """字符级假分词器：字母各占一格，其余字符统一给同一个号，右补空洞。

    只实现 `HfCausalLMBackend` 真用到的那几个动作（encode / 批量编码 / padding_side /
    chat_template 探测），故意不提供模板，好让 `_prepare` 走"无模板加思考关闭后缀"分支。
    """

    padding_side = "right"
    chat_template = None
    pad_token_id = PAD_ID
    eos_token = "</s>"

    def __init__(self, with_template: bool = False) -> None:
        if with_template:
            self.chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"

    @staticmethod
    def _id(ch: str) -> int:
        if ch in LETTERS:
            return LETTERS.index(ch) + 1
        return UNKNOWN_ID

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:  # noqa: FBT002, FBT001
        return [self._id(ch) for ch in text]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:  # noqa: FBT001, ARG002
        return "".join(chr(i + 61) for i in ids if 0 < i < 26)

    def __call__(self, texts, return_tensors: str = "pt", padding: bool = True, add_special_tokens: bool = False):  # noqa: FBT001, ARG002
        if isinstance(texts, str):
            texts = [texts]
        rows = [self.encode(t) for t in texts]
        width = max(len(r) for r in rows)
        ids, mask = [], []
        for row in rows:
            pad = [PAD_ID] * (width - len(row))
            ids.append(row + pad)          # 右补：空洞在真符号之后
            mask.append([1] * len(row) + [0] * len(pad))
        return _Encoding(
            input_ids=torch.tensor(ids, dtype=torch.long),
            attention_mask=torch.tensor(mask, dtype=torch.long),
        )


class _Encoding(dict):
    """假装是批编码结果：按下标取张量、`.to()` 原地自返（假件不真的搬设备）。"""

    def to(self, device=None):  # noqa: ARG002
        return self


class _FakeLM:
    """只在"每行真实末位"放分数的假模型：读错位置就全是 0，单测当场发现。"""

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.scores = scores or RAW_SCORES
        self.calls = 0

    def __call__(self, input_ids=None, attention_mask=None, use_cache: bool = False, **kwargs):  # noqa: ARG002, FBT002
        self.calls += 1
        batch, width = input_ids.shape
        logits = torch.zeros(batch, width, VOCAB)
        for row in range(batch):
            pos = int(attention_mask[row].sum().item()) - 1   # 真末位下标
            for letter, value in self.scores.items():
                logits[row, pos, LETTERS.index(letter) + 1] = float(value)
        return SimpleNamespace(logits=logits)

    def generate(self, **kwargs):
        raise AssertionError("主路不该调 generate（只有 answer 兜底路径才生成）")


def make_backend(scores: dict[str, float] | None = None, *, tokenizer: _FakeTokenizer | None = None) -> HfCausalLMBackend:
    """装一台假教师：真后端类 + 假分词器 + 假模型（走真实取位与折份额逻辑）。"""
    return HfCausalLMBackend(_FakeLM(scores), tokenizer or _FakeTokenizer(), "fake-teacher")


def expected_dist(raw: dict[str, float], *, temperature: float = 1.0) -> dict[str, float]:
    """独立复算期望分布：先对原始分数做 log_softmax，再按温度折份额（与实现无共享代码）。"""
    z = [raw[k] for k in KEYS]
    shift = max(z)
    denom = sum(math.exp(v - shift) for v in z)
    logp = {k: v - shift - math.log(denom) for k, v in zip(KEYS, z)}
    return softmax_over(logp, temperature=temperature)


def tiny_png(seed: int = 1) -> bytes:
    """造一张极小但合法的 PNG 字节（标准库 zlib 手搓，不依赖 PIL）。"""
    import struct
    import zlib

    raw = bytes([0] + [seed, 255 - seed, 7] * 2)  # 2x1 图像，每行前置 filter 字节
    chunk = lambda tag, data: struct.pack(  # noqa: E731
        ">I", len(data)
    ) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200, text: str = "") -> None:
        self._payload, self.status_code, self.text = payload, status, text or json.dumps(payload)

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    """假 HTTP 会话：按脚本依次回话，并记下每次请求体（用于断言"没把密钥写进正文"）。"""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []

    def post(self, url, json=None, headers=None, timeout=None):  # noqa: ARG002
        self.requests.append({"url": url, "json": json, "headers": dict(headers or {})})
        text = self.replies.pop(0) if self.replies else ""
        return _FakeResponse({"choices": [{"message": {"content": text}}], "usage": {
            "prompt_tokens": 11, "completion_tokens": 7}})


def integration_only(reason: str) -> None:
    """真实路径统一门控：没打标记或没显式开环境变量就 skip（绝不静默假通过）。"""
    if os.environ.get("SYS1_TEACHER_INTEGRATION") != "1":
        pytest.skip(f"{reason}（设 SYS1_TEACHER_INTEGRATION=1 并加 -m integration 才会真跑）")


# ── A1 接口契约 ──────────────────────────────────────────────────────────────
def test_text_score_options_returns_normalized_distribution():
    """主入口交出"每个字母各占多少"的份额，和为 1、键与候选一致。"""
    teacher = TextTeacher(make_backend())
    dist = teacher.score_options("Question: which one?\nOptions:\nA. x\nB. y\nC. z", KEYS)
    assert list(dist) == KEYS
    assert sum(dist.values()) == pytest.approx(1.0)
    assert teacher.stats.forward_calls == 1


def test_text_score_options_rejects_empty_option_keys():
    """候选清单为空必须直接报错——没有候选就没有分布，绝不返回空字典糊弄上层。"""
    teacher = TextTeacher(make_backend())
    with pytest.raises(ValueError, match="不能为空"):
        teacher.score_options("p", [])


def test_text_right_padding_readout_uses_true_last_position():
    """一批长度不齐的请求：必须按每行真实长度取末位，补空洞的行不能读到空洞上。"""
    backend = make_backend()
    short, long_ = "A", "A" + "x" * 11
    rows = backend.last_letter_logits([short, long_], KEYS)
    assert rows[0] == {k: RAW_SCORES[k] for k in KEYS}
    assert rows[1] == {k: RAW_SCORES[k] for k in KEYS}
    # 反证：若实现取了补出来的空洞位（全 0），三行分数会挤成一样——这里显式检查不是那样
    assert rows[0]["A"] != rows[0]["B"]


def test_text_backend_rejects_left_padding():
    """分词器若是左补就当场拒绝：线性注意力/因果卷积会把空洞读进状态，读数被污染。"""
    tok = _FakeTokenizer()
    tok.padding_side = "left"
    with pytest.raises(ValueError, match="right"):
        HfCausalLMBackend(_FakeLM(), tok, "fake-teacher")


def test_text_backend_rejects_multi_token_option_key():
    """候选字母必须独占一格；多 token 的键（如 "AB" 或中文词）直接报错，不静默降级。"""
    backend = make_backend()
    with pytest.raises(ValueError, match="不是单 token"):
        backend.last_letter_logits(["Question: hi"], ["A", "AB"])


def test_text_messages_and_flat_prompt_agree():
    """messages 写法与整段文字写法必须算出完全一样的份额（同一模型同一请求）。"""
    teacher = TextTeacher(make_backend())
    prompt = "Question: which one?\nOptions:\nA. x\nB. y"
    flat = teacher.score_options(prompt, ["A", "B"], use_cache=False)
    via_messages = teacher.score_messages(
        [{"role": "system", "content": "Apply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only."},
         {"role": "user", "content": prompt}],
        ["A", "B"],
        use_cache=False,
    )
    assert flat == via_messages


def test_text_check_letters_accepts_real_seam_and_rejects_bad_vocab():
    """26 个字母接缝自检：合格的交回号表，被吞成半词的当场指名是哪个字母。"""
    ids = check_letters(_FakeTokenizer())
    assert len(ids) == 26 and len(set(ids)) == 26

    class _MergeY(_FakeTokenizer):
        def encode(self, text, add_special_tokens=False):  # noqa: ARG002
            if text == "Y":
                return [7, 8]
            return super().encode(text)

    with pytest.raises(ValueError, match="Y"):
        check_letters(_MergeY())


def test_text_batch_scores_in_one_forward_and_rejects_mixed_keys(tmp_path):
    """批量打分：候选集必须一致，命中缓存的行不占这趟前向。"""
    teacher = TextTeacher(make_backend(), cache=DistCache(root=tmp_path).load())
    first = teacher.score_options("prompt-1", KEYS)
    out = teacher.score_batch([("prompt-1", KEYS), ("prompt-2", KEYS)], qtype="choice")
    assert out[0] == first
    assert teacher.stats.forward_calls == 2          # 第二次批量只补了没算过的那条
    with pytest.raises(ValueError, match="一致"):
        teacher.score_batch([("p", KEYS), ("q", ["A", "B"])])


def test_text_generative_fallback_retries_then_returns_none():
    """兜底路径两次都抽不出答案就返回 None——绝不编一份份额出来。"""
    class _Chatty:
        model_id = "fake-gen"
        replies = ["I think A is plausible, but B.", ""]
        calls = 0

        def generate_text(self, prompt, *, max_new_tokens=12):  # noqa: ARG002
            text = self.replies[min(self.calls, len(self.replies) - 1)]
            self.calls += 1
            return text

    stub = _Chatty()
    teacher = TextTeacher(stub)
    assert teacher.score_options_generative("Question: x", ["A", "B"]) is None
    assert teacher.stats.generations == 2


def test_text_generative_fallback_hard_distribution():
    """兜底路径抽到确定字母时交近 one-hot：押中项拿 mass，其余均摊。"""
    class _Talker:
        model_id = "fake-gen"

        def generate_text(self, prompt, *, max_new_tokens=12):  # noqa: ARG002
            return "reasoning here <answer>b</answer>"

    dist = TextTeacher(_Talker()).score_options_generative("Question: x", ["A", "B", "C"])
    assert dist["B"] == pytest.approx(0.9)
    assert dist["A"] == pytest.approx(dist["C"])
    assert sum(dist.values()) == pytest.approx(1.0)


# ── A2 数值一致性 ─────────────────────────────────────────────────────────────
def test_numeric_distribution_matches_softmax_of_letter_logprobs():
    """核心口径：返回分布 == 字母末位 log_softmax 再过一次 softmax（本测试独立复算）。"""
    teacher = TextTeacher(make_backend())
    dist = teacher.score_options("Question: which?\nOptions:\nA. a\nB. b\nC. c", KEYS, use_cache=False)
    assert dist == pytest.approx(expected_dist(RAW_SCORES))


def test_numeric_temperature_only_rescales_confidence_not_argmax():
    """温度是单调缩放：谁排第一不变，但份额的胖瘦随温度变。"""
    base = expected_dist(RAW_SCORES)
    hot = softmax_over({k: math.log(v) for k, v in base.items()}, temperature=2.0)
    assert max(hot, key=hot.get) == max(base, key=base.get)
    assert hot[max(base, key=base.get)] < base[max(base, key=base.get)]


def test_numeric_explicit_temperature_beats_decision_config(tmp_path):
    """倍数优先级：调用方明说 > 教师小抄（按题型） > 构造默认。"""
    d = tmp_path / "w"
    d.mkdir()
    (d / "decision_config.json").write_text(json.dumps({"temperature_by_type": {"choice": 1.3872}}), encoding="utf-8")
    teacher = TextTeacher(make_backend(), settings=load_decision_config(d))
    via_config = teacher.score_options("p-num-1", KEYS, qtype="choice", use_cache=False, temperature=None)
    via_explicit = teacher.score_options("p-num-1", KEYS, temperature=1.5, use_cache=False)
    assert via_config == pytest.approx(expected_dist(RAW_SCORES, temperature=1.3872))
    assert via_explicit == pytest.approx(expected_dist(RAW_SCORES, temperature=1.5))


def test_numeric_decision_config_temperature_read_from_disk(tmp_path):
    """教师小抄直读：decision_config.json 里的题型倍数必须被真的读出来并用上。"""
    d = tmp_path / "weights"
    d.mkdir()
    (d / "decision_config.json").write_text(
        json.dumps({"temperature_by_type": {"choice": 1.25, "noul": 2.0}, "letter_token_ids": [1, 2]}), encoding="utf-8"
    )
    settings = load_decision_config(d)
    assert settings.temperature_for("choice") == (1.25, True)
    assert settings.temperature_for("score") == (1.0, False)   # 小抄没写的类型：如实标未标定
    assert settings.letter_token_ids == (1, 2)


def test_numeric_softmax_over_rejects_bad_temperature():
    """倍数必须为正有限数：0/负数/NaN 都会让份额失去意义，必须直接报错。"""
    with pytest.raises(ValueError):
        softmax_over({"A": 1.0, "B": 2.0}, temperature=0.0)
    with pytest.raises(ValueError):
        softmax_over({}, temperature=1.0)


# ── 分域抽取（交付项 2）───────────────────────────────────────────────────────
def test_protocol_answer_prompt_keeps_original_and_adds_rule():
    """约束话术只追加不改写：原请求逐字保留，尾部多一行格式要求。"""
    asked = proto.build_answer_prompt("Question: x\nOptions:\nA. a\nB. b", ["A", "B"], qtype="choice")
    assert asked.startswith("Question: x")
    assert "<answer>X</answer>" in asked and "one of {A/B}" in asked
    assert "true or false" in proto.build_answer_prompt("q", ["A", "B"], qtype="noul")
    assert "a single number" in proto.build_answer_prompt("q", ["A", "B"], qtype="score")
    with pytest.raises(ValueError):
        proto.build_answer_prompt("q", ["A"], qtype="bogus")


def test_protocol_choice_domain_exact_match():
    """字母域：装饰写法都收，但必须在候选集内，且只许出现一个字母。"""
    assert proto.extract_answer("<answer>A</answer>", qtype="choice", option_keys=["A", "B"]) == "A"
    assert proto.extract_answer("<answer>(b)</answer>", qtype="choice", option_keys=["A", "B"]) == "B"
    assert proto.extract_answer("<answer>Answer: C.</answer>", qtype="choice", option_keys=["A", "B", "C"]) == "C"
    assert proto.extract_answer("<answer>Z</answer>", qtype="choice", option_keys=["A", "B"]) is None
    assert proto.extract_answer("<answer>A or B</answer>", qtype="choice", option_keys=["A", "B"]) is None


def test_protocol_noul_domain_boolean():
    """真假域：中英文是/否说法折成 true/false；同现两类词判不可判定。"""
    for body, want in (("<answer>true</answer>", "true"), ("<answer>否</answer>", "false"),
                       ("<answer>YES</answer>", "true"), ("<answer>不对</answer>", "false")):
        assert proto.extract_answer(body, qtype="noul") == want
    assert proto.extract_answer("<answer>是的，但不是全部</answer>", qtype="noul") is None


def test_protocol_score_domain_with_tolerance():
    """数值域：剥单位/千分位后收第一个数；判分按相对容差带。"""
    assert proto.extract_answer("<answer>7.2 / 10</answer>", qtype="score") == repr(7.2)
    assert proto.extract_answer("<answer>1,024</answer>", qtype="score") == repr(1024.0)
    assert proto.judge("7.2", "7.4", qtype="score", rel_tol=0.05) is True
    assert proto.judge("7.2", "8.0", qtype="score", rel_tol=0.05) is False
    assert proto.judge("7.2", "7.4", qtype="score", rel_tol=0.0, abs_tol=0.3) is True


def test_protocol_nested_or_empty_answer_returns_none():
    """嵌套标签与空内容都算"没答"（GLM-V verifier 同款口径）；重试话术只加不改。"""
    assert proto.extract_answer("<answer><answer>A</answer></answer>", qtype="choice") is None
    assert proto.extract_answer("<answer>   </answer>", qtype="choice") is None
    assert proto.extract_answer("no tags here A", qtype="choice", option_keys=["A"]) is None
    assert proto.extract_answer("A", qtype="choice", option_keys=["A"], strict=False) == "A"
    retry = proto.retry_prompt("Question: x", ["A", "B"], qtype="choice")
    assert retry.startswith("Question: x") and "ONLY <answer>" in retry


def test_protocol_hard_distribution_rejects_out_of_set_answer():
    """答案不在候选集里时折分布必须报错，交给调用方按缺失处理。"""
    with pytest.raises(ValueError, match="不在候选"):
        proto.hard_distribution("D", ["A", "B", "C"])
    assert proto.hard_distribution("A", ["A"]) == {"A": 1.0}


# ── B1 parquet 缓存 ──────────────────────────────────────────────────────────
def test_cache_key_is_stable_and_sensitive_to_every_part():
    """键 = sha256(模型名+请求+候选)截 16 位：换任一项必变键，换顺序不变键。"""
    k1 = make_key("m1", "prompt", ["B", "A"])
    assert re.fullmatch(r"[0-9a-f]{16}", k1)
    assert k1 == make_key("m1", "prompt", ["A", "B"])          # 候选顺序无关（内部排序）
    assert k1 != make_key("m2", "prompt", ["A", "B"])          # 换教师必 miss
    assert k1 != make_key("m1", "prompt ", ["A", "B"])         # 请求逐字敏感
    assert k1 != make_key("m1", "prompt", ["A", "B"], image_hash="deadbeef")


def test_cache_second_call_zero_forward(tmp_path):
    """孙任务 B1 的核心断言：同一请求第二次命中缓存，前向计数不再增长。"""
    cache = DistCache(root=tmp_path).load()
    teacher = TextTeacher(make_backend(), cache=cache, stats=TeacherStats())
    prompt = "Question: again?\nOptions:\nA. a\nB. b"
    first = teacher.score_options(prompt, ["A", "B"])
    mid = teacher.stats.forward_calls
    second = teacher.score_options(prompt, ["A", "B"])
    assert second == pytest.approx(first)
    assert teacher.stats.forward_calls == mid                   # 一次都没多花
    assert teacher.stats.cache_hits == 1 and teacher.stats.cache_misses == 1
    assert teacher.stats.api_calls == 0


def test_cache_flush_then_reload_from_parquet(tmp_path):
    """落盘与再载入：新进程只读目录也能查到上次的份额（json 列往返一致）。"""
    cache = DistCache(root=tmp_path).load()
    teacher = TextTeacher(make_backend(), cache=cache)
    teacher.score_options("Question: disk\nOptions:\nA. a", ["A", "B"])
    assert cache.flush() == 1
    assert list(tmp_path.glob("*.parquet"))

    reopened = DistCache(root=tmp_path).load()
    hit = reopened.lookup(teacher.model_id, "Question: disk\nOptions:\nA. a", ["A", "B"])
    assert hit and sum(hit.values()) == pytest.approx(1.0)


def test_cache_read_only_refuses_writes(tmp_path):
    """离线伪标包按只读打开：任何写入尝试都要明确拒绝，不改脏别人的产标。"""
    writer = DistCache(root=tmp_path).load()
    writer.record(model_id="m", prompt="p", option_keys=["A"], dist={"A": 1.0})
    writer.flush()
    packed = DistCache(root=tmp_path, read_only=True).load()
    assert packed.lookup("m", "p", ["A"]) == {"A": 1.0}
    with pytest.raises(CacheMissError, match="只读"):
        packed.record(model_id="m", prompt="q", option_keys=["A"], dist={"A": 1.0})


def test_cache_merge_brings_foreign_rows_without_overwriting(tmp_path):
    """伪标包合并：同键保留本地已有行，只并进新键（防止外部页覆盖既有产标）。"""
    home = DistCache(root=tmp_path / "home").load()
    home.record(model_id="m", prompt="same", option_keys=["A"], dist={"A": 0.7}, source="local")
    home.flush()
    foreign = DistCache(root=tmp_path / "foreign").load()
    foreign.record(model_id="m", prompt="same", option_keys=["A"], dist={"A": 0.1})
    foreign.record(model_id="m", prompt="new", option_keys=["A"], dist={"A": 0.9})
    foreign.flush()
    assert home.merge(tmp_path / "foreign") == 1
    assert home.lookup("m", "same", ["A"]) == {"A": 0.7}
    assert home.lookup("m", "new", ["A"]) == {"A": 0.9}


def test_cache_default_root_is_bench_teacher_cache():
    """默认目录写在 bench 下（已被 .gitignore 覆盖），不会把产标结果混进版本库。"""
    assert DistCache().root == "bench/teacher_cache"
    ignored = Path(__file__).resolve().parents[2] / ".gitignore"
    text = ignored.read_text(encoding="utf-8") if ignored.is_file() else ""
    assert "bench" in text or "bench/" in text


# ── C1 视觉教师 ──────────────────────────────────────────────────────────────
def test_vision_pack_replay_zero_api_calls(tmp_path):
    """回放模式：包里有标签就原样交出，一次在线请求都不发（§11-4 红线）。"""
    cache = DistCache(root=tmp_path).load()
    image = tiny_png(3)
    digest = image_digest(image)
    cache.record(model_id="glm-5.3-flash", prompt="what color?", option_keys=["A", "B"],
                 dist={"A": 0.75, "B": 0.25}, image_hash=digest, source="offline_pack")
    cache.flush()
    teacher = VisionTeacher(DistCache(root=tmp_path).load(), mode="pack")
    dist = teacher.score(image, "what color?", ["A", "B"])
    assert dist == {"A": 0.75, "B": 0.25}
    assert teacher.stats.api_calls == 0 and teacher.stats.cache_hits == 1


def test_vision_pack_miss_raises_instead_of_going_online(tmp_path):
    """回放模式查不到必须显式失败：训练/评测链不得偷偷补一次在线请求。"""
    teacher = VisionTeacher(DistCache(root=tmp_path).load(), mode="pack")
    with pytest.raises(TeacherUnreachableError, match="不得在线"):
        teacher.score(tiny_png(9), "unknown question", ["A", "B"])


def test_vision_api_confidence_distribution_main_path(tmp_path, monkeypatch):
    """产标主路：逐候选 0–10 信心 → 线性摊成份额，并回写缓存（键含图指纹）。"""
    monkeypatch.setenv("ZAI_API_KEY", "dummy-key-for-unit-test")
    session = _FakeSession(["A=8\nB=4\nC=0"])
    backend = GlmApiBackend(session=session, stats=TeacherStats())
    cache = DistCache(root=tmp_path).load()
    teacher = VisionTeacher(cache, mode="api", backend=backend)
    dist = teacher.score(tiny_png(5), "Question: which?\nOptions:\nA. a\nB. b\nC. c", ["A", "B", "C"])
    assert dist == pytest.approx(confidence_distribution({"A": 8.0, "B": 4.0, "C": 0.0}))
    assert teacher.stats.api_calls == 1 and teacher.stats.forward_calls == 0
    assert cache.flush() == 1
    again = teacher.score(tiny_png(5), "Question: which?\nOptions:\nA. a\nB. b\nC. c", ["A", "B", "C"])
    assert again == pytest.approx(dist) and teacher.stats.api_calls == 1   # 第二次零在线
    # 密钥只该出现在请求头，绝不进请求正文
    assert "dummy-key-for-unit-test" not in json.dumps(session.requests[0]["json"])


def test_vision_api_answer_fallback_after_bad_format(tmp_path, monkeypatch):
    """主路格式不对 → 退 answer 协议；第一次仍不对则重试一轮（共两次在线请求）。"""
    monkeypatch.setenv("ZAI_API_KEY", "dummy")
    session = _FakeSession(["I am not sure", "plain prose again", "<answer>B</answer>"])
    backend = GlmApiBackend(session=session)
    teacher = VisionTeacher(DistCache(root=tmp_path).load(), mode="api", backend=backend)
    dist = teacher.score(tiny_png(7), "Question: pick", ["A", "B"])
    assert dist["B"] == pytest.approx(0.9)
    assert teacher.stats.api_calls == 3      # 1 次主路 + 2 次兜底
    assert "ONLY <answer>" in session.requests[2]["json"]["messages"][0]["content"][1]["text"]


def test_vision_api_two_failures_raises(tmp_path, monkeypatch):
    """外部教师两次都不守格式就显式失败——不产出半截标签，也不静默退均匀。"""
    monkeypatch.setenv("ZAI_API_KEY", "dummy")
    session = _FakeSession(["junk", "junk", "junk"])
    teacher = VisionTeacher(DistCache(root=tmp_path).load(), mode="api", backend=GlmApiBackend(session=session))
    with pytest.raises(TeacherUnreachableError, match="两次"):
        teacher.score(tiny_png(8), "Question: pick", ["A", "B"])


def test_vision_api_requires_env_key_and_never_bakes_one(tmp_path, monkeypatch):
    """没有环境变量密钥就停：绝不回退到任何内置密钥（防把凭据写进代码）。"""
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    teacher = VisionTeacher(DistCache(root=tmp_path).load(), mode="api",
                            backend=GlmApiBackend(session=_FakeSession(["A=1\nB=1"])))
    with pytest.raises(MissingApiKeyError, match="ZAI_API_KEY"):
        teacher.score(tiny_png(2), "Question: pick", ["A", "B"])


def test_vision_api_budget_circuit_breaker(tmp_path, monkeypatch):
    """产标预算用尽即熔断：批量打标跑偏时最坏只花到约定次数。"""
    monkeypatch.setenv("ZAI_API_KEY", "dummy")
    backend = GlmApiBackend(session=_FakeSession(["A=9\nB=1"] * 5), budget=1)
    teacher = VisionTeacher(DistCache(root=tmp_path).load(), mode="api", backend=backend)
    assert teacher.score(tiny_png(1), "q1", ["A", "B"])["A"] > 0.5
    with pytest.raises(TeacherUnreachableError, match="预算"):
        teacher.score(tiny_png(2), "q2", ["A", "B"])


def test_vision_http_error_is_exp_failure(tmp_path, monkeypatch):
    """非 200 一律收成显式不可达，且错误正文截断、不回显请求头。"""
    monkeypatch.setenv("ZAI_API_KEY", "dummy")

    class _Boom:
        def post(self, url, json=None, headers=None, timeout=None):  # noqa: ARG002
            return _FakeResponse({}, status=429, text="rate limited")

    backend = GlmApiBackend(session=_Boom())
    with pytest.raises(TeacherUnreachableError, match="429"):
        backend.confidences("AAAA", "q", ["A"])


def test_vision_parse_confidences_rejects_incomplete_or_out_of_range():
    """信心表解析：缺候选、越界（>10）都判整份不作数。"""
    assert parse_confidences("A=8\nB=4", ["A", "B"]) == {"A": 8.0, "B": 4.0}
    assert parse_confidences("A=8", ["A", "B"]) is None
    assert parse_confidences("A=8\nB=11", ["A", "B"]) is None
    assert parse_confidences("A : 8.5\nB=2", ["A", "B"]) == {"A": 8.5, "B": 2.0}
    assert parse_confidences("", ["A", "B"]) is None


def test_vision_confidence_distribution_is_linear_with_floor():
    """摊份额是线性而非指数：8 分就是 4 分的两倍（垫底后仍保持单调与和为 1）。"""
    dist = confidence_distribution({"A": 8.0, "B": 4.0, "C": 0.0}, floor=0.0)
    assert dist["A"] == pytest.approx(2 * dist["B"])
    assert dist["C"] == pytest.approx(0.0)
    assert sum(dist.values()) == pytest.approx(1.0)
    floored = confidence_distribution({"A": 0.0, "B": 0.0})
    assert floored == {"A": 0.5, "B": 0.5}      # 全 0 分退均匀，不出现纯零伪标
    assert min(confidence_distribution({"A": 10.0, "B": 0.0}).values()) > 0.0


def test_vision_encode_image_accepts_bytes_and_path(tmp_path):
    """图像入参收字节与路径，并给出内容指纹：同名换图必然换号。"""
    raw = tiny_png(4)
    b64, digest = encode_image(raw)
    assert isinstance(b64, str) and len(b64) > 0
    path = tmp_path / "x.png"
    path.write_bytes(raw)
    assert encode_image(path)[1] == digest                     # 同一内容同号
    assert encode_image(tiny_png(5))[1] != digest                      # 换内容必换号
    with pytest.raises(ValueError):
        encode_image(b"")
    with pytest.raises(ValueError):
        encode_image(object())


def test_vision_pack_rows_reads_back_shard(tmp_path):
    """包内容自检：掀盖数几行，供冒烟脚本与 run notes 记证据。"""
    cache = DistCache(root=tmp_path).load()
    cache.record(model_id="m", prompt="p1", option_keys=["A", "B"], dist={"A": 0.5, "B": 0.5}, source="offline_pack")
    cache.record(model_id="m", prompt="p2", option_keys=["A", "B"], dist={"A": 0.2, "B": 0.8}, source="offline_pack")
    cache.flush()
    rows = pack_rows(tmp_path, limit=1)
    assert len(rows) == 1 and rows[0]["source"] == "offline_pack"


# ── B3 真实路径（默认跳过；显式开启才真跑）──────────────────────────────────
@pytest.mark.integration
def test_integration_startlux_4b_tokenizer_and_config_seam():
    """真权重目录：26 字母接缝 + decision_config 直读 + 模板尾巴，一项不合格就红。"""
    integration_only("需要已下载的 StartLux-4B 快照")
    assert STARTLUX_SNAPSHOT.is_dir(), f"快照目录不存在：{STARTLUX_SNAPSHOT}"
    from transformers import AutoTokenizer

    settings = load_decision_config(STARTLUX_SNAPSHOT)
    assert settings.temperature_by_type, "decision_config.json 应带题型倍数"
    tok = AutoTokenizer.from_pretrained(str(STARTLUX_SNAPSHOT), trust_remote_code=True)
    ids = check_letters(tok, expected=settings.letter_token_ids)
    assert ids[0] == 32 and ids[-1] == 57                       # 与魔搭仓库实算一致
    prepared = HfCausalLMBackend.__new__(HfCausalLMBackend)
    prepared.tokenizer, prepared.apply_chat_template = tok, True
    text = prepared._prepare("Question: probe\nOptions:\nA. a\nB. b")
    assert text.endswith("<|im_end|>\n") or "Question: probe" in text


@pytest.mark.integration
def test_integration_glm_api_live_connectivity():
    """真打一次接口（1 条图文）：证明协议与端点可用；密钥只从环境变量取。"""
    integration_only("需要 ZAI_API_KEY 与外网连通")
    if os.environ.get("ZAI_API_KEY", "").strip() == "":
        pytest.skip("ZAI_API_KEY 未设置")
    backend = GlmApiBackend(budget=1)
    conf, info = backend.confidences(encode_image(tiny_png(1))[0], "Question: what color is this 2x1 image?", ["A", "B"])
    assert isinstance(conf, dict) or conf is None      # 只验连通与用量，不验对错
    assert info["seconds"] > 0 and backend.usage["calls"] == 1
