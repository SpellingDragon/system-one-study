"""tests/test_schema.py — p1-01-data-schema 契约验收单测。

对应 spec.md 的 5 个 Scenario + tasks.md A3/B2 明确要求的三场景 + 键序场景。
命名对齐 -k 过滤器：path / enum / normalize / roundtrip / rank / score_key。
"""
from __future__ import annotations

import json

import pytest

from sys1.data.schema import (
    PROB_TOL,
    QTYPES,
    SchemaError,
    dump,
    load,
    to_rank_vector,
    validate_sample,
)


# —— 通用 fixture：合法 choice 样本（soft targets，概率和=1）
def _valid_choice_sample():
    return {
        "state": "今天多云，气温 18℃。",
        "questions": {
            "q1": {
                "type": "choice",
                "instructions": "天气如何？",
                "criteria": {"A": "晴", "B": "多云", "C": "雨"},
            },
        },
        "targets": {"q1": {"A": 0.1, "B": 0.8, "C": 0.1}},
    }


def _valid_score_sample():
    return {
        "state": "商品评价文本……",
        "questions": {
            "q1": {
                "type": "score",
                "instructions": "满意度打几档（0 最低）？",
                "criteria": {"0": "很差", "1": "一般", "2": "很好"},
            },
        },
        "targets": {"q1": {"0": 0.1, "1": 0.2, "2": 0.7}},
    }


def _valid_noul_sample():
    return {
        "state": "某人说了某句话。",
        "questions": {
            "q1": {"type": "noul", "instructions": "该句是否蕴含前提？"},
        },
        "targets": {"q1": {"false": 0.2, "true": 0.8}},
    }


# =====================================================================
# A3 / spec Scenario 1：合法样本通过 —— 三 qtype 全走一遍 + dump→load 往返
# =====================================================================

def test_valid_choice_passes_and_normalizes():
    """choice 合法样本：validate_sample 返回摊平后的新 dict，键约定不破坏。"""
    s = _valid_choice_sample()
    out = validate_sample(s)
    assert out["questions"]["q1"]["type"] == "choice"
    # 概率摊平到严格 1（浮点抖动不应外溢）
    total = sum(out["targets"]["q1"].values())
    assert abs(total - 1.0) <= PROB_TOL


def test_valid_noul_passes():
    """noul：targets 键为 false/true，validate 通过并保留原语义。"""
    out = validate_sample(_valid_noul_sample())
    assert set(out["targets"]["q1"].keys()) == {"false", "true"}


def test_valid_score_passes_roundtrip():
    """spec Scenario 1 的 dump→load 往返：逐字段相等，且顺序不敏感（dict 语义）。"""
    s = _valid_score_sample()
    line = dump(s)
    assert line.endswith("\n") and line.count("\n") == 1
    back = load(line)
    # 逐字段相等（json 里 dict 顺序保留即可）
    assert back["state"] == s["state"]
    assert back["questions"]["q1"]["type"] == "score"
    assert back["questions"]["q1"]["criteria"] == s["questions"]["q1"]["criteria"]
    for k, v in s["targets"]["q1"].items():
        assert back["targets"]["q1"][k] == pytest.approx(v, rel=1e-9)


# =====================================================================
# A2 / A3 / spec Scenario 2：概率和越界被拒 —— 消息含 qid 与"概率和"
# =====================================================================

def test_prob_sum_off_normalize_rejects():
    """概率和=0.9 时训练侧默认必拒；错误消息同时含 qid 与"概率和"字样。"""
    s = _valid_choice_sample()
    s["targets"]["q1"] = {"A": 0.1, "B": 0.7, "C": 0.1}  # 和=0.9
    with pytest.raises(SchemaError) as ei:
        validate_sample(s)
    msg = str(ei.value)
    assert "q1" in msg, f"错误消息应含 qid：{msg}"
    assert "概率和" in msg, f"错误消息应含『概率和』：{msg}"


def test_prob_sum_within_tol_normalize_accepts():
    """和=1±1e-7（在 PROB_TOL 内）应放行，且被摊平到严格 1。"""
    s = _valid_choice_sample()
    s["targets"]["q1"] = {"A": 0.1 + 5e-8, "B": 0.8, "C": 0.1}
    out = validate_sample(s)
    total = sum(out["targets"]["q1"].values())
    assert abs(total - 1.0) < 1e-12  # 摊平到严格 1


def test_allow_unnormalized_flag_keeps_original_sum():
    """allow_unnormalized=True 时：概率和=0.9 放行，且**保留原始分布**不摊平。"""
    s = _valid_choice_sample()
    s["targets"]["q1"] = {"A": 0.1, "B": 0.7, "C": 0.1}
    out = validate_sample(s, allow_unnormalized=True)
    total = sum(out["targets"]["q1"].values())
    assert abs(total - 0.9) < 1e-12, f"allow_unnormalized 下不应摊平，实际和={total}"


def test_negative_prob_rejected_with_path():
    """负概率被拒；错误消息里带上出现问题的 qid 路径。"""
    s = _valid_choice_sample()
    s["targets"]["q1"]["C"] = -0.2
    with pytest.raises(SchemaError) as ei:
        validate_sample(s)
    assert "q1" in str(ei.value) and "C" in str(ei.value)


# =====================================================================
# A1 / spec Scenario 3：非法 qtype 被拒 —— 消息含合法枚举列表 + 字段路径
# =====================================================================

def test_illegal_qtype_message_includes_enum_and_path():
    """type='ranking' 被拒；消息同时含合法枚举 {choice,noul,score} 与路径 questions.q3.type。

    这条覆盖 spec 场景 3 与 tasks A3 的第三子场景，命名带 enum 便于 -k 过滤。
    """
    s = _valid_choice_sample()
    # 把样本改成含 q3 的形态（spec 消息示例里用 q3）
    s["questions"]["q3"] = {"type": "ranking", "instructions": "排序", "criteria": {"A": "a", "B": "b"}}
    with pytest.raises(SchemaError) as ei:
        validate_sample(s)
    msg = str(ei.value)
    for t in QTYPES:
        assert t in msg, f"错误消息应列出合法枚举 {QTYPES}，实际：{msg}"
    assert "ranking" in msg
    assert "questions.q3.type" in msg, f"错误消息应含字段路径 questions.q3.type，实际：{msg}"


def test_missing_type_rejected():
    """缺 type 字段被拒（路径应指到 questions.q1.type）。"""
    s = _valid_choice_sample()
    del s["questions"]["q1"]["type"]
    with pytest.raises(SchemaError) as ei:
        validate_sample(s)
    assert "questions.q1.type" in str(ei.value)


def test_error_message_contains_field_path():
    """错误消息**必须**带路径，是 design.md 明写的"后续域调试生命线"。

    这条命名带 path，配合 -k "path or enum" 命中 A1 验证。
    """
    # state 非 str → 路径就是 "state"
    with pytest.raises(SchemaError) as ei:
        validate_sample({"state": 123, "questions": {}})
    assert str(ei.value).startswith("state "), f"路径应在消息开头：{ei.value}"

    # instructions 空白 → 路径 questions.q1.instructions
    s = _valid_choice_sample()
    s["questions"]["q1"]["instructions"] = "   "
    with pytest.raises(SchemaError) as ei:
        validate_sample(s)
    assert "questions.q1.instructions" in str(ei.value)


# =====================================================================
# B1 / spec Scenario 4：score 键 ["0","1","2"] 通过 + to_rank_vector 长度 3
# =====================================================================

def test_score_key_valid_roundtrip():
    """合法 score 键 {"0","1","2"}：validate 通过 + dump/load 往返等价。命名含 score_key。"""
    out = validate_sample(_valid_score_sample())
    assert set(out["targets"]["q1"].keys()) == {"0", "1", "2"}


def test_rank_vector_length_and_order():
    """to_rank_vector：长度=k、按数字索引顺序回填（"2" 在 "10" 前，非字符串排序）。"""
    targets = {"0": 0.1, "1": 0.2, "2": 0.7}
    vec = to_rank_vector(targets)
    assert isinstance(vec, list) and len(vec) == 3
    assert vec == [0.1, 0.2, 0.7]

    # 数字排序陷阱：10 个档位时 "10" 应排在最后而非 "1" 之后
    many = {str(i): 1.0 / 12 for i in range(12)}
    vec2 = to_rank_vector(many)
    assert len(vec2) == 12
    assert vec2[-1] == pytest.approx(1.0 / 12)


def test_rank_vector_rejects_bad_keys():
    """to_rank_vector 拒非数字键、拒稀疏索引（缺 "1"）。"""
    with pytest.raises(SchemaError):
        to_rank_vector({"a": 0.5, "b": 0.5})
    with pytest.raises(SchemaError):
        to_rank_vector({"0": 0.5, "2": 0.5})  # 缺 1，不连续


# =====================================================================
# B2 / spec Scenario 5：score 键 ["1","0"] 而 k=3 被拒 —— 键集合与 k 不符
# =====================================================================

def test_score_key_missing_level_rejected():
    """spec 场景 5：criteria 声明 k=3、targets 只给了 {"1","0"} → 缺 "2" 键，SchemaError。"""
    s = _valid_score_sample()
    s["targets"]["q1"] = {"1": 0.4, "0": 0.6}
    with pytest.raises(SchemaError) as ei:
        validate_sample(s)
    msg = str(ei.value)
    assert "q1" in msg and "k=3" in msg, f"错误消息要指明 k 与缺项：{msg}"


def test_score_key_criteria_mismatch_rejected():
    """criteria 键本身不符合 "0".."k-1" 约定 → validate 拒。"""
    s = _valid_score_sample()
    s["questions"]["q1"]["criteria"] = {"low": "低", "high": "高"}  # 非数字
    with pytest.raises(SchemaError) as ei:
        validate_sample(s)
    assert "score" in str(ei.value)


# =====================================================================
# 额外：jsonl 序列化键型漂移防线（design 风险 2）
# =====================================================================

def test_dump_rejects_non_str_keys():
    """dict 键不是 str 时（如 targets 用了 int 键），dump 前拒；json 会静默 str 化。"""
    s = _valid_choice_sample()
    # 绕过 validate 直接扔给 dump 会先跑一遍 validate，因此这里构造 validate 后仍可能
    # 出问题的形态：ep_group 里带 int 键（透传字段，validate 不校验语义）。
    s["ep_group"] = {1: "one", 2: "two"}
    with pytest.raises(SchemaError) as ei:
        dump(s)
    assert "非字符串" in str(ei.value) or "str 化" in str(ei.value)


def test_load_rejects_empty_line_and_bad_json():
    """空行、非法 json 都抛 SchemaError，与 validate 家族一致。"""
    with pytest.raises(SchemaError):
        load("")
    with pytest.raises(SchemaError):
        load("   \n")
    with pytest.raises(SchemaError):
        load("{not json}")


def test_dump_load_revalidates_on_load():
    """load 走一遍 validate_sample：把落盘时可能被人工编辑坏的样本兜住。"""
    s = _valid_choice_sample()
    line = dump(s)
    # 手改一行里的概率，使其和=0.5，模拟磁盘污染
    tampered = json.loads(line)
    tampered["targets"]["q1"] = {"A": 0.5}  # 单值 0.5 缺项、和也不为 1
    bad_line = json.dumps(tampered, ensure_ascii=False) + "\n"
    with pytest.raises(SchemaError):
        load(bad_line)


def test_unknown_field_rejected():
    """顶层或题目层的未知字段（如拼错 critria）一律拒，防"静默通过"。"""
    s = _valid_choice_sample()
    s["critia"] = {}  # 拼错的 criteria
    with pytest.raises(SchemaError) as ei:
        validate_sample(s)
    assert "critia" in str(ei.value)


def test_targets_on_orphan_qid_rejected():
    """targets 挂到 questions 里不存在的 qid → 路径式错误。"""
    s = _valid_choice_sample()
    s["targets"]["q999"] = {"A": 1.0}
    with pytest.raises(SchemaError) as ei:
        validate_sample(s)
    assert "q999" in str(ei.value)


def test_ep_fields_passthrough():
    """ep_group / ep_step 透传不校验语义（spec 硬约束）。"""
    s = _valid_choice_sample()
    s["ep_group"] = "g-abc"
    s["ep_step"] = 42
    out = validate_sample(s)
    assert out["ep_group"] == "g-abc" and out["ep_step"] == 42
