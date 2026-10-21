"""p1-09 工作项 A（样本转写）单测：`-k rules` 锁定映射规则、其余覆盖三型转写。

覆盖 tasks.md：
  A1 XNLI→noul / 分类→choice / 星级→score 三条转写路径，产物均过样本契约。
  A2 转写规则锁定（neutral→{"false":1.0}、contradiction→{"false":1.0}、entailment→{"true":1.0}）。
"""
from __future__ import annotations

import pytest

from sys1.data.schema import validate_sample
from sys1.data.transcribe import (
    TRANSCRIBE_VERSION,
    XNLI_TO_NOUL,
    TranscribeError,
    classification_to_choice,
    rating_to_score,
    transcribe,
    xnli_to_noul,
)


# ---------------------------------------------------------------- A1：三型转写
def test_xnli_neutral_becomes_false():
    rec = {"id": "xnli-1", "premise": "他站在门口。", "hypothesis": "有人在屋里睡觉。", "label": "neutral"}
    out = xnli_to_noul(rec)
    assert out["id"] == "xnli-1" and out["task"] == "noul"
    assert out["sample"]["targets"]["q"] == {"false": 1.0}


def test_xnli_state_premise_and_question_aligned():
    rec = {"id": "x", "premise": "猫在垫子上。", "hypothesis": "有动物在垫子上。", "label": "entailment"}
    out = xnli_to_noul(rec)
    sample = out["sample"]
    assert sample["state"] == "猫在垫子上。"            # state 对齐 premise
    assert "有动物在垫子上。" in sample["questions"]["q"]["instructions"]
    assert sample["questions"]["q"]["type"] == "noul"


def test_choice_one_hot_on_label():
    rec = {"id": "c1", "text": "帮我订一张明天的票", "options": ["订票", "天气", "翻译"], "label": "订票"}
    out = classification_to_choice(rec)
    targets = out["sample"]["targets"]["q"]
    assert targets == {"订票": 1.0}
    assert out["sample"]["questions"]["q"]["type"] == "choice"


def test_choice_dict_options_preserved():
    rec = {"id": "c2", "text": "t", "options": {"A": "晴", "B": "雨"}, "label": "B"}
    out = classification_to_choice(rec)
    assert out["sample"]["questions"]["q"]["criteria"] == {"A": "晴", "B": "雨"}
    assert out["sample"]["targets"]["q"] == {"B": 1.0}


def test_score_star_level_full_support():
    rec = {"id": "s1", "text": "很好用的产品", "rating": 3, "scale": 5}
    out = rating_to_score(rec)
    sample = out["sample"]
    assert sample["questions"]["q"]["type"] == "score"
    # score 契约要求 0..k-1 全档齐活：实得档（3 星→下标 2）押 1，其余写 0
    assert sample["targets"]["q"] == {"0": 0.0, "1": 0.0, "2": 1.0, "3": 0.0, "4": 0.0}
    assert set(sample["questions"]["q"]["criteria"]) == {"0", "1", "2", "3", "4"}


# ---------------------------------------------------------------- A2：规则锁定
@pytest.mark.parametrize(
    ("label", "verdict"),
    [("entailment", "true"), ("neutral", "false"), ("contradiction", "false")],
)
def test_rules_xnli_map(label, verdict):
    assert XNLI_TO_NOUL[label] == verdict
    out = xnli_to_noul({"id": f"r-{label}", "premise": "p", "hypothesis": "h", "label": label})
    assert out["sample"]["targets"]["q"] == {verdict: 1.0}


def test_rules_xnli_map_is_exactly_three_keys():
    assert set(XNLI_TO_NOUL) == {"entailment", "neutral", "contradiction"}


def test_rules_label_case_and_space_insensitive():
    for raw in ("Entailment", " entailment ", "ENTAILMENT"):
        out = xnli_to_noul({"id": f"cc-{raw}", "premise": "p", "hypothesis": "h", "label": raw})
        assert out["sample"]["targets"]["q"] == {"true": 1.0}


def test_rules_star_min_is_one():
    # 1 星（下限）→ 档位下标 0，其余档写 0
    out = rating_to_score({"id": "r-low", "text": "糟", "rating": 1, "scale": 3})
    assert out["sample"]["targets"]["q"] == {"0": 1.0, "1": 0.0, "2": 0.0}


# ---------------------------------------------------------------- 契约接入与鲁棒性
@pytest.mark.parametrize(
    "factory",
    [
        lambda: xnli_to_noul({"id": "1", "premise": "p", "hypothesis": "h", "label": "neutral"}),
        lambda: classification_to_choice({"id": "2", "text": "t", "options": ["a", "b"], "label": "a"}),
        lambda: rating_to_score({"id": "3", "text": "t", "rating": 2, "scale": 3}),
    ],
)
def test_output_passes_schema(factory):
    out = factory()
    validate_sample(out["sample"])                       # 转写产物必须已在契约内


def test_transcribe_dispatch_by_task_field():
    out = transcribe({"task": "score", "id": "d", "text": "t", "rating": 5, "scale": 5})
    assert out["task"] == "score"
    assert out["sample"]["targets"]["q"] == {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 1.0}


def test_transcribe_infers_task_from_fields():
    out = transcribe({"id": "i", "premise": "p", "hypothesis": "h", "label": "entailment"})
    assert out["task"] == "noul" and out["sample"]["targets"]["q"] == {"true": 1.0}


def test_unknown_xnli_label_rejected():
    with pytest.raises(TranscribeError):
        xnli_to_noul({"id": "u", "premise": "p", "hypothesis": "h", "label": "maybe"})


def test_choice_label_not_in_options_rejected():
    with pytest.raises(TranscribeError):
        classification_to_choice({"id": "u", "text": "t", "options": ["a", "b"], "label": "c"})


def test_rating_out_of_range_rejected():
    with pytest.raises(TranscribeError):
        rating_to_score({"id": "u", "text": "t", "rating": 6, "scale": 5})


def test_missing_id_rejected():
    with pytest.raises(TranscribeError):
        xnli_to_noul({"premise": "p", "hypothesis": "h", "label": "neutral"})


def test_version_constant_present():
    assert TRANSCRIBE_VERSION == "dmlaya_transcribe_v1"


def test_schema_error_surfaces_as_valueerror():
    # 转写把校验失败原样交给契约：SchemaError 仍是 ValueError 子类，调用方可同族兜底
    with pytest.raises(ValueError):
        transcribe({"id": "bad", "task": "choice", "text": "t", "options": ["only"], "label": "only"})
