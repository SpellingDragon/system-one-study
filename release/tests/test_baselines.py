"""双基线亲跑与对照表的单元测试（p2-04 baselines-dual，五场景）。

这些测试一律不下载任何模型、不碰 MPS/GPU：用假 runner、假 laya 应答、monkeypatch 载入与
选题，把"读数折份额 / 两条线各出一行 / 权重不可得时降级 / 对照表六字段且 DML 留空"这几件
事钉死。目的只有一个——管线与表结构能脱机自证，绿灯不是靠真模型跑出来的侥幸。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sys1.eval.baselines import run_baseline as rb
from sys1.eval.baselines import table as tb
from sys1.eval.predict import SamplePrediction


# ---------------------------------------------------------------- 造数据的小工具
def _fake_records(n: int = 6) -> list[dict]:
    """造几条只带题型的假记录，够 pick_records 分堆用（不涉渲染，脱机可跑）。"""
    recs = []
    for i in range(n):
        qtype = "choice" if i % 2 == 0 else "noul"
        recs.append({"id": f"r{i}", "sample": {"state": "s", "questions": {f"q{i}": {"type": qtype}}, "targets": {}}})
    return recs


def _mk_pred(pid: str, *, qtype: str = "choice", k: int = 3, hit: bool = True) -> SamplePrediction:
    """造一个字段齐全的富样本：默认让首选即标准答案（hit=True）以便断言准确率。"""
    opts = ["A", "B", "C", "D"][:k]
    probs = [0.8] + [0.2 / (k - 1)] * (k - 1)
    if not hit:
        probs = [0.2] + [0.8] + [0.0] * (k - 2)
        probs = probs[:k]
        s = sum(probs)
        probs = [p / s for p in probs]
    target = [1.0] + [0.0] * (k - 1)
    return SamplePrediction(
        id=pid, qid="q", qtype=qtype, k=k, options=opts, scores=probs, probs=list(probs),
        probs_uncalibrated=list(probs), target=target, temperature=1.0, uncalibrated=False, ms=120.0,
    )


# ---------------------------------------------------------------- 场景：读数折份额（可单测的核心算子）
def test_fold_letter_scores_sums_and_keeps_top():
    """同除一个正温度不改谁排第一，但折出来的份额要各自摊成一锅汤（和为 1）。"""
    probs_cal, probs_raw = rb.fold_letter_scores([3.0, 1.0, 0.0], 2.0)
    assert len(probs_cal) == 3 and len(probs_raw) == 3
    assert pytest.approx(sum(probs_cal), abs=1e-6) == 1.0
    assert pytest.approx(sum(probs_raw), abs=1e-6) == 1.0
    assert probs_cal.index(max(probs_cal)) == 0  # 首选不变（同除不改序）


# ---------------------------------------------------------------- 场景：选题 split 两类到场且确定
def test_pick_records_both_types_present_and_deterministic():
    """取小样本时 choice 与 noul 都得在场，且同样的输入两次取到同一串（两基线才可比）。"""
    picked = rb.pick_records(_fake_records(6), 4)
    types = {next(iter(r["sample"]["questions"].values()))["type"] for r in picked}
    assert {"choice", "noul"} <= types
    again = rb.pick_records(_fake_records(6), 4)
    assert [r["id"] for r in picked] == [r["id"] for r in again]


# ---------------------------------------------------------------- 场景：laya 应答按字母序对齐
def test_laya_prob_map_aligns_to_order():
    """laya 报回的候选顺序未必合我方字母序，这里按 order 逐格讨回来（判断题补足 false）。"""
    runner = rb.LayaRunner(agent=None)
    noul = runner._prob_map({"type": "noul", "noul": 0.7}, ["false", "true"])
    assert pytest.approx(noul, abs=1e-6) == [0.3, 0.7]
    choice = runner._prob_map(
        {"type": "choice", "probabilities": {"C": 0.5, "A": 0.2, "B": 0.3}}, ["A", "B", "C"],
    )
    assert pytest.approx(choice, abs=1e-6) == [0.2, 0.3, 0.5]


# ---------------------------------------------------------------- 假 runner（脱机替身）
class _FakeRunner:
    """顶替 StartLux/Laya 的假引擎：predict 直接回一批现成富样本，绝不做真前向。"""

    def __init__(self, preds: list[SamplePrediction]) -> None:
        self._preds = preds
        self.temp_table = {"choice": 1.75, "noul": 3.29}
        self.describe_meta = {"backend": "fake", "temperature_by_type": self.temp_table}

    def predict(self, records, *, batch_size: int = 4):  # noqa: ARG002 records 用不上（假数据）
        return list(self._preds)


# ---------------------------------------------------------------- 场景一：laya 对照行（真跑通那条形）
def test_run_laya_produces_row_with_provenance(tmp_path, monkeypatch):
    """laya 载得到就出亲跑行：source=harness、gate=true，带 run-id/采样参数/commit 与指标。"""
    preds = [_mk_pred("A1"), _mk_pred("A2", qtype="noul", k=2), _mk_pred("A3", hit=False)]
    monkeypatch.setattr(rb, "load_records", lambda path: _fake_records(6))
    monkeypatch.setattr(rb.LayaRunner, "load", classmethod(lambda cls, device="cpu", alias=None: _FakeRunner(preds)))
    row = rb.run("laya", data=tmp_path / "test.jsonl", limit=6, device="cpu", split="s6",
                 out_root=tmp_path, startlux_dir=None, allow_degrade=False)
    assert row["source"] == "harness" and row["gate"] is True
    assert row["sampling_params"] == rb.SAMPLING_PARAMS and row["commit"] is not None
    assert "p2-04" in row["run_id"]
    assert row["n_samples"] == len(preds)
    assert (tmp_path / "laya" / "s6.preds.jsonl").is_file()
    assert (tmp_path / "laya" / "s6.row.json").is_file()


# ---------------------------------------------------------------- 场景二：StartLux 对照行（同 harness 读数）
def test_run_startlux_produces_row_with_temperature(tmp_path, monkeypatch):
    """StartLux 载得到就出亲跑行，且把自带 decision_config 的倍数读进来（温度直读）。"""
    preds = [_mk_pred("S1"), _mk_pred("S2", qtype="noul", k=2)]
    monkeypatch.setattr(rb, "load_records", lambda path: _fake_records(6))
    monkeypatch.setattr(
        rb.StartLuxRunner, "load",
        classmethod(lambda cls, model_dir=None, device="cpu": _FakeRunner(preds)),
    )
    row = rb.run("startlux", data=tmp_path / "test.jsonl", limit=6, device="cpu", split="s6",
                 out_root=tmp_path, startlux_dir=tmp_path / "sl", allow_degrade=False)
    assert row["source"] == "harness" and row["gate"] is True
    assert "temperature_by_type" in row["note"]  # 倍数表被如实记进出处
    assert row["metrics"]["acc"] is not None


# ---------------------------------------------------------------- 场景三：权重不可得 → 降级不冒充实测
def test_run_degrade_marks_card_and_no_metrics(tmp_path, monkeypatch):
    """引擎载不动又允许降级：该行 source=card、gate=false，指标全空，notes 仍留结论。"""
    def _boom(cls, device="cpu", alias=None):  # noqa: ANN001, ARG001
        raise RuntimeError("权重不可得（测试注入）")

    monkeypatch.setattr(rb, "load_records", lambda path: [])
    monkeypatch.setattr(rb.LayaRunner, "load", classmethod(_boom))
    row = rb.run("laya", data=tmp_path / "test.jsonl", limit=6, device="cpu", split="s6",
                 out_root=tmp_path, startlux_dir=None, allow_degrade=True)
    assert row["source"] == "card" and row["gate"] is False
    assert row["metrics"]["acc"] is None and row["n_samples"] == 0
    notes = list((tmp_path / "runs").rglob("notes.md"))
    jie = chr(0x7ed3) + chr(0x8bba) + chr(0xff1a)  # 结论：
    assert notes and jie in notes[0].read_text(encoding="utf-8")


def test_run_no_degrade_reraises(tmp_path, monkeypatch):
    """不允许降级时，载入失败必须原样抛出，绝不悄悄写一行假数据糊弄过去。"""
    def _boom(cls, device="cpu", alias=None):  # noqa: ANN001, ARG001
        raise RuntimeError("权重不可得（测试注入）")

    monkeypatch.setattr(rb, "load_records", lambda path: [])
    monkeypatch.setattr(rb.LayaRunner, "load", classmethod(_boom))
    with pytest.raises(RuntimeError):
        rb.run("laya", data=tmp_path / "test.jsonl", limit=6, device="cpu", split="s6",
               out_root=tmp_path, startlux_dir=None, allow_degrade=False)


# ---------------------------------------------------------------- 场景四：对照表六字段 + DML 留空 + 全量占位
def _write_row(root: Path, baseline: str, split: str, **over) -> None:
    """往表生成器要读的位置落一张 row.json，方便脱机拼表。"""
    metrics = {"acc": 0.5833, "ece_after": 0.079, "ece_before": 0.062, "ms_p50": 250.7,
               "ms_mean": 300.0, "ece": 0.079, "throughput_qps": 3.06}
    metrics.update(over.pop("metrics", {}))
    row = rb.build_row(baseline=baseline, dataset="test", split=split, n=24, device="cpu",
                       sampling=rb.SAMPLING_PARAMS, metrics=metrics, run_id=f"x-{baseline}",
                       commit="c" * 40, source="harness", gate=True, note="测试行")
    row.update(over)
    (root / baseline).mkdir(parents=True, exist_ok=True)
    (root / baseline / f"{split}.row.json").write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")


def test_compare_table_six_fields_dml_placeholder(tmp_path):
    """对照表每指标行恰六字段，DML 未跑处一律短横，全量档显式标待跑并留 D11 脚注。"""
    _write_row(tmp_path, "laya", "smoke24")
    _write_row(tmp_path, "startlux", "smoke24",
               metrics={"acc": 0.5833, "ece_after": 0.0616, "ece_before": 0.0956,
                        "ms_p50": 5490.9, "ms_mean": 5400.0, "ece": 0.0616, "throughput_qps": 0.184})
    md = tb.build_compare_md(tmp_path, "smoke24", "full")
    body = [ln for ln in md.splitlines() if ln.startswith("| ") and "---" not in ln]
    data_rows = [ln for ln in body if ln.count("|") == 7]  # 六字段 → 7 根竖线
    assert data_rows, "应存在六字段指标行"
    for ln in md.splitlines():
        if ln.startswith("| 准确率") or ln.startswith("| ECE") or ln.startswith("| 逐题") or ln.startswith("| 吞吐"):
            cells = [c.strip() for c in ln.split("|")][1:-1]
            assert len(cells) == 6, f"每行须六字段：{cells}"
            assert cells[1] in ("—", "待 full"), f"DML 格须显式占位：{cells[1]}"
    assert "待 full" in md
    jie = chr(0x7ed3) + chr(0x8bba)
    assert ("D11" in md) and (chr(0x4ec5) in md)  # 含 D11 与"仅"字，边界写明


def test_compare_table_missing_baseline_is_placeholder(tmp_path):
    """只有一条基线落了盘，另一条那一格也须显式短横，不能留空或臆造。"""
    _write_row(tmp_path, "laya", "smoke24")
    md = tb.build_compare_md(tmp_path, "smoke24", "full")
    meta = [ln for ln in md.splitlines() if ln.startswith("| startlux ")]
    assert meta and "未跑" in meta[0]
