"""p1-10 评测出口验收单测（唯一预测出口 + 四轴 + 溯源 + 一键复现契约）。

`-k` 过滤名即 tasks.md 条目：
  A1 `-k predict`        预测行三键契约、概率和为 1、--model/--endpoint 二选一、
                         endpoint 协议转换（mock HTTP，不依赖真服务）
  A2 `-k quality`        qtype×k 分桶：k=3 与 k=20 绝不分列合并、基线现取、Δ 符号按指标方向
  A3 `-k calibration`    ECE before/after/Δ 三列同屏、倍数出处、未调类如实标注
  A4 `-k parity` (-m mps) 内核路 vs 纯 torch 路 argmax 一致率 100%（真实 SFT 权重打靶）
  A5 `-k speed`          单请求串行、warm-up 剔除、P50/P95、设备/精度必注、吞吐只算输入
  B1 `-k provenance`     裸权重拒评、豁免标记、三处来源任一命中即认、拒评发生在载权重之前

真实打靶用例依赖前序域产物（runs/1004-s2-decision-sft-17ac + bench/p1-09），缺失即 skip
而不是伪造通过；合成用例全部离线自造，不下载任何数据集。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from sys1.calibrate import bucket_baselines
from sys1.decision import to_probs
from sys1.eval import calibration, parity, predict, quality, report, speed

REPO_ROOT = Path(__file__).resolve().parents[1]
SFT_MODEL_DIR = REPO_ROOT / "runs" / "1004-s2-decision-sft-17ac" / "model"
TYPED_DATA = REPO_ROOT / "bench" / "p1-09" / "typed_decisions" / "test.jsonl"
requires_artifacts = pytest.mark.skipif(
    not (SFT_MODEL_DIR.is_dir() and TYPED_DATA.is_file()),
    reason=f"缺前序域产物：{SFT_MODEL_DIR} 或 {TYPED_DATA}",
)


# ---------------------------------------------------------------- 桩件
def _record(idx: int = 0, qtype: str = "choice", options: tuple[str, ...] = ("晴", "雨", "阴")) -> dict:
    """一条最小合法样本（typed-decisions 的信封形状：{id, task, sample}）。"""
    criteria = {o: o for o in options}
    return {
        "id": f"synthetic-{qtype}-{idx:04d}",
        "task": qtype,
        "sample": {
            "state": f"报道{idx}：天空的状况记录。",
            "questions": {"q": {"type": qtype, "instructions": "这段话属于哪一类？", "criteria": criteria}},
            "targets": {"q": {options[0]: 1.0}},
        },
    }


def _pred(qid: str, qtype: str, options: list[str], scores: list[float], target: list[float], *,
          temperature: float = 1.0, uncalibrated: bool = True) -> predict.SamplePrediction:
    """造一条富样本：probs 由同一把尺子（to_probs）折出，避免测试里另定口径。"""
    t = torch.tensor(scores, dtype=torch.float64) / temperature
    probs = to_probs(t.unsqueeze(0), 1.0)[0].tolist()
    return predict.SamplePrediction(
        id=f"{qtype}-{''.join(options)}-{scores[0]:.2f}", qid=qid, qtype=qtype, k=len(options),
        options=options, scores=list(scores), probs=list(probs), probs_uncalibrated=list(
            to_probs(torch.tensor(scores, dtype=torch.float64).unsqueeze(0), 1.0)[0].tolist()),
        target=list(target), temperature=temperature, uncalibrated=uncalibrated, ms=1.0,
    )


class _FakePredictor:
    """只满足四轴所需接口的假出口：predict 按脚本返回耗时，describe 交设备标注。"""

    def __init__(self, ms_list: list[float], *, mode: str = "model") -> None:
        self.ms_list = ms_list
        self.mode = mode
        self.calls = 0
        self.temperatures = {"choice": 2.0}

    def describe(self) -> dict:
        return {"mode": self.mode, "device": "cpu", "dtype": "float32", "backend": "torch_eager",
                "temperatures": dict(self.temperatures),
                "provenance": {"run_id": "fake-run", "exempt": False, "commit": "deadbeef"}}

    def predict(self, records, *, batch_size: int = 1):   # 假出口不在乎捆大小，只按脚本交耗时
        idx = self.calls
        self.calls += 1
        ms = self.ms_list[min(idx, len(self.ms_list) - 1)]
        rec = records[0]
        opts = list(rec["sample"]["questions"]["q"]["criteria"])
        one = _pred("q", "choice", opts, [3.0, 1.0], [1.0, 0.0], temperature=1.0)
        one.id = rec["id"]
        one.ms = float(ms)
        return [one]


# ---------------------------------------------------------------- A1 predict
@pytest.fixture(scope="module")
def local_predictor():
    """真实 SFT 权重的预测出口（module 级复用：载一次权重够全组用例打靶）。"""
    if not (SFT_MODEL_DIR.is_dir() and TYPED_DATA.is_file()):
        pytest.skip(f"缺前序域产物：{SFT_MODEL_DIR} 或 {TYPED_DATA}")
    return predict.LocalPredictor(SFT_MODEL_DIR, device="cpu")


@requires_artifacts
def test_predict_local_row_contract(local_predictor):
    """--model 真实打靶：交出的每一行恰有 {id, answers, ms} 三键，且各候选份额和为 1。"""
    records = predict.load_records(TYPED_DATA)[:8]
    preds = local_predictor.predict(records, batch_size=4)
    assert len(preds) == len(records)
    rows = [p.to_row() for p in preds]
    for row, src in zip(rows, records):
        assert set(row) == set(predict.ROW_KEYS), f"预测行键不合规: {sorted(row)}"
        assert row["id"] == src["id"]
        assert row["ms"] > 0.0
        for answers in row["answers"].values():
            assert abs(sum(answers.values()) - 1.0) < 1e-4, "份额没归一"
            assert all(v >= 0.0 for v in answers.values())


@requires_artifacts
def test_predict_rows_roundtrip_and_contract_gate(local_predictor, tmp_path):
    """写盘再读回必须逐行等价；缺键文件要当场拒绝，不许"看着像预测文件"就往下算。"""
    preds = local_predictor.predict(predict.load_records(TYPED_DATA)[:4], batch_size=2)
    path = predict.write_rows(tmp_path / "rows.jsonl", preds)
    assert predict.read_rows(path) == [p.to_row() for p in preds]
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "x", "answers": {}}) + "\n", encoding="utf-8")
    with pytest.raises(KeyError, match="缺键"):
        predict.read_rows(bad)


def test_predict_requires_exactly_one_target():
    """--model 与 --endpoint 必须二选一，两个都给或都不给都拒绝启动。"""
    from argparse import Namespace

    with pytest.raises(SystemExit):
        predict.build_predictor(Namespace(model=None, endpoint=None, tokenizer=None,
                                          device="cpu", allow_missing_run_id=False))
    with pytest.raises(SystemExit):
        predict.build_predictor(Namespace(model="m", endpoint="http://x", tokenizer=None,
                                          device="cpu", allow_missing_run_id=False))


def test_predict_endpoint_protocol_with_mock():
    """endpoint 模式：请求体形状、回包按本地候选序对齐、ms 优先服务端——全用 mock HTTP 验。"""
    seen: list[tuple[str, dict]] = []

    def fake_post(url: str, payload: dict, timeout: float) -> dict:      # 三参签名 → 走超时传递
        seen.append((url, payload))
        assert timeout == 5.0
        return {"answers": {"q": {"晴": 0.7, "雨": 0.3}}, "ms": 12.5}    # 故意少答一个候选

    predictor = predict.EndpointPredictor("http://svc.local:8080", timeout=5.0, post=fake_post)
    records = [_record(0, "choice", ("晴", "雨", "阴"))]
    preds = predictor.predict(records)
    assert seen[0][0].endswith(predict.SYSTEMONE_PATH), "地址没补 /v1/systemone"
    sent = seen[0][1]
    assert set(sent) == {"id", "state", "questions"} and sent["id"] == records[0]["id"]
    p = preds[0]
    assert p.k == 3 and set(p.options) == {"晴", "雨", "阴"}
    assert abs(sum(p.probs) - 1.0) < 1e-6, "回包缺项也要归一成一份答案"
    assert p.ms == 12.5, "耗时优先取服务端报的那个数"
    assert p.uncalibrated and p.from_endpoint
    desc = predictor.describe()
    assert desc["mode"] == "endpoint" and desc["provenance"]["source"] == "endpoint"


# ---------------------------------------------------------------- A2 quality
def test_quality_keeps_k3_and_k20_apart():
    """分桶表必须把 k=3 与 k=20 各列一行、各配自己的随机基线，绝不合并成一个总分。"""
    preds = []
    for i in range(4):
        preds.append(_pred("q", "choice", ["a", "b", "c"], [3.0 - i, 1.0, 0.5], [1.0, 0.0, 0.0]))
    twenty = [f"o{j}" for j in range(20)]
    for i in range(3):
        scores = [0.0] * 20
        scores[5 if i else 0] = 2.0
        target = [0.0] * 20
        target[0] = 1.0
        preds.append(_pred("q", "choice", twenty, scores, target))

    rep = quality.score_quality(preds)
    buckets = {(row["qtype"], row["k"]): row for row in rep["buckets"]}
    assert set(buckets) == {("choice", 3), ("choice", 20)}, f"分桶键不合规: {sorted(buckets)}"
    base = bucket_baselines()["choice"]
    assert buckets[("choice", 3)]["baseline"] == pytest.approx(base[3].value, abs=1e-6)
    assert buckets[("choice", 20)]["baseline"] == pytest.approx(1.0 / 20.0, abs=1e-6)
    assert buckets[("choice", 3)]["n"] == 4 and buckets[("choice", 20)]["n"] == 3
    assert "合并" in rep["honest_note"]
    table = quality.format_table(rep)
    assert table.count("choice") == 2, "两张桶必须各占一行"


def test_quality_noul_and_score_bucket_baselines():
    """noul 基线恒 0.5；score 看平均偏差（越小越好），Δ 的符号随之翻转。"""
    noul = [_pred("q", "noul", ["是", "否"], [2.0, -2.0], [1.0, 0.0]),
            _pred("q", "noul", ["是", "否"], [-2.0, 2.0], [1.0, 0.0])]
    rep = quality.score_quality(noul)
    row = rep["buckets"][0]
    assert row["baseline"] == pytest.approx(bucket_baselines()["noul"][2].value, abs=1e-6)
    assert row["accuracy"] == pytest.approx(0.5)
    assert row["delta_pp"] == pytest.approx(0.0, abs=1e-6) and not row["beats_baseline"]

    digits = ["1", "2", "3", "4", "5"]
    score = [_pred("q", "score", digits, [0.0, 0.0, 5.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]),  # 命中
             _pred("q", "score", digits, [0.0, 5.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0])]  # 差一档
    rep2 = quality.score_quality(score)
    row2 = rep2["buckets"][0]
    assert row2["mae"] == pytest.approx(0.5) and row2["within_1"] == pytest.approx(1.0)
    assert row2["value_source"] == "numeric-code"
    assert row2["mae_delta_vs_baseline"] == pytest.approx(row2["baseline"] - 0.5)
    assert row2["beats_baseline"] is True


def test_quality_rejects_mixed_value_source_in_one_bucket():
    """同一 score 桶里"数字档"与"位次档"混排必须拒绝——两口径不许进同一个平均。"""
    mixed = [_pred("q", "score", ["1", "2", "3"], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
             _pred("q", "score", ["低", "中", "高"], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0])]
    with pytest.raises(ValueError, match="口径不唯一"):
        quality.score_quality(mixed)


# ---------------------------------------------------------------- A3 calibration
def test_calibration_reports_before_after_and_delta():
    """三列同屏：过度自信的原始分数 ECE 高，按倍数折软之后必须降下来（Δ 为负）。"""
    preds = [_pred("q", "choice", ["a", "b", "c"], [9.0, 1.0, 0.5], [0.5, 0.3, 0.2]),     # 答对但过自信
             _pred("q", "choice", ["a", "b", "c"], [9.0, 8.0, 0.5], [0.0, 1.0, 0.0])]     # 答错且过自信
    for p in preds:
        p.uncalibrated = False
        p.temperature = 1.0
    rep = calibration.score_calibration(preds, table={"choice": 8.0})
    row = rep["by_type"][0]
    assert row["ece_before"] > row["ece_after"], "折软之后差距必须收窄"
    assert row["ece_delta"] == pytest.approx(row["ece_after"] - row["ece_before"], abs=1e-6)
    assert row["improved"] is True and row["temperature"] == pytest.approx(8.0)
    assert row["temperature_source"] == "caller-table"      # 由调用方给的表，不是模型自带
    assert row["mode"] == calibration.EVAL_MODE and row["bins"] == 15
    assert rep["overall"]["scope"] == "all-rows-pooled"
    assert "top-label" in calibration.format_table(rep)


def test_calibration_marks_uncalibrated_and_empty():
    """没倍数可查的题型要如实标未调；一条题都没有时出空表而不是编一个 0。"""
    preds = [_pred("q", "noul", ["是", "否"], [2.0, -2.0], [1.0, 0.0], temperature=1.0, uncalibrated=True)]
    rep = calibration.score_calibration(preds, table={"choice": 2.0})
    row = rep["by_type"][0]
    assert row["uncalibrated"] is True and row["temperature_source"] == "default(1.0)"

    empty = calibration.score_calibration([])
    assert empty["empty"] is True and empty["by_type"] == []
    assert "空表" in calibration.format_table(empty)


def test_calibration_does_not_rewrite_ece_formula():
    """本域不重写公式：与直接调用 sys1.calibrate.ECE（同一份把握行）逐值一致。"""
    from sys1.calibrate import ECE

    preds = [_pred("q", "choice", ["a", "b"], [4.0, 1.0], [1.0, 0.0]) for _ in range(3)]
    preds[0].scores = [1.0, 4.0]
    rows = [to_probs(torch.tensor(x.scores, dtype=torch.float64).unsqueeze(0), 1.0)[0].tolist() for x in preds]
    manual = float(ECE(rows, [x.target for x in preds], 15, weighted=True))
    rep = calibration.score_calibration(preds, table={"choice": 1.0})
    assert rep["by_type"][0]["ece_before"] == pytest.approx(round(manual, 6), abs=1e-6)


# ---------------------------------------------------------------- A4 parity
@requires_artifacts
@pytest.mark.mps
def test_parity_argmax_agreement_on_mps(local_predictor):
    """内核路 vs 纯 torch 路在 MPS 上必须 100% 名次一致，且报告带着"没走回退"的凭据。"""
    if not torch.backends.mps.is_available():
        pytest.skip("本机没有 MPS")
    records = predict.load_records(TYPED_DATA)[:8]
    encs = local_predictor.encode(records)
    rep = parity.run_parity(local_predictor.model, encs, device="mps",
                           head_weight=local_predictor.head_weight_cpu)
    assert rep["total"] > 0
    assert rep["passed"] is True and rep["argmax_agreement"] == pytest.approx(1.0)
    assert rep["device"] == "mps" and rep["backend"] in ("tilelang", "torch_eager")
    if rep["backend"] == "tilelang":
        assert rep["dialect_compiles_this_run"] > 0 and rep["compiled_keys"], "方言通过要有编译凭据"
    assert rep["max_abs_err_scores"] < rep["fp16_tolerance"]
    assert "gelu" in rep["external_ops"], "模具没覆盖的算子必须如实列出"
    assert "parity" in parity.format_table(rep)


@requires_artifacts
def test_parity_cpu_fallback_is_labelled(local_predictor):
    """CPU 上入口层回退到普通写法：一致率照算，但 backend 必须写 torch_eager，不许冒充方言。"""
    records = predict.load_records(TYPED_DATA)[:4]
    encs = local_predictor.encode(records)
    rep = parity.run_parity(local_predictor.model, encs, device="cpu",
                           head_weight=local_predictor.head_weight_cpu)
    assert rep["argmax_agreement"] == pytest.approx(1.0) and rep["passed"] is True
    assert {"backend", "device", "external_ops", "blockers", "compiled_keys"} <= set(rep)
    assert rep["backend"] == "torch_eager", "CPU 上没有 Metal 方言，必须如实标注为回退路径"


# ---------------------------------------------------------------- A5 speed
def test_speed_excludes_warmup_and_reports_percentiles():
    """warm-up 条数不进分布；P50/P95 由实测序列算出，设备与精度必须写在报告头。"""
    ms = [50.0, 40.0, 10.0, 20.0, 30.0, 60.0]
    predictor = _FakePredictor(ms)
    records = [_record(i) for i in range(len(ms))]
    rep = speed.measure_speed(predictor, records, warmup=2)
    assert predictor.calls == len(ms)
    assert rep["n"] == 4 and rep["warmup_excluded"] == 2
    tail = sorted(ms[2:])                                  # 去掉两条热身后的读数分布
    assert rep["readout_ms"]["p50"] == pytest.approx((tail[1] + tail[2]) / 2.0)
    assert rep["readout_ms"]["p95"] <= tail[-1] and rep["readout_ms"]["p95"] >= tail[-2]
    assert rep["e2e_ms"]["p95"] >= rep["e2e_ms"]["p50"] > 0
    assert rep["device"] == "cpu" and rep["dtype"] == "float32" and rep["backend"] == "torch_eager"
    assert rep["serial"] is True and rep["output_tokens"] == 0
    assert "P50" in speed.format_table(rep)

def test_speed_flags_small_sample_and_empty_input():
    """不足 N=30 要显式标注"只当参考"；没有样本时不出数也不猜默认值。"""
    predictor = _FakePredictor([5.0])
    rep = speed.measure_speed(predictor, [_record(0), _record(1)], warmup=0)
    assert rep["meets_min_samples"] is False and "参考" in rep["honest_note"]
    empty = speed.measure_speed(predictor, [])
    assert empty.get("empty") is True
    assert "空" in speed.format_table(empty)


# ---------------------------------------------------------------- B1 provenance
def test_provenance_rejects_bare_weights(tmp_path):
    """裸权重（config.json 里没有任何来源字段）必须当场拒评，错误消息还要交代怎么补。"""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"d": 8, "L": 2}), encoding="utf-8")
    with pytest.raises(predict.MissingProvenanceError, match="run_id"):
        predict.resolve_provenance(model)
    # 拒评发生在载权重之前：构造 LocalPredictor 也同样炸，不会先把权重读进内存
    with pytest.raises(predict.MissingProvenanceError):
        predict.LocalPredictor(model)


def test_provenance_cli_refuses_with_exit_code_2(tmp_path, capsys):
    """命令行口径的拒评：一句人话提示 + 退出码 2，绝不把 traceback 当成"拒绝"。"""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"d": 8, "L": 2}), encoding="utf-8")
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps(_record(0), ensure_ascii=False) + "\n", encoding="utf-8")

    assert predict.main(["--model", str(model), "--data", str(data), "--device", "cpu"]) == 2
    assert "拒绝出数" in capsys.readouterr().err
    assert report.main(["--model", str(model), "--data", str(data), "--device", "cpu",
                        "--axes", "quality", "--no-record"]) == 2
    assert "拒绝出数" in capsys.readouterr().err


def test_provenance_exempt_smoke_keeps_marker(tmp_path):
    """--allow-missing-run-id 放行时必须带着显式豁免标记，报告里据此判这次不算通过。"""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"d": 8}), encoding="utf-8")
    prov = predict.resolve_provenance(model, allow_missing=True)
    assert prov.run_id == predict.EXEMPT_RUN_ID and prov.exempt is True
    assert prov.as_dict()["exempt"] is True


@pytest.mark.parametrize("carrier", ["config.json", "decision_config.json", "ledger"])
def test_provenance_accepts_any_of_the_three_carriers(tmp_path, carrier):
    """三处来源（模型设置单 / 决策说明书 / 上一层 run 账本）任一命中即算有溯源。"""
    run_dir = tmp_path / "1004-s2-fake"
    model = run_dir / "model"
    model.mkdir(parents=True)
    if carrier == "config.json":
        (model / "config.json").write_text(json.dumps({"d": 8, "run_id": "run-from-config"}), encoding="utf-8")
    elif carrier == "decision_config.json":
        (model / "config.json").write_text(json.dumps({"d": 8}), encoding="utf-8")
        (model / "decision_config.json").write_text(
            json.dumps({"provenance": {"run_id": "run-from-decision"},
                        "calibration": {"run_id": "run-cal", "status": "fitted"}}), encoding="utf-8")
    else:
        (model / "config.json").write_text(json.dumps({"d": 8}), encoding="utf-8")
        (run_dir / "config.yaml").write_text("run_id: run-from-ledger\ncommit: abc123\n", encoding="utf-8")

    prov = predict.resolve_provenance(model)
    assert prov.exempt is False
    assert prov.run_id.startswith("run-from")
    if carrier == "decision_config.json":
        assert prov.chain["calibration_run_id"] == "run-cal"
    if carrier == "ledger":
        assert prov.commit == "abc123"


@requires_artifacts
def test_provenance_on_real_sft_run(local_predictor):
    """真实 SFT 产物必须被认出来，并顺出"s2 ← s3 校准"的来源链。"""
    prov = local_predictor.provenance
    assert prov.run_id == "1004-s2-decision-sft-17ac" and prov.exempt is False
    assert prov.chain.get("calibration_run_id") == "1004-s3-calibrate-a3a2"
    assert local_predictor.temperatures, "decision_config 里的倍数表要跟着一起带出来"


# ---------------------------------------------------------------- 汇总出口（B2 依赖）
def test_report_verdict_requires_provenance(local_predictor, tmp_path):
    """豁免来的数不算通过：溯源 exempt 时 verdict 必判负（退出码非零的依据）。"""
    from sys1.eval import report

    result = report.run_eval(local_predictor, predict.load_records(TYPED_DATA)[:6],
                             device="cpu", speed_samples=6, parity_samples=4,
                             axes=("quality", "calibration", "speed"))
    assert result["verdict"]["passed"] is True
    assert result["axes"]["quality"]["buckets"] and result["axes"]["speed"]["n"] > 0

    # 真断言：裸目录 + 豁免标记重跑一遍，verdict 必须判负（而非仅检查既有数据在场）。
    # 裸目录 = 复制真 SFT 产物后抹净溯源键（上层 run 账本因搬到 tmp 天然断开）。
    import shutil
    bare = tmp_path / "bare_model"
    shutil.copytree(SFT_MODEL_DIR, bare)
    for name in ("config.json", "decision_config.json"):
        p = bare / name
        if not p.is_file():
            continue
        obj = json.loads(p.read_text("utf-8"))
        for k in predict.PROVENANCE_KEYS:
            obj.pop(k, None)
        cal = obj.get("calibration")
        if isinstance(cal, dict):
            for k in predict.PROVENANCE_KEYS:
                cal.pop(k, None)
        p.write_text(json.dumps(obj), encoding="utf-8")
    exemptor = predict.LocalPredictor(bare, device="cpu", allow_missing_run_id=True)
    r2 = report.run_eval(exemptor, predict.load_records(TYPED_DATA)[:6],
                         device="cpu", speed_samples=6, parity_samples=4,
                         axes=("quality", "calibration", "speed"))
    assert r2["verdict"]["passed"] is False, "豁免溯源的评测不得判通过"
    assert r2["verdict"]["exempt"] is True and r2["verdict"]["provenance_ok"] is False


def test_report_table_shape_matches_bucket_baselines(local_predictor):
    """汇总里的质量表与单独跑 quality 一致（同一份分母，四轴不许各自取数）。"""
    from sys1.eval import report

    records = predict.load_records(TYPED_DATA)[:6]
    result = report.run_eval(local_predictor, records, device="cpu", speed_samples=6,
                             parity_samples=0, axes=("quality",))
    direct = quality.score_quality(local_predictor.predict(records, batch_size=4))
    assert result["axes"]["quality"]["buckets"] == direct["buckets"]
    assert math.isclose(result["n_predictions"], len(records))
