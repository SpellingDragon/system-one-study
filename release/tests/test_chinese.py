"""中文轨（p2-09）单测：转写规则锁定 / registry pin 收口 / 底账可溯源 / 配比两档 / G3 骨架。

分层说明（与 tests/test_registry.py 同口径，别把"没数据"写成"通过"）：
  **结构层**只查代码与登记这本"应然"（映射常量、题型与分账通道、pin 字段、CLI 按钮、配比档），
  任何时候都必须绿；里面出现的手写行是**单元构造**（给一条最小题面看折得对不对），不冒充实测。
  **事实层**查盘上真装配出来的东西（条数、逐题型、真值覆盖、sha256、按集分账），副本不在就
  skip 并给出可执行的装配命令。数字一律从 `manifest.json` 与 `assembled/*.jsonl` 现场取，
  不写死在本文件里——写死的数字会随着复跑变成假账。
  另有一类"源头对拍"用例：拿 `raw/clue-subset/*/validation-*.parquet` 里那份 ClassLabel names
  跟本模块常量逐位比。真值口径（哪个下标算 entailment、十五个类别码是什么）只能来自源头，
  上游改版即红；这条比"我记得 CLUE 长什么样"可靠。

运行方式（务必在 release/ 下用 -m）：
    cd release && .venv/bin/python -m pytest tests/test_chinese.py -q
    ... -k transcribe          # A1：决策化转写 + registry pin（孙任务判据）
    ... -k g3                  # B2：G3 骨架与失败样本挂列（出分待 C5）
    ... -k mix                 # B1：配比两档参数化（消融本体待 C5）
事实层前置：`.venv/bin/python -m sys1.eval.registry fetch --zh`
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from sys1.eval import chinese as zh
from sys1.eval import registry
from sys1.eval import run as evalrun
from sys1.eval import scoring

ASM = registry.DATA_DIR / registry.ASSEMBLED_DIR_NAME
RAW = registry.DATA_DIR / registry.RAW_DIR_NAME
CONFIGS = registry.REPO_ROOT / "production" / "configs"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _digest(rows: list[dict[str, Any]]) -> str:
    """一摞信封的契约字段（id/task/qtype/sample）摊成摘要——比散列文件更严：只认题面与真值。

    落盘那行还带 registry 记账时补的 split/溯源字段，直接比文件散列会把"记账字段"和"转写口径"
    混在一起谈；这里只摊四枚契约键，所以复跑与盘上不一致时点名的是转写规则本身。
    """
    canon = [{k: r[k] for k in ("id", "task", "qtype", "sample")} for r in rows]
    blob = json.dumps(canon, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _need_decision_sets() -> None:
    """中文决策信封还没装配就 skip，并给出装配命令（绝不拿空表算出 0 分当成实测）。"""
    man = registry.load_manifest()
    sets = man.get("sets") or {}
    missing = [pid for pid in zh.G3_SETS
               if not ((sets.get(pid) or {}).get("envelope_ready"))]
    if missing:
        pytest.skip(f"中文决策信封未装配：{missing}（先跑 `python -m sys1.eval.registry fetch --zh`）")


# ─────────────────── 结构层：三型转写规则（单元构造，不冒充实测） ───────────────────
def test_transcribe_cmmlu_becomes_choice_k4_with_option_text():
    """CMMLU 四选一 → choice(k=4)：criteria 是选项文字，真值押在 answer 那个字母上，问法换对味。"""
    env = zh.cmmlu_to_envelope({
        "id": "cmmlu:college_medicine:test/1", "question": "下列哪项为正确?",
        "options": {"A": "甲", "B": "乙", "C": "丙", "D": "丁"}, "answer": "B",
        "subject": "college_medicine"})
    q = env["sample"]["questions"][zh.QID]
    assert env["qtype"] == "choice" and env["task"] == zh.ZH_CHANNEL["cmmlu"]
    assert list(q["criteria"]) == list(zh.CMMLU_OPTION_KEYS), "候选代号必须就是 A/B/C/D（渲染层按它出字母）"
    assert q["criteria"]["B"] == "乙" and q["instructions"] == zh.CMMLU_INSTRUCTIONS
    assert env["sample"]["targets"][zh.QID] == {"B": 1.0}
    assert env["id"] == "cmmlu:college_medicine:test/1" and env["zh_source"] == "cmmlu-subset"
    assert env["sample"]["state"].startswith("下列哪项")


def test_transcribe_tnews_becomes_choice_with_source_label_codes():
    """tnews → choice(k=15)：候选文字用源头给的类别码本身，真值 = 下标映出的那枚码。"""
    env = zh.tnews_to_envelope({"id": "clue:tnews:validation:7", "sentence": "某球队赢了",
                                 "label": 8, "gold": "8", "task_name": "tnews",
                                 "label_space": [str(i) for i in range(15)]})
    q = env["sample"]["questions"][zh.QID]
    assert env["qtype"] == "choice" and env["task"] == zh.ZH_CHANNEL["tnews"]
    assert list(q["criteria"]) == list(zh.TNEWS_LABEL_CODES)
    assert env["label_code"] == zh.TNEWS_LABEL_CODES[8] == "109"
    assert q["criteria"]["109"] == "109", "沿用码本身当候选文字（源头没给中文名，不凭记忆补）"
    assert env["sample"]["targets"][zh.QID] == {"109": 1.0}


@pytest.mark.parametrize("index,verdict", [(1, "true"), (0, "false"), (2, "false")])
def test_transcribe_ocnli_becomes_noul_only_entailment_true(index: int, verdict: str):
    """ocnli → noul：只有 entailment（源头顺序的第 1 号）折成 true，中立与矛盾都记 false。"""
    env = zh.ocnli_to_envelope({"id": "clue:ocnli:validation:3", "sentence1": "他在家",
                                "sentence2": "有人在屋里", "label": index, "task_name": "ocnli",
                                "label_space": ["0", "1", "2"]})
    q = env["sample"]["questions"][zh.QID]
    assert env["qtype"] == "noul" and env["task"] == zh.ZH_CHANNEL["ocnli"]
    assert q["criteria"] == zh.NOUL_CRITERIA, "noul 必须带 false/true 两格说明，否则打分层拆出 k=0 假题"
    assert env["relation"] == zh.OCNLI_RELATIONS[index]
    assert env["sample"]["targets"][zh.QID] == {verdict: 1.0}


def test_transcribe_rows_feed_the_shared_scorer_without_zero_k():
    """转写产物直接喂公共打分件：每题拆出的候选数 k≥2、task 带出分账通道（同 harness 同口径）。"""
    rows = [zh.cmmlu_to_envelope({"id": "x/1", "question": "题干", "answer": "A",
                                  "options": {"A": "甲", "B": "乙", "C": "丙", "D": "丁"}}),
            zh.ocnli_to_envelope({"id": "x/2", "sentence1": "前句", "sentence2": "后句",
                                  "label": 1, "task_name": "ocnli"})]
    flat = scoring.rows_from_envelopes(rows)
    assert [r["k"] for r in flat] == [4, 2], f"拆题候选数异常：{[(r['id'], r['k']) for r in flat]}"
    assert {r["task"] for r in flat} == {zh.ZH_CHANNEL["cmmlu"], zh.ZH_CHANNEL["ocnli"]}
    preds = scoring.uniform_predictions(flat)
    scores = scoring.score_rows(flat, preds)
    assert zh.ZH_CHANNEL["cmmlu"] in scores and zh.ZH_CHANNEL["ocnli"] in scores, "按集分账没出这两本"
    assert scores["all"]["n"] == 2


def test_transcribe_rejects_unusable_rows():
    """折不动的行一律拒收并带 id：未知任务名、越界下标、无真值(-1)、缺候选、候选缺项。"""
    with pytest.raises(zh.ChineseTranscribeError, match="未登记决策化规则"):
        zh.clue_to_envelope({"id": "clue:cslr:validation:1", "task_name": "cslr"})
    with pytest.raises(zh.ChineseTranscribeError, match="越出本模块登记的类别码表"):
        zh.tnews_to_envelope({"id": "clue:tnews:validation:2", "sentence": "文", "label": 99})
    with pytest.raises(zh.ChineseTranscribeError, match="真值下标为负"):
        zh.ocnli_to_envelope({"id": "clue:ocnli:test:9", "sentence1": "a", "sentence2": "b", "label": -1})
    with pytest.raises(zh.ChineseTranscribeError, match="没有真值下标"):
        zh.tnews_to_envelope({"id": "clue:tnews:validation:3", "sentence": "文", "label": None})
    with pytest.raises(zh.ChineseTranscribeError, match="缺候选"):
        zh.cmmlu_to_envelope({"id": "cmmlu:x:test/1", "question": "题干", "answer": "A",
                              "options": {"A": "甲", "B": "乙"}})
    with pytest.raises(zh.ChineseTranscribeError, match="未登记的中文源副本集"):
        zh.transcribe_subset("xlongbench-zh", [])


# ─────────────────── 源头对拍：真值口径只认盘上那份 metadata ───────────────────
def test_transcribe_label_tables_match_source_metadata():
    """tnews 十五枚类别码 / ocnli 三枚关系名，必须与 raw parquet 的 ClassLabel names 逐位相同。"""
    import pyarrow.parquet as pq

    for task, table in (("tnews", zh.TNEWS_LABEL_CODES), ("ocnli", zh.OCNLI_RELATIONS)):
        files = sorted((RAW / "clue-subset" / task).glob("validation-*.parquet"))
        if not files:
            pytest.skip(f"源头 parquet 不在盘上：{RAW / 'clue-subset' / task}"
                        "（先跑 `python -m sys1.eval.registry fetch --cn`）")
        md = pq.ParquetFile(files[0]).schema_arrow.metadata or {}
        names = ((json.loads(md[b"huggingface"].decode()).get("info") or {}).get("features") or {}) \
            .get("label", {}).get("names")
        assert names is not None, f"{task} 的 parquet 没带 ClassLabel names，形态变了"
        assert tuple(str(n) for n in names) == table, \
            f"{task} 标签表与源头不符：源头 {names} vs 本模块 {table}（上游改版即红，先取证再改常量）"


# ─────────────────── 结构层：registry pin 收口（派生集 / 零流量 / CLI） ───────────────────
def test_transcribe_registry_pins_registered_and_version_locked():
    """两个派生集登记齐：derived_from 指向原件、版本串两侧一致、qtypes 与转写口径一致。"""
    assert registry.ZH_DECISION_VERSION == zh.ZH_DECISION_VERSION, "两处版号必须一个字不差"
    for src_pid, dec_pid in zh.SOURCE_TO_DECISION.items():
        pin = registry.REGISTRY[dec_pid]
        src = registry.REGISTRY[src_pid]
        assert pin.kind == "derived" and pin.derived_from == src_pid
        assert pin.revision == zh.ZH_DECISION_VERSION
        assert pin.split == src.split, f"{dec_pid} 的档位必须跟着原件走（{src.split}）"
        assert pin.axis == src.axis == "quality"
        assert pin.seed == registry.NEEDLE_SEED and pin.assembler in registry.ASSEMBLERS
        expect = ("choice",) if src_pid == "cmmlu-subset" else ("choice", "noul")
        assert pin.qtypes == expect, f"{dec_pid} 登记题型 {pin.qtypes} 与转写口径 {expect} 不符"
        assert set(pin.qtypes) <= set(registry.QTYPES), f"{dec_pid} 登记了公共契约外的题型"
    assert registry.FLAG_TO_IDS["zh"] == tuple(zh.SOURCE_TO_DECISION.values())


def test_transcribe_cli_has_zh_button_and_extra_ids_cover_it():
    """`--zh` 在 argparse 上真有这个按钮（只在 FLAG_TO_IDS 里加键，按下去是不会生效的）；--all 覆盖到。"""
    args = registry.build_parser().parse_args(["fetch", "--zh"])
    assert args.zh is True
    assert registry.registry_ids(groups=("zh",)) == tuple(zh.SOURCE_TO_DECISION.values())
    assert set(zh.SOURCE_TO_DECISION.values()) <= set(registry.EXTRA_IDS), "--all 重建不到中文信封"
    assert set(zh.SOURCE_TO_DECISION.values()) <= set(registry.registry_ids(all_sets=True))


def test_transcribe_derived_fetch_is_zero_traffic_but_keeps_evidence(tmp_path):
    """派生路零流量：bytes=0、cached、resolved 指到原件，并把原件散列写进 derived_from.json 当凭据。"""
    origin = ASM / "cmmlu-subset" / "cmmlu-subset.jsonl"
    if not origin.is_file():   # skip-when-missing（R-P1-4 统一口径，同本文件 :146-147 的 parquet 判据）：
        # 原件是 bench/ 装配副本，gitignored 按设计不入库；缺件时 _fetch_derived 报 FetchError
        # （见下方 :202 那条报错路用例），那不是"零流量派生"要验收的事实层回归。
        pytest.skip(f"cmmlu-subset 题面原件不在盘上：{origin}（先跑 `python -m sys1.eval.registry fetch --cn`）")
    pin = registry.REGISTRY["cmmlu-decision"]
    got = registry._fetch_derived(pin, tmp_path / pin.id)
    assert got["bytes"] == 0 and got["cached"] is True and got["endpoint"] == "local"
    assert got["resolved"].startswith("derived:cmmlu-subset@")
    claim = json.loads((tmp_path / pin.id / "derived_from.json").read_text(encoding="utf-8"))
    assert claim["from"] == "cmmlu-subset" and claim["origin_bytes"] == origin.stat().st_size
    assert claim["origin_sha256"] == registry._sha256(origin), "版本凭证必须是从原件实测出来的"


def test_transcribe_missing_origin_raises_actionable_command(tmp_path, monkeypatch):
    """缺件时的三条报错路都得指到能敲的命令（把 DATA_DIR 挪进空目录来模拟"还没 fetch"）。"""
    monkeypatch.setattr(registry, "DATA_DIR", tmp_path / "not_fetched_yet")
    with pytest.raises(zh.ChineseTranscribeError, match="fetch --cn"):
        zh.load_subset("cmmlu-subset")                     # 题面原件没拉过
    with pytest.raises(zh.ChineseTranscribeError, match="fetch --zh"):
        zh.load_decision_records("cmmlu-decision")         # 中文信封没装配过
    with pytest.raises(registry.FetchError, match="fetch --sets cmmlu-subset"):
        registry._fetch_derived(registry.REGISTRY["cmmlu-decision"], tmp_path / "dst")


# ─────────────────── 事实层：底账三口一致、id 可回溯、复跑同结果 ───────────────────
def test_transcribe_assembled_ledger_three_mouths_agree():
    """qtype 三口一致（登记口 pin.qtypes / 产物口行级 qtype / 账本口 manifest.qtype_counts）。"""
    _need_decision_sets()
    man = (registry.load_manifest().get("sets") or {})
    for src_pid, dec_pid in zh.SOURCE_TO_DECISION.items():
        assert registry.REGISTRY[dec_pid].split == registry.REGISTRY[src_pid].split, \
            f"{dec_pid} 的档位与题面原件 {src_pid} 脱钩"
        entry = man[dec_pid]
        rows = _read_jsonl(ASM / dec_pid / f"{dec_pid}.jsonl")
        counts = entry["qtype_counts"]
        assert entry["samples"] == len(rows), f"{dec_pid} 账面 {entry['samples']} 条 vs 盘上 {len(rows)} 条"
        seen = {r["qtype"] for r in rows}
        assert seen <= set(registry.REGISTRY[dec_pid].qtypes), f"{dec_pid} 行级题型越出登记：{seen}"
        assert {q for q, n in counts.items() if q != "_other" and n > 0} == seen, \
            f"{dec_pid} 账本 {counts} 与行级 {seen} 不一口"
        assert counts.get("_other", 0) == 0, f"{dec_pid} 有 {counts['_other']} 条题型不认识"
        cov = entry["gold_coverage"]
        assert cov["with_gold"] == cov["total"] == len(rows), f"{dec_pid} 有题没真值：{cov}"
        assert entry["envelope_ready"] is True
        assert entry["assembled_bytes"] == (ASM / dec_pid / f"{dec_pid}.jsonl").stat().st_size
        assert entry["assembled_sha256"] == registry._sha256(ASM / dec_pid / f"{dec_pid}.jsonl")
        assert {r.get("split") for r in rows} == {registry.REGISTRY[dec_pid].split}, "split 未显式随行走"


def test_transcribe_ids_traceable_to_source_rows():
    """每条信封的 id 都能在源副本里找到原题面——中文分要能逐题追问回题面原件。"""
    _need_decision_sets()
    for src_pid, dec_pid in zh.SOURCE_TO_DECISION.items():
        src_ids = {r["id"] for r in _read_jsonl(ASM / src_pid / f"{src_pid}.jsonl")}
        rows = _read_jsonl(ASM / dec_pid / f"{dec_pid}.jsonl")
        assert len(rows) == len(src_ids), f"{dec_pid} 条数与原件不等（{len(rows)} vs {len(src_ids)}）"
        orphan = [r["id"] for r in rows if r["id"] not in src_ids]
        assert not orphan, f"{dec_pid} 有 {len(orphan)} 条 id 追不回原件：{orphan[:3]}"


def test_transcribe_is_deterministic_rerun_builds_same_rows():
    """同一套规则复跑必然得到同一批信封（逐行等值，散列才谈得上可比）。"""
    _need_decision_sets()
    for src_pid, dec_pid in zh.SOURCE_TO_DECISION.items():
        rows = zh.transcribe_subset(src_pid, zh.load_subset(src_pid))
        disk = _read_jsonl(ASM / dec_pid / f"{dec_pid}.jsonl")
        assert len(rows) == len(disk), f"{dec_pid} 复跑 {len(rows)} 条 vs 盘上 {len(disk)} 条"
        for built, stored in zip(rows, disk):
            assert built["id"] == stored["id"] and built["qtype"] == stored["qtype"]
            assert built["sample"] == stored["sample"], f"{built['id']} 复跑结果漂移（样本体不等）"
        assert _digest(rows) == _digest(disk), f"{dec_pid} 整批摘要与盘上不符（现推≠装配口）"


def test_transcribe_quality_axis_carries_chinese_and_d2_isolation_holds():
    """中文决策信封经唯一的读题口进质量轴；D2 训测隔离断言不破（train 档不混进评测轴）。"""
    _need_decision_sets()
    records = evalrun.load_axis_records("quality")
    by_task = {r.get("task") for r in records}
    assert {zh.ZH_CHANNEL["cmmlu"]} <= by_task, f"质量轴没读到中文信封：{sorted(by_task)}"
    train_ids = set(evalrun.axis_ids(evalrun.TRAIN_AXIS))
    for axis in evalrun.AXES_SIX:
        assert not set(evalrun.axis_ids(axis)) & train_ids, f"中文集混进了训练轴 {axis}"
    assert not (set(zh.SOURCE_TO_DECISION.values()) & set(registry.TRAIN_IDS))
    train = evalrun.load_train_records()
    overlap = {r["id"] for r in train} & {r["id"] for r in records}
    assert not overlap, f"训测同集：{sorted(overlap)[:3]}"


def test_transcribe_channel_counts_measured():
    """按集分账的实测数：cmmlu 200 题全 choice；clue = tnews + ocnli 两通道之和（数字现场取，不写死）。"""
    _need_decision_sets()
    rows = _read_jsonl(ASM / "clue-decision" / "clue-decision.jsonl")
    n_choice = sum(1 for r in rows if r["qtype"] == "choice")
    n_noul = sum(1 for r in rows if r["qtype"] == "noul")
    assert n_choice + n_noul == len(rows) and n_noul > 0 and n_choice > 0
    by_task: dict[str, int] = {}
    for r in rows:
        by_task[r["task"]] = by_task.get(r["task"], 0) + 1
    assert set(by_task) == {zh.ZH_CHANNEL["tnews"], zh.ZH_CHANNEL["ocnli"]}
    assert by_task[zh.ZH_CHANNEL["tnews"]] == n_choice
    assert by_task[zh.ZH_CHANNEL["ocnli"]] == n_noul
    cmmlu = _read_jsonl(ASM / "cmmlu-decision" / "cmmlu-decision.jsonl")
    assert all(r["qtype"] == "choice" and len(r["sample"]["questions"][zh.QID]["criteria"]) == 4
               for r in cmmlu)


# ─────────────────── B2 骨架层：G3 表结构与失败样本挂列 ───────────────────
def test_g3_skeleton_has_laya_columns_and_pending_marks():
    """骨架：两个集各一块表，含 laya 行、ms 行、采样参数列与三格失败样本位；DML 格明写待 C5。"""
    md = zh.build_g3_md(rows={pid: None for pid in zh.G3_SETS}, ledger=registry.load_manifest())
    for pid in zh.G3_SETS:
        assert f"## {pid}" in md
    assert f"| 指标 | {zh.G3_DML_LABEL} | laya |" in md
    assert "逐题耗时中位(ms)" in md and "采样参数" in md and "吞吐(题/秒)" in md
    assert md.count(f"| {zh.PENDING_LABEL} |") > 0
    assert md.count("| 待 laya 中文亲跑落预测 jsonl 后挂载 |") == 2 * zh.FAILURE_SAMPLE_SLOTS
    assert "—" in md, "没实测的 laya 格要是空着就分不清'没跑'与'跑砸了'"
    assert "zh_decision_v1" in md and "沿用源头给的类别码" in md


def test_g3_fills_laya_cell_when_rowjson_exists(tmp_path):
    """挂列机制真在取数：喂一份 laya 的 row.json，表上就该出现它的实测值与 run-id/采样参数。"""
    (tmp_path / "laya").mkdir(parents=True)
    (tmp_path / "laya" / "zh-cmmlu.row.json").write_text(json.dumps({
        "dataset": "cmmlu-decision", "split": "zh-cmmlu", "n_samples": 200,
        "metrics": {"acc": 0.2731, "ece_after": 0.11, "ms_p50": 188.5, "throughput_qps": 5.305},
        "sampling_params": "T=1/top_p=1(readout)", "on_device": False,
        "run_id": "1006-p209-laya-probe", "source": "local-run", "gate": False,
    }), encoding="utf-8")
    rows = zh.collect_g3_rows(root=tmp_path)
    md = zh.build_g3_md(rows=rows, ledger=registry.load_manifest())
    assert "0.2731" in md and "188.5" in md and "5.305" in md
    assert "1006-p209-laya-probe" in md and "T=1/top_p=1(readout)" in md
    assert zh.PENDING_LABEL in md, "DML 那一列还没数，仍须显式挂着，不能被 laya 的数顶掉"


def test_g3_failure_samples_pick_wrong_items_and_truncate_state():
    """失败样本挂列：只挑人工≠模型的那几题，至多 3 例，题面按字数截断；没预测就交空表。"""
    records = [zh.cmmlu_to_envelope({
        "id": f"cmmlu:x:test/{i}", "question": "很长的一段中文题干" * 20, "answer": "A",
        "options": {"A": "甲", "B": "乙", "C": "丙", "D": "丁"}}) for i in range(5)]
    # 预测键取打分层拆出来的 "<题号>/<qid>" 形状（与 predictions.jsonl 同形，不是本用例自造的）
    wrong = {f"cmmlu:x:test/{i}/{zh.QID}": {"A": 0.1, "B": 0.7, "C": 0.1, "D": 0.1} for i in range(5)}
    picked = zh.pick_failure_samples(records, wrong)
    assert len(picked) == zh.FAILURE_SAMPLE_SLOTS, "spec 要 3 例，多给少给都不算达标"
    assert all(s["gold"] == "A" and s["pred"] == "B" and s["k"] == 4 for s in picked)
    assert all(s["id"].endswith(f"/{zh.QID}") for s in picked), "题号须与分数表同形，便于逐题追问"
    assert all(len(s["state"]) <= zh.FAILURE_STATE_CHARS + 1 for s in picked)
    assert picked[0]["state"].endswith("…"), "题面按字数截断并留省略号（不整段外抄）"
    right = {f"cmmlu:x:test/{i}/{zh.QID}": {"A": 0.9, "B": 0.03, "C": 0.03, "D": 0.04}
             for i in range(5)}
    assert zh.pick_failure_samples(records, right) == []
    assert zh.pick_failure_samples(records, {}) == []
    mixed = dict(right, **{f"cmmlu:x:test/2/{zh.QID}": {"A": 0.1, "B": 0.8, "C": 0.05, "D": 0.05}})
    only_one = zh.pick_failure_samples(records, mixed)
    assert [s["id"] for s in only_one] == [f"cmmlu:x:test/2/{zh.QID}"], "答对的题不该混进失败样本"
    lines = zh._failure_lines([])
    assert sum(1 for ln in lines if zh.PENDING_LABEL in ln) == zh.FAILURE_SAMPLE_SLOTS


# ─────────────────── B1 骨架层：配比两档参数化与卫生闸 ───────────────────
def test_mix_configs_are_two_slots_differing_only_in_ratio():
    """两档齐备、配比取 30%/50%、除 zh_ratio 外逐键相同（消融只许动一个因子）。"""
    docs = [zh.load_mix_config(CONFIGS / f"zh_mix_{int(r * 100)}.yaml") for r in zh.MIX_RATIOS]
    assert [d["zh_ratio"] for d in docs] == list(zh.MIX_RATIOS)
    cmp = zh.compare_mix_configs(docs[0], docs[1])
    assert cmp["identical"] is True and cmp["unexpected_diffs"] == {}
    for doc in docs:
        zh.assert_mix_hygiene(doc)
        assert doc["mix"]["zh_corpus"] == zh.MIX_CORPUS_PENDING, "中文 train 档还没登记，必须挂显式占位"


def test_mix_configs_legal_keys_are_accepted_by_sft_loader(tmp_path):
    """摘掉两枚待接入键后，其余键今天就能被 `production/sft.py::load_config` 认下（拼错即报错）。"""
    import yaml

    from production import sft

    for r in zh.MIX_RATIOS:
        doc = zh.load_mix_config(CONFIGS / f"zh_mix_{int(r * 100)}.yaml")
        path = tmp_path / f"legal_{int(r * 100)}.yaml"
        path.write_text(yaml.safe_dump(doc["config"], allow_unicode=True), encoding="utf-8")
        cfg = sft.load_config(path)
        assert cfg["seed"] == doc["config"]["seed"] and abs(cfg["lr"] - doc["config"]["lr"]) < 1e-12
        with pytest.raises(ValueError, match="未知配置键"):
            sft.load_config(CONFIGS / f"zh_mix_{int(r * 100)}.yaml")   # 待接入键今天必被拒


def test_mix_hygiene_refuses_eval_sets_as_train_corpus(tmp_path):
    """卫生闸：把中文考卷(cmmlu-decision)当训练语料写进配比档，当场拒绝（训测同集=假分）。"""
    import yaml

    base = zh.load_mix_config(CONFIGS / "zh_mix_30.yaml")["mix"]
    for bad, why, pattern in ((zh.SOURCE_TO_DECISION["cmmlu-subset"], "evalset", "训测同集"),
                              ("not-registered-at-all", "unknown", "没在 registry 登记")):
        data = dict(yaml.safe_load((CONFIGS / "zh_mix_30.yaml").read_text(encoding="utf-8")))
        data["zh_corpus"] = bad
        path = tmp_path / f"bad_{why}.yaml"
        path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        doc = zh.load_mix_config(path)
        assert doc["mix"]["zh_corpus"] == bad, "脏值得真进配比栏，否则这一闸是在空转"
        assert doc["mix"]["zh_ratio"] == base["zh_ratio"], "只该改语料引用，别把配比也顶掉了"
        with pytest.raises(zh.ChineseTranscribeError, match=pattern):
            zh.assert_mix_hygiene(doc)
    assert base["zh_corpus"] == zh.MIX_CORPUS_PENDING, "正例：在案可查的占位符必须放行"
    zh.assert_mix_hygiene(zh.load_mix_config(CONFIGS / "zh_mix_30.yaml"))


def test_mix_config_rejects_unknown_ratio_and_stray_key(tmp_path):
    """配比写档外的数（如 0.4）、或多写一枚两边都没登记的键，都当场报错（写了没生效最难查）。"""
    import yaml

    data = yaml.safe_load((CONFIGS / "zh_mix_30.yaml").read_text(encoding="utf-8"))
    off = tmp_path / "off.yaml"
    off.write_text(yaml.safe_dump({**data, "zh_ratio": 0.4}, allow_unicode=True), encoding="utf-8")
    with pytest.raises(zh.ChineseTranscribeError, match="不在设计给定的两档"):
        zh.load_mix_config(off)
    stray = tmp_path / "stray.yaml"
    stray.write_text(yaml.safe_dump({**data, "zh_weight": 3}, allow_unicode=True), encoding="utf-8")
    with pytest.raises(zh.ChineseTranscribeError, match="都没登记的键"):
        zh.load_mix_config(stray)
    missing = tmp_path / "missing.yaml"
    missing.write_text(yaml.safe_dump({k: v for k, v in data.items() if k != "zh_corpus"},
                                      allow_unicode=True), encoding="utf-8")
    with pytest.raises(zh.ChineseTranscribeError, match="必须齐备配比键"):
        zh.load_mix_config(missing)
