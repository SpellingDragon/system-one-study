"""评测集登记与六轴调度单测：pin 台账 / 扩展集自持 / Uniform 锚点 / 六轴不缺席。

分层说明（为什么有的用例查盘、有的不查盘）：
  本域的数据产物（`bench/eval_data/`）是 gitignored 的下载结果，新机器上默认不存在。因此把
  用例分成两层：**结构层**只查注册表这个"应然"（pin 版本、分割、种子、轴归属、导出接口形状），
  任何时候都必须绿；**事实层**查盘上真拉下来的副本（条数、题型字段、真值覆盖、锚点复现），
  副本不在就直接 skip 并说明"先跑 fetch"——绝不把"没数据"写成"通过"，也绝不为了跑测试去联网。
  锚点用例是熔断器：它验的不是"算得出数"，而是"算得出的数必须等于上游卡面那行 Uniform"。

运行方式（务必在 release/ 下用 -m）：
    cd release && .venv/bin/python -m pytest tests/test_registry.py -q
    ... -k versions   # A1 骨架 + export_versions
    ... -k extras     # A3 中文/长文/多模态注册与子集自持
    ... -k anchor     # B1 Uniform 锚点复现（容差 0.01）
    ... -k axes       # B2 六轴调度入口（含 B3 一致轴裁定与 npu 占位）
    ... -k split_isolation   # D2 train/test 隔离（训练档不入评测轴）
事实层前置：`.venv/bin/python -m sys1.eval.registry fetch --all`
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sys1.eval import registry
from sys1.eval import run as evalrun
from sys1.eval import scoring

ASM = registry.DATA_DIR / registry.ASSEMBLED_DIR_NAME
SET_FILES = {pid: ASM / pid / f"{pid}.jsonl" for pid in registry.REGISTRY if pid != "mmbench-cn-subset"}


def _manifest() -> dict[str, Any]:
    return registry.load_manifest()


def _fetched(man: dict[str, Any], pid: str) -> bool:
    return ((man.get("sets") or {}).get(pid) or {}).get("status") in ("fetched", "cached")


def _need_disk(*pids: str) -> None:
    """盘上缺副本就 skip（并给出可执行的补救命令），不假装通过。"""
    man = _manifest()
    missing = [pid for pid in pids if not _fetched(man, pid)]
    if missing:
        pytest.skip(f"评测副本未下载：{missing}（先跑 `python -m sys1.eval.registry fetch --all`）")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ───────────────────────────── A1：三件套 pin 与版本导出 ─────────────────────────────
def test_versions_three_sheet_pins_are_exact():
    """三件套的版本号、分割、装配器必须是变更文档定死的那一组，改动即失败。"""
    expect = {
        "typed-decisions": (registry.PIN_TYPED, "f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8"),
        "intern-decision": (registry.PIN_INTERN, "2f815802058b2464144218859ea9221c1bc0d2a8"),
        "jevbench": (registry.PIN_JEV, "7ce310c7262ed49cc85853339a8a42459298e3f3"),
    }
    for pid, (short, full) in expect.items():
        pin = registry.REGISTRY[pid]
        assert pin.revision == short, f"{pid} revision 漂移：{pin.revision}"
        assert pin.full == full, f"{pid} 全长散列漂移：{pin.full}"
        assert pin.split == "test", f"{pid} 必须钉在 test 分割：{pin.split}"
        assert pin.assembler in registry.ASSEMBLERS
        assert set(pin.qtypes) <= set(registry.QTYPES), f"{pid} 题型越界：{pin.qtypes}"


def test_versions_export_keeps_unfetched_sets_visible():
    """一个集都没拉过的时候，导出表必须逐集占行、实测字段为 None（不是 0，也不是省略）。"""
    rows = registry.export_versions(manifest={"sets": {}, "fetch": {}})
    assert len(rows) == len(registry.REGISTRY)
    for row in rows:
        assert row["status"] == "absent"
        assert row["samples"] is None, "没拉过应写 None（0 会被读成'这个集零道题'）"
        assert row["bytes"] is None
        # 轴只能是六条评测轴之一，或训练登记轴 train（D1：train 不入评测表，但必须占行）
        assert row["axis"] in evalrun.AXES_SIX + (evalrun.TRAIN_AXIS,), \
            f"{row['id']} 挂在不存在的轴上：{row['axis']}"


def test_versions_export_merges_measured_facts():
    """账本里的条数/字节/散列要并进同一行，报告才只有一个数据真源。"""
    fake = {"sets": {"jevbench": {"status": "fetched", "resolved": "7ce310c7",
                                  "bytes_downloaded": 4640431, "samples": 231,
                                  "qtype_counts": {"choice": 139, "noul": 74, "score": 18},
                                  "assembled_sha256": "abc123", "assembled": "/tmp/jev"}}}
    row = next(r for r in registry.export_versions(manifest=fake) if r["id"] == "jevbench")
    assert (row["status"], row["samples"], row["bytes"]) == ("fetched", 231, 4640431)
    assert row["sha256"] == "abc123" and row["qtype_counts"]["score"] == 18


def test_versions_table_shows_three_sheet_and_counts():
    """人读版表格：三件套各占一行、样本数与字节都在，缺数据用短横线而不是留空。"""
    table = registry.format_versions_table()
    for pid in registry.THREE_SHEET_IDS:
        assert pid in table
    header, first = table.splitlines()[0], table.splitlines()[2]
    for col in ("id", "revision", "split", "seed", "axis", "samples", "bytes"):
        assert col in header
    assert first.strip()


# ─────────────────────── A3：中文 / 长文 / 多模态注册与子集自持 ───────────────────────
def test_extras_registered_with_fixed_seed():
    """扩展集一律固定 seed、只取子集，且轴归属正确（长文/多模态不能混进质量轴）。"""
    expect_axis = {"cmmlu-subset": "quality", "clue-subset": "quality",
                   "mmbench-cn-subset": "multimodal", "longbench-zh": "longctx",
                   "needle-synthetic": "longctx"}
    for pid, axis in expect_axis.items():
        pin = registry.REGISTRY[pid]
        assert pin.axis == axis, f"{pid} 轴错位：{pin.axis}"
        assert pin.seed == registry.NEEDLE_SEED, f"{pid} 种子不固定：{pin.seed}"
        assert pin.kind in ("modelscope", "synthetic"), f"{pid} 来源异常：{pin.kind}"
        if pin.kind == "modelscope":
            assert pin.patterns, f"{pid} 没写文件白名单，会把整仓拖下来"
            assert str(registry.MIN_SUBSET) in pin.sampler or "subset" in pin.sampler
    assert registry.REGISTRY["cmmlu-subset"].split == "test"
    # CLUE 的 test 档官方隐藏答案（实测 label 整列 -1），因此登记在 validation
    assert registry.REGISTRY["clue-subset"].split == "validation"


def test_extras_needle_plan_is_deterministic():
    """合成针的档位/针数/派生 seed 全由总种子决定，两次调用逐字相同。"""
    a, b = registry.needle_plan(), registry.needle_plan(seed=registry.NEEDLE_SEED)
    assert a == b
    assert [x["ctx"] for x in a["buckets"]] == list(registry.NEEDLE_BUCKETS)
    assert registry._assemble_needle(Path("/nonexistent")) == registry._assemble_needle(
        Path("/also-nonexistent"))


def test_extras_subset_files_on_disk():
    """每个扩展集都要有自持副本落在 bench/eval_data/assembled/ 下，题量不低于子集下限。"""
    _need_disk(*registry.EXTRA_IDS)
    for pid in registry.EXTRA_IDS:
        man_entry = (_manifest().get("sets") or {})[pid]
        root = Path(man_entry["assembled"])
        files = sorted(p for p in root.rglob("*") if p.is_file())
        assert files, f"{pid} 自持副本目录是空的：{root}"
        assert man_entry.get("samples", 0) >= registry.MIN_SUBSET or pid == "needle-synthetic", \
            f"{pid} 只有 {man_entry.get('samples')} 条，低于下限 {registry.MIN_SUBSET}"
    mm = ASM / "mmbench-cn-subset"
    assert (mm / "subset.parquet").is_file() and (mm / "index.jsonl").is_file()
    assert len(list((mm / "images").iterdir())) > 0, "图文集必须自带图文件，否则 p2-08 取不到像素"


def test_extras_qtype_field_uniform():
    """题型字段一律叫 qtype（下游只认这一个口），且取值在 choice/noul/score 或显式 untyped。"""
    _need_disk(*registry.EXTRA_IDS)
    allowed = set(registry.QTYPES) | {"untyped"}
    for pid in registry.EXTRA_IDS:
        entry = (_manifest().get("sets") or {})[pid]
        if pid == "mmbench-cn-subset":
            rows = _read_jsonl(Path(entry["assembled"]) / "index.jsonl")
        else:
            rows = _read_jsonl(SET_FILES[pid])
        assert rows, f"{pid} 副本为空"
        for row in rows:
            assert "qtype_hint" not in row, f"{pid} 仍有旧字段名 qtype_hint"
            assert row.get("qtype") in allowed, f"{pid}/{row.get('id')} 题型非法：{row.get('qtype')}"


def test_extras_cn_sets_carry_human_gold():
    """中文子集不留无真值的行（CLUE test 那种 label=-1 的档一律不收），账本要能核对。"""
    _need_disk("cmmlu-subset", "clue-subset")
    man = _manifest()
    for pid in ("cmmlu-subset", "clue-subset"):
        entry = (man.get("sets") or {})[pid]
        cov = entry.get("gold_coverage") or {}
        assert cov.get("with_gold") == entry.get("samples"), \
            f"{pid} 带真值 {cov.get('with_gold')} 条 / 共 {entry.get('samples')} 条"
        rows = _read_jsonl(SET_FILES[pid])
        for row in rows:
            assert str(row.get("answer", row.get("label", ""))) not in ("", "-1", "None")


# ───────────────────────────── B1：Uniform 锚点自检 ─────────────────────────────
def test_anchor_empty_rows_is_not_fake_green():
    """没有题面时自检必须判红并指路 fetch——不给"分母为零所以全对"这种假绿。"""
    rep = scoring.anchor_selfcheck([])
    assert rep["ok"] is False and rep["n"] == 0
    assert "fetch" in rep["detail"]


def test_anchor_uniform_card_reproduction():
    """均匀瞎猜 typed-decisions 全量，KL/TV/Brier 必须复现上游卡面 0.444/0.381/0.238。"""
    _need_disk("typed-decisions")
    rows = scoring.rows_from_envelopes(scoring.load_assembled(str(SET_FILES["typed-decisions"])))
    assert len(rows) == 2000, f"锚点集应为 400 案例×5 问=2000 决策，实得 {len(rows)}"
    rep = scoring.anchor_selfcheck(rows)
    assert rep["ok"] is True, f"锚点复现失败：{rep}"
    assert rep["expected"] == {"kl": 0.444, "tv": 0.381, "brier": 0.238}
    for key, diff in rep["abs_diff"].items():
        assert diff <= rep["tolerance"], f"{key} 偏离卡面 {diff} > {rep['tolerance']}"


def test_anchor_harness_catches_wrong_assignment():
    """把人为均匀的答案换成 one-hot，锚点必须当场判红（证明熔断器真的会熔断）。"""
    rows = [{"id": f"q{i}", "task": "t", "qtype": "choice", "keys": ["a", "b", "c", "d"], "k": 4,
             "gold": {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0}} for i in range(50)]
    rep = scoring.anchor_selfcheck(rows)
    assert rep["ok"] is False, f"one-hot 真值竟被判成锚点通过：{rep}"


# ───────────────────────────── B2：六轴调度入口 ─────────────────────────────
def test_axes_six_table_has_all_six():
    """`--axes all` 的表必须六轴齐全、顺序固定，出分的轴给 status，没数的轴给 reason。"""
    result = evalrun.run_six(axes="all", backend="cpu", manifest=_manifest())
    assert tuple(result["axes"]) == evalrun.AXES_SIX
    assert evalrun.AXES_SIX == ("quality", "calibration", "longctx", "multimodal", "speed", "parity")
    table = evalrun.format_table(result)
    for axis in evalrun.AXES_SIX:
        assert axis in table, f"{axis} 从表上消失了"
    for axis, st in result["axes"].items():
        assert st["status"] in ("data-ready", "scored", "needs-model", "n/a"), f"{axis}: {st['status']}"
        if st["status"] != "data-ready":
            assert st["reason"], f"{axis} 标了 {st['status']} 却没说为什么"


def test_axes_unfetched_manifest_still_lists_every_axis():
    """一个集都没拉的账本上，六轴也必须在场并各自说明缺什么——spec 禁止静默省略。"""
    empty = {"sets": {}, "fetch": {}}
    statuses = evalrun.collect_axes("all", manifest=empty)
    assert len(statuses) == 6
    for axis in ("quality", "calibration"):
        assert statuses[axis]["status"] == "n/a"
        assert "未拉取" in statuses[axis]["reason"] or "没有挂" in statuses[axis]["reason"]
    assert statuses["parity"]["status"] == "n/a", "一致轴被裁定为 n/a，不受数据有无影响"


def test_axes_registered_but_not_fetched_reason():
    """注册了但没拉取的轴，reason 要指到具体哪几集和 fetch 命令。"""
    man = {"sets": {}, "fetch": {}, "registered": list(registry.REGISTRY)}
    st = evalrun.axis_status("longctx", manifest=man)
    assert st["sets"], "长文轴应能从 registry 推出挂载的集"
    assert st["status"] == "n/a"
    assert "未拉取" in st["reason"] and "fetch --all" in st["reason"]


def test_axes_unknown_axis_rejected():
    """不认识的轴名当场报错，而不是悄悄少跑一轴。"""
    with pytest.raises(ValueError):
        evalrun.collect_axes("quality,magic", manifest={"sets": {}})


def test_axes_axis_ids_derive_from_registry():
    """轴与集的对应只从 registry.axis 推，不许在两处各维护一份清单。"""
    for axis in evalrun.AXES_SIX:
        ids = evalrun.axis_ids(axis)
        assert all(registry.REGISTRY[pid].axis == axis for pid in ids)
        expect = tuple(pid for pid, pin in registry.REGISTRY.items() if pin.axis == axis)
        assert ids == expect
    assert evalrun.axis_ids("parity") == (), "一致轴本域不挂数据（对象裁定见 B3）"


def test_axes_parity_adjudication_written_down():
    """B3 裁定必须在入口上可见：结论、测什么、不测什么、n/a 时报告怎么写，四样齐。"""
    adj = evalrun.PARITY_ADJUDICATION
    for key in ("verdict", "measures", "not_measured", "na_note"):
        assert adj.get(key), f"裁定缺 {key}"
    assert len(adj["not_measured"]) >= 2
    result = evalrun.run_six(axes="all", backend="cpu", manifest=_manifest())
    assert result["parity_adjudication"] == adj
    assert result["axes"]["parity"]["status"] == "n/a"
    assert adj["na_note"][:20] in result["axes"]["parity"]["reason"]


def test_axes_backend_npu_is_placeholder():
    """D6 v3：后端只有 cpu/mps/npu 三个取值；npu 必须当场拒绝并指出接入域（不静默回退 CPU）。"""
    assert scoring.BACKENDS == ("cpu", "mps", "npu")
    assert scoring.resolve_backend("cpu") == "cpu"
    with pytest.raises(NotImplementedError, match="p2-13"):
        scoring.resolve_backend("npu")
    with pytest.raises(NotImplementedError):
        evalrun.run_six(axes="quality", backend="npu", manifest={"sets": {}})
    with pytest.raises(ValueError):
        scoring.resolve_backend("tpu")


def test_axes_data_revision_counts_fetched_sets():
    """写进 run 档案的数据版本串：三件套 pin + 实拉集数 + 总条数，与账本一致。"""
    man = _manifest()
    rev = evalrun.data_revision(man)
    sets = man.get("sets") or {}
    got = [pid for pid, s in sets.items() if s.get("status") in ("fetched", "cached")]
    total = sum(int(sets[pid].get("samples") or 0) for pid in got)
    assert f"sets={len(got)}" in rev and f"samples={total}" in rev
    for pid in registry.THREE_SHEET_IDS:
        assert f"{pid}@{registry.REGISTRY[pid].revision}" in rev


def test_axes_report_template_carries_adjudication():
    """报告模板里必须带裁定四要素与数据底账——报告不是每次手写，漏一句就成了悬空轴。"""
    man = _manifest()
    result = evalrun.run_six(axes="all", backend="cpu", manifest=man)
    body = evalrun.format_report_md(result, manifest=man)
    for key in ("verdict", "measures", "na_note"):
        assert evalrun.PARITY_ADJUDICATION[key] in body, f"模板漏了裁定项 {key}"
    for item in evalrun.PARITY_ADJUDICATION["not_measured"]:
        assert item in body, f"模板没写明不测对象：{item[:20]}"
    for section in ("数据底账", "六轴表", "一致轴裁定", "读表须知"):
        assert section in body
    assert f"`{evalrun.data_revision(man)}`" in body


def test_axes_report_template_counts_real_bytes_and_gold():
    """模板里的字节量/条数/带真值条数必须与账本逐位相符（报告数字要能溯源回 manifest）。"""
    man = _manifest()
    sets = man.get("sets") or {}
    body = evalrun.format_report_md(evalrun.run_six(axes="all", manifest=man), manifest=man)
    nbytes = sum(int(v.get("bytes_downloaded") or v.get("bytes") or 0) for v in sets.values())
    ngold = sum(int((v.get("gold_coverage") or {}).get("with_gold") or 0) for v in sets.values())
    nsamples = sum(int(v.get("samples") or 0) for v in sets.values())
    assert f"累计下载字节：{nbytes:,}" in body
    assert f"登记条数合计：{nsamples:,}（其中带人工答案 {ngold:,}）" in body


# ─────────────────── D2：train/test 隔离（训练档不得混进评测轴） ───────────────────
def test_split_isolation_train_axis_never_on_eval_axes():
    """结构层：train 轴上的集不许出现在任何一条评测轴名下；quality 轴不得挂 split=train 的登记项。"""
    train_ids = set(evalrun.axis_ids(evalrun.TRAIN_AXIS))
    assert "typed-decisions-train" in train_ids, "D1 的 train 登记项缺席"
    assert registry.REGISTRY["typed-decisions-train"].split == "train"
    assert registry.REGISTRY["typed-decisions-train"].axis == evalrun.TRAIN_AXIS
    for axis in evalrun.AXES_SIX:
        assert not (set(evalrun.axis_ids(axis)) & train_ids), f"train 集混进了评测轴 {axis}"
    for pid in evalrun.axis_ids("quality"):
        assert registry.REGISTRY[pid].split != "train", f"quality 轴挂了 train 档：{pid}"
    st = evalrun.collect_axes("all", manifest=_manifest())
    assert evalrun.TRAIN_AXIS not in st, "六轴表上不该出现 train 轴"
    # D3 中文对：train 语料那两格必须在册且只挂 train 轴（与 typed train 同一道闸）
    for pid in registry.ZH_TRAIN_IDS_TUPLE:
        assert pid in registry.REGISTRY, f"D3 中文 train 登记项缺席：{pid}"
        assert registry.REGISTRY[pid].split == "train", f"{pid} 档位必须钉在 train"
        assert registry.REGISTRY[pid].axis == evalrun.TRAIN_AXIS, f"{pid} 必须挂 train 轴"
        assert pid in train_ids, f"{pid} 没进 train 轴消费表：{sorted(train_ids)}"
    # 探查结论必须在案（无 train 档的集只许留结论，不许留一个改名凑数的 pin）
    assert registry.ZH_TRAIN_PROBE["verdict"], "D3 中文 train 探查结论缺席"
    assert registry.ZH_TRAIN_PROBE["evidence"], "D3 探查取证行为空（结论不可复跑即无效）"
    assert "cmmlu" in registry.ZH_TRAIN_PROBE["absent"], "CMMLU 无 train 这条事实得挂在探查表上"
    faked = [pid for pid, pin in registry.REGISTRY.items()
             if pin.axis == evalrun.TRAIN_AXIS and "cmmlu" in pid]
    assert not faked, f"CMMLU 被凭空登记了 train 档（dev 改名凑 train？）：{faked}"


def test_split_isolation_train_ids_disjoint_from_quality():
    """事实层：任何 train 记录 id 不得出现在 quality 评测轴集内；qtype 三口一致、split 显式。

    白话：这一条是 D1/D3 共用的那道隔离闸的"真数据版"——上面那条结构层只保证登记面上
    train 轴没串进六轴，本条要把盘上的信封真读一遍：训练口 scoop 出来的每一条都得写着
    split=train，且这些 id 与考卷（quality 轴，含中文 validation 那 400 题）一个都不能重。
    中文那半单独再断一次：train 语料得真在训练口里（C5 前置闸要的"有米可下锅"），而且
    "练的题"（id 里带 :train:）与"考的题"（id 里带 :validation:）不能是同一批——重了一个，
    训练分与中文评测分就互相漏题，报告上却看不出破口。
    """
    _need_disk("typed-decisions", "typed-decisions-train", *registry.ZH_TRAIN_IDS_TUPLE)
    train = evalrun.load_train_records()
    assert train, "train 消费口 scoop 出空表（先跑 fetch --train）"
    assert all(r.get("split") == "train" for r in train), "数据类记录必须显式带 split 语义"
    assert {"choice", "noul", "score"} <= {r.get("qtype") for r in train}, "train 档 qtype 三口不全"
    quality = evalrun.load_axis_records("quality")
    overlap = {r["id"] for r in train} & {r["id"] for r in quality}
    assert not overlap, f"训测同集：{len(overlap)} 个 train id 出现在 quality 评测轴集内 {sorted(overlap)[:5]}"
    # ── 中文对（D3）：训练口里得有中文档，且与中文考卷逐条互斥 ──
    zh_train = [r for r in train if r["id"].startswith("clue:")]
    assert zh_train, "load_train_records() 里没有中文 train 语料（D3 交付缺口，先跑 fetch --zh-train）"
    assert all(r.get("split") == "train" for r in zh_train), "中文 train 记录必须带 split=train"
    assert all(":train:" in r["id"] for r in zh_train), "中文 train 记录 id 的档位段必须是 train"
    zh_exam = {r["id"] for r in quality if r["id"].startswith("clue:")}
    assert zh_exam, "中文考卷缺席——零交集断言就退化成空断言（先跑 fetch --cn）"
    clash = {r["id"] for r in zh_train} & zh_exam
    assert not clash, f"中文训测同集：{len(clash)} 条 train 语料出现在考卷里 {sorted(clash)[:5]}"


# ─────────────────── A2 补充：幂等复跑不动底账（离线可验的那一半） ───────────────────
def test_fetch_cached_archive_reports_real_bytes(tmp_path):
    """命中缓存的归档路必须仍报出包体真实字节，而不是 0（0 会把真下载量从底账里抹掉）。"""
    dst = tmp_path / "jevbench"
    dst.mkdir(parents=True)
    (dst / "_source.tar.gz").write_bytes(b"x" * 4096)
    (dst / ".pinned_sha").write_text(registry.FULL_JEV, encoding="utf-8")

    got = registry._fetch_archive(registry.REGISTRY["jevbench"], dst)   # 散列相符 => 零流量
    assert got["cached"] is True, "散列一致却没认出缓存"
    assert got["bytes"] == 4096, f"缓存路该报包体真实大小，实得 {got.get('bytes')}"


def test_fetch_ledger_bytes_match_disk_truth():
    """账本里每个归档集的累计字节，必须等于盘上那份原始包的大小（数字可溯源，非估算）。"""
    _need_disk("jevbench", "intern-decision")
    man = _manifest()
    for pid in ("jevbench", "intern-decision"):
        blob = registry.DATA_DIR / registry.RAW_DIR_NAME / pid / "_source.tar.gz"
        entry = (man.get("sets") or {})[pid]
        assert blob.is_file(), f"原始归档包不在盘上：{blob}"
        assert int(entry.get("bytes_downloaded") or 0) == blob.stat().st_size, \
            f"{pid} 账本 {entry.get('bytes_downloaded')} != 包体 {blob.stat().st_size}"
