"""sys1/eval/run.py — 六轴调度入口：P1 四轴原样保留，长文与多模态两轴新挂，缺数据的轴显式 n/a。

【做什么】
    一条命令把六个评测轴（quality / calibration / longctx / multimodal / speed / parity）
    各归各位摆出来：每轴先查 registry 登记的数据是否就绪（哪几集、多少条、多少条真带人工
    答案），再决定这一轴是"能出数"还是"显式 n/a 并说清为什么"。给了 --model 或 --endpoint
    就把题目交给一阶段的预测出口打分、按 P1 口径汇总；没给就只交数据就绪度——本域不出模型分。

【怎么做】
    ① 轴与评测集的对应关系不写死在这里，从 registry 每项的 axis 字段推出来：注册即上轴，
       删一项就少一集，不会出现两处清单各自漂移。
    ② 数据就绪度直接读 `bench/eval_data/manifest.json`（fetch 落的账），本入口不再走一次网络，
       也不重新装配——同一份账既能被人复跑核对，也保证六轴看到的是同一个分母。
    ③ 质量与把握两轴复用 P1 现成口径（`quality.score_quality` / `calibration.score_calibration`），
       输入是 registry 装配出的统一信封；一致轴复用 `report.run_eval` 里那套双路对拍。长文与
       多模态两轴在本域只交就绪度与题目来源，判分件分别是 p2-07 / p2-08 的交付物。
    ④ 一致轴按 p2-03-B3 的裁定走（见 PARITY_ADJUDICATION）：被测体没有第二条实现路径可对拍时
       显式标 n/a 并附裁定，禁止把"没得对拍"混进"对拍通过"。
    ⑤ 设备后端由 `scoring.resolve_backend` 收口：cpu / mps / npu 三个取值，npu 当场拒绝并说清
       接入域在哪，云端换卡时只改这一个参数。
    ⑥ `--record` 用 `sys1.runs.new_run` 开一页评测档案，把 pin 表、字节账与六轴表写进 notes。

【为什么】
    被否方案一：六轴各配一个 CLI、由外层脚本拼表——缺数据的轴会因为"没人调用它"而从表上静默
        消失，这正是 spec 明令禁止的"省略"；只有单一入口才能保证六轴永远都在表上。
    被否方案二：本域顺手把长文与多模态的分也算了——针召回与图文判分分别是 p2-07、p2-08 的
        交付物，在这里先算一份就等于立了第二个口径，日后两边对不上数。
    被否方案三：一致轴对 HF 主干也"跑一遍对拍"——单一实现路径自己跟自己比恒等于全对，是一枚
        假绿；裁定成 n/a 才是实话，也是 P1 教训（悬空轴）的回写。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sys1.eval import registry, scoring

#: 六轴 = 一阶段四轴（原样保留）+ 长文 + 多模态；顺序即表上呈现顺序
AXES_SIX = ("quality", "calibration", "longctx", "multimodal", "speed", "parity")

#: 训练登记轴（p2-03-D1 补录）：只作训练侧取题的登记位，**不在**六轴评测表上——
#: 训练装配复用评测题面=训测同集出假分（p2-05 发现），隔离断言见 split_isolation 用例
TRAIN_AXIS = registry.TRAIN_AXIS

#: P2 一致轴裁定（B3）：写清楚"对谁拍、什么情况才算数、拍不了怎么办"
PARITY_ADJUDICATION = {
    "verdict": "P2 一致轴只测 P1 自训 decoder 的延续路径；HF 主干显式 n/a",
    "measures": "同一份输入下，自研算子路与纯 torch 路的名次是否换人（沿用一阶段 parity 口径）",
    "not_measured": [
        "HF 主干（经 transformers 载入的那条前向）——它只有一条实现路径，自己跟自己比恒等于全对",
        "昇腾卡上的算子对拍——归 p2-13 的梯度对拍验收（G9），不在本域",
        "远端服务与本地权重的一致性——归 p2-10 serving",
    ],
    "na_note": ("parity=n/a：backbone 前向是单一实现路径（本机 CPU/MPS；云端换卡由 p2-13 承接），"
                "无双路可对拍；自研件的 parity 验收见 p2-13 G9"),
}

#: 每条轴"想吃什么数据"的说明（进表注脚，免得读者以为某轴没数就是漏跑）
AXIS_EXPECTATION = {
    "quality": "带人工份额的决策题（typed-decisions / intern-decision 全量 + 中文子集题面）",
    "calibration": "把握度真值可核的集（jevbench 三层 + known-distribution 已知分布试点）",
    "longctx": "长上下文题面（longbench-zh 自持副本 + 合成针位计划）",
    "multimodal": "图文题（mmbench-cn-subset：子集 parquet + 索引 + 图文件）",
    "speed": "任意就绪集若干条即可计时；要真数必须有被测模型",
    "parity": "被测体必须自带第二条实现路径（见 PARITY_ADJUDICATION）",
}


def axis_ids(axis: str, *, reg: dict[str, Any] | None = None) -> tuple[str, ...]:
    """这一轴挂着哪几个评测集 id（数据来自 registry 的 axis 字段，注册即上轴）。"""
    src = reg if reg is not None else registry.REGISTRY
    return tuple(pid for pid, pin in src.items() if pin.axis == axis)


def axis_status(axis: str, *, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """交一轴的数据就绪度：哪几集、共几条、多少条带真值、缺什么——不猜也不省。

    白话：把这一轴要点名的评测集逐个查账，账面写着拉到过几条、其中人工答案在不在，都加总成
    一句话；要是账上一集都没有，就老实说"还没拉"，而不是交一张看着齐全的空白表。
    """
    man = manifest if manifest is not None else registry.load_manifest()
    sets = man.get("sets") or {}
    ids = axis_ids(axis)
    rows = [(pid, sets.get(pid) or {}) for pid in ids]
    present = [(pid, s) for pid, s in rows if s.get("status") in ("fetched", "cached")]
    samples = sum(int(s.get("samples") or 0) for _, s in present)
    with_gold = sum(int((s.get("gold_coverage") or {}).get("with_gold") or 0) for _, s in present)
    env = [pid for pid, s in present if s.get("envelope_ready")]
    out: dict[str, Any] = {
        "axis": axis, "sets": list(ids), "present": [pid for pid, _ in present],
        "samples": samples, "with_gold": with_gold, "envelope_ready_sets": env,
        "expect": AXIS_EXPECTATION.get(axis, ""), "status": "n/a", "reason": "",
    }
    if axis == "parity":                       # B3 裁定优先：不是缺数据，是没得对拍
        out["status"] = "n/a"
        out["reason"] = PARITY_ADJUDICATION["na_note"]
    elif not ids:
        out["reason"] = "registry 里没有挂在这条轴上的集"
    elif axis == "speed":
        out["status"] = "needs-model"
        out["reason"] = "计时轴必须有被测模型；只给数据时不交这份数"
    elif not present:
        out["reason"] = f"已注册但未拉取：{list(ids)}（跑 `python -m sys1.eval.registry fetch --all`）"
    elif axis in ("quality", "calibration") and with_gold == 0:
        out["reason"] = "数据在盘上但一条人工答案都没有，判不了分"
    elif axis in ("longctx", "multimodal"):
        # 这两轴的判分件分属 p2-07 / p2-08；本域只保证题面在盘上、来源可回溯
        out["status"] = "data-ready"
        out["reason"] = (f"题面已自持；判分件属 {'p2-07' if axis == 'longctx' else 'p2-08'}，"
                         f"本域不代打")
    else:
        out["status"] = "data-ready"
    if out["status"] == "data-ready" and len(env) < len(present):
        # 题面在盘上 ≠ 能直接判分：非信封的那几集要等对应域做决策化，这里把分母说清
        note = f"可判分信封 {len(env)}/{len(present)} 集（其余为原始题面副本，决策化归对应专域）"
        out["reason"] = f"{out['reason']}；{note}" if out["reason"] else note
    out["scorable"] = bool(out["status"] == "data-ready" and with_gold > 0)
    return out


def collect_axes(axes: tuple[str, ...] | str = "all",
                 *, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """按点名收集六轴就绪度；`all` 就是六条一个都不许少。

    白话：点名要哪几条轴，就逐条去查账、逐条交一份说明；说 all 就把六条全数点齐。轴名写错会
    当场停下来报错，而不是少交一条还装作交全了——表上少一条轴，看的人压根不知道该问谁。
    """
    names = AXES_SIX if axes == "all" or axes is None else tuple(
        a.strip() for a in (axes if isinstance(axes, (list, tuple)) else str(axes).split(","))
        if a.strip())
    unknown = [a for a in names if a not in AXES_SIX]
    if unknown:
        raise ValueError(f"不认识的轴：{unknown}（可选：{list(AXES_SIX)} 或 all）")
    return {a: axis_status(a, manifest=manifest) for a in names}


def load_axis_records(axis: str, *, data_dir: str | Path | None = None,
                      limit: int = 0) -> list[dict[str, Any]]:
    """取这一轴的全部统一信封（一题一行），`limit` 每条集封顶；不在此处造题。

    白话：把这条轴点名的那些集，从装配好的副本里原样读出来交给打分口。读到一条算一条，
    读不到就交空表——这里绝不为了"表上好看"补几条假题，题目从哪来 registry 已经记了账。
    """
    root = Path(data_dir) if data_dir is not None else registry.DATA_DIR
    out: list[dict[str, Any]] = []
    man = registry.load_manifest(root / registry.MANIFEST_NAME)
    sets = man.get("sets") or {}
    for pid in axis_ids(axis):
        entry = sets.get(pid) or {}
        if not entry.get("envelope_ready"):
            continue
        asm = Path(entry.get("assembled") or (root / registry.ASSEMBLED_DIR_NAME / pid))
        rows = scoring.load_assembled(asm / f"{pid}.jsonl")
        out.extend(rows[:limit] if limit else rows)
    return out


def load_train_records(*, data_dir: str | Path | None = None,
                       limit: int = 0) -> list[dict[str, Any]]:
    """训练数据消费口：只回 `split=='train'` 的统一信封（typed train + 中文 train 语料，D3）。

    白话：训练侧从这里拿题，评测侧从 quality 那个口拿题，两口各走各路——这个口只认
    train 轴上登过记的集，永远 scoop 不到考卷；每条信封都带着档位名（split 字段），
    哪条没带就当场停下来报缺，绝不把来路不明的题混进训练分母里凑数。
    """
    records = load_axis_records(TRAIN_AXIS, data_dir=data_dir, limit=limit)
    bad = [r.get("id") for r in records if r.get("split") != "train"]
    if bad:
        raise ValueError(f"train 消费口收到 {len(bad)} 条非 split=='train' 记录"
                         f"（前几个：{bad[:5]}）——split 语义缺失即拒绝入训练分母")
    return records


def train_ledger(manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """汇总 train 轴账面（集/条数/真实字节/通道 + Intern 探查结论），run notes 与报告共用。

    白话：把 train 轴上每个集的账凑成一句能核对的话：真拉了多少字节、装出多少条、走的
    哪个通道；还没拉的集如实报 absent 与 0——底账宁可空着，也不许把没拿到的写成拿到了。
    """
    man = manifest if manifest is not None else registry.load_manifest()
    sets = man.get("sets") or {}
    rows = []
    for pid, pin in registry.REGISTRY.items():
        if pin.axis != TRAIN_AXIS:
            continue
        s = sets.get(pid) or {}
        rows.append({"id": pid, "status": s.get("status", "absent"),
                     "samples": int(s.get("samples") or 0),
                     "bytes": int(s.get("bytes_downloaded") or s.get("bytes") or 0),
                     "qtype_counts": s.get("qtype_counts") or {},
                     "channel": s.get("endpoint") or s.get("source_url") or "-"})
    # 中文那一格单独列出（D3）：C5 的 zh_corpus 要指着它，缺口也必须能单独读出来——
    # 混在总数里就看不出"typed 有了、中文还欠着 CMMLU 那一档"这句实话
    zh_rows = [r for r in rows if r["id"] in registry.ZH_TRAIN_IDS_TUPLE]
    return {"sets": rows, "samples": sum(r["samples"] for r in rows),
            "bytes": sum(r["bytes"] for r in rows),
            "zh_sets": zh_rows, "zh_samples": sum(r["samples"] for r in zh_rows),
            "zh_bytes": sum(r["bytes"] for r in zh_rows),
            "intern_probe": registry.INTERN_TRAIN_PROBE["verdict"],
            "zh_probe": registry.ZH_TRAIN_PROBE["verdict"]}


def score_axis(axis: str, preds: list[Any]) -> dict[str, Any]:
    """用一阶段的现成口径给一条轴出数；口径不在本域重写，只转手。

    白话：质量与把握两轴的算法早就定在一阶段，这里只把答卷递过去、把结果接回来，接不到
    合适的算法就明写"本入口不会算这条轴"，不自己另发明一套判分。
    """
    from sys1.eval import calibration, quality

    if not preds:
        return {"axis": axis, "status": "n/a", "reason": "没有预测行可算"}
    if axis == "quality":
        rep = quality.score_quality(preds)
        return {"axis": axis, "status": "scored", "report": rep,
                "table": quality.format_table(rep)}
    if axis == "calibration":
        rep = calibration.score_calibration(preds)
        return {"axis": axis, "status": "scored", "report": rep,
                "table": calibration.format_table(rep)}
    return {"axis": axis, "status": "n/a", "reason": f"本入口不代打 {axis}（判分件属专域）"}


def run_six(*, axes: tuple[str, ...] | str = "all", predictor: Any = None,
            backend: str = "cpu", data_dir: str | Path | None = None,
            limit: int = 0, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """六轴总入口：先交就绪度，有预测出口就把能算的两轴真算出来。

    白话：一次调用把六条轴排好队，每条轴都占一格——能出分的出分，只有题面的写清"题面在这儿、
    分要等专业判分件"，一样都拿不到的写清是还没拉数据还是压根没注册。参数 backend 只管一件事：
    这些数打算在哪类设备上跑；写 npu 会当场停下来，因为那条路还没接通。
    """
    resolved = scoring.resolve_backend(backend)
    statuses = collect_axes(axes, manifest=manifest)
    result: dict[str, Any] = {"backend": resolved, "axes": statuses, "scored": {},
                              "parity_adjudication": PARITY_ADJUDICATION}
    if predictor is None:
        return result
    preds_cache: dict[str, list[Any]] = {}
    for axis in ("quality", "calibration"):
        if axis not in statuses:
            continue
        records = load_axis_records(axis, data_dir=data_dir, limit=limit)
        if not records:
            continue
        preds = predictor.predict(records) if hasattr(predictor, "predict") else []
        preds_cache[axis] = list(preds)
        statuses[axis].update(score_axis(axis, preds_cache[axis]))
    return result


def format_table(result: dict[str, Any]) -> str:
    """把六轴摘要压成一张等宽表：每轴一行，缺数据的轴照样占行并写明原因。

    白话：把每条轴的名字、状态、拉到几个集、共几条题、其中几条带人工答案、以及"为什么没数"
    排成对齐的几列，人和终端都能直接读。表尾固定带两句：这批数打算在哪类设备上算，以及一致轴
    的裁定是什么——这两句漏掉过一次，就多了一条没人认领的悬空轴。
    """
    lines = [f"{'axis':<12} {'status':<11} {'sets(present/registered)':<26} "
             f"{'samples':>9} {'w/gold':>7} reason", "-" * 118]
    for name in AXES_SIX:
        st = (result.get("axes") or {}).get(name)
        if st is None:
            lines.append(f"{name:<12} {'未点名':<11} {'-':<26} {'-':>9} {'-':>7} "
                         f"这次 --axes 没包含它")
            continue
        got = f"{len(st.get('present') or [])}/{len(st.get('sets') or [])}"
        status = st.get("status", "n/a")
        if st.get("report") is not None:
            status = "scored"
        reason = str(st.get("reason") or "")
        lines.append(f"{name:<12} {status:<11} {got:<26} {st.get('samples', 0):>9} "
                     f"{st.get('with_gold', 0):>7} {reason[:60]}")
    lines.append("-" * 118)
    lines.append(f"backend={result.get('backend')}  （npu 尚未接入：p2-13 sys1.kernels.backends）")
    adj = result.get("parity_adjudication") or {}
    lines.append(f"parity 裁定：{adj.get('verdict', '未裁定')}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """搭命令行：轴、集、后端、是否记录成 run 档案都在这里一次给齐。

    白话：这里摆的是入口全部的旋钮——点哪几条轴、这批数打算在哪类设备上算、要不要真起被测出口
    （给模型目录或服务地址才算分）、数据副本在哪个目录、这次要不要顺手记一页档案。一个都不填
    也能跑：照样出六轴表，只是不出模型分。
    """
    parser = argparse.ArgumentParser(prog="sys1.eval.run",
                                     description="六轴评测调度入口（数据就绪度 + 可算轴真算）",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--axes", default="all", help="逗号分隔的轴名，或 all")
    parser.add_argument("--backend", default="cpu", choices=scoring.BACKENDS,
                        help="这批数打算在哪类设备上跑（npu 尚未接入）")
    parser.add_argument("--model", default="", help="被测本地模型目录（给了才真出模型分）")
    parser.add_argument("--endpoint", default="", help="远端 /v1/systemone 服务地址")
    parser.add_argument("--limit", type=int, default=0, help="每集最多取几题（冒烟用）")
    parser.add_argument("--data-dir", default="", help="评测数据根目录（默认 bench/eval_data）")
    parser.add_argument("--record", action="store_true", help="把这次调度记成一页 run 档案")
    parser.add_argument("--runs-root", default="", help="run 记录本根目录（默认仓库 runs/）")
    parser.add_argument("--run-prefix", default="p2-03", help="run 档案名前缀")
    parser.add_argument("--json", action="store_true", help="改交 JSON（机器读）")
    return parser


def make_predictor(args: argparse.Namespace) -> Any:
    """按参数决定要不要真起一个预测出口；两个都没给就交 None（只走数据就绪度）。

    白话：本域默认不碰模型——不写 --model 也不写 --endpoint 时，这里干脆什么都不建，六轴表
      照样出，只是那两轴停在"有题面没分数"。真要出分时才按一阶段的规矩把出口建起来，规矩
      本身一条没改。
    """
    if not args.model and not args.endpoint:
        return None
    from sys1.eval import predict

    return predict.build_predictor(args)


def data_revision(manifest: dict[str, Any] | None = None) -> str:
    """把"这次的数用的是哪一版数据"压成一行短码，填进 run 档案（P1 留的未定版占位由此填实）。

    白话：档案上原先写着数据来源版本未定，那是数据还没登记时的老规矩。现在每个集钉在哪个版本、
    装出多少条都记在账上了，就压成一行写进档案：三件套各带一个版本短码，再补一句共几集几条。
    日后有人问这分数是用哪批题打的，档案里当场有据可查，不用回去翻聊天记录。
    """
    man = manifest if manifest is not None else registry.load_manifest()
    sets = man.get("sets") or {}
    pins = ",".join(f"{pid}@{registry.REGISTRY[pid].revision}"
                    for pid in registry.THREE_SHEET_IDS if pid in registry.REGISTRY)
    got = [pid for pid, s in sets.items() if s.get("status") in ("fetched", "cached")]
    total = sum(int(sets[pid].get("samples") or 0) for pid in got)
    return f"eval-registry[{pins or 'none'}] sets={len(got)} samples={total}"


def fill_notes_skeleton(run: Any, *, hypothesis: str, observation: list[str]) -> None:
    """把 notes 骨架里的"假设/观察"两行从待填写换成这次的真话（结论行仍由 conclude 产生）。

    白话：记录本发下来时那两行写着"待填写"，那是给人填的空。这一页是数据登记与调度就绪的凭证，
    为什么这么做（假设）与看出了什么数字（观察）都来自账本，不填就等于交一张空白凭证。结论行
    故意不在这里写——它必须走 conclude()，这样 finish() 的"没结论不许收工"这道门还留着。
    """
    text = run.notes_file.read_text(encoding="utf-8")
    obs = "\n".join(f"- {line}" for line in observation)
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("- 假设"):
            line = f"- 假设：{hypothesis}"
        elif line.strip().startswith("- 观察"):
            line = f"- 观察：\n{obs}"
        lines.append(line)
    run.notes_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_report_md(result: dict[str, Any], *, manifest: dict[str, Any] | None = None) -> str:
    """交一页评测报告模板（run 档案的 notes/report.md 都由它渲染，裁定与 n/a 说明在其中）。

    白话：报告不能每次手写——手写就会漏掉"这条轴为什么没数"。模板把四块固定下来：用的是哪版
    数据（真下载字节与条数）、六轴表、一致轴的对象裁定（含报告脚注原句）、以及读表须知（n/a 不等于
    没跑）。以后任何一次 `--record` 都长同一个样子，读者不必猜哪一栏是临时加的。
    """
    man = manifest if manifest is not None else registry.load_manifest()
    sets = man.get("sets") or {}
    tl = train_ledger(man)
    table = format_table(result)
    versions = registry.format_versions_table()
    total_bytes = sum(int(v.get("bytes_downloaded") or v.get("bytes") or 0) for v in sets.values())
    total_samples = sum(int(v.get("samples") or 0) for v in sets.values())
    total_gold = sum(int((v.get("gold_coverage") or {}).get("with_gold") or 0) for v in sets.values())
    adj = result.get("parity_adjudication") or PARITY_ADJUDICATION
    lines = [
        "## 数据底账（真实下载量，非估算）", "", "```", versions, "```", "",
        f"- 数据版本串：`{data_revision(man)}`",
        f"- 累计下载字节：{total_bytes:,}（出自 fetch 的真实网络计数；命中缓存时报的是当初真下载的那份）",
        f"- 登记条数合计：{total_samples:,}（其中带人工答案 {total_gold:,}）",
        f"- train 底账（训练侧消费面，不入六轴评测表）：{len(tl['sets'])} 集 / {tl['samples']:,} 条 / "
        f"真实下载 {tl['bytes']:,} 字节（通道 "
        f"{'、'.join(str(r['channel']) for r in tl['sets'] if r['status'] in ('fetched', 'cached')) or '未拉'}）；"
        f"Intern-Decision train 探查：{tl['intern_probe']}",
        f"- 中文 train 语料（D3 · C5 前置闸之二）：{len(tl['zh_sets'])} 集 / "
        f"{tl['zh_samples']:,} 条 / 真实下载 {tl['zh_bytes']:,} 字节"
        f"（通道 {'、'.join(str(r['channel']) for r in tl['zh_sets'] if r['status'] in ('fetched', 'cached')) or '未拉'}）"
        f"，逐集 {[(r['id'], r['status'], r['samples']) for r in tl['zh_sets']]}——"
        f"条数按集累计，训练口只吃决策信封（原件档 envelope_ready=False，不入分母）；"
        f"CMMLU train 探查：{tl['zh_probe']}",
        f"- 执行后端：`{result.get('backend')}`；npu 为云端占位，接入前必被拒绝（见 p2-13）",
        "", "## 六轴表", "", "```", table, "```", "",
        "## 一致轴裁定（B3 · P1 悬空轴教训回写）", "",
        f"- 裁定：{adj['verdict']}",
        f"- 测什么：{adj['measures']}",
    ]
    lines += [f"- 不测：{x}" for x in adj["not_measured"]]
    lines += ["", f"- 报告脚注（parity 为 n/a 时原样带上）：{adj['na_note']}",
              "", "## 读表须知", "",
              "- `n/a` 不等于没跑：要么是没得测（单一实现路径），要么是数据还没拉，reason 列写清是哪种。",
              "- `data-ready` 只保证题面与真值在盘上、来源可回溯；长文与多模态的判分件分属 p2-07 / p2-08。",
              "- 本域不出模型分：给了 `--model` 或 `--endpoint` 才走一阶段预测出口，口径复用不改写。",
              ""]
    return "\n".join(lines)


def write_run_record(result: dict[str, Any], *, prefix: str = "p2-03",
                     root: str | Path | None = None) -> Path:
    """把这次六轴调度记成一页 run 档案（pin 表 + 数据账 + 轴表 + 裁定全进 notes 与 report.md）。

    白话：开一页记录本，写清这次用的哪一版数据、真走了多少字节流量、登记了多少条题、六轴各自
    是什么状态、一致轴按什么口径裁定，最后把看出了什么落成结论行。骨架由 new_run 发下来，这一
    页只往后面填事实，不动它写好的提示语；结论必须走 conclude()，好让"没结论不许收工"那道门生效。
    """
    from sys1.runs import new_run

    man = registry.load_manifest()
    cfg = {"axes": list((result.get("axes") or {}).keys()), "backend": result.get("backend"),
           "registry_source": str(registry.DATA_DIR / registry.MANIFEST_NAME),
           "data_revision": data_revision(man)}
    run = new_run(f"{prefix}-eval-registry-six-axes", cfg, root=root)
    (run.path / "axes.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    body = format_report_md(result, manifest=man)
    (run.path / "report.md").write_text(body, encoding="utf-8")   # 模板产物单独留一份，报告直接抄
    with run.notes_file.open("a", encoding="utf-8") as fh:   # 骨架后面接，不覆盖 new_run 写好的四件套
        fh.write("\n" + body)
    sets = man.get("sets") or {}
    tl = train_ledger(man)
    fill_notes_skeleton(
        run,
        hypothesis="评测集按版本钉住、可复跑装配并统一登记后，六轴表能在不出模型分的前提下"
                   "自证数据就绪度（缺数据的轴显式 n/a 而非省略）",
        observation=[
            f"实拉 {len([1 for v in sets.values() if v.get('status') in ('fetched', 'cached')])} 集 / "
            f"登记 {sum(int(v.get('samples') or 0) for v in sets.values()):,} 条 / 带人工答案 "
            f"{sum(int((v.get('gold_coverage') or {}).get('with_gold') or 0) for v in sets.values()):,} 条",
            f"真实下载累计 {sum(int(v.get('bytes_downloaded') or v.get('bytes') or 0) for v in sets.values()):,} 字节"
            f"（幂等复跑状态记 cached、字节不被清零或翻倍）",
            "六轴表：" + "、".join(f"{a}={st.get('status')}" for a, st in (result.get('axes') or {}).items()),
            "一致轴按 B3 裁定显式 n/a；npu 后端当场拒绝并指向 p2-13 接入点（不静默回退 CPU）",
            f"train 底账（D1，训练侧消费面，不入六轴评测表）："
            f"{len(tl['sets'])} 集 / {tl['samples']:,} 条 / 真实下载 {tl['bytes']:,} 字节（通道 "
            f"{'、'.join(str(r['channel']) for r in tl['sets'] if r['status'] in ('fetched', 'cached')) or '未拉'}）；"
            f"qtype 逐集 {[(r['id'], r['qtype_counts']) for r in tl['sets'] if r['samples']]}；"
            f"Intern-Decision train 探查结论：{tl['intern_probe']}"
            f"（取证见 registry.INTERN_TRAIN_PROBE）——查无即如实记，不硬造分区凑数",
            f"中文 train 语料底账（D3，C5 前置闸之二）："
            f"{[(r['id'], r['status'], r['samples']) for r in tl['zh_sets']]} / "
            f"真实下载 {tl['zh_bytes']:,} 字节；中文档 id 与考卷 id 逐条互斥"
            f"（档位段 train vs validation/test 写在 id 里，断言见 tests/test_registry.py "
            f"-k split_isolation）；语料缺口上报：{tl['zh_probe']}"
            f"（取证见 registry.ZH_TRAIN_PROBE）——查无即如实记，不拿 dev 档改名凑 train",
        ])
    na = [a for a, v in (result.get("axes") or {}).items() if v.get("status") == "n/a"]
    run.log_metrics(0, axes_requested=len(result.get("axes") or {}), axes_na=len(na),
                    axes_scored=sum(1 for v in (result.get("axes") or {}).values()
                                    if v.get("status") == "scored"))
    run.conclude(f"六轴调度入口就绪；显式 n/a 的轴：{na or '无'}；一致轴按 B3 裁定降级。")
    run.finish()
    return run.path


def main(argv: list[str] | None = None) -> int:
    """CLI 出口：`--axes all` 必须六条都在表上，否则非零退出。

    白话：把前面的零件串成给外部用的一条命令：出表（或出机器读的 json）、需要时记一页档案。
    表印完还要回头核一遍"被点名的轴是不是每条都在表上"，少一条就以失败退出——轴被悄悄省略比
    缺一格数字更糟，因为没人会去数表上该有几行。
    """
    args = build_parser().parse_args(argv)
    try:
        predictor = make_predictor(args)
    except SystemExit as exc:
        print(f"[run] 预测出口建不起来：{exc}")
        return 2
    result = run_six(axes=args.axes, predictor=predictor, backend=args.backend,
                     data_dir=args.data_dir or None, limit=args.limit)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_table(result))
    wanted = AXES_SIX if str(args.axes).strip() == "all" else tuple(
        a.strip() for a in str(args.axes).split(",") if a.strip())
    missing = [a for a in wanted if a not in (result.get("axes") or {})]
    if missing:
        print(f"[run] 轴缺位（被静默省略）：{missing}")
        return 1
    if args.record:
        path = write_run_record(result, prefix=args.run_prefix, root=args.runs_root or None)
        print(f"[run] 档案 -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
