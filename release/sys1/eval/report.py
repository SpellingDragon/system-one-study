"""sys1/eval/report.py — 四轴汇总出口：一次调用吃一份预测行，交一张四轴表 + 一个评测 run。

【做什么】
    把 quality / calibration / speed / parity 四条轴串成一次评测：读一批题、跑一遍模型、
    出四份数、并在终端把四轴摘要连排打出来（这就是 repro 脚本要的"四轴摘要表"）。同时
    把这次评测本身记成一页实验档案（runs/ 里一个新 run 目录），里面带回被测模型的来源
    run 号——报告里的每个数因此都能指回"哪次实验、哪份权重、哪个代码版本"。

【怎么做】
    ① 建预测出口（复用 `predict.build_predictor` 的本地/远端二选一），溯源体检在载权重
    之前完成，不过就直接非零退出。② 一次读数拿富样本，四条轴全从这同一份分母出：
    质量与把握直接吃富样本；耗时轴挑 `--speed-samples` 条**计时**、热身条数另加（保证分
    位数有足够有效读数）；一致轴只挑前 `--parity-samples` 条把内核路与纯 torch 路各走一遍。
    ③ 判定（`verdict`）
    只由三件事决定：溯源不是豁免来的、四轴都真的出了数、一致轴名次没有换人。质量与把握
    是**呈现值**，不参与判定——tiny 档的小模型质量本来就贴着基线，拿它当复现脚本的通过
    条件会把"链路是否通"与"模型好不好"两件事混成一件事。④ 落盘：`--record`（默认开）时
    用 `sys1.runs.new_run` 开目录，四份指标写 `report.json`、四轴表写 `report.md`、
    关键数追加进 `metrics.jsonl`，最后 conclude + finish 把结论行补齐。
    一致轴在 CPU 机器上会走入口层的普通写法（`backend=torch_eager`）：一致率照样算，但
    表里明写"本轮未开启 Metal 方言，方言验收见 @mps 用例"，绝不把回退当通过。

【为什么】
    被否方案一：让 repro_p1.sh 自己拼四次 CLI 再用 awk 凑表——四轴的入参口径（同一份
    预测行、同一批样本）在 shell 里凑不出强一致，一旦两次调用取的数据不同，四轴数字就
    不是同一分母上的；汇总必须在一个进程里一次做完。
    被否方案二：把评测结果并进被测模型那一页档案（写回 s2/s3 的 run 目录）——评测自身
    也是一次实验，混记会让"谁的数字"重新变得说不清；故评测自己开新 run，并在 config 里
    带上被测 run 号，形成链条而不是覆盖。
    被否方案三：判定里加"质量必须超基线 10pp"——一阶段合格线是对**真训练档**的要求，
    复现脚本跑的是 tiny 冒烟档，用它当退出码会把"评测链路坏了"与"这次只训了 20 步"
    混成同一个红灯，反而看不出该修哪个。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sys1.eval import calibration, parity, predict, quality, speed
from sys1.runs import new_run

#: 四轴的呈现顺序（终端摘要与 md 表都按这个次序）
AXES = ("quality", "calibration", "speed", "parity")


def run_eval(pred_model: Any, records: list[dict[str, Any]], *, device: str = "auto",
             speed_samples: int = speed.MIN_SAMPLES, parity_samples: int = 12,
             axes: tuple[str, ...] = AXES, temperatures: dict[str, float] | None = None,
             head_weight: Any = None, batch_size: int = 8) -> dict[str, Any]:
    """在一份预测行上跑齐四轴，交 `{axes, verdict, provenance, header}` 的汇总字典。

    参数 `pred_model` 是已建好的预测出口（本地或远端）；`temperatures` 缺省用出口自带的
    倍数表。返回里的 `verdict.passed` 决定 CLI 退出码，`axes` 按轴名存各轴的原始报告。

    白话：题目读一遍、模型答一遍，然后从同一份答卷上量四件事——答得准不准、嘴上几成
    把握配不配、一次要花多少毫秒、换条算法路子答案会不会换人。最后给一句总评：来路
    查得到、四件事都量出来了、名次没换人，就算这次评测出口是通的。
    """
    preds = pred_model.predict(records, batch_size=batch_size)
    desc = pred_model.describe()
    out: dict[str, Any] = {"axes": {}, "tables": {}}

    if "quality" in axes:
        rep = quality.score_quality(preds)
        out["axes"]["quality"] = rep
        out["tables"]["quality"] = quality.format_table(rep)
    if "calibration" in axes:
        # temperatures 只作显式覆盖；不给就采信预测出口随样本带出来的那份（出处标 decision_config）
        rep = calibration.score_calibration(preds, table=temperatures)
        out["axes"]["calibration"] = rep
        out["tables"]["calibration"] = calibration.format_table(rep)
    if "speed" in axes:
        rep = speed.measure_speed(pred_model, records[: speed_samples + speed.WARMUP])
        out["axes"]["speed"] = rep
        out["tables"]["speed"] = speed.format_table(rep)
    if "parity" in axes:
        encs = pred_model.encode(records[:parity_samples]) if hasattr(pred_model, "encode") else []
        if encs:
            hw = head_weight if head_weight is not None else getattr(pred_model, "head_weight_cpu", None)
            rep = parity.run_parity(pred_model.model, encs, device=device, head_weight=hw)
        else:                                       # 远端模式没有本地权重可对拍，如实记空
            rep = {"axis": "parity", "skipped": "endpoint 模式无本地权重", "total": 0,
                   "consistent": 0, "argmax_agreement": 0.0, "passed": True, "backend": "n/a",
                   "device": desc.get("device", "remote"), "external_ops": [], "blockers": {},
                   "compiled_keys": [], "dialect_compiles_this_run": 0, "max_abs_err_scores": 0.0}
        out["axes"]["parity"] = rep
        out["tables"]["parity"] = parity.format_table(rep) if rep.get("total") else \
            f"parity   跳过：{rep.get('skipped', '无可对拍的本地样本')}"

    prov = desc.get("provenance") or {}
    empty_axes = [name for name, rep in out["axes"].items() if not _has_data(name, rep)]
    provenance_ok = prov.get("source") == "endpoint" or not prov.get("exempt")
    out["provenance"] = prov
    out["header"] = desc
    out["n_predictions"] = len(preds)
    out["verdict"] = {
        "passed": bool(out["axes"]) and provenance_ok and not empty_axes
                  and _parity_ok(out["axes"].get("parity")),
        "provenance_ok": bool(provenance_ok),
        "criteria": ["溯源不是豁免来的（--allow-missing-run-id 出的数按规则不算通过）",
                     "被点名的轴都真的出了数（没有空轴）",
                     "一致轴 argmax 名次未换人"],
        "exempt": bool(prov.get("exempt")),
        "empty_axes": empty_axes,
        "dialect_verified": (out["axes"].get("parity") or {}).get("backend") == "tilelang",
        "note": "质量/把握为呈现值，不参与退出码判定（tiny 档模型质量贴基线属预期）",
    }
    return out


def _has_data(name: str, rep: dict[str, Any]) -> bool:
    """这一轴到底有没有拿到数（每轴的空表标志字段名不同，集中在这里判一次）。"""
    if name == "quality":
        return bool(rep.get("buckets"))
    if name == "calibration":
        return not rep.get("empty") and bool(rep.get("by_type"))
    if name == "speed":
        return bool(rep.get("n")) and not rep.get("empty")
    if name == "parity":
        return bool(rep.get("total")) or bool(rep.get("skipped"))
    return bool(rep)


def _parity_ok(rep: dict[str, Any] | None) -> bool:
    """一致轴的判定：跳过也算过，跑了就必须名次全对。"""
    if not rep or rep.get("skipped") or not rep.get("total"):
        return True
    return bool(rep.get("passed"))


def format_summary(result: dict[str, Any]) -> str:
    """把四轴摘要与判定连排成一段终端文本（repro 脚本要的那张表）。

    白话：先说这份数是打哪来的（哪个模型目录、哪次实验、哪个代码版本、在什么设备上），
    再把四张摘要表上下摞起来，最后一行给一句总评：三条判定各自成不成、以及这次有没有
    真的用上机器模具。没用上模具时这里会多一句提醒，免得把回退路径当成方言验收。
    """
    head = result["header"]
    prov = result["provenance"]
    title = (f"eval report  mode={head.get('mode')} run-id={prov.get('run_id')} "
             f"commit={(prov.get('commit') or '')[:12]} exempt={prov.get('exempt')}")
    subtitle = (f"             device={head.get('device')} dtype={head.get('dtype')} "
                f"backend={head.get('backend')} n={result['n_predictions']}")
    lines = ["=" * 78, title, subtitle]
    chain = prov.get("chain") or {}
    if chain:
        lines.append(f"             chain={chain}")
    for name in AXES:
        if name in result["tables"]:
            lines.append("-" * 78)
            lines.append(result["tables"][name])
    v = result["verdict"]
    lines.append("-" * 78)
    lines.append(f"verdict    passed={v['passed']}  criteria={'; '.join(v['criteria'])}")
    if not v.get("dialect_verified", False):
        lines.append("           ⚠ 本轮一致轴未走 Metal 方言（回退路径），方言验收以 @mps 用例为准")
    lines.append("=" * 78)
    return "\n".join(lines)


def write_record(result: dict[str, Any], *, root: str | Path | None = None) -> Path:
    """把这次评测本身记成一页 run 档案（四件套 + report.json + report.md），返回目录。

    白话：评测自己也是一次实验，得单独开一页档案：档案里写明这次测的是哪个 run 的权重、
    用了什么设备、几条题、判过没判过，再把四轴原始数与那张连排表各存一份，最后在笔记页
    留一行结论。这样报告里任何一个数都能顺着这页档案倒回被测的那次训练。
    """
    prov = result["provenance"]
    cfg = {
        "stage": "eval",
        "evaluated_run_id": prov.get("run_id"),
        "evaluated_commit": prov.get("commit"),
        "mode": result["header"].get("mode"),
        "device": result["header"].get("device"),
        "backend": result["header"].get("backend"),
        "n_predictions": result["n_predictions"],
        "axes": list(result["axes"]),
        "passed": result["verdict"]["passed"],
    }
    ctx = new_run("eval-report", cfg, root=root)
    ctx.path.mkdir(parents=True, exist_ok=True)
    (ctx.path / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                          encoding="utf-8")
    (ctx.path / "report.md").write_text(format_summary(result) + "\n", encoding="utf-8")
    ctx.log_metrics(0, **_headline_metrics(result))
    ctx.conclude(f"评测出口四轴跑通：passed={result['verdict']['passed']}，"
                 f"被测 run={prov.get('run_id')}，判定详情见 report.md")
    return ctx.finish()


def _headline_metrics(result: dict[str, Any]) -> dict[str, float]:
    """挑几个能画线的标量写进 metrics.jsonl（键带轴名与桶名，避免同 step 撞名）。"""
    out: dict[str, float] = {}
    q = result["axes"].get("quality") or {}
    for row in q.get("buckets", []):
        out[f"acc_{row['qtype']}_k{row['k']}"] = row["accuracy"]
        out[f"base_{row['qtype']}_k{row['k']}"] = row["baseline"]
    cal = result["axes"].get("calibration") or {}
    for row in cal.get("by_type", []):
        out[f"ece_before_{row['qtype']}"] = row["ece_before"]
        out[f"ece_after_{row['qtype']}"] = row["ece_after"]
    sp = result["axes"].get("speed") or {}
    for key in ("p50", "p95"):
        if (sp.get("e2e_ms") or {}).get(key) is not None:
            out[f"e2e_{key}_ms"] = sp["e2e_ms"][key]
    pa = result["axes"].get("parity") or {}
    if pa.get("total"):
        out["parity_agreement"] = pa["argmax_agreement"]
        out["parity_max_abs_err"] = pa["max_abs_err_scores"]
    return out


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    """搭出四轴汇总的 CLI 参数表（被测对象、评测集、各轴样本量、是否记 run）。

    白话：把命令行能敲的选项摆清楚——测哪个模型目录或哪个服务地址、题目从哪份文件读、
    耗时轴量几条、对拍轴比几条、要不要给这次评测单独开一页档案。默认值按"笔记本上能
    跑完"的规模给，不默认全量。
    """
    parser = argparse.ArgumentParser(
        prog="report.py",
        description="四轴汇总出口：一次调用出 quality/calibration/speed/parity 四份数与总判定。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=None, help="被测模型目录（runs/<id>/model）")
    parser.add_argument("--endpoint", default=None, help="被测服务地址（…/v1/systemone）")
    parser.add_argument("--data", required=True, help="评测集 jsonl（一行一个 {id,task,sample}）")
    parser.add_argument("--tokenizer", default=None, help="词表目录（默认自动定位）")
    parser.add_argument("--device", default="auto", help="计算设备（auto/cpu/mps）")
    parser.add_argument("--batch-size", type=int, default=8, help="读数每捆样本数")
    parser.add_argument("--limit", type=int, default=0, help="评测集只取前 N 条（0=全取）")
    parser.add_argument("--axes", default=",".join(AXES), help=f"要跑的轴（逗号分隔，可选 {','.join(AXES)}）")
    parser.add_argument("--speed-samples", type=int, default=speed.MIN_SAMPLES, help="耗时轴计时条数（热身条数另加，不计入分位）")
    parser.add_argument("--parity-samples", type=int, default=12, help="一致轴样本条数")
    parser.add_argument("--no-record", action="store_true", help="不为本次评测开 run 档案")
    parser.add_argument("--record-root", default=None, help="run 档案根目录（默认仓库 runs/）")
    parser.add_argument("--allow-missing-run-id", action="store_true",
                        help="无溯源也放行（仅冒烟；此判定会让 verdict.passed=False）")
    parser.add_argument("--timeout", type=float, default=30.0, help="远端模式单请求超时秒数（透传 predict）")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：建出口 → 跑四轴 → 打表 → 记 run → 按判定给退出码。

    白话：命令行跑起来后，先查这份权重来路——查不到就一句提示加退出码 2，一道题都不测；
    带着豁免标记硬要测也行，但总评判按规则算不过。来路查清后把题目读一遍、四件事各量
    一遍，终端连排打出四张摘要表，最后把这次评测本身记成一页档案。判定全满足退出码 0，
    差一条就是 1——复现脚本据此决定这次闭环算不算成。
    """
    args = build_parser().parse_args(argv)
    ns = argparse.Namespace(model=args.model, endpoint=args.endpoint, tokenizer=args.tokenizer,
                            device=args.device, allow_missing_run_id=args.allow_missing_run_id,
                            timeout=args.timeout)
    try:
        predictor = predict.build_predictor(ns)  # 内含溯源体检：查不到来源就拒评
    except predict.MissingProvenanceError as exc:
        print(f"[report] 拒绝出数：{exc}", file=sys.stderr)
        return 2
    records = predict.load_records(args.data)
    if args.limit:
        records = records[: args.limit]
    axes = tuple(a.strip() for a in args.axes.split(",") if a.strip())
    result = run_eval(predictor, records, device=args.device, speed_samples=args.speed_samples,
                      parity_samples=args.parity_samples, axes=axes, batch_size=args.batch_size)
    print(format_summary(result))
    if not args.no_record:
        path = write_record(result, root=args.record_root)
        print(f"[report] run record -> {path}")
    return 0 if result["verdict"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
