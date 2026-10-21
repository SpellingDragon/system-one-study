"""learning/s3_calibrate.py — S3 温度校准 CLI：SFT 模型 → 逐类拟温度 → 写回 decision_config.json。

【做什么】
    接过上一步微调好的模型目录，在一批带软标签的决策样本上，按题型各找一个"把胆量分
    同除之后最说到做到"的倍数（温度），把量出来的"把握几成、结果几成"差距（ECE）连同
    拟合前后一并记进实验账，最后把这份 `题型 → 倍数` 表补进模型随附的决策配置温度栏。

【怎么做】
    四步。① 编码：复用 s2 的转写样本编码（渲染排成死文字、编成编号、折出与字母列等长
    且和为 1 的软标签），把校准集摊成一批批可组窗的样本。② 读数：只在读出位取分数——
    复用 `sys1.decision.option_scores` 拿每个样本末位那串数字点 26 字母行得到候选分，
    逐样本收齐（原始分、软标签、题型）三列对齐清单。③ 拟合：把三列交给
    `sys1.calibrate.fit_temperature` 做网格穷举（逐类独立定倍数，样本不足某类自动池化
    并在报告里标注），一次拿到温度表与逐类 ECE 前后。④ 落盘：把温度写进模型目录那份
    decision_config.json 的 `temperatures` 栏、把校准摘要（前后 ECE、池化情况、run-id）
    并进 `calibration` 栏，其余字段一律不动；ECE 前后同步入 run-notebook。

【为什么】
    校准与微调分家，是因为"选对答案"（SFT 学 argmax）与"报准把握"（温度管置信度标尺）
    是两把独立的尺子；温度是对分值的严格单调缩放，不换 argmax，故绝不该和改权的 SFT 混在
    一步里做。被否方案一：把温度并进 SFT 损失一起梯度优化——多一套超参、结果不可复现，
    且与 p1-05 已定的网格穷举同构口径分叉；网格法无梯度、同输入必同输出。
    被否方案二：s3 里自造一套"取末位、点字母行、算 softmax 份额"——与 p1-07 读出层两份
    真源必然漂移，故强制复用 `option_scores` 与 p1-05 的 `fit_temperature`（本域零平行实现）。
    被否方案三：在小样本题型上也硬拟独立温度——把噪声当规律、跨次乱跳；遵 p1-05 规则
    样本不足即池化并如实记 `pooled=True`，报告不掩盖。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

# 仓库根 + 脚本同目录引导：既保证 `python learning/s3_calibrate.py` 从任意 cwd 可跑
# （editable 安装在部分环境未把仓库根挂上 sys.path），也让本域复用兄弟脚本 s2 的编码件。
_REPO_ROOT = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from s2_decision_sft import (
    DEFAULT_TOKENIZER_DIR,
    PAD_TOKEN,
    encode_example,
    group_by_k,
    load_records,
    make_batch,
)

from sys1.calibrate import fit_temperature
from sys1.decision import option_scores
from sys1.lang import bpe
from sys1.model import DECISION_CONFIG_FILE, Decoder
from sys1.runs import new_run

CALIBRATE_VERSION = "dmlaya_s3_calibrate_v1"


# ---------------------------------------------------------------- 读数（复用读出层，零平行实现）
@torch.no_grad()
def collect_scores(model: Decoder, examples: list[dict[str, Any]], pad_id: int,
                   batch_size: int) -> tuple[list[list[float]], list[list[float]], list[str]]:
    """把编码样本按候选数分桶组批、只在读出位取分，收齐 (原始分列, 软标签列, 题型列) 三张对齐清单。

    白话：让模型把每道题只读末尾那一格，抄下这题几个候选各自的胆量分；再把人工标的"各候选
    该占几成"和这题归哪一类一并记下。三样按同一顺序排齐，回头一类一类去找"该同除几"的倍数。

    :returns: 三个等长列表，逐样本对齐——fit_temperature 要的 (logits, targets) 加逐行 qtypes。
    """
    model.eval()
    logits: list[list[float]] = []
    targets: list[list[float]] = []
    qtypes: list[str] = []
    for rows in group_by_k(examples).values():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            batch = make_batch(chunk, pad_id)
            last_hidden = model(batch["input_ids"], batch["attn_mask"])
            # 末位读出唯一入口仍是 option_scores——与 SFT/引擎同一份真源，杜绝校准口径漂移。
            scores = option_scores(last_hidden, model.lm_head.weight, batch["letter_ids"], batch["lengths"])
            logits.extend(scores.float().cpu().tolist())
            targets.extend(batch["targets"].cpu().tolist())
            qtypes.extend(ex["qtype"] for ex in chunk)
    return logits, targets, qtypes


# ---------------------------------------------------------------- 温度写回决策配置
def merge_decision_config(model_dir: str | Path, report: Any, run_id: str, bins: int) -> Path:
    """把拟合出的温度表与校准摘要并回模型目录的 decision_config.json，保留其余字段不动。

    白话：翻开随模型那张说明书，找到"温度"一栏，把刚算好每一类该同除几的数填进去；再把
    "校准"一栏从"还没校"改写成"已校"，附上前后每一类说到做到的差距、是否跟着大盘走的标注
    和这次实验的编号。其余栏目一个字不改，免得把决策域写好的读点位口径给弄乱了。

    :returns: 写回后的 decision_config.json 路径。
    """
    path = Path(model_dir) / DECISION_CONFIG_FILE
    config: dict[str, Any] = {}
    if path.is_file():
        config = json.loads(path.read_text(encoding="utf-8"))
    config["temperatures"] = {q: float(t) for q, t in report.temperatures.items()}
    config["calibration"] = {
        "status": "fitted",
        "version": CALIBRATE_VERSION,
        "run_id": run_id,
        "bins": bins,
        "pooled_temperature": float(report.pooled_temperature),
        "pooled_types": list(report.pooled_types),
        "ece": {
            q: {"before": f.ece_before, "after": f.ece_after, "delta": f.ece_delta,
                "improved": f.improved, "temperature": f.temperature, "n": f.n, "pooled": f.pooled}
            for q, f in report.by_type.items()
        },
    }
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- 主体
def run_calibrate(args: argparse.Namespace) -> dict[str, Any]:
    """校准主体：载 SFT 模型 → 校准集读末位分 → 逐类拟温度 → ECE 入 run → 温度写回决策配置。

    白话：先接过上一步微调好的模型，拿一批带人工份额的题目让它逐题读末尾那格；按题型各找
    一个"说到做到"的除数，把调整前后"嘴上几成把握、结果几成对"的差距记进实验账本；最后把
    各题型的倍数补进模型那份说明书的温度栏，供引擎以后读数时同除，使置信度名副其实。

    :returns: 摘要字典（含温度表、逐类 ECE 前后、产出路径与 run-id）。
    """
    torch.manual_seed(args.seed)
    tokenizer = bpe.load(args.tokenizer)
    letter_map = bpe.check_tokenizer(tokenizer)
    pad_id = int(tokenizer.token_to_id(PAD_TOKEN) or 0)

    model = Decoder.load(args.model_dir)
    records = load_records(args.data)
    examples = [encode_example(tokenizer, letter_map, r) for r in records]
    logits, targets, qtypes = collect_scores(model, examples, pad_id, args.batch_size)

    report = fit_temperature(logits, targets, qtypes=qtypes, min_samples=args.min_samples, bins=args.bins)

    run = new_run("s3-calibrate", _run_config(args, len(examples)))
    run.log_metrics(0, **_ece_metrics(report))
    path = merge_decision_config(args.model_dir, report, run.run_id, args.bins)

    summary = _summarize(report, args, len(examples))
    run.conclude(summary["conclusion"])
    run.finish()

    _print_summary(run.run_id, str(path), report)
    return {"run_id": run.run_id, "decision_config": str(path), "temperatures": report.temperatures,
            "report": report, "summary": summary}


def _run_config(args: argparse.Namespace, n_examples: int) -> dict[str, Any]:
    """把这次校准的可复现要素攒成 run config（含校准集来源，如实入 data_revision）。"""
    return {
        "stage": "s3-calibrate",
        "model_dir": str(args.model_dir),
        "tokenizer": str(args.tokenizer),
        "data": str(args.data),
        "batch_size": args.batch_size,
        "min_samples": args.min_samples,
        "bins": args.bins,
        "seed": args.seed,
        "n_calibration_samples": n_examples,
        # 校准集来源（父 design D8 防泄漏）：只用与 SFT 训练 id 互斥的集，且绝不参与改权
        "data_revision": {
            "calibration": "held-out transcribed set (NOT typed-decisions test, NOT SFT train)",
            "leak_policy": "temperature fit only; no gradient/weight update on this data",
        },
    }


def _ece_metrics(report: Any) -> dict[str, float]:
    """把逐类 ECE 前后与温度摊成一组可入 notebook 的标量指标（键名对 run 曲线友好）。"""
    metrics: dict[str, float] = {}
    for q, f in report.by_type.items():
        metrics[f"ece_before_{q}"] = round(f.ece_before, 6)
        metrics[f"ece_after_{q}"] = round(f.ece_after, 6)
        metrics[f"ece_delta_{q}"] = round(f.ece_delta, 6)
        metrics[f"temperature_{q}"] = round(f.temperature, 4)
    return metrics


def _summarize(report: Any, args: argparse.Namespace, n_examples: int) -> dict[str, Any]:
    """把逐类 ECE 前后/是否改善与池化情况攒成一句如实结论（校准验收看 Δ≤0，非绝对线）。"""
    per_type = {
        q: {"temperature": round(f.temperature, 4), "ece_before": round(f.ece_before, 4),
            "ece_after": round(f.ece_after, 4), "improved": f.improved, "pooled": f.pooled}
        for q, f in report.by_type.items()
    }
    all_improved = all(f.improved for f in report.by_type.values())
    pooled = list(report.pooled_types)
    conclusion = (
        f"s3 校准完成：n={n_examples} min_samples={args.min_samples} bins={args.bins}；"
        f"温度表={ {q: round(t, 4) for q, t in report.temperatures.items()} }；"
        f"逐类 ECE 前后={per_type}；池化题型={pooled or '无'}；"
        f"{'各类 ECE 均未变差（Δ≤0）' if all_improved else '存在 ECE 变差类（样本小/弱模型，如实记录）'}。"
    )
    return {"per_type": per_type, "temperatures": dict(report.temperatures),
            "all_improved": all_improved, "pooled_types": pooled, "conclusion": conclusion,
            "n_calibration_samples": n_examples}


def _print_summary(run_id: str, config_path: str, report: Any) -> None:
    """把关键数字打到 stdout（smoke 日志一眼可读，也供端到端测试抓取）。"""
    print(f"[s3] run-id          : {run_id}")
    print(f"[s3] decision config : {config_path}")
    print(f"[s3] pooled          : {list(report.pooled_types) or '无'}")
    for q, f in report.by_type.items():
        print(f"[s3] {q}: T={f.temperature:.4f} ECE {f.ece_before:.4f} -> {f.ece_after:.4f} "
              f"(Δ={f.ece_delta:+.4f}, improved={f.improved}, pooled={f.pooled})")


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    """搭出 s3 CLI 参数表（模型目录/词表/校准集/批规模/池化阈值/ECE 桶数），供 --help 与 main 共用。

    白话：把能敲的选项和默认值一次列清楚，让人敲一下 --help 就看到一份说明书：接哪个微调好的
    模型目录、用哪张表、拿哪份带标题目来校、一捆多大、某类题少到多少条就跟着大盘走、把握分几格数。
    """
    parser = argparse.ArgumentParser(
        prog="s3_calibrate.py",
        description="温度校准：载 SFT 模型目录 → 校准集逐类拟温度 → 写回 decision_config.json（ECE 前后入 run）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", required=True, help="S2 产出的模型目录（含 decision_config.json，温度将写回此处）")
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER_DIR), help="S0 自训 BPE 产物目录")
    parser.add_argument("--data", required=True, help="校准集 jsonl（一行一个 {id,task,sample}，须与 SFT 训练 id 互斥）")
    parser.add_argument("--batch-size", type=int, default=32, help="每个候选数分桶内读数批的规模")
    parser.add_argument("--min-samples", type=int, default=200, help="某类低于此样本量则池化到全局温度")
    parser.add_argument("--bins", type=int, default=15, help="top-label ECE 的等宽桶数（上游口径）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（读数可复现）")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：解析参数 → 校验模型目录/校准集存在 → 跑 run_calibrate → 退出码 0。

    白话：命令行跑起来先核对两样东西在不在——要写回温度的模型目录、用来校的那份带标题目；
    缺任何一样就停下报错，绝不拿半截输入硬跑（免得把不存在的目录当模型、或校出一份空温度）。
    都在就走完整条"读数—拟倍数—记账—写回说明书"，正常收工；smoke 档不设"必须 ECE 变好才放行"
    的硬闸门，变没变好都在摘要与 run 记录里如实报出。
    """
    args = build_parser().parse_args(argv)
    if not Path(args.model_dir).is_dir():
        raise SystemExit(f"模型目录不存在：{args.model_dir}")
    if not Path(args.data).is_file():
        raise SystemExit(f"校准集不存在：{args.data}")
    run_calibrate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
