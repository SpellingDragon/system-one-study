"""sys1/eval/calibration.py — 把握轴：同一份预测，报"调之前"与"调之后"各差多少。

【做什么】
    量"嘴上说几成把握、结果真有几成对"之间的差距（ECE），并把两份数一起交出去：一份是
    胆量分原样折算的把握（调整前），一份是按随附说明书里的题型倍数折算的把握（调整后），
    再加一列两者之差。分数小＝说到做到；差值为负＝这一步真的把把握调准了。

【怎么做】
    全部算术交给 `sys1.calibrate.ECE`（top-label 口径、等宽桶、按桶占比加权），本域一行
    公式都不重写。两份把握从同一把末位胆量分折出：调整前用原始分数（等价于倍数按 1.0），
    调整后用"同除该题型倍数"之后的分数——倍数可以来自调用方给的表，也可以直接采信预测
    出口从模型目录带出来的那一份（表不给时逐样本用自带 `temperature`）。配置里没有这一
    类倍数时如实标 `uncalibrated` 并按 1.0 计，绝不悄悄冒充"调过"。目标一律带人工份额，
    故口径固定为质量加权（`weighted=True`）：命中份额取目标压在获胜项上的质量、桶权重取
    该行总质量；报告里以 `mode` 与 `bins` 写死，避免与硬命中口径混报。
    分组只按题型（同一题型里三选与二十选并存是常态，宽度不等的行由尺子按行处理，不填零）；
    另附全体行合并的一份 ECE 作对照，`scope` 字段把它与逐类数区分开。除 ECE 外还顺手报
    "平均把握 vs 实际命中"两个数，让可靠性图的结论不用读者自己心算。

【为什么】
    把握轴是评测出口里唯一"数字方向有含义"的轴：只报调整后一份数，读者无从判断这一步
    到底是调准了还是调歪了；只报 Δ 又不承认绝对水平。所以 before/after/Δ 三列必须同屏。
    被否方案一：直接抄 s3 校准报告里的数——那是校准集上的数，评测集上重新量一遍才是
    本轴的独立证据（同一份权重、两份数据，抄数等于自己给自己打分）。
    被否方案二：把"没倍数"的题型按 1.0 算完不打标注——和"拟过且恰好消息是 1.0"混成一谈，
    报告就分不清"没调"与"调过"；故逐类带 `uncalibrated` 布尔与 `temperature_source` 出处。
    被否方案三：ECE 用硬命中口径（目标当 0/1）——人工份额常带小数（多人投票的分布照录），
    硬折成对/错会把"部分正确"当全错，与 p1-05 的"两口径禁止混报"冲突。
    被否方案四：题型分组也按候选数拆桶——把握只看"报几成、对几成"，与候选个数无关，
    拆开后每桶样本寥寥，15 个等宽桶大部分是空的，ECE 反而被小样本噪声顶高。
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from sys1.calibrate import DEFAULT_BINS, ECE
from sys1.decision import to_probs

#: 报告里写死的口径标注（与 typed-decisions 上游一致：top-label、质量加权、等宽 15 桶）
EVAL_MODE = "top-label/weighted"


def _temperature_for(pred: Any, temperature: float | dict[str, float] | None) -> float:
    """把"这一条该同除几"算清楚：数字全体同除、字典按题型查、None 采信样本自带倍数。"""
    if isinstance(temperature, dict):
        return float(temperature.get(pred.qtype, getattr(pred, "temperature", 1.0) or 1.0))
    if temperature is None:
        return float(getattr(pred, "temperature", 1.0) or 1.0)
    return float(temperature)


def _rows(preds: Sequence[Any], *, temperature: float | dict[str, float] | None
          ) -> tuple[list[list[float]], list[list[float]]]:
    """把富样本折成 (把握行, 目标行) 两张等长清单；给了倍数就照倍数重折，没给用现成那份。

    白话：一摞是"每道题各候选分到几成"，另一摞是"人工觉得各候选该占几成"，两摞按同一
    顺序排齐好一起送进量把握的尺子。要看"调整前"就把倍数按一（原样），要看"调整后"就
    照说明书上这一类的倍数同除一遍再折。
    """
    probs: list[list[float]] = []
    for p in preds:
        if temperature is None:
            probs.append([float(v) for v in p.probs])
            continue
        t = _temperature_for(p, temperature)
        scaled = torch.tensor([float(v) for v in p.scores], dtype=torch.float64) / max(t, 1e-12)
        probs.append([float(v) for v in to_probs(scaled.unsqueeze(0), 1.0)[0]])
    targets = [[float(v) for v in p.target] for p in preds]
    return probs, targets


def _one_group(preds: Sequence[Any], *, temperature: float | dict[str, float] | None,
               bins: int) -> dict[str, Any]:
    """一组样本的一份 ECE（含平均把握、实际命中与样本数）；空组回 None 而不是回 0。

    白话：把一摞题送进尺子，量出"嘴上几成、结果几成对"的平均差距；顺手报两个辅助数——
    模型平均有多自信、按人工份额算平均对了几成，让人不必自己看图心算。一组题都没有
    时给 None，因为"零道题的差距是零"是句谎话。
    """
    if not preds:
        return {"n": 0, "ece": None}
    probs, targets = _rows(preds, temperature=temperature)
    ece = ECE(probs, targets, bins, weighted=True)
    conf = sum(max(row) for row in probs) / len(probs)
    hit = sum(t[max(range(len(p)), key=lambda j: p[j])] for t, p in zip(targets, probs)) / len(probs)
    return {"n": len(preds), "ece": round(float(ece), 6), "mean_confidence": round(conf, 6),
            "accuracy_mass": round(hit, 6)}


def score_calibration(preds: Sequence[Any], *, bins: int = DEFAULT_BINS,
                      table: dict[str, float] | None = None) -> dict[str, Any]:
    """逐题型的 ECE 前后与 Δ；`table` 缺省采信预测出口从模型目录带出来的倍数。

    入参每项需带 `scores / probs / target / qtype / temperature / uncalibrated`（即富样本）。
    空集合回一份带 `empty=True` 的报告而不是抛错——"评测集里没这类题"是事实，必须能被记录。

    白话：按题型分堆，每堆各量两遍"嘴上几成、结果几成对"的差距：一遍照模型报出的原始
    胆量分直接折成把握，一遍先按说明书上这一类的倍数同除再折。两遍的差就是这一步调温
    换来的改进；差为负说明把握更名副其实了，为零说明倍数恰等于没调。
    """
    by_type: dict[str, list[Any]] = {}
    for p in preds:
        by_type.setdefault(p.qtype, []).append(p)

    rows: list[dict[str, Any]] = []
    for qtype in sorted(by_type):
        group = by_type[qtype]
        temps = {float(p.temperature) for p in group}
        used = temps.pop() if len(temps) == 1 else None      # 逐样本倍数不一致时不假装有一个统一值
        override = None if table is None else table.get(qtype)
        applied = override if override is not None else used
        before = _one_group(group, temperature=1.0, bins=bins)
        after = _one_group(group, temperature=(override if override is not None else None), bins=bins)
        uncalibrated = all(p.uncalibrated for p in group) and override is None
        delta = (after["ece"] - before["ece"]) if before["ece"] is not None and after["ece"] is not None else None
        rows.append({
            "qtype": qtype, "n": len(group),
            "bins": bins, "mode": EVAL_MODE,
            "temperature": (round(float(applied), 6) if applied is not None else None),
            "temperature_source": ("default(1.0)" if uncalibrated else
                                   ("caller-table" if override is not None else "decision_config")),
            "uncalibrated": uncalibrated,
            "ece_before": before["ece"], "ece_after": after["ece"],
            "ece_delta": (round(float(delta), 6) if delta is not None else None),
            "improved": (delta <= 0.0 if delta is not None else None),
            "mean_confidence_before": before["mean_confidence"],
            "mean_confidence_after": after["mean_confidence"],
            "accuracy_mass_before": before["accuracy_mass"],
            "accuracy_mass_after": after["accuracy_mass"],
        })

    all_before = _one_group(preds, temperature=1.0, bins=bins)
    all_after = _one_group(preds, temperature=table, bins=bins) if preds else {"n": 0, "ece": None}
    return {
        "axis": "calibration",
        "mode": EVAL_MODE,
        "bins": bins,
        "by_type": rows,
        "overall": {"scope": "all-rows-pooled", "before": all_before, "after": all_after},
        "empty": not rows,
        "honest_note": "口径固定为 top-label + 质量加权（与硬命中口径不混报）；"
                       "before 恒取倍数按一的那一份，逐类倍数出处随报告带出；"
                       "overall 是全体行合并的一份，与逐类数不可互推",
    }


def format_table(report: dict[str, Any]) -> str:
    """把把握轴压成一张等宽文本表（Δ 列直接写符号，"是否改善"给一句人话）。

    白话：一行一类题：先报这道题用了哪个倍数、是说明书里拟来的还是压根没调；再摆
    调整前后的差距与差值，最后一列写"没变差"或"变差了"。差距越小越诚实，差值为负才说明
    这一步确实把胆量调到了配得上战绩的位置。表末另附全体行合并的一份作对照。
    """
    lines = ["qtype     n    倍数(出处)              ECE前   ECE后   Δ         改善"]
    for row in report["by_type"]:
        temp = "-" if row["temperature"] is None else f"{row['temperature']:.4f}"
        src = "未调" if row["uncalibrated"] else "已调"
        improved = "-" if row["improved"] is None else ("是" if row["improved"] else "否")
        lines.append(
            f"{row['qtype']:<9} {row['n']:<4} {temp}({src},{row['temperature_source']:<14}) "
            f"{_fmt(row['ece_before'])}  {_fmt(row['ece_after'])}  {_fmt(row['ece_delta'], signed=True)}  {improved}"
        )
    if len(lines) == 1:
        lines.append("(空表：评测集里一道题都没落到这一轴上)")
    overall = report.get("overall") or {}
    bef, aft = overall.get("before") or {}, overall.get("after") or {}
    delta = None
    if bef.get("ece") is not None and aft.get("ece") is not None:
        delta = aft["ece"] - bef["ece"]
    lines.append(f"{'pooled':<9} {bef.get('n', 0):<4} ({report.get('scope_note', 'all rows'):<22}) "
                 f"{_fmt(bef.get('ece'))}  {_fmt(aft.get('ece'))}  {_fmt(delta, signed=True)}")
    lines.append(f"口径 {report.get('mode')} × {report.get('bins')} 桶；{report.get('honest_note', '')}")
    return "\n".join(lines)


def _fmt(value: float | None, *, signed: bool = False) -> str:
    """统一的数值排版（None 写成 '-'，带符号时显式给正负号）。"""
    if value is None:
        return "-"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


__all__ = [
    "EVAL_MODE",
    "format_table",
    "score_calibration",
]
