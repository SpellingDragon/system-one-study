"""sys1/eval/quality.py — 决策质量轴：按 `题型 × 候选数` 分桶，每桶拿自己的随机基线量。

【做什么】
    把预测行与人工标准答案逐题对上，分桶报数：choice 与 noul 报"名次对不对"的比例，
    score 另外报"档位差多少"的平均偏差与"差一档以内"的比例。每个桶旁边必列两样东西：
    这一桶闭眼乱选的随机基线，以及实测与基线相差几个百分点。

【怎么做】
    桶的钥匙是 `(题型, 候选个数)` 这一对，且**绝不跨 k 合并**：三个里蒙一个有一分之一
    的机会，二十个里蒙一个只有二十分之一，把两档平均成一格，就等于让三选一的成绩去
    冒充二十选一的及格线（GUIDE §6 写死的防假通过口径）。基线一律现取
    `sys1.calibrate.bucket_baselines`，不在本域重算第二份公式。
    三个指标的取法：命中 = 模型名次（胆量分最大那格）与人工名次是否为同一格；score 的
    平均偏差取"模型档 − 人工档"的绝对值再按桶求平均，差一档以内取绝对值 ≤1 的份额。
    档位数字优先从候选代号里读（"1"…"9"、星级都是数字串），读不出来才退回候选在队列里
    的位次，并在报告里以 `value_source` 标注用的是哪一种——两档数不能混在同一个平均里。
    Δ 的符号约定按指标方向定：命中率类是"实测 − 基线"（正数是好消息，写成百分点）；
    平均偏差类是"基线 − 实测"（同样正数是好消息，单位是档位）。

【为什么】
    质量轴唯一的价值是"比底线强多少"，所以基线必须与实测同桶同尺；把基线写成常数或
    把 Δ 统一成一种符号，都是让弱模型混进合格线的通道。
    被否方案一：给一个跨 k 的总准确率做"头条数字"——三选一的题在集合里占多数时，总准确率
    会被 1/3 那条线主导，二十选一桶的真实水平被抹平；要总量必须由读者自己按桶加权，
    故本域只交分桶表，不交合并数。
    被否方案二：score 也只看名次对不对——评分档是有顺序的，把"预测 5 星、真值 1 星"与
    "预测 2 星、真值 1 星"记成同一种错，等于丢掉这门题唯一的难度信息。
    被否方案三：把人工份额的期望值当预测档（份额加权平均）——那样同一份预测既能算出
    偏差又能算出命中率，两套口径混在一桶里，跨报告对不上数；本轴统一取"名次对应的档"。
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sys1.calibrate import bucket_baselines

#: 一阶段合格线的参考间距（GUIDE §6：该桶基线 + ≥10pp 才算"过程信号"过关）
REFERENCE_MARGIN_PP = 10.0
_METRIC_BY_QTYPE = {"score": "mae"}


def _hit(pred: int, gold: int) -> int:
    """名次是否对上（对=1，错=0）。"""
    return int(pred == gold)


def _option_value(options: Sequence[str], index: int) -> float:
    """候选对应的"档"：代号是数字串就用数字，否则用在队列里的位次（从 0 起）。"""
    text = str(options[index])
    try:
        return float(text)
    except ValueError:
        return float(index)


def _value_source(options: Sequence[str]) -> str:
    """这一桶的档位是从代号数字读的还是从位次推的（两档口径不许混进同一个平均）。"""
    try:
        [float(str(o)) for o in options]
    except ValueError:
        return "rank"
    return "numeric-code"


def score_quality(preds: Sequence[Any], *, margin_pp: float = REFERENCE_MARGIN_PP) -> dict[str, Any]:
    """按 `(qtype, k)` 分桶出质量表；每桶带实测、随机基线、Δ 与是否超基线/超参考线。

    入参是预测出口的富样本（有 `pred / gold / k / qtype / options`）；空集合回空表而非报错，
    因为"这一桶一条题都没有"是常见事实，必须原样写在报告里而不是被平均数藏掉。

    白话：把题目按"哪一类、有几个候选"分堆，一堆一堆地数对了几道；每一堆旁边都摆一条
    闭眼乱选能得的分数，再看看实际比这条线高出多少。三个候选一堆、二十个候选一堆，
    各量各的，绝不把两堆合成一堆——不然蒙对三选一的战绩会被当成二十选一也合格。
    """
    baselines = bucket_baselines()
    buckets: dict[tuple[str, int], list[Any]] = {}
    for pred in preds:
        buckets.setdefault((pred.qtype, int(pred.k)), []).append(pred)

    rows: list[dict[str, Any]] = []
    for (qtype, k) in sorted(buckets):
        group = buckets[(qtype, k)]
        hits = [_hit(p.pred, p.gold) for p in group]
        acc = sum(hits) / len(hits)
        base = baselines.get(qtype, {}).get(k)
        base_val = float(base.value) if base is not None else (1.0 / k if k else 0.0)
        metric = _METRIC_BY_QTYPE.get(qtype, "accuracy")
        row: dict[str, Any] = {
            "qtype": qtype, "k": k, "n": len(group),
            "accuracy": round(acc, 6), "hits": int(sum(hits)),
            "baseline_metric": metric, "baseline": round(base_val, 6),
            # Δ 的符号按"正数=好消息"定：命中率类看高多少，平均偏差类看低多少
            "delta_pp": round((acc - base_val) * 100.0, 4),
            "beats_baseline": acc > base_val,
            "meets_reference": (acc - base_val) * 100.0 >= margin_pp,
            "baseline_source": "sys1.calibrate.bucket_baselines",
            "margin_pp_reference": margin_pp,
        }
        if qtype == "score":
            sources = {_value_source(p.options) for p in group}
            if len(sources) != 1:
                raise ValueError(f"score 桶 k={k} 内档位口径不唯一 {sources}：禁止混进同一个平均")
            src = sources.pop()
            errs = [abs(_option_value(p.options, p.pred) - _option_value(p.options, p.gold)) for p in group]
            mae = sum(errs) / len(errs)
            row.update({
                "value_source": src,
                "mae": round(mae, 6),
                "within_1": round(sum(int(e <= 1.0) for e in errs) / len(errs), 6),
                # 平均偏差是"越小越好"，故 Δ 取基线减实测，仍保持"正数=好消息"
                "mae_delta_vs_baseline": round(base_val - mae, 6),
                "beats_baseline": mae < base_val,
                "delta_pp": round((mae - base_val) * 100.0, 4),
            })
        rows.append(row)

    return {
        "axis": "quality",
        "buckets": rows,
        "n_predictions": len(preds),
        "empty_buckets": not rows,
        "honest_note": "分桶呈现：任何两个 k 都未合并；无跨 k 加权总分（合并由读者按 n 自行加权）",
    }


_HEADER = "qtype   k     n    acc     base    Δ        within1  mae     超基线  超参考线(+{margin}pp)"
_ROW = "{qtype:<7} {k:<3} {n:<4} {acc:<7} {base:<7} {delta:<8} {w1:<8} {mae:<7} {beat:<6} {ref:<6}"


def format_table(report: dict[str, Any], *, margin_pp: float = REFERENCE_MARGIN_PP) -> str:
    """把分桶表压成一张等宽文本表（表头写清每一列口径，空桶如实留白而不藏进平均数）。

    白话：一行一个堆：先说这是什么题、几个候选、共几道，再摆实际对了几成、闭眼乱选能对
    几成、高出多少个百分点；评分那类还多两列，分别报"差一档以内的占几成"和"平均差几档"。
    最后两列是一句人话结论：有没有强过乱选、有没有强过参考线。
    """
    lines = [_HEADER.format(margin=margin_pp)]
    for row in report["buckets"]:
        is_score = row["qtype"] == "score"
        # 基线列跟着指标走：命中率的基线是 1/k（noul 恒 0.5），评分档的基线是均匀猜测的平均偏差
        base = row["baseline"]
        delta = f"{row['mae_delta_vs_baseline']:+.3f}档" if is_score else f"{row['delta_pp']:+.2f}pp"
        lines.append(_ROW.format(
            qtype=row["qtype"], k=row["k"], n=row["n"],
            acc=f"{row['accuracy']:.4f}", base=f"{base:.4f}", delta=delta,
            w1=(f"{row['within_1']:.4f}" if "within_1" in row else "-"),
            mae=(f"{row['mae']:.4f}" if "mae" in row else "-"),
            beat="是" if row["beats_baseline"] else "否",
            ref="是" if row["meets_reference"] else "否",
        ))
    if len(lines) == 1:
        lines.append("(空表：没有任何一个桶拿到样本——如实记录，不用平均数遮掩)")
    if any(row["qtype"] == "score" for row in report["buckets"]):
        lines.append("注：score 行的 base/Δ 走均匀猜测平均偏差口径（单位＝档，越小越好）；"
                     "acc/within1 两列仍是份额")
    lines.append(report.get("honest_note", ""))
    return "\n".join(lines)


__all__ = [
    "REFERENCE_MARGIN_PP",
    "format_table",
    "score_quality",
]
