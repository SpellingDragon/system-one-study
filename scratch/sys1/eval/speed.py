"""sys1/eval/speed.py — 耗时轴：单请求串行走一遍，报 P50/P95 与"每秒读多少 token"。

【做什么】
    把同一批评测样本一条一条、前后不重叠地送进预测出口，量每条从"交题目"到"拿到各
    候选几成把握"花了多少毫秒；再把这些耗时折成两个分位（一半请求快于 P50、九成九请求
    快不到 P95 的那个位置由 P95 守着）与一个吞吐数（每秒能吃进多少个字的题目）。
    报告里必须同时写明这次是在什么设备、什么位宽、哪条后端路上测的。

【怎么做】
    ① 串行：每条单独调一次 `predictor.predict([record], batch_size=1)`，绝不并批也绝不
    并发——一并批就变成"摊薄后的均摊耗时"，与真实单次请求不是一回事。② 预热豁免：
    头 `warmup` 条只跑不记，躲开首次载内核、建调度、页缓存冷启的那几笔一次性开销。
    ③ 两份耗时同屏：`e2e_ms`（本域自己掐表，含抄题与折算）是对外主口径；`readout_ms`
    取预测行里自带的 `ms`（出口只包前向+读数那一段），两列并排放，读者看得见包装成本
    占了多少。④ 分位取法用现成的 `torch.quantile`（线性插值口径），主报 P50/P95，
    P99/最小/最大作附表；**不报 mean 作头条**，因为一次磁盘换页就能把平均数拖花。
    ⑤ 吞吐按"总入参 token ÷ 总耗时"给一个整体数，另给逐条比值的中位数，两列分开标名，
    避免被当成同一件事。⑥ 样本不足 `min_samples`（默认 30，design 的风险条款）时照常
    出数，但 `meets_min_samples=False` 与提示一起写进报告——分位数在 20 条以下本就是噪声。

【为什么】
    速度是零生成方案唯一"可能比对手慢"的地方，也是最容易被机器噪声骗过去的地方：
    被否方案一：报均值——均值对长尾没有抵抗力，一次后台换页就能把"看着很快"变成假账，
    而线上体验恰恰由尾部决定，故 P95 与 P50 同级、均值只进附列。
    被否方案二：把 warm-up 计入——第一次要编译内核/首次触碰权重，那笔钱是一次性的，
    计入就等于用一次冷启污染整条分布，故预热样本单独计数、不进分位。
    被否方案三：报"生成 tok/s"——一阶段末位一次读数即出全部候选份额，压根没有逐字生成
    （零生成红线 `output_tokens=0`），把它写成生成速度就是虚构指标；故这里的 tok/s 明确
    标注是"输入吞吐（read-once prefill）"，并附 `output_tokens=0` 自证。
    被否方案四：设备与位宽不写——同一份权重 cpu/mps、fp32/fp16 差一个数量级，
    不带出处的耗时数字过不了跨机对比，也不配进对照表。
"""
from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from typing import Any

import torch

#: 低于这个条数分位数就是噪声（design 风险条款：固定 warm-up + N≥30）
MIN_SAMPLES = 30
#: 默认预热条数（不计入分布，只在报告里单列）
WARMUP = 3


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    """一组耗时的分位数（P50/P95/P99 + 最小/最大/均值），线性插值口径。"""
    t = torch.tensor([float(v) for v in values], dtype=torch.float64)
    q = torch.quantile(t, torch.tensor([0.5, 0.95, 0.99], dtype=torch.float64))
    return {
        "p50": float(q[0]), "p95": float(q[1]), "p99": float(q[2]),
        "min": float(t.min()), "max": float(t.max()), "mean": float(t.mean()),
    }


def _tokens_of(predictor: Any, record: dict[str, Any]) -> int:
    """一条样本有多少个"入参 token"（编码后的编号串长度；测不到就回 0 而不猜）。"""
    try:
        enc = predictor.encode([record]) if hasattr(predictor, "encode") else None
    except Exception:  # noqa: BLE001  取不到长度只影响吞吐列，不该拖垮整条耗时测量
        return 0
    return len(enc[0].input_ids) if enc else 0


def measure_speed(predictor: Any, records: Sequence[dict[str, Any]], *, warmup: int = WARMUP,
                  min_samples: int = MIN_SAMPLES) -> dict[str, Any]:
    """逐条串行计时，交一份带设备与精度标注的耗时报告（P50/P95 为主口径）。

    参数 `predictor` 只要有 `predict([record], batch_size=1)` 与 `describe()` 即可，
    本地模式与远端模式都收；`warmup` 条只跑不记，`min_samples` 条以下照常出数但如实标注。

    白话：一道题一道题地问，问完掐一次表，前几次的问答只当热身不算数。把所有秒表读数
    排成一队，取正中间那个和九成那个当结论——中间那位代表日常手感，九成那位代表偶尔
    的卡顿；再把题目总字数除以总时间，得到"一秒钟能吃进多少字的题目"。哪些热身、
    在什么机器上测的、有没有算生成速度，都写在报告里。
    """
    recs = list(records)
    if not recs:
        return {"axis": "speed", "n": 0, "empty": True,
                "honest_note": "没有样本可测：耗时轴不出数，也不猜一个默认值"}

    desc = predictor.describe() if hasattr(predictor, "describe") else {}
    e2e: list[float] = []
    readout: list[float] = []
    row_tokens: list[int] = []
    warmed = 0
    for idx, rec in enumerate(recs):
        t0 = time.perf_counter()
        preds = predictor.predict([rec], batch_size=1)
        elapsed = (time.perf_counter() - t0) * 1000.0
        if idx < warmup:                       # 热身：载内核/首次触权重的一次性开销不进分布
            warmed += 1
            continue
        e2e.append(elapsed)
        readout.append(float(preds[0].ms) if preds else elapsed)
        row_tokens.append(_tokens_of(predictor, rec))

    q_e2e = _quantiles(e2e)
    q_read = _quantiles(readout)
    total_ms = sum(e2e)
    tokens = sum(row_tokens)
    per_row = [t / ms * 1000.0 for t, ms in zip(row_tokens, e2e) if ms > 0 and t > 0]
    report: dict[str, Any] = {
        "axis": "speed",
        "mode": desc.get("mode", "unknown"),
        "device": desc.get("device", "unknown"),           # cpu / mps / remote——必注
        "dtype": desc.get("dtype", "unknown"),             # 位宽与精度模式——必注
        "backend": desc.get("backend", "unknown"),         # tilelang / torch_eager（同一份代码两条路）
        "serial": True,                                    # 单请求串行，不并批不并发
        "warmup_excluded": warmed,
        "n": len(e2e),
        "min_samples_reference": min_samples,
        "meets_min_samples": len(e2e) >= min_samples,
        "e2e_ms": {k: round(v, 3) for k, v in q_e2e.items()},
        "readout_ms": {k: round(v, 3) for k, v in q_read.items()},
        # 两列吞吐口径不同：aggregate 是总字数÷总时间，median-per-row 是逐条比值取中位
        "tokens_in": tokens,
        "throughput_tps_aggregate": round(tokens / total_ms * 1000.0, 3) if total_ms > 0 else 0.0,
        "throughput_tps_median_per_row": round(statistics.median(per_row), 3) if per_row else None,
        "output_tokens": 0,                                 # 零生成红线：一次读数即出全部候选
        "throughput_semantics": "input-tokens read once (no autoregressive decoding)",
        "honest_note": "主口径为 P50/P95（不报均值为头条）；热身样本已剔除；"
                       "设备/位宽/后端随报告带出，跨机对比必须连这三项一起看",
    }
    if not report["meets_min_samples"]:
        report["honest_note"] += f"；样本仅 {len(e2e)} 条 < {min_samples}，分位数噪声大，只当参考"
    return report


def format_table(report: dict[str, Any]) -> str:
    """把耗时轴压成两行摘要（第一行是分位，第二行是吞吐与出处标注）。

    白话：给人一眼看完的两句：第一句说"热身丢了几条、正式量了几条、一半请求快过多少
    毫秒、九成快过多少毫秒"；第二句说"这是在哪台机器、什么位宽、哪条后端路上量的，
    一秒能吃多少字的题目，以及确认没有逐字生成"。样本不够时第二句末尾会带一句提醒。
    """
    if report.get("empty"):
        return "speed    (空：没有样本可测，不出数也不猜)"
    e, r = report["e2e_ms"], report["readout_ms"]
    flag = "" if report["meets_min_samples"] else f"  ⚠ n<{report['min_samples_reference']}"
    head = (f"speed     serial n={report['n']} (warmup-{report['warmup_excluded']})  "
            f"P50={e['p50']:.2f}ms P95={e['p95']:.2f}ms P99={e['p99']:.2f}ms "
            f"[readout P50={r['p50']:.2f} P95={r['p95']:.2f}]{flag}")
    detail = (f"          device={report['device']} dtype={report['dtype']} backend={report['backend']}  "
              f"tok/s={report['throughput_tps_aggregate']:.1f} (aggregate input-only, output_tokens=0)")
    return f"{head}\n{detail}"


__all__ = [
    "MIN_SAMPLES",
    "WARMUP",
    "format_table",
    "measure_speed",
]
