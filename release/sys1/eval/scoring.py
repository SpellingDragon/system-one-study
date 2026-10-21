"""sys1/eval/scoring.py — 决策打分口径与执行后端枚举：把"一份预测怎么算分、能在哪种机器上算"写成一处公共件。

【做什么】
    本文件是 p2-03 到 p2-13 的打分对接面，只管两件事。① 口径：把一条题的人工份额与模型给的
    份额折成 KL / TV / Brier / acc / soft / macro-F1 / ECE / 档位 MAE / within-1 这组数，算法与
    上游 `StartLux-Decision/eval/typed_decisions.py::score` 逐式对齐（上游卡面的 Uniform 行
    0.444/0.381/0.238 就是这组函数的自检锚点）。② 后端：把"在哪儿算这些分"抽象成 `backend`
    参数（cpu / mps / npu 三枚枚举），npu 现在是显式占位、调用即抛 `NotImplementedError`，
    云端 910B 到位后只在这里接一次，调用方（registry / run / p2-13）一行不改。

【怎么做】
    分数全部是纯 stdlib 的浮点运算，不引张量库：`dist_pair()` 先把两份分布对齐成同序两列（缺失
    候选记 0、预测列归一），再交 `kl_given_pred()` / `tv()` / `brier()` 逐题算；宏平均 F1 按
    `(集名, 题名)` 分组、每组各算再平均；ECE 用 top-label 15 桶（与 `sys1.eval.calibration` 的
    15 桶口径同源；JevBench-hard 层是 10 桶，禁止混报，桶数由参数显式带入）。
    `uniform_predictions()` 从 registry 装配好的信封直接产出"均匀猜"的预测行，用来验锚点；
    `score_rows()` 一次吃完整批预测与标准答案，回 `{"all": {...}, "<qtype>": {...}}` 两层层级，
    逐题型字段（choice/noul/score）由信封里的 `qtype` 带出——下游分桶（p2-04 基线、p2-09 中文轨）
    吃的就是这两个字段，不再各自解析题面。
    `resolve_backend()` 只做解析与拒绝：cpu/mps 原样回，npu 抛错并说明"云端 C 轨接入点在此"；
    `auto_backend()` 按本机可见性挑（有 mps 用 mps，否则 cpu），验证一律走 cpu。

【为什么】
    被否方案一：让各域自己抄一份打分公式——同一批预测在两个域算出两个 KL，双基线对照表与
    卡面锚点当场对不上，而这类漂移只会以"某个数怎么跟昨天不一样"的形式暴露。
    被否方案二：把 backend 做成设备字符串直接塞给某个张量库——本域不出模型分（打分对接 p2-13
    与 P1 的 predict 出口），口径件里塞张量依赖会让纯 CPU 的自检也得装全套；故这里只留枚举与
    拒绝点，真正的 NPU 数据搬运归 p2-13 的 backends。
    被否方案三：用硬标签（one-hot）当人工标准答案算 KL——上游的 gold 是多人投票的软分布，换成
    one-hot 后锚点必然复现不出卡面三数，自检熔断器就废了。
    被否方案四：锚点自检写死在测试里不看盘上数据——测试会随数据缺失一起坏；故自检函数从
    registry 装配结果出发，数据没拉时明确报"先 fetch --typed"，而不是给个假绿。
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

#: 上游卡面（typed-decisions revision f7a2487e，test 分割）的 Uniform 行——本域的熔断锚点
CARD_UNIFORM = {"kl": 0.444, "tv": 0.381, "brier": 0.238}
ANCHOR_TOLERANCE = 0.01           # design 定的容差：复现不准就说明 harness 接错，不接受"接近"
EPS = 1e-9                        # 与上游同款的 KL 分母地板值
TOP_LABEL_BINS = 15               # typed 的 ECE 桶数（JevBench-hard 是 10 桶，禁止混报）
HARD_BINS = 10

#: 打分入口允许的执行后端；npu 是 D6 v3 的云端占位（现在必被拒绝，接入点在 resolve_backend）
BACKENDS = ("cpu", "mps", "npu")


def resolve_backend(name: str) -> str:
    """校验并回显后端名：cpu/mps 原样通过，npu 抛 `NotImplementedError`。

    白话：问一句"这批分数打算在哪儿算"。说本地机器就照办；说昇腾那张卡，今天还没接上，
    就当场把话挑明并指出接的位置，免得跑了一小时才发现用的是另一条路。
    """
    if name not in BACKENDS:
        raise ValueError(f"未知 backend {name!r}，可选：{BACKENDS}")
    if name == "npu":
        raise NotImplementedError(
            "backend='npu' 尚未接入：昇腾 910B 的数据搬运归 p2-13 的 sys1.kernels.backends "
            "（云端 C 轨），届时本函数改为返回实际设备描述符，调用方无需改动。"
        )
    return name


def auto_backend(*, prefer_cpu: bool = True) -> str:
    """按本机可见性挑一个能用的后端名（默认偏 cpu；验证一律 cpu）。

    白话：不给后端名时该用哪个？这台机器上苹果的统一内存能就用它，否则就用普通 CPU。
    加了 prefer_cpu 这个开关，是为了让"跑验证"的人能强制走最朴素那条路，别去抢正在长跑的卡。
    """
    if prefer_cpu:
        return "cpu"
    try:
        import torch

        if getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)():
            return "mps"
    except Exception:  # noqa: BLE001  装了却导不出、或压根没装，都退到 cpu
        return "cpu"
    return "cpu"


def dist_pair(keys: list[str], pred: dict[str, Any], gold: dict[str, Any]) -> tuple[list[float], list[float]]:
    """把两份 {候选: 份额} 按同一候选序折成两列浮点（缺的记 0，预测列归一，人工列照录）。

    白话：模型报的和人工标的都写成"哪个选项几成"这样的本子，两本子上的选项名要对齐才能比。
    这里按题目原本的选项顺序重抄一遍：谁没写就补个零，模型那份加不满一的补平（它常因四舍五入
    差一丝），人工那份不改动——那是标准答案，动它就等于改考卷。
    """
    p = [max(0.0, float(pred.get(k, 0.0) or 0.0)) for k in keys]
    g = [max(0.0, float(gold.get(k, 0.0) or 0.0)) for k in keys]
    z = sum(p)
    if z <= 0.0:
        p = [1.0 / len(keys)] * len(keys)
    else:
        p = [x / z for x in p]
    return p, g


def kl_gpred(pred: list[float], gold: list[float]) -> float:
    """KL(gold ‖ pred)：逐候选 `b·log(b/max(a,EPS))` 求和（与上游同式，只在 b>0 处累加）。"""
    return sum(b * math.log(b / max(a, EPS)) for a, b in zip(pred, gold) if b > 0)


def tv(pred: list[float], gold: list[float]) -> float:
    """全变差距离 = 两列逐格差的绝对值之和的一半（上游同款）。"""
    return 0.5 * sum(abs(a - b) for a, b in zip(pred, gold))


def brier(pred: list[float], gold: list[float]) -> float:
    """Brier 分 = 逐候选差的平方和（上游同款，越小越好）。"""
    return sum((a - b) ** 2 for a, b in zip(pred, gold))


def top_label_ece(items: list[tuple[float, bool]], *, bins: int = TOP_LABEL_BINS) -> float:
    """按最高把握那一档的值分桶，回各桶"平均把握减平均命中率"的加权和。

    白话：把题目按模型自己报的把握高低切成几格，每格里比一下"平均有多自信"和"平均对多少"，
    差得越远说明这份把握越虚。桶数要显式给：不同集的口径不一样，混在一格子里算出来的数没法解释。
    """
    if not items:
        return 0.0
    buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for conf, hit in items:
        buckets[min(bins - 1, int(conf * bins))].append((conf, hit))
    n = len(items)
    return sum(len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(1.0 if h else 0.0 for _, h in b) / len(b))
               for b in buckets.values())


def _macro_f1(groups: dict[Any, list[tuple[int, int]]]) -> float:
    """按题分组算宏平均 F1（每组各算逐候选 F1 再平均；与上游 `score()` 里的 f1 段同式）。"""
    per_q: list[float] = []
    for pairs in groups.values():
        labels = sorted({pred for pred, _ in pairs} | {gold for _, gold in pairs})
        f1s: list[float] = []
        for lab in labels:
            tp = sum(1 for pr, go in pairs if pr == lab and go == lab)
            fp = sum(1 for pr, go in pairs if pr == lab and go != lab)
            fn = sum(1 for pr, go in pairs if pr != lab and go == lab)
            f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        if f1s:
            per_q.append(sum(f1s) / len(f1s))
    return sum(per_q) / len(per_q) if per_q else 0.0


def rows_from_envelopes(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 registry 的信封行摊平成打分用的窄表（id/qtype/k/keys/gold/task）。

    白话：装配好的文件里一行套着一整包信息，判分时只想看几样：这是哪道题、属于哪一类、
    有哪几个选项、人工各给了几成。这一层把包装拆开摊平，下游（p2-04 的基线表、p2-09 的中文
    曲线）直接吃这张窄表，不必各自去拆包，也拆不出两种样子。
    """
    out: list[dict[str, Any]] = []
    for rec in lines:
        sample = rec.get("sample") or {}
        questions = sample.get("questions") or {}
        targets = sample.get("targets") or {}
        for qid, q in questions.items():
            keys = [str(k) for k in (q.get("criteria") or {})]
            out.append({"id": f"{rec.get('id')}/{qid}", "task": rec.get("task") or rec.get("qtype"),
                        "qtype": q.get("type", "choice"), "keys": keys, "k": len(keys),
                        "gold": {k: float(v) for k, v in (targets.get(qid) or {}).items()},
                        "instructions": q.get("instructions", "")})
    return out


def uniform_predictions(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """由窄表产出"均匀猜"的预测（每题每个候选同分），用于复现卡面锚点。"""
    return {r["id"]: {k: 1.0 / r["k"] for k in r["keys"]} for r in rows}


def score_rows(rows: list[dict[str, Any]], preds: dict[str, dict[str, float]],
               *, bins: int = TOP_LABEL_BINS) -> dict[str, dict[str, float]]:
    """一批预测对一批标准答案打分，回 `{"all": {...}, "<qtype>": {...}, "<集>": {...}}`。

    逐题先算 KL/TV/Brier/命中，再按"总体 / 题型 / 所属集"三个口径分别平均；`mf1` 按题分组做
    宏平均，`ece` 用 top-label 分桶（桶数由参数带入，禁止跨集混算），`mae`/`within1` 只对
    档位题（score）有意义，其他题型不给数。

    白话：把"答得准不准、把握实不实、差得远不远"三件事一次算齐，并且分三本账记：总体一本、
    按题型一本、按数据来源一本。这样后面看报告的人能问出"是选择题不行还是档位题不行"，
    而不是一句总分糊过去；少一题预测也不装作没发生，会写进 missing 那一栏。
    """
    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    ece_items: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    f1_groups: dict[str, dict[Any, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    missing = 0
    for r in rows:
        pred = preds.get(r["id"])
        if pred is None:
            missing += 1
            continue
        p, g = dist_pair(r["keys"], pred, r["gold"])
        amax = max(range(len(p)), key=lambda i: (p[i], -i))
        lab = max(range(len(g)), key=lambda i: (g[i], -i))
        hit = amax == lab
        conf = p[amax]
        vals = {"kl": kl_gpred(p, g), "tv": tv(p, g), "brier": brier(p, g),
                "acc": 1.0 if hit else 0.0, "soft": g[amax]}
        if r["qtype"] == "score":
            gold_score = r["gold_score"] if "gold_score" in r else sum(i * v for i, v in enumerate(g))
            vals["mae"] = abs(sum(i * v for i, v in enumerate(p)) - gold_score)
            vals["within1"] = 1.0 if abs(amax - lab) <= 1 else 0.0
        for group in ("all", r["qtype"], r["task"] or "unknown"):
            for key, value in vals.items():
                agg[group][key].append(value)
            ece_items[group].append((conf, hit))
            f1_groups[group][r["id"].rsplit("/", 1)[0]].append((amax, lab))
    out: dict[str, dict[str, float]] = {}
    for group, metrics in agg.items():
        out[group] = {k: sum(v) / len(v) for k, v in metrics.items()}
        out[group]["mf1"] = _macro_f1(f1_groups[group])
        out[group]["ece"] = top_label_ece(ece_items[group], bins=bins)
        out[group]["n"] = float(len(metrics["kl"]))
    out["all"]["missing"] = float(missing)
    out["all"]["bins"] = float(bins)
    return out


def anchor_selfcheck(rows: list[dict[str, Any]], *, tolerance: float = ANCHOR_TOLERANCE) -> dict[str, Any]:
    """Uniform 锚点自检：均匀猜一遍这批题，KL/TV/Brier 必须落在卡面值的容差内。

    白话：harness 接错没有别的征兆——分数会"看着还行"。所以拿一个已知答案的笨办法当试纸：
    每道题都均匀瞎猜，这时算出来的三个数应当正好是卡面上那行 Uniform。对不上就说明题面拆错
    了或者公式抄错了，这时后面所有模型分都不能信。

    返回 `{"measured":..., "expected":..., "ok":..., "n":..., "detail":...}`；rows 为空时
    `ok=False` 并附一句"先 fetch"，不给假绿。
    """
    if not rows:
        return {"ok": False, "n": 0, "measured": {}, "expected": dict(CARD_UNIFORM),
                "detail": "没有可判分的题：先跑 python -m sys1.eval.registry fetch --typed"}
    measured = score_rows(rows, uniform_predictions(rows))["all"]
    got = {"kl": measured["kl"], "tv": measured["tv"], "brier": measured["brier"]}
    detail = {k: round(abs(got[k] - v), 6) for k, v in CARD_UNIFORM.items()}
    return {"ok": all(detail[k] <= tolerance for k in detail), "n": int(measured["n"]),
            "measured": {k: round(v, 4) for k, v in got.items()}, "expected": dict(CARD_UNIFORM),
            "abs_diff": detail, "tolerance": tolerance}


def load_assembled(path: str) -> list[dict[str, Any]]:
    """读 registry 装配好的 jsonl（一行一封信封），回原始字典列表。

    白话：把装配好的那摞卡片从磁盘上一张张读进内存，不做任何加工——加工口径全在 `rows_from_envelopes`
    那一层，这样同一份副本能被打分、分桶、抽样几种用途共用，各读各改容易改出几个版本。
    """
    import json
    from pathlib import Path

    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
