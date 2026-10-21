"""multimodal.py — MMBench-CN 图文决策评测器（p2-08 B2，含无图对照列）。

【做什么】
    对装配好的图文决策行做末位字母读出打分，交逐题记录与 acc；同一接口
    走纯文本对照路（加/去图对照、G5 归因的取数口）。

【怎么做】
    打分只依赖"前向函数"注入（真 HF 模型或假引擎皆可）：从末位 logits
    取候选字母行 → softmax 成份额 → argmax 判分；无图路由 mm.py 同口径
    装配，两路只差像素。

【为什么】
    读出契约与 decision/ 完全同构（字母行摘取 softmax），评测器不许自带
    第二套读出——"整词 logits 平均"式臆造方案被否（与训练口径漂移，
    对照即失真）。
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import torch


def letter_probs(logits_last: torch.Tensor, letter_ids: Sequence[int]) -> list[float]:
    """白话：末位 logits 上按候选字母编号摘行、摊成一锅份额（温度 1 口径）。

    输入：logits_last（一行向量）、letter_ids（候选字母的 token 号）。
    输出：与 letter_ids 同序的概率列表（和为 1）。
    """
    picks = logits_last[list(letter_ids)].float()
    return torch.softmax(picks, dim=-1).tolist()


def evaluate_mm(rows: Sequence[dict[str, Any]], encode: Callable[[dict], dict],
                forward: Callable[[dict], torch.Tensor], letter_id_of: Callable[[str], int],
                limit: int | None = None) -> dict[str, Any]:
    """白话：一批行过"装配→前向→字母读出"三段流水线，回逐题记录与 acc 总账。

    输入：rows（含 gold/letter_keys 边车）、encode（行→模型输入）、
    forward（输入→末位 logits 一行）、letter_id_of（字母→token 号）。
    输出：{"n","acc","results"}；空批 acc=None 不装数。
    """
    res = []
    for i, row in enumerate(rows):
        if limit is not None and i >= limit:
            break
        enc = encode(row)
        probs = letter_probs(forward(enc), [letter_id_of(ch) for ch in row["letter_keys"]])
        pred = row["letter_keys"][max(range(len(probs)), key=lambda j: probs[j])]
        res.append({"id": row.get("id"), "gold": row["gold"], "pred": pred,
                    "probs": probs, "has_image": row.get("has_image"),
                    "ok": pred == row["gold"]})
    acc = (sum(r["ok"] for r in res) / len(res)) if res else None
    return {"n": len(res), "acc": acc, "results": res}
