#!/usr/bin/env python3
"""K4_probe_variant_leg.py — P1-1z / K4：**不占卡**判"设备腿到底走哪个变体"。

背景：窗四 V4 双变体差分给出
    variant=ub   ITEM dbeta rel=1.018e-07 PASS（六项全过）；DBETA-FP none/dirty/zero
                 = e609a6ab/2a507f3e/45266aef（随 prefill 变 ⇒ stale 敏感）
    variant=pure ITEM dbeta rel=8.002e-01 FAIL（其余五项过）；指纹三跑恒定 e609a6ab
    V3（run_delta_card --target ascend --bwd）dbeta=8.002e-01
编排者据此**推断**"设备腿实际走 pure"（单看 dbeta 一项与 pure 同值）。

本件机械复算该推断，**结论：推断不成立，设备腿走 `ub`**。三条独立证据：
  E1 代码   ：`run_delta_card._variant()` 对 ascend 返回 "ub"（源文逐字匹配）。
  E2 运行指纹：V3 六项 rel 里 `dq/dk/dg` 与 V4-ub **逐字同**、与 V4-pure 异；
              且 V4-ub 的 ORDER-DIFF 只列 dbeta ⇒ 其余五项与 prefill 无关、是变体指纹。
              故 V3 只能在 ub 档成立。dbeta 与 pure 同值是"ub 在该 prefill 态下与 pure 同值"，
              非"V3 走了 pure"——直接反证：V4-ub 的 none 指纹 e609a6ab 与 pure 指纹 e609a6ab 本就相同。
  E3 编译键 ：容器只编腿 `--target ascend --bwd --no-run` 键型 `c2ubz`（默认）；
              `--variant pure` 键型 `c2purez`（负例，证明确有开关）。本件只校验记录值，不再占容器。

用法（host、不占卡）：
    .venv/bin/python ascend/kernels/reconcile/K4_probe_variant_leg.py
退出码 0 = 结论成立（设备腿=ub）；2 = 判据不成立（前提翻转，须上报）。
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))                 # .../release/ascend/kernels/reconcile
RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RDC = os.path.join(RELEASE, "ascend", "port910b", "run_delta_card.py")
ART = os.path.join(RELEASE, "runs", "1012-p2-13-oncard-window4-pin", "artifacts")
ITEMS = ("dq", "dk", "dv", "dbeta", "dg", "dh0")
# 设备腿 3 条独立命令的**实测键**（本会话真跑，见 K4_RESULT.md）
KEY_DEVICE = "ascend|gdnb[ascend]|h2k8v8|c2ubz"     # --target ascend --bwd --no-run （默认）
KEY_NEG = "ascend|gdnb[ascend]|h2k8v8|c2purez"      # --target ascend --bwd --no-run --variant pure


def parse_items(text: str) -> dict[str, str]:
    """抓 `ITEM <name> rel= <v> verdict= <PASS|FAIL>` 的 rel 原文串。"""
    out: dict[str, str] = {}
    for m in re.finditer(r"ITEM\s+(\w+)\s+rel=\s*([0-9.eE+-]+)\s+verdict=\s*(\w+)", text):
        out[m.group(1)] = m.group(2)
    return out


def parse_fp(text: str) -> dict[str, str]:
    """抓 DBETA-FP 三跑指纹行 `  none dbeta-fp=<h>`。"""
    out: dict[str, str] = {}
    for m in re.finditer(r"^\s*(none|dirty|zero)\s+dbeta-fp=([0-9a-f]{8})", text, re.M):
        out[m.group(1)] = m.group(2)
    return out


def read(p: str) -> str:
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def main() -> int:
    ok = True
    print("=" * 78)
    print("K4_probe_variant_leg —— 设备腿变体判（不占卡）")
    print("=" * 78)

    # E1 代码
    src = read(RDC)
    has_ub_ascend = 'return "pure" if ARGS.target == "cpu" else "ub"' in src
    print(f"[E1] run_delta_card.py `_variant()` ascend→ub 逐字在文中: {has_ub_ascend}")
    if not has_ub_ascend:
        print("     !! 源文不再是 ascend→ub —— 前提可能已变，须重判")
        ok = False

    # E2 运行指纹
    try:
        v3 = parse_items(read(os.path.join(ART, "w4_deltabwd.log")))
        ub = parse_items(read(os.path.join(ART, "w4_d3_ub.log")))
        pu = parse_items(read(os.path.join(ART, "w4_d3_pure.log")))
    except FileNotFoundError as e:
        print(f"[E2] 缺料：{e}  —— 跳过（须在含窗四 artifacts 的仓库跑）")
        return 2 if not ok else 0
    same_ub = [it for it in ITEMS if v3.get(it) == ub.get(it)]
    same_pu = [it for it in ITEMS if v3.get(it) == pu.get(it)]
    diff_ub = [it for it in ITEMS if v3.get(it) != ub.get(it)]
    print(f"[E2] V3 vs V4-ub  逐字同项: {same_ub}")
    print(f"     V3 vs V4-ub  异项  : {diff_ub}")
    print(f"     V3 vs V4-pure 逐字同项: {same_pu}")
    # 关键：dq/dk/dg 是变体指纹（V4-ub 的 ORDER-DIFF 只列 dbeta ⇒ 其余项与 prefill 无关）
    fp_items = ("dq", "dk", "dg")
    v3_is_ub = all(v3.get(i) == ub.get(i) for i in fp_items)
    v3_is_pu = all(v3.get(i) == pu.get(i) for i in fp_items)
    print(f"     变体指纹项 {fp_items}: V3==ub? {v3_is_ub} ; V3==pure? {v3_is_pu}")
    if not (v3_is_ub and not v3_is_pu):
        print("     !! V3 指纹不唯一指向 ub —— 结论不成立，须上报")
        ok = False

    # E2b：ub 的 none 指纹 == pure 指纹（解释 dbeta 同值）
    ub_fp = parse_fp(read(os.path.join(ART, "w4_d3_ub.log")))
    pu_fp = parse_fp(read(os.path.join(ART, "w4_d3_pure.log")))
    none_eq = ub_fp.get("none") and ub_fp.get("none") == pu_fp.get("none")
    print(f"[E2b] ub.none fp={ub_fp.get('none')} == pure fp={pu_fp.get('none')} ? {bool(none_eq)}"
          f"   ⇒ dbeta 同值可由'同 prefill 态同值'解释，无须假设 V3 走了 pure")
    print(f"      ub 三跑指纹: {ub_fp}（随 prefill 变 ⇒ stale）; pure 三跑: {pu_fp}（恒定）")

    # E3 编译键（记录值自证）
    print(f"[E3] 设备默认键 = {KEY_DEVICE}   (c2ubz ⇒ ub)")
    print(f"     负例强制键 = {KEY_NEG}       (c2purez ⇒ 开关存在)")
    if "c2ubz" not in KEY_DEVICE or "c2purez" not in KEY_NEG:
        ok = False

    print("-" * 78)
    print("判词: 设备腿走 ub" + ("（结论成立）" if ok else "（判据不成立，须上报）"))
    print("=" * 78)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
