#!/usr/bin/env python3
"""reconcile/M2_probe_patch_gemm.py — P1-1p 修复候选的**落件器**（改 ascend/kernels/gemm_asc.py）。

只做一件事：把 `plan()` 里 ascend 轨的 K 块宽选择，从"ladder 里取最大整除者"换成
`_pick_bk_ascend()`（整行连续优先 ⇒ 单发突发档；放不下才允许够宽的跨步档；都不满足 ⇒ None 落 eager）。
接口、I/O、交货位宽、正文算式**一行未改**，只改一个 tiling 常数的取法。

`--revert` 逐字回到改前态（A/B 对照用）；`--check` 只报当前态；**apply 是无参默认**（没有 `--apply` 这个
旗标，误传会 argparse 报错而不是改文件）。幂等：重复 apply/revert 报 no-op。
判"是否已应用"用的是**新增块本身**是否在场（ANCHOR_A 在改后态里仍然存在，不能拿它当标记）。
凭据行：`M2-PATCH <apply|revert> gemm_asc.py ok md5=<前8> lines=<n>` / `M2-PATCH-STATE <态>`
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
TARGET = os.path.join(RELEASE, "ascend", "kernels", "gemm_asc.py")

# ── ①新增块：ANCHOR_A 之后的插入体（BLOCK_A）────────────────────────────────
ANCHOR_A = '''def _largest_block(total: int) -> int | None:
    """在候选块宽里取能整除 total 的最大值；取不到说明这个维度表达不了（返回 None）。"""
    for blk in BLOCK_LADDER:
        if total % blk == 0:
            return blk
    return None
'''

BLOCK_A = '''

# ── P1-1p / P1-1q：W 块装填档的选择纪律 ────────────────────────────────────────
# 取证出处：release/runs/1011-p2-13-oncard-accept-2002/artifacts/sys1_kernels.tgz（归档真机生成码）
# 与 reconcile/M2_FP16_NAN_EVIDENCE.md。要点是生成码原文可复核的两点对置，不是推测：
#   · `bk < k` 时 `T.copy(W[mx*bn, ko*bk], w_ub)` 编成**多突发跨步 GM→UB DMA**，单突发宽 = bk*字节宽。
#     卡上实测出错的两枚同形：`ea1059e6…` device_kernel.asc:31（fp16 dA n32k96bk32 ⇒ 32 突发 ×64B、
#     跨距 192B）＝P1-1p 的 NaN 起点；`5d732a8f…` :31（fp32 n96k48bk16 ⇒ 同一组 32/64/192/64）＝P1-1q
#     的真机错值。经 compat `TL910B_BYTES_TO_BLK`（port910b_compat.h:599，÷32）换算即
#     (blockLen, srcStride, dstStride) = (2, 6, 2) 个 32B 块。
#   · `bk == k` 时 tilelang `src/ascend/op/copy.cc` 的 `plan_single_row` 折叠成**整块单发连续**
#     （blockCount=1），同位宽在卡上实测清白：fp16 前向 `8f30a21f…`/`e07da648…` 各 `1×2048B`
#     （H 案同窗前向 15 次调用零 NaN 零 inf），fp32 见 `3249db2f…` `1×8192B`、`32d6c39c…` `1×16384B`。
_MIN_STRIDED_BURST_B = 128
#: 动态 UB 预算 = 归档里**已在卡上编出产物**的最大申报值（`32d6c39c…` host_kernel.c 申报 24576B；
#: 与 reconcile/S_RESULT.md 算过的 gemm 最坏档 24KB 同量级）。是工程预算，不是硬件规范。
_UB_BUDGET_B = 24576


def _ub_bytes(bm: int, bn: int, bk: int, elem: int) -> int:
    """`gemm_scalar_body` 每核块的 UB 需求：a_ub(bm×bk, 恒 fp32) + w_ub(bn×bk, 随源位宽)
    + c_ub(bm×bn, fp32)。与生成码里 `buf_dyn_shmem` 的分块一一对应（对账见取证探针）。"""
    return bm * bk * 4 + bn * bk * elem + bm * bn * 4


def _pick_bk_ascend(bm: int, bn: int, k: int, elem: int) -> int | None:
    """ascend 轨的 K 块宽选择：**先试整行连续（bk==k ⇒ 单发突发档）**。

    白话：竖条整块一次取进小格子这种做法在卡上是证过清白的；切成几段、跨步去取，段宽只剩 64 字节
    的两处实测都错了。所以能整块取就整块取；整块取装不下小格子才退回分段，且分段只允许每段够宽
    （>=128 字节）的那几种；两条都不满足就交 None，由入口层换普通写法（不硬凑）。
    """
    if _ub_bytes(bm, bn, k, elem) <= _UB_BUDGET_B:
        return k
    ladder = _largest_block(k)
    if ladder is not None and ladder * elem >= _MIN_STRIDED_BURST_B:
        return ladder
    return None
'''

# ── ②plan() 里换 bk（这一对没有前缀包含关系，可直接对调）────────────────────
ANCHOR_B = '''    dt = _TL_DT.get(A.dtype, "float32")   # plan 已判 C.dtype == A.dtype 且位宽在白名单内 ⇒ 单段位宽
    kwargs = {"n": n, "k": k, "bm": BLOCK_M, "bn": bn, "bk": bk, "act_mode": ACT_MODE[act]}
'''

NEW_B = '''    dt = _TL_DT.get(A.dtype, "float32")   # plan 已判 C.dtype == A.dtype 且位宽在白名单内 ⇒ 单段位宽
    if name == ascend_env.TARGET_ASCEND:
        # 装填档纪律（P1-1p/P1-1q，取证见 `_pick_bk_ascend` 上方）：只动 bk 的取法，
        # 接口/正文算式/交货位宽不动；cpu 轨（产物里没有该 DMA 面）维持 ladder 原样。
        bk = _pick_bk_ascend(BLOCK_M, bn, k, 2 if dt == "float16" else 4)
        if bk is None:
            return None
    kwargs = {"n": n, "k": k, "bm": BLOCK_M, "bn": bn, "bk": bk, "act_mode": ACT_MODE[act]}
'''


def replace_once(src, frm, to, what):
    if src.count(frm) != 1:
        raise SystemExit(f"M2-PATCH-FAIL {what}: 锚点命中 {src.count(frm)} 次（期望 1）")
    out = src.replace(frm, to, 1)
    if out == src:
        raise SystemExit(f"M2-PATCH-FAIL {what}: 替换后文本无变化")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--check", action="store_true", help="只报告当前是改后态还是改前态")
    a = ap.parse_args()
    src = open(TARGET).read()
    applied = BLOCK_A in src and NEW_B in src
    if a.check:
        print(f"M2-PATCH-STATE {'POST-FIX' if applied else 'PRE-FIX'} "
              f"md5={hashlib.md5(src.encode()).hexdigest()[:8]}", flush=True)
        return 0
    if a.revert:
        if not applied:
            print("M2-PATCH revert no-op（本来就是改前态）", flush=True)
            return 0
        out = replace_once(src, BLOCK_A, "", "摘除新增块")
        out = replace_once(out, NEW_B, ANCHOR_B, "还原 plan")
    else:
        if applied:
            print("M2-PATCH apply no-op（已是改后态）", flush=True)
            return 0
        out = replace_once(src, ANCHOR_A, ANCHOR_A + BLOCK_A, "插入新增块")
        out = replace_once(out, ANCHOR_B, NEW_B, "改写 plan")
    tmp = TARGET + ".m2tmp"
    with open(tmp, "w") as f:
        f.write(out)
    os.replace(tmp, TARGET)
    print(f"M2-PATCH {'revert' if a.revert else 'apply'} gemm_asc.py ok "
          f"md5={hashlib.md5(out.encode()).hexdigest()[:8]} lines={out.count(chr(10)) + 1}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
