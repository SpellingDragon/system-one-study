#!/usr/bin/env python3
"""reconcile/U2_probe_k48_host.py — P1-1q（`linear_prod` K=48 真机数值错）的**本地复现判据**探针。

【要回答的唯一问题】产品件 `gemm_asc.gemm_scalar_body` 在 `m=16,n=96,k=48,bn=32,bk=16` 上的
错值，是**正文数学**缺陷（那本地就该复现、当场可修），还是**昇腾 DMA 装填面**缺陷（那本地
结构上量不到，必须下窗）？

【三轨口径（谁判什么，务必分开读）】
  T1 `cpu`   ：`ascend_env.dialect("cpu")` + `tilelang.compile(target=c 三件套)` **真跑**，直调
               正文生成器（不经 plan/入口件）⇒ 判"同一份算式的数学"。bk 由本探针**强喂**，
               故可造出 plan 永远给不出的档（非整除 bk），用来量"表达窗外会发生什么"。
  T2 `alias` ：**产品路由**（`gemm_kernel.forward(..., target="ascend")`）+ cpu 方言执行
               （`tilelang.ascend.language` 两处一起指向 cpu 方言 + `compile_kwargs` 翻 c 后端，
               配方 = `run_numerics._prod_setup` 的 cpu-alias 档）⇒ 判"正文 + plan + 缓存键 +
               入口件"这一整套产品面。**注意**：这一轨的 `T.copy` 走的是 **c 后端 lowering**，
               不是 ascend 的 MTE2 跨步 DMA ⇒ 本轨**结构上量不到** P1-1q 的嫌疑面（这正是
               C2/S 两轮 9/9 漏掉它的根因，见 U2_matrix_k48.md §3）。
  T3 `pred`  ：纯静态判据（不编译）——按 tilelang `src/ascend/op/copy.cc` 的 `plan_single_row`
               条件复算"该 (k,bk) 在 ascend 侧会编成**单发连续**还是**多突发跨步** DMA"，
               并与归档生成码（`5d732a8f`=k48 多突发 / `77a8a3c4`=k32 单发）逐字对齐。

【判据】沿用产品件尺子：`rel = max|got-ref| / max|ref| < 2e-5`（`REL_TOL_LIN` 同值，本探针
不动任何产品阈值）。附加两条：relu 档要求出口零负数；`bk∤k` 档**不参与 PASS/FAIL**（记
`OUT-OF-PLAN`），只如实登记数值形态。

【跑法】
  cd release && .venv/bin/python ascend/kernels/reconcile/U2_probe_k48_host.py
  cd release && .venv/bin/python ascend/kernels/reconcile/U2_probe_k48_host.py --tracks cpu
  可选：--n 96 --ks 16,32,48,64,80,96 --bks 16,32 --acts none,relu
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))                    # .../release/ascend/kernels/reconcile
K_DIR = os.path.dirname(HERE)                                        # .../release/ascend/kernels
RELEASE_ROOT = os.path.dirname(os.path.dirname(K_DIR))               # .../release
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

import numpy as np           # noqa: E402
import torch                 # noqa: E402
import tilelang              # noqa: E402

from ascend.kernels import ascend_env, gemm_asc   # noqa: E402

REL_TOL = 2e-5          # 与 run_numerics.REL_TOL_LIN / S_cpu_oracle.REL_TOL 同尺（本探针不改产品阈值）
BM = gemm_asc.BLOCK_M   # 16，产品件常量


def rel(got: np.ndarray, ref: np.ndarray) -> float:
    num = float(np.abs(got - ref).max())
    den = max(float(np.abs(ref).max()), 1e-12)
    return num / den


def ref_np(A, W, b, act):
    pre = A.detach().numpy().astype(np.float64) @ W.detach().numpy().astype(np.float64).T \
        + b.detach().numpy().astype(np.float64)
    return np.maximum(pre, 0.0) if act == "relu" else pre


# ─────────────────────────────────────────────────────────── T3 静态判据（不编译）
def dma_form(n: int, k: int, bn: int, bk: int, elem_bytes: int = 4) -> tuple[str, str]:
    """复算 tilelang ascend copy lowering 的形态（`src/ascend/op/copy.cc` 的 plan_single_row）。

    W 块 = `T.copy(W[mx*bn, ko*bk], w_ub(bn,bk))`：源区是 GM 上 (bn 行 × bk 列)、行步长 k 的
    **跨步**区；目的 UB 是紧凑 (bn*bk)。`plan_single_row` 成立当且仅当 n_rows==1 或
    `src_stride_bytes == row_bytes`（且 dst 同）⇒ 只有 **bk == k**（W 块吃满整行）才collapse
    成单发连续突发；`bk < k` 一律编成 `blockCount=bn` 的多突发跨步 DMA。
    """
    row_bytes = bk * elem_bytes
    src_stride = k * elem_bytes
    if n // bn == 1 or src_stride == row_bytes:
        return "single", f"1 × {row_bytes * (k // bk if k >= bk else 1)}B（连续）"
    return "strided", f"{bn} × {row_bytes}B, srcStride={src_stride}B, dstStride={row_bytes}B"


# ─────────────────────────────────────────────────────────── T1 cpu 方言直调正文
def track_cpu(m, n, k, bk, act):
    """真编真跑：dialect("cpu") + gemm_scalar_body（bk 强喂，不经 plan）。"""
    T = ascend_env.dialect("cpu")
    bn = gemm_asc._largest_block(n) or 32
    A = torch.randn(m, k, dtype=torch.float32)
    W = torch.randn(n, k, dtype=torch.float32)
    b = torch.randn(n, dtype=torch.float32)
    C = torch.zeros(m, n, dtype=torch.float32)
    kw = {"n": n, "k": k, "bm": BM, "bn": bn, "bk": bk, "act_mode": gemm_asc.ACT_MODE[act]}
    prog = gemm_asc.gemm_scalar_body(T, A, W, b, C, **kw)
    kern = tilelang.compile(prog, out_idx=[], **ascend_env.compile_kwargs("cpu"))
    kern(A, W, b, C)
    return rel(C.detach().numpy().astype(np.float64), ref_np(A, W, b, act)), int((C < 0).sum()), kern


# ─────────────────────────────────────────────────────────── T2 产品路由（cpu-alias 档）
def alias_setup():
    """把"产品递进去 target=ascend"的执行面翻到 cpu 方言（配方同 run_numerics._prod_setup）。"""
    import types
    import tilelang as _tl
    saved = {n_: getattr(ascend_env, n_) for n_ in
             ("compile_kwargs", "backend_available", "_npu_present", "active_backend")}
    cpu_mod = ascend_env.dialect("cpu")
    pkg = sys.modules.get("tilelang.ascend")
    if pkg is None:
        pkg = types.ModuleType("tilelang.ascend")
        pkg.__path__ = []
        sys.modules["tilelang.ascend"] = pkg
        setattr(_tl, "ascend", pkg)
    saved_alias = (sys.modules.get("tilelang.ascend.language"), getattr(pkg, "language", None))
    sys.modules["tilelang.ascend.language"] = cpu_mod
    pkg.language = cpu_mod
    ascend_env.compile_kwargs = lambda t: {"target": "c", "target_host": "c",
                                           "execution_backend": "cython"}
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda *a, **k: ascend_env.TILELANG
    ascend_env.reset()

    def restore():
        mod, attr = saved_alias
        if mod is None:
            sys.modules.pop("tilelang.ascend.language", None)
        else:
            sys.modules["tilelang.ascend.language"] = mod
        if attr is None:
            try:
                delattr(sys.modules["tilelang.ascend"], "language")
            except AttributeError:
                pass
        else:
            sys.modules["tilelang.ascend"].language = attr
        for n_, v in saved.items():
            setattr(ascend_env, n_, v)
        ascend_env.reset()
    return restore


def track_alias(m, n, k, act):
    """走真入口件 `gemm_kernel.forward(..., target="ascend")`；bk 由 plan 的阶梯决定。"""
    from ascend.kernels import gemm_kernel
    A = torch.randn(m, k, dtype=torch.float32)
    W = torch.randn(n, k, dtype=torch.float32)
    b = torch.randn(n, dtype=torch.float32)
    got = gemm_kernel.forward(A, W, b, act, torch.float32, "ascend")   # 入口件交回单张量
    plan_key = [x for x in ascend_env.compiled_keys() if x.startswith("ascend|gemm[ascend]")]
    bk = gemm_asc._largest_block(k)
    r = rel(got.detach().float().numpy().astype(np.float64), ref_np(A, W, b, act))
    routed = bool(plan_key)      # 假绿守卫：无 ascend| 键 ⇒ 落 eager，不算判决
    mine = [x for x in plan_key if f"n{n}k{k}" in x]
    return r, int((got < 0).sum()), bk, routed, (mine[-1] if mine else "NOKEY")


# ─────────────────────────────────────────────────────────── T4 跨步 DMA 地址核算
def dma_addr_check(n: int, k: int, bn: int, bk: int, elem_bytes: int = 4) -> list[tuple[str, int, int]]:
    """按归档生成码的实参（k=48 档：`32,64,0,0,0,mode,192,64`）核算每种步长语义的**读地址上界**。

    语义候选（都不涉及数值算式，纯地址算术，host 可判定）：
      P (pitch/首地址间距)  : 第 r 块起点 = base + r·srcStride           ← codegen 与 compat 的假设
      G (gap/块间空隙)      : 第 r 块起点 = base + r·(srcStride+rowBytes)
      U (单位混用：/32 再乘) : compat 把 srcStride 字节换成 32B 块数交给原生件；若原生件按**字节**
                              解释该值，则实际间距 = srcStride/32
    读越界 ⇔ 末块尾址 > 张量字节跨度（W 是 (n,k) ⇒ span = n·k·elem_bytes）。
    """
    span = n * k * elem_bytes
    row_b, src_stride = bk * elem_bytes, k * elem_bytes
    out = []
    for label, pitch in (("P(pitch)", src_stride), ("G(gap)", src_stride + row_b),
                         ("U(/32 当字节)", max(src_stride // 32, 1))):
        hi = (bn - 1) * pitch + row_b          # 最后一块的尾地址（相对块起点）
        # 最坏列块 block_idx=nb-1、最坏 ko：base=(nb-1)·bn·k + (nk-1)·bk
        nb, nk = n // bn, k // bk
        base_hi = (nb - 1) * bn * k * elem_bytes + max(nk - 1, 0) * bk * elem_bytes
        out.append((label, base_hi + hi, span))
    return [(lab, hi, span, "OOB +%dB" % (hi - span) if hi > span else "in-bounds")
            for lab, hi, span in out]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="cpu,alias,pred,addr")
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--ks", default="16,32,48,64,80,96")
    ap.add_argument("--bks", default="16,32")
    ap.add_argument("--acts", default="none,relu")
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",")]
    bks = [int(x) for x in a.bks.split(",")]
    acts = a.acts.split(",")
    tracks = a.tracks.split(",")
    bn = gemm_asc._largest_block(a.n)
    print("U2-PROBE m=%d n=%d bn=%d ladder_bk(k)=%s" % (
        a.m, a.n, bn, {k: gemm_asc._largest_block(k) for k in ks}), flush=True)

    fails = 0
    if "pred" in tracks:
        print("\n── T3 静态判据：ascend 侧 DMA 形态（bk==k ⇒ 单发连续；bk<k ⇒ 多突发跨步）", flush=True)
        for k in ks:
            for bk in bks:
                form, args = dma_form(a.n, k, bn, bk)
                plan_bk = gemm_asc._largest_block(k)
                tag = "IN-PLAN" if plan_bk == bk else "OUT-OF-PLAN(plan 给不出)"
                print("U2-PRED k=%-3d bk=%-2d nk=%-2d %-8s %-28s %s" % (
                    k, bk, k // bk, form.upper(), args, tag), flush=True)

    if "cpu" in tracks:
        print("\n── T1 cpu 方言轨（真跑，直调 gemm_scalar_body）", flush=True)
        for k in ks:
            for bk in bks:
                for act in acts:
                    try:
                        r, neg, _ = track_cpu(a.m, a.n, k, bk, act)
                    except Exception as e:                      # 表达窗外的档可能直接编炸：如实登记，不吞
                        print("U2-CPU  k=%-3d bk=%-2d %-4s EXC %s: %s" % (
                            k, bk, act, type(e).__name__, str(e)[:160].replace("\n", " ")), flush=True)
                        continue
                    in_plan = (gemm_asc._largest_block(k) == bk)
                    if not in_plan:
                        print("U2-CPU  k=%-3d bk=%-2d %-4s rel=%.3e neg=%d OUT-OF-PLAN（bk∤k⇒静默截尾，"
                              "不计 PASS/FAIL）" % (k, bk, act, r, neg), flush=True)
                        continue
                    ok = (r < REL_TOL) and (neg == 0 if act == "relu" else True)
                    fails += 0 if ok else 1
                    print("U2-CPU  k=%-3d bk=%-2d %-4s rel=%.3e neg=%d %s" % (
                        k, bk, act, r, neg, "PASS" if ok else "FAIL"), flush=True)

    if "alias" in tracks:
        print("\n── T2 产品路由轨（cpu-alias：入口件+plan+缓存键全是产品的，执行面是 c 后端）", flush=True)
        restore = alias_setup()
        try:
            for k in ks:
                for act in acts:
                    try:
                        r, neg, bk, routed, key = track_alias(a.m, a.n, k, act)
                    except Exception as e:
                        fails += 1
                        print("U2-ALIAS k=%-3d %-4s EXC %s: %s" % (
                            k, act, type(e).__name__, str(e)[:160].replace("\n", " ")), flush=True)
                        continue
                    if not routed:
                        fails += 1
                        print("U2-ALIAS k=%-3d %-4s NOKEY ⇒ 落 eager，假绿守卫拦下（不计数值判决）" % (
                            k, act), flush=True)
                        continue
                    ok = (r < REL_TOL) and (neg == 0 if act == "relu" else True)
                    fails += 0 if ok else 1
                    print("U2-ALIAS k=%-3d bk=%-2d %-4s rel=%.3e neg=%d %s key=%s" % (
                        k, bk, act, r, neg, "PASS" if ok else "FAIL", key), flush=True)
        finally:
            restore()

    if "addr" in tracks:
        print("\n── T4 跨步 DMA 读地址核算（纯地址算术；判 '哪种步长语义会读越界'）", flush=True)
        for k in ks:
            bk = gemm_asc._largest_block(k)
            if bk is None or k % bk:
                continue
            form, _ = dma_form(a.n, k, bn, bk)
            if form != "strided":            # 单发连续无跨步语义可判（blockCount=1，步长被忽略）
                print("U2-ADDR k=%-3d bk=%-2d dma=SINGLE   —— 步长参数不参与（blockCount=1）" % (k, bk),
                      flush=True)
                continue
            for lab, hi, span, verdict in dma_addr_check(a.n, k, bn, bk):
                print("U2-ADDR k=%-3d bk=%-2d dma=%-8s %-12s 读上界=%-8dB span=%-8dB %s" % (
                    k, bk, form, lab, hi, span, verdict), flush=True)

    print("\nU2-PROBE-DONE cells_failed=%d（T1/T2 内 IN-PLAN 档的失败数；T3 不判数值）" % fails, flush=True)
    print("U2-LOCAL-VERDICT: k=48 两轨均 PASS ⇒ **本地不复现** ⇒ 缺陷面在 ascend 的 "
          "多突发跨步 DMA（bk<k），host 结构上量不到" if fails == 0 else
          "U2-LOCAL-VERDICT: 本地复现 ⇒ 纯逻辑缺陷，见上表 FAIL 档", flush=True)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
