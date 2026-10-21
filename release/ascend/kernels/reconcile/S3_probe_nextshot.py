#!/usr/bin/env python3
"""reconcile/S3_probe_nextshot.py — P1-1s 下窗一枪（S3 交付物 3）。

**不改 U2 探针的产品语义**：本文件独立重写同一「门牌号」手法（encode/decode/predict 与
`U2_probe_k48_dma_semantics.py` 逐式同构，但本文件不改它一个字），另加三件事：
  (1) 提案 A 前/后：把实测图与 **P(=提案A落地后的预期语义)** 对拍，看失配是否归零；
  (2) **H-b（UB 未清零/stale）判决实验**：目标档前先跑一遍"填充趟"（同形状、不同内容），
      再跑目标档；若"填充趟内容"会改变目标档的实测图 ⇒ 目标档残留**含 stale 成分** ⇒ H-b 真。
      三种填充：`none`(不填) / `zero`(全 0 门牌号) / `perm`(行置换门牌号，仍在合法域内以便被 decode 看到)。
  (3) **对称度量**：除 U2 的 `mm`(双有限且异) + `nn`(候选有限/实测NaN) 外，补 `sym`
      =(**实测有限 & 候选NaN**)——即 U2 漏罚的一侧（见 S3_STRIDED_SECOND_FACTOR.md §4）。

跑法：
    # 自测解码器（不占卡；应与 U2 --dev cpu 同样给出 P 且全 OK）
    .venv/bin/python ascend/kernels/reconcile/S3_probe_nextshot.py --dev cpu
    # 下窗卡上判决（单进程；单卡一次）
    python3 ascend/kernels/reconcile/S3_probe_nextshot.py --dev npu
    python3 ascend/kernels/reconcile/S3_probe_nextshot.py --dev npu --ks 48 --acts none
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

KS = (48, 96, 32)
LADDER = (64, 32, 16)
PREFILLS = ("none", "zero", "perm")


def largest_block(v: int) -> int:
    for c in LADDER:
        if v % c == 0:
            return c
    raise SystemExit(f"n/k={v} 不在 BLOCK_LADDER 表达窗内")


def encode_W(n: int, k: int, mode: str = "plain") -> np.ndarray:
    """门牌号 W[i][j]。mode: plain=i*1000+j；zero=0；perm=(n-1-i)*1000+j（行置换，仍落合法 decode 域）。"""
    i = np.arange(n, dtype=np.float64)[:, None]
    j = np.arange(k, dtype=np.float64)[None, :]
    if mode == "zero":
        return np.zeros((n, k), dtype=np.float64)
    if mode == "perm":
        return (n - 1 - i) * 1000.0 + j
    return i * 1000.0 + j


def onehot_A(m: int, k: int, bk: int) -> np.ndarray:
    a = np.zeros((m, k), dtype=np.float64)
    for p in range(min(m, bk)):
        a[p, p] = 1.0
    return a


def decode(C: np.ndarray, n: int, k: int, bk: int):
    probes = min(C.shape[0], bk)
    src_row = np.full((probes, n), np.nan)
    src_col = np.full((probes, n), np.nan)
    for p in range(probes):
        for ix in range(n):
            v = float(C[p, ix])
            if not np.isfinite(v):
                continue
            e = int(round(v))
            r, c = e // 1000, e % 1000
            if 0 <= r < n and 0 <= c < k:
                src_row[p, ix], src_col[p, ix] = r, c
    return src_row, src_col


def predict(nb: int, n: int, k: int, bn: int, bk: int, sp: int, dp: int):
    nb_out = dst_oob = 0
    src_row = np.full((bk, n), np.nan)
    src_col = np.full((bk, n), np.nan)
    span = n * k
    for mx in range(nb):
        base = mx * bn * k
        for r in range(bn):
            for c in range(bk):
                sflat, dslot = base + r * sp + c, r * dp + c
                j_1, kk = dslot // bk, dslot % bk
                if j_1 >= bn:
                    dst_oob += 1
                    continue
                if sflat < 0 or sflat >= span:
                    nb_out += 1
                    continue
                src_row[kk, mx * bn + j_1] = sflat // k
                src_col[kk, mx * bn + j_1] = sflat % k
    return src_row, src_col, nb_out, dst_oob


def cands(k: int, bk: int):
    return [("P(pitch,pitch)", k, bk), ("C(srcStride被忽略)", bk, bk),
            ("G(双gap)", k + bk, 2 * bk), ("Gs(源gap)", k + bk, bk),
            ("Gd(目的gap)", k, 2 * bk), ("Z(dstStride被忽略)", k, 0),
            # —— S3 新增（把 log 里"每突发按连续写"这句叙述真正编码出来，与 P 同值以便对照）
            ("Zc(dst连续,src pitch)", k, bk)]


def stats(mrow, mcol, bk, n):
    probes = mrow.shape[0]
    expect = np.tile(np.arange(n, dtype=np.float64), (probes, 1))
    fin = np.isfinite(mrow)
    ok = int((fin & (mrow == expect)).sum())
    rowm = int(((mrow != expect) & fin).sum())
    colm = int(((mcol != np.arange(probes)[:, None]) & fin).sum())
    un = int((~fin).sum())
    return ok, rowm, colm, un


def score_all(mrow, n, k, bn, bk):
    out = []
    for tag, sp, dp in cands(k, bk):
        pr, _pc, oob, doob = predict(n // bn, n, k, bn, bk, sp, dp)   # nb=n//bn, sp/dp 以**元素**计
        take = pr[:min(16, bk)]
        mm = int(((take != mrow) & np.isfinite(mrow) & np.isfinite(take)).sum())
        nn = int((np.isfinite(take) & ~np.isfinite(mrow)).sum())
        sym = int((np.isfinite(mrow) & ~np.isfinite(take)).sum())   # U2 漏罚的一侧
        out.append((tag, mm + nn, sym, oob, doob))
    return out


def fingerprint(mrow, mcol):
    fin = np.isfinite(mrow)
    h = 0
    for a, b in zip(mrow[fin].astype(int).ravel().tolist(),
                    mcol[fin].astype(int).ravel().tolist()):
        h = (h * 1000003 + a * 1000 + b) & 0xFFFFFFFF
    return h


def emit(m, n, k, bn, bk, dev, act, Wmat):
    """发射一次；返回 C(m,n)。dev=npu 走 gemm_asc（**强钉 bk**，绕开 plan 选档，不正产品件一字）。"""
    import torch  # lazy
    A = torch.from_numpy(onehot_A(m, k, bk)).float()
    W = torch.from_numpy(Wmat).float()
    bias = torch.zeros(n, dtype=torch.float32)
    from ascend.kernels import gemm_asc
    if dev == "npu":
        A, W, bias = A.npu(), W.npu(), bias.npu()
        C = torch.zeros(m, n, dtype=torch.float32).npu()
        spec = {"target": "ascend",
                "key": f"gemm[ascend]|S3NEXT-n{n}k{k}b{gemm_asc.BLOCK_M}x{bn}x{bk}a{act}",
                "kwargs": {"n": n, "k": k, "bm": gemm_asc.BLOCK_M, "bn": bn, "bk": bk,
                           "act_mode": gemm_asc.ACT_MODE[act], "idt": "float32", "odt": "float32"},
                "impl": gemm_asc.gemm_asc_impl, "shape": (n, k)}
        if not gemm_asc.run(A, W, bias, C, spec):
            raise SystemExit("S3-PROBE-ABORT ascend 轨编译失败：%s" % spec["key"])
        torch.npu.synchronize()
        return C.detach().float().cpu().numpy().astype(np.float64)
    from ascend.kernels import ascend_env  # cpu 方言直调同一正文
    import tilelang
    C = torch.zeros(m, n, dtype=torch.float32)
    prog = gemm_asc.gemm_scalar_body(ascend_env.dialect("cpu"), A, W, bias, C,
                                     n=n, k=k, bm=gemm_asc.BLOCK_M, bn=bn, bk=bk,
                                     act_mode=gemm_asc.ACT_MODE[act])
    tilelang.compile(prog, out_idx=[], **ascend_env.compile_kwargs("cpu"))(A, W, bias, C)
    return C.detach().float().cpu().numpy().astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="cpu", choices=("cpu", "npu"))
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--ks", default=",".join(str(x) for x in KS))
    ap.add_argument("--acts", default="none,relu")
    ap.add_argument("--prefills", default=",".join(PREFILLS),
                    help="目标档前先跑的填充趟内容：none/zero/perm（H-b 判决用；cpu 轨忽略）")
    a = ap.parse_args()
    os.environ.setdefault("DMLAYA_ASCEND_TARGET", "ascend")
    import torch  # noqa: F401
    if a.dev == "npu" and hasattr(torch, "npu"):
        torch.npu.init()
    from ascend.kernels import gemm_asc  # noqa: F401

    print("S3-NEXT dev=%s m=%d n=%d ks=%s acts=%s prefills=%s" %
          (a.dev, a.m, a.n, a.ks, a.acts, a.prefills), flush=True)
    prefs = [p for p in a.prefills.split(",") if p] if a.dev == "npu" else ["none"]

    for k in [int(x) for x in a.ks.split(",")]:
        bn, bk = largest_block(a.n), largest_block(k)
        for act in a.acts.split(","):
            print("---- k=%d/%s bn=%d bk=%d nb=%d nk=%d dma=%s ----" %
                  (k, act, bn, bk, a.n // bn, k // bk, "strided" if bk < k else "single"), flush=True)
            fp = {}
            for pf in prefs:
                if pf != "none":                       # 先"弄脏" w_ub（同形状、不同内容）
                    emit(a.m, a.n, k, bn, bk, a.dev, act, encode_W(a.n, k, pf))
                got = emit(a.m, a.n, k, bn, bk, a.dev, act, encode_W(a.n, k, "plain"))
                mrow, mcol = decode(got, a.n, k, bk)
                ok, rowm, colm, un = stats(mrow, mcol, bk, a.n)
                fp[pf] = fingerprint(mrow, mcol)
                sc = score_all(mrow, a.n, k, bn, bk)
                print("  PREFILL=%-4s MEASURE OK=%d 行=%d 列=%d UNATTR=%d fp=%08x" %
                      (pf, ok, rowm, colm, un, fp[pf]), flush=True)
                print("      cands(U2步法 mm+nn | S3对称sym): " +
                      "  ".join("%s=%d|%d" % (t, m2, s) for t, m2, s, _o, _d in sc), flush=True)
            if a.dev == "npu" and len(prefs) > 1:
                base = fp["none"]
                d = [pf for pf in prefs if fp[pf] != base]
                print("  STALE-DIFF 与 none 不同的 prefill=%s ⇒ %s" %
                      (d or "无", "H-b 真（UB 残留受上次发射影响）" if d else
                       "H-b 未被本实验触发（残留与上次 UB 内容无关）"), flush=True)
    print("S3-NEXT-TRUTH（把实测对到下表判机制）:")
    print("  · 某候选 mm+nn==0 且 sym==0 → 该语义即硬件真值；若=P ⇒ 提案A(§6)语义成立、H-b 无关。")
    print("  · 若 =G(双gap) ⇒ 硬件忠实 gap、codegen 喂 pitch ⇒ 提案A 为必要修复。")
    print("  · 所有候选 mm+nn>0（含 sym=0 的最优仍>0）⇒ 非单一几何 ⇒ 看 STALE-DIFF。")
    print("  · STALE-DIFF>0 ⇒ H-b 真：残留含 stale ⇒ 修复须同时处理 UB 预清零/落趟拆解。")
    print("  · STALE-DIFF=0 且无候选命中 ⇒ 退提案 B（每突发单发连续 DMA，blockSize 语义无关）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
