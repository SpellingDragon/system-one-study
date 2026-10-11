#!/usr/bin/env python3
"""reconcile/S3_probe_semantics_repro.py — S3 离线判读件（P1-1s 第二层因素）。

不改任何产品语义：本文件只**离线复算** U2_probe_k48_dma_semantics.py 的口径，
用来回答一个问题——"哪种硬件步长语义能同时复现归档日志 line16/line30 的
聚合数（OK/行错位/列错位/UNATTRIBUTED）与六候选失配？"

判据来源（实测原文，非转述）：
  runs/1011-p2-13-oncard-window2-throughput/artifacts/w2_pitchgap.log
  k=48/none : OK=0 行错位=1536 列错位=1353 UNATTR=0
              Z=45 G=384 Gd=768 Gs=1408 P=1536 C=1536
  k=96/none : OK=24 行错位=872 列错位=752 UNATTR=640
              Z=48 G=384 Gd=696 Gs=1380 C=1504 P=1512

跑法： cd release && .venv/bin/python ascend/kernels/reconcile/S3_probe_semantics_repro.py
"""
import numpy as np


def make(n, k, bn, bk):
    Wflat = np.array([i * 1000 + j for i in range(n) for j in range(k)], dtype=np.float64)
    return dict(n=n, k=k, bn=bn, bk=bk, nb=n // bn, Wflat=Wflat)


def run_hw(g, sp, dp, blen):
    """硬件模型：**每个 n-block 独立一次 DMA**（各自 UB 副本，如真机 block_idx）。
    突发 r 读 src_base+r*sp 处 blen 个元素，写 dst_base+r*dp（连续）。
    返回 {mx: w_mx}, oobR, oobW。"""
    n, k, bn, bk, nb, Wflat = g["n"], g["k"], g["bn"], g["bk"], g["nb"], g["Wflat"]
    out, oobR, oobW = {}, 0, 0
    for mx in range(nb):
        w = np.full(bn * bk, np.nan)
        base = mx * bn * k
        for r in range(bn):
            for c in range(blen):
                s, d = base + r * sp + c, r * dp + c
                if d >= bn * bk:
                    oobW += 1
                    continue
                if s < 0 or s >= n * k:
                    oobR += 1
                    w[d] = np.nan
                    continue
                w[d] = Wflat[s]
        out[mx] = w
    return out, oobR, oobW


def measure(g, wper):
    """C[p][ix] = w_mx[j_1*bk + p]，ix = mx*bn + j_1；探针只读前 16 行（m=16）。"""
    n, k, bn, bk, nb = g["n"], g["k"], g["bn"], g["bk"], g["nb"]
    probes = min(16, bk)
    mrow = np.full((probes, n), np.nan)
    mcol = np.full((probes, n), np.nan)
    for mx in range(nb):
        w = wper[mx]
        for j1 in range(bn):
            for p in range(probes):
                v = w[j1 * bk + p]
                if not np.isfinite(v):
                    continue
                e = int(round(float(v)))
                rr, cc = e // 1000, e % 1000
                if 0 <= rr < n and 0 <= cc < k:
                    mrow[p, mx * bn + j1] = rr
                    mcol[p, mx * bn + j1] = cc
    return mrow, mcol


def predict(g, sp, dp):
    n, k, bn, bk, nb = g["n"], g["k"], g["bn"], g["bk"], g["nb"]
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


def cands(k, bk):
    return [("P(pitch,pitch)", k, bk), ("C(srcStride被忽略)", bk, bk),
            ("G(双gap)", k + bk, 2 * bk), ("Gs(源gap)", k + bk, bk),
            ("Gd(目的gap)", k, 2 * bk), ("Z(dstStride被忽略)", k, 0)]


def score(g, mrow, mcol):
    n, k, bn, bk = g["n"], g["k"], g["bn"], g["bk"]
    probes = min(16, bk)
    expect = np.tile(np.arange(n, dtype=np.float64), (probes, 1))
    unattr = int((~np.isfinite(mrow)).sum())
    mism = int(((mrow != expect) & np.isfinite(mrow)).sum())
    exp_col = np.arange(probes, dtype=np.float64)[:, None]
    colm = int(((mcol != exp_col) & np.isfinite(mrow)).sum())
    okc = int((np.isfinite(mrow) & (mrow == expect)).sum())
    rows = []
    for tag, sp_e, dp_e in cands(k, bk):
        pr, pc, oob, doob = predict(g, sp_e, dp_e)
        take = pr[:probes] if pr.shape[0] >= probes else pr
        mm = int(((take != mrow) & np.isfinite(mrow) & np.isfinite(take)).sum())
        nn = int((np.isfinite(take) & ~np.isfinite(mrow)).sum())
        rows.append((tag, mm + nn, oob, doob))
    rows.sort(key=lambda s: s[1])
    return okc, mism, colm, unattr, rows


def main():
    cases = [
        ("k=48/bk=16", make(96, 48, 32, 16),
         "OK=0 行错位=1536 列错位=1353 UNATTR=0 | Z=45 G=384 Gd=768 Gs=1408 P=1536 C=1536"),
        ("k=96/bk=32", make(96, 96, 32, 32),
         "OK=24 行错位=872 列错位=752 UNATTR=640 | Z=48 G=384 Gd=696 Gs=1380 P=1512 C=1504"),
    ]
    # (sp, dp, blen) — sp/dp are the *effective* per-burst advances in elements
    models = {
        "PITCH codegen-intent": lambda k, bk: (k, bk, bk),
        "GAP  (blen+stride)":   lambda k, bk: (bk + k, bk + bk, bk),
        "SRCGAP only":          lambda k, bk: (bk + k, bk, bk),
        "DSTGAP only":          lambda k, bk: (k, bk + bk, bk),
        "Z ignored-dst":        lambda k, bk: (k, 0, bk),
        "C ignored-src":        lambda k, bk: (0, bk, bk),
        "dstStride-as-pitch, src gap (mixed)": lambda k, bk: (bk + k, bk, bk),
    }
    for name, g, expect in cases:
        k, bk = g["k"], g["bk"]
        print("=" * 78)
        print("%s  LOG: %s" % (name, expect))
        for mname, f in models.items():
            sp, dp, blen = f(k, bk)
            wper, oobR, oobW = run_hw(g, sp, dp, blen)
            mrow, mcol = measure(g, wper)
            okc, mism, colm, unattr, rows = score(g, mrow, mcol)
            print("  H=%-32s meas OK=%d 行=%d 列=%d UNATTR=%d" % (mname, okc, mism, colm, unattr))
            print("      cands: " + "  ".join("%s=%d" % (t, m) for t, m, _o, _d in rows))


if __name__ == "__main__":
    main()
