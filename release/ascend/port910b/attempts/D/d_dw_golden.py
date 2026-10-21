#!/usr/bin/env python3
"""attempts/D/d_dw_golden.py — dW = dYᵀ @ X（权重梯度）的 CPU fp32/fp64 golden 与**上卡判据**。

本机无 NPU，本件是 d_dw_910b.py 上卡对拍的唯一参考真值。语义（生产尺子
release/sys1/kernels/gemm_bwd_dw_mps.py:39-63 同口径）：

    dY (T, N) fp16   上游梯度（行=token）
    X  (T, K) fp16   前向激活输入（行=token）
    dW (N, K) fp32   dW[n,k] = SUM_{t=0..T-1} dY[t,n] * X[t,k]      ← 归约轴 = token

三路真值同框，用来把"语义/转置方向错"与"累加次序数值差"分开定位：
  A) exact      —— 先把输入舍到 fp16（kernel 看到的即这组数），再用 float64 精确归约。
                  这是**语义真值**：任何"转置摆反/轴用错"都会在这里现形。
  B) fp32-tiled —— 复刻 kernel 的分块次序：token 轴按 TT 分块，块内 float32 matmul，
                  块间 float32 累加（等价于 L0C 跨 kt clear_accum=首次 + 串行累加）。
                  与 A 的差 = **纯累加次序误差**，是判据的标尺本身。
  C) fp16-out   —— 若入口层把出口降位宽（生产里 dW 也有 fp16 消费方），给出的容差档。

上卡判据（P0/P1/P2 三级，见 main() 打印）：
  P0 转置方向（**方案级**判据，必须先过）：
      relFro(DW_kern, exact) < 1e-2  **且**  relFro(DW_kern, exact.T) 明显更大
      → 若 DW_kern 更接近 exact.T，说明 dn2nz（GM→L1 转置载入）或 l0c2gm 的
        NZ 位映射（compat VERIFY V1/V3）把方向搞反了，属布局 bug 而非精度 bug。
  P1 累加精度（fp16 入 / fp32 累加 / fp32 出）：
      relFro(DW_kern, B) <= 2e-6（同分块次序应近乎 bit 级）；
      relFro(DW_kern, A) <= 1e-5（相对 exact 的次序误差上界，实测见打印）。
  P2 逐元素（torch.testing.assert_close 口径）：
      atol=1e-3, rtol=1e-5 对 B；放宽到 atol=1e-2, rtol=1e-3 对 A（尾块/边界用）。
  出口若降 fp16：P1 阈值换 5e-4（fp16 eps≈4.9e-4），P2 换 atol=5e-2, rtol=5e-3。

用法：
    python3 d_dw_golden.py                      # 自检（A/B/C 互校 + 阈值实标定 + 转置检测）
    python3 d_dw_golden.py --npz /tmp/dw_pairs.npz [--T 2048 --N 256 --K 256 --TT 64]
        # 产上卡对拍件（dY/X fp16 + DW_expected fp32 + exact.T 参照 + 阈值）

上卡对拍参考片段（有 NPU 的机器）：
    import numpy as np, torch, tilelang
    z = np.load("/tmp/dw_pairs.npz")
    import d_dw_910b as D
    fn = D.dw_l0tr(int(z["T"]), int(z["N"]), int(z["K"]), 128, 128, int(z["TT"]), 64,
                  "float16", "float32")
    k = tilelang.compile(fn, target="ascend", out_idx=-1)
    dY = torch.from_numpy(z["dY"]).npu(); X = torch.from_numpy(z["X"]).npu()
    got = k(dY, X).float().cpu().numpy()
    rel = lambda a, b: float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
    print("P0", rel(got, z["exact"]) < 1e-2 and rel(got, z["exact"]) < rel(got, z["exact_T"]))
    print("P1", rel(got, z["tiled"]), "P2", float(np.abs(got - z["tiled"]).max()))
"""
import argparse
import sys

import numpy as np

F16 = np.float16
F32 = np.float32
F64 = np.float64


def make_inputs(T, N, K, seed=0):
    """固定种子造 (dY, X)：值域 [-4,4]/8 —— 与主仓 ascend 用例同方（小整数比，
    既让 fp16 精确可表示（舍入零损失），又便于把布局 bug 一眼看出来）。"""
    g = np.random.RandomState(seed)
    dY = (g.randint(-4, 5, size=(T, N)) / 8.0).astype(F16)
    X = (g.randint(-4, 5, size=(T, K)) / 8.0).astype(F16)
    return dY, X


def exact_dw(dY, X):
    """A) 语义真值：fp16 输入 → float64 精确归约 → float32 出口。"""
    return (dY.astype(F64).T @ X.astype(F64)).astype(F32)


def tiled_dw(dY, X, TT, BN, BK):
    """B) 复刻 kernel 分块次序：token 轴 TT 分块、块间 fp32 累加、输出按 (BN,BK) 分格。

    与 L0C 的对应关系：每格 acc 先 clear（kt==0 的 clear_accum=True），随后
    每块 fp32 matmul 结果相加——与 cube 的 sub-K 累加同序（硬件内部按 16 行
    分块，这里用整块 fp32 已足够贴近，差值远小于 fp16 输入舍入）。
    """
    T, N = dY.shape
    K = X.shape[1]
    out = np.zeros((N, K), dtype=F32)
    for n0 in range(0, N, BN):
        for k0 in range(0, K, BK):
            acc = np.zeros((min(BN, N - n0), min(BK, K - k0)), dtype=F32)
            for t0 in range(0, T, TT):
                dyb = dY[t0:t0 + TT, n0:n0 + BN].astype(F32)
                xkb = X[t0:t0 + TT, k0:k0 + BK].astype(F32)
                acc = acc + dyb.T @ xkb          # fp32 累加，块内一次归约
            out[n0:n0 + BN, k0:k0 + BK] = acc
    return out


def relfro(a, b):
    return float(np.linalg.norm(a.astype(F64) - b.astype(F64)) /
                 (np.linalg.norm(b.astype(F64)) + 1e-12))


def selfcheck(args):
    T, N, K, TT = args.T, args.N, args.K, args.TT
    BN = BK = 128
    dY, X = make_inputs(T, N, K, args.seed)
    A = exact_dw(dY, X)
    B = tiled_dw(dY, X, TT, BN, BK)
    C = B.astype(F16)

    print("D-GOLDEN-SELFCHK")
    print(f"  T={T} N={N} K={K} TT={TT} BN={BN} BK={BK} seed={args.seed} "
          f"dtype_in=float16 acc=float32 out=float32")
    print(f"  exact   : norm={np.linalg.norm(A.astype(F64)):.6g} "
          f"maxabs={np.abs(A).max():.6g}")
    print(f"  tiled   : relFro(vs exact)={relfro(B, A):.3e}   "
          f"<== 这就是'纯累加次序误差'上界，P1 阈值按它标定")
    print(f"  fp16out : relFro(vs tiled)={relfro(C, B):.3e}")
    # 逐元素口径
    d = np.abs(B.astype(F64) - A.astype(F64))
    denom = np.abs(A.astype(F64)) + 1e-12
    print(f"  per-elem: max_abs={d.max():.3e} max_rel(>=1e-3 分母)={np.max(d / np.where(np.abs(A) > 1e-3, denom, np.inf)):.3e}")

    # 转置方向自检（P0 判据的"阳性对照"：真值 vs 真值ᵀ 必须能被区分）
    if A.shape == A.T.shape:
        r_same = relfro(A, A)
        r_trans = relfro(A, A.T)
        print(f"  transposition-discriminator: relFro(exact, exact)={r_same:.3e} "
              f"relFro(exact, exact.T)={r_trans:.3e} "
              f"=> {'可区分(P0 有效)' if r_trans > 100 * max(r_same, 1e-12) else '不可区分:换形状(方阵对称巧合), 上卡请用非方阵 N!=K'}")
    else:
        print(f"  transposition-discriminator: 非方阵 ({N}x{K}) → DW_kern 与 exact.T 形状都不合，"
              f"P0 直接由形状+relFro 判")

    # 语义自证：golden 必须等于 dYᵀ@X 的 torch 口径（用两种独立算法互校）
    alt = np.einsum("tn,tk->nk", dY.astype(F64), X.astype(F64)).astype(F32)
    print(f"  einsum-vs-matmul: relFro={relfro(alt, A):.3e} (必须 0/1e-16 量级)")

    print("  ---- 上卡判据（照抄，勿放宽） ----")
    print("  P0 转置方向: relFro(DW,exact) < 1e-2 且 relFro(DW,exact) < relFro(DW,exact.T)")
    print(f"  P1 累加精度: relFro(DW,tiled) <= 2e-6 ; relFro(DW,exact) <= {max(1e-5, 10 * relfro(B, A)):.1e}")
    print("  P2 逐元素  : assert_close(DW, tiled, atol=1e-3, rtol=1e-5)  (对 exact 放宽 atol=1e-2)")
    print("  出口降 fp16: P1 阈值 5e-4, P2 atol=5e-2 rtol=5e-3")
    return A, B, C, dY, X


def emit_npz(args, A, B, C, dY, X):
    import os
    path = args.npz
    np.savez(path, dY=dY, X=X, exact=A, exact_T=A.T.copy(), tiled=B, fp16out=C,
             T=np.array(args.T), N=np.array(args.N), K=np.array(args.K),
             TT=np.array(args.TT), BN=np.array(128), BK=np.array(128),
             p1_tiled=np.array(2e-6), p1_exact=np.array(1e-5),
             atol=np.array(1e-3), rtol=np.array(1e-5))
    print(f"D-GOLDEN-NPZ {path} ({os.path.getsize(path)} bytes) keys=dY,X,exact,exact_T,tiled,fp16out,T,N,K,TT,BN,BK,p1_*,atol,rtol")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--TT", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--npz", default="")
    args = ap.parse_args()
    A, B, C, dY, X = selfcheck(args)
    if args.npz:
        emit_npz(args, A, B, C, dY, X)
    return 0


if __name__ == "__main__":
    sys.exit(main())
