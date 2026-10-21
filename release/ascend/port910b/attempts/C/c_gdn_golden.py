#!/usr/bin/env python3
"""attempts/C/c_gdn_golden.py — GDN 短卷积（causal depthwise conv1d K=4 + SiLU）CPU 语义 golden。

本机无 NPU，本件是**上卡对拍的唯一参考真值**（与 c_gdn_conv_910b.py 一一对应）：

    Y[c, t] = silu( bias[c] + SUM_{k=0..K-1} W[c, k] * X[c, t-(K-1)+k] )
    silu(z) = z * sigmoid(z) = z / (1 + exp(-z))
    左补零宽度 = K-1；窗 = x[t-(K-1) .. t]；t-(K-1)+k < 0 视为 0（因果，不镜像不周期）

与 kernel 的传参对应：
    X (C,L) f32 channels-first / W (C,K) f32 / BIAS (C,) f32 → Y (C,L) f32

三种真值同框，便于把"语义错"与"compat 数值近似"分开定位：
  A) exact   —— float64 精确式（语义真值）
  B) fp32    —— float32 逐步乘加 + numpy fp32 exp（贴近 kernel 的累加次序）
  C) compat  —— **逐行复刻** attempts/C/compat_patch_C.h 的 tl910b_expf（范围规约 +
                6 阶 Taylor + 指数域拼 2^n）与 rational 近似 SiLU。
                上卡对拍若与 B 不符而与 C 符 → 是 compat 近似；与 C 也不符 → 是搬运/
                索引 bug（优先查 G-C4 单位、G-C5 §10b 槽位）。
                本件在写 kernel 之前就先抓到过 compat exp 的一个真 bug（指数域双
                重加 127），所以 C 这条路**必须**和 compat_patch_C.h 同步维护。

用法：
    python3 c_gdn_golden.py                     # 自检（三真值互校 + exp 精度扫描 + 边界）
    python3 c_gdn_golden.py --npz out.npz --C 2048 --L 512 --K 4 --seed 0
                                                # 产出上卡对拍的输入/期望输出
"""
import argparse
import sys

import numpy as np

K_DOC = "GDN/Qwen3-Next 短卷积约定 K=4"


# ── compat 复刻（与 compat_patch_C.h 的 tl910b_expf 逐步同序） ──────────────
_F2U = lambda v: np.frombuffer(np.float32(v).tobytes(), dtype=np.uint32)[0]  # noqa: E731
_U2F = lambda u: np.frombuffer(np.uint32(u & 0xFFFFFFFF).tobytes(), dtype=np.float32)[0]  # noqa: E731
FLT_MAX = np.float32(3.4028234663852886e38)


def tl910b_expf(x):
    """复刻 GAP-C 软件 expf：x = n*ln2 + r，6 阶 Taylor 求 e^r∈[1,2)，指数域拼 2^n。"""
    flat = np.asarray(x, dtype=np.float32).ravel().astype(np.float32).copy()
    kLog2e = np.float32(1.44269504088896340736)
    kLn2 = np.float32(0.69314718055994530942)
    for i in range(flat.size):
        v = flat[i]
        if v <= np.float32(-104.0):
            flat[i] = np.float32(0.0)
            continue
        if v >= np.float32(88.0):
            flat[i] = FLT_MAX
            continue
        t = np.float32(v * kLog2e)
        n = int(t + 0.5) if t >= 0 else int(t - 0.5)     # C 侧 static_cast<int> 亦截断
        r = np.float32(v - np.float32(n) * kLn2)
        p = np.float32(1.0 + r * (1.0 + r * (0.5 + r * (0.16666666666666666 +
                    r * (0.041666666666666664 + r * (0.008333333333333333 +
                    r * 0.001388888888888889))))))
        if p < np.float32(1.0):
            p = np.float32(p + p)
            n -= 1
        if n < -126:
            flat[i] = np.float32(0.0)
            continue
        if n > 127:
            flat[i] = FLT_MAX
            continue
        mant = _F2U(p)                      # p∈[1,2) ⇒ 指数域已是 127，故只需再叠 n
        flat[i] = np.float32(_U2F(mant - np.uint32(-n << 23)) if n < 0
                             else _U2F(mant + np.uint32(n << 23)))
    return flat.reshape(np.asarray(x).shape)


def tl910b_rational_sigmoid(u):
    """compat --silu rational 的 sigmoid(u) ≈ 0.5*(u/(1+|u|)+1)。"""
    u = np.float32(u)
    return np.float32(0.5 * (u / np.float32(1.0 + np.abs(u)) + 1.0))


# ── 三种真值 ────────────────────────────────────────────────────────────────
def _conv_z(x, w, bias, dtype):
    """公共前段：z[c,t] = bias[c] + sum_k w[c,k]*xp[c,t+k]（xp 左补 pad 个 0）。"""
    C, L = x.shape
    K = w.shape[1]
    xp = np.pad(x.astype(dtype), ((0, 0), (K - 1, 0)), mode="constant")
    z = np.zeros((C, L), dtype=dtype)
    for c in range(C):
        for t in range(L):
            acc = dtype(0)
            for k in range(K):
                acc = dtype(acc + dtype(w[c, k]) * dtype(xp[c, t + k]))
            z[c, t] = dtype(acc + dtype(bias[c]))
    return z


def short_conv_exact(x, w, bias):
    """A) 语义真值（float64 + 精确 sigmoid）。"""
    z = _conv_z(x, w, bias, np.float64)
    return z / (1.0 + np.exp(-z))


def short_conv_fp32(x, w, bias):
    """B) fp32 逐步乘加 + fp32 精确 exp（kernel 的累加次序，不含 compat 近似）。"""
    z = _conv_z(x, w, bias, np.float32)
    return (z / np.float32(1.0 + np.float32(np.exp(np.float32(-z))))).astype(np.float32)


def short_conv_compat(x, w, bias, silu="exp"):
    """C) kernel 落地真值：SiLU 走 compat 的 exp / rational 路径。"""
    z = _conv_z(x, w, bias, np.float32)
    if silu == "exp":
        return (z / np.float32(1.0 + tl910b_expf(-z))).astype(np.float32)
    return (z * tl910b_rational_sigmoid(-z)).astype(np.float32)


def short_conv_torch(x, w, bias):
    """D) torch F.conv1d 交叉验证（channels-first depthwise + 因果左补零 + SiLU）。

    这就是 910B 上 torch_npu Conv2D 崩板的那条路；本机（CPU torch）可用，作为
    "语义锚点"证明 A/B/C 的窗方向与 groups 约定无误。
    """
    try:
        import torch
        import torch.nn.functional as F
    except Exception:  # noqa: BLE001
        return None
    C, L = x.shape
    K = w.shape[1]
    xt = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)        # (1,C,L)
    wt = torch.from_numpy(w.astype(np.float32)).unsqueeze(1)        # (C,1,K) groups=C
    bt = torch.from_numpy(bias.astype(np.float32))
    z = F.conv1d(F.pad(xt, (K - 1, 0)), wt, bt, groups=C)[0]        # (C,L)
    y = z * torch.sigmoid(z)
    return y.numpy()


# ── 度量与自检 ──────────────────────────────────────────────────────────────
def _errs(ref, got):
    """返回 (max_abs, max_rel_on|ref|>=1)。"""
    ref = ref.astype(np.float64)
    got = got.astype(np.float64)
    d = np.abs(ref - got)
    big = np.abs(ref) >= 1.0
    rel = float((d[big] / np.abs(ref[big])).max()) if big.any() else 0.0
    return float(d.max()), rel


def exp_accuracy_sweep():
    """compat exp 复刻的精度实测（上卡对拍容差的直接来源）。"""
    xs = np.linspace(-30.0, 30.0, 601).astype(np.float32)
    got = tl910b_expf(xs)
    ref = np.exp(xs.astype(np.float64))
    rel = np.abs(got - ref) / ref
    ok = float(rel.max()) < 1e-5
    print("EXP-SWEEP |x|<=30: max_rel_err=%.3e %s" % (rel.max(), "OK" if ok else "FAIL"))
    # 边界行为
    b = [("expf(-200)", float(tl910b_expf(np.float32(-200.0)))),
         ("expf(200)", float(tl910b_expf(np.float32(200.0)))),
         ("expf(-104)", float(tl910b_expf(np.float32(-104.0)))),
         ("expf(88)", float(tl910b_expf(np.float32(88.0)))),
         ("expf(0)", float(tl910b_expf(np.float32(0.0))))]
    print("EXP-EDGE  " + "  ".join("%s=%g" % (k, v) for k, v in b))
    return ok


def exp_max_rel():
    """compat exp 复刻的实测最大相对误差（供 npz 提示语引用，避免硬编码）。"""
    xs = np.linspace(-30.0, 30.0, 601).astype(np.float32)
    ref = np.exp(xs.astype(np.float64))
    return float((np.abs(tl910b_expf(xs) - ref) / ref).max())


def selfcheck():
    ok = exp_accuracy_sweep()
    cases = [("typical", 8, 64, 4, 0), ("L==K", 4, 4, 4, 1), ("L<K", 3, 2, 4, 2),
             ("long", 5, 512, 4, 3), ("zerow", 6, 32, 4, 4), ("bigz", 4, 24, 4, 5)]
    for name, C, L, K, seed in cases:
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((C, L)).astype(np.float32) * (8.0 if name == "bigz" else 2.0)
        w = rng.standard_normal((C, K)).astype(np.float32) * 0.5
        if name == "zerow":
            w = np.zeros_like(w)
        bias = rng.standard_normal((C,)).astype(np.float32) * 0.1
        ea = short_conv_exact(x, w, bias)
        eb = short_conv_fp32(x, w, bias)
        ec = short_conv_compat(x, w, bias, "exp")
        er = short_conv_compat(x, w, bias, "rational")
        et = short_conv_torch(x, w, bias)
        ab, rb = _errs(ea, eb)
        ac, rc = _errs(ea, ec)
        ar, _ = _errs(ea, er)
        line = ("%s(C=%d,L=%d,K=%d) fp32:abs %.2e/rel %.2e | compatexp:abs %.2e/rel %.2e"
                " | rational(仅参考):abs %.2e" % (name, C, L, K, ab, rb, ac, rc, ar))
        if et is not None:
            at, rt = _errs(ea, et)
            line += " | torch:abs %.2e/rel %.2e" % (at, rt)
            ok = ok and rt < 1e-5
        print(line)
        ok = ok and rb < 1e-5 and rc < 1e-5
        # 因果左边界硬校验：t=0 只该吃 x[c,0]*w[c,K-1]
        z0 = (x[:, 0].astype(np.float64) * w[:, K - 1].astype(np.float64)
              + bias.astype(np.float64))
        ref0 = z0 / (1 + np.exp(-z0))
        if not np.allclose(ea[:, 0], ref0, rtol=1e-12, atol=1e-12):
            print("  !! t=0 边界不符（应为 bias + w[:,K-1]*x[:,0]）")
            ok = False
        # 全零权重 ⇒ 输出必须恒等于 silu(bias)
        if name == "zerow":
            silu_b = (bias / (1 + np.exp(-bias.astype(np.float32)))).astype(np.float32)
            if not np.array_equal(eb, np.repeat(silu_b[:, None], L, axis=1)):
                print("  !! zerow 用例：全零权重未退化为 silu(bias)")
                ok = False
    print("GOLDEN-SELFCHK " + ("OK" if ok else "FAIL"))
    return ok


def emit_npz(path, C, L, K, seed):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((C, L)) * 2.0).astype(np.float32)
    w = (rng.standard_normal((C, K)) * 0.5).astype(np.float32)
    bias = (rng.standard_normal((C,)) * 0.1).astype(np.float32)
    out = dict(X=x, W=w, BIAS=bias,
               Y_FP32=short_conv_fp32(x, w, bias),
               Y_COMPAT_EXP=short_conv_compat(x, w, bias, "exp"),
               Y_COMPAT_RATIONAL=short_conv_compat(x, w, bias, "rational"),
               META=np.array([C, L, K, seed]))
    np.savez(path, **out)
    print("GOLDEN-EMITTED %s X%s W%s BIAS%s -> Y%s + Y_FP32/Y_COMPAT_EXP/Y_COMPAT_RATIONAL"
          % (path, x.shape, w.shape, bias.shape, out["Y_FP32"].shape))
    print("  建议对拍判据: allclose(Y_kernel, Y_COMPAT_EXP, rtol=2e-5, atol=2e-5)；"
          "实测 compat exp 精度见上方 EXP-SWEEP 行（本次 max_rel_err=%.2e，"
          "2e-5 约有 18 倍余量）" % exp_max_rel())


def main():
    ap = argparse.ArgumentParser(description=__doc__[:400])
    ap.add_argument("--npz", default="", help="产出对拍张量到该路径")
    ap.add_argument("--C", type=int, default=2048)
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.K != 4:
        print("NOTE: 本波 kernel 硬断言 K==4（%s）" % K_DOC)
    ok = selfcheck()
    if args.npz:
        emit_npz(args.npz, args.C, args.L, args.K, args.seed)
    print("GOLDEN-PASS" if ok else "GOLDEN-FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
