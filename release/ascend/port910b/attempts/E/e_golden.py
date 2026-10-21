#!/usr/bin/env python3
"""attempts/E/e_golden.py — rope / attn_sw 的 CPU 语义 golden（numpy，无卡可跑，上卡对拍判据源）。

三件套纪律（承 C 波）：**compat 软件件与 golden 复刻体成对维护**——本件含
`tl910b_expf` 的逐行 numpy 复刻（compat v3 §11，真源 port910b_compat.h:724），
kernel 数值验收对标的是"复刻路径真值"，它已经 C 波定标 max_rel_err=1.103e-06。

rope 语义（track-A 口径，rope_ref.py/rope_asc.py 双向锚定）：
    rotate-half：x1' = x1*c - x2*s；x2' = x2*c + x1*s（c/s=预计算表 (tokens,half)）
    slot2=V 原样透传；sign=-1 走同式（逆旋转=sin 取负），正反对合应回到原值。
attn_sw 语义（track-A 口径，attn_sw_asc.py/_eager）：
    查询 i 可见键 kk ∈ [max(0,i-window+1), i]；score=scale*(Qi·Kk)；
    行内 softmax（减最大→exp→归一）→ 加权 V；不可见键权重严格 0。

上卡判据（写死在本件输出里，卡边直接抄）：
  R1 rope  : allclose(O_kernel[:,0:2], GOLD_ROPE_F32, rtol=2e-5, atol=2e-5)（纯四则，
             误差预算只含 fp32 舍入与潜在 FMA 收缩）
  R2 rope  : O_kernel[:,2] == QKV[:,2] **按位相等**（V 透传，结构性判据）
  R3 rope  : 反对合 kernel 链（sign=+1 件 ∘ sign=-1 件）‖x-x0‖inf <= 5e-6*‖x0‖inf
  R4 attn  : allclose(O_kernel, GOLD_ATTN_F32（复刻路径）, rtol=2e-5, atol=2e-5)
             （预算含 expf 1.1e-6 相对误差与行和放大，取 2 倍余量）
  R5 attn  : 不可见扰动不变性——逐行逐头把该行不可见键的 K/V 加有限扰动，
             该行输出**按位不变**（哨兵 expf→0 的结构性零权重；对 kernel 同样成立）
  R6 attn  : 行分和 l_kernel ∈ [1, window]（自见格 exp(0)=1 下界；哨兵格恰贡献 0）

自检口径（本件内部）：sim-vs-f64 的 rel 用 1%|ref|max 分母地板（实测 naive 口径在
近零输出分量上报 8e-5~9e-5 假阳，垫地板后 attn rel=3.6e-6 / 且 swexp 与 true-exp 两路
误差逐位同级，证明是度量伪影而非 expf/语义误差——分离取证件见 _dbg_e.py 同口径复跑）。
用法：python3 e_golden.py [--seed 7] [--npz /tmp/e_pairs.npz]
判据行：GOLDEN-ROPE-PASS / GOLDEN-ATTN-PASS / E-GOLDEN-PASS
"""
import argparse
import struct
import sys

import numpy as np

FLT_MAX = np.float32(3.4028234663852886e38)
NEG = np.float32(-1.0e30)
KLOG2E = np.float32(1.44269504088896340736)
KLN2 = np.float32(0.69314718055994530942)


def f2u(x):
    return struct.unpack("<I", struct.pack("<f", np.float32(x)))[0]


def u2f(u):
    return np.float32(struct.unpack("<f", struct.pack("<I", int(u) & 0xFFFFFFFF))[0])


def expf_sw(x):
    """compat §11 tl910b_expf 的逐行 numpy 复刻（float32 位语义）。"""
    x = np.float32(x)
    if x <= np.float32(-104.0):
        return np.float32(0.0)
    if x >= np.float32(88.0):
        return FLT_MAX
    t = np.float32(x * KLOG2E)
    n = int(np.floor(t + np.float32(0.5))) if t >= 0 else int(np.ceil(t - np.float32(0.5)))
    r = np.float32(x - np.float32(n) * KLN2)
    p = np.float32(1.0) + r * (np.float32(1.0) + r * (np.float32(0.5) + r * (
        np.float32(0.16666666666666666) + r * (np.float32(0.041666666666666664) + r * (
            np.float32(0.008333333333333333) + r * np.float32(0.001388888888888889))))))
    if p < 1.0:
        p = np.float32(p + p)
        n -= 1
    if n < -126:
        return np.float32(0.0)
    if n > 127:
        return FLT_MAX
    mant = f2u(p)
    if n < 0:
        return u2f(mant - (np.uint32(-n) << np.uint32(23)))
    return u2f(mant + (np.uint32(n) << np.uint32(23)))


# ─────────────────────────── rope ───────────────────────────

def rope_tables(tokens, half, theta=10000.0):
    """生产 rope_angle_tables 同式：freq_i=theta^(-2i/D)，pos×freq → cos/sin fp32 分表。"""
    i = np.arange(half, dtype=np.float64)
    freq = np.float32(theta ** (-2.0 * i / (2.0 * half)))
    pos = np.arange(tokens, dtype=np.float32)
    ang = np.float32(pos[:, None] * freq[None, :])
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def rope_ref_f64(qkv, cos, sin, sign=1):
    """float64 精确式（真值锚）。"""
    half = qkv.shape[-1] // 2
    out = qkv.astype(np.float64).copy()
    c = cos.astype(np.float64)[:, None, :]      # (tokens,1,half)
    s = sin.astype(np.float64)[:, None, :] * sign
    for slot in (0, 1):
        x1 = out[:, slot, :, :half].copy()
        x2 = out[:, slot, :, half:].copy()
        out[:, slot, :, :half] = x1 * c - x2 * s
        out[:, slot, :, half:] = x2 * c + x1 * s
    return out


def rope_kernel_sim(qkv, cos, sin, sign=1):
    """kernel 发射顺序复刻：fp32 标量四则，先物化 x1/x2 再写两路（与 e_rope_910b 同序）。"""
    half = qkv.shape[-1] // 2
    out = qkv.astype(np.float32).copy()
    sgn = np.float32(sign)
    for t in range(qkv.shape[0]):
        for slot in (0, 1):
            for h in range(qkv.shape[2]):
                for j in range(half):
                    x1 = np.float32(qkv[t, slot, h, j])
                    x2 = np.float32(qkv[t, slot, h, j + half])
                    c = np.float32(cos[t, j])
                    sn = np.float32(sgn * sin[t, j])
                    out[t, slot, h, j] = np.float32(x1 * c - x2 * sn)
                    out[t, slot, h, j + half] = np.float32(x2 * c + x1 * sn)
    out[:, 2] = qkv[:, 2]                        # V 透传
    return out


def check_rope(rng, tokens=16, heads=4, dim=32, theta=10000.0):
    half = dim // 2
    qkv = rng.standard_normal((tokens, 3, heads, dim)).astype(np.float32)
    cos, sin = rope_tables(tokens, half, theta)
    ref64 = rope_ref_f64(qkv, cos, sin, +1)
    sim32 = rope_kernel_sim(qkv, cos, sin, +1)
    err = np.abs(sim32.astype(np.float64) - ref64)
    # rel 口径：分母垫 1%|ref|max 地板（近零分量的 naive rel 是除零假阳，C 波同类坑；
    # naive 口径同时打印留痕）。上卡判据 R1/R4 用 allclose(rtol=atol=2e-5)，不受此口径影响。
    denom = np.maximum(np.abs(ref64), 1e-2 * np.abs(ref64).max())
    abs_e = float(err.max())
    rel_e = float((err / denom).max())
    rel_naive = float((err / np.maximum(np.abs(ref64), 1e-12)).max())
    v_exact = bool(np.array_equal(sim32[:, 2], qkv[:, 2]))     # 结构性：V 按位透传
    rt = rope_kernel_sim(sim32, cos, sin, -1)                  # 反对合（kernel 两件链）
    rt_err = float(np.abs(rt.astype(np.float64) - qkv.astype(np.float64)).max())
    rt_scale = float(np.abs(qkv).max())
    ok = (rel_e <= 2e-5) and (abs_e <= 2e-5) and v_exact and (rt_err <= 5e-6 * max(rt_scale, 1.0))
    print(f"ROPE tokens={tokens} heads={heads} dim={dim} theta={theta:g}")
    print(f"  sim32-vs-f64: abs={abs_e:.3e} rel={rel_e:.3e} (预算 2e-5) [naive-rel={rel_naive:.3e}，近零分量假阳，仅留痕]")
    print(f"  V-slot 按位透传: {v_exact}")
    print(f"  正反旋转对合 max|Δ|={rt_err:.3e} (预算 {5e-6 * max(rt_scale, 1.0):.3e})")
    print("  GOLDEN-ROPE-" + ("PASS" if ok else "FAIL"))
    return ok, dict(qkv=qkv, cos=cos, sin=sin, rope_f32=sim32, rope_f64=ref64)


# ─────────────────────────── attn_sw ───────────────────────────

def attn_row(q, k, v, hh, i, window, scale):
    """e_attn_sw stage 变体单行复刻：哨兵窗 + 复刻 expf + 两步归一。返回 (o_row, ps, l)。"""
    dim = q.shape[-1]
    lo = max(0, i - window + 1)
    n = i - lo + 1
    sc = []
    for j in range(window):
        if j < n:
            acc = np.float32(0.0)
            qh, kh = q[hh, i].astype(np.float32), k[hh, lo + j].astype(np.float32)
            for c in range(dim):
                acc = np.float32(acc + np.float32(qh[c] * kh[c]))
            sc.append(np.float32(scale) * acc)
        else:
            sc.append(NEG)
    m = np.float32(NEG)
    for x in sc:
        m = np.float32(max(m, x))
    ps = [expf_sw(np.float32(x - m)) for x in sc]
    l = np.float32(0.0)
    for p in ps:
        l = np.float32(l + p)
    o = np.zeros(dim, dtype=np.float32)
    for c in range(dim):
        acc = np.float32(0.0)
        for j in range(window):
            acc = np.float32(acc + np.float32(ps[j] * v[hh, lo + j, c]))
        o[c] = np.float32(acc / l)
    return o, ps, l


def attn_ref_f64(q, k, v, window, scale):
    """全量带掩码 softmax 精确式（真值锚，_eager/forward_weights 同口径）。"""
    heads, seq, dim = q.shape
    o = np.zeros((heads, seq, dim), dtype=np.float64)
    for hh in range(heads):
        score = q[hh].astype(np.float64) @ k[hh].astype(np.float64).T * scale
        for i in range(seq):
            lo = max(0, i - window + 1)
            m = score[i, lo:i + 1].max()
            w = np.zeros(seq)
            w[lo:i + 1] = np.exp(score[i, lo:i + 1] - m)
            w /= w.sum()
            o[hh, i] = w @ v[hh].astype(np.float64)
    return o


def check_attn(rng, heads=4, seq=24, dim=16, window=8):
    scale = dim ** -0.5
    q = rng.standard_normal((heads, seq, dim)).astype(np.float32)
    k = rng.standard_normal((heads, seq, dim)).astype(np.float32)
    v = rng.standard_normal((heads, seq, dim)).astype(np.float32)
    ref64 = attn_ref_f64(q, k, v, window, scale)
    sim32 = np.zeros((heads, seq, dim), dtype=np.float32)
    l_rows = np.zeros((heads, seq), dtype=np.float32)
    zero_w_ok = True                              # 窗外哨兵权重恰 0.0
    for hh in range(heads):
        for i in range(seq):
            o, ps, l = attn_row(q, k, v, hh, i, window, scale)
            sim32[hh, i], l_rows[hh, i] = o, l
            n = i - max(0, i - window + 1) + 1
            if any(p != 0.0 for p in ps[n:]):
                zero_w_ok = False
    err = np.abs(sim32.astype(np.float64) - ref64)
    denom = np.maximum(np.abs(ref64), 1e-2 * np.abs(ref64).max())
    abs_e = float(err.max())
    rel_e = float((err / denom).max())
    rel_naive = float((err / np.maximum(np.abs(ref64), 1e-12)).max())
    # 不可见扰动不变性（逐行逐头，全局扰动不成立的口径修正见 RESULT §③）
    perturb_exact = True
    for hh in range(heads):
        for i in range(seq):
            lo = max(0, i - window + 1)
            if lo == 0:
                continue
            k2, v2 = k.copy(), v.copy()
            k2[hh, :lo] += np.float32(3.0)
            v2[hh, :lo] -= np.float32(5.0)
            o2, _, _ = attn_row(q, k2, v2, hh, i, window, scale)
            if not np.array_equal(o2, sim32[hh, i]):
                perturb_exact = False
    l_ok = bool((l_rows >= 1.0 - 1e-6).all() and (l_rows <= window + 1e-4).all())
    ok = (rel_e <= 2e-5) and (abs_e <= 2e-5) and zero_w_ok and perturb_exact and l_ok
    print(f"ATTN heads={heads} seq={seq} dim={dim} window={window} scale={scale:.4f}")
    print(f"  sim32(复刻expf)-vs-f64: abs={abs_e:.3e} rel={rel_e:.3e} (预算 2e-5) [naive-rel={rel_naive:.3e}，近零分量假阳，仅留痕]")
    print(f"  窗外权重恰0={zero_w_ok} / 不可见扰动该行输出按位不变={perturb_exact}")
    print(f"  行分和 l∈[1,window]: {l_ok} (min={l_rows.min():.4f} max={l_rows.max():.4f})")
    print("  GOLDEN-ATTN-" + ("PASS" if ok else "FAIL"))
    return ok, dict(q=q, k=k, v=v, attn_f32=sim32, attn_f64=ref64, l_rows=l_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--npz", default="")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    ok_r, dr = check_rope(rng)
    ok_a, da = check_attn(rng)
    if args.npz:
        np.savez(args.npz,
                 QKV=dr["qkv"], COS=dr["cos"], SIN=dr["sin"],
                 ROPE_F32=dr["rope_f32"], ROPE_F64=dr["rope_f64"],
                 AQ=da["q"], AK=da["k"], AV=da["v"],
                 ATTN_F32=da["attn_f32"], ATTN_F64=da["attn_f64"], L_ROWS=da["l_rows"])
        print(f"GOLDEN-EMITTED {args.npz}")
    print("E-GOLDEN-" + ("PASS" if (ok_r and ok_a) else "FAIL"))
    return 0 if (ok_r and ok_a) else 1


if __name__ == "__main__":
    sys.exit(main())
