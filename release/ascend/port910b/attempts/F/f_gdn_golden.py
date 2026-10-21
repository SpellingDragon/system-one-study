#!/usr/bin/env python3
"""attempts/F/f_gdn_golden.py — GDN delta rule 前向/反向的 **CPU 分步对拍**件（无卡可跑，numpy-only）。

三段独立判决（缺任何一段都不算证真）：
  [A] 前向语义：`fwd_kernel_order`（fp32，**逐 op 复刻 DSL 发射次序**）
      vs `fwd_exact`（float64 精确递推）→ rel(O)、rel(S_final)、rel(HIST)。
  [B] 反向公式正确性：`bwd_adjoint`（float64，本波推导的伴随递推）
      vs **中心差分有限梯度检查**（∂L/∂x，L=Σ w·O 随机线性泛函，对 Q/K/V/BETA/G/H0 全体）
      → 这是与推导过程**独立**的锚（不是"我自己推的公式对我自己推的公式"）。
      torch 可用时再叠一路 autograd（第三条独立锚，同一 fp64 前向图）。
  [C] fp32 + 910B 软件 expf 的真实漂移：`bwd_adjoint(order=ub/pure)` fp32（decay 走
      `tl910b_expf` 的**逐行 numpy 复刻**，与 trunk `port910b_compat.h:724` 成对维护）
      vs fp64 真值 → 给出上卡可期望的 rel 量级；并单独量化"递推链把 expf 的
      ~1e-6 相对误差放大成什么"（decay 是连乘 ⇒ 误差随 T 线性累积，见 EXPF-DRIFT 行）。

次序纪律（[C] 的意义所在）：kernel 里 pred 吃的是**已衰减**的 S、o 吃的是**已更新**的 S，
且 dq/dk/da 的累加嵌套次序（ub=i 外 j 内 / pure=j 外 i 内）决定 fp32 求和顺序。
⇒ 复刻体必须与 DSL 同序，否则 rel 数字没有解释力（C 波 §2 的 expf 双加 127 就是这么抓到的）。

判据行：`F-GDN-GOLDEN-FWD rel=…` / `F-GDN-GOLDEN-BWD-ADJOINT rel=…` /
        `F-GDN-GOLDEN rel=…`（总体）/ `F-GDN-GOLDEN-FAIL …`（任一超阈）。
用法：python3 f_gdn_golden.py [--H 8] [--T 32] [--DK 16] [--DV 16] [--fd 1]
                             [--npz /tmp/f_gdn_pairs.npz] [--seed 7]
"""
import argparse
import sys

import numpy as np

TOL_F32 = 2e-4      # fp32 递推 + 软件 expf 的放宽阈（实测远低于此，见判决行）
TOL_ADJ = 1e-8      # fp64 伴随 vs 有限差分
FLT_MAX = np.float32(3.402823466385286e38)

# ── trunk port910b_compat.h §11 `tl910b_expf` 的逐行 numpy 复刻（成对维护，禁改语义）──
_F2U = lambda v: np.frombuffer(np.float32(v).tobytes(), dtype=np.uint32)[0]  # noqa: E731
_U2F = lambda u: np.frombuffer(np.uint32(int(u) & 0xFFFFFFFF).tobytes(), dtype=np.float32)[0]  # noqa: E731


def tl910b_expf(x):
    xs = np.asarray(x, dtype=np.float32)
    flat = xs.ravel().astype(np.float32).copy()
    kLog2e = np.float32(1.44269504088896340736)
    kLn2 = np.float32(0.69314718055994530942)
    for i in range(flat.size):
        v = np.float32(flat[i])
        if v <= np.float32(-104.0):
            flat[i] = np.float32(0.0); continue
        if v >= np.float32(88.0):
            flat[i] = FLT_MAX; continue
        t = np.float32(v * kLog2e)
        n = int(t + np.float32(0.5)) if t >= 0 else int(t - np.float32(0.5))
        r = np.float32(v - np.float32(n) * kLn2)
        p = np.float32(1.0 + r * (1.0 + r * (0.5 + r * (0.16666666666666666 +
                    r * (0.041666666666666664 + r * (0.008333333333333333 +
                    r * 0.001388888888888889))))))
        if p < np.float32(1.0):
            p = np.float32(p + p); n -= 1
        if n < -126:
            flat[i] = np.float32(0.0); continue
        if n > 127:
            flat[i] = FLT_MAX; continue
        mant = _F2U(p)
        flat[i] = np.float32(_U2F(mant - (np.uint32(-n) << 23)) if n < 0
                             else _U2F(mant + (np.uint32(n) << 23)))
    return flat.reshape(xs.shape)


# ══════════════════════════════════════ 前向 ══════════════════════════════════════
def fwd_exact(H, T, DK, DV, Q, K, V, BETA, G, H0):
    """float64 精确递推（语义真值）。返回 O, HIST(=S_{t−1}), S_final。"""
    a_f = np.exp(G)
    O = np.zeros((H, T, DV), np.float64)
    HIST = np.zeros((H, T, DV, DK), np.float64)
    Sf = np.zeros((H, DV, DK), np.float64)
    for h in range(H):
        st = H0[h].astype(np.float64, copy=True)
        for t in range(T):
            HIST[h, t] = st
            a = a_f[h, t]
            sp = st * a
            pred = sp @ K[h, t]                     # [DV]
            uu = BETA[h, t] * (V[h, t] - pred)
            snew = sp + np.outer(uu, K[h, t])
            O[h, t] = snew @ Q[h, t]
            st = snew
        Sf[h] = st
    return O, HIST, Sf


def fwd_kernel_order(H, T, DK, DV, Q, K, V, BETA, G, H0, expf, dt):
    """**逐 op 复刻 DSL 发射次序**的 fp32 前向（= kernel 应产出什么）。

    对应 f_gdn_delta_fwd_910b.py：per (h,j) 列，趟1 `st*=a; pred+=st*k`（同趟融合），
    趟2 `st+=k*u; o+=st*q`（同趟融合）。j 是并行单元 ⇒ j 之间无归约，次序无关。
    """
    O = np.zeros((H, T, DV), dt)
    HIST = np.zeros((H, T, DV, DK), dt)
    Sf = np.zeros((H, DV, DK), dt)
    for h in range(H):
        for j in range(DV):
            st = H0[h, j].astype(dt, copy=True)
            for t in range(T):
                a = dt(expf(np.array([G[h, t]], dt))[0])
                bt = dt(BETA[h, t]); vj = dt(V[h, t, j])
                HIST[h, t, j] = st
                pred = dt(0.0)
                kt = K[h, t].astype(dt, copy=False)
                for i in range(DK):
                    st[i] = dt(st[i] * a)
                    pred = dt(pred + st[i] * kt[i])
                uu = dt(bt * (vj - pred))
                o = dt(0.0)
                qt = Q[h, t].astype(dt, copy=False)
                for i in range(DK):
                    st[i] = dt(st[i] + kt[i] * uu)
                    o = dt(o + st[i] * qt[i])
                O[h, t, j] = o
            Sf[h, j] = st
    return O, HIST, Sf


# ══════════════════════════════════════ 反向（伴随递推）══════════════════════════════════════
def bwd_adjoint(H, T, DK, DV, Q, K, V, BETA, G, HIST, dO, dS_final=None,
                expf=np.exp, dt=np.float64, order="ub"):
    """delta rule 反向的伴随递推实现（本波推导，与 DSL 同序）。

    前向：S'_t=a·S_{t−1}; p_t=S'_tᵀk_t; u_t=β(v_t−p_t); S_t=S'_t+k_t⊗u_t; o_t=S_tᵀq_t
    反向（Λ_t=∂L/∂S_t，t 递减；Λ^tot=Λ_{t+1}+dO_t⊗q_t）：
      dq_i=Σ_j S_t[j,i]dO_j ; du_j=Σ_i Λ^tot[j,i]k_i ; dp_j=−β·du_j
      dv_j=β·du_j ; dβ=Σ_j du_j(v_j−p_j) ; dk_i=Σ_j(u_j Λ^tot[j,i]+S'_t[j,i]dp_j)
      Λ_{t−1}[j,i]=a(Λ^tot[j,i]+k_i dp_j) ; da=Σ_{j,i}S_{t−1}[j,i]·(Λ^tot[j,i]+k_i dp_j) ; dg=da·a
    order="ub"   — 趟1(j 外，算 u/dp 落暂存) + 趟2(i 外 j 内，Λ 就地递推) → 同 f_gdn_delta_bwd_910b.py ub
    order="pure" — j 外单趟融合（du 只读未更新的第 j 行）+ dq/dk 跨 j 累加 → 同 pure 变体
    """
    a_all = expf(np.asarray(G, dt))
    dqa = np.zeros((H, T, DK), dt)
    dka = np.zeros((H, T, DK), dt)
    dva = np.zeros((H, T, DV), dt)
    dba = np.zeros((H, T), dt)
    dga = np.zeros((H, T), dt)
    dh0 = np.zeros((H, DV, DK), dt)
    for h in range(H):
        lam = (np.zeros((DV, DK), dt) if dS_final is None
               else np.asarray(dS_final[h], dt).astype(dt, copy=True))
        for tt in range(T):
            t = T - 1 - tt
            a = dt(a_all[h, t]); bt = dt(BETA[h, t])
            kt = np.asarray(K[h, t], dt); qt = np.asarray(Q[h, t], dt)
            vt = np.asarray(V[h, t], dt); dot = np.asarray(dO[h, t], dt)
            hist = np.asarray(HIST[h, t], dt)
            u = np.zeros(DV, dt); dp = np.zeros(DV, dt)
            db = dt(0.0); da = dt(0.0)
            if order == "ub":
                for j in range(DV):                    # 趟 1
                    sp = hist[j] * a
                    p = dt(np.dot(sp, kt))
                    u[j] = dt(bt * (vt[j] - p))
                    du = dt(np.dot(lam[j] + dot[j] * qt, kt))
                    dp[j] = dt(-bt * du)
                    dva[h, t, j] = dt(bt * du)
                    db = dt(db + du * (vt[j] - p))
                for i in range(DK):                    # 趟 2
                    acc_q = dt(0.0); acc_k = dt(0.0)
                    for j in range(DV):
                        sp = hist[j, i] * a
                        acc_q = dt(acc_q + (sp + u[j] * kt[i]) * dot[j])
                        lamt = dt(lam[j, i] + dot[j] * qt[i])   # Λ^tot = Λ_{t+1}+dO⊗q
                        acc_k = dt(acc_k + u[j] * lamt + sp * dp[j])
                        dst = lamt + kt[i] * dp[j]
                        da = dt(da + hist[j, i] * dst)
                        lam[j, i] = dt(a * dst)
                    dqa[h, t, i] = acc_q; dka[h, t, i] = acc_k
            else:                                      # pure：j 外单趟
                for j in range(DV):
                    sp_row = hist[j] * a
                    p = dt(np.dot(sp_row, kt))
                    du = dt(np.dot(lam[j] + dot[j] * qt, kt))
                    u[j] = dt(bt * (vt[j] - p)); dp[j] = dt(-bt * du)
                    dva[h, t, j] = dt(bt * du)
                    db = dt(db + du * (vt[j] - p))
                    for i in range(DK):
                        sp = sp_row[i]
                        lamji = dt(lam[j, i] + dot[j] * qt[i])  # Λ^tot
                        acc_q = (sp + u[j] * kt[i]) * dot[j]
                        acc_k = u[j] * lamji + sp * dp[j]
                        dst = lamji + kt[i] * dp[j]
                        dqa[h, t, i] = dt(dqa[h, t, i] + acc_q) if j else dt(acc_q)
                        dka[h, t, i] = dt(dka[h, t, i] + acc_k) if j else dt(acc_k)
                        da = dt(da + hist[j, i] * dst)
                        lam[j, i] = dt(a * dst)
            dba[h, t] = db
            dga[h, t] = dt(da * a)
        dh0[h] = lam
    return dqa, dka, dva, dba, dga, dh0


def fwd_scalar(L_args):
    """把 (Q,K,V,BETA,G,H0) 打平成一个向量 → L=Σ w·O（有限差分/autograd 的公共口径）。"""
    H, T, DK, DV, Q, K, V, BETA, G, H0, w = L_args
    O, _, _ = fwd_exact(H, T, DK, DV, Q, K, V, BETA, G, H0)
    return float(np.sum(w * O))


# ══════════════════════════════════════ 独立锚 ══════════════════════════════════════
def gradcheck_fd(args, eps=1e-6):
    """中心差分 ∂L/∂θ（float64），返回与 adjoint 同一 key 结构的梯度字典。"""
    H, T, DK, DV, Q, K, V, BETA, G, H0, w = args
    ref = {"Q": Q, "K": K, "V": V, "BETA": BETA, "G": G, "H0": H0}
    out = {}
    for name, arr in ref.items():
        g = np.zeros_like(arr, dtype=np.float64)
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            base = float(arr[idx])
            for which, sgn in (("p", 1.0), ("m", -1.0)):
                mut = {k: v.copy() for k, v in ref.items()}
                mut[name][idx] = base + sgn * eps
                out_v = fwd_scalar((H, T, DK, DV, mut["Q"], mut["K"], mut["V"],
                                    mut["BETA"], mut["G"], mut["H0"], w))
                if which == "p": fp = out_v
                else: fm = out_v
            g[idx] = (fp - fm) / (2 * eps)
        out[name] = g
    return out


def gradcheck_torch(args):
    """torch autograd（float64）第三独立锚；torch 不可用时返回 None。"""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    H, T, DK, DV, Q, K, V, BETA, G, H0, w = args
    tq = torch.tensor(Q, dtype=torch.float64, requires_grad=True)
    tk = torch.tensor(K, dtype=torch.float64, requires_grad=True)
    tv = torch.tensor(V, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(BETA, dtype=torch.float64, requires_grad=True)
    tg = torch.tensor(G, dtype=torch.float64, requires_grad=True)
    th = torch.tensor(H0, dtype=torch.float64, requires_grad=True)
    O = torch.zeros(H, T, DV, dtype=torch.float64)
    for h in range(H):
        st = th[h]
        for t in range(T):
            a = torch.exp(tg[h, t])
            sp = st * a
            pred = torch.mv(sp, tk[h, t])
            uu = tb[h, t] * (tv[h, t] - pred)
            st = sp + torch.outer(uu, tk[h, t])
            O[h, t] = torch.mv(st, tq[h, t])
    L = torch.sum(torch.tensor(w, dtype=torch.float64) * O)
    L.backward()
    return {"Q": tq.grad.numpy(), "K": tk.grad.numpy(), "V": tv.grad.numpy(),
            "BETA": tb.grad.numpy(), "G": tg.grad.numpy(), "H0": th.grad.numpy()}


def rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    den = max(float(np.max(np.abs(b))), 1e-30)
    return float(np.max(np.abs(a - b)) / den)


# ══════════════════════════════════════ 主流程 ══════════════════════════════════════
def sample(H, T, DK, DV, seed):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((H, T, DK)).astype(np.float64) / np.sqrt(DK)
    K = rng.standard_normal((H, T, DK)).astype(np.float64)
    K /= np.maximum(np.linalg.norm(K, axis=-1, keepdims=True), 1e-6)   # GDN 惯例：k 单位化
    V = rng.standard_normal((H, T, DV)).astype(np.float64)
    BETA = (1.0 / (1.0 + np.exp(-rng.standard_normal((H, T))))).astype(np.float64)
    G = (-np.abs(rng.standard_normal((H, T))) * 0.5 - 0.05).astype(np.float64)  # g<=0
    H0 = (rng.standard_normal((H, DV, DK)) * 0.1).astype(np.float64)
    w = rng.standard_normal((H, T, DV)).astype(np.float64)
    return Q, K, V, BETA, G, H0, w


def as_f32(*arrs):
    return [np.asarray(a, np.float32) for a in arrs]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--DK", type=int, default=16)
    ap.add_argument("--DV", type=int, default=16)
    ap.add_argument("--fd", type=int, default=1, help="有限差分 gradcheck（小形状）")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--npz", default="")
    args = ap.parse_args()

    H, T, DK, DV = args.H, args.T, args.DK, args.DV
    Q, K, V, BETA, G, H0, w = sample(H, T, DK, DV, args.seed)
    dO = w                                        # 线性泛函权重即 L 对 O 的梯度

    # ── [A] 前向：fp32 同序复刻 vs fp64 真值 ──
    O64, HIST64, Sf64 = fwd_exact(H, T, DK, DV, Q, K, V, BETA, G, H0)
    q32, k32, v32, b32, g32, h032 = as_f32(Q, K, V, BETA, G, H0)
    Oe, HISTe, Sfe = fwd_kernel_order(H, T, DK, DV, q32, k32, v32, b32, g32, h032,
                                      lambda x: np.exp(x), np.float32)
    Oc, HISTc, Sfc = fwd_kernel_order(H, T, DK, DV, q32, k32, v32, b32, g32, h032,
                                      tl910b_expf, np.float32)
    print(f"[A] FWD  fp32(exp)  vs fp64 : rel_O={rel(Oe, O64):.3e} "
          f"rel_HIST={rel(HISTe, HIST64):.3e} rel_Sfin={rel(Sfe, Sf64):.3e}")
    print(f"[A] FWD  fp32(expf) vs fp64 : rel_O={rel(Oc, O64):.3e} "
          f"rel_HIST={rel(HISTc, HIST64):.3e} rel_Sfin={rel(Sfc, Sf64):.3e}")
    print(f"[A] FWD  fp32(expf) vs fp32(exp) : rel_O={rel(Oc, Oe):.3e} "
          f"rel_Sfin={rel(Sfc, Sfe):.3e}   ← 纯 expf 软件件贡献")

    # ── [B] 反向公式：fp64 伴随 vs 有限差分（+ torch autograd 第三锚）──
    fdH, fdT, fdK, fdD = 2, 6, 4, 4
    Qf, Kf, Vf, Bf, Gf, H0f, wf = sample(fdH, fdT, fdK, fdD, args.seed + 1)
    _, HISTf, _ = fwd_exact(fdH, fdT, fdK, fdD, Qf, Kf, Vf, Bf, Gf, H0f)
    adj = bwd_adjoint(fdH, fdT, fdK, fdD, Qf, Kf, Vf, Bf, Gf, HISTf, wf, order="ub")
    names = ["Q", "K", "V", "BETA", "G", "H0"]
    fdg = gradcheck_fd((fdH, fdT, fdK, fdD, Qf, Kf, Vf, Bf, Gf, H0f, wf))
    worst = 0.0
    for n, arr in zip(names, adj):
        r = rel(arr, fdg[n]); worst = max(worst, r)
        print(f"[B] ADJ-FD {n:5s} rel={r:.3e}")
    tg = gradcheck_torch((fdH, fdT, fdK, fdD, Qf, Kf, Vf, Bf, Gf, H0f, wf))
    if tg is not None:
        for n, arr in zip(names, adj):
            print(f"[B] ADJ-TORCH {n:5s} rel={rel(arr, tg[n]):.3e}")
    else:
        print("[B] ADJ-TORCH skipped (no torch)")
    ok_b = worst < TOL_ADJ
    print(f"[B] BWD-ADJOINT {'OK' if ok_b else 'FAIL'} worst_rel_vs_FD={worst:.3e} (tol {TOL_ADJ:.0e})")

    # ── [C] 真实可期望面：fp32 + 软件 expf（ub/pure 两种发射次序）vs fp64 真值 ──
    adj_ub = bwd_adjoint(H, T, DK, DV, q32, k32, v32, b32, g32, HISTc, dO,
                         expf=tl910b_expf, dt=np.float32, order="ub")
    adj_pu = bwd_adjoint(H, T, DK, DV, q32, k32, v32, b32, g32, HISTc, dO,
                         expf=tl910b_expf, dt=np.float32, order="pure")
    adj_64 = bwd_adjoint(H, T, DK, DV, Q, K, V, BETA, G, HIST64, dO, order="ub")
    spread = max(rel(u, p) for u, p in zip(adj_ub, adj_pu))
    worst_c = 0.0
    for n, arr, ref64 in zip(names, adj_ub, adj_64):
        r = rel(arr, ref64)
        worst_c = max(worst_c, r)
        print(f"[C] fp32+expf {n:5s} rel_vs_fp64={r:.3e}")
    print(f"[C] fp32 发射次序 spread(ub vs pure) = {spread:.3e}")
    # expf 漂移沿递推链的放大：T=1 与 T=完整 对比
    a_err = rel(tl910b_expf(np.asarray(G, np.float32)), np.exp(G))
    print(f"[C] EXPF-DRIFT a_t 单点 max_rel={a_err:.3e}（compat §11 件本体）；"
          f"沿 T={T} 连乘后进 O 的 rel={rel(Oc, O64):.3e}")

    worst_all = max(worst_c, rel(Oc, O64))
    ok_a = rel(Oc, O64) < TOL_F32
    if ok_a and ok_b:
        print(f"F-GDN-GOLDEN rel={worst_all:.3e} "
              f"(fwd {rel(Oc, O64):.3e} / bwd_fp32 worst {worst_c:.3e} / "
              f"adjoint-vs-FD {worst:.3e}) TOL_F32={TOL_F32:.0e}")
    else:
        print(f"F-GDN-GOLDEN-FAIL fwd_ok={ok_a} adj_ok={ok_b}")

    if args.npz:
        np.savez_compressed(args.npz, Q=Q, K=K, V=V, BETA=BETA, G=G, H0=H0, dO=dO,
                            O=Oc, HIST=HISTc, Sfin=Sfc,
                            dQ=adj_ub[0], dK=adj_ub[1], dV=adj_ub[2],
                            dBeta=adj_ub[3], dG=adj_ub[4], dH0=adj_ub[5],
                            O64=O64, dQ64=adj_64[0], dK64=adj_64[1], dV64=adj_64[2],
                            dBeta64=adj_64[3], dG64=adj_64[4], dH064=adj_64[5])
        print(f"GOLDEN-EMITTED {args.npz} (fp32 同序件 + fp64 真值，供上卡对拍)")
    return 0 if (ok_a and ok_b) else 1


if __name__ == "__main__":
    sys.exit(main())
