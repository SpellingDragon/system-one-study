#!/usr/bin/env python3
"""attempts/F/f_gdn_delta_bwd_910b.py — GDN(Gated DeltaNet) **delta rule 反向（伴随递推）**的 910B tilelang DSL 件。

前向（同 f_gdn_delta_fwd_910b.py，逐 head，state 转置存 Sd[j,i] ≡ S_std[i,j]）：
    a_t = exp(g_t);  S'_t = a_t·S_{t−1};  p_t[j] = Σ_i S'_t[j,i]·k_t[i]
    u_t[j] = β_t(v_t[j] − p_t[j]);  S_t[j,i] = S'_t[j,i] + u_t[j]·k_t[i]
    o_t[j] = Σ_i S_t[j,i]·q_t[i]

反向（伴随态 Λ_t ≡ ∂L/∂S_t，t 从 T−1 递减；Λ^tot[j,i] = Λ_{t+1}[j,i] + dO_t[j]·q_t[i]）：
    (1) o_t 直贡献  : Λ^tot[j,i] += dO[j]·q[i] ;  dq[i] += Σ_j S_t[j,i]·dO[j]
    (2) S_t=S'+k⊗u  : du[j] = Σ_i Λ^tot[j,i]·k[i] ;  dk[i] += Σ_j (u[j]·Λ^tot[j,i] + S'_t[j,i]·dp[j])
                      dS'[j,i] = Λ^tot[j,i] + k[i]·dp[j]
    (3) u=β(v−p)    : dv[j] = β·du[j] ;  dβ = Σ_j du[j]·(v[j]−p[j]) ;  dp[j] = −β·du[j]
    (4) p=S'ᵀk      : dk[i] 的 Σ_j S'_t[j,i]·dp[j] 项（已并入 (2) 行）
    (5) S'=a·S_{t−1}: Λ_{t−1}[j,i] = a·dS'[j,i] ;  da = Σ_{j,i} S_{t−1}[j,i]·dS'[j,i] ;
                      dg_t = da·a_t
    入口 Λ_{T−1} = 末状态梯度（--dstate zero 置零 / gm 读 DSOUT）；出口 dH0 = Λ_{−1}。

**为什么反向必须换工作分解**（本波结构性结论，登记 G-F2）：
  前向在值维 j 上逐列独立 ⇒ 可按 (h,j) 切核；但反向 dq[i]、dk[i]、dβ、dg 都是
  **对 j 的归约**（dq[i]=Σ_j S_t[j,i]·dO[j] 等），跨列不可分。故本件按 **head 并行**
  （unit = head），整块伴随态 (DV,DK) 常驻单核（ub 变体放 UB / pure 变体放 GM）。

910B 建模取向：
  * 只用 AIV 标量面：`T.Kernel(cores)` + `T.Vector()` + 纯 `T.serial`；不用
    T.Parallel/SimtVF/T.simd/T.copy（零 MTE ⇒ 不吃 G-C4/G-C7 未证真的单位/stride 面）。
  * decay 求导链只吃 **exp**（a=exp(g)，dg=da·a）⇒ 不需要 logf（见 gap_F.md G-F0）。
  * 变体（两者的**读写次序**不同，这是本件的核心纪律）：
      ub   — Λ 与 k/q/v/dO/u/dp 暂存全在 UB（7 个都被访问的 shared，避开 G-C8 空句柄）。
             趟1（j 外）算 u[j]/dp[j]（读**未更新**的 Λ）→ 落 u_scr/dp_scr；
             趟2（i 外 j 内）逐元素读 Λ^tot 后**立刻**回写 Λ_{t−1}（每个元素只读一次即写，
             无二次读 ⇒ 无自噬）；dq/dk 在 i 外层用 var 累加 ⇒ 单存，无 GM RMW。
      pure — 零 alloc_shared。次序换成 **j 外层单趟融合**：第 j 行的 du[j] 只读第 j 行 Λ
             （此刻该行尚未被任何更新污染，前 j−1 行的更新与它无关）⇒ 可安全就地 RMW；
             dq/dk 跨 j 累加走 **GM 读改写**，首列 j==0 用 T.if_then_else 走"首存"，
             其余走"累加" ⇒ **不要求宿主预清零**。
  * HIST 是前向逐 token 落下的 S_{t−1}（本件只读）。若前向 `--hist 0` 则本件不可用
     （da 项必须要 S_{t−1}）⇒ 反向对 HIST 的依赖是前向落状态的必要性凭据（G-F4）。
  * **不能** `from __future__ import annotations`（tilelang eager builder get_type_hints 坑）。

判据：容器内 compile PASS（本机无 NPU；数值与 partial 口径见 f_gdn_golden.py / RESULT.md）。
用法：python3 f_gdn_delta_bwd_910b.py [--variant ub|pure] [--dstate zero|gm] [--H 8] [--T 32]
                                      [--DK 16] [--DV 16] [--cores 8] [--dump f]
"""
import argparse
import os
import re
import sys
import traceback

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402


def gdn_delta_rule_bwd(H=8, T_LEN=32, DK=16, DV=16, cores=8, dtype="float32",
                       variant="ub", dstate="zero"):
    """构造 GDN delta rule 反向 prim_func（q/k/v/beta/g + 初状态全梯度）。

    variant:
      ub   — Λ/暂存在 UB，GM 纯只读 + 旁路写（首推，无 GM 就地 RMW）
      pure — 零 UB，Λ 就地存 GM 工作缓冲 DSOUT，dq/dk 走 GM 读改写累加（免宿主预清零）
    dstate:
      zero — 末状态梯度视为 0（截断 BPTT 常规；件内把 Λ 初值写 0）
      gm   — 末状态梯度取自 DSOUT（ub: 载入 UB；pure: DSOUT 即工作缓冲，
             **要求宿主把 DSOUT 预置为该梯度**）
    """
    if H % cores:
        raise ValueError(f"H={H} 必须能被 cores={cores} 整除（反向按 head 并行）")
    UPC = H // cores

    @T.prim_func
    def gdn_delta_bwd(
        Q: T.Tensor((H, T_LEN, DK), dtype),
        K: T.Tensor((H, T_LEN, DK), dtype),
        V: T.Tensor((H, T_LEN, DV), dtype),
        DO: T.Tensor((H, T_LEN, DV), dtype),
        BETA: T.Tensor((H, T_LEN), dtype),
        G: T.Tensor((H, T_LEN), dtype),
        HIST: T.Tensor((H, T_LEN, DV, DK), dtype),
        DSOUT: T.Tensor((H, DV, DK), dtype),
        DQ: T.Tensor((H, T_LEN, DK), dtype),
        DKG: T.Tensor((H, T_LEN, DK), dtype),
        DVV: T.Tensor((H, T_LEN, DV), dtype),
        DBETA: T.Tensor((H, T_LEN), dtype),
        DG: T.Tensor((H, T_LEN), dtype),
        DH0: T.Tensor((H, DV, DK), dtype),
    ):
        with T.Kernel(cores) as core_id:
            a = T.alloc_var(dtype, T.float32(0.0))
            bt = T.alloc_var(dtype, T.float32(0.0))
            hp = T.alloc_var(dtype, T.float32(0.0))
            sp = T.alloc_var(dtype, T.float32(0.0))
            lam = T.alloc_var(dtype, T.float32(0.0))
            ki = T.alloc_var(dtype, T.float32(0.0))
            qi = T.alloc_var(dtype, T.float32(0.0))
            p = T.alloc_var(dtype, T.float32(0.0))
            uj = T.alloc_var(dtype, T.float32(0.0))
            du = T.alloc_var(dtype, T.float32(0.0))
            dp = T.alloc_var(dtype, T.float32(0.0))
            dst = T.alloc_var(dtype, T.float32(0.0))
            vj = T.alloc_var(dtype, T.float32(0.0))
            doj = T.alloc_var(dtype, T.float32(0.0))
            dq_acc = T.alloc_var(dtype, T.float32(0.0))
            dk_acc = T.alloc_var(dtype, T.float32(0.0))
            db_acc = T.alloc_var(dtype, T.float32(0.0))
            da_acc = T.alloc_var(dtype, T.float32(0.0))

            if variant == "ub":
                lam_ub = T.alloc_shared((DV, DK), dtype)   # 伴随态 Λ（本核独占整块）
                u_scr = T.alloc_shared((DV,), dtype)      # u_t[j]
                dp_scr = T.alloc_shared((DV,), dtype)     # dp_t[j]
                k_ub = T.alloc_shared((DK,), dtype)
                q_ub = T.alloc_shared((DK,), dtype)
                v_scr = T.alloc_shared((DV,), dtype)
                do_scr = T.alloc_shared((DV,), dtype)

            with T.Vector():
                for rr in T.serial(UPC):
                    h = core_id * UPC + rr

                    # ── Λ 初值：末状态梯度（zero=写 0 / gm=读 DSOUT）──
                    for j in T.serial(DV):
                        for i in T.serial(DK):
                            if variant == "ub":
                                if dstate == "gm":
                                    lam_ub[j, i] = DSOUT[h, j, i]
                                else:
                                    lam_ub[j, i] = T.float32(0.0)
                            else:
                                if dstate == "zero":
                                    DSOUT[h, j, i] = T.float32(0.0)

                    for tt in T.serial(T_LEN):
                        t = (T_LEN - 1) - tt              # 逆序遍历 token
                        a = T.exp(G[h, t])                # → codegen expf（compat §11）
                        bt = BETA[h, t]
                        db_acc = T.float32(0.0)
                        da_acc = T.float32(0.0)

                        if variant == "ub":
                            for i in T.serial(DK):
                                k_ub[i] = K[h, t, i]
                                q_ub[i] = Q[h, t, i]
                            for j in T.serial(DV):
                                v_scr[j] = V[h, t, j]
                                do_scr[j] = DO[h, t, j]

                            # 趟 1（j 外）：p[j] / u[j] / du[j] / dp[j] / dv[j] / dβ 累加
                            for j in T.serial(DV):
                                p = T.float32(0.0)
                                du = T.float32(0.0)
                                for i in T.serial(DK):
                                    p = p + (a * HIST[h, t, j, i]) * k_ub[i]
                                    du = du + (lam_ub[j, i] + do_scr[j] * q_ub[i]) * k_ub[i]
                                uj = bt * (v_scr[j] - p)
                                dp = T.float32(0.0) - bt * du
                                u_scr[j] = uj
                                dp_scr[j] = dp
                                DVV[h, t, j] = bt * du
                                db_acc = db_acc + du * (v_scr[j] - p)

                            # 趟 2（i 外）：Λ 递推 + dq[i] / dk[i] / da
                            for i in T.serial(DK):
                                dq_acc = T.float32(0.0)
                                dk_acc = T.float32(0.0)
                                for j in T.serial(DV):
                                    # Λ^tot = Λ_{t+1} + dO⊗q（o_t 对 S_t 的直贡献，必须补）
                                    lam = lam_ub[j, i] + do_scr[j] * q_ub[i]
                                    hp = HIST[h, t, j, i]
                                    sp = a * hp                       # S'_t[j,i]
                                    dq_acc = dq_acc + (sp + u_scr[j] * k_ub[i]) * do_scr[j]
                                    dk_acc = dk_acc + u_scr[j] * lam + sp * dp_scr[j]
                                    dst = lam + k_ub[i] * dp_scr[j]   # dS'_t,total[j,i]
                                    da_acc = da_acc + hp * dst
                                    lam_ub[j, i] = a * dst            # Λ_{t−1}[j,i]
                                DQ[h, t, i] = dq_acc
                                DKG[h, t, i] = dk_acc

                        else:
                            # pure：**j 外层单趟融合**。第 j 行的 du/p 只读第 j 行 Λ
                            # （该行此刻必未被更新：本行的写发生在读完之后）⇒ 就地 RMW 安全。
                            for j in T.serial(DV):
                                vj = V[h, t, j]
                                doj = DO[h, t, j]
                                p = T.float32(0.0)
                                du = T.float32(0.0)
                                for i in T.serial(DK):
                                    ki = K[h, t, i]
                                    qi = Q[h, t, i]
                                    sp = (a * HIST[h, t, j, i])
                                    p = p + sp * ki
                                    du = du + (DSOUT[h, j, i] + doj * qi) * ki
                                uj = bt * (vj - p)
                                dp = T.float32(0.0) - bt * du
                                DVV[h, t, j] = bt * du
                                db_acc = db_acc + du * (vj - p)
                                for i in T.serial(DK):
                                    ki = K[h, t, i]
                                    # Λ^tot = Λ_{t+1} + dO⊗q（本行 DSOUT 仍是未更新的 Λ_{t+1}）
                                    lam = DSOUT[h, j, i] + doj * Q[h, t, i]
                                    hp = HIST[h, t, j, i]
                                    sp = a * hp
                                    dq_acc = (sp + uj * ki) * doj
                                    dk_acc = uj * lam + sp * dp
                                    dst = lam + ki * dp
                                    DQ[h, t, i] = T.if_then_else(
                                        j == 0, dq_acc, DQ[h, t, i] + dq_acc)
                                    DKG[h, t, i] = T.if_then_else(
                                        j == 0, dk_acc, DKG[h, t, i] + dk_acc)
                                    da_acc = da_acc + hp * dst
                                    DSOUT[h, j, i] = a * dst          # Λ_{t−1} 就地回写

                        DBETA[h, t] = db_acc
                        DG[h, t] = da_acc * a

                    # ── 出口：dH0 = Λ_{−1} ──
                    for j in T.serial(DV):
                        for i in T.serial(DK):
                            if variant == "ub":
                                DH0[h, j, i] = lam_ub[j, i]
                            else:
                                DH0[h, j, i] = DSOUT[h, j, i]

    return gdn_delta_bwd


def scan_source(src):
    """codegen 产物体检：哪些面被真实触到（结构结论/缺口清单的凭据来源）。"""
    asc = sorted(set(re.findall(r"\basc_[A-Za-z0-9_]+", src)))
    tlf = sorted(set(re.findall(r"\btl::[A-Za-z0-9_]+", src)))
    mathf = sorted(set(re.findall(
        r"\b(expf|exp2f|logf|sqrtf|rsqrtf|fabsf?|tanhf|__cce_[a-z0-9_]+)\s*\(", src)))
    m = re.search(r'extern "C" (__global__[^(]*)', src)
    # G-C8 的病灶是**具名** shared 落成 (__ubuf__ T*)0；buf_dyn_shmem 是正常动态池基址，排除
    named_null = re.findall(
        r"__ubuf__ [A-Za-z_0-9]+ \*(?!buf_dyn_shmem)(\w+) = \(__ubuf__ [A-Za-z_0-9]+ \*\)0", src)
    feats = {
        "prefix": m.group(1).strip() if m else "?",
        "has_mix": "__mix__" in src,
        "has_ASC_IS_": "ASC_IS_" in src,
        "has_vf_call": "asc_vf_call" in src,
        "has_threadIdx": "threadIdx" in src,
        "has_asc_sync": bool(re.search(r"\basc_sync", src)),
        "has_gm_bypass": "gm_bypass_dcache" in src,
        "has_ubuf": "__ubuf__" in src,
        "has_mte_copy": bool(re.search(r"\basc_copy_[a-z0-9]+_align", src)),
        "named_null_ubuf": named_null,
        "gm_rmw": bool(re.search(r"read_gm_bypass_dcache", src)),
        "lines": src.count("\n"),
    }
    return asc, tlf, mathf, feats


def run(args):
    func = gdn_delta_rule_bwd(H=args.H, T_LEN=args.T, DK=args.DK, DV=args.DV,
                              cores=args.cores, dtype=args.dtype,
                              variant=args.variant, dstate=args.dstate)
    kernel = tilelang.compile(func, target="ascend", out_idx=-1)
    src = kernel.get_kernel_source()
    asc, tlf, mathf, feats = scan_source(src)
    print("F-GDN-DELTA-BWD-COMPILE-PASS")
    print(f"  variant={args.variant} dstate={args.dstate} H={args.H} T={args.T} "
          f"DK={args.DK} DV={args.DV} cores={args.cores} dtype={args.dtype}")
    for k, v in feats.items():
        print(f"  {k}: {v}")
    print("  emitted asc_* :", " ".join(asc) or "(none)")
    print("  emitted tl::* :", " ".join(tlf) or "(none)")
    print("  math/intrin   :", " ".join(mathf) or "(none)")
    if args.dump:
        with open(args.dump, "w") as f:
            f.write(src)
        print(f"  source dumped -> {args.dump}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default=os.environ.get("TL_F_VARIANT", "ub"))
    ap.add_argument("--dstate", default=os.environ.get("TL_F_DSTATE", "zero"))
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--DK", type=int, default=16)
    ap.add_argument("--DV", type=int, default=16)
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--dump", default=os.environ.get("TL_F_DUMP", ""))
    args = ap.parse_args()
    try:
        run(args)
    except Exception as e:  # noqa: BLE001
        print("F-GDN-DELTA-BWD-COMPILE-FAIL")
        s = str(e)
        print("---- first error lines ----")
        err = [ln for ln in s.splitlines()
               if re.search(r"error|undefined|undeclared|no member|FATAL|Check failed|NameError", ln)]
        for ln in (err[:8] or s.splitlines()[:8]):
            print("  " + ln.strip()[:220])
        print("---- tail 800 ----")
        print(s[-800:])
        if os.environ.get("TL_F_TRACE"):
            traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
