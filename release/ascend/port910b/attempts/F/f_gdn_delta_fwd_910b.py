#!/usr/bin/env python3
"""attempts/F/f_gdn_delta_fwd_910b.py — GDN(Gated DeltaNet) **delta rule 前向递推**的 910B tilelang DSL 件。

语义（标准 Gated DeltaNet recurrent form，逐 head，state S ∈ [dk, dv]；tasks P1-3）：
    a_t   = exp(g_t)                        # g_t <= 0，log 域衰减门
    S     = S * a_t                         # 衰减
    pred  = Sᵀ k_t    pred[j]=Σ_i S[i,j]k[i]
    u_t   = beta_t * (v_t - pred)           # delta 修正（值域向量）
    S     = S + outer(k_t, u_t)             # 外积更新
    o_t   = Sᵀ q_t                          # 输出 ∈ [dv]
（pred 取"已衰减未更新"的 state；o 取"已更新"的 state —— 与 FLA / Qwen3-Next GDN 一致。）

**布局约定（本波定型，重要）**：state 一律以**转置**形态存储/遍历
    Sd[h, j, i] ≡ S_std[i, j]，j ∈ DV（值维）、i ∈ DK（键维）
理由：delta rule 的递推在**值维 j 上逐列独立**（pred[j]/u[j]/o[j] 只吃第 j 列），
按 j 切分工作单元可零跨核归约；转置后第 j 列 = DK 个**连续**元素，标量面顺址读写。

910B 建模取向（沿用 C 波 G-C1/G-C8 与 E 波 pure/stage 双变体纪律）：
  * 只用 AIV 标量面：`T.Kernel(cores)` + `T.Vector()` + 纯 `T.serial` 嵌套。
    **不用** T.Parallel / SimtVF / T.simd.*（910B 无该头链 / clang15 冲突）。
  * 工作分解：unit = (head h, 值维列 j)，共 H*DV 个；`u // DV`、`u % DV` 反解。
  * exp：`T.exp(g)` → codegen `expf` → compat §11 软件 expf（trunk 已并，真源实测在
    `port910b_compat.h:724`）。decay 只吃 **exp**，**不需要 logf**（g 以 log 域直接入参）
    → tasks 里"decay 的 log 缺件"在本语义下不成立，见 gap_F.md G-F0。
  * 变体：
      ub    — 状态列常驻 UB（st_ub/k_ub/q_ub 三个**都被访问**的 shared，避开 G-C8
              「恰好 1 个被访问 shared → (__ubuf__ T*)0 空句柄」）；GM 只在首尾各碰一次，
              **全程零 T.copy**（不走 MTE2/MTE3 → 不吃 G-C4/G-C7 的单位/stride 未证真风险）。
      pure  — 零 alloc_shared，状态**就地**存 GM 工作缓冲 SOUT（宿主预置为 H0）。
              代价：同一地址 read-modify-write 跨 GM，**上卡须证真 bypass-dcache 读后写**（G-F3）。
  * 逐 token 融合次序：`S=S*a; p+=S*k` 在同一趟里完成（p 用的就是已衰减的 S），
    `S+=k*u; o+=S*q` 同理（o 用的是已更新的 S）。golden 复刻体**必须同序**。
  * **不能** `from __future__ import annotations`（tilelang eager builder 的 get_type_hints
    会把闭包变量 H/T/DK/DV/dtype 当模块全局名解析 → NameError，C 波实测）。

判据：容器内 compile PASS（本机无 NPU；数值见 f_gdn_golden.py）。
用法：python3 f_gdn_delta_fwd_910b.py [--variant ub|pure] [--H 8] [--T 32] [--DK 16]
                                      [--DV 16] [--cores 8] [--hist 1] [--dump f]
"""
import argparse
import os
import re
import sys
import traceback

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # ascend 方言（Kernel 1-D 核栅格 / Vector 区域）  # noqa: E402


def gdn_delta_rule_fwd(H=8, T_LEN=32, DK=16, DV=16, cores=8, dtype="float32",
                       variant="ub", emit_hist=True):
    """构造 GDN delta rule 前向 prim_func。

    variant:
      ub   — 状态列常驻 UB，纯标量访问，零 T.copy（首推）
      pure — 零 UB，状态就地读写 GM 工作缓冲 SOUT（约定宿主预置 SOUT=H0）
    """
    units = H * DV
    if units % cores:
        raise ValueError(f"units=H*DV={units} 必须能被 cores={cores} 整除")
    UPC = units // cores

    @T.prim_func
    def gdn_delta_fwd(
        Q: T.Tensor((H, T_LEN, DK), dtype),
        K: T.Tensor((H, T_LEN, DK), dtype),
        V: T.Tensor((H, T_LEN, DV), dtype),
        BETA: T.Tensor((H, T_LEN), dtype),
        G: T.Tensor((H, T_LEN), dtype),
        H0: T.Tensor((H, DV, DK), dtype),
        HIST: T.Tensor((H, T_LEN, DV, DK), dtype),
        SOUT: T.Tensor((H, DV, DK), dtype),
        O: T.Tensor((H, T_LEN, DV), dtype),
    ):
        with T.Kernel(cores) as core_id:
            a = T.alloc_var(dtype, T.float32(0.0))
            bt = T.alloc_var(dtype, T.float32(0.0))
            si = T.alloc_var(dtype, T.float32(0.0))
            ki = T.alloc_var(dtype, T.float32(0.0))
            qi = T.alloc_var(dtype, T.float32(0.0))
            vj = T.alloc_var(dtype, T.float32(0.0))
            pred = T.alloc_var(dtype, T.float32(0.0))
            uu = T.alloc_var(dtype, T.float32(0.0))
            o = T.alloc_var(dtype, T.float32(0.0))

            if variant == "ub":
                # 三个都被访问的 shared → 不触发 G-C8 单死缓冲空句柄
                st_ub = T.alloc_shared((DK,), dtype)
                k_ub = T.alloc_shared((DK,), dtype)
                q_ub = T.alloc_shared((DK,), dtype)

            with T.Vector():
                for rr in T.serial(UPC):
                    u = core_id * UPC + rr
                    h = u // DV
                    j = u % DV

                    if variant == "ub":
                        # 入口：初值 H0[h,j,:] → UB（标量循环，不用 T.copy ⇒ 零 MTE2）
                        for i in T.serial(DK):
                            st_ub[i] = H0[h, j, i]

                    for t in T.serial(T_LEN):
                        a = T.exp(G[h, t])           # → codegen expf（compat §11 软件件）
                        bt = BETA[h, t]
                        vj = V[h, t, j]

                        if variant == "ub":
                            for i in T.serial(DK):
                                k_ub[i] = K[h, t, i]
                                q_ub[i] = Q[h, t, i]

                        if emit_hist:
                            # HIST[h,t,:,:] = 进 t 前的状态 = S_{t-1}（bwd 求 da_t 用）
                            for i in T.serial(DK):
                                if variant == "ub":
                                    HIST[h, t, j, i] = st_ub[i]
                                else:
                                    HIST[h, t, j, i] = SOUT[h, j, i]

                        # 趟 1：衰减 + 预测（同一趟，pred 吃已衰减的 S）
                        pred = T.float32(0.0)
                        for i in T.serial(DK):
                            if variant == "ub":
                                st_ub[i] = st_ub[i] * a
                                pred = pred + st_ub[i] * k_ub[i]
                            else:
                                si = SOUT[h, j, i] * a
                                SOUT[h, j, i] = si
                                pred = pred + si * K[h, t, i]

                        uu = bt * (vj - pred)

                        # 趟 2：外积更新 + 输出（o 吃已更新的 S）
                        o = T.float32(0.0)
                        for i in T.serial(DK):
                            if variant == "ub":
                                st_ub[i] = st_ub[i] + k_ub[i] * uu
                                o = o + st_ub[i] * q_ub[i]
                            else:
                                si = SOUT[h, j, i] + K[h, t, i] * uu
                                SOUT[h, j, i] = si
                                o = o + si * Q[h, t, i]

                        O[h, t, j] = o

                    if variant == "ub":
                        # 出口：末状态 UB → GM（标量循环，不用 T.copy ⇒ 零 MTE3）
                        for i in T.serial(DK):
                            SOUT[h, j, i] = st_ub[i]

    return gdn_delta_fwd


def scan_source(src):
    """codegen 产物体检：哪些面被真实触到（结构结论/缺口清单的凭据来源）。"""
    asc = sorted(set(re.findall(r"\basc_[A-Za-z0-9_]+", src)))
    tlf = sorted(set(re.findall(r"\btl::[A-Za-z0-9_]+", src)))
    mathf = sorted(set(re.findall(
        r"\b(expf|exp2f|logf|sqrtf|rsqrtf|fabsf?|tanhf|__cce_[a-z0-9_]+)\s*\(", src)))
    m = re.search(r'extern "C" (__global__[^(]*)', src)
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
        # G-C8 病灶是**具名** shared 落成 (__ubuf__ T*)0；buf_dyn_shmem 是正常动态池基址，排除
        "named_null_ubuf": re.findall(
            r"__ubuf__ [A-Za-z_0-9]+ \*(?!buf_dyn_shmem)(\w+) = \(__ubuf__ [A-Za-z_0-9]+ \*\)0", src),
        "gm_rmw": bool(re.search(r"read_gm_bypass_dcache", src)),
        "lines": src.count("\n"),
    }
    return asc, tlf, mathf, feats


def run(args):
    func = gdn_delta_rule_fwd(H=args.H, T_LEN=args.T, DK=args.DK, DV=args.DV,
                              cores=args.cores, dtype=args.dtype,
                              variant=args.variant, emit_hist=bool(args.hist))
    kernel = tilelang.compile(func, target="ascend", out_idx=-1)
    src = kernel.get_kernel_source()
    asc, tlf, mathf, feats = scan_source(src)
    print("F-GDN-DELTA-FWD-COMPILE-PASS")
    print(f"  variant={args.variant} H={args.H} T={args.T} DK={args.DK} DV={args.DV} "
          f"cores={args.cores} dtype={args.dtype} hist={args.hist}")
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
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--DK", type=int, default=16)
    ap.add_argument("--DV", type=int, default=16)
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--hist", type=int, default=1)
    ap.add_argument("--dump", default=os.environ.get("TL_F_DUMP", ""))
    args = ap.parse_args()
    try:
        run(args)
    except Exception as e:  # noqa: BLE001
        print("F-GDN-DELTA-FWD-COMPILE-FAIL")
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
