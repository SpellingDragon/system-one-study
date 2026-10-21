#!/usr/bin/env python3
"""attempts/C/c_gdn_conv_910b.py — GDN(Gated DeltaNet) 短卷积前向的 910B tilelang DSL 件。

语义（Qwen3.5 GDN / Mamba 系 short conv，channels-first depthwise causal）：
    Y[c, t] = silu( bias[c] + SUM_{k=0..K-1} W[c,k] * X[c, t-(K-1)+k] )   (t-(K-1)+k < 0 视作 0)
    silu(z) = z * sigmoid(z) = z / (1 + exp(-z))
K=4（Qwen3-Next/GDN 约定 conv_kernel=4）。

910B 建模取向（本波）：
  * 只用 AIV 标量面：`T.Kernel(cores)` + `T.Vector()` 区域 + 纯 `T.serial` 三重循环
    （通道块 → 时间步 → 窗内 4 抽头）。不用 T.Parallel / SimtVF（其 codegen 走
    c_api/simt_api 面，910B 无该头链），也不用 T.simd.*（128b vec 属性与 910B clang15 冲突）。
  * 累加走 `T.alloc_var`（local.var），不做 UB 整块 vector-value store（会撞
    ScalarDcacheBypass 的 "vector-valued buffer store not implemented"）。
  * 因果左边界用 `T.max(idx,0)` 夹地址 + `T.if_then_else` 置零，不出负下标。
  * **死缓冲纪律**：alloc_shared 若只写不读，codegen 会把它落成
    `__ubuf__ T *buf = (__ubuf__ T *)0;` 的**命名空句柄**（真实写 UB 偏移 0，且不进
    动态共享内存布局；多个死缓冲会互相别名）。故 shared 声明按变体分支，scalar
    变体一个 alloc_shared 都不留。取证见 probe_ubhandle.py（G-C8）。
  * SiLU：--silu exp 用 T.exp（codegen→expf，实付件见 compat_patch_C.h 的软件 expf）；
    --silu rational 退化为 sigmoid(u)≈0.5*(u/(1+|u|)+1) 的有理近似（只吃四则+fabs，
    供 exp 面缺失时保编译，数值非对拍真值；golden 仍以 exp 版为准）。
  * 注意：**不能** `from __future__ import annotations` —— tilelang eager builder
    用 get_type_hints 求值 `T.Tensor((C, L), dtype)` 这类形状注解，PEP563 延迟求值
    会把闭包变量(C/L/K/dtype)当模块全局名解析 → NameError: name 'C' is not defined。

判据：compile PASS（本机无 NPU；上卡对拍件见 c_gdn_golden.py）。
用法：python3 c_gdn_conv_910b.py [--variant scalar|ubstage] [--silu exp|rational]
                                 [--C 2048] [--L 512] [--K 4] [--cores 64] [--dump f]
"""
import argparse
import os
import re
import sys
import traceback

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # ascend 方言（Kernel 1-D 核栅格 / Vector 区域）  # noqa: E402


def _silu(z, mode):
    """SiLU 两种发射面。"""
    if mode == "exp":
        # z * sigmoid(z) = z / (1 + exp(-z))；T.exp(float32) → codegen `expf`
        return z / (T.float32(1.0) + T.exp(T.float32(0.0) - z))
    # 兜底（保编译用）：sigmoid(u) ≈ 0.5*(u/(1+|u|)+1)，只用四则 + fabs
    u = T.float32(0.0) - z
    sig = T.float32(0.5) * (u / (T.float32(1.0) + T.abs(u)) + T.float32(1.0))
    return z * sig


def gdn_short_conv_fwd(C=2048, L=512, K=4, cores=64, dtype="float32",
                       variant="scalar", silu="exp"):
    """构造 GDN 短卷积前向 prim_func。

    variant:
      scalar  — 直读 GM 标量面（最小可编面，零 UB 参与）
      ubstage — 每通道行先 T.copy 进 UB（滑窗读 UB），算完 T.copy 回 GM
                （多走 MTE2/MTE3 搬运面，验证 asc_copy_*_align 通路）
    """
    if C % cores:
        raise ValueError(f"C={C} 必须能被 cores={cores} 整除")
    CB = C // cores
    assert K == 4, "GDN 短卷积约定 kernel=4"
    pad = K - 1  # 因果左补零宽度

    @T.prim_func
    def gdn_conv_fwd(
        X: T.Tensor((C, L), dtype),
        W: T.Tensor((C, K), dtype),
        BIAS: T.Tensor((C,), dtype),
        Y: T.Tensor((C, L), dtype),
    ):
        with T.Kernel(cores) as core_id:
            acc = T.alloc_var(dtype, T.float32(0.0))
            z = T.alloc_var(dtype, T.float32(0.0))
            xv = T.alloc_var(dtype, T.float32(0.0))
            wt = T.alloc_var(dtype, T.float32(0.0))
            bi = T.alloc_var(dtype, T.float32(0.0))

            if variant == "ubstage":
                # 窗缓存：4 抽头权重 / 整行输入 / 输出行常驻 UB（仅 ubstage 需要；
                # 死缓冲会落成 (__ubuf__ T*)0 空句柄，见模块 docstring 与 G-C8）
                w_ub = T.alloc_shared((K,), dtype)
                row_ub = T.alloc_shared((L + pad,), dtype)  # 左端 pad 个 0 = 因果补零
                out_ub = T.alloc_shared((L,), dtype)

            with T.Vector():
                for cb in T.serial(CB):
                    ch = core_id * CB + cb
                    bi = BIAS[ch]

                    if variant == "ubstage":
                        for k in T.serial(K):
                            w_ub[k] = W[ch, k]
                        # 因果补零：row_ub[0:pad]=0，row_ub[pad:pad+L]=X[ch,:]
                        # 注意必须写**区间**（X[ch, 0:L]）：写成 X[ch, 0] 时
                        # T.copy 只搬 1 个元素（实测发射退化成一个标量 store，
                        # asc_copy_gm2ub_align 根本不会出现）。
                        for k in T.serial(pad):
                            row_ub[k] = T.float32(0.0)
                        T.copy(X[ch, 0:L], row_ub[pad:pad + L])
                        for t in T.serial(L):
                            acc = T.float32(0.0)
                            for k in T.serial(K):
                                acc = acc + w_ub[k] * row_ub[t + k]
                            z = acc + bi
                            out_ub[t] = _silu(z, silu)
                        T.copy(out_ub[0:L], Y[ch, 0:L])
                    else:
                        for t in T.serial(L):
                            acc = T.float32(0.0)
                            for k in T.serial(K):
                                idx = t - pad + k
                                # 左边界：idx<0 → 补零（地址先夹到 0，不出负下标）
                                xv = T.if_then_else(idx >= 0, X[ch, T.max(idx, 0)],
                                                    T.float32(0.0))
                                wt = W[ch, k]
                                acc = acc + wt * xv
                            z = acc + bi
                            Y[ch, t] = _silu(z, silu)

    return gdn_conv_fwd


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
        "lines": src.count("\n"),
    }
    return asc, tlf, mathf, feats


def run(args):
    func = gdn_short_conv_fwd(C=args.C, L=args.L, K=args.K, cores=args.cores,
                              dtype=args.dtype, variant=args.variant, silu=args.silu)
    kernel = tilelang.compile(func, target="ascend", out_idx=-1)
    src = kernel.get_kernel_source()
    asc, tlf, mathf, feats = scan_source(src)
    print("C-GDN-CONV-COMPILE-PASS")
    print(f"  variant={args.variant} silu={args.silu} C={args.C} L={args.L} "
          f"K={args.K} cores={args.cores} dtype={args.dtype}")
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default=os.environ.get("TL_C_VARIANT", "scalar"))
    ap.add_argument("--silu", default=os.environ.get("TL_C_SILU", "exp"))
    ap.add_argument("--C", type=int, default=2048)
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--cores", type=int, default=64)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--dump", default=os.environ.get("TL_C_DUMP", ""))
    args = ap.parse_args()
    try:
        run(args)
    except Exception as e:  # noqa: BLE001
        print("C-GDN-CONV-COMPILE-FAIL")
        s = str(e)
        print("---- first error lines ----")
        err = [ln for ln in s.splitlines()
               if re.search(r"error|undefined|undeclared|no member|FATAL|Check failed|NameError", ln)]
        for ln in (err[:8] or s.splitlines()[:8]):
            print("  " + ln.strip()[:220])
        print("---- tail 800 ----")
        print(s[-800:])
        if os.environ.get("TL_C_TRACE"):
            traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
