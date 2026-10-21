#!/usr/bin/env python3
"""attempts/E/e_attn_sw_910b.py — 因果滑窗注意力 softmax 侧件（attn_sw）910B tilelang DSL。

语义（与 system-one track-A 生产件 attn_sw_asc.py 同口径）：
    Q/K/V (heads, seq, dim) fp32；查询 i 只看得见键 kk ∈ [max(0,i-window+1), i]
    （可见性 = `i0 >= kk` & `i0 - kk < window`，两个方向都拦）；
    score = scale * (Q_i · K_kk)，行内 softmax（减行最大→exp→归一）后加权 V 写 O。
    不可见键权重**严格为 0**：scalar 形态按行精确圈定可见窗，窗外套哨兵
    NEG=-1e30，`expf(哨兵-真实最大)→0`（compat §11 软件 expf 下溢钳位），
    无需生产件的"重算那一步再判可见性"补丁路径（本波逐行窗天然含真实最大值格）。

910B 建模取向（沿用 C 波 G-C1/G-C8 纪律）：
  * `T.Kernel(cores)` + `T.Vector()` + 纯 `T.serial`；每核串行领 heads*seq/cores 行。
  * softmax 侧数学件只吃：四则 + `T.exp`→`expf`（compat v3 §11 软件件，C 波已入
    真源并数值定标 max_rel_err=1.103e-06）+ `T.max`；**不碰 log/sqrt/rsqrt/三角**。
  * 变体：
      pure  — 零 alloc_shared 纯标量（G-C8 最稳形态）；代价=分数重算（编译波不关心
              性能，只锁可编面与语义面）。
      stage — 每行分数/权重量各一个 UB 数组（sc_ub/ps_ub 共 2 个且都被访问，
              避开 G-C8 单死缓冲空句柄形态；分数只算一遍）。
    两变体数值口径一致（stage 是生产 online-softmax 的"整行两趟"简化版：逐行形态
    下行最大一次即得，无需修正因子）。
  * scale 作为编译期 python float 折进一次乘法（与生产同口径）。
  * 判据件本机无 NPU，只看 compile PASS；数值真值与上卡判据见 e_golden.py。

用法：python3 e_attn_sw_910b.py [--variant pure|stage] [--heads 4] [--seq 64]
                                [--dim 32] [--window 8] [--cores 64] [--dump f]
"""
import argparse
import os
import re
import sys
import traceback

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402

NEG = -1.0e30   # 不可见哨兵（与生产 attn_sw_asc.NEG 同值）


def attn_sw_fwd(heads=4, seq=64, dim=32, window=8, cores=64, dtype="float32",
                variant="stage"):
    """构造因果滑窗注意力（softmax 侧）前向 prim_func。"""
    if window < 1:
        raise ValueError(f"window 需为正整数，实得 {window}")
    if window > seq:
        raise ValueError(f"window={window} 不得超过 seq={seq}（本波取等宽窗上界）")
    if heads * seq % cores:
        raise ValueError(f"heads*seq={heads * seq} 必须能被 cores={cores} 整除")
    RPC = heads * seq // cores       # 每核行数
    WIN = window                     # 窗上界（编译期常量数组宽）
    scale = dim ** -0.5              # 与生产默认口径一致：1/sqrt(dim)

    @T.prim_func
    def attn_sw_kernel(
        Q: T.Tensor((heads, seq, dim), dtype),
        K: T.Tensor((heads, seq, dim), dtype),
        V: T.Tensor((heads, seq, dim), dtype),
        O: T.Tensor((heads, seq, dim), dtype),
    ):
        with T.Kernel(cores) as core_id:
            sc = T.alloc_var(dtype, T.float32(0.0))
            m = T.alloc_var(dtype, T.float32(NEG))
            l = T.alloc_var(dtype, T.float32(0.0))
            w = T.alloc_var(dtype, T.float32(0.0))
            acc = T.alloc_var(dtype, T.float32(0.0))
            acc2 = T.alloc_var(dtype, T.float32(0.0))
            if variant == "stage":
                # 2 个都被访问的 UB 数组 → 不触发 G-C8 单死缓冲空句柄
                sc_ub = T.alloc_shared((WIN,), dtype)
                ps_ub = T.alloc_shared((WIN,), dtype)

            with T.Vector():
                for rr in T.serial(RPC):
                    u = core_id * RPC + rr
                    hh = u // seq
                    i = u % seq                      # 查询行号
                    lo = T.max(i - window + 1, 0)    # 可见窗左端（因果+滑窗）
                    n = i - lo + 1                   # 可见键数 ∈ [1, WIN]

                    if variant == "stage":
                        # 趟1：分数进 UB——窗外（j>=n）写哨兵；lo+j ≤ lo+WIN-1，
                        # 当 lo=0 时 lo+WIN-1=window-1 ≤ seq-1，地址恒合法
                        for j in T.serial(WIN):
                            acc = T.float32(0.0)
                            for c in T.serial(dim):
                                acc = acc + Q[hh, i, c] * K[hh, lo + j, c]
                            sc_ub[j] = T.if_then_else(j < n, scale * acc, T.float32(NEG))
                        # 趟2：行最大（哨兵不会赢：窗内至少 1 个真实分数）
                        m = T.float32(NEG)
                        for j in T.serial(WIN):
                            m = T.max(m, sc_ub[j])
                        # 趟3：权重与分母（哨兵侧 expf(NEG-m) 下溢成 0，窗内自带 exp(0)>=1）
                        l = T.float32(0.0)
                        for j in T.serial(WIN):
                            ps_ub[j] = T.exp(sc_ub[j] - m)
                            l = l + ps_ub[j]
                        # 趟4：PV 累加并归一
                        for c in T.serial(dim):
                            acc = T.float32(0.0)
                            for j in T.serial(WIN):
                                acc = acc + ps_ub[j] * V[hh, lo + j, c]
                            O[hh, i, c] = acc / l
                    else:
                        # pure：零 UB，分数三趟各重算一遍（语义同 stage，性能非取向）
                        m = T.float32(NEG)
                        for j in T.serial(WIN):
                            acc = T.float32(0.0)
                            for c in T.serial(dim):
                                acc = acc + Q[hh, i, c] * K[hh, lo + j, c]
                            sc = T.if_then_else(j < n, scale * acc, T.float32(NEG))
                            m = T.max(m, sc)
                        l = T.float32(0.0)
                        for j in T.serial(WIN):
                            acc = T.float32(0.0)
                            for c in T.serial(dim):
                                acc = acc + Q[hh, i, c] * K[hh, lo + j, c]
                            sc = T.if_then_else(j < n, scale * acc, T.float32(NEG))
                            l = l + T.exp(sc - m)
                        for c in T.serial(dim):
                            acc = T.float32(0.0)
                            for j in T.serial(WIN):
                                acc2 = T.float32(0.0)
                                for cc in T.serial(dim):
                                    acc2 = acc2 + Q[hh, i, cc] * K[hh, lo + j, cc]
                                sc = T.if_then_else(j < n, scale * acc2, T.float32(NEG))
                                w = T.exp(sc - m)
                                acc = acc + w * V[hh, lo + j, c]
                            O[hh, i, c] = acc / l

    return attn_sw_kernel


def scan_source(src):
    asc = sorted(set(re.findall(r"\basc_[A-Za-z0-9_]+", src)))
    tlf = sorted(set(re.findall(r"\btl::[A-Za-z0-9_]+", src)))
    mathf = sorted(set(re.findall(
        r"\b(expf|exp2f|logf|sqrtf|rsqrtf|sinf|cosf|fabsf?|tanhf|__cce_[a-z0-9_]+)\s*\(", src)))
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
        "null_ubuf_handle_named": bool(re.search(r"__ubuf__\s+\w+\s*\*\s*(?!buf_dyn_shmem)\w+\s*=\s*\(__ubuf__\s+\w+\s*\*\)\s*0", src)),
        "lines": src.count("\n"),
    }
    return asc, tlf, mathf, feats


def run(args):
    func = attn_sw_fwd(heads=args.heads, seq=args.seq, dim=args.dim,
                       window=args.window, cores=args.cores, dtype=args.dtype,
                       variant=args.variant)
    kernel = tilelang.compile(func, target="ascend", out_idx=[])
    src = kernel.get_kernel_source()
    asc, tlf, mathf, feats = scan_source(src)
    print("E-ATTNSW-COMPILE-PASS")
    print(f"  variant={args.variant} heads={args.heads} seq={args.seq} dim={args.dim} "
          f"window={args.window} cores={args.cores} dtype={args.dtype}")
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
    ap.add_argument("--variant", default=os.environ.get("TL_E_ATTN_VARIANT", "stage"))
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--cores", type=int, default=64)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--dump", default=os.environ.get("TL_E_DUMP", ""))
    args = ap.parse_args()
    try:
        run(args)
    except Exception as e:  # noqa: BLE001
        print("E-ATTNSW-COMPILE-FAIL")
        s = str(e)
        print("---- first error lines ----")
        err = [ln for ln in s.splitlines()
               if re.search(r"error|undefined|undeclared|no member|FATAL|Check failed|NameError", ln)]
        for ln in (err[:8] or s.splitlines()[:8]):
            print("  " + ln.strip()[:220])
        print("---- tail 800 ----")
        print(s[-800:])
        if os.environ.get("TL_E_TRACE"):
            traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
