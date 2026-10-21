#!/usr/bin/env python3
"""attempts/E/e_rope_910b.py — RoPE（旋转位置编码）910B tilelang DSL 件（AIV 标量面）。

语义（与 system-one track-A 生产件 rope_asc.py / torch_ref rope_ref.py 完全同口径）：
    packed QKV (tokens,3,heads,dim)，只旋转 slot0=Q、slot1=K，slot2=V 原样透传；
    rotate-half（前后半）：x1' = x1*cos - x2*sign*sin；x2' = x2*cos + x1*sign*sin
    （sign=+1 前向、-1 反向=逆旋转；rot 即半维交错取负）。cos/sin 表 (tokens, dim//2) fp32。

910B 建模取向（本波，沿用 C 波 G-C1/G-C8 实证纪律）：
  * 只用 AIV 标量面：`T.Kernel(cores)` + `T.Vector()` 区域 + 纯 `T.serial`（核块 →
    (token,head) 单元 → slot → 半维）。不用 T.Parallel/SimtVF/T.copy。
  * **cos/sin 走预计算表从 GM 标量直读**（G-C1：910B 标量面 sinf/cosf 整条不存在；
    表加载是生产 rope_asc/rope_ref 既有形态，无需任何 transcendental）。
  * 累加/暂存全走 `T.alloc_var`，**零 alloc_shared**（G-C8：单死缓冲落 (__ubuf__ T*)0
    命名空句柄；纯标量件 refs=0 最稳）。
  * 就地 vs 出地：生产件就地改 QKV；本波 910B 件 **出地**（O 另表）——就地语义依赖
    "先物化 x1/x2 再写回"的顺序纪律（_eager 注释里踩过自噬坑），fp32 编译波不需要
    这个额外风险面；透传 slot2 反而多验一条标量 GM 读→bypass 写通路。
  * --mode sinf / --mode rational 是**取证变体**（论证"表加载"取舍）：
      sinf    : T.cos/T.sin(fp32) → codegen `cosf/sinf`（intrin_rule_ascend.cc:29
                AscendMath float32→name+'f'，:56-57 已注册）→ compat v3 无此二符号
                → 预期 **FAIL**（"自算"路线在本硬件可复跑的反证）。
      rational: 无三角函数库件时的多项式兜底骨架（只吃四则；数值非对拍真值）。

用法：python3 e_rope_910b.py [--mode table|sinf|rational] [--tokens 64] [--heads 8]
                             [--dim 64] [--cores 64] [--sign 1|-1] [--dump f]
判据：compile PASS（本机无 NPU；数值对拍件见 e_golden.py）。
注意：**不能** `from __future__ import annotations`（tilelang eager builder 的
get_type_hints 会把闭包变量当模块全局名 → NameError，C 波实测）。
"""
import argparse
import os
import re
import sys
import traceback

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402


def rope_fwd(tokens=64, heads=8, dim=64, cores=64, dtype="float32",
             mode="table", sign=1):
    """构造 RoPE 前向/反向 prim_func（sign 为编译期常量，与生产 rope_asc 同参数化）。"""
    if dim % 2:
        raise ValueError(f"dim 需为偶数（折半旋转），实得 {dim}")
    half = dim // 2
    units = tokens * heads
    if units % cores:
        raise ValueError(f"units=tokens*heads={units} 必须能被 cores={cores} 整除")
    UPC = units // cores

    @T.prim_func
    def rope_kernel(
        QKV: T.Tensor((tokens, 3, heads, dim), dtype),
        O: T.Tensor((tokens, 3, heads, dim), dtype),
        Cos: T.Tensor((tokens, half), dtype),
        Sin: T.Tensor((tokens, half), dtype),
        FREQ: T.Tensor((half,), dtype),
    ):
        with T.Kernel(cores) as core_id:
            x1 = T.alloc_var(dtype, T.float32(0.0))
            x2 = T.alloc_var(dtype, T.float32(0.0))
            c = T.alloc_var(dtype, T.float32(1.0))
            sn = T.alloc_var(dtype, T.float32(0.0))
            ang = T.alloc_var(dtype, T.float32(0.0))

            with T.Vector():
                for uc in T.serial(UPC):
                    u = core_id * UPC + uc
                    t = u // heads          # token 行
                    h = u % heads           # 头号
                    for s in T.serial(2):   # slot0=Q, slot1=K；slot2=V 走透传段
                        for j in T.serial(half):
                            x1 = QKV[t, s, h, j]
                            x2 = QKV[t, s, h, j + half]
                            if mode == "table":
                                c = Cos[t, j]
                                sn = Sin[t, j]
                            elif mode == "sinf":
                                # 取证变体：核内自算角度 → T.cos/T.sin → cosf/sinf（缺件）
                                ang = T.float32(t) * FREQ[j]
                                c = T.cos(ang)
                                sn = T.sin(ang)
                            else:  # rational：5 阶泰勒 sin / 4 阶 cos（|x|<=pi 精度退化，非真值）
                                ang = T.float32(t) * FREQ[j]
                                a2 = ang * ang
                                c = T.float32(1.0) + a2 * (T.float32(-0.5) +
                                    a2 * (T.float32(0.041666668) + a2 * T.float32(-0.0013888889)))
                                sn = ang * (T.float32(1.0) + a2 * (T.float32(-0.16666667) +
                                    a2 * (T.float32(0.008333334) + a2 * T.float32(-0.0001984127))))
                            if sign < 0:
                                sn = T.float32(0.0) - sn
                            O[t, s, h, j] = x1 * c - x2 * sn
                            O[t, s, h, j + half] = x2 * c + x1 * sn
                    # V（slot2）原样透传：同时验标量 GM 读→写通路
                    for j in T.serial(dim):
                        O[t, 2, h, j] = QKV[t, 2, h, j]

    return rope_kernel


def scan_source(src):
    """codegen 产物体检（与 C 波同格式，供结构结论溯源）。"""
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
        "lines": src.count("\n"),
    }
    return asc, tlf, mathf, feats


def run(args):
    func = rope_fwd(tokens=args.tokens, heads=args.heads, dim=args.dim,
                    cores=args.cores, dtype=args.dtype, mode=args.mode, sign=args.sign)
    kernel = tilelang.compile(func, target="ascend", out_idx=[])
    src = kernel.get_kernel_source()
    asc, tlf, mathf, feats = scan_source(src)
    print("E-ROPE-COMPILE-PASS")
    print(f"  mode={args.mode} sign={args.sign} tokens={args.tokens} heads={args.heads} "
          f"dim={args.dim} cores={args.cores} dtype={args.dtype}")
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
    ap.add_argument("--mode", default=os.environ.get("TL_E_ROPE_MODE", "table"))
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--cores", type=int, default=64)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--sign", type=int, default=1, choices=[1, -1])
    ap.add_argument("--dump", default=os.environ.get("TL_E_DUMP", ""))
    args = ap.parse_args()
    try:
        run(args)
    except Exception as e:  # noqa: BLE001
        print("E-ROPE-COMPILE-FAIL")
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
