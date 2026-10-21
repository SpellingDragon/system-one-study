#!/usr/bin/env python3
"""attempts/D/d_dw_910b.py — dW（权重梯度 gemm：**dW = dYᵀ @ X**）的 910B tilelang DSL 件。

语义（对齐生产尺子 release/sys1/kernels/gemm_bwd_dw_mps.py:39-63）：
    dY (T_tok, N)  上游梯度，行=token
    X  (T_tok, K)  该层前向的激活输入，行=token
    dW (N, K)      权重梯度，dW[n,k] = SUM_t dY[t,n] * X[t,k]
    归约轴 = **token 数**（生产件里是动态 m，本件按 910B 模板约束取编译期常量）；
    累加出口 fp32（降位宽由入口层负责——保累加精度是纪律）。

910B 建模取向（方案论证全文见 RESULT.md §2，此处只留结论）：
  * Ascend 的 cube mad 只吃 **NT 形态**（A 在 L0A/L1 里 K-major、B 在 L0B/L1 里 K-major），
    `tilelang/ascend/op/gemm/gemm_mad.py:162` 对 shared.l1 路直接 assert
    `not trans_A and trans_B`。dW 天然 trans_A=True（TN），正面撞墙。
  * **本波判决（实测，见 RESULT.md §2）**：转置必须下移到 **DMA 阶段**，且落点选
    **L1→L0 装填**（`asc_copy_l12l0a_transpose` / `asc_copy_l12l0b_transpose`，官方
    dav_c220 8 参件）——主案变体 = **`l0tr`**：L1 存 dY/X 的自然 (token块, 列) 布局，
    装进 L0A/L0B 时转置成 NT 要求的 (N块, token块)/(K块, token块)，mad 永远吃 NT。
    出口 `T.copy(acc, DW[...])` 按 dW 自然朝向直出 ⇒ 等价于转置恒等式
    dW = (Xᵀ @ dY)ᵀ 但将恒等式的"再转置"消灭在入口 DMA，不给出口加第二次重排。
  * `madl1`（把转置交给布局推断：L0 缓冲按自然形状声明 + `T.gemm(..., transpose_A=True)`）
    生成码与 `l0tr` **逐字节相同**（仅 kernel 名不同，dw_l0tr.asc vs dw_madl1.asc）
    ⇒ "direct mad 的 trans_A" 在 codegen 里被降级成同一对 L1→L0 转置件，不是新通路。
  * `madta`（装填不转置、让 mad 直接吃 MN-major A）= **FAIL**：GM→L0A/L0B 无 DMA 面，
    codegen 退化成标量循环解引用 `__ca__` ⇒ CCE `only __ubuf__/__gm__/local memory
    pointer can be dereferenced`（登记 G-D2）。
  * `ntt`（GM→L1 的 **dn2nz** 转置载入 + shared.l1 融合模板，DSL 最短、模板自带 sub-K
    双缓冲）编译绿，但 compat 的 `asc_copy_gm2l1_dn2nz` 现在只是 `nd2nz` 的**同名别名
    stub**（port910b_compat.h:878-882，注 VERIFY V1）⇒ 编译绿≠数值绿，列为
    "性能后续"（登记 G-D3），不作主案。
  * `baseline`：无转置的纯 NT 件（A 已是 (N,token) 朝向），建材回归对照。

已知坑遵守：fp16 输入（bf16 标量面 cast 后端缺陷）；不碰 T.Parallel/SimtVF；
tile 尺寸 16 对齐（`gemm.h:52-53` static_assert M%16==0,N%16==0 与 TILE_K_SUB%16==0）；
L0C→GM 走 `T.copy(acc, ...)`（A-4 已把 asc_copy_l0c2gm 落地）。

判据：compile PASS（本机无 NPU；上卡对拍参考见 d_dw_golden.py）。
用法：python3 d_dw_910b.py [--variant ntt|l0tr|madta|madl1|baseline] [--T 2048] [--N 1024]
                           [--K 1024] [--BN 128] [--BK 128] [--TT 64] [--cores 64]
                           [--dtype float16] [--dump /tmp/dw.asc]
"""
import argparse
import os
import re
import sys
import traceback

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # ascend 方言：1-D 核栅格 Kernel / Cube / copy(transpose=) / gemm  # noqa: E402

VARIANTS = ("ntt", "l0tr", "madta", "madl1", "baseline")


def _require_aligned(name, val, mult=16):
    if val % mult:
        raise ValueError(f"{name}={val} 必须 {mult} 对齐（gemm.h static_assert M/N%16==0）")


def dw_ntt(T_tok, N, K, BN, BK, TT, cores, dtype, acc_dtype):
    """主案：GM→L1 dn2nz 转置载入 + shared.l1 融合 NT 模板路（ascend_gemm_l1）。

    L1 里 A_l1=(BN,TT)、B_l1=(BK,TT) 均以 **token 轴为 C0**（K-major），
    正是模板要求的 NT 形态；dY/X 的 (TT,BN)/(TT,BK) 切片由 transpose=True 转置装入。
    """
    NT = N // BN
    NK = K // BK
    KS = T_tok // TT
    TILES = NT * NK
    BLOCKS = min(TILES, cores)
    ITER = TILES // BLOCKS

    @T.prim_func
    def dW_dw(
        dY: T.Tensor((T_tok, N), dtype),
        X: T.Tensor((T_tok, K), dtype),
        DW: T.Tensor((N, K), acc_dtype),
    ):
        with T.Kernel(BLOCKS) as bid:
            a_l1 = T.alloc_l1((BN, TT), dtype)   # A = dYᵀ 块 (M=BN, K=TT) K-major
            b_l1 = T.alloc_l1((BK, TT), dtype)   # B = Xᵀ  块 (N=BK, K=TT) K-major
            acc = T.alloc_l0c((BN, BK), acc_dtype)
            with T.Cube():
                for it in T.serial(ITER):
                    tid = bid + it * BLOCKS
                    n_tile = tid // NK
                    k_tile = tid % NK
                    for kt in T.serial(KS):
                        T.copy(dY[kt * TT:(kt + 1) * TT, n_tile * BN:(n_tile + 1) * BN],
                               a_l1, transpose=True)
                        T.copy(X[kt * TT:(kt + 1) * TT, k_tile * BK:(k_tile + 1) * BK],
                               b_l1, transpose=True)
                        # C = A @ Bᵀ = dYᵀ @ X  →  天然就是 dW，无需出口再转置
                        T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=kt == 0)
                    T.copy(acc, DW[n_tile * BN:(n_tile + 1) * BN,
                                   k_tile * BK:(k_tile + 1) * BK])

    return dW_dw


def dw_baseline(T_tok, N, K, BN, BK, TT, cores, dtype, acc_dtype):
    """建材回归对照：同尺寸、无转置的**纯 NT 前向件**（A=(BN,TT) 直读）。

    语义 = baseline C = dY_tᵀ @ X_t 的"假装"版（不做真转置载入，切片按 (BN,TT) 取），
    只用来判定"dn2nz 转置面"以外的一切是否绿：baseline 绿 + ntt 挂 ⇒ 缺口在转置面。
    """
    NT = N // BN
    NK = K // BK
    KS = T_tok // TT
    TILES = NT * NK
    BLOCKS = min(TILES, cores)
    ITER = TILES // BLOCKS

    @T.prim_func
    def dW_base(
        A: T.Tensor((N, T_tok), dtype),
        B: T.Tensor((K, T_tok), dtype),
        DW: T.Tensor((N, K), acc_dtype),
    ):
        with T.Kernel(BLOCKS) as bid:
            a_l1 = T.alloc_l1((BN, TT), dtype)
            b_l1 = T.alloc_l1((BK, TT), dtype)
            acc = T.alloc_l0c((BN, BK), acc_dtype)
            with T.Cube():
                for it in T.serial(ITER):
                    tid = bid + it * BLOCKS
                    n_tile = tid // NK
                    k_tile = tid % NK
                    for kt in T.serial(KS):
                        T.copy(A[n_tile * BN:(n_tile + 1) * BN, kt * TT:(kt + 1) * TT], a_l1)
                        T.copy(B[k_tile * BK:(k_tile + 1) * BK, kt * TT:(kt + 1) * TT], b_l1)
                        T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=kt == 0)
                    T.copy(acc, DW[n_tile * BN:(n_tile + 1) * BN,
                                   k_tile * BK:(k_tile + 1) * BK])

    return dW_base


def dw_l0tr(T_tok, N, K, BN, BK, TT, cores, dtype, acc_dtype):
    """备选一：L1 存**自然布局** (TT,BN)/(TT,BK)，转置下移到 L1→L0 装填。

    发射面 = asc_copy_l12l0a_transpose / asc_copy_l12l0b_transpose（codegen_ascend.cc:1563-1567）。
    主仓有该形的用例 testing/ascend/layout/test_ascend_l0_transpose.py:29（只 L0A）。
    """
    NT = N // BN
    NK = K // BK
    KS = T_tok // TT
    TILES = NT * NK
    BLOCKS = min(TILES, cores)
    ITER = TILES // BLOCKS

    @T.prim_func
    def dW_l0tr(
        dY: T.Tensor((T_tok, N), dtype),
        X: T.Tensor((T_tok, K), dtype),
        DW: T.Tensor((N, K), acc_dtype),
    ):
        with T.Kernel(BLOCKS) as bid:
            dy_l1 = T.alloc_l1((TT, BN), dtype)   # 自然 (token, n)
            x_l1 = T.alloc_l1((TT, BK), dtype)    # 自然 (token, k)
            a_l0 = T.alloc_l0a((BN, TT), dtype)   # NT 要求：(M=BN, K=TT)
            b_l0 = T.alloc_l0b((BK, TT), dtype)   # NT 要求：(N=BK, K=TT)
            acc = T.alloc_l0c((BN, BK), acc_dtype)
            with T.Cube():
                for it in T.serial(ITER):
                    tid = bid + it * BLOCKS
                    n_tile = tid // NK
                    k_tile = tid % NK
                    for kt in T.serial(KS):
                        T.copy(dY[kt * TT:(kt + 1) * TT, n_tile * BN:(n_tile + 1) * BN], dy_l1)
                        T.copy(X[kt * TT:(kt + 1) * TT, k_tile * BK:(k_tile + 1) * BK], x_l1)
                        T.copy(dy_l1, a_l0, transpose=True)
                        T.copy(x_l1, b_l0, transpose=True)
                        T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=kt == 0)
                    T.copy(acc, DW[n_tile * BN:(n_tile + 1) * BN,
                                   k_tile * BK:(k_tile + 1) * BK])

    return dW_l0tr


def dw_madta(T_tok, N, K, BN, BK, TT, cores, dtype, acc_dtype):
    """备选二（面探测）：direct mad，L0 里直接 `transpose_A=True`，零转置搬运。

    A 缓冲按 (K=TT, M=BN)、B 按 (K=TT, N=BK) 声明，布局推断走
    `make_ascend_major_mn_layout`（gemm_mad.py:132-133）。编译面通不代表数值对：
    compat 的 asc_mmad 把 kDirectionAlign/btbuf_ctrl 硬编码 false（port910b_compat.h:913-919），
    MN-major A 是否被 910B mad 真吃需上卡证真 → 只作对照，不当主案。
    """
    NT = N // BN
    NK = K // BK
    KS = T_tok // TT
    TILES = NT * NK
    BLOCKS = min(TILES, cores)
    ITER = TILES // BLOCKS

    @T.prim_func
    def dW_madta(
        dY: T.Tensor((T_tok, N), dtype),
        X: T.Tensor((T_tok, K), dtype),
        DW: T.Tensor((N, K), acc_dtype),
    ):
        with T.Kernel(BLOCKS) as bid:
            a_l0 = T.alloc_l0a((TT, BN), dtype)   # (K, M)，transpose_A=True
            b_l0 = T.alloc_l0b((TT, BK), dtype)   # (K, N)，transpose_B=False
            acc = T.alloc_l0c((BN, BK), acc_dtype)
            with T.Cube():
                for it in T.serial(ITER):
                    tid = bid + it * BLOCKS
                    n_tile = tid // NK
                    k_tile = tid % NK
                    for kt in T.serial(KS):
                        T.copy(dY[kt * TT:(kt + 1) * TT, n_tile * BN:(n_tile + 1) * BN], a_l0)
                        T.copy(X[kt * TT:(kt + 1) * TT, k_tile * BK:(k_tile + 1) * BK], b_l0)
                        T.gemm(a_l0, b_l0, acc, transpose_A=True, transpose_B=False,
                               clear_accum=kt == 0)
                    T.copy(acc, DW[n_tile * BN:(n_tile + 1) * BN,
                                   k_tile * BK:(k_tile + 1) * BK])

    return dW_madta


def dw_madl1(T_tok, N, K, BN, BK, TT, cores, dtype, acc_dtype):
    """备选三（判决用）：L1 存自然布局，**装填不转置**，gemm 直接 `transpose_A=True`。

    这一条是"direct mad 方案"的 DSL 正写：转置交给 L0A 的 MN-major 布局推断
    （gemm_mad.py:132-133 `trans_A -> make_ascend_major_mn_layout`），搬运面零转置。
    编得过 ⇒ direct mad 有路；编不过/发射退化 ⇒ 方案论证里"转置恒等式"胜出有实证。
    """
    NT = N // BN
    NK = K // BK
    KS = T_tok // TT
    TILES = NT * NK
    BLOCKS = min(TILES, cores)
    ITER = TILES // BLOCKS

    @T.prim_func
    def dW_madl1(
        dY: T.Tensor((T_tok, N), dtype),
        X: T.Tensor((T_tok, K), dtype),
        DW: T.Tensor((N, K), acc_dtype),
    ):
        with T.Kernel(BLOCKS) as bid:
            dy_l1 = T.alloc_l1((TT, BN), dtype)
            x_l1 = T.alloc_l1((TT, BK), dtype)
            a_l0 = T.alloc_l0a((TT, BN), dtype)   # (K, M) 原样装填，靠 MN-major 布局吃转置
            b_l0 = T.alloc_l0b((TT, BK), dtype)   # (K, N) 原样装填
            acc = T.alloc_l0c((BN, BK), acc_dtype)
            with T.Cube():
                for it in T.serial(ITER):
                    tid = bid + it * BLOCKS
                    n_tile = tid // NK
                    k_tile = tid % NK
                    for kt in T.serial(KS):
                        T.copy(dY[kt * TT:(kt + 1) * TT, n_tile * BN:(n_tile + 1) * BN], dy_l1)
                        T.copy(X[kt * TT:(kt + 1) * TT, k_tile * BK:(k_tile + 1) * BK], x_l1)
                        T.copy(dy_l1, a_l0)
                        T.copy(x_l1, b_l0)
                        T.gemm(a_l0, b_l0, acc, transpose_A=True, transpose_B=False,
                               clear_accum=kt == 0)
                    T.copy(acc, DW[n_tile * BN:(n_tile + 1) * BN,
                                   k_tile * BK:(k_tile + 1) * BK])

    return dW_madl1


BUILDERS = {"ntt": dw_ntt, "l0tr": dw_l0tr, "madta": dw_madta,
          "madl1": dw_madl1, "baseline": dw_baseline}


def scan_source(src):
    """codegen 产物体检：转置面/mad 面/L0C 出口面各自落到哪个符号（缺口取证来源）。"""
    asc = sorted(set(re.findall(r"\basc_[A-Za-z0-9_]+|ascend_gemm_l1<[^>]*>", src)))
    m = re.search(r'extern "C" (__global__[^(]*)', src)
    feats = {
        "prefix": (m.group(1).strip() if m else "?"),
        "lines": src.count("\n"),
        "has_mix": "__mix__" in src,
        "has_ASC_IS_": "ASC_IS_" in src,
        "has_threadIdx": "threadIdx" in src,
        "has_vf_call": "asc_vf_call" in src,
        "has_ubuf": "__ubuf__" in src,
        "has_dn2nz": "asc_copy_gm2l1_dn2nz" in src,
        "has_nd2nz": "asc_copy_gm2l1_nd2nz" in src,
        "has_l1tmpl": "ascend_gemm_l1<" in src,
        "has_mmad": "asc_mmad" in src,
        "has_l0c2gm": "asc_copy_l0c2gm" in src,
        "has_l12l0": bool(re.search(r"asc_copy_l12l0[ab]", src)),
        "has_l12l0_tr": bool(re.search(r"asc_copy_l12l0[ab]_transpose", src)),
        "clear_accum_0": src.count("true") + src.count("false") > 0,
    }
    return asc, feats


def build(args):
    _require_aligned("BN", args.BN)
    _require_aligned("BK", args.BK)
    _require_aligned("TT", args.TT)
    _require_aligned("N", args.N)
    _require_aligned("K", args.K)
    _require_aligned("T", args.T)
    if args.T % args.TT:
        raise ValueError(f"T={args.T} 必须被 TT={args.TT} 整除（L1 模板 K%TILE_K_SUB==0）")
    func = BUILDERS[args.variant](args.T, args.N, args.K, args.BN, args.BK, args.TT,
                                  args.cores, args.dtype, args.acc_dtype)
    kernel = tilelang.compile(func, target="ascend", out_idx=-1)
    src = kernel.get_kernel_source()
    asc, feats = scan_source(src)
    print("D-DW-COMPILE-PASS")
    print(f"  variant={args.variant} T={args.T} N={args.N} K={args.K} "
          f"BN={args.BN} BK={args.BK} TT={args.TT} cores={args.cores} "
          f"dtype={args.dtype} acc={args.acc_dtype}")
    for k in ("lines", "has_mix", "has_ASC_IS_", "has_threadIdx", "has_vf_call",
              "has_ubuf", "has_dn2nz", "has_nd2nz", "has_l1tmpl", "has_mmad",
              "has_l0c2gm", "has_l12l0", "has_l12l0_tr"):
        print(f"  {k}: {feats[k]}")
    print("  emitted asc_* :", " ".join(asc) or "(none)")
    if args.dump:
        with open(args.dump, "w") as f:
            f.write(src)
        print(f"  source dumped -> {args.dump}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # 默认主案 l0tr（转置下移到 L1→L0，官方 8 参 transpose 件；论证见 RESULT.md §2）
    ap.add_argument("--variant", default=os.environ.get("TL_D_VARIANT", "l0tr"),
                    choices=VARIANTS)
    ap.add_argument("--T", type=int, default=int(os.environ.get("TL_D_T", 2048)))
    ap.add_argument("--N", type=int, default=int(os.environ.get("TL_D_N", 1024)))
    ap.add_argument("--K", type=int, default=int(os.environ.get("TL_D_K", 1024)))
    ap.add_argument("--BN", type=int, default=128)
    ap.add_argument("--BK", type=int, default=128)
    ap.add_argument("--TT", type=int, default=64)
    ap.add_argument("--cores", type=int, default=64)
    ap.add_argument("--dtype", default=os.environ.get("TL_D_DTYPE", "float16"))
    ap.add_argument("--acc-dtype", dest="acc_dtype", default="float32")
    ap.add_argument("--dump", default=os.environ.get("TL_D_DUMP", ""))
    args = ap.parse_args()
    try:
        build(args)
    except Exception as e:  # noqa: BLE001
        print("D-DW-COMPILE-FAIL")
        print("  variant:", args.variant)
        s = str(e)
        err = [ln for ln in s.splitlines()
               if re.search(r"error|undefined|undeclared|no matching|no member|"
                            r"assert|Check failed|ValueError|TypeError|FATAL", ln)]
        print("---- first error ----")
        for ln in (err[:6] or s.splitlines()[:6]):
            print("  " + ln.strip()[:260])
        print("---- tail 1200 ----")
        print(s[-1200:])
        if os.environ.get("TL_D_TRACE"):
            traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
