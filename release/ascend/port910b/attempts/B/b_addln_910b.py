#!/usr/bin/env python3
"""attempts/B/b_addln_910b.py — add_ln（残差相加 + 层内整形）的 910B tilelang DSL kernel

语义对齐 sys1/kernels/add_ln_mps.py（Metal 方言参照）与 sys1/testing/torch_ref/add_ln_ref.py：
  v = X + Res                     （X: fp16 本层输出；Res: fp16 或 fp32 老残差路）
  Hout = v  (fp32)                （残差流出口，全程 fp32，不许降位宽）
  mu = mean_j(v) ; var = mean_j((v-mu)^2)      ← **有偏方差**，与参照逐字一致
  rs = rsqrt(var + eps)
  Y  = cast((v-mu)*rs*G + Bt, ODT)             （ODT=fp16；末层出口走 fp32，见 B_OUT_FP16=0）

三趟扫描（不求一趟 E[x²]-E[x]²）：参照文件第 11 行写明理由——一趟式在均值大、波动小的行上
会灾难性抵消，方差走负 → rsqrt 出 NaN 整层崩。这里逐字保留三趟口径。

910B 方言要点（与 Metal 参照的差异，均为编译面必需，见 RESULT.md ④）：
  · Ascend 的 T.Kernel **没有 threads= 形参**（tilelang/ascend/language/kernel.py:139，只吃 1-D
    core grid），所以"一行一线程 + if tx==r"改成 **按行分块的 bx + 块内 T.serial 逐行**；
  · 逐元素读走 T.copy 把 X/Res/G/Bt 搬进 UB（__ubuf__）后再扫，不再三趟直读 GM；
    数值口径不变（搬的是同一批数），代价是 UB 占用（见 UB_BUDGET）；
  · **T.Parallel 在 Ascend 方言必须位于 VF 块内**（probe_910b_faces.py P9 的语义检查原文），
    910B 非 SIMT 面走不通 → 全部 T.serial 嵌套；
  · store 走纯标量（T.vectorized 段内整块 vector-value store 会撞
    ScalarDcacheBypass: vector-valued buffer store not implemented）；
  · **激活位宽取 fp16 不取 bf16**：bisheng 15.0.5 在 dav-2201 vector 面对 bf16 标量转换直接
    `fatal error: error in backend: not support bf16 type cast`（dump_bf16.py 取证），
    而生产口径本就是 fp16（model.py:329 `.half()`、parity.py:136 `h.to(torch.float16)`）。

依赖 GAP-B 补丁（compat_gap_B.md）：asc_copy_gm2ub_align / asc_sync_notify / asc_sync_wait /
asc_load_l2_cache_mode / 标量 rsqrtf。环境准备见 env_setup.sh + apply_compat_gapB.py。

用法（宿主机）：bash attempts/B/run_b.sh b_addln_910b.py 20
可调环境变量：B_ROWS B_DIM B_BM B_RES_FP32 B_OUT_FP16 B_EPS；`--dyn` 改动态行数再编一次。
"""
import os
import sys

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")

import tilelang
import tilelang.ascend.language as T

ROWS = int(os.environ.get("B_ROWS", "64"))
DIM = int(os.environ.get("B_DIM", "512"))
BM = int(os.environ.get("B_BM", "8"))
RES_FP32 = int(os.environ.get("B_RES_FP32", "1"))   # 老残差路位宽开关，与参照 res_fp32 同义
OUT_FP16 = int(os.environ.get("B_OUT_FP16", "1"))   # 末层出口 fp32 时置 0，与参照 out_fp16 同义
EPS = float(os.environ.get("B_EPS", "1e-5"))
DYN = "--dyn" in sys.argv

F16, F32 = "float16", "float32"
RDT = F32 if RES_FP32 else F16
ODT = F16 if OUT_FP16 else F32

#: UB 占用（字节）= BM*DIM*2(X) + BM*DIM*sizeof(RDT) + 2*DIM*4(G/Bt)
UB_BUDGET = BM * DIM * 2 + BM * DIM * (4 if RES_FP32 else 2) + 2 * DIM * 4


def build():
    """追踪期构造 kernel；rows 可静态（默认）或 T.dynamic（--dyn）。"""
    rows = T.dynamic("rows") if DYN else ROWS

    @T.prim_func
    def main(X: T.Tensor((rows, DIM), F16), Res: T.Tensor((rows, DIM), RDT),
             G: T.Tensor((DIM,), F32), Bt: T.Tensor((DIM,), F32),
             Y: T.Tensor((rows, DIM), ODT), Hout: T.Tensor((rows, DIM), F32)):
        with T.Kernel(T.ceildiv(rows, BM)) as bx:
            x_ub = T.alloc_shared((BM, DIM), F16)
            r_ub = T.alloc_shared((BM, DIM), RDT)
            g_ub = T.alloc_shared((DIM,), F32)
            b_ub = T.alloc_shared((DIM,), F32)
            T.copy(X[bx * BM, 0], x_ub)
            T.copy(Res[bx * BM, 0], r_ub)
            T.copy(G[0:DIM], g_ub)
            T.copy(Bt[0:DIM], b_ub)
            for i in T.serial(BM):
                row = bx * BM + i
                val = T.alloc_var(F32)
                acc = T.alloc_var(F32)
                mu = T.alloc_var(F32)
                dv = T.alloc_var(F32)
                rs = T.alloc_var(F32)
                # 第一趟：残差相加，fp32 原样交出口，同时攒行和
                acc = 0.0
                for j in T.serial(DIM):
                    val = T.cast(x_ub[i, j], F32) + T.cast(r_ub[i, j], F32)
                    Hout[row, j] = val
                    acc = acc + val
                mu = acc / DIM
                # 第二趟：偏差平方和（有偏方差）
                acc = 0.0
                for j in T.serial(DIM):
                    dv = T.cast(x_ub[i, j], F32) + T.cast(r_ub[i, j], F32) - mu
                    acc = acc + dv * dv
                rs = T.rsqrt(acc / DIM + EPS)
                # 第三趟：缩放 + 仿射，按 ODT 交货
                for j in T.serial(DIM):
                    dv = T.cast(x_ub[i, j], F32) + T.cast(r_ub[i, j], F32) - mu
                    Y[row, j] = T.cast(dv * rs * g_ub[j] + b_ub[j], ODT)

    return main


def main_():
    mode = "dynamic-rows" if DYN else f"static rows={ROWS}"
    print(f"B-ADDLN-BEGIN [{mode} dim={DIM} bm={BM} res_dtype={RDT} out_dtype={ODT} "
          f"eps={EPS} UB~{UB_BUDGET // 1024}KB]")
    try:
        k = tilelang.compile(build(), target="ascend", out_idx=-1)
    except Exception as e:  # noqa: BLE001
        lines = [ln.strip() for ln in str(e).splitlines() if ln.strip()]
        errs = [ln for ln in lines if ": error:" in ln or "fatal error" in ln]
        print("B-ADDLN-COMPILE-FAIL")
        for ln in (errs[:2] or lines[-2:]):
            print(f"  ! {ln[:260]}")
        return 1
    src = k.get_kernel_source()
    need = ("asc_copy_gm2ub_align", "asc_sync_notify", "asc_sync_wait", "rsqrtf")
    miss = [s for s in need if s not in src]
    outlets = src.count("Hout") + src.count("Y[")
    print(f"B-ADDLN-GEN len={len(src)} miss={miss} H/Y_refs={outlets} "
          f"kernel='main_kernel' in src: {'main_kernel' in src}")
    if miss:
        print("B-ADDLN-COMPILE-FAIL")
        print(f"  ! 生成码缺少预期符号（GAP-B 未生效？）: {miss}")
        return 1
    print("B-ADDLN-COMPILE-PASS")
    return 0


sys.exit(main_())
