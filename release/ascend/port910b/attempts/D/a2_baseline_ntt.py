#!/usr/bin/env python3
"""attempts/D/a2_baseline_ntt.py — 建材回归对照件（A案 A2 形态，fp16 版）。

作用：判定"shared.l1 融合 NT 模板路 + L0C→GM 出口"在 cann910b-d 里是否仍绿。
d_dw_910b.py 的 ntt/l0tr 挂而本件绿 ⇒ 缺口在**转置面**（dn2nz / l12l0_transpose），
本件也挂 ⇒ 容器 compat/bisheng 装配或 tilelang 模板树回归问题（非 dW 方案本身）。
判据：D-A2-BASELINE-PASS/FAIL。
"""
import os

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402

ROWS = COLS = 64


def kern():
    @T.prim_func
    def main(A: T.Tensor((ROWS, COLS), "float16"),
             B: T.Tensor((COLS, COLS), "float16"),
             C: T.Tensor((ROWS, COLS), "float32")):
        with T.Kernel(1) as bid:
            a_l1 = T.alloc_l1((ROWS, COLS), "float16")
            b_l1 = T.alloc_l1((COLS, COLS), "float16")
            c_l0 = T.alloc_l0c((ROWS, COLS), "float32")
            with T.Cube():
                T.copy(A, a_l1)
                T.copy(B, b_l1)
                T.gemm(a_l1, b_l1, c_l0, transpose_B=True, clear_accum=True)
                T.copy(c_l0, C)
    return main


try:
    k = tilelang.compile(kern(), target="ascend", out_idx=-1)
    src = k.get_kernel_source()
    print("D-A2-BASELINE-PASS")
    print("  lines:", src.count("\n"))
    print("  has_l1tmpl:", "ascend_gemm_l1<" in src, " has_mmad:", "asc_mmad" in src,
          " has_l0c2gm:", "asc_copy_l0c2gm" in src, " has_mix:", "__mix__" in src)
except Exception as e:  # noqa: BLE001
    print("D-A2-BASELINE-FAIL")
    print(str(e)[:600])
