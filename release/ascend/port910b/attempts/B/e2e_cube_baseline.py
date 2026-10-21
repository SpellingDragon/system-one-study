#!/usr/bin/env python3
"""P0-2 e2e verdict v2: explicit single-work Cube kernel (T.Cube block =>
codegen kCube mode => no __mix__/no cross-core-sync registration). This is the
ALPHA route's phase-1 shape for 910B: AIC-only gemm."""
import os
os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
import tilelang
import tilelang.language as T
from tilelang.ascend.language import Cube  # explicit Cube work block


def gemm(M, N, K, dtype, out_dtype):
    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype),
             C: T.Tensor((M, N), out_dtype)):
        with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=1) as (bx, by):
            A_s = T.alloc_shared((128, K), "bfloat16")
            B_s = T.alloc_shared((K, 128), "bfloat16")
            C_l = T.alloc_shared((128, 128), "float")
            with Cube():
                T.copy(A[by * 128, 0], A_s)
                T.copy(B[0, bx * 128], B_s)
                T.gemm(A_s, B_s, C_l, clear_accum=True)
                for i in T.serial(128):
                    for j in T.serial(128):
                        C[by * 128 + i, bx * 128 + j] = T.cast(C_l[i, j], out_dtype)
    return main


try:
    k = tilelang.compile(gemm(256, 256, 256, "bfloat16", "bfloat16"),
                         target="ascend", out_idx=-1)
    print("E2E-CUBE-ONLY PASS")
    src = k.get_kernel_source()
    print("HAS_MIX:", "__mix__" in src, " HAS_ASC_IS:", "ASC_IS_" in src)
except Exception as e:
    print("E2E-CUBE-ONLY-FAIL:", str(e)[:200])
    s = str(e)
    i = s.find("Compiler output")
    print(s[i:i + 700] if i >= 0 else "")
