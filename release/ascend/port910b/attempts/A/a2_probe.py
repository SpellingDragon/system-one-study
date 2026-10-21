#!/usr/bin/env python3
"""A-2 probe: dump the emission surface of the shared.l1 gemm path (alloc_l1 +
alloc_l0c + transpose_B) on 910B — tells us exactly which symbols compat must
back for numeric-correct gemm."""
import os
os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
import tilelang
import tilelang.language as T
from tilelang.ascend.language import Cube, alloc_l1, alloc_l0c

ROWS = COLS = 64


def kern():
    @T.prim_func
    def main(A: T.Tensor((ROWS, COLS), "bfloat16"),
             B: T.Tensor((COLS, COLS), "bfloat16"),
             C: T.Tensor((ROWS, COLS), "float32")):
        with T.Kernel(1):
            a_l1 = alloc_l1((ROWS, COLS), "bfloat16")
            b_l1 = alloc_l1((COLS, COLS), "bfloat16")
            c_l0 = alloc_l0c((ROWS, COLS), "float32")
            with Cube():
                T.copy(A, a_l1)
                T.copy(B, b_l1)
                T.gemm(a_l1, b_l1, c_l0, transpose_B=True, clear_accum=True)
                T.copy(c_l0, C)
    return main


try:
    k = tilelang.compile(kern(), target="ascend", out_idx=-1)
    src = k.get_kernel_source()
    print("A2-COMPILE-PASS")
except Exception as e:
    s = str(e)
    i = s.find("Source:")
    if i >= 0:
        src = s[i + 7:].lstrip("\n")
        print("A2-DUMP-VIA-EXC")
    else:
        print("A2-FAIL:", s[:200])
        src = ""

import re
if src:
    emits = sorted(set(re.findall(r"\basc_[a-z0-9_]+|ascend_gemm_l1<[^>]*>", src)))
    print("EMISSION SURFACE:", emits)
    print("--- first 24 lines ---")
    print("\n".join(src.split("\n")[:24]))
