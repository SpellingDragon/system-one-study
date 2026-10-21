#!/usr/bin/env python3
"""attempts/B/dump_p7.py — T.gemm 在 Cube() 之外的失败全文取证（P7 判决的补充材料）

只打印首个错误的**完整行**（probe_910b_faces.py 为了排版截到 220 字，这里不截）。
目的：判定这是 compat 的 asc_mmad 缺口、还是方言用法问题（gemm 必须落在 Cube 面）。
"""
import os

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
import tilelang
import tilelang.ascend.language as T

BT, KT, DIM = 16, 16, 64


@T.prim_func
def main(A: T.Tensor((BT, DIM), "float32"), B: T.Tensor((DIM, KT), "float32"),
         C: T.Tensor((BT, KT), "float32")):
    with T.Kernel(1) as bx:
        a = T.alloc_shared((BT, DIM), "float32")
        b = T.alloc_shared((DIM, KT), "float32")
        c = T.alloc_shared((BT, KT), "float32")
        T.copy(A[0, 0], a)
        T.copy(B[0, 0], b)
        T.gemm(a, b, c, clear_accum=True)
        T.copy(c, C[0, 0])


try:
    tilelang.compile(main, target="ascend", out_idx=-1)
    print("P7: PASS")
except Exception as e:  # noqa: BLE001
    errs = [ln.strip() for ln in str(e).splitlines() if ": error:" in ln]
    print(f"P7: FAIL ({len(errs)} error lines)")
    for ln in errs[:3]:
        print("  FULL:", ln)
