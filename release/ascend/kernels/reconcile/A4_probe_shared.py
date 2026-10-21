#!/usr/bin/env python3
"""A4_probe_shared.py — 探针：定位 add_ln 产品正文在 cpu-alias 轨编不过的机理。

假设：`add_ln_asc_impl` 里 4 个 `T.alloc_shared`（scope=shared.dyn，见
tilelang/language/allocate.py:49）在 c 后端只允许 1 个 dynamic shared ⇒ 报
`InternalError: Only one dynamic shared memory allocation is allowed.`；
据此验证：(a) n 个 alloc_shared 在 c 后端是否随 n 变红；(b) alloc_local 替代后是否编过。
"""
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

import tilelang  # noqa: E402
import tilelang.cpu.language as T  # noqa: E402

KW = {"target": "c", "target_host": "c", "execution_backend": "cython"}


def build_shared(n):
    @T.prim_func
    def main(A: T.Tensor((64, 512), "float32"), O: T.Tensor((64, 512), "float32")):
        with T.Kernel(1) as bx:
            bufs = [T.alloc_shared((16, 512), "float32") for _ in range(n)]
            T.copy(A[0, 0], bufs[0])
            for i in T.serial(16):
                for j in T.serial(512):
                    O[i, j] = bufs[0][i, j]
    return main


def build_local(n):
    @T.prim_func
    def main(A: T.Tensor((64, 512), "float32"), O: T.Tensor((64, 512), "float32")):
        with T.Kernel(1) as bx:
            bufs = [T.alloc_local((16, 512), "float32") for _ in range(n)]
            T.copy(A[0, 0], bufs[0])
            for i in T.serial(16):
                for j in T.serial(512):
                    O[i, j] = bufs[0][i, j]
    return main


def build_dyn_shared(n, kind="shared"):
    """动态 rows + 动态网格 + 从动态张量 T.copy —— 与 add_ln_asc_impl 同形。

    注意每个 buffer 都要真被读（否则 DCE 会删掉未用 alloc，掩盖真实并发数）。
    """
    alloc = T.alloc_shared if kind == "shared" else T.alloc_local

    rows = T.dynamic("rows")

    @T.prim_func
    def main(A: T.Tensor((rows, 512), "float32"), O: T.Tensor((rows, 512), "float32")):
        with T.Kernel(T.ceildiv(rows, 16)) as bx:
            bufs = [alloc((16, 512), "float32") for _ in range(n)]
            for bi in range(n):
                T.copy(A[bx * 16, 0], bufs[bi])
            for i in T.serial(16):
                row = bx * 16 + i
                if row < rows:
                    for j in T.serial(512):
                        acc = T.alloc_var("float32")
                        acc = bufs[0][i, j]
                        for bi in range(1, n):
                            acc = acc + bufs[bi][i, j]
                        O[row, j] = acc
    return main


def main():
    for n in (1, 2, 4):
        try:
            tilelang.compile(build_shared(n), **KW)
            print(f"static alloc_shared n={n}: OK")
        except Exception as e:  # noqa: BLE001
            print(f"static alloc_shared n={n}: FAIL {type(e).__name__}: {str(e)[:140]}")
    for n in (1, 2, 4):
        try:
            tilelang.compile(build_local(n), **KW)
            print(f"static alloc_local  n={n}: OK")
        except Exception as e:  # noqa: BLE001
            print(f"static alloc_local  n={n}: FAIL {type(e).__name__}: {str(e)[:140]}")
    for n in (1, 2, 4):
        for kind in ("shared", "local"):
            try:
                tilelang.compile(build_dyn_shared(n, kind), **KW)
                print(f"dynamic alloc_{kind:6s} n={n}: OK")
            except Exception as e:  # noqa: BLE001
                print(f"dynamic alloc_{kind:6s} n={n}: FAIL {type(e).__name__}: {str(e)[:140]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
