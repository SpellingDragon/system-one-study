#!/usr/bin/env python3
"""attempts/B/dump_one.py — 编译单个探针 kernel，dump 生成码并打印 **错误行上下文**

用法: python3 dump_one.py <case>   （case ∈ vec|ub2gm|gather|exp|rsqrt）
输出：首个 `: error:` 行 + 该错误指向的生成码行 ±2，用于区分 compat 缺口 / 方言用法错。
"""
import os
import re
import sys

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
from tilelang.contrib import bisheng as bs

CASE = sys.argv[1] if len(sys.argv) > 1 else "exp"
OUT = f"/tmp/agentB/gen/gen_{CASE}.asc"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
_orig = bs.compile_ascend


def spy(code, *a, **kw):
    with open(OUT, "w") as f:
        f.write(code)
    return _orig(code, *a, **kw)


bs.compile_ascend = spy

import tilelang
import tilelang.ascend.language as T

N, DIM, BM = 1024, 256, 4
BT, TT, KT = 8, 16, 32


def k_exp():
    @T.prim_func
    def main(S: T.Tensor((BT, KT), "float32"), C: T.Tensor((BT, KT), "float32"),
             O: T.Tensor((BT, KT), "float32")):
        with T.Kernel(1) as bx:
            s = T.alloc_shared((BT, KT), "float32")
            T.copy(S[0, 0], s)
            for i in T.serial(BT):
                mx = T.alloc_var("float32")
                sm = T.alloc_var("float32")
                mx = s[i, 0]
                sm = 0.0
                for k in T.serial(KT):
                    mx = T.max(mx, s[i, k])
                for k in T.serial(KT):
                    sm = sm + T.exp(s[i, k] - mx)
                for k in T.serial(KT):
                    C[i, k] = sm
                    O[i, k] = T.exp(s[i, k] - mx) / sm
    return main


def k_rsqrt():
    @T.prim_func
    def main(A: T.Tensor((N, DIM), "float32"), C: T.Tensor((N, DIM), "float32"),
             O: T.Tensor((N, DIM), "float32")):
        with T.Kernel(T.ceildiv(N, BM)) as bx:
            a = T.alloc_shared((BM, DIM), "float32")
            T.copy(A[bx * BM, 0], a)
            for i in T.serial(BM):
                acc = T.alloc_var("float32")
                rs = T.alloc_var("float32")
                acc = 0.0
                for j in T.serial(DIM):
                    acc = acc + a[i, j] * a[i, j]
                rs = T.rsqrt(acc / DIM + 1e-5)
                for j in T.serial(DIM):
                    C[i, j] = rs
                    O[bx * BM + i, j] = a[i, j] * rs
    return main


FN = {"exp": k_exp, "rsqrt": k_rsqrt}[CASE]
try:
    tilelang.compile(FN(), target="ascend", out_idx=-1)
    print("COMPILE-PASS")
except Exception as e:  # noqa: BLE001
    s = str(e)
    errs = [ln for ln in s.splitlines() if ": error:" in ln]
    print("COMPILE-FAIL, first errors:")
    for ln in errs[:4]:
        print("   ", ln.strip()[:200])
    m = re.search(r"tl_kernel\.asc:(\d+):", errs[0]) if errs else None
    if m and os.path.exists(OUT):
        n = int(m.group(1))
        lines = open(OUT).read().splitlines()
        print(f"=== 生成码 {OUT} 第 {n} 行上下文 ===")
        for i in range(max(0, n - 3), min(len(lines), n + 2)):
            mark = ">>" if i == n - 1 else "  "
            print(f"{mark}{i+1:4d}: {lines[i][:200]}")
    else:
        print("=== 生成码全文（前 40 行）===")
        if os.path.exists(OUT):
            for i, ln in enumerate(open(OUT).read().splitlines()[:40], 1):
                print(f"  {i:4d}: {ln[:200]}")
