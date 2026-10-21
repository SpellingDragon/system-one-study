#!/usr/bin/env python3
"""attempts/B/dump_generated.py — 把 910B 面 codegen 生成的 .asc 源码 dump 出来，
统计 emit 了哪些 `asc_*` / `tl::` / CCE 符号，用于定位 port910b_compat 的缺口全集。

做法：patch tilelang.contrib.bisheng.compile_ascend（tilelang/ascend/backend.py:41 走
属性查找，故补丁生效），在真正调用编译器前把 code 落盘；编译失败也照样拿到源码。
"""
import os
import re

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
from tilelang.contrib import bisheng as bs

OUT = "/tmp/agentB/gen"
os.makedirs(OUT, exist_ok=True)
_orig = bs.compile_ascend
CUR = {"tag": "x", "path": None}


def spy(code, *a, **kw):
    with open(CUR["path"], "w") as f:
        f.write(code)
    return _orig(code, *a, **kw)


bs.compile_ascend = spy

import tilelang
import tilelang.ascend.language as T

N, DIM, BM = 1024, 256, 4
BT, TT, KT = 8, 16, 32


def k_vec():
    """gm->ub copy + 标量 serial + 标量 gm store（add_ln 骨架）"""
    @T.prim_func
    def main(A: T.Tensor((N, DIM), "float32"), B: T.Tensor((N, DIM), "float32"),
             C: T.Tensor((N, DIM), "float32")):
        with T.Kernel(T.ceildiv(N, BM)) as bx:
            a = T.alloc_shared((BM, DIM), "float32")
            b = T.alloc_shared((BM, DIM), "float32")
            T.copy(A[bx * BM, 0], a)
            T.copy(B[bx * BM, 0], b)
            for i in T.serial(BM):
                for j in T.serial(DIM):
                    C[bx * BM + i, j] = a[i, j] + b[i, j]
    return main


def k_ub2gm():
    """ub->gm T.copy（asc_copy_ub2gm_align / store 侧枚举）"""
    @T.prim_func
    def main(A: T.Tensor((N, DIM), "float32"), B: T.Tensor((N, DIM), "float32"),
             C: T.Tensor((N, DIM), "float32")):
        with T.Kernel(T.ceildiv(N, BM)) as bx:
            a = T.alloc_shared((BM, DIM), "float32")
            b = T.alloc_shared((BM, DIM), "float32")
            c = T.alloc_shared((BM, DIM), "float32")
            T.copy(A[bx * BM, 0], a)
            T.copy(B[bx * BM, 0], b)
            for i in T.serial(BM):
                for j in T.serial(DIM):
                    c[i, j] = a[i, j] + b[i, j]
            T.copy(c, C[bx * BM, 0])
    return main


def k_gather():
    """3D 动态下标 gather（readout 取末位）"""
    @T.prim_func
    def main(H: T.Tensor((BT, TT, DIM), "float32"), IDX: T.Tensor((BT,), "int32"),
             C: T.Tensor((BT, DIM), "float32")):
        with T.Kernel(1) as bx:
            pos = T.alloc_shared((BT,), "int32")
            row = T.alloc_shared((BT, DIM), "float32")
            T.copy(IDX[0:BT], pos)
            for i in T.serial(BT):
                for j in T.serial(DIM):
                    row[i, j] = H[i, pos[i], j]
            T.copy(row, C[0, 0])
    return main


def k_gemm_nocube():
    """T.gemm 不在 Cube() 内（readout 若走投影）"""
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
    return main


CASES = [("vec", k_vec), ("ub2gm", k_ub2gm), ("gather", k_gather), ("gemm", k_gemm_nocube)]

print("=== DUMP-GENERATED-910B ===")
for tag, fn in CASES:
    path = os.path.join(OUT, f"gen_{tag}.asc")
    CUR["path"] = path
    try:
        tilelang.compile(fn(), target="ascend", out_idx=-1)
        st = "COMPILE-PASS"
    except Exception as e:  # noqa: BLE001
        s = str(e)
        eline = ""
        for ln in s.splitlines():
            if "error" in ln.lower():
                eline = ln.strip()[:180]
                break
        st = f"COMPILE-FAIL | {eline or s.strip().splitlines()[0][:180] if s.strip() else repr(e)[:180]}"
    if not os.path.exists(path):
        print(f"[{tag}] NO-SOURCE-DUMPED {st}")
        continue
    src = open(path).read()
    print(f"[{tag}] {st}")
    print("   ASC_SYMS:", sorted(set(re.findall(r"\b(asc_[a-z0-9_]+)\b", src))))
    print("   TL_SYMS :", sorted(set(re.findall(r"\btl::([a-z0-9_]+)\b", src))))
    print("   CCE_SYMS:", sorted(set(re.findall(r"\b(__cce_[a-z0-9_]+|copy_gm_to_ubuf|copy_ubuf_to_gm|copy_gm_to_cbuf|mad)\b", src))))
    print(f"   KEPT: {path} ({len(src)} bytes)")
