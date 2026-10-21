#!/usr/bin/env python3
"""attempts/B/dump_bf16.py — bf16 面前端崩溃（exit code 70）的完整诊断取证

打印：① 生成码全文 ② tilelang 抛出的**完整**编译日志（不截断），用于判定这是
compat/类型问题还是 bisheng 对 bf16 标量面支持的硬缺陷（fp16 同形已 PASS，见 dump_p8.py）。
"""
import os

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
from tilelang.contrib import bisheng as bs

OUT = "/tmp/agentB/gen/gen_bf16.asc"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
_orig = bs.compile_ascend


def spy(code, *a, **kw):
    with open(OUT, "w") as f:
        f.write(code)
    return _orig(code, *a, **kw)


bs.compile_ascend = spy

import tilelang
import tilelang.ascend.language as T

N, DIM, BM = 64, 32, 2


@T.prim_func
def main(A: T.Tensor((N, DIM), "bfloat16"), C: T.Tensor((N, DIM), "float32")):
    with T.Kernel(T.ceildiv(N, BM)) as bx:
        a = T.alloc_shared((BM, DIM), "bfloat16")
        T.copy(A[bx * BM, 0], a)
        for i in T.serial(BM):
            for j in T.serial(DIM):
                C[bx * BM + i, j] = T.cast(a[i, j], "float32")


print("=== GEN CODE ===")
try:
    tilelang.compile(main, target="ascend", out_idx=-1)
    print("BF16-COMPILE-PASS")
except Exception as e:  # noqa: BLE001
    print("BF16-COMPILE-FAIL, full log below (last 40 lines):")
    lines = [ln for ln in str(e).splitlines() if ln.strip()]
    for ln in lines[-40:]:
        print("   ", ln.strip()[:240])
if os.path.exists(OUT):
    print("=== GENERATED .asc ===")
    for i, ln in enumerate(open(OUT).read().splitlines(), 1):
        print(f"  {i:4d}: {ln[:200]}")
