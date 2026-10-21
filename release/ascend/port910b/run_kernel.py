#!/usr/bin/env python3
"""ON-CARD runtime verdict: compile + EXECUTE the T.Cube gemm, numeric match vs
torch matmul, and tflops. Run inside bundle_ondemand.sh on the 910B instance.
Prints exactly one verdict line: RUNTIME-PASS {...json...} or RUNTIME-FAIL ..."""
import os, json, time
os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
import tilelang
import tilelang.language as T
from tilelang.ascend.language import Cube
import torch
import torch_npu  # noqa: F401  (device backend registration)


def gemm(M, N, K, blk, dtype, out_dtype):
    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype),
             C: T.Tensor((M, N), out_dtype)):
        with T.Kernel(T.ceildiv(N, blk), T.ceildiv(M, blk), threads=1) as (bx, by):
            A_s = T.alloc_shared((blk, K), "bfloat16")
            B_s = T.alloc_shared((K, blk), "bfloat16")
            C_l = T.alloc_shared((blk, blk), "float")
            with Cube():
                T.copy(A[by * blk, 0], A_s)
                T.copy(B[0, bx * blk], B_s)
                T.gemm(A_s, B_s, C_l, clear_accum=True)
                for i in T.serial(blk):
                    for j in T.serial(blk):
                        C[by * blk + i, bx * blk + j] = T.cast(C_l[i, j], out_dtype)
    return main


M = N = K = 2048
try:
    k = tilelang.compile(gemm(M, N, K, 128, "bfloat16", "bfloat16"),
                         target="ascend", out_idx=-1)
    dev = "npu"
    A = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    B = torch.randn(K, N, dtype=torch.bfloat16, device=dev)
    C = k(A, B)
    torch.npu.synchronize()
    ref = (A.float() @ B.float()).to(torch.bfloat16)
    rel = ((C.float() - ref.float()).norm() / (ref.float().norm() + 1e-9)).item()
    # timing
    for _ in range(3):
        k(A, B)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    iters = 20
    for _ in range(iters):
        k(A, B)
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
    ok = rel < 0.02
    print("RUNTIME-PASS " + json.dumps(
        {"shape": [M, N, K], "rel_err": round(rel, 5), "ms": round(ms, 3),
         "tflops": round(tflops, 2)})) if ok else print(
        f"RUNTIME-FAIL numeric rel_err={rel:.5f} (>0.02) ms={ms:.3f}")
except Exception as e:
    print("RUNTIME-FAIL", str(e)[:400])
