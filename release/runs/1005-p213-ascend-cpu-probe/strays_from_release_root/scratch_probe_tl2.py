"""临时探查 #8：cpu(c target) 可靠构件（serial 标量/守卫/gemm epilogue/就地/双 gemm/gather）。"""
import sys

import torch
import tilelang
import tilelang.cpu.language as T

CFGP = dict(target="c", target_host="c", execution_backend="cython")


def cmp_(prog, out):
    return tilelang.compile(prog, out_idx=out, **CFGP)


def go(name, fn):
    try:
        print(f"[{name}] -> {fn()}", flush=True)
    except Exception as e:
        first = [ln.strip() for ln in str(e).splitlines() if ln.strip()]
        print(f"[{name}] FAIL {type(e).__name__}: {first[0][:240] if first else e}", flush=True)


def serial_math():
    @T.prim_func
    def main(A: T.Tensor((4, 8), "float32"), C: T.Tensor((4, 8), "float32")):
        with T.Kernel(1) as bx:
            al = T.alloc_local((4, 8), "float32")
            mx = T.alloc_local((4,), "float32")
            sm = T.alloc_local((4,), "float32")
            T.copy(A[0, 0], al)
            for i in T.serial(4):
                mx[i] = al[i, 0]
                sm[i] = T.float32(0)
            for i, j in T.serial(4, 8):
                mx[i] = T.max(mx[i], al[i, j])
                al[i, j] = T.exp(al[i, j] - mx[i])
            for i, j in T.serial(4, 8):
                al[i, j] = al[i, j] / sm[i]
            T.copy(al, C[0, 0])
    k = cmp_(main, [1])
    a = torch.randn(4, 8)
    return "exp_div_err=%.2e" % float((k(a) - torch.softmax(a, -1)).abs().max())


def tail_guard():
    m = T.dynamic("m")

    @T.prim_func
    def main(A: T.Tensor((m, 8), "float32"), C: T.Tensor((m, 8), "float32")):
        with T.Kernel(T.ceildiv(m, 4)) as bx:
            al = T.alloc_local((4, 8), "float32")
            T.copy(A[bx * 4, 0], al)
            for i, j in T.serial(4, 8):
                if bx * 4 + i < m:
                    C[bx * 4 + i, j] = al[i, j] * 2 + 1
    k = cmp_(main, [1])
    errs = []
    for rows in (4, 7, 13):
        a = torch.rand(rows, 8)
        errs.append(round(float((k(a) - (a * 2 + 1)).abs().max()), 7))
    return "tail_errs=%s" % errs


def gemm_epilogue():
    M, N, K = 8, 8, 8

    @T.prim_func
    def main(A: T.Tensor((M, K), "float32"), W: T.Tensor((N, K), "float32"), B: T.Tensor((N,), "float32"),
             Y: T.Tensor((M, N), "float16"), Z: T.Tensor((M, N), "float32")):
        with T.Kernel(1) as bx:
            al = T.alloc_local((M, K), "float32")
            wl = T.alloc_local((N, K), "float32")
            bl = T.alloc_local((N,), "float32")
            cl = T.alloc_local((M, N), "float32")
            T.copy(A[0, 0], al)
            T.copy(W[0, 0], wl)
            T.copy(B[0], bl)
            T.clear(cl)
            T.gemm(al, wl, cl, transpose_B=True)
            for i, j in T.serial(M, N):
                cl[i, j] = T.max(cl[i, j] + bl[j], T.float32(0))
            T.copy(cl, Z[0, 0])
            T.copy(cl, Y[0, 0])
    k = cmp_(main, [3, 4])
    a, w, b = torch.rand(8, 8), torch.rand(8, 8), torch.rand(8)
    ref = (a @ w.T + b).clamp(min=0)
    y, z = k(a, w, b)
    return "z=%.2e y=%.2e ydtype=%s" % (float((z - ref).abs().max()), float((y.float() - ref).abs().max()), y.dtype)


CASES = {"serial_math": serial_math, "tail_guard": tail_guard, "gemm_epilogue": gemm_epilogue}

for nm in (sys.argv[1:] or sorted(CASES)):
    go(nm, CASES[nm])
