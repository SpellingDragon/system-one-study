"""临时探查 #7：ascend 后端本地编译能力（哪些构件能 lower 出源码，哪些不能）。argv 选做，一案例一进程。"""
import sys

import tilelang
import tilelang.ascend.language as T


def show(tag, program, out_idx):
    kernel = tilelang.compile(program, out_idx=out_idx, target="ascend")
    src = str(kernel.get_kernel_source())
    marks = [m for m in ("AscendC", "kernel_c", "__aicore__", "T.gemm", "asc_reduce", "Pipe", "SetAtomic") if m in src]
    return f"{tag}: src_len={len(src)} marks={marks} head={src[:80]!r}"


def gemm_nt(M=128, N=128, K=128, bm=64, bn=64, bk=64):
    @T.prim_func
    def main(A: T.Tensor((M, K), "bfloat16"), W: T.Tensor((N, K), "bfloat16"), C: T.Tensor((M, N), "float32")):
        with T.Kernel((N // bn) * (M // bm)) as bx:
            n_tiles = M // bm
            m_tile = bx // n_tiles
            n_tile = bx % n_tiles
            a_l1 = T.alloc_l1((bm, bk), "bfloat16")
            w_l1 = T.alloc_l1((bn, bk), "bfloat16")
            c_l0c = T.alloc_l0c((bm, bn), "float32")
            ub = T.alloc_shared((bm, bn), "float32")
            T.clear(c_l0c)
            for kt in T.Pipelined(K // bk, num_stages=2):
                T.copy(A[m_tile * bm:(m_tile + 1) * bm, kt * bk:(kt + 1) * bk], a_l1)
                T.copy(W[n_tile * bn:(n_tile + 1) * bn, kt * bk:(kt + 1) * bk], w_l1)
                T.gemm(a_l1, w_l1, c_l0c, transpose_B=True, clear_accum=(kt == 0))
            T.copy(c_l0c, ub)
            T.copy(ub, C[m_tile * bm, n_tile * bn])
    return main


def vecadd_simt(n=2048, blocks=8, threads=128):
    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(blocks) as bx:
            tile = n // blocks
            a = T.alloc_shared((tile,), "float32")
            b = T.alloc_shared((tile,), "float32")
            c = T.alloc_shared((tile,), "float32")
            begin = bx * tile
            T.copy(A[begin:begin + tile], a)
            T.copy(B[begin:begin + tile], b)
            with T.SimtVF(threads=threads):
                for i in T.Parallel(tile):
                    c[i] = a[i] + b[i]
            T.copy(c, C[begin:begin + tile])
    return main


def vecadd_simd(n=2048, blocks=8):
    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(blocks) as bx:
            tile = n // blocks
            a = T.alloc_shared((tile,), "float32")
            b = T.alloc_shared((tile,), "float32")
            c = T.alloc_shared((tile,), "float32")
            begin = bx * tile
            T.copy(A[begin:begin + tile], a)
            T.copy(B[begin:begin + tile], b)
            with T.SimdVF():
                mask = T.simd.pset(32)
                for i in range(tile // 64):
                    r0 = T.simd.vld(a[i * 64])
                    r1 = T.simd.vld(b[i * 64])
                    T.simd.vsts(c[i * 64], T.simd.vadd(r0, r1, mask), mask)
            T.copy(c, C[begin:begin + tile])
    return main


def rms_reducer(rows=8, d=64, threads=64):
    @T.prim_func
    def main(X: T.Tensor((rows, d), "float32"), Y: T.Tensor((rows, d), "float32"), eps: T.float32):
        with T.Kernel(1) as bx:
            x_ub = T.alloc_shared((rows, d), "float32")
            y_ub = T.alloc_shared((rows, d), "float32")
            T.copy(X[0:rows, 0:d], x_ub)
            with T.SimtVF(threads=threads):
                xf = T.alloc_fragment((rows, d), "float32")
                for r, i in T.Parallel(rows, d):
                    xf[r, i] = x_ub[r, i]
                acc = T.alloc_reducer((rows,), "float32", op="sum")
                T.reducer_init(acc)
                for r, i in T.Parallel(rows, d):
                    T.reducer_update(acc[r], xf[r, i] * xf[r, i])
                rs = T.alloc_fragment((rows,), "float32")
                T.finalize_reducer(acc, rs)
                for r, i in T.Parallel(rows, d):
                    y_ub[r, i] = xf[r, i] * T.rsqrt(rs[r] / d + eps)
            T.copy(y_ub, Y[0:rows, 0:d])
    return main


def reduce_op_rows(rows=8, d=64):
    """T.reduce_sum 直接对 UB 做行归约（不经过 reducer）。"""
    @T.prim_func
    def main(X: T.Tensor((rows, d), "float32"), S: T.Tensor((rows, 1), "float32")):
        with T.Kernel(1) as bx:
            x_ub = T.alloc_shared((rows, d), "float32")
            s_ub = T.alloc_shared((rows, 1), "float32")
            T.copy(X[0:rows, 0:d], x_ub)
            T.reduce_sum(x_ub, s_ub, dim=1)
            T.copy(s_ub, S[0:rows, 0:1])
    return main


def stage_manual(rows=8, d=64):
    @T.prim_func
    def main(X: T.Tensor((rows, d), "float32"), Y: T.Tensor((rows, d), "float32")):
        with T.Kernel(1) as bx:
            x_ub = T.alloc_shared((rows, d), "float32")
            y_ub = T.alloc_shared((rows, d), "float32")
            with T.Stage(0):
                T.copy(X[0:rows, 0:d], x_ub)
            with T.Stage(1):
                with T.SimtVF(threads=64):
                    for i, j in T.Parallel(rows, d):
                        y_ub[i, j] = x_ub[i, j] * 2
            T.copy(y_ub, Y[0:rows, 0:d])
    return main


CASES = {
    "gemm": lambda: show("gemm_nt", gemm_nt(), [2]),
    "vecadd_simt": lambda: show("vecadd_simt", vecadd_simt(), [2]),
    "vecadd_simd": lambda: show("vecadd_simd", vecadd_simd(), [2]),
    "rms_reducer": lambda: show("rms_reducer", rms_reducer(), None),
    "reduce_op": lambda: show("reduce_sum_ub", reduce_op_rows(), None),
    "stage": lambda: show("stage_manual", stage_manual(), None),
}

for name in (sys.argv[1:] or sorted(CASES)):
    try:
        print(f"[asc/{name}] -> {CASES[name]()}", flush=True)
    except Exception as e:
        first = [ln for ln in str(e).splitlines() if ln.strip()]
        print(f"[asc/{name}] FAIL {type(e).__name__}: {first[0][:300] if first else e}", flush=True)
