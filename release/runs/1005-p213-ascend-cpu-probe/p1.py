import tilelang, torch
import tilelang.cpu.language as T
def k(A, B, C, n, bk):
    m = T.dynamic("m")
    @T.prim_func
    def f(A: T.Tensor((m, n), "float32"), B: T.Tensor((n, n), "float32"), C: T.Tensor((m, n), "float32")):
        with T.Kernel(T.ceildiv(m, 16)) as bx:
            As = T.alloc_local((16, bk), "float32")
            Bs = T.alloc_local((n, bk), "float32")
            Cs = T.alloc_local((16, n), "float32")
                T.clear(Cs)
                T.copy(B[0, 0], Bs)
                for ko in T.serial(n // bk):
                    T.copy(A[bx * 16, ko * bk], As)
                    T.gemm(As, Bs, Cs, transpose_B=True)
                T.copy(Cs, C[bx * 16, 0])
    return f
ker = tilelang.jit(out_idx=[])(k)
A, B, C = torch.rand(8, 32), torch.rand(16, 32), torch.zeros(8, 16)
ker(A, B, C, n=16, bk=32)
print("PROBE1 ERR", float((C - A @ B.T).abs().max()))
