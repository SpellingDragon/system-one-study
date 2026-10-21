import tilelang, torch
import tilelang.cpu.language as T
def k(A, B, C, nn, kk, bk):
    m = T.dynamic("m")
    @T.prim_func
    def f(A: T.Tensor((m, kk), "float32"), B: T.Tensor((nn, kk), "float32"), C: T.Tensor((m, nn), "float32")):
        with T.Kernel(T.ceildiv(m, 16)) as bx:
            As = T.alloc_local((16, bk), "float32")
