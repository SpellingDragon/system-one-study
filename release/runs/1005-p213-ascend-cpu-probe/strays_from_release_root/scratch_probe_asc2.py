"""临时探查 #5b：ascend 编译返回对象到底是什么，能否本机拿到方言源码/强制 lowering。"""
import tilelang
import tilelang.ascend.language as T


@T.prim_func
def trivial(A: T.Tensor((64,), "float32"), C: T.Tensor((64,), "float32")):
    with T.Kernel(1) as bx:
        a = T.alloc_shared((64,), "float32")
        c = T.alloc_shared((64,), "float32")
        T.copy(A[0:64], a)
        with T.SimtVF(threads=64):
            for i in T.Parallel(64):
                c[i] = a[i] * 2
        T.copy(c, C[0:64])


k = tilelang.compile(trivial, target="ascend", out_idx=-1)
print("type:", type(k), "mro:", [x.__name__ for x in type(k).__mro__][:6])
print("attrs:", sorted(n for n in dir(k) if not n.startswith("__"))[:40])
for n in ("kernel", "_kernel", "artifact", "adapter", "func", "libpath", "target", "source"):
    v = getattr(k, n, None)
    if v is not None:
        print(f"  .{n} -> {type(v).__name__}")
inner = getattr(k, "kernel", None) or getattr(k, "_kernel", None)
if inner is not None:
    print("inner type:", type(inner))
    print("inner attrs:", sorted(n for n in dir(inner) if not n.startswith("__"))[:40])
    try:
        src = inner.get_kernel_source()
        print("INNER SOURCE len:", len(src))
        print(src[:600])
    except Exception as e:
        print("inner get_kernel_source FAIL:", type(e).__name__, str(e).splitlines()[0][:300])
