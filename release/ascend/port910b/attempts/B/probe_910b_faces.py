#!/usr/bin/env python3
"""attempts/B/probe_910b_faces.py — 910B 非 Cube（vector/scalar）面构件可用性探针

目的：写 add_ln / readout 前，逐个确认哪些 DSL 构件能在 dav-2201 + port910b_compat(GAP-B) 下
编译通过（E2E-COMPILE 级）。每项独立 try/except，输出 <name>: PASS/FAIL + 首错误行。

方言要点（tilelang/ascend/language/kernel.py:139）：Ascend 的 T.Kernel 只吃 1-D core
grid、没有 threads 形参；`with T.Kernel(N) as bx`。故这里同时探两种发射形态。
错误提取只用 `": error:"`（早期版本按 "error" 子串匹配会抓到 toolchain 的
`kernel_log.h ... warning: inline function ... is not defined`，把判决掩盖成假错误）。
"""
import os

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
import tilelang
import tilelang.ascend.language as T
import tilelang.language as TC  # common 方言（带 threads=），对照用

RESULTS = []


def verdict(name, fn):
    try:
        k = tilelang.compile(fn(), target="ascend", out_idx=-1)
        src = k.get_kernel_source()
        syms = sorted({w for w in ("asc_copy_gm2ub_align", "asc_copy_ub2gm_align",
                                   "asc_sync_notify", "asc_sync_wait", "rsqrtf", "expf")
                       if w in src})
        RESULTS.append((name, "PASS", f"len={len(src)} syms={syms}"))
    except Exception as e:  # noqa: BLE001
        s = str(e)
        errs = [ln.strip() for ln in s.splitlines() if ": error:" in ln]
        if errs:
            RESULTS.append((name, "FAIL", errs[0][:220]))
        else:
            lines = [x.strip() for x in s.splitlines() if x.strip()]
            kind = "DIALECT-REJECT" if lines and "error" not in lines[0].lower() else "FAIL"
            RESULTS.append((name, kind, (lines[0] if lines else repr(e))[:220]))


N, DIM, BM = 1024, 256, 4
BT, TT, KT = 8, 16, 32


# ── P0a: ascend 方言标量核（T.Kernel(N) as bx）+ 纯标量 store 直写 GM
def p0a():
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


# ── P0b: common 方言标量核（threads=1，与 e2e_cube 同发射形态）
def p0b():
    @TC.prim_func
    def main(A: TC.Tensor((N, DIM), "float32"), B: TC.Tensor((N, DIM), "float32"),
             C: TC.Tensor((N, DIM), "float32")):
        with TC.Kernel(TC.ceildiv(N, BM), 1, threads=1) as (bx, _by):
            a = TC.alloc_shared((BM, DIM), "float32")
            b = TC.alloc_shared((BM, DIM), "float32")
            TC.copy(A[bx * BM, 0], a)
            TC.copy(B[bx * BM, 0], b)
            for i in TC.serial(BM):
                for j in TC.serial(DIM):
                    C[bx * BM + i, j] = a[i, j] + b[i, j]
    return main


# ── P2: 结果落 UB 后 T.copy ub->gm（asc_copy_ub2gm_align 路径）
def p2():
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


# ── P3: alloc_var 标量累加 + T.rsqrt（行内归约口径，add_ln 核心）
def p3():
    @T.prim_func
    def main(A: T.Tensor((N, DIM), "float32"), B: T.Tensor((N, DIM), "float32"),
             C: T.Tensor((N, DIM), "float32")):
        with T.Kernel(T.ceildiv(N, BM)) as bx:
            a = T.alloc_shared((BM, DIM), "float32")
            T.copy(A[bx * BM, 0], a)
            for i in T.serial(BM):
                acc = T.alloc_var("float32")
                mu = T.alloc_var("float32")
                rs = T.alloc_var("float32")
                acc = 0.0
                for j in T.serial(DIM):
                    acc = acc + a[i, j]
                mu = acc / DIM
                acc = 0.0
                for j in T.serial(DIM):
                    dv = a[i, j] - mu
                    acc = acc + dv * dv
                rs = T.rsqrt(acc / DIM + 1e-5)
                for j in T.serial(DIM):
                    B[bx * BM + i, j] = mu * rs
                    C[bx * BM + i, j] = (a[i, j] - mu) * rs
    return main


# ── P4: 一维 tensor 的 T.copy（GM 行向量 -> UB，add_ln 的 G/Bt）
def p4():
    @T.prim_func
    def main(A: T.Tensor((N, DIM), "float32"), G: T.Tensor((DIM,), "float32"),
             C: T.Tensor((N, DIM), "float32")):
        with T.Kernel(T.ceildiv(N, BM)) as bx:
            a = T.alloc_shared((BM, DIM), "float32")
            g = T.alloc_shared((DIM,), "float32")
            T.copy(A[bx * BM, 0], a)
            T.copy(G[0:DIM], g)
            for i in T.serial(BM):
                for j in T.serial(DIM):
                    C[bx * BM + i, j] = a[i, j] * g[j]
    return main


# ── P5: 动态下标 gather（3D GM 读，行号来自 UB 的 index 数组）——readout 取末位
def p5():
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


# ── P6: 标量 math intrin（exp/max）——readout softmax 侧路
def p6():
    @T.prim_func
    def main(S: T.Tensor((BT, KT), "float32"), W: T.Tensor((KT, DIM), "float32"),
             C: T.Tensor((BT, KT), "float32")):
        with T.Kernel(1) as bx:
            s = T.alloc_shared((BT, KT), "float32")
            w = T.alloc_shared((KT, DIM), "float32")
            T.copy(S[0, 0], s)
            T.copy(W[0, 0], w)
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
                    C[i, k] = T.exp(s[i, k] - mx) / sm
    return main


# ── P7: T.gemm 在 Cube() 之外（readout 若走矩阵乘投影）
def p7():
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


# ── P8: bf16 混合位宽（add_ln 的 X fp16 / Res fp32 / 输出 bf16 口径）
def p8():
    @T.prim_func
    def main(A: T.Tensor((N, DIM), "bfloat16"), B: T.Tensor((N, DIM), "float32"),
             C: T.Tensor((N, DIM), "bfloat16"), D: T.Tensor((N, DIM), "float32")):
        with T.Kernel(T.ceildiv(N, BM)) as bx:
            a = T.alloc_shared((BM, DIM), "bfloat16")
            b = T.alloc_shared((BM, DIM), "float32")
            T.copy(A[bx * BM, 0], a)
            T.copy(B[bx * BM, 0], b)
            for i in T.serial(BM):
                for j in T.serial(DIM):
                    C[bx * BM + i, j] = T.cast(T.cast(a[i, j], "float32") + b[i, j], "bfloat16")
                    D[bx * BM + i, j] = T.cast(a[i, j], "float32") + b[i, j]
    return main


# ── P9: T.Parallel（已知坑：Ascend 方言里必须在 VF 块内）——负面确认
def p9():
    @T.prim_func
    def main(A: T.Tensor((N, DIM), "float32"), C: T.Tensor((N, DIM), "float32")):
        with T.Kernel(T.ceildiv(N, BM)) as bx:
            a = T.alloc_shared((BM, DIM), "float32")
            T.copy(A[bx * BM, 0], a)
            for i in T.Parallel(BM):
                for j in T.Parallel(DIM):
                    C[bx * BM + i, j] = a[i, j] * 2.0
    return main


# ── P10: 双输出 fp32+bf16（add_ln 的 H 残差出口 + Y 降位宽出口）
def p10():
    @T.prim_func
    def main(A: T.Tensor((N, DIM), "float32"), G: T.Tensor((DIM,), "float32"),
             Y: T.Tensor((N, DIM), "float16"), H: T.Tensor((N, DIM), "float32")):
        with T.Kernel(T.ceildiv(N, BM)) as bx:
            a = T.alloc_shared((BM, DIM), "float32")
            g = T.alloc_shared((DIM,), "float32")
            T.copy(A[bx * BM, 0], a)
            T.copy(G[0:DIM], g)
            for i in T.serial(BM):
                acc = T.alloc_var("float32")
                rs = T.alloc_var("float32")
                acc = 0.0
                for j in T.serial(DIM):
                    acc = acc + a[i, j] * a[i, j]
                rs = T.rsqrt(acc / DIM + 1e-5)
                for j in T.serial(DIM):
                    H[bx * BM + i, j] = a[i, j]
                    Y[bx * BM + i, j] = T.cast(a[i, j] * rs * g[j], "float16")
    return main


for nm, f in [("P0a_ascend_kernel_scalar_store", p0a), ("P0b_common_kernel_threads1", p0b),
              ("P2_copy_ub2gm", p2), ("P3_alloc_var_rsqrt", p3), ("P4_1d_copy", p4),
              ("P5_dyn_idx_gather", p5), ("P6_exp_max_softmax", p6), ("P7_gemm_no_cube", p7),
              ("P8_mixed_dtype_cast", p8), ("P9_parallel_negative", p9),
              ("P10_two_outlet_rsqrt", p10)]:
    verdict(nm, f)

print("=== PROBE-910B-FACES ===")
for nm, st, info in RESULTS:
    print(f"{nm}: {st} | {info}")
npass = sum(1 for _, st, _ in RESULTS if st == "PASS")
print(f"PROBE-DONE {npass}/{len(RESULTS)} PASS")
