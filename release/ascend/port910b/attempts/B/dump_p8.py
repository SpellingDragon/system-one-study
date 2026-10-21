#!/usr/bin/env python3
"""attempts/B/dump_p8.py — 混合位宽（bf16/fp16）面逐个形态取证

起因：probe_910b_faces.py 的 P8（bf16 入 + fp32 入 → bf16 出 + fp32 出）报
  `bisheng: error: clang frontend command failed with exit code 70`（前端崩溃，非普通诊断），
而 add_ln 的真实口径正是 X(bf16/fp16) + Res(fp32) → Y(降位宽) + H(fp32 残差出口)，
必须弄清崩在哪一环：bf16 store？fp16 store？bf16 T.copy？还是 fp32+bf16 双出口混排？
每个 case 独立 try/except，FAIL 时打印 **完整** 诊断尾部（不只首 error 行）。
"""
import os
import sys

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
import tilelang
import tilelang.ascend.language as T

N, DIM, BM = 256, 128, 4


def c_b16_in_f32_out():
    @T.prim_func
    def main(A: T.Tensor((N, DIM), "bfloat16"), C: T.Tensor((N, DIM), "float32")):
        with T.Kernel(T.ceildiv(N, BM)) as bx:
            a = T.alloc_shared((BM, DIM), "bfloat16")
            T.copy(A[bx * BM, 0], a)
            for i in T.serial(BM):
                for j in T.serial(DIM):
                    C[bx * BM + i, j] = T.cast(a[i, j], "float32")
    return main


def c_b16_in_b16_out():
    @T.prim_func
    def main(A: T.Tensor((N, DIM), "bfloat16"), C: T.Tensor((N, DIM), "bfloat16")):
        with T.Kernel(T.ceildiv(N, BM)) as bx:
            a = T.alloc_shared((BM, DIM), "bfloat16")
            T.copy(A[bx * BM, 0], a)
            for i in T.serial(BM):
                for j in T.serial(DIM):
                    C[bx * BM + i, j] = T.cast(a[i, j] * 2.0, "bfloat16")
    return main


def c_f16_in_f16_out():
    @T.prim_func
    def main(A: T.Tensor((N, DIM), "float16"), C: T.Tensor((N, DIM), "float16")):
        with T.Kernel(T.ceildiv(N, BM)) as bx:
            a = T.alloc_shared((BM, DIM), "float16")
            T.copy(A[bx * BM, 0], a)
            for i in T.serial(BM):
                for j in T.serial(DIM):
                    C[bx * BM + i, j] = T.cast(a[i, j] * 2.0, "float16")
    return main


def c_f32acc_b16_and_f32_out():
    """add_ln 的真实形状：bf16 X + fp32 Res -> fp32 h；再写 bf16 Y 与 fp32 Hout"""
    @T.prim_func
    def main(A: T.Tensor((N, DIM), "bfloat16"), R: T.Tensor((N, DIM), "float32"),
             Y: T.Tensor((N, DIM), "bfloat16"), H: T.Tensor((N, DIM), "float32")):
        with T.Kernel(T.ceildiv(N, BM)) as bx:
            a = T.alloc_shared((BM, DIM), "bfloat16")
            r = T.alloc_shared((BM, DIM), "float32")
            T.copy(A[bx * BM, 0], a)
            T.copy(R[bx * BM, 0], r)
            for i in T.serial(BM):
                for j in T.serial(DIM):
                    h = T.alloc_var("float32")
                    h = T.cast(a[i, j], "float32") + r[i, j]
                    H[bx * BM + i, j] = h
                    Y[bx * BM + i, j] = T.cast(h, "bfloat16")
    return main


def c_f32acc_f16_and_f32_out():
    """同上但 Y 用 fp16"""
    @T.prim_func
    def main(A: T.Tensor((N, DIM), "float16"), R: T.Tensor((N, DIM), "float32"),
             Y: T.Tensor((N, DIM), "float16"), H: T.Tensor((N, DIM), "float32")):
        with T.Kernel(T.ceildiv(N, BM)) as bx:
            a = T.alloc_shared((BM, DIM), "float16")
            r = T.alloc_shared((BM, DIM), "float32")
            T.copy(A[bx * BM, 0], a)
            T.copy(R[bx * BM, 0], r)
            for i in T.serial(BM):
                for j in T.serial(DIM):
                    h = T.alloc_var("float32")
                    h = T.cast(a[i, j], "float32") + r[i, j]
                    H[bx * BM + i, j] = h
                    Y[bx * BM + i, j] = T.cast(h, "float16")
    return main


def c_ub_b16_copy_out():
    """bf16 结果落 UB 再 T.copy ub->gm（P2 路径的 bf16 版）"""
    @T.prim_func
    def main(A: T.Tensor((N, DIM), "bfloat16"), C: T.Tensor((N, DIM), "bfloat16")):
        with T.Kernel(T.ceildiv(N, BM)) as bx:
            a = T.alloc_shared((BM, DIM), "bfloat16")
            c = T.alloc_shared((BM, DIM), "bfloat16")
            T.copy(A[bx * BM, 0], a)
            for i in T.serial(BM):
                for j in T.serial(DIM):
                    c[i, j] = T.cast(a[i, j] * 2.0, "bfloat16")
            T.copy(c, C[bx * BM, 0])
    return main


CASES = [(n, f) for n, f in [
    ("bf16_in_f32_out", c_b16_in_f32_out),
    ("bf16_in_b16_out", c_b16_in_b16_out),
    ("fp16_in_fp16_out", c_f16_in_f16_out),
    ("f32acc_bf16_out_2outlets", c_f32acc_b16_and_f32_out),
    ("f32acc_fp16_out_2outlets", c_f32acc_f16_and_f32_out),
    ("bf16_ub_copy_out", c_ub_b16_copy_out),
]]

sel = sys.argv[1] if len(sys.argv) > 1 else None
print("=== DUMP-P8 ===")
for name, fn in CASES:
    if sel and sel not in name:
        continue
    try:
        k = tilelang.compile(fn(), target="ascend", out_idx=-1)
        src = k.get_kernel_source()
        print(f"{name}: PASS | len={len(src)}")
    except Exception as e:  # noqa: BLE001
        tail = [ln.strip() for ln in str(e).splitlines() if ln.strip()]
        errs = [ln for ln in tail if ": error:" in ln or "Aborted" in ln or "crash" in ln.lower()]
        print(f"{name}: FAIL | nlines={len(tail)}")
        for ln in (errs[:3] or tail[-3:]):
            print(f"    ! {ln[:300]}")
print("DUMP-P8-DONE")
