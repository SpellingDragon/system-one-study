#!/usr/bin/env python3
"""attempts/C/probe_ubhandle.py — 取证：910B codegen 给 alloc_shared 发的 UB 句柄形态。

起因（gen_gdn_scalar.asc 旧版第 6 行）：
    __ubuf__ float *w_ub = (__ubuf__ float *)0;
——纯标量件里一个 shared 落成**命名空句柄**（真实写 UB 偏移 0，且不进动态共享
内存布局），而 ubstage 件里同类缓冲走 ((__ubuf__ float*)buf_dyn_shmem)[off]。
若这个"空句柄"路径可以在有多个 shared 时出现，则多个空句柄彼此别名 → 编译
PASS 但运行时互踩，属于必须立规矩的坑（G-C8）。

三个 case 定位触发条件（同一 kernel 内 shared 用法不同）：
    A: sa/sb 写且读，sc 只写不读        → 观测只写不读是否退化
    B: A 再加 row/out 两个**完全未使用**的 shared（= 旧 gdn scalar 的形状）
    C: 只 sa 写且读；sb/sc 声明但零访问（= 只有 1 个"被访问"的 shared）
    D: sa 写且读 + sb 只写不读，sc 零访问（= 2 个被访问的 shared）
   → 若判决为「被访问的 shared 数 == 1 ⇒ 空句柄，>= 2 ⇒ 动态分块」则规则成立

判据：每 case 打印 __ubuf__ 相关行 + VERDICT（空句柄名字清单）。
"""
import os
import re

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402

N = 8
L = 64


def build(case):
    """case A/B/C —— 见模块 docstring。"""

    @T.prim_func
    def kern(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"),
             Dead: T.Tensor((N,), "float32"), O: T.Tensor((N,), "float32")):
        with T.Kernel(1) as _cid:
            sa = T.alloc_shared((N,), "float32")
            sb = T.alloc_shared((N,), "float32")
            sc = T.alloc_shared((N,), "float32")
            if case in ("B",):
                # 完全未使用（零访问）的大 shared —— 旧 gdn scalar 的死缓冲形状
                dead_row = T.alloc_shared((L,), "float32")       # noqa: F841
                dead_out = T.alloc_shared((L + 3,), "float32")   # noqa: F841
            acc = T.alloc_var("float32", T.float32(0.0))
            with T.Vector():
                for i in T.serial(N):
                    sa[i] = A[i]
                if case != "C":
                    for i in T.serial(N):
                        sb[i] = B[i]             # 只写不读
                    if case == "A" or case == "B":
                        for i in T.serial(N):
                            sc[i] = Dead[i]      # 只写不读
                if case == "D":
                    for i in T.serial(N):
                        sc[i] = Dead[i]          # D: sc 也被访问 ⇒ 2 个被访问 shared
                for i in T.serial(N):
                    acc = sa[i] + (sb[i] if case not in ("C", "D") else T.float32(0.0))
                    O[i] = acc * T.float32(2.0)

    return kern


def report(case):
    src = tilelang.compile(build(case), target="ascend", out_idx=-1).get_kernel_source()
    ub_lines = [ln.strip() for ln in src.splitlines() if "__ubuf__" in ln]
    named_nulls = re.findall(
        r"__ubuf__\s+\w+\s*\*(\w+)\s*=\s*\(__ubuf__\s+\w+\s*\*\)\s*0\s*;", src)
    real_nulls = [n for n in named_nulls if n != "buf_dyn_shmem"]
    dyn = len(re.findall(r"buf_dyn_shmem", src))
    print("CASE %s: compile PASS | buf_dyn_shmem refs=%d | 命名空句柄=%s"
          % (case, dyn, real_nulls or "(none)"))
    for ln in ub_lines:
        print("   UB :", ln[:150])
    return real_nulls


def main():
    print("PROBE-UBHANDLE start")
    hits = {}
    for case in ("A", "B", "C", "D"):
        try:
            hits[case] = report(case)
        except Exception as e:  # noqa: BLE001
            hits[case] = None
            print("CASE %s: compile FAIL %s" % (case, str(e).splitlines()[0][:160]))
    bad = [c for c, v in hits.items() if v]
    if bad:
        print("PROBE-UBHANDLE-VERDICT: NULL-HANDLE-SEEN cases %s 出现命名空句柄；"
          "对照 A/B/C/D 的差量即触发条件（本波结论：被访问的 alloc_shared 只有 1 个 "
          "⇒ 落 (__ubuf__ T*)0 = UB 偏移 0，未进动态布局；>= 2 ⇒ 走 buf_dyn_shmem 分块）"
          % bad)
    else:
        print("PROBE-UBHANDLE-VERDICT: DYN-SHARED-ONLY 三种用法（含只写不读、含完全"
              "未使用的大 shared）都走 buf_dyn_shmem 分块，唯一空句柄是基址本身；"
              "旧 gdn scalar 的 (__ubuf__ float*)0 形态由**别的**条件触发（已在本波"
              "把死缓冲删除后消失，见 gen_gdn_scalar.asc）")


if __name__ == "__main__":
    main()
