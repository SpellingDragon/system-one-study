#!/usr/bin/env python3
"""attempts/D2_probe_matrix.py — p2-13/P1-1n（D2）：§12 三变体的**离线判别力矩阵**（不需卡）。

产物三张判决表 + 一张覆盖关系小结，全部可复跑：
  ① 默认路径**逐字节等价**：同一份 TU 源码、同一个落盘路径，只换头树
     （ref=D2 改头前 pristine f56432b0  vs  trunk=D2 落地后 d0547c0b，两者都不加 -D）
     → 整只 `.o` 的 sha256 **必须全等**（不是"宏隔离所以安全"的口头断言）。
  ② 变体确实改变生成码：trunk 头树 default vs -D A/B/C…  逐 TU 比 .o sha256 → DIFF/SAME。
     SAME 不是 bug，而是**覆盖关系证据**（该 TU 不吃这组开关）。
  ③ 每组的编译判决行（rc=0 给 .o 大小+sha；rc!=0 给原文错误首行）。

TU 口径：`L1/L2` 是**真生成码原文逐字照抄**（从 A2 基线缓存
`/tmp/d2_cache_base/.../device_kernel.asc` 取，含它自己的 3 条 #include），
`M*` 是为"定位到底是哪一件变了"而配的最小单件 TU。全部经真包含链、真件名真签名。

  L1  gemm_l1_intrinsic 真载荷（asc_copy_gm2l1_nd2nz ×1 + ascend_gemm_l1 模板）
  L2  dw_intrinsic 真载荷（nd2nz ×2 + l12l0a/b**_transpose** + mmad + l0c2gm）
  M1  gm2l1 bf16 单发          M2  gCol 越界兜底支      M3  l12l0a 拆趟支
  M4  l12l0a kStep=1 退化支    M5  l12l0b K-major       M6  dn2nz 委托
  M7  l12l0a_transpose 真形    ——  **应吃不到 B/C**（Z 未改该件）→ 覆盖关系实证

复跑（容器 cann910b-k，离线、无卡）：
  docker cp attempts/D2_probe_matrix.py cann910b-k:/tmp/
  docker exec cann910b-k bash -c 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \
      python3 /tmp/D2_probe_matrix.py'
参照树（pristine，建法见 attempts/D2_variants.md §5）：/tmp/d2_ref_src/src/{tl_templates,c_api}
"""
import argparse
import hashlib
import os
import subprocess
import sys

BISHENG = os.environ.get(
    "BISHENG", "/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/bin/bisheng")
ROOT = os.environ.get("ASCEND_CANN", "/usr/local/Ascend/cann-8.5.0/aarch64-linux")
STD = ["-std=c++20", "-fPIC", "-O2"]
ARCH = ["--npu-arch=dav-2201"]
INC_TL = "#include <tl_templates/ascend/common.h>\n"
MODE_L = "static_cast<asc_load_l2_cache_mode>(0)"

# ── 真生成码原文（照抄 A2 基线缓存的 device_kernel.asc）────────────────────────
L1_GEMM_L1 = """#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/gemm.h>
extern "C" __global__ __cube__ void main_kernel(__gm__ bfloat16_t* A, __gm__ bfloat16_t* B, __gm__ float* O) {
  asc_init();
  __cbuf__ uint8_t *buf_dyn_l1 = (__cbuf__ uint8_t *)0;
  __cc__ float *c_l0 = (__cc__ float *)0;
  asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(64), 0);
  asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1), (&(A[0])), 128, static_cast<asc_load_l2_cache_mode>(0), 64, 64, 0, 0);
  asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(64), 0);
  asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + 4096), (&(B[0])), 128, static_cast<asc_load_l2_cache_mode>(0), 64, 64, 0, 0);
  ascend_gemm_l1<64, 64, 64, 64, true, bfloat16_t>(((__cc__ float*)c_l0), ((__cbuf__ bfloat16_t*)buf_dyn_l1), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 4096), (bool)1, 0, 0);
  asc_set_l0c_copy_nz_para(1, 0, 0);
  asc_copy_l0c2gm((&(O[0])), ((__cc__ float*)c_l0), 64, 64, 64, 64, static_cast<asc_store_l2_cache_mode>(0), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
}
"""

L2_DW = """#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
extern "C" __global__ __cube__ void dW_l0tr_kernel(__gm__ float* DW, __gm__ half* X, __gm__ half* dY) {
  asc_init();
  __cbuf__ uint8_t *buf_dyn_l1 = (__cbuf__ uint8_t *)0;
  __ca__ half *a_l0 = (__ca__ half *)0;
  __cb__ half *b_l0 = (__cb__ half *)0;
  __cc__ float *acc = (__cc__ float *)0;
  for (int32_t kt = 0; kt < 32; ++kt) {
    asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(64), 0);
    asc_copy_gm2l1_nd2nz(((__cbuf__ half*)buf_dyn_l1), (&(dY[((kt * 16384) + ((((int32_t)block_idx) >> 1) * 128))])), 512, static_cast<asc_load_l2_cache_mode>(0), 64, 128, 0, 0);
    asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(64), 0);
    asc_copy_gm2l1_nd2nz(((__cbuf__ half*)buf_dyn_l1 + 8192), (&(X[((kt * 16384) + ((((int32_t)block_idx) & 1) * 128))])), 512, static_cast<asc_load_l2_cache_mode>(0), 64, 128, 0, 0);
    asc_copy_l12l0a_transpose(((__ca__ half*)a_l0), ((__cbuf__ half*)buf_dyn_l1), 0, 0, 4, 8, 4, 8);
    asc_copy_l12l0b_transpose(((__cb__ half*)b_l0), ((__cbuf__ half*)buf_dyn_l1 + 8192), 0, 0, 4, 8, 4, 8);
    asc_mmad(((__cc__ float*)acc), ((__ca__ half*)a_l0), ((__cb__ half*)b_l0), 128, 64, 128, 0, 1, 0, (kt == 0));
  }
  asc_set_l0c_copy_nz_para(1, 0, 0);
  asc_copy_l0c2gm((&(DW[(((((int32_t)block_idx) >> 1) * 32768) + ((((int32_t)block_idx) & 1) * 128))])), ((__cc__ float*)acc), 128, 128, 256, 128, static_cast<asc_store_l2_cache_mode>(0), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
}
"""

TUS = {
    "L1_real_gemm_l1_intrinsic": L1_GEMM_L1,
    "L2_real_dw_intrinsic": L2_DW,
    "M1_gm2l1_bf16": INC_TL + """
extern "C" __global__ __cube__ void probe(__gm__ bfloat16_t *g) {
  __cbuf__ bfloat16_t *c = (__cbuf__ bfloat16_t *)(uintptr_t)0x10000;
  asc_copy_gm2l1_nd2nz(c, g, 128, %s, 64, 64, 0, 0);
  (void)g; (void)c;
}
""" % MODE_L,
    "M2_gm2l1_gcol_fallback": INC_TL + """
extern "C" __global__ __cube__ void probe(__gm__ bfloat16_t *g) {
  __cbuf__ bfloat16_t *c = (__cbuf__ bfloat16_t *)(uintptr_t)0x10000;
  asc_copy_gm2l1_nd2nz(c, g, 131072, %s, 16, 32, 0, 0);  // gColW>=UINT16_MAX → 逐行拆趟支
  (void)g; (void)c;
}
""" % MODE_L,
    "M3_l12l0a_msplit": INC_TL + """
extern "C" __global__ __cube__ void probe(__gm__ bfloat16_t *g) {
  asc_copy_l12l0a((__ca__ bfloat16_t *)(uintptr_t)0x40000,
                  (__cbuf__ bfloat16_t *)(uintptr_t)0x10000, 1, 2, 2, 4, 8, 4);
  (void)g;
}
""",
    "M4_l12l0a_kstep1": INC_TL + """
extern "C" __global__ __cube__ void probe(__gm__ bfloat16_t *g) {
  asc_copy_l12l0a((__ca__ bfloat16_t *)(uintptr_t)0x40000,
                  (__cbuf__ bfloat16_t *)(uintptr_t)0x10000, 0, 0, 3, 1, 8, 1);
  (void)g;
}
""",
    "M5_l12l0b_kmajor": INC_TL + """
extern "C" __global__ __cube__ void probe(__gm__ bfloat16_t *g) {
  asc_copy_l12l0b((__cb__ bfloat16_t *)(uintptr_t)0x48000,
                  (__cbuf__ bfloat16_t *)(uintptr_t)0x10000, 1, 0, 1, 4, 8, 4);
  (void)g;
}
""",
    "M6_dn2nz_delegate": INC_TL + """
extern "C" __global__ __cube__ void probe(__gm__ bfloat16_t *g) {
  __cbuf__ bfloat16_t *c = (__cbuf__ bfloat16_t *)(uintptr_t)0x10000;
  asc_copy_gm2l1_dn2nz(c, g, 512, %s, 64, 128, 0, 0);
  (void)g; (void)c;
}
""" % MODE_L,
    "M7_l12l0a_transpose": INC_TL + """
extern "C" __global__ __cube__ void probe(__gm__ half *g) {
  // dw_intrinsic 生成码原文实参：(dst, src, 0, 0, 4, 8, 4, 8)
  asc_copy_l12l0a_transpose((__ca__ half *)(uintptr_t)0x40000,
                            (__cbuf__ half *)(uintptr_t)0x10000, 0, 0, 4, 8, 4, 8);
  (void)g;
}
""",
}

COMBOS = [
    ("default", []),
    ("-D A", ["-DTL910B_VARIANT_A"]),
    ("-D B", ["-DTL910B_VARIANT_B"]),
    ("-D B+SID16", ["-DTL910B_VARIANT_B", "-DTL910B_VARIANT_B_SID=16"]),
    ("-D C", ["-DTL910B_VARIANT_C"]),
    ("-D A+C", ["-DTL910B_VARIANT_A", "-DTL910B_VARIANT_C"]),
    ("-D A+B+C", ["-DTL910B_VARIANT_A", "-DTL910B_VARIANT_B", "-DTL910B_VARIANT_C"]),
]


def compile_tu(tu, body, src_root, extra, scratch, archive=None):
    """固定落盘路径 scratch/tu.asc + scratch/tu.o（**所有**组合同一路径）后编译。

    .o 内嵌的是源文件路径 ⇒ 若每组合用不同文件名，sha 天然不可比（D2 首跑踩过）。
    故：同路径编 → 立刻 sha256 → 再按需把 .o 另存为 archive 名留档。
    """
    path = os.path.join(scratch, "tu.asc")
    with open(path, "w") as f:
        f.write(body)
    out = os.path.join(scratch, "tu.o")
    if os.path.exists(out):
        os.remove(out)
    cmd = [BISHENG] + STD + ARCH + ["-I%s/src" % src_root, "-DTL_PORT910B_NATIVE_TYPES",
                                    "-I%s/include" % ROOT, "-I%s/asc/impl" % ROOT,
                                    "-I%s/asc/include" % ROOT] + extra + \
          ["--cce-aicore-only", "-c", path, "-o", out]
    p = subprocess.run(cmd, capture_output=True, text=True)
    log = (p.stdout or "") + (p.stderr or "")
    sha, size = "", 0
    if p.returncode == 0 and os.path.exists(out):
        raw = open(out, "rb").read()
        sha, size = hashlib.sha256(raw).hexdigest()[:16], len(raw)
        if archive:
            with open(archive, "wb") as f:
                f.write(raw)
    err = ""
    for line in log.splitlines():
        if "error" in line.lower():
            err = line.strip()[:170]
            break
    return p.returncode, size, sha, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tilelang", help="trunk 仓根（含 src/tl_templates）")
    ap.add_argument("--ref", default="/tmp/d2_ref_src", help="pristine 参照树根（D2 改头前快照）")
    ap.add_argument("--workdir", default="/tmp/d2_matrix")
    a = ap.parse_args()
    scratch = os.path.join(a.workdir, "scratch")
    arch = os.path.join(a.workdir, "objects")
    os.makedirs(scratch, exist_ok=True)
    os.makedirs(arch, exist_ok=True)

    ref_md5 = os.popen("md5sum %s/src/tl_templates/ascend/port910b_compat.h" % a.ref).read().split()
    src_md5 = os.popen("md5sum %s/src/tl_templates/ascend/port910b_compat.h" % a.src).read().split()
    print("== D2 判别器矩阵  bisheng=%s" % os.path.basename(BISHENG))
    print("   ref  %s  compat md5=%s" % (a.ref, (ref_md5[0] if ref_md5 else "?")))
    print("   src  %s  compat md5=%s" % (a.src, (src_md5[0] if src_md5 else "?")))
    have_ref = os.path.isdir(os.path.join(a.ref, "src", "tl_templates"))

    res = {}
    for cname, extra in COMBOS:
        for tu, body in sorted(TUS.items()):
            tag = "%s__%s" % (tu, cname.replace(" ", "").replace("-", "").replace("+", "_"))
            rc, size, sha, err = compile_tu(tu, body, a.src, extra, scratch,
                                            os.path.join(arch, tag + ".o"))
            res.setdefault(cname, {})[tu] = (rc, size, sha, err)
    ref = {}
    if have_ref:
        for tu, body in sorted(TUS.items()):
            ref[tu] = compile_tu(tu, body, a.ref, [], scratch)

    print("\n① 默认路径逐字节等价（同 TU 同路径；ref=改头前 vs trunk=D2 落地后，均不加 -D）")
    print("   %-28s %-20s %-20s %s" % ("TU", "ref .o sha/size", "trunk .o sha/size", "判决"))
    if not have_ref:
        print("   SKIP：参照树不存在（%s/src/tl_templates）→ 建法见 D2_variants.md §5" % a.ref)
        verdict1 = "SKIP"
    else:
        bad = 0
        for tu in sorted(TUS):
            r, d = ref[tu], res["default"][tu]
            ok = (r[0] == d[0]) and (r[2] == d[2]) and d[0] == 0
            bad += 0 if ok else 1
            print("   %-28s %-20s %-20s %s" % (
                tu, "%s/%d" % (r[2] or "rc%d" % r[0], r[1]),
                "%s/%d" % (d[2] or "rc%d" % d[0], d[1]),
                "IDENTICAL" if ok else "!!DIFF!!"))
        verdict1 = "PASS" if bad == 0 else "FAIL"
        print("   → 默认路径逐字节等价：%s（%d/%d TU 整只 .o sha256 全等）"
              % (verdict1, len(TUS) - bad, len(TUS)))

    print("\n② 变体是否改变生成码（trunk 头树，与 default 比 .o sha256；SAME=该 TU 不吃这组开关）")
    print("   %-28s %s" % ("TU", " ".join("%-11s" % c for c, _ in COMBOS[1:])))
    for tu in sorted(TUS):
        row = []
        for cname, _ in COMBOS[1:]:
            rc, size, sha, err = res[cname][tu]
            row.append("RC%d!" % rc if rc != 0 else
                       ("DIFF" if sha != res["default"][tu][2] else "SAME"))
        print("   %-28s %s" % (tu, " ".join("%-11s" % x for x in row)))

    print("\n③ 编译判决行（rc!=0 给原文错误首行）")
    for cname, _ in COMBOS:
        npass = sum(1 for tu in TUS if res[cname][tu][0] == 0)
        print("   %-11s %d/%d rc=0" % (cname, npass, len(TUS)))
        for tu in sorted(TUS):
            rc, size, sha, err = res[cname][tu]
            if rc != 0:
                print("        FAIL %-28s %s" % (tu, err))
            else:
                print("        PASS %-28s .o=%6dB sha=%s" % (tu, size, sha))

    print("\n④ 判别力小结（活 = 与 default 的 rc 或 .o sha256 有差）")
    for cname, _ in COMBOS[1:]:
        live = [tu for tu in sorted(TUS)
                if res[cname][tu][0] != res["default"][tu][0]
                or res[cname][tu][2] != res["default"][tu][2]]
        dead = [tu for tu in sorted(TUS) if tu not in live]
        print("   %-11s 活: %s" % (cname, ", ".join(live) or "（无——本载荷集上完全空开关）"))
        if dead:
            print("   %-11s 空: %s" % ("", ", ".join(dead)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
