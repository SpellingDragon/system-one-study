#!/usr/bin/env python3
"""attempts/X_probe_proposal.py — p2-13 P1-1j：X_sec12_proposal.md 的 compile-only 自证。

纪律：**不落地**（不触碰 trunk port910b_compat.h）。本脚本把提案里的 wrapper 原文粘进独立 TU，
在 dav-2201 上编译，证明「提案形态在 2201 语法/语义面可编」，并逐槽回读 sema 约束。

复跑（容器 cann910b-k）：
  docker cp X_probe_proposal.py cann910b-k:/tmp/
  docker exec cann910b-k bash -c 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \
    cd /tmp && python3 /tmp/X_probe_proposal.py'
"""
import subprocess

BISHENG = "/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/bin/bisheng"
ROOT = "/usr/local/Ascend/cann-8.5.0/aarch64-linux"
INC = ["-I/tilelang/src", "-DTL_PORT910B_NATIVE_TYPES",
       "-I%s/include" % ROOT, "-I%s/asc/impl" % ROOT, "-I%s/asc/include" % ROOT]
STD = ["-std=c++20", "-fPIC", "-O2"]
ARCH = ["--npu-arch=dav-2201"]
HDR = "#include <tl_templates/ascend/common.h>\n"

# ── 提案件原文（与 X_sec12_proposal.md 的 diff 一一对应）────────────────────
GM2L1 = r"""
// ── 提案件 P1：GM→L1 走官方 2201 ND2NZ 硬件转换件（11 参）
template <typename DT, typename ST>
__aicore__ inline void pr_copy_gm2l1_nd2nz(__cbuf__ DT *dst, __gm__ ST *src,
                                           int rowBytes, int nRows, int cols) {
  uint16_t const gCol = (uint16_t)((unsigned)rowBytes / (unsigned)sizeof(ST));  // GM 行宽(元素)
  uint16_t const c0s = (uint16_t)((((unsigned)nRows + 15u) / 16u) * 16u);       // dstNzC0Stride(32B 块)
  if constexpr (sizeof(ST) == 1) {
    copy_gm_to_cbuf_multi_nd2nz_b8((__cbuf__ int8_t *)(uintptr_t)dst, (__gm__ int8_t *)(uintptr_t)src,
        (int8_t)0, (uint16_t)1, (uint16_t)nRows, (uint16_t)cols,
        (uint16_t)0, gCol, c0s, (uint16_t)1, (uint16_t)0);
  } else if constexpr (sizeof(ST) == 2) {
    copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ bfloat16_t *)(uintptr_t)dst, (__gm__ bfloat16_t *)(uintptr_t)src,
        (int8_t)0, (uint16_t)1, (uint16_t)nRows, (uint16_t)cols,
        (uint16_t)0, gCol, c0s, (uint16_t)1, (uint16_t)0);
  } else {
    copy_gm_to_cbuf_multi_nd2nz_b32s((__cbuf__ float *)(uintptr_t)dst, (__gm__ float *)(uintptr_t)src,
        (int8_t)0, (uint16_t)1, (uint16_t)nRows, (uint16_t)cols,
        (uint16_t)0, gCol, c0s, (uint16_t)1, (uint16_t)0);
  }
}
"""

L12L0 = r"""
// ── 提案件 P2：L1→L0A 按官方 V1 模板（起点折指针 + K 向 repeat + M 向拆趟）
template <typename T>
__aicore__ inline void pr_copy_l12l0a(__ca__ T *dst, __cbuf__ T *src, int mStart, int kStart,
                                      int mStep, int kStep, int srcStride, int dstStride) {
  (void)dstStride;  // VERIFY U3：假定 L0 紧凑（dst 侧 pitch == kStep），非紧凑须改走 dstGap/再拆趟
  int const blkElems = 16 * (int)(32 / (int)sizeof(T));      // 分形块元素数（bf16=256，=16×32B）
  __cbuf__ T *base = (__cbuf__ T *)(uintptr_t)src + (int64_t)(kStart * srcStride + mStart) * blkElems;
  __ca__ T *dbase = (__ca__ T *)(uintptr_t)dst;
  if (kStep == 1) {
    load_cbuf_to_ca(dbase, base, (uint16_t)0, (uint8_t)mStep, (uint16_t)1, (uint16_t)0,
                    (uint8_t)0, false, (__cce_scalar::addr_cal_mode_t)0);
  } else {
    for (int i = 0; i < mStep; ++i) {
      load_cbuf_to_ca(dbase + (int64_t)i * kStep * blkElems,
                      base + (int64_t)i * blkElems,
                      (uint16_t)0, (uint8_t)kStep, (uint16_t)srcStride, (uint16_t)0,
                      (uint8_t)0, false, (__cce_scalar::addr_cal_mode_t)0);
    }
  }
}

template <typename T>
__aicore__ inline void pr_copy_l12l0b(__cb__ T *dst, __cbuf__ T *src, int mStart, int kStart,
                                      int mStep, int kStep, int srcStride, int dstStride) {
  (void)mStart; (void)mStep; (void)dstStride;  // B 侧本轮只走 K-major 单趟形态；VERIFY U2
  int const blkElems = 16 * (int)(32 / (int)sizeof(T));
  __cbuf__ T *base = (__cbuf__ T *)(uintptr_t)src + (int64_t)(kStart * srcStride) * blkElems;
  load_cbuf_to_cb((__cb__ T *)(uintptr_t)dst, base, (uint16_t)0, (uint8_t)kStep,
                  (uint16_t)srcStride, (uint16_t)0, (uint8_t)0, false,
                  (__cce_scalar::addr_cal_mode_t)0);
}
"""

# ── 止血件 P0（E-3 对照实验用）：只把 sid 钉 0，其余一行不动
BLEED = r"""
template <typename T>
__aicore__ inline void pr_bleed_l12l0a(__ca__ T *dst, __cbuf__ T *src, int mStart, int kStart,
                                       int mStep, int kStep, int srcStride, int dstStride) {
  (void)kStep;
  load_cbuf_to_ca(dst, src, (uint16_t)kStart, (uint8_t)mStep, (uint16_t)srcStride,
                  (uint16_t)dstStride, (uint8_t)0, false,   // ← 唯一改动：sid 槽不再吃 mStart
                  (__cce_scalar::addr_cal_mode_t)0);
}
"""

DRIVER = (
    'extern "C" __global__ __cube__ void probe(__gm__ bfloat16_t *g) {\n'
    "  __cbuf__ bfloat16_t *c = (__cbuf__ bfloat16_t *)(uintptr_t)0x10000;\n"
    "  __ca__ bfloat16_t *ca = (__ca__ bfloat16_t *)(uintptr_t)0x40000;\n"
    "  __cb__ bfloat16_t *cb = (__cb__ bfloat16_t *)(uintptr_t)0x48000;\n"
    "  pr_copy_gm2l1_nd2nz<c16, c16>(c, g, 32 * 2, 16, 32);\n"
    "  pr_copy_l12l0a(ca, c, 1, 2, 2, 4, 8, 4);\n"
    "  pr_copy_l12l0b(cb, c, 0, 0, 1, 4, 8, 4);\n"
    "  pr_bleed_l12l0a(ca, c, 1, 2, 2, 4, 8, 4);\n"
    "  (void)g; (void)c; (void)ca; (void)cb;\n"
    "}\n"
)


def run(name, body):
    path = "/tmp/xpp_%s.asc" % name
    with open(path, "w") as f:
        f.write(body)
    out = "/tmp/xpp_%s.o" % name
    cmd = [BISHENG] + STD + ARCH + INC + ["--cce-aicore-only", "-c", path, "-o", out]
    p = subprocess.run(cmd, capture_output=True, text=True)
    log = ((p.stdout or "") + (p.stderr or ""))
    errs = [l for l in log.splitlines() if "error" in l.lower()][:4]
    print("── %-26s %s (rc=%d)" % (name, "PASS" if p.returncode == 0 else "FAIL", p.returncode))
    for e in errs:
        print("   ERR| " + e.strip())
    return p.returncode


if __name__ == "__main__":
    c16 = "bfloat16_t"
    # 全量提案件
    run("all_in_one", HDR + GM2L1 + L12L0 + BLEED + DRIVER.replace("c16", c16))
    # 逐件隔离（便于定位是哪一件编不过）
    run("P1_gm2l1_nd2nz", HDR + GM2L1 + (
        'extern "C" __global__ __cube__ void probe(__gm__ bfloat16_t *g) {\n'
        "  pr_copy_gm2l1_nd2nz((__cbuf__ bfloat16_t *)(uintptr_t)0x10000, g, 64, 16, 32);\n"
        "  (void)g; }\n"))
    run("P1_gm2l1_nd2nz_fp32", HDR + GM2L1 + (
        'extern "C" __global__ __cube__ void probe(__gm__ float *g) {\n'
        "  pr_copy_gm2l1_nd2nz((__cbuf__ float *)(uintptr_t)0x10000, g, 128, 16, 32);\n"
        "  (void)g; }\n"))
    run("P1_gm2l1_nd2nz_fp8", HDR + GM2L1 + (
        'extern "C" __global__ __cube__ void probe(__gm__ int8_t *g) {\n'
        "  pr_copy_gm2l1_nd2nz((__cbuf__ int8_t *)(uintptr_t)0x10000, g, 32, 16, 32);\n"
        "  (void)g; }\n"))
    run("P2_l12l0a", HDR + L12L0 + (
        'extern "C" __global__ __cube__ void probe(__gm__ bfloat16_t *g) {\n'
        "  pr_copy_l12l0a((__ca__ bfloat16_t *)(uintptr_t)0x40000,\n"
        "                 (__cbuf__ bfloat16_t *)(uintptr_t)0x10000, 1, 2, 2, 4, 8, 4);\n"
        "  (void)g; }\n"))
    run("P2_l12l0a_kstep1", HDR + L12L0 + (
        'extern "C" __global__ __cube__ void probe(__gm__ bfloat16_t *g) {\n'
        "  pr_copy_l12l0a((__ca__ bfloat16_t *)(uintptr_t)0x40000,\n"
        "                 (__cbuf__ bfloat16_t *)(uintptr_t)0x10000, 0, 0, 3, 1, 8, 1);\n"
        "  (void)g; }\n"))
    run("P2_l12l0b", HDR + L12L0 + (
        'extern "C" __global__ __cube__ void probe(__gm__ bfloat16_t *g) {\n'
        "  pr_copy_l12l0b((__cb__ bfloat16_t *)(uintptr_t)0x48000,\n"
        "                 (__cbuf__ bfloat16_t *)(uintptr_t)0x10000, 0, 0, 1, 4, 8, 4);\n"
        "  (void)g; }\n"))
    run("P0_bleed_sid0", HDR + BLEED + (
        'extern "C" __global__ __cube__ void probe(__gm__ bfloat16_t *g) {\n'
        "  pr_bleed_l12l0a((__ca__ bfloat16_t *)(uintptr_t)0, \n"
        "                  (__cbuf__ bfloat16_t *)(uintptr_t)0x10000, 1, 2, 2, 4, 8, 4);\n"
        "  (void)g; }\n"))
