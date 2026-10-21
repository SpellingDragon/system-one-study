#!/usr/bin/env python3
"""attempts/Z_probe_sec12.py — p2-13 P1-1k：落地后 §12 的 compile-only 自证（dav-2201）。

与 X_probe_proposal.py 的差别：X 探针编的是**提案粘贴文本**（pr_* 独立件）；本探针编的是
**真实 trunk 落地件**——经 `common.h → port910b_compat.h` 包含链直接调用
asc_copy_gm2l1_nd2nz / asc_copy_gm2l1_dn2nz / asc_copy_l12l0a / asc_copy_l12l0b，
覆盖 P1 三 dtype 族 + gCol>=UINT16_MAX 逐行兜底支 + P2 拆趟支 + kStep==1 退化支。

复跑（容器 cann910b-k；前提：/tilelang 即 trunk 仓 virtiofs 挂载）：
  docker cp Z_probe_sec12.py cann910b-k:/tmp/
  docker exec cann910b-k bash -c 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \
    cd /tmp && python3 /tmp/Z_probe_sec12.py'
"""
import subprocess

BISHENG = "/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/bin/bisheng"
ROOT = "/usr/local/Ascend/cann-8.5.0/aarch64-linux"
INC = ["-I/tilelang/src", "-DTL_PORT910B_NATIVE_TYPES",
       "-I%s/include" % ROOT, "-I%s/asc/impl" % ROOT, "-I%s/asc/include" % ROOT]
STD = ["-std=c++20", "-fPIC", "-O2"]
ARCH = ["--npu-arch=dav-2201"]
HDR = "#include <tl_templates/ascend/common.h>\n"
MODE = "static_cast<asc_load_l2_cache_mode>(0)"

# 驱动体：真件名、真签名（与 codegen_ascend.cc:1458/1570-1572 的实参位序一致）
GM = """
extern "C" __global__ __cube__ void probe(__gm__ %s *g) {
  __cbuf__ %s *c = (__cbuf__ %s *)(uintptr_t)0x10000;
  asc_copy_gm2l1_nd2nz(c, g, %s, %s, 16, 32, 0, 0);
  (void)g; (void)c;
}
"""
GM_FALLBACK = """
extern "C" __global__ __cube__ void probe(__gm__ bfloat16_t *g) {
  __cbuf__ bfloat16_t *c = (__cbuf__ bfloat16_t *)(uintptr_t)0x10000;
  // rowBytes=131072 → gColW=65536 ≥ UINT16_MAX → 官方同款 M 向逐行拆趟支
  asc_copy_gm2l1_nd2nz(c, g, 131072, %s, 16, 32, 0, 0);
  (void)g; (void)c;
}
"""
DN = """
extern "C" __global__ __cube__ void probe(__gm__ bfloat16_t *g) {
  __cbuf__ bfloat16_t *c = (__cbuf__ bfloat16_t *)(uintptr_t)0x10000;
  asc_copy_gm2l1_dn2nz(c, g, 64, %s, 16, 32, 0, 0);
  (void)g; (void)c;
}
"""
L0A = """
extern "C" __global__ __cube__ void probe(__gm__ bfloat16_t *g) {
  asc_copy_l12l0a((__ca__ bfloat16_t *)(uintptr_t)0x40000,
                  (__cbuf__ bfloat16_t *)(uintptr_t)0x10000, %s);
  (void)g;
}
"""
L0B = """
extern "C" __global__ __cube__ void probe(__gm__ bfloat16_t *g) {
  asc_copy_l12l0b((__cb__ bfloat16_t *)(uintptr_t)0x48000,
                  (__cbuf__ bfloat16_t *)(uintptr_t)0x10000, 0, 0, 1, 4, 8, 4);
  (void)g;
}
"""


def run(name, body):
    path = "/tmp/zps_%s.asc" % name
    with open(path, "w") as f:
        f.write(body)
    out = "/tmp/zps_%s.o" % name
    cmd = [BISHENG] + STD + ARCH + INC + ["--cce-aicore-only", "-c", path, "-o", out]
    p = subprocess.run(cmd, capture_output=True, text=True)
    log = ((p.stdout or "") + (p.stderr or ""))
    errs = [l for l in log.splitlines() if "error" in l.lower()][:4]
    print("── %-26s %s (rc=%d)" % (name, "PASS" if p.returncode == 0 else "FAIL", p.returncode))
    for e in errs:
        print("   ERR| " + e.strip())
    return p.returncode


if __name__ == "__main__":
    rc = []
    rc.append(run("Z1_gm2l1_bf16_nd2nz", HDR + (GM % ("bfloat16_t", "bfloat16_t", "bfloat16_t", "64", MODE))))
    rc.append(run("Z2_gm2l1_fp32_nd2nz", HDR + (GM % ("float", "float", "float", "128", MODE))))
    rc.append(run("Z3_gm2l1_int8_nd2nz", HDR + (GM % ("int8_t", "int8_t", "int8_t", "32", MODE))))
    rc.append(run("Z4_gm2l1_gcol_fallback", HDR + (GM_FALLBACK % MODE)))
    rc.append(run("Z5_dn2nz_delegation", HDR + (DN % MODE)))
    rc.append(run("Z6_l12l0a_msplit", HDR + (L0A % "1, 2, 2, 4, 8, 4")))
    rc.append(run("Z7_l12l0a_kstep1", HDR + (L0A % "0, 0, 3, 1, 8, 1")))
    rc.append(run("Z8_l12l0b_kmajor", HDR + L0B))
    print("=== rc=0: %d/8" % sum(1 for x in rc if x == 0))
