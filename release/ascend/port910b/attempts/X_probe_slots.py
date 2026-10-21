#!/usr/bin/env python3
"""attempts/X_probe_slots.py — p2-13 P1-1j 路径3 追加：逐槽类型/区间判别 + 跨 arch 交叉验证。

要回答的两个裁决项：
  E1..E7：dav-2201 上 `load_cbuf_to_ca` 的**每个槽**要求的类型/取值区间
          （用「故意放错类型 → 读 sema 报错里给出的期望类型」这一判别器，
           等价于逐槽反查签名，L 代理在 set_l1_2d 上已验证此法有效）。
  X1..X3：同一份 3101 显式形源码在 dav-3101 / dav-2201 下的可编性差异
          （若 3101 可编 & 2201 报 target feature/类型错 → 证该形是 arch 专属）。

复跑（容器 cann910b-k）：
  docker cp X_probe_slots.py cann910b-k:/tmp/
  docker exec cann910b-k bash -c 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \
    cd /tmp && python3 /tmp/X_probe_slots.py'
"""
import os
import subprocess

BISHENG = "/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/bin/bisheng"
ROOT = "/usr/local/Ascend/cann-8.5.0/aarch64-linux"
INC = [
    "-I/tilelang/src", "-DTL_PORT910B_NATIVE_TYPES",
    "-I%s/include" % ROOT, "-I%s/asc/impl" % ROOT, "-I%s/asc/include" % ROOT,
]
STD = ["-std=c++20", "-fPIC", "-O2"]
HDR = "#include <tl_templates/ascend/common.h>\n"
PRE = (
    'extern "C" __global__ __cube__ void probe(__gm__ float *g) {\n'
    "  (void)g;\n"
    "  __ca__ bfloat16_t *d = (__ca__ bfloat16_t *)(uintptr_t)0;\n"
    "  __cbuf__ bfloat16_t *c = (__cbuf__ bfloat16_t *)(uintptr_t)0x10000;\n"
    "  (void)d; (void)c;\n"
)
TAIL = "}\n"
# 基准：官方 V1 9 参形，合法取值
BASE = ("  load_cbuf_to_ca(d, c, (uint16_t)0, (uint8_t)1, (uint16_t)1, (uint16_t)0,\n"
        "                  (uint8_t)0, false, (__cce_scalar::addr_cal_mode_t)0);\n")


def slot_case(k, expr):
    """把第 k 个实参（1 起，含 dst/src）替换成 expr，其余保持基准。"""
    args = ["d", "c", "(uint16_t)0", "(uint8_t)1", "(uint16_t)1", "(uint16_t)0",
            "(uint8_t)0", "false", "(__cce_scalar::addr_cal_mode_t)0"]
    args[k - 1] = expr
    return "  load_cbuf_to_ca(%s);\n" % ", ".join(args)


CASES = [
    ("E0_base_v1_legal", PRE + BASE + TAIL, "dav-2201"),
    ("E1_slot3_float", PRE + slot_case(3, "1.5f") + TAIL, "dav-2201"),
    ("E2_slot4_float", PRE + slot_case(4, "1.5f") + TAIL, "dav-2201"),
    ("E3_slot5_float", PRE + slot_case(5, "1.5f") + TAIL, "dav-2201"),
    ("E4_slot6_float", PRE + slot_case(6, "1.5f") + TAIL, "dav-2201"),
    ("E5_slot7_uint16_16", PRE + slot_case(7, "(uint16_t)16") + TAIL, "dav-2201"),
    ("E6_slot8_int", PRE + slot_case(8, "1") + TAIL, "dav-2201"),
    ("E7_arity_8args", PRE + (
        "  load_cbuf_to_ca(d, c, (uint16_t)0, (uint8_t)1, (uint16_t)1, (uint16_t)0,\n"
        "                  (uint8_t)0, false);\n") + TAIL, "dav-2201"),
    ("E8_arity_10args", PRE + (
        "  load_cbuf_to_ca(d, c, (uint16_t)0, (uint8_t)1, (uint16_t)1, (uint16_t)0,\n"
        "                  (uint8_t)0, false, (__cce_scalar::addr_cal_mode_t)0, 0);\n") + TAIL, "dav-2201"),
    # X: 3101 显式形（mStart,kStart,mStep,kStep,srcStride,dstStride,transpose）跨 arch
    ("X1_explicit_on_2201", PRE + (
        "  load_cbuf_to_ca(d, c, (uint16_t)1, (uint16_t)2, (uint8_t)1, (uint8_t)2,\n"
        "                  (int16_t)1, (uint16_t)1, false);\n") + TAIL, "dav-2201"),
    ("X2_explicit_on_3101", PRE + (
        "  load_cbuf_to_ca(d, c, (uint16_t)1, (uint16_t)2, (uint8_t)1, (uint8_t)2,\n"
        "                  (int16_t)1, (uint16_t)1, false);\n") + TAIL, "dav-3101"),
    ("X3_v1_on_3101", PRE + BASE + TAIL, "dav-3101"),
]


def run(name, body, arch):
    path = "/tmp/xsl_%s.asc" % name
    with open(path, "w") as f:
        f.write(HDR + body)
    out = "/tmp/xsl_%s.o" % name
    cmd = [BISHENG] + STD + ["--npu-arch=" + arch] + INC + ["--cce-aicore-only", "-c", path, "-o", out]
    p = subprocess.run(cmd, capture_output=True, text=True)
    log = ((p.stdout or "") + (p.stderr or ""))
    errs = [ln for ln in log.splitlines() if "error" in ln.lower()]
    print("── %-24s [%s] %s" % (name, arch, "PASS" if p.returncode == 0 else "FAIL"))
    for e in errs[:3]:
        print("   ERR| " + e.strip())


if __name__ == "__main__":
    for n, b, a in CASES:
        run(n, b, a)
