#!/usr/bin/env python3
"""attempts/X_probe_nd2nz.py — p2-13 P1-1j 路径3 追加判别：GM→L1 的 ND2NZ 件在 dav-2201 的可编性。

背景（本探针要钉死的问题）：
  §12 的 asc_copy_gm2l1_nd2nz 走的是 copy_gm_to_cbuf 8 参裸形（线性块拷贝，等价于
  官方 DataCopyGM2L1Impl / DataCopyParams 路），而官方 Matmul 的 ND 输入走的是
  CopyND2NZ → DataCopy(…, Nd2NzParams) → copy_gm_to_cbuf_multi_nd2nz_b16（11 参）。
  两者是否都能在 dav-2201 编过？谁才是 load_cbuf_to_ca(V1 分形语义) 的合法上游？

复跑（容器 cann910b-k）：
  docker cp X_probe_nd2nz.py cann910b-k:/tmp/
  docker exec cann910b-k bash -c 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \
    cd /tmp && python3 /tmp/X_probe_nd2nz.py'
"""
import os
import subprocess

BISHENG = os.environ.get(
    "BISHENG", "/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/bin/bisheng"
)
ROOT = (os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-8.5.0")) + "/aarch64-linux"
TPL = os.environ.get("TPL_INC", "/tilelang/src")
INC = [
    "-I" + TPL,
    "-DTL_PORT910B_NATIVE_TYPES",
    "-I%s/include" % ROOT,
    "-I%s/asc/impl" % ROOT,
    "-I%s/asc/include" % ROOT,
]
ARCH = ["--npu-arch=dav-2201"]
STD = ["-std=c++20", "-fPIC", "-O2"]

HDR = "#include <tl_templates/ascend/common.h>\n"
# __cbuf__ 不能作 __global__ 形参（前序探针已证），用 §12 同款整型中转句柄
KERNEL = (
    'extern "C" __global__ __cube__ void probe(__gm__ float *g) {\n'
    "  __cbuf__ bfloat16_t *c = (__cbuf__ bfloat16_t *)(uintptr_t)0x10000;\n"
    "  __gm__ bfloat16_t *gs = (__gm__ bfloat16_t *)g;\n"
    "  (void)g;\n"
)
TAIL = "  (void)c;\n}\n"


def build(name, body):
    src = HDR + body + TAIL
    path = "/tmp/xnz_%s.asc" % name
    with open(path, "w") as f:
        f.write(src)
    out = "/tmp/xnz_%s.o" % name
    cmd = [BISHENG] + STD + ARCH + INC + ["--cce-aicore-only", "-c", path, "-o", out]
    p = subprocess.run(cmd, capture_output=True, text=True)
    log = (p.stdout or "") + (p.stderr or "")
    return {"name": name, "rc": p.returncode, "log": log.strip(), "out": out}


def report(res, show=8):
    print("── %-30s %s (rc=%d)" % (res["name"], "PASS" if res["rc"] == 0 else "FAIL", res["rc"]))
    for ln in res["log"].splitlines()[:show]:
        print("   | " + ln)


# D1: 官方 dav_c220/kernel_operator_data_copy_impl.h:275 的 11 参形（Nd2NzParams 全 uint16_t）
D1 = KERNEL + (
    "  // DataCopyGM2L1ND2NZImplBase: (dst,src,sid, ndNum,nValue,dValue,\n"
    "  //   srcNdMatrixStride,srcDValue, dstNzC0Stride,dstNzNStride,dstNzMatrixStride)\n"
    "  copy_gm_to_cbuf_multi_nd2nz_b16(c, gs, (uint8_t)0,\n"
    "      (uint16_t)1, (uint16_t)16, (uint16_t)16, (uint16_t)0, (uint16_t)64,\n"
    "      (uint16_t)16, (uint16_t)1, (uint16_t)0);\n"
)
# D2: sid 越界（探 sid 槽的合法区间，与 load_cbuf_to_ca 的 [0,15] 对照）
D2 = KERNEL + (
    "  copy_gm_to_cbuf_multi_nd2nz_b16(c, gs, (uint8_t)200,\n"
    "      (uint16_t)1, (uint16_t)16, (uint16_t)16, (uint16_t)0, (uint16_t)64,\n"
    "      (uint16_t)16, (uint16_t)1, (uint16_t)0);\n"
)
# D3: 逐槽类型判别——把第 5 槽（nValue）换成 float，看 sema 要求的类型
D3 = KERNEL + (
    "  copy_gm_to_cbuf_multi_nd2nz_b16(c, gs, (uint8_t)0,\n"
    "      (uint16_t)1, 1.5f, (uint16_t)16, (uint16_t)0, (uint16_t)64,\n"
    "      (uint16_t)16, (uint16_t)1, (uint16_t)0);\n"
)
# D4: 3101 的 9 参 multi_nd2nz（对照用，前序已 FAIL，这里复现以证明失败非探针伪报）
D4 = KERNEL + (
    "  copy_gm_to_cbuf_multi_nd2nz(c, gs, (uint8_t)0, (uint64_t)64,\n"
    "      (uint8_t)0, (uint16_t)2, (uint32_t)8, (uint64_t)0, false);\n"
)
# D5: b32s（float）与 b8（int8）两变体是否同样存在（判 §12 若改 ND2NZ 需按位宽分派）
D5 = KERNEL + (
    "  copy_gm_to_cbuf_multi_nd2nz_b32s((__cbuf__ float *)c, (__gm__ float *)g, (int8_t)0,\n"
    "      (uint16_t)1, (uint16_t)16, (uint16_t)16, (uint16_t)0, (uint16_t)64,\n"
    "      (uint16_t)16, (uint16_t)1, (uint16_t)0);\n"
)
D6 = KERNEL + (
    "  copy_gm_to_cbuf_multi_nd2nz_b8((__cbuf__ int8_t *)c, (__gm__ int8_t *)g, (int8_t)0,\n"
    "      (uint16_t)1, (uint16_t)16, (uint16_t)16, (uint16_t)0, (uint16_t)64,\n"
    "      (uint16_t)16, (uint16_t)1, (uint16_t)0);\n"
)
# D7: §12 现用 8 参裸形（对照：官方 DataCopyGM2L1Impl 同款，必须 PASS）
D7 = KERNEL + (
    "  copy_gm_to_cbuf((__cbuf__ void *)c, (__gm__ void *)g, (int8_t)0,\n"
    "      (uint16_t)1, (uint16_t)32, (uint16_t)32, (uint16_t)32, (pad_t)0);\n"
)

CASES = [
    ("D1_nd2nz_b16_official11", D1),
    ("D2_nd2nz_b16_sidOOB", D2),
    ("D3_nd2nz_b16_slot5_float", D3),
    ("D4_multi_nd2nz_3101_9arg", D4),
    ("D5_nd2nz_b32s", D5),
    ("D6_nd2nz_b8", D6),
    ("D7_gm2l1_sec12_8arg", D7),
]

if __name__ == "__main__":
    print("bisheng =", BISHENG)
    print("ARCH    =", " ".join(ARCH))
    for name, body in CASES:
        report(build(name, body))
