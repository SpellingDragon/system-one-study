#!/usr/bin/env python3
"""attempts/X_probe_forms.py — p2-13 P1-1j 取证探针（路径3：bisheng 编译判别）。

目的（对应任务书三条待裁决项）：
  A. dav-2201 下 `cce_aicore_intrinsics_3101.h` 的显式重载是否可见
     （__clang_cce_types.h:231 `#if (__NPU_ARCH__ == 3101)` 的进入条件）。
  B. `load_cbuf_to_ca` 三种候选形的可编性与**绑定证据**：
     F1 = P1-1d 现 trunk 形 (dst,src,u16,u8,u16,u16,u8,bool,addr_cal_mode_t)
     F2 = 3101 显式形     (dst,src,mStart u16,kStart u16,mStep u8,kStep u8,srcStride i16,dstStride u16,transpose bool)
     F3 = config 形       (dst,src,config0 u64,config1 u64,transpose bool)
     各用**可辨识的立即数**，编过后 dump asm，从 LOAD_L1_TO_L0A_* 行的
     编码位段反推"实参落到了哪个槽"。
  C. GM→L1 三候选：§12 现用的 copy_gm_to_cbuf 8 参裸形 vs
     copy_gm_to_cbuf_multi_nd2nz 显式 9 参形 vs load_gm_to_cbuf_2dv2。

复跑（容器 cann910b-k）：
  docker exec cann910b-k bash -c 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \
    cd /tmp && python3 /work/release/ascend/port910b/attempts/X_probe_forms.py'
（若 /work 非 system-one 挂载，则 docker cp 本文件进 /tmp 后 python3 /tmp/X_probe_forms.py）
"""
import os
import subprocess

BISHENG = os.environ.get(
    "BISHENG", "/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/bin/bisheng"
)
ROOT = (os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-8.5.0")) + "/aarch64-linux"
TPL = os.environ.get("TPL_INC", "/tilelang/src")  # <tl_templates/ascend/common.h>
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
KERNEL = 'extern "C" __global__ __cube__ void probe(__gm__ float *g) {\n' + \
  "  __cbuf__ bfloat16_t *c = (__cbuf__ bfloat16_t *)(uintptr_t)0x10000;\n\n"
TAIL = "  (void)g; (void)c;\n}\n"


def build(name, body, emit_asm=True):
    """compile one TU; return dict(status, log, asm)"""
    src = HDR + body + TAIL
    path = "/tmp/xprobe_%s.asc" % name
    with open(path, "w") as f:
        f.write(src)
    out = ("/tmp/xprobe_%s.s" % name) if emit_asm else ("/tmp/xprobe_%s.o" % name)
    flags = ["-S"] if emit_asm else ["-c"]
    cmd = [BISHENG] + STD + ARCH + INC + flags + [path, "-o", out]
    p = subprocess.run(cmd, capture_output=True, text=True)
    log = (p.stdout or "") + (p.stderr or "")
    asm = ""
    if p.returncode == 0 and emit_asm and os.path.exists(out):
        with open(out) as f:
            asm = f.read()
    return {"name": name, "rc": p.returncode, "log": log.strip(), "asm": asm, "path": path, "out": out}


def report(res, keywords=("LOAD_L1_TO_L0A", "LOAD_L1_TO_L0B", "COPY_GM_TO", "LOAD_GM_TO", "SET_L1_2D")):
    st = "PASS" if res["rc"] == 0 else "FAIL"
    print("── %-28s %s" % (res["name"], st))
    if res["rc"] != 0:
        for ln in res["log"].splitlines()[:6]:
            print("   | " + ln)
    else:
        hits = [ln.strip() for ln in res["asm"].splitlines() if any(k in ln for k in keywords)]
        for h in hits[:8]:
            print("   ASM| " + h)
        if not hits:
            print("   (asm 中无关键行，见 %s)" % res["out"])


# ═══ A. 3101 头可见性（宏存在性两段探针）═══════════════════════════════════
VIS = KERNEL + (
    "#ifdef CCE_AICORE_INTRINSICS_TYPES_3101\n"
    '  #error "XARCH-VISIBLE: 3101 header IS included on dav-2201"\n'
    "#else\n"
    '  #error "XARCH-ABSENT: 3101 header NOT included"\n'
    "#endif\n"
)
# 附加：探 __NPU_ARCH__ 真值
ARCHVAL = KERNEL + "  static_assert(__NPU_ARCH__ == 99991, \"XNPUARCH\");\n"

# ═══ B. load_cbuf_to_ca 三形 ═════════════════════════════════════════════
CA_DECL = (
    "  __ca__ bfloat16_t *d = (__ca__ bfloat16_t *)(uintptr_t)0;\n"
    "  (void)d;\n"
)
F1 = KERNEL + CA_DECL + (
    "  // P1-1d trunk 形：(dst,src,(u16)101,(u8)102,(u16)103,(u16)104,(u8)105,false,(addr_cal_mode_t)0)\n"
    "  load_cbuf_to_ca(d, (__cbuf__ bfloat16_t *)c, (uint16_t)101, (uint8_t)102,\n"
    "                  (uint16_t)103, (uint16_t)104, (uint8_t)105, false,\n"
    "                  (__cce_scalar::addr_cal_mode_t)0);\n"
)
F2 = KERNEL + CA_DECL + (
    "  // 3101 显式形：mStart=101,kStart=102,mStep=103,kStep=104,srcStride=105,dstStride=106,transpose=false\n"
    "  load_cbuf_to_ca(d, (__cbuf__ bfloat16_t *)c, (uint16_t)101, (uint16_t)102,\n"
    "                  (uint8_t)103, (uint8_t)104, (int16_t)105, (uint16_t)106, false);\n"
)
F3 = KERNEL + CA_DECL + (
    "  // config 形：cfg0=0x0000_0100_0066_0065 (mStart=0x65=101,kStart=0x66=102,mStep=0x00,kStep=0x01)\n"
    "  load_cbuf_to_ca(d, (__cbuf__ bfloat16_t *)c, (uint64_t)0x0000010000660065ULL,\n"
    "                  (uint64_t)0x00000000006b0069ULL, false);\n"
)
F1R = KERNEL + CA_DECL + (
    "  // P1-1d trunk 形，sid 槽改用合法值 2（105>15 是探针常量的 sema 伪报）\n"
    "  load_cbuf_to_ca(d, (__cbuf__ bfloat16_t *)c, (uint16_t)101, (uint8_t)102,\n"
    "                  (uint16_t)103, (uint16_t)104, (uint8_t)2, true,\n"
    "                  (__cce_scalar::addr_cal_mode_t)0);\n"
)
CB_F2 = KERNEL + (
    "  __cb__ bfloat16_t *d = (__cb__ bfloat16_t *)(uintptr_t)0; (void)d;\n"
    "  load_cbuf_to_cb(d, (__cbuf__ bfloat16_t *)c, (uint16_t)201, (uint16_t)202,\n"
    "                  (uint8_t)203, (uint8_t)204, (int16_t)205, (uint16_t)206, false);\n"
)

# ═══ C. GM→L1 候选形 ═════════════════════════════════════════════════════
G1 = KERNEL + (
    "  // §12 现用 8 参裸形（data_copy_impl.h:92 同款）\n"
    "  copy_gm_to_cbuf((__cbuf__ void *)c, (__gm__ void *)g, (int8_t)0,\n"
    "                  (uint16_t)1, (uint16_t)32, (uint16_t)32, (uint16_t)32, (pad_t)0);\n"
)
G2 = KERNEL + (
    "  // 3101 显式 multi_nd2nz：(dst,src,sid,loop1_src_stride,l2, n_value,d_value,loop4_src_stride,smallc0_en)\n"
    "  copy_gm_to_cbuf_multi_nd2nz(c, (__gm__ bfloat16_t *)g, (uint8_t)0, (uint64_t)64,\n"
    "      (uint8_t)0, (uint16_t)2, (uint32_t)8, (uint64_t)0, false);\n"
)
G3 = KERNEL + (
    "  // GM→L1 2D-v2（LoadData2DGM2L1Cal 底件）：(dst,src,startIndex,repeatTimes,srcStride,dstGap,sid,inc)\n"
    "  load_gm_to_cbuf_2dv2((__cbuf__ void *)c, (__gm__ void *)g, (uint16_t)1, (uint16_t)2,\n"
    "      (uint16_t)3, (uint16_t)4, (uint8_t)0, (uint8_t)0);\n"
)

CASES = [
    ("A1_visibility", VIS),
    ("A2_archval", ARCHVAL),
    ("B1_ca_p11d_form", F1),
    ("B2_ca_explicit_form", F2),
    ("B3_ca_config_form", F3),
    ("B4_cb_explicit_form", CB_F2),
    ("C1_gm2l1_sec12_8arg", G1),
    ("C2_gm2l1_multi_nd2nz", G2),
    ("C3_gm2l1_2dv2", G3),
    ("B5_ca_p11d_sidinrange", F1R),
]

if __name__ == "__main__":
    print("bisheng =", BISHENG)
    print("INC     =", " ".join(INC))
    print("ARCH    =", " ".join(ARCH))
    for name, body in CASES:
        res = build(name, body, emit_asm=True)
        report(res)
