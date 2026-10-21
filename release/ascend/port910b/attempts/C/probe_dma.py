#!/usr/bin/env python3
"""attempts/C/probe_dma.py — 910B(dav-2201) UB<->GM 搬运面原生件探针。

背景：tilelang 的 `T.copy`（GM<->UB）在 codegen 里落到 compat §10b 的
`asc_copy_gm2ub_align / asc_copy_ub2gm_align`，而后两者内部又调 CCE 原生
`copy_gm_to_ubuf / copy_ubuf_to_gm`。§10b 是按 **950 的 "packed config 三参"**
签名写的，且从未被真实调用过（e2e_cube 只走 Cube，GDN scalar 变体只走标量 GM
直读）。ubstage 变体首次真实触达即挂：
  error: unknown type name 'asc_store_l2_cache_mode'
本探针把这一层做实：逐个候选写法编一个最小 .asc（aibin 全链，与 tilelang
codegen 同一条 bisheng 通路），PASS/FAIL 直接进缺口清单与 GAP-C 补丁依据。

顺带取证 codegen 的 arity 事实（src/ascend/codegen/codegen_ascend.cc）：
  EmitGmToUbCopy_   -> asc_copy_gm2ub_align(...) 共 10 参（含 static_cast<asc_load_l2_cache_mode>）
  EmitUbufToGmCopy_ -> asc_copy_ub2gm_align(dst, src, burst_num, burst_len,
                        static_cast<asc_store_l2_cache_mode>(ctl),
                        burst_dst_stride, burst_src_stride) 共 7 参
=> 与 §10b 声明（10 参）不符，本探针把两种 arity 都试一遍。

UB 句柄写法取证：CCE 既不接受 `__ubuf__ T ub[N]`（"array ub is only valid for
Local Memory"），也不接受 `__shared__`（unknown type name）；tilelang 的
GetAscendScopeQualifier 路径发的是 `__ubuf__ T *buf = (__ubuf__ T *)0;`，本探针
沿用同形态。
"""
import os
import re

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")

from tilelang.contrib import bisheng  # noqa: E402
from tilelang.env import TILELANG_TEMPLATE_PATH  # noqa: E402

ROOT = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest") + "/aarch64-linux"
INC = ["-I" + TILELANG_TEMPLATE_PATH, "-DTL_PORT910B_NATIVE_TYPES",
       "-I%s/asc/impl" % ROOT, "-I%s/asc/include" % ROOT]
HDR = '#include <tl_templates/ascend/common.h>\n'

PRE = ('extern "C" __global__ __vector__ void probe(__gm__ float *out, __gm__ float *in) {\n'
       '  __ubuf__ uint8_t *ub = (__ubuf__ uint8_t *)0;\n'
       '  __ubuf__ float *uf = (__ubuf__ float *)0;\n'
       '  ub[0] = 0; uf[0] = 0.f;\n')

# (名字, 调用体) —— 每个都在 __vector__ kernel 里真调用一次
CASES = [
    # --- 950 风格：packed uint64 config（compat §10b 目前就这么写的） ---
    ("gm2ub_cfg3_voidptr",
     '  copy_gm_to_ubuf((__ubuf__ void *)uf, (__gm__ void *)in, (uint64_t)0);'),
    ("gm2ub_cfg3_u8ptr",
     '  copy_gm_to_ubuf(ub, (__gm__ uint8_t *)in, (uint64_t)0);'),
    ("ub2gm_cfg3_voidptr",
     '  copy_ubuf_to_gm((__gm__ void *)out, (__ubuf__ void *)uf, (uint64_t)0);'),
    ("ub2gm_cfg3_u8ptr",
     '  copy_ubuf_to_gm((__gm__ uint8_t *)out, ub, (uint64_t)0);'),
    # --- 910B/C220 风格：(dst, src, sid, burst_num, burst_len, srcStride, dstStride) ---
    ("gm2ub_7arg_voidptr",
     '  copy_gm_to_ubuf((__ubuf__ void *)uf, (__gm__ void *)in, (int8_t)0, (uint16_t)1, '
     '(uint16_t)32, (uint16_t)0, (uint16_t)0);'),
    ("gm2ub_7arg_u8ptr",
     '  copy_gm_to_ubuf(ub, (__gm__ uint8_t *)in, (int8_t)0, (uint16_t)1, '
     '(uint16_t)32, (uint16_t)0, (uint16_t)0);'),
    ("ub2gm_7arg_voidptr",
     '  copy_ubuf_to_gm((__gm__ void *)out, (__ubuf__ void *)uf, (int8_t)0, (uint16_t)1, '
     '(uint16_t)32, (uint16_t)0, (uint16_t)0);'),
    ("ub2gm_7arg_u8ptr",
     '  copy_ubuf_to_gm((__gm__ uint8_t *)out, ub, (int8_t)0, (uint16_t)1, '
     '(uint16_t)32, (uint16_t)0, (uint16_t)0);'),
    ("ub2gm_7arg_f32ptr",
     '  copy_ubuf_to_gm((__gm__ float *)out, uf, (int8_t)0, (uint16_t)1, '
     '(uint16_t)32, (uint16_t)0, (uint16_t)0);'),
    # --- 连续拷贝原生件（无 config/stride，最保守的 UB<->GM 通道） ---
    ("copy_data_gm2ub",
     '  copy_data(ub, (__gm__ uint8_t *)in, (uint64_t)256);'),
    ("copy_data_ub2gm",
     '  copy_data((__gm__ uint8_t *)out, ub, (uint64_t)256);'),
    ("copy_data_align64_gm2ub",
     '  copy_data_align64(ub, (__gm__ uint8_t *)in, (uint64_t)256);'),
    # --- compat §10b 现成适配器（10 参声明，内部走 cfg3） ---
    ("compat_gm2ub_align_10arg",
     '  asc_copy_gm2ub_align(ub, (__gm__ uint8_t *)in, 0, 32, 0, 0, 1, '
     'ASC_LOAD_L2_CACHE_ALLOC, 1, 32);'),
    ("compat_ub2gm_align_10arg",
     '  asc_copy_ub2gm_align((__gm__ uint8_t *)out, ub, 0, 32, 0, 0, 1, '
     '(int)0, 1, 32);'),
    # --- codegen 真实发射形态（ub2gm 只有 7 参 + asc_store_l2_cache_mode） ---
    ("codegen_ub2gm_7arg_as_emitted",
     '  asc_copy_ub2gm_align((__gm__ uint8_t *)out, ub, 1, 2048, '
     'static_cast<asc_store_l2_cache_mode>(4), 2048, 2048);'),
    ("asc_store_l2_cache_mode_exists",
     '  asc_store_l2_cache_mode m = static_cast<asc_store_l2_cache_mode>(4); (void)m;'),
    # --- 负形对照：故意错的 arity。若也 PASS => 原生声明是变参 builtin alias，
    #     则上面所有 PASS 只证明"名字存在"，不证明"参数形态合法"（G-C4 必上卡）。
    ("arity_control_ub2gm_2arg",
     '  copy_ubuf_to_gm((__gm__ void *)out, (__ubuf__ void *)uf);'),
    ("arity_control_ub2gm_9arg",
     '  copy_ubuf_to_gm((__gm__ void *)out, (__ubuf__ void *)uf, (int8_t)0, (uint16_t)1, '
     '(uint16_t)32, (uint16_t)0, (uint16_t)0, (uint8_t)4, (uint8_t)5);'),
    ("arity_control_gm2ub_5arg",
     '  copy_gm_to_ubuf((__ubuf__ void *)uf, (__gm__ void *)in, (int8_t)0, (uint16_t)1, '
     '(uint16_t)32);'),
]


def run():
    npass = nfail = 0
    for name, body in CASES:
        src = HDR + PRE + body + '\n  out[0] = in[0];\n}\n'
        try:
            bisheng.compile_ascend(src, target_format="aibin", npu_arch="dav-2201", options=INC)
            print("PROBE %-34s PASS" % name)
            npass += 1
        except Exception as e:  # noqa: BLE001
            s = str(e)
            errs = re.findall(r"[^\n]*(?:error:|undefined symbol|Fatal Err)[^\n]*", s)
            first = errs[0][:170].strip() if errs else s.splitlines()[-1][:170]
            print("PROBE %-34s FAIL :: %s" % (name, first))
            nfail += 1
    print("PROBE-DMA-SUMMARY pass=%d fail=%d" % (npass, nfail))


if __name__ == "__main__":
    run()
