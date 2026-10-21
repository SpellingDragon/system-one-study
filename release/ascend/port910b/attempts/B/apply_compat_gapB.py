#!/usr/bin/env python3
"""attempts/B/apply_compat_gapB.py — 把 GAP-B 补丁打进 cann910b-b 的 pip compat 副本

主仓 /tilelang 是模板真源（禁改），本脚本只改容器内 pip 运行副本，幂等：
  · 已打过（含 TL910B_GAPB_APPLIED）→ 只做校验，不重复打
  · 未打过 → 备份 .preGapB，然后做四处改动（依据写在 GAPB 注释里）

改动清单（详见 compat_gap_B.md）：
  G1 替换 asc_copy_gm2ub_align / asc_copy_ub2gm_align：910B(__DAV_C220_VEC__) 用
     7 参 copy_gm_to_ubuf/copy_ubuf_to_gm（长度/步长 32B 单位），原 3 参 config 形态属
     __DAV_L310__/M310/L311 段；且 codegen 的 ub2gm 实发 7 参，与 compat 原 10 参声明不符
  G2 asc_sync_notify / asc_sync_wait → **函数形宏**转 __cce_scalar::set_flag/wait_flag
     （910B 该 builtin 要求 pipe 实参为字面枚举常量，inline 函数转调会 FAIL，见 probe_setflag.sh）
  G3 增 enum asc_store_l2_cache_mode（codegen 以 static_cast<asc_store_l2_cache_mode>(4) 发射）
  G4 增标量 rsqrtf shim（910B 标量面无 rsqrtf；T.rsqrt 落码为 rsqrtf(float)）
"""
import pathlib
import sys

MARK = "TL910B_GAPB_APPLIED"

GAPB = r'''
// ── 10b-GAPB（P1-1 执行代理 B 于 cann910b-b 实测补齐；证据见 attempts/B/compat_gap_B.md）
#ifndef TL910B_GAPB_APPLIED
#define TL910B_GAPB_APPLIED

// G3: store 侧 L2 cache 模式枚举。发射点 codegen_ascend.cc:1430 以
// `static_cast<asc_store_l2_cache_mode>(n)` 传值（n=4 来自 copy.cc:316 kStoreDefaultL2CacheCtrl），
// 910B 的 GM<-UB 搬运原生件无 L2 ctrl 形参，本值在此层被丢弃。
enum asc_store_l2_cache_mode {
  ASC_STORE_L2_CACHE_ALLOC = 0,
  ASC_STORE_L2_CACHE_NORMAL = 1,
  ASC_STORE_L2_CACHE_FREE = 2,
  ASC_STORE_L2_CACHE_DEFAULT = 4
};

// G1: GM<->UB 搬运的 910B 正确形态（7 参原生件，长度/步长以 32B 为单位）。
// codegen 侧实发形态（src/ascend/codegen/codegen_ascend.cc:1405-1433 + src/ascend/op/copy.cc:300-328）：
//   asc_copy_gm2ub_align(dst, src, n_rows, row_bytes, 0, 0, 0, load_mode, src_stride_bytes, dst_stride_bytes)
//   asc_copy_ub2gm_align(dst, src, burst_num, burst_len, store_mode, burst_dst_stride, burst_src_stride)  // 7 参!
// 910B 原生件（asc/impl/basic_api/dav_c220/kernel_operator_data_copy_impl.h:51/124/221）：
//   copy_gm_to_ubuf(__ubuf__ void*, __gm__ void*, sid, blockCount, blockLen, srcStride, dstStride)
//   copy_ubuf_to_gm(__gm__  void*, __ubuf__ void*, sid, blockCount, blockLen, srcStride, dstStride)
//   —— 官方 :221 传 `tensorSize*sizeof(T)/32` 作 blockLen，故单位=32B；名字见
//      cce_aicore_intrinsics.h:992/1042（namespace __cce_scalar，无 arch guard）。
#define TL910B_BYTES_TO_BLK(x) ((uint16_t)(((int)(x)) / 32))

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ uint8_t *dst, __gm__ uint8_t *src,
                                            int blockCount, int blockLenB, int rsvd0,
                                            int rsvd1 /*right_pad*/, int rsvd2,
                                            asc_load_l2_cache_mode mode, int srcStrideB,
                                            int dstStrideB) {
  (void)rsvd0; (void)rsvd1; (void)rsvd2; (void)mode;
  __cce_scalar::copy_gm_to_ubuf((__ubuf__ void *)dst, (__gm__ void *)src, (uint8_t)0,
                                (uint16_t)blockCount, TL910B_BYTES_TO_BLK(blockLenB),
                                TL910B_BYTES_TO_BLK(srcStrideB), TL910B_BYTES_TO_BLK(dstStrideB));
}

__aicore__ inline void asc_copy_ub2gm_align(__gm__ uint8_t *dst, __ubuf__ uint8_t *src,
                                            int blockCount, int blockLenB,
                                            asc_store_l2_cache_mode mode, int dstStrideB,
                                            int srcStrideB) {
  (void)mode;
  __cce_scalar::copy_ubuf_to_gm((__gm__ void *)dst, (__ubuf__ void *)src, (uint8_t)0,
                                (uint16_t)blockCount, TL910B_BYTES_TO_BLK(blockLenB),
                                TL910B_BYTES_TO_BLK(srcStrideB), TL910B_BYTES_TO_BLK(dstStrideB));
}

// G2: 跨 pipe 事件对。发射点：T.copy(gm->ub) 后 `asc_sync_notify(PIPE_MTE2, PIPE_S,
// static_cast<event_t>(0))`，首个消费点 asc_sync_wait(...)；ub->gm 前为 (PIPE_S, PIPE_MTE3)
// （见 gen_vec.asc / gen_ub2gm.asc）。910B 依据：AscendC dav_c220 官方搬运即
//   SetFlag<HardEvent::MTE2_S>/S_MTE3 成对使用（kernel_operator_data_copy_impl.h:167-168,
//   kernel_event.h:57 MTE2_S）；toolchain 自身样板 __clang_cce_aicore_functions.h:2152-2159。
// **必须用宏、不能用 inline 函数**：__cce_scalar::set_flag/wait_flag 是
//   `__attribute__((clang_builtin_alias(...))) void set_flag(...)`（cce_aicore_intrinsics.h:2072/2742），
//   CCE 前端校验要求 pipe 实参为字面枚举常量；实测（attempts/B/probe_setflag.sh）：
//     V1 字面量直调        COMPILE-OK
//     V2 inline 函数转调   COMPILE-FAIL "the 1st parameter maybe need a type 'pipe_t'"
//     V3/V4 宏（带/不带括号）COMPILE-OK
//     V5 static_cast<event_t>(3) COMPILE-OK（event 可非 0）
//   这也解释了 toolchain 的 __cce_set_flag(:2796) 为何 (void)p;(void)tp 丢弃实参、硬编码
//   PIPE_M->PIPE_V —— 它无法转发变量形 pipe 实参。我们靠宏保住真实的 MTE2->S / S->MTE3 语义。
#ifndef asc_sync_notify
#define asc_sync_notify(from, to, evt) __cce_scalar::set_flag(from, to, evt)
#endif
#ifndef asc_sync_wait
#define asc_sync_wait(from, to, evt) __cce_scalar::wait_flag(from, to, evt)
#endif

// G4: 910B(dav-2201) **标量面没有 rsqrtf**。一符号一 TU 实测（attempts/B/probe_math_symbols.sh）：
//     sqrt(float)=OK；rsqrtf / sqrtf / expf / exp / logf / log / powf / fabs / floorf /
//     ceilf / tanhf / __nv_rsqrtf / __nv_expf 全 MISSING；__builtin_sqrt FAIL（aicore 函数禁 double）。
// tilelang 的 T.rsqrt 在 vector 面落码为标量 `rsqrtf(float)`（见 gen_rsqrt.asc），故补最小 shim。
// 数值口径：1/sqrt(x)，走已验证的标量 sqrt；**不等价于 RCRS/RSQRT 近似指令**，
//   编译面等价、上卡后与 AscendC 原生 rsqrt 可能有 ~ulp 级差异（RESULT.md 已申报）。
__aicore__ inline float rsqrtf(float x) { return 1.0f / sqrt(x); }

#endif // TL910B_GAPB_APPLIED
'''

START = "// codegen: asc_copy_gm2ub_align(dst, src, sid, lenBurst, srcStride, dstStride, blockGroup, mode, nBurst, totalLen)"
END = "  copy_ubuf_to_gm(dst, src, config);\n}\n"
TAIL = "#endif // TL_PORT910B_NATIVE_TYPES (AIC adapters)"


def pip_compat() -> pathlib.Path:
    for root in sys.path:
        if not root:
            continue
        p = pathlib.Path(root) / "tilelang" / "src" / "tl_templates" / "ascend" / "port910b_compat.h"
        if p.is_file():
            return p
    raise SystemExit("FATAL: pip compat not found")


p = pip_compat()
s = p.read_text()
if MARK in s:
    print(f"ALREADY-PATCHED {p}")
else:
    for anchor in (START, END, TAIL):
        if anchor not in s:
            raise SystemExit(f"FATAL: anchor missing -> {anchor[:60]}")
    i, j = s.index(START), s.index(END) + len(END)
    s2 = s[:i] + GAPB.lstrip("\n") + s[j:]
    if not pathlib.Path(str(p) + ".preGapB").exists():
        pathlib.Path(str(p) + ".preGapB").write_text(p.read_text())
    p.write_text(s2)
    print(f"PATCHED {p}  (+{len(s2) - len(s)} bytes)")

t = p.read_text()
for sym in ("asc_sync_notify", "asc_sync_wait", "asc_store_l2_cache_mode",
            "asc_copy_gm2ub_align", "asc_copy_ub2gm_align",
            "#define asc_sync_notify", "__cce_scalar::copy_ubuf_to_gm", "float rsqrtf"):
    print(f"  present {sym:30s}: {t.count(sym)}")
if "pipe_t from" in t:
    print("  !! WARN: 旧 G2 inline 函数形态仍在（会导致 set_flag 前端校验 FAIL）")
