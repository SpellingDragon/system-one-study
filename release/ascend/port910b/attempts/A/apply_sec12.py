#!/usr/bin/env python3
"""A-3: inject compat §12 — shared.l1 gemm path 910B backend (Cal-layer assembly).
Idempotent via anchor TL910B_SEC12_L1GEMM."""
import pathlib

TL = pathlib.Path("$WORKSPACE_ROOT/tilelang/src/tl_templates/ascend/port910b_compat.h")
s = TL.read_text("utf-8")
if "TL910B_SEC12_L1GEMM" in s:
    print("ALREADY"); raise SystemExit

# 追加在文件最末（§11 GAP-C 块之后；自带守卫）
SEC12 = r'''
// ════════════════════════════════════════════════════════════════════════════
// §12 shared.l1 gemm path — 910B backend (P1-1 A案). Anchor: TL910B_SEC12_L1GEMM
// codegen `tl.ascend_gemm_l1` lowering（alloc_l1/alloc_l0c + transpose_B）发射面：
//   asc_set_gm2l1_nz_para / asc_copy_gm2l1_nd2nz / ascend_gemm_l1<...>(模板体)
//   asc_copy_l12l0a/b(_transpose) / asc_mmad / asc_lock/unlock / asc_copy_l0c2gm
// 建材（全部 raw 指针，无需 TPipe/LocalTensor；a11/a16/a17 实测签名）：
//   copy_gm_to_cbuf 原生 7 参   —— dav_c220 data_copy_impl.h:92 同款调法
//   LoadData2DL12L0ACal/B2Cal    —— asc/impl/basic_api/dav_c220/kernel_operator_mm_impl.h
//   MmadCal<T,U,S>(cc,ca,cb,MmadParams) —— bf16/fp16/fp32/int8 组合官方支持
//   FixpipeL0C2UBImpl + copy_ubuf_to_gm —— L0C→GM 两段式（fix 不支持 32bit 直出 GM）
// 上卡待证真点（首版 best-effort，注释处标 VERIFY）：
//   V1 gm2l1 参数单位（rowBytes/rows 的 32B 块换算）与 NZ padFuncMode
//   V2 L12L0 startIndex/repeatTimes 与 950 mStep/kStep 的映射方向
//   V3 l0c2gm 的 nSize/mSize/srcStride 分块步距
// ════════════════════════════════════════════════════════════════════════════
#ifndef TL_ASCEND_SIMT
#ifdef TL_PORT910B_NATIVE_TYPES
#ifndef TL910B_SEC12_L1GEMM
#define TL910B_SEC12_L1GEMM

#include "basic_api/dav_c220/kernel_operator_mm_impl.h"
#include "basic_api/dav_c220/kernel_operator_fixpipe_v2_impl.h"

// codegen 发射的 mode 枚举（存在性面，值域保守）
enum asc_unit_flag_mode : unsigned char { ASC_UF_MODE_DEFAULT = 0 };
enum asc_quant_mode : unsigned char { ASC_QUANT_MODE_DEFAULT = 0 };
enum asc_relu_pre_mode : unsigned char { ASC_RELU_MODE_OFF = 0, ASC_RELU_MODE_ON = 1 };
#ifndef ASC_LOCK_BLOCK
#define ASC_LOCK_BLOCK 0
#endif

// NZ 参数缓存：950 是有状态 set→copy 模型，保持形态
struct Tl910bNzState { uint16_t c0_stride = 16; int enable = 1; };
static Tl910bNzState tl910b_gm2l1_nz;

__aicore__ inline void asc_set_gm2l1_nz_para(int en, int rchg, uint16_t c0_stride, int pad) {
  (void)rchg; (void)pad;
  tl910b_gm2l1_nz.enable = en;
  tl910b_gm2l1_nz.c0_stride = c0_stride;
}
__aicore__ inline void asc_set_l0c_copy_nz_para(int en, int a, int b) { (void)en; (void)a; (void)b; }
__aicore__ inline void asc_set_copy_pad_val(int v) { (void)v; }

// GM→L1（ND/DN 首版同路：行块连续搬运。V1：NZ 语义在 padFuncMode 与块步距上卡定）
__aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ uint8_t *dst, __gm__ uint8_t *src,
                                            int rowBytes, asc_load_l2_cache_mode mode,
                                            int nRows, int cols, int pad1, int pad2) {
  (void)mode; (void)cols; (void)pad1; (void)pad2;
  uint16_t const blk = (uint16_t)((unsigned)rowBytes >> 5);      // VERIFY V1: 32B 单位
  copy_gm_to_cbuf((__cbuf__ void *)dst, (__gm__ void *)src, (int8_t)0,
                  (uint16_t)nRows, blk, blk, blk);
}
__aicore__ inline void asc_copy_gm2l1_dn2nz(__cbuf__ uint8_t *dst, __gm__ uint8_t *src,
                                            int rowBytes, asc_load_l2_cache_mode mode,
                                            int nRows, int cols, int pad1, int pad2) {
  asc_copy_gm2l1_nd2nz(dst, src, rowBytes, mode, nRows, cols, pad1, pad2);  // VERIFY V1
}

// L1→L0A/L0B 装填（gemm.h 模板体 8 参发射：dst,src,sid,kStartFrac,mStep,kStep,srcStrideBlk,dstStrideBlk）
template <typename T>
__aicore__ inline void asc_copy_l12l0a(__ca__ T *dst, __cbuf__ T *src, int sid, int kStart,
                                       int mStep, int kStep, int srcStride, int dstStride) {
  (void)kStep;
  // VERIFY V2：startIndex=行块起点(kStart 语义按发射名取 m 起点)，repeatTimes=mStep
  AscendC::LoadData2DParams p((uint16_t)kStart, (uint8_t)mStep, (uint16_t)srcStride,
                              (uint8_t)sid, (uint16_t)dstStride, false, 0);
  AscendC::LoadData2DL12L0ACal(dst, src, p);
}
template <typename T>
__aicore__ inline void asc_copy_l12l0b(__cb__ T *dst, __cbuf__ T *src, int sid, int kStart,
                                       int mStep, int kStep, int srcStride, int dstStride) {
  (void)kStep;
  AscendC::LoadData2DParams p((uint16_t)kStart, (uint8_t)mStep, (uint16_t)srcStride,
                              (uint8_t)sid, (uint16_t)dstStride, false, 0);
  AscendC::LoadData2DL12L0BCal(dst, src, p);
}
template <typename T>
__aicore__ inline void asc_copy_l12l0b_transpose(__cb__ T *dst, __cbuf__ T *src, int mStart,
                                                 int kStart, int mStep, int kStep, int srcStride,
                                                 int dstStride) {
  (void)kStart; (void)kStep;
  AscendC::LoadData2DParams p((uint16_t)mStart, (uint8_t)mStep, (uint16_t)srcStride,
                              (uint8_t)0, (uint16_t)dstStride, true /*ifTranspose*/, 0);
  AscendC::LoadData2DL12L0BCal(dst, src, p);   // VERIFY V2: transpose 走 ifTranspose 形
}

// cube 计算：910B 官方 Cal（MmadParams(m,n,k,unitFlag,cmatrixSource,cmatrixInitVal)）
template <typename CT, typename AT, typename BT>
__aicore__ inline void asc_mmad(CT *cc, AT *ca, BT *cb, uint16_t m, uint16_t k, uint16_t n,
                                int unit_flag_ctrl, bool gemv_ctrl, bool btbuf_ctrl,
                                bool zero_c_ctrl) {
  (void)gemv_ctrl; (void)btbuf_ctrl;
  AscendC::MmadParams p(m, n, k, (uint8_t)unit_flag_ctrl, false /*cmatrixSource*/,
                        zero_c_ctrl /*cmatrixInitVal: 1=清零起算*/);
  AscendC::MmadCal(cc, ca, cb, p);
}

// L0C→GM：fix 不支持 32bit 直 GM（native static_assert 实证）→ L0C→UB→GM 两段流水
// 发射形态：asc_copy_l0c2gm(gm, cc, mRows?, nCols?, m?, n?, store_mode, uf, quant, relu, a,b,c,d)
__aicore__ inline void asc_copy_l0c2gm(__gm__ float *dst, __cc__ float *src, int mRows,
                                      int nCols, int srcM, int srcN,
                                      asc_store_l2_cache_mode smode, asc_unit_flag_mode uf,
                                      asc_quant_mode q, asc_relu_pre_mode r, int a, int b,
                                      int c, int d) {
  (void)srcM; (void)srcN; (void)smode; (void)uf; (void)q; (void)r; (void)a; (void)b; (void)c; (void)d;
  // VERIFY V3：16 行分块（L0C 行粒度=16），行内 nCols 连续
  uint16_t const nBlk = (uint16_t)((unsigned)nCols >> 4);       // 16-col blocks? 单位上卡定
  __ubuf__ float staging[16 * 64];                               // 单块 ≤ 16x64 f32
  for (int rb = 0; rb < (mRows >> 4); ++rb) {
    AscendC::FixpipeParamsV220 fp((uint16_t)nCols, 16, (uint16_t)nCols, 0, false);
    AscendC::FixpipeL0C2UBImpl(staging, src + rb * 16 * nCols, fp);
    uint16_t const blk = (uint16_t)((unsigned)(nCols * 4) >> 5);
    copy_ubuf_to_gm((__gm__ void *)(dst + rb * 16 * nCols), (const __ubuf__ void *)staging,
                    (int8_t)0, (uint16_t)16, blk, blk, blk);
  }
}

#endif  // TL910B_SEC12_L1GEMM
#endif  // TL_PORT910B_NATIVE_TYPES
#endif  // !TL_ASCEND_SIMT (§12)
'''
s = s + SEC12
TL.write_text(s, "utf-8")
print("§12 injected:", s.count("TL910B_SEC12_L1GEMM"), "anchors; lines:", s.count(chr(10)))
