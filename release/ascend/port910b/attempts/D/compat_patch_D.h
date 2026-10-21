// attempts/D/compat_patch_D.h — 910B port910b_compat.h 缺口补丁（GAP-D，dW 权重梯度波）
// 追加位置：pip 运行副本 <tilelang>/src/tl_templates/ascend/port910b_compat.h **文件末尾**。
// 只在 §12 之后追加，不动任何既有行；由 attempts/D/env_setup_D.sh [2b] 幂等装配
// （每次先 cp 真源重置，再按 marker 追加）。
//
// ── G-D1 ─────────────────────────────────────────────────────────────────────
// 缺口：`asc_copy_l12l0a_transpose` **未定义**。
// 触发形态：DSL `T.copy(l1_tile, l0a_tile, transpose=True)`（L1→L0A 转置装填）。
//   codegen 发射点 src/ascend/codegen/codegen_ascend.cc:1563-1567 —— transpose=1 且
//   is_l0a ⇒ 名字 "asc_copy_l12l0a_transpose"（arity 8）；compat §12 只给了
//   非转置 asc_copy_l12l0a(:886) 与 L0B 的 asc_copy_l12l0b_transpose(:902)。
// 实测：cann910b-d / d_dw_910b.py --variant l0tr ⇒
//   tl_kernel.asc:14:5: error: use of undeclared identifier 'asc_copy_l12l0a_transpose';
//   did you mean 'asc_copy_l12l0b_transpose'?      （编译器自己都提示了同族件存在）
// 910B 原生依据（同族件在官方 Cal 层就在）：
//   asc/impl/basic_api/dav_c220/kernel_operator_mm_impl.h:132
//     load_cbuf_to_ca_transpose(dst, src, loadDataParam.startIndex, repeatTimes,
//                               srcStride, dstGap, inc, dstFracGap);
//   asc/impl/basic_api/dav_c220/kernel_operator_cube_others_impl.h:319-322
//     LoadCbufToCaTranspose(dst, src, uint16_t indexID, uint8_t repeatTime,
//                           uint16_t srcStride, uint16_t dstStride, bool addrmode,
//                           uint16_t dstFracStride)
// 位序映射依据（codegen 侧槽名）：src/ascend/op/builtin.h:247-249
//   ascend_load_cbuf_to_ca(dst, src, mStartPosition, kStartPosition, mStep, kStep,
//                          srcStride, dstStride, transpose)
// ⇒ 与 compat 既有 asc_copy_l12l0b_transpose 取同一选择（indexID=mStart、
//   repeatTime=mStep），**保持族内一致**；kStep/dstStride 的去留标 VERIFY V2'，
//   上卡对拍（d_dw_golden.py P0/P1）若转置方向对而数值散，则先查这一位序。
#ifndef TL_PORT910B_COMPAT_GAP_D_H
#define TL_PORT910B_COMPAT_GAP_D_H

#ifndef TL_ASCEND_SIMT  // 与 §12 同域：仅 910B native 面（非 SIMT）装配
#ifdef TL_PORT910B_NATIVE_TYPES

template <typename T>
__aicore__ inline void asc_copy_l12l0a_transpose(__ca__ T *dst, __cbuf__ T *src, int mStart,
                                                 int kStart, int mStep, int kStep, int srcStride,
                                                 int dstStride) {
  (void)kStart; (void)kStep;
  (void)dstStride;  // VERIFY V2'：与 asc_copy_l12l0b_transpose 同款取舍
  load_cbuf_to_ca_transpose(dst, src, (uint16_t)mStart, (uint8_t)mStep, (uint16_t)srcStride,
                            (uint16_t)dstStride, false, (uint16_t)0);  // VERIFY V2' 位序
}

#endif  // TL_PORT910B_NATIVE_TYPES
#endif  // !TL_ASCEND_SIMT
#endif  // TL_PORT910B_COMPAT_GAP_D_H
