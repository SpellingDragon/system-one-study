// compat_patch_K.h — agent K 波 §12 缺口上报（P1-4 cube/矩阵三件 910B 编译合流）
//
// 【缺口】asc_fill_l1 未定义（trunk port910b_compat.h 971 行无此件）。
//
// 【现象】dW l0tr 主案在接口"动态 m"下编译失败：
//   CCE error: use of undeclared identifier 'asc_fill_l1'
// codegen 发射面（每次 asc_copy_gm2l1_nd2nz 之后，对本 token 块未填满的 L1 尾区补零）：
//   asc_fill_l1( (__cbuf__ uint16_t*)dst, (uint32_t)value,
//                { .repeat=..., .blk_num=(16 - 实际载入行数), .dst_gap=... } );
//   即 3 参：(dst: __cbuf__ 指针, value: uint32_t, params 聚合体 {repeat,blk_num,dst_gap})。
//
// 【触发条件（实测）】
//   - dW 归约轴 = 动态 token 数 m；末个 token 块 m % TC(=16) != 0 → L1 尾块需零填充补位。
//   - 静态且 m 整除 TC（原版 d_dw l0tr T=64/TT=16）→ 不发射 asc_fill_l1 → 编译 PASS（本容器实测）。
//   - gemm 前向走 alloc_shared(UB) 操作数（asc_copy_gm2ul_align 自带尾处理）→ 不触此件（故前向可编）。
//
// 【约束：不能用标量填充】补 __cbuf__(L1) 尾区的实现若写成标量 for(dst[i]=0) 会命中 CCE
//   "only __ubuf__ / __gm__ / local memory pointer can be dereferenced"（本波已在容器 pip 树试证）。
//   → 真件必须走官方 DMA/DataCopy-pad 类 builtin（如 __builtin_cce_fill_data / DataCopyPad，
//     配合 asc_set_copy_pad_val 家族），语义与位宽单位由 §12 owner + 上卡定夺。
//
// 【本波结论】dW 的 l0tr 主案 DSL 形态正确、且是 §12 尾填充的唯一缺口挡住编译；补 asc_fill_l1
//   （DMA-fill 版）后 l0tr 应即编过。前向件（gemm）不受此件影响、已编译合流。
#ifndef TL_PORT910B_COMPAT_GAP_K_H
#define TL_PORT910B_COMPAT_GAP_K_H

// 聚合体成员照抄 codegen 发射面；真实填充须委托官方 DMA-fill builtin，勿用标量循环。
struct asc_fill_l1_params_K {
  uint64_t repeat;
  uint64_t blk_num;
  uint64_t dst_gap;
};

// TODO(§12 owner + card): 用 __builtin_cce_fill_data / DataCopyPad 实现 L1 尾区按 value 补位。
// 下面签名仅为与 codegen 发射面对齐的占位声明，函数体故意留空标量（会命中 __cbuf__ 不可解引用，
// 故不可原样并入 trunk——仅示意 arity；编排者合入时须替换为 DMA-fill 实现）。
template <typename T>
__aicore__ inline void asc_fill_l1(__cbuf__ T *dst, uint32_t value, asc_fill_l1_params_K p) {
  // placeholder — MUST be replaced by DMA fill builtin before merging into §12 trunk.
  (void)dst; (void)value; (void)p;
}

#endif  // TL_PORT910B_COMPAT_GAP_K_H
