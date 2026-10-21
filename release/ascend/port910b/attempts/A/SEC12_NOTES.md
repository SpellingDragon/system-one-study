# A 案（shared.l1 gemm 数值路）侦察卷宗 — 2026-10-09

## 已定的完整事实链（全部实测/原文证据）
1. **触发**：A/B scope `shared.l1` + `transpose_B=True` → codegen `_lower_l1()`（gemm_mad.py:161-166）发 `ascend_gemm_l1<M,K,N,tile,trans,dtype>` 模板路（`__cube__` 单工 kernel，无 mix——发射面 9 符号，见 a2_probe EMISSION SURFACE）。
2. **建材真名**（官方 Cal 层解包所得，a29c/a30c）：
   - GM→L1：native `copy_gm_to_cbuf(dst,src,sid,blockCount,blockLen,srcStride,dstStride)`（32B 单位；§12 已用）
   - L1→L0A/B：`load_cbuf_to_ca(dst,src,startIndex,repeatTimes,srcStride,...)` / `load_cbuf_to_cb`（**bisheng 自带头无通用形声明**——是 `CCE_SCALAR()` 的 clang builtin，需 extern 声明或找到其声明头）；另有 **`load_gm_to_ca/cb`（GM→L0 直载）** 可走捷径
   - cube：官方形 `mad(c,a,b,m,k,n,unitFlag,kDirectionAlign,cmatrixSource,cmatrixInitVal)`（10 参，注意 ≠ §10 自拼形——多 kDirAlign 位）
   - L0C→UB：Fixpipe Cal 的 native 件待最后一层（a30 grep 未中——在 load_cc family 或 `fix_matrix_cc_to_cbufubuf_dualout`）
   - Params 结构：LoadData2DParams{startIndex,repeatTimes,srcStride,sid,dstGap,ifTranspose,addrMode}；MmadParams(m,n,k,unitFlag,cmatrixSource,cmatrixInitVal)；FixpipeParamsV220(nSize,mSize,srcStride,dstStride,reluEn)
3. **框架 include 的组装难题**（本轮所停）：
   - 单拎 `dav_c220/*_impl.h` → 缺 LocalTensor（interface 前置）
   - `kernel_operator.h` 全链 → `kernel_operator_dump_tensor_impl.h:605` illegal initializer（`ASCENDC_DUMP` 被 kernel_macros.h:118 默认定义=1，dump 模板体解析出错——官方 cmake 有自己的 -D/-U 矩阵未穷尽）
   - 破解方向（下轮）：**绕框架，直接 extern 声明 `__builtin_cce_*` 系列**（官方 Cal 即其薄包装，签名可从 builtin 定义反查 `CCE_SCALAR` 宏展开=`__builtin_cce_##name`）

## 主线守护态
§12 已加 `TL910B_SEC12_ENABLE` 门（默认关）；B/C/e2e_cube 回归全绿后才入账。

## 上卡验证点（V1/V2/V3 已标 §12 注释）
NZ 布局单位、L12L0 映射方向、l0c2gm 步距——数值真后 gemm rel_err 判据转绿。

## 追加（同日深夜）：硬件边界负结果 + v2 native 化
- **b7 探针**：compat 链内（无框架）`load_cbuf_to_ca/load_gm_to_ca/mad(10参官方形)/fix_matrix_cc_to_cbufubuf_dualout/copy_gm_to_cbuf` 5/5 可编 → §12 v2 全 native，框架 include 废弃（其 dump 头/`__aicore__` 宏互斥、ASCENDC_DUMP 默认开等多重环境依赖不值得追）。
- **b10 负结果**：官方 `FixpipeL0C2UBImpl` 在 2201 = **assert(false)——L0C→UB 硬件不支持**；b11：`FixpipeL0C2GMImpl` 存在但体内无直连 builtin → 正解应为 L0C→L1(fix 16bit 或 s32 系)→MTE3(copy_cbuf... _to_gm 方向名再考古)→GM。
- CCE 禁可变成体全局（a2k3）→ set_gm2l1_nz_para 无状态化。
- **本波收**：l0c2gm 暂 stub（编译链先通），输出三段路 = A-4 下一轮 + 上卡清单同批。

## A-3 收官（同日破局）
- **A2-GEMM-L1-PASS 2776B**：shared.l1 模板路编译贯通。全部官方形定档：
  - `copy_gm_to_cbuf` 8 参（data_copy_impl.h:92：sid,blockCount,blockLen,srcStride,dstStride,pad_t）
  - `load_cbuf_to_ca/cb` **9 参**（mm_impl.h:24：startIndex,repeatTimes,srcStride,dstGap,sid,transpose,**addr_cal_mode_t**）
  - mad 10 参官方形已入；MX/blockscaled 段 NATIVE 罩除（gemm.h，fp8 类型 950-only）
  - l0c2gm=stub（b10 硬件边界：L0C→UB 不支持，正解三段 L0C→L1→MTE3→GM 待 A-4/上卡）
- 主线回归（§12 无门控合入后）：cube/B×2/GDN 全绿。

## A-4 破局（同日续）：L0C→GM 真件 = `copy_matrix_cc_to_gm`（12 参变长 alias builtin）
- 此前两轮否定检索（fix_matrix_cc_to_gm 名称错、FixpipeL0C2UB=assert-false 只封了 Cal 包装层）——真件在 cce_aicore_intrinsics.h:1020 `__builtin_cce_copy_matrix_cc_to_gm`，位序照抄 fixpipe_v2_impl.h:433 官方 Cal。同族 `copy_matrix_cc_to_cbuf/ubuf` 亦在（输出全向可用）。
- **A2-COMPILE-PASS（tilelang 全管线）**：gemm 链 GM→L1→L0A/B→mad→L0C→GM 编译面完全贯通；主线 4/4 无回归；三容器已同步。
- 上卡数值证真清单：V1（gm2l1 32B 单位/块）·V2（L12L0 位序）·V3（l0c2gm stride 单位与位映射）。
