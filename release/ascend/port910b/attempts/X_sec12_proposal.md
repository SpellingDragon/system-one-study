# X_sec12_proposal — §12 shared.l1（910B cube intrinsic 路）修正提案【**不落地**】

配套证据：`attempts/X_FRACTAL_EVIDENCE.md`（本提案的每一条改动都由该文件的 §2/§5/§6/§7 支撑）
自证：`attempts/X_probe_proposal.py` —— 提案 wrapper 原文在 **dav-2201 compile-only 8/8 rc=0**
纪律：本文件**只是 diff**。trunk `src/tl_templates/ascend/port910b_compat.h` 一行未改（Y 代理在改 `ascend/kernels/*`、`run_numerics.py`、`compile_all_asc.py`、`card_run.sh`，本窗严禁触碰产品码）。

---

## 0. 一句话结论

**建议改，但分三步走且第一步必须在真机判别后才走第二步**：
- **P0（止血/判别件，建议先上卡）**：`asc_copy_l12l0a/b` 的 sid 槽不再吃 IR `m_start`，钉 `(uint8_t)0`。改动 1 处 × 2 件，纯消歧。
- **P1（结构性必要条件）**：`asc_copy_gm2l1_nd2nz` 从 `copy_gm_to_cbuf` 8 参（ND 线性块拷贝）改走 2201 官方 ND2NZ 件 `copy_gm_to_cbuf_multi_nd2nz_b8/b16/b32s`（11 参），把被丢掉的 IR `d_value` 与 NZ c0 pitch 真正落到参数上。
- **P2（与 P1 成套）**：`asc_copy_l12l0a/b` 改为官方 V1 模板形态 —— **二维起点折进 src 指针** + `repeatTimes = k_step`（K/C0 向）+ **M 向外层 for 拆趟**。

P1 与 P2 **不可只做其一**：P1 产出 NZ 而 P2 仍按错位槽读 → 依旧非法访问；P2 正确而 P1 仍产 ND → 读到的是被当分形块解释的 ND 字节，数值必错（不再是异常，而是静默错值，更难查）。

---

## 1. P0 —— sid 止血（等价于 E-3 判别实验的载荷）

```diff
--- a/src/tl_templates/ascend/port910b_compat.h
+++ b/src/tl_templates/ascend/port910b_compat.h
@@ -891,9 +891,11 @@ template <typename T>
 __aicore__ inline void asc_copy_l12l0a(__ca__ T *dst, __cbuf__ T *src, int sid, int kStart,
                                        int mStep, int kStep, int srcStride, int dstStride) {
   (void)kStep;
-  // VERIFY V2：startIndex=kStart(发射行块起点)、repeatTimes=mStep、srcStride 单位待卡定
+  // P0 止血：第 7 槽是 4-bit SMMU sid（sema 实证 range [0,15]，且**运行期值不检查**）。
+  // codegen 该位传的是 IR m_start（row16 块起点，可 0..127），非 sid —— 越界即非法 L1 分区
+  // → aicore 507015。此处先钉 0（官方 Matmul 也恒 sid=0）。P2 落地后 m_start 改由指针承担。
   load_cbuf_to_ca(dst, src, (uint16_t)kStart, (uint8_t)mStep, (uint16_t)srcStride,
-                  (uint16_t)dstStride, (uint8_t)sid, false,
+                  (uint16_t)dstStride, (uint8_t)0, false,
                   (__cce_scalar::addr_cal_mode_t)0);  // 官方 9 参位序（mm_impl.h:35）
 }
@@ -900,8 +902,8 @@ template <typename T>
 __aicore__ inline void asc_copy_l12l0b(__cb__ T *dst, __cbuf__ T *src, int sid, int kStart,
                                        int mStep, int kStep, int srcStride, int dstStride) {
   (void)kStep;
   load_cbuf_to_cb(dst, src, (uint16_t)kStart, (uint8_t)mStep, (uint16_t)srcStride,
-                  (uint16_t)dstStride, (uint8_t)sid, false,
+                  (uint16_t)dstStride, (uint8_t)0, false,
                   (__cce_scalar::addr_cal_mode_t)0);  // 对称 LoadData2DL12L0BCal
 }
```

**判别价值**（这是本提案最要紧的用途）：
- 真机上 P0-only 后若 `507015` **消失**（哪怕数值仍错）→ sid 越界是异常的直接触发者，考古链闭合，继续 P1+P2 修数值；
- 若 `507015` **仍在** → sid 非触发者，主嫌收敛到 GM→L1 的 ND/NZ 布局（P1），可据此决定后续投入方向。
一次真机窗即可二选一，成本 ≈ 0.5 页 diff + 1 次编译。

**风险**：`sid=0` 意味着不做 L1 分区（SMMU）。若 §12 将来要分区并发，必须另立显式形参（不能借 m_start 混用）。

---

## 2. P1 —— GM→L1 走官方 2201 ND2NZ 件

```diff
--- a/src/tl_templates/ascend/port910b_compat.h
+++ b/src/tl_templates/ascend/port910b_compat.h
@@ -866,17 +866,30 @@ __aicore__ inline void asc_set_copy_pad_val(int v) { (void)v; }
 // GM→L1（ND/DN 首版同路：行块连续搬运。V1：NZ 语义在 padFuncMode 与块步距上卡定）
 template <typename DT, typename ST>
 __aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ DT *dst, __gm__ ST *src,
                                             int rowBytes, asc_load_l2_cache_mode mode,
                                             int nRows, int cols, int pad1, int pad2) {
-  (void)mode; (void)cols; (void)pad1; (void)pad2;
-  uint16_t const blk = (uint16_t)((unsigned)rowBytes >> 5);  // VERIFY V1: 32B 单位
-  // 官方 8 参（dav_c220 data_copy_impl.h:92）：sid,blockCount,blockLen,srcStride,dstStride,pad
-  copy_gm_to_cbuf((__cbuf__ void *)(uintptr_t)dst, (__gm__ void *)(uintptr_t)src, (int8_t)0,
-                  (uint16_t)nRows, blk, blk, blk, (pad_t)0);
+  // P1（考古定论，X_FRACTAL_EVIDENCE §6）：copy_gm_to_cbuf 8 参 = 官方 DataCopyGM2L1Impl，
+  //   只做 ND 线性块拷贝；910B 官方 Matmul 写 L1 只有 ND2NZ/NZ2NZ 两路，V1 load_cbuf_to_ca 的
+  //   startIndex 文档语义 = Fractal matrix ID → L1 必须是 NZ。
+  // 字段单位取自官方文档 + 地址模型（kernel_check_data_copy_overflow.h:505-520）：
+  //   dValue/srcDValue/srcNdMatrixStride 单位=元素；dstNzC0Stride/dstNzNStride 单位=32B 块；
+  //   dstNzMatrixStride 单位=元素。dstNzC0Stride 取值照抄 data_copy_wrapper_nd.h:93。
+  (void)mode; (void)pad1; (void)pad2;   // l2_cache_ctrl 在 2201 ND2NZ 件无对应槽
+  uint16_t const gCol = (uint16_t)((unsigned)rowBytes / (unsigned)sizeof(ST)); // srcDValue：GM 行宽(元素)
+  uint16_t const c0s  = (uint16_t)((((unsigned)nRows + 15u) / 16u) * 16u);      // dstNzC0Stride(32B 块)
+  if constexpr (sizeof(ST) == 1) {
+    copy_gm_to_cbuf_multi_nd2nz_b8((__cbuf__ int8_t *)(uintptr_t)dst, (__gm__ int8_t *)(uintptr_t)src,
+        (int8_t)0, (uint16_t)1, (uint16_t)nRows, (uint16_t)cols,
+        (uint16_t)0, gCol, c0s, (uint16_t)1, (uint16_t)0);
+  } else if constexpr (sizeof(ST) == 2) {
+    copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ bfloat16_t *)(uintptr_t)dst, (__gm__ bfloat16_t *)(uintptr_t)src,
+        (int8_t)0, (uint16_t)1, (uint16_t)nRows, (uint16_t)cols,
+        (uint16_t)0, gCol, c0s, (uint16_t)1, (uint16_t)0);
+  } else {
+    copy_gm_to_cbuf_multi_nd2nz_b32s((__cbuf__ float *)(uintptr_t)dst, (__gm__ float *)(uintptr_t)src,
+        (int8_t)0, (uint16_t)1, (uint16_t)nRows, (uint16_t)cols,
+        (uint16_t)0, gCol, c0s, (uint16_t)1, (uint16_t)0);
+  }
 }
 template <typename DT, typename ST>
 __aicore__ inline void asc_copy_gm2l1_dn2nz(__cbuf__ DT *dst, __gm__ ST *src,
                                             int rowBytes, asc_load_l2_cache_mode mode,
                                             int nRows, int cols, int pad1, int pad2) {
-  asc_copy_gm2l1_nd2nz(dst, src, rowBytes, mode, nRows, cols, pad1, pad2);  // VERIFY V1
+  // 本提案**不改** dn2nz：2201 另有 copy_gm_to_cbuf_multi_dn2nz（cce_aicore_intrinsics.h:980），
+  //   其 Nd2NzParams 同族字段文档为 dn 版（srcDnMatrixStride/srcDValue）。语义与 nValue/dValue
+  //   互换关系需一次 compile+真机判别（X_FRACTAL_EVIDENCE §8-E1 的 dn 扩展），先保持 ND 路。
+  asc_copy_gm2l1_nd2nz(dst, src, rowBytes, mode, nRows, cols, pad1, pad2);  // VERIFY U5
 }
```

**参数字典（照抄官方，逐项可溯源）**

| 槽 | 本提案取值 | 官方出处 | 单位 |
|---|---|---|---|
| sid | `0` | `data_copy_impl.h:275` 传字面 0 | — |
| ndNum | `1` | `data_copy_wrapper_nd.h:75` | 矩阵块数 |
| nValue | `nRows` ← IR `n_value`(=src 行数) | `wrapper_nd.h:76` `=height` | 元素（行）|
| dValue | `cols` ← IR `d_value`（现被 `(void)` 丢）| `wrapper_nd.h:77` `=width` | 元素 |
| srcNdMatrixStride | `0` | `wrapper_nd.h:78`（默认入参 0）| 元素 |
| srcDValue | `gCol = rowBytes/sizeof(ST)` | `wrapper_nd.h:79` `=gCol` | 元素 |
| dstNzC0Stride | `Ceil(nRows,16)*16` | `wrapper_nd.h:93` 同式 | **32B 块** |
| dstNzNStride | `1` | `wrapper_nd.h:96` | 32B 块 |
| dstNzMatrixStride | `0` | `wrapper_nd.h:97` | 元素 |

**已知边界（提案未覆盖，须在实现时补守卫或留 ICHECK）**
- 全部字段是 `uint16_t`（`kernel_struct_data_copy.h:185-192`）→ `gCol > 65535` 时官方 2201 自己**逐行拆趟**（`wrapper_nd.h:99-107`：`nValue=1; srcDValue=width; for i<height DataCopy(dst[i*c0Size_], …)`）。本提案是单发形态，K 很宽的 tile 需照此加兜底，否则 sema 的 `[0,15]` 之类区间不报错但值被截。
- `dstNzC0Stride` 与 P2 的 `srcStride` 是**同一物理量的两种单位**：`srcStride(分形块) = dstNzC0Stride(32B 块) / 16`（bf16 一块 = 16×32B=512B）。若 codegen 侧 L1 buffer 的 `outer1` extent 与这里手算的 `Ceil(nRows,16)` 不一致，P1/P2 会互相对不上 → **实现时加一处 static 一致性检查或日志**（这是本路最容易复发的坑）。

---

## 3. P2 —— L1→L0A/B 改官方 V1 模板形态（折指针 + K 向 repeat + M 向拆趟）

```diff
--- a/src/tl_templates/ascend/port910b_compat.h
+++ b/src/tl_templates/ascend/port910b_compat.h
@@ -884,24 +897,42 @@ __aicore__ inline void asc_copy_gm2l1_dn2nz(...
-// P1-1d 修复（2026-10-10）：官方 LoadData2DL12L0ACal（mm_impl.h:32/35）发 **9 参**
-//   (dst,src,startIndex,repeatTimes,srcStride,dstGap,sid,transpose(0/1),inc)。
-// 旧 trunk 只发 7 参……（略）
 template <typename T>
-__aicore__ inline void asc_copy_l12l0a(__ca__ T *dst, __cbuf__ T *src, int sid, int kStart,
+__aicore__ inline void asc_copy_l12l0a(__ca__ T *dst, __cbuf__ T *src, int mStart, int kStart,
                                        int mStep, int kStep, int srcStride, int dstStride) {
-  (void)kStep;
-  load_cbuf_to_ca(dst, src, (uint16_t)kStart, (uint8_t)mStep, (uint16_t)srcStride,
-                  (uint16_t)dstStride, (uint8_t)sid, false,
-                  (__cce_scalar::addr_cal_mode_t)0);  // 官方 9 参位序（mm_impl.h:35）
+  // P2：dav-2201 无 2DV2，V2 语义只能按官方 V1 模板降级（X_FRACTAL_EVIDENCE §5.2 四条硬规则）：
+  //   (1) 二维起点折进指针，startIndex 恒 0；(2) repeatTimes 沿 K/C0 = kStep；
+  //   (3) srcStride = L1 的 row16 块数（= IR src_stride，口径本就一致）；
+  //   (4) M 向 mStep 趟外层循环，每趟 src 前进 1 个分形块、dst 前进 kStep 个分形块。
+  // 形参 `sid` → `mStart` 更名：codegen 第 3 实参是 IR m_start（copy.cc:457），旧名是错位之源。
+  (void)dstStride;  // VERIFY U2：假定 L0 目的紧凑（pitch == kStep）；非紧凑须改用 dstGap 或再拆趟
+  int const blkElems = 16 * (int)(32 / (int)sizeof(T));  // 分形块元素数（bf16: 256；16×32B 恒等）
+  __cbuf__ T *base = (__cbuf__ T *)(uintptr_t)src + (int64_t)(kStart * srcStride + mStart) * blkElems;
+  __ca__ T *dbase = (__ca__ T *)(uintptr_t)dst;
+  if (kStep == 1) {                      // 官方同款退化：repeat 改沿 M、srcStride=1
+    load_cbuf_to_ca(dbase, base, (uint16_t)0, (uint8_t)mStep, (uint16_t)1, (uint16_t)0,
+                    (uint8_t)0, false, (__cce_scalar::addr_cal_mode_t)0);
+  } else {
+    for (int i = 0; i < mStep; ++i) {
+      load_cbuf_to_ca(dbase + (int64_t)i * kStep * blkElems,
+                      base + (int64_t)i * blkElems,
+                      (uint16_t)0, (uint8_t)kStep, (uint16_t)srcStride, (uint16_t)0,
+                      (uint8_t)0, false, (__cce_scalar::addr_cal_mode_t)0);
+    }
+  }
 }
 template <typename T>
-__aicore__ inline void asc_copy_l12l0b(__cb__ T *dst, __cbuf__ T *src, int sid, int kStart,
+__aicore__ inline void asc_copy_l12l0b(__cb__ T *dst, __cbuf__ T *src, int mStart, int kStart,
                                        int mStep, int kStep, int srcStride, int dstStride) {
-  (void)kStep;
-  load_cbuf_to_cb(dst, src, (uint16_t)kStart, (uint8_t)mStep, (uint16_t)srcStride,
-                  (uint16_t)dstStride, (uint8_t)sid, false,
-                  (__cce_scalar::addr_cal_mode_t)0);  // 官方 9 参位序（对称 LoadData2DL12L0BCal）
+  // P2-B：官方 L0B 非转置路 repeatTimes=blockUseN、srcStride=Ceil(bL1K,16)；转置由 L1 写侧
+  //   dn2nz 承担而非 load 侧 transpose 位（load_to_l0b_load2d.h:38-92）。
+  //   本轮 §12 的 B 侧几何是 K-major 单趟形态，先按 A 式同款降级；N 向拆趟留 U3/E-4 卡定。
+  (void)mStart; (void)mStep; (void)dstStride;  // VERIFY U3
+  int const blkElems = 16 * (int)(32 / (int)sizeof(T));
+  __cbuf__ T *base = (__cbuf__ T *)(uintptr_t)src + (int64_t)(kStart * srcStride) * blkElems;
+  load_cbuf_to_cb((__cb__ T *)(uintptr_t)dst, base, (uint16_t)0, (uint8_t)kStep,
+                  (uint16_t)srcStride, (uint16_t)0, (uint8_t)0, false,
+                  (__cce_scalar::addr_cal_mode_t)0);
 }
```

**不改的部分（证据不足，明说不改）**
- `asc_copy_l12l0a_transpose` / `asc_copy_l12l0b_transpose`：hivmc 名字表显示 transpose 件是**双 cfg 字**打包（`_load_cbuf_to_ca_transpose ... m m`），与 V1 非转置件不同族；本窗未取到 2201 该件的逐槽 sema 表（X_FRACTAL_EVIDENCE §8-U5/E-5）。**保持原样**，只把 `VERIFY` 注释指向 E-5。
- `asc_set_gm2l1_nz_para`（空实现）：P1 已自算 `dstNzC0Stride`，无需激活它；若要改用 codegen 传来的 IR `nz_c0_stride`（args[10]），则要给 §12 件加形参 → **属接口扩张，需 codegen 同窗改动**，不在本提案（本窗禁改 codegen/kernels）。
- `asc_fill_l1`（§12 补件）：与 P1 的 pad 区有交互（U4），本提案不动。

---

## 4. compile-only 自证（本窗实测）

```bash
cd release/ascend/port910b/attempts && docker cp X_probe_proposal.py cann910b-k:/tmp/
docker exec cann910b-k bash -c 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; cd /tmp && python3 /tmp/X_probe_proposal.py'
```
```
── all_in_one                 PASS (rc=0)
── P1_gm2l1_nd2nz             PASS (rc=0)   # bf16 → _b16
── P1_gm2l1_nd2nz_fp32        PASS (rc=0)   # float → _b32s
── P1_gm2l1_nd2nz_fp8         PASS (rc=0)   # int8 → _b8
── P2_l12l0a                  PASS (rc=0)   # mStep=2,kStep=4 → 拆趟循环版
── P2_l12l0a_kstep1           PASS (rc=0)   # kStep==1 退化分支
── P2_l12l0b                  PASS (rc=0)
── P0_bleed_sid0              PASS (rc=0)
```
（编译命令与 §1 同；产物 `/tmp/xpp_*.o`。**compile-only 只证语法/sema 面可编，不证数值** —— 数值判据必须走真机 E-1/E-2/E-3。）

---

## 5. 落地路线建议（给编排者的排序，非本窗执行）

1. **本窗**：仅归档本提案 + 证据（已完成）。
2. **下一真机窗（最小判别）**：落 P0（2 行改动，`asc_copy_l12l0a/b` 各 1 处）→ 跑 gemm_l1 单窗 → 记 `507015` 是否消失。产出即 E-3 结论，直接决定投入方向。
3. **随后**：落 P1+P2（成套），先跑 **E-1**（16×16 GM→L1→GM 回读，证 NZ 位型），再跑 **E-2/E-4**（16×16×16 mad 对拍 + L0A 步进标定）。E-1 独立于 L0 侧，是最低风险的第一次真机验证。
4. **禁止**：只落 P1 或只落 P2（见 §0）；禁止在未跑 E-1 前用 gemm_l1 端到端当首验（异常与数值错会混在一起，回到 P1-1d 的老路）。
5. 若 E-3 显示 P0 无效且 E-1 显示 ND2NZ 位型正确 → 主嫌转移到 L1 侧 `srcStride` 与 `dstNzC0Stride` 的不同源（§2 末的一致性坑），届时先补 static 检查再谈其它。

## 6. 与 §12 现有 VERIFY 标记的关闭关系

| 旧标记 | 本提案处置 |
|---|---|
| `V1 gm2l1 参数单位（rowBytes/rows 的 32B 块换算）与 NZ padFuncMode` | **可关闭为定论**：单位=32B 块（overflow 模型），且 GM→L1 必须换 ND2NZ 件；padFuncMode 一支作废（2201 ND2NZ 件无该槽）。 |
| `V2 L12L0 startIndex/repeatTimes 与 950 mStep/kStep 的映射方向` | **方向定论**（startIndex 恒 0+折指针；repeatTimes=kStep；M 向拆趟），剩 dst 侧步进留 `U2/U3`。 |
| `V3 l0c2gm 的 nSize/mSize/srcStride 分块步距` | 本提案不涉及（另一件，未取证）。 |
| `V1'/V2' 各件注释` | 建议随 P0/P1/P2 落地时同步更新为 `U#`（与 X_FRACTAL_EVIDENCE §8 编号一致），避免 VERIFY 语义漂移。 |
