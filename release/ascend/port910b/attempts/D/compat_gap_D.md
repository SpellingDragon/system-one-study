# compat_gap_D.md — 910B（dav-2201）缺口清单与补丁片段回传（dW 权重梯度波）

执行代理 D / p2-13-ascend-runtime 甲路 P1-1。容器 `cann910b-d`，CANN 8.5.0 + bisheng(clang) 15，
pip tilelang 0.1.15，注入态 `-DTL_PORT910B_NATIVE_TYPES -I<asc>/asc/impl -I<asc>/asc/include`，
`ASCEND_NPU_ARCH=dav-2201 target_format=aibin`。

**主仓真源一字未改**（硬边界）：`/tilelang/src/tl_templates/ascend/port910b_compat.h`
= **944 行**，md5 `8faeb7c17fca…`（本波全程只读，只做只读挂载比对）。
补丁只打在容器内 pip 运行副本
`/usr/local/lib/python3.10/dist-packages/tilelang/src/tl_templates/ascend/port910b_compat.h`，
由 `attempts/D/env_setup_D.sh` 幂等装配（每次先 `cp -f /tilelang/.../*.h` 重置真源，再重打）：

| 阶段 | 动作 | 副本行数 |
|---|---|---|
| [2] overlay | pip 副本 ← 主仓真源 | 944 |
| [2b] **GAP-D1** | 追加 `compat_patch_D.h`（新增 `asc_copy_l12l0a_transpose`，自带守卫，不动既有行） | 944 → **991** |
| [2c] **GAP-D1b** | `patch_compat_D.py` 就地改写 `asc_copy_l12l0b_transpose` 体内原生调用（**签名一字不动**） | 991 → **994** |
| [3] bisheng | `patch_bisheng.py` 注入 -D/-I（幂等自愈 + py_compile 自证） | — |

复跑取证：`bash attempts/D/run_D.sh --setup && bash attempts/D/run_D.sh --matrix`
（判决日志 `attempts/D/verdict_log_D.txt`）。

---

## 0. 结论一览

| # | 缺口 | 触发形态（DSL 写法 → codegen 发射） | 真源 compat 现状 | 处置 | 910B 原生依据 / 取证 |
|---|---|---|---|---|---|
| **G-D1** | `asc_copy_l12l0a_transpose` **整条未定义** | `T.copy(l1_tile, l0a_tile, transpose=True)`（L1→L0A 转置装填）→ `EmitL1ToL0Copy_`（`src/ascend/codegen/codegen_ascend.cc:1563-1567`）在 `transpose==1 && is_l0a` 时**硬发**这个名字，arity 8 | §12 只给了非转置 `asc_copy_l12l0a`(:886) 与 L0B 的 `asc_copy_l12l0b_transpose`(:902)，**A 侧转置件从未写过** | **补 1 个模板件**（`compat_patch_D.h`，8 参转 native 8 参；位序族内一致，标 VERIFY V2'） | 官方 Cal 三处独立证据：`asc/impl/basic_api/dav_c220/kernel_operator_mm_impl.h:132`、`kernel_operator_cube_others_impl.h:314/321`（`LoadCbufToCaTranspose(dst,src,uint16 indexID,uint8 repeatTime,uint16 srcStride,uint16 dstStride,bool addrmode,uint16 dstFracStride)`）、`asc/include/pto/npu/a2a3/TExtract.hpp:48-55`。首挂原文：`tl_kernel.asc:14:5: error: use of undeclared identifier 'asc_copy_l12l0a_transpose'; did you mean 'asc_copy_l12l0b_transpose'?`（编译器自己点出同族件存在） |
| **G-D1b** | 真源 `asc_copy_l12l0b_transpose` **体内 arity 错**（6 参），该发射体**从未可编** | 同上，`is_l0b && transpose==1` → `asc_copy_l12l0b_transpose(...)` | `port910b_compat.h:907` 发 `load_cbuf_to_cb_transpose(dst,src,mStart,mStep,srcStride,(uint8_t)0)` = **6 参** | `patch_compat_D.py` **就地改写函数体**为官方 8 参形 `(dst,src,indexID=mStart,repeatTime=mStep,srcStride,dstStride,false,0)`；签名不删不改，可整块回退 | 首挂原文：`port910b_compat.h:907:29: error: parameters too many`（clang 只匹配到 4 参 packed-config 形 ⇒ 6 参形态在 910B 不存在）。官方 8 参证据：`dav_c220/kernel_operator_mm_impl.h:152`、`kernel_operator_cube_others_impl.h:388`、`TExtract.hpp:154`（`load_cbuf_to_cb_transpose(dstAddr, srcAddr, startIdx0+i*srcRowNum, dstRowNum, 1, dstGap, false, 0)`）。**为什么今天才炸**：真源里该件只被 `gemm.h:83` 的 **TRANS_B=false（NN）** 分支调用，而 A案立的用例全走 `TRANS_B=true`，`if constexpr` 死分支从不实例化 ⇒ 这行代码从未进过编译器 |
| **G-D2** | **GM→L0A/L0B 没有 DMA 通路**（direct-mad 方案的硬阻塞） | `T.copy(gm_slice, l0a_tile)`（跳过 L1 直装 L0A） | 非 compat 层问题：codegen 把该 copy **降级成标量 element-wise 循环**，直接解引用 L0 空间指针 | **本波不在 compat 解决**（改不了），主案绕开：一律 GM→L1→L0 两段。建议 upstream 在 `src/ascend/op/copy.cc:533 DMAPath` 判定处对 `GM→kL1ToL0A/B` 给**显式编译期错误**，而不是静默退化成非法标量体 | 原文：`tl_kernel.asc:11:11: error: only __ubuf__ and __gm__ and local memory pointer can be dereferenced`（生成码体见 `verdict_log_D.txt` 的 `variant=madta` 段：`a_l0[...] = dY[...]` 的 `#pragma unroll` 8192 次标量写，目标指针是 `__ca__ half *a_l0`） |
| **G-D3** | `asc_copy_gm2l1_dn2nz` 是 `nd2nz` 的**同名别名 stub** ⇒ GM→L1 转置搬运（NT 融合模板路的 NN/转置载入前提）**语义未落地** | `T.copy(gm_slice, l1_tile, transpose=True)` → `EmitGmToL1Copy_`（`codegen_ascend.cc:1438-1441`）按 annotation `transpose` 选发 `dn2nz` | `port910b_compat.h:878-882`：`asc_copy_gm2l1_dn2nz(...)` 体内**直接转调** `asc_copy_gm2l1_nd2nz(...)`，注释 `VERIFY V1` ⇒ 编译绿但**不会真转置** | **未补**（补它需要 910B `copy_gm_to_cbuf` 是否带 transpose 位的考古，本波不作主案因此不做）。`ntt` 变体因此在数值上不可信，已降级为"性能后续候选" | 实测：`variant=ntt` 生成码发 `asc_copy_gm2l1_dn2nz((__cbuf__ half*)buf_dyn_l1, &dY[...], 2048, ..., 128, 64, 0, 0)`（`dw_ntt.asc`），而 compat 侧该函数体=nd2nz；对照 `dw_baseline.asc` 的 nd2nz 发射仅 rowBytes/nRows 不同 ⇒ 两者当前**编出的行为一致**，转置丢失 |
| **G-D4** | L0C→GM **无转置直出**（`copy_matrix_cc_to_gm` 行优先直写，`nz2nd` 只做 NZ→ND，不做矩阵转置） | 若把方案写成"先算 `dWᵀ = Xᵀ @ dY`，出口再转回 dW" ⇒ 出口需要第二次重排 | `port910b_compat.h:923-940`（A-4 件，`asc_copy_l0c2gm`，VERIFY V3） | **方案层规避**：主案把转置全压在入口 DMA（L1→L0），出口按 dW 自然朝向直出。若将来必须出口转置（例如 fused dW 直接写 NZ 布局给下一层），需另考古 L0C→L1→MTE3 三段路 | 取证：`dw_l0tr.asc:19` 出口发射 `asc_copy_l0c2gm(&DW[n_tile*1024 + k_tile*128 ...], acc, 128, 128, 1024, 128, ...)` 行主序直写；本波未测转置出口形态 |
| **G-D5** | `asc_mmad` 把 `kDirectionAlign=false`、`btbuf_ctrl` 丢弃**硬编码** ⇒ mad 无法吃 MN-major A/B，"硬件级 trans_A"无从谈起 | 任何想靠 mad 自身转置的写法（G-D2 的 `madta` 意图） | `port910b_compat.h:913-919`（A案 b7 实证形 `mad(c,a,b,m,k,n,unitFlag,kDirAlign,cmatrixSource,cmatrixInitVal)`） | **未补**（补它属硬件语义考古，超出本波 compile 判范围）。这条是"转置必须下移到 DMA"的**根据**而非待办 | 同 §0 G-D2；本波把 trans_A 交给 codegen 布局推断（`gemm_mad.py:132-133`），生成码自动变成显式转置装填（见 RESULT.md §2 的 madl1≡l0tr 实证） |

> **继承 A案的未证真项**（本波新增一处同类）：`VERIFY V1`（gm2l1 32B 单位/NZ pad 模式）、
> `VERIFY V2`（L12L0 位序）、`VERIFY V3`（l0c2gm stride 单位）。
> 本波追加 **`VERIFY V2'`** = `asc_copy_l12l0a_transpose` / 改写后 `asc_copy_l12l0b_transpose`
> 的 8 参位序（`indexID` 取 codegen 发射槽 3 `mStart`、`repeatTime` 取槽 5 `mStep`，
> 与族内已用取法一致；`addrmode=false`、`dstFracStride=0`）。
> codegen 侧槽名依据 `src/ascend/op/builtin.h:247-249`
> （`ascend_load_cbuf_to_ca(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, transpose)`），
> 实测发射值（BN=128/TT=64）：`asc_copy_l12l0a_transpose(a_l0, buf, 0, 0, 4, 8, 4, 8)`
> ——注意 **mStep/kStep 在转置形里的数值与直觉相反**（4=TT/16 出现在"mStep"槽、8=BN/16 出现在"kStep"槽），
> 说明转置形下 codegen 的 m/k 是**按源（L1）朝向**计的。上卡若转置方向对而数值散，先查这里（V2'）。

---

## 1. GAP-D1 补丁片段（可直接贴真源 `port910b_compat.h` §12 末尾）

同 `attempts/D/compat_patch_D.h`（自带 `TL_PORT910B_COMPAT_GAP_D_H` / `!TL_ASCEND_SIMT` /
`TL_PORT910B_NATIVE_TYPES` 三重守卫，不与 §0–§12 任何行冲突）：

```cpp
template <typename T>
__aicore__ inline void asc_copy_l12l0a_transpose(__ca__ T *dst, __cbuf__ T *src, int mStart,
                                                 int kStart, int mStep, int kStep, int srcStride,
                                                 int dstStride) {
  (void)kStart; (void)kStep;
  (void)dstStride;  // VERIFY V2'：与 asc_copy_l12l0b_transpose 同款取舍
  load_cbuf_to_ca_transpose(dst, src, (uint16_t)mStart, (uint8_t)mStep, (uint16_t)srcStride,
                            (uint16_t)dstStride, false, (uint16_t)0);  // VERIFY V2' 位序
}
```

## 2. GAP-D1b 真源回传映射（建议主仓合并的确切改法）

`src/tl_templates/ascend/port910b_compat.h:901-909`，**只替换体内两行**：

```diff
 template <typename T>
 __aicore__ inline void asc_copy_l12l0b_transpose(__cb__ T *dst, __cbuf__ T *src, int mStart,
                                                  int kStart, int mStep, int kStep, int srcStride,
                                                  int dstStride) {
   (void)kStart; (void)kStep;
-  (void)dstStride;  // VERIFY V2
-  load_cbuf_to_cb_transpose(dst, src, (uint16_t)mStart, (uint8_t)mStep, (uint16_t)srcStride,
-                            (uint8_t)0);
+  (void)dstStride;  // VERIFY V2'（与 asc_copy_l12l0a_transpose 同族取法）
+  // GAP-D1b: 6 参形态在 910B 不存在（dav_c220 官方三处证据均 8 参）→ 改 8 参
+  load_cbuf_to_cb_transpose(dst, src, (uint16_t)mStart, (uint8_t)mStep,
+                            (uint16_t)srcStride, (uint16_t)dstStride,
+                            false, (uint16_t)0);  // VERIFY V2' 位序
 }
```

容器侧生效件 = `attempts/D/patch_compat_D.py`（幂等：命中 `GAP-D1b` marker 即 no-op；
锚不到则 `D-PATCH-FAIL` 明确报错，绝不静默）。装配日志：

```
[2b] GAP-D1 compat: 991 lines (marker=3)
D-PATCH-COMPAT-D1b APPLIED /usr/local/lib/python3.10/dist-packages/tilelang/src/tl_templates/ascend/port910b_compat.h (991 -> 994 lines)
```

## 3. GAP-D2 / D3 / D4 / D5 的处置建议（交回主仓/编排者定夺）

- **G-D2（GM→L0 直连）**：属 codegen/IR 层缺陷，compat 补不了。最小上游修法 =
  `src/ascend/op/copy.cc` 在解析 copy 路径时，若 `src scope == global` 且 `dst scope ∈
  {shared.l0a, shared.l0b}` ⇒ 直接 `ICHECK` 报"必须经 L1 中转"，别把语义错误的标量体发给后端。
- **G-D3（dn2nz 真件）**：要启用 `ntt`（融合 L1 模板 + GM→L1 转置载入，DSL 最短、
  模板自带 sub-K 双缓冲，性能面最优）就必须把 `asc_copy_gm2l1_dn2nz` 从别名改成真转置搬运。
  考古入口：`copy_gm_to_cbuf` 的 `pad_t/stride` 组里是否携带 transpose/NZ 选项，或
  `LoadData2DParams.ifTranspose`（SEC12 §2 已记）在 GM→L1 侧的对应件。本波**未做**，
  因为主案不需要它。
- **G-D4（出口转置）**：若生产里出现"必须产出 dWᵀ 朝向"的下游，再立一段 L0C→L1→MTE3→GM 的
  三段路（A案 A-4 已列同类待办）。本波通过方案选择把它绕开了，属**有意规避**，非解决。
- **G-D5（mad 转置能力）**：`asc_mmad` 的 `kDirAlign`/`btbuf_ctrl` 位需要 910B 硬件手册级证据
  才能开；在此之前，任何"让 mad 自己转置"的方案（含 `trans_A=True` 走 MN-major 布局）
  都**只能靠 codegen 降级成 DMA 转置**才成立。

## 4. 本波 compat 覆盖面自检（防"假绿"）

- 判决一律 `TILELANG_CACHE_DIR=$(mktemp -d)` 前缀（`run_D.sh::run_one`）——compat 头内容
  **不进 key**，不换 cache 就是陈旧 PASS。
- 打完 GAP-D 后重跑建材回归 3 件全绿（证明补丁没伤既有面）：
  `D-A2-BASELINE-PASS` / `E2E-CUBE-ONLY PASS` / `variant=baseline` `D-DW-COMPILE-PASS`。
- `l0tr` 从 FAIL→PASS 的**唯一变化**就是 GAP-D1+D1b（同 cache 纪律、同形状、同 tile），
  因果链干净；`madta` 在打完补丁后仍 FAIL 且错误不变 ⇒ G-D2 确为**另一条**独立缺口。
