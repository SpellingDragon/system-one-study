# INVENTORY.md — TileLang 昇腾后端的 950-only 依赖面穷举与 910B 兼容层设计

> 域：`p2-13-ascend-runtime` · 孙任务：**P0-1**（甲路主线首件，纯离线静态分析）
> 输入判决：run `1007-p2-13-b1-910b-port-attempt-7793`（五层剥四层过）+ run `1007-p2-13-c1-ascend-probe-de15`
> 移植对象：tilelang 主仓 `src/tl_templates/ascend/`（模板层 11 头 2694 行）+ `src/ascend/`（C++ lowering/codegen ~51521 行）+ `tilelang/ascend/`（python 门面）
> 纪律：**主仓全程只读**（本次零改动，B1 遗产的 3 个 M + 2 个 ?? 目录非本次所动）；本机无 NPU，不编译不运行任何 ascend 目标。

---

## §0 一页判决（先说结论）

| 项 | 结论 |
|---|---|
| **清单规模** | 跨族去重后 **349** 个 950-only 符号：**A=35 / B=20 / C=250 / D=44**；`./checklist.sh` 报的 **354** 是 12 族 uniq 的**族内**累加（族间有重叠，354−349=5 个符号被两族同时命中）。逐符号见 `inventory_symbols.md`（`./checklist.sh symbols` 生成） |
| **头号事实** | **C 类占 72%**。910B 移植的主体不是"缺头/缺类型别名"（那是 A/B 的 55 个符号，A35+B20），而是 **TileLang 的向量侧整个压在 950 SIMT/SIMD 方言上**：160 个 `asc_*` MicroAPI + 谓词寄存器 + 线程模型。 |
| **最大未知数** | **D2-AIC-API（14 符号）**：`asc_init` / `asc_copy_gm2ub_align` / `asc_mmad` / `asc_lock|unlock` / `asc_copy_l12l0a|b` 这批 **AIC(cube) 通路 API 在 910B/CANN 8.5.2 是否可用，静态不可判**。它与 design.md 的判断正面冲突（见 §4-墙0），并且是 P0-2 第一个 vecadd 级 kernel 的**生死线**——`asc_init();` 由 codegen **无条件**发射在每个 kernel 函数体首行（`codegen_ascend.cc:711`，函数 `PreFunctionBody` @`:710`）。 |
| **第二墙** | **SIMT 不可移植**：`T.SimtVF` 是本仓七件 kernel 的**唯一向量执行载体**（§5），而 SIMT 面（`__simt_vf__`/`threadIdx`/`asc_shfl_xor`/`asc_ballot`/`asc_vf_call`/`cooperative_groups`/sub_block 1:2）在 910B 无对应硬件模型 → 必须**改道**（SIMD 化或标量化），不是加头文件能解决的。 |
| **兼容层能兑现多少** | `port910b_compat.h` 覆盖 **B 类 20 符号**，即"bisheng 类型系统第 5 层墙"（bf16/fp16 代理与转换内建）**可离线闭环**——已由 `host_selfcheck.cc` 在主机 clang16 三种后端配置下跑出 `SELFCHECK PASS`（bf16 RNE 位型表 + fp16 与 `_Float16` 神谕 20 万样本 mismatch=0 + codegen 的 `((bfloat16x2_t*)&v)[i].x` lane 位视图），故 B 类的**形状/数值**判决是实证级，只有"bisheng 是否已自带同名内建"仍需 P9 打真值；C 类 250 符号里，本兼容层只能提供 `#if !defined(TL_ASCEND_SIMT)` 下的**桩/禁用面**，真正的 910B 向量通路要等 P0-2 的 D 类裁决后再立项。 |
| **建议的甲路下一步** | P0-2 上卡先跑 `probe910b.sh`（本目录，10 个静态探针，零训练成本，≈15min 卡时）→ 拿到 D1/D2/D3 三组存在性真值 → 再决定"910B 向量侧走 SIMD-intrinsic 翻译层"还是"vector 侧整体让位 torch_npu 混合栈"。 |

---

## §1 方法与分类学

### 1.1 扫描范围（可复跑）

| 层 | 路径 | 体量 |
|---|---|---|
| 模板层（device 头） | `src/tl_templates/ascend/*.h`（11 文件） | 2694 行 |
| C++ lowering/codegen | `src/ascend/**/*.{cc,h}` | ~51521 行 |
| python 门面/pass 序 | `tilelang/ascend/**/*.py` | language 4542 行（`simd.py` 1488） |

族定义（`F1..F12`，含 ERE 与扫描范围）落在 `checklist.sh` 的 `FAMS` 表——**族即"依赖面"的切分维度**，改族必须同步改本文件。

### 1.2 四分类语义（判决口径，勿混用）

| 类 | 定义 | 判据 |
|---|---|---|
| **A 已解** | **无需新增工程**：或 B1 四层处置法已落补丁（主仓工作区，未 commit），或 910B/CANN 8.5.2 编译器实勘**已证存在**，或是扫描副产物（同名字串、非外部符号） | 补丁位置可指到 `文件:行`；或实勘清单点名 |
| **B 兼容层可解** | 纯头文件工程可闭环：类型别名 / union 位视图 / `__builtin_bit_cast` 桥 / 转换内建 / 常量宏 | 不需要编译器新增硬件能力，只需给 C++ 一个等价表达 |
| **C 需 lowering 侧改造** | **架构概念在 910B 不存在**（SIMT 线程模型、谓词寄存器、mx/fp8/fp4 MMA、dual-copy/unit-flag 模式、1:2 sub-block 分组、950 事件枚举……），必须降级、关特性或换实现路径 | 即便补齐头文件声明也无法直译；改动点在 `src/ascend/`（pass/codegen） |
| **D 静态不可判** | 唯一悬而未决的是"910B 的 CANN 8.5.2 是否**声明**该符号"。本机无 CANN 头（已核实：`/usr/local/Ascend` 不存在，全工作区无 `c_api/asc_simd.h` 真身），无法静态裁决 | 只能上卡探针（§6） |

**证据强度列**（附表未展开，此处统一声明）：
`实证`=B1 编译推进过程中该点已被跨过（run 7793 config `patches_applied`）；
`静态`=本仓源码 `文件:行` 举证；
`推断`=按公开 AscendC/CANN API 形态记忆类比，**未在本机核实**（凡"910B 等价物"字样均为此级，P0-2 必须逐条打真值）。

### 1.3 完备性自证（三件套）

```bash
cd release/ascend/port910b
./checklist.sh symbols   # 生成/刷新逐符号附表（349 行，分类来自 classify()）
./checklist.sh           # audit：族穷举 → 反查清单，unlisted=0 才 PASS
./checklist.sh gap       # 族外残差体检（模板层全部标识符 - 族覆盖 = 人工复核队列）
```

`audit` 的**逻辑边界要如实交代**：附表由同一族穷举生成，故 audit 证明的是
"族内符号 100% 未漏抄 + 分类规则全覆盖"，**不等于**"族集合本身无漏"。后者由 `gap` 模式独立兜：
它把模板层出现的、形如外部依赖的标识符（`__*`、`*_t`、全大写宏、前缀白名单）与族覆盖做差集，
残差逐个人工复核（复核记录与结论见 §7）。

---

## §2 族表与规模（数字全部可 `./checklist.sh tok F#` 复跑）

| 族 | 含义 | 范围 | uniq | 主要落点 |
|---|---|---|---|---|
| F1 | 950 代际头 include（`c_api/`、`simt_api/`） | 全三层 | 6 | `common.h:11-15`、`debug.h:12`、`gemm.h:3`、`reduce.h:3`、`simd_inst.h:5`、`ascend_fp8.h:5-6`；codegen 发射 `<simt_api/cooperative_groups.h>` @`codegen_ascend.cc:633` |
| F2 | device 属性 / 函数限定宏 | 模板+codegen | 16 | `__simt_vf__` `__simt_callee__` `__simd_vf__` `__simd_callee__` `__aicore__` `__launch_bounds__` `__global__` `__mix__` `__cube__` + 存储限定符 `__gm__/__ubuf__/__cc__/__cbuf__/__ca__/__cb__` |
| F3 | SIMT 内建变量与核身份 | 全三层 | 11 | `block_idx` `blockIdx.` `threadIdx.` `asc_get_sub_block_id` `ASC_IS_AIC/AIV` `cce::dim3` `laneid` `cooperative_groups` |
| F4 | SIMD 寄存器类型 `vector_*` | 模板+codegen | 19 | `simd_inst.h:8-78` 的 `vec<T>` 特化表；`vector_bool` 命中 141 次（谓词寄存器） |
| F5 | 类型代理（bf16/fp16/fp8/fp4/宽向量元组） | 全三层 | 29 | `bfloat16_t` `bfloat16x2_t` `half` `float2/4` `fp8_e*_t` `float4_e*_t` `float8_e*_t` |
| F6 | `asc_*` C API（950 MicroAPI 命名空间） | 全三层 | 160 | SIMD 方言主体（`simd_inst.h` 独占 ~110）+ AIC 面（`gemm.h`/codegen）+ SIMT 面 |
| F7 | 双下划线内建（转换/存储/常量） | 全三层 | 22 | `__bfloat162float` `__float22bfloat162_rn` `__asc_cvt_float2_to_fp8x2` `make_float2/4` `make_ulonglong4` |
| F8 | 硬件枚举/tag/mode 类型与事件 | 全三层 | 47 | `PAT_*`(14) `MODE_ZEROING/MERGING` `ROUND_R/RS_DISABLE/PART_*/POST_UPDATE/HIGHER/LOWER/POS_LOWEST/INC_ORDER/Bin_N0` `PIPE_*`(11) `asc_*_mode`(7) `event_t` `ASC_LOCK_*` `__ASC_(E4M3/E5M2/SATFINITE)` `__NPU_ARCH__` `DATA_BLOCK_COPY` |
| F9 | C++ 标准库依赖（`std::bit_cast` / `<bit>`） | 模板层 | 3 | `numeric_limits.h:3,6,8` |
| F10 | 950 专属语言构造与 pass 名 | 全三层 | 15 | `SimtVF`(101 处) `SimdVF`(147 处) `blockscaled`(120) `nd2nz`(86) `hf32`(61) `Nd2Nz`(45) `set_atomic`(41) `dual_copy`(20) `ascend_mad_mx`(18) + 4 个 pass 名 |
| F11 | 全局作用域 CCE 内建与 950 运行时件 | 模板+codegen | 19 | `::vcvt` `::vpack` `::vsstb` `::asc_mem_bar` `::print_var` `::print_buffer` `::BoxMullerFloat` `::ALG_KEY_SIZE` `cast_float_to_fp8_*` `IDX_2/IDX_3` `RAND_2POW32_INV` |
| F12 | 分形/布局与同步容量前提 | 全三层 | 7 | `C0`(82) `Fractal`(137) `fractal`(48) `TRANS_B` `mix_aiv_count` `nBlk` `kIntraCoreFlagLimit` |

> 族间有重叠（同一符号可被多族命中），**附表按符号去重后为 349 行**；族内 uniq 之和 = audit 输出的 **354**（6+16+11+19+29+160+22+47+3+15+19+7），二者差 5 属正常。

---

## §3 逐组处置方案（组号即附表"组"列；举证为 `文件:行`）

### A 类（33）— 已解，不动工

| 组 | 内容 | 举证 / 处置落点 | 强度 |
|---|---|---|---|
| **A1-HEAD-STUB** | 5 个 950 代际头的 include 语句 | 桩已落：`src/c_api/asc_simd.h`、`src/simt_api/{asc_bf16,asc_fp16,asc_fp8,asc_simt}.h`（均 `#pragma once` + 说明注释，纯新增未改上游）；include 站点 `common.h:11-15`、`debug.h:12`、`gemm.h:3`、`reduce.h:3`、`simd_inst.h:5`、`ascend_fp8.h:5-6` | 实证（"file not found" 已消除） |
| **A2-ATTR-ALIAS** | `__aicore__` | `debug.h:5-7` `#ifndef __aicore__ / #define __aicore__`（空展开）；调用面 `dcache_bypass.h:14` 等 29 处 | 实证（第 3 层墙已跨） |
| **A3-BLOCKIDX-0** | `block_idx` | `debug.h:290` `TL_DEFINE_ASCEND_DEBUG(__aicore__, 0)`（debug 打印丢块号，语义无损）；SIMT 变体 `debug.h:293-297` 已罩 | 实证（第 4 层墙已跨） |
| **A4-BITCAST** | `std::bit_cast` / `<bit>` / `__builtin_bit_cast` | `numeric_limits.h:3`（`#include <bit>` 仍在，靠 libstdc++ 提供）、`:8` `#define TL_BIT_CAST(D,x) __builtin_bit_cast(D,x)`；调用面 `numeric_limits.h:14-29`（`kBf16Inf` 等常量） | 实证（第 2 层墙已跨） |
| **A5-GATE** | `TL_ASCEND_SIMT` 总开关（本域自建） | `common.h:6-17`（950 头组 + `ascend_fp8.h`）、`common.h:20-22`（`simd_inst.h`）、`debug.h:294`；**注**：全仓无人注入该宏（`bisheng.py`/`libgen.py` 均无 `-DTL_ASCEND_SIMT`），默认关闭 = 默认 910B 面 | 静态 |
| **A6-SCAN-ARTIFACT** | 15 个"看着像 950 符号其实是别的东西"的串 | 头文件名子串（`asc_bf16/asc_fp16/asc_fp8/asc_simd/asc_simt`）、tilelang 自有命名空间（`__asc_aicore`/`__asc_simt_vf` @`debug.h:16,289,293`）、codegen `FreshName()` 生成的局部变量名（`__asc_pad_val` @`codegen_ascend.cc:1524`、`__asc_rng_state` @`:2545`）、python 侧 `normalize_asc_target`（`target.py:78`）、注释引用（`asc_loadalign_v2_impl` @`simd_inst.h:193`、`__asc_fp8_interpretation_t` @`target_utils.cc:20`）、**`::print_var`/`::print_buffer`**（实为 `debug.h:24,30` 宏体内 `PrintTraits<T>::` 的静态成员，被 F11 的 `::` 前缀误捕——本次写兼容层时取证改判，原列 B4） | 静态 |
| **A7-ATTR-PROVEN** | 存储限定符 `__gm__`(75) `__ubuf__`(86) `__cbuf__`(17) `__cc__`(3) `__ca__`(10) `__cb__` | 910B 编译器地图（CANN 8.5.2 bisheng clang15.0.5）实勘俱备；发射点 `codegen_ascend.cc:152-166` `GetAscendScopeQualifier()` | 实证（实勘） |

> **A5 的残留缺口（本次新发现，P0-2 第一个错误大概率在这里）**：`reduce.h` 被 `common.h:19` **无条件** include，而 `reduce.h` 自身**没有任何 `TL_ASCEND_SIMT` 罩**（全文只有 `HUGE_VALF` 一个 `#ifndef`），却含 29 处 `__simt_callee__`、`asc_shfl_xor`（`:32-56`）、`asc_reduce_add/max/min`（`:62-88`）、`threadIdx.x`（`:192,232,261`）、`asc_syncthreads()`（`:238,247,249,270,296,301,303`）。→ 910B 面上 `common.h` 这条 TU **编不过**，除非把 reduce.h 也纳入罩内并给出 910B 归约路径（方案见 C1）。同类漏罩还有 `gemm.h:3`、`simd_inst.h:5`、`nd2nz_copy.h:3`、`debug.h:12`、`ascend_fp8.h:5-6`（它们只在特定 op 下被 codegen include，见 §6 探针 P2）。

### B 类（20）— 兼容层可解，落 `port910b_compat.h`（主机侧已 `SELFCHECK PASS`）

| 组 | 符号 | 910B 等价物方案（**全部为"推断"级，须 P0-2 打真值**） |
|---|---|---|
| **B1-TYPE** | `bfloat16_t` `bfloat16x2_t` `half` `half2` `half_t` `float2` `float4` `make_float2/4` `make_int2` `make_{u}longlong4` `__fp16` | ① `bfloat16_t` → clang15 builtin `__bf16` 的 **typedef + 成员转换算子**（第 5 层墙的正解：bisheng 自带 `BFloat16` 代理但缺 `(float)`/`operator=` 隐式转换，我们用 `struct bfloat16_t { __nv_bfloat16_storage… }` 或直接 alias + 自由函数补齐 `PrintTraits`/`numeric_limits` 所需面）；② `bfloat16x2_t`/`half2`/`float2/4` → **union 位视图**（`{T data[2]; struct{T x,y} }`，与 `codegen_ascend.cc:3593,3630` 发射的 `((bfloat16x2_t*)&(…))[i/2]` 的取址-下标习惯严格兼容）；③ `make_*` → `constexpr` 函数（`{x,y}` 聚合初始化）。 |
| **B2-CVT** | `__bfloat162float` `__float2bfloat16_rn` `__bfloat1622float2` `__float22bfloat162_rn` `__half22float2` `__float22half2_rn` | 标量对：`static inline` + `__builtin_bit_cast` 保位；bf16↔f32 用**截尾舍入**（`+0x8000` 后取高 16 位，`__float2bfloat16_rn` 语义）；成对版拆成两次标量调用。发射点：`codegen_ascend.cc:3333,3355,3403,3411,3419,3428`。 |
| **B3-LOCKMODE** | `ASC_LOCK_BLOCK` `ASC_LOCK_NON_BLOCK`（+前缀截断命中 `ASC_LOCK_NON`） | 常量宏 `#define ASC_LOCK_BLOCK 0` / `ASC_LOCK_NON_BLOCK 1`——**但语义仍取决于 `asc_lock/asc_unlock` 是否存在**（那两个在 D2，故本组只解决"枚举值"，不解决"调用可用性"）。发射点 `codegen_ascend.cc:1853-1870`（含 `rewrite_flag_to_buf.cc:20-23` 的 `PIPE_PROD/PIPE_CONS`）。 |
| **B4-DECL-MACRO** | `__SIMT_DEVICE_FUNCTIONS_DECL__`（1 个） | `#define __SIMT_DEVICE_FUNCTIONS_DECL__`（空，`#ifndef` 保护）以解 `ascend_fp8.h:9` 的 `TL_DEVICE` 定义。**原列在本组的 `::print_var`/`::print_buffer` 经取证改判为 A6**：`debug.h:260,266` 的 `::` 是 `PrintTraits<T>::` 限定，两函数由 `TL_DEFINE_ASCEND_DEBUG` 宏在 `debug.h:24,30` 自行定义，不是外部符号。 |

### C 类（250）— 需 lowering/codegen 改造（本域真正的工程主体）

按"发射者 → 910B 处置"组织。**每条给出：谁会发射它（pass/op/codegen 站点）+ 910B 上降级成什么或关哪个特性。**

| 组 | 规模 | 符号代表（举证锚点） | 谁发射 | 910B 处置建议 |
|---|---|---|---|---|
| **C1-SIMT** | 24 | `__simt_vf__`(52 处) `__simt_callee__` `threadIdx.`(43) `blockIdx.`(13) `asc_vf_call` `asc_ballot` `asc_activemask` `asc_shfl_xor` `asc_syncthreads` `asc_threadfence` `asc_reduce_{add,max,min}` `asc_atomic_*` `asc_update_addr_reg_b` `ASC_IS_AIV` `cooperative_groups::coalesced_threads()` `laneid()` `cce::dim3` `__launch_bounds__` `SimtVF`(101) `PIPE_PROD/CONS` | `codegen_ascend.cc:978`（`__simt_vf__` + `__launch_bounds__(n)`）、`:986`（`asc_vf_call<…>(cce::dim3(...))`）、`:881`（`if ASC_IS_AIV`）、`:1814-1845`（threadfence/ballot/activemask）、`:1948-1950`（`tl.warp_reduce_*` → `asc_reduce_*`）、`:3557`（`asc_syncthreads`）、`:633`（include `simt_api/cooperative_groups.h`）；模板面 `reduce.h` 全篇、`philox_rng.h`、`random_kernel_base.h`；语言面 `tilelang/ascend/language/frame.py` `T.SimtVF` | **整条 SIMT 路径在 910B 关闭**：① `T.SimtVF(...)` 的 frame 降级为"单控制流 VF"或直接拒绝（`frame.py` + `vf_checker.py` 加 arch 判据）；② `tl.warp_reduce_*`/`T.reduce_sum(dim=1)` 改走**显式归约**——注意 `reduce.h:231 ub_reduce` 虽名为 UB 回退，但仍用 `threadIdx.x`+`asc_syncthreads()`（`:232,238,247`），**并非 SIMT-free**，910B 需新写"单流 UB 归约"（纯循环，无线程身份）；③ `asc_ballot/activemask/shfl` 无硬件对应 → 依赖它们的写法必须改写（本仓七算子仅 add_ln 的 reduce 触及，见 §5）；④ 桩头 `simt_api/cooperative_groups.h` 不补，改为 codegen 在 910B 目标下**不发**该 include。 |
| **C2-SIMD-REG** | 95 | `asc_loadalign*`(8 变体) `asc_storealign*`(6) `asc_{add,sub,mul,div,max,min,and,or,xor,not,neg,abs,relu,exp,ln,sqrt}`(+`*_scalar`) `asc_select` `asc_duplicate*` `asc_create_mask_b{8,16,32}` `asc_update_mask_b*` `asc_gather*` `asc_scatter` `asc_intlv*`/`asc_deintlv*` `asc_pack_to_{high,low}` `asc_unpack_*` `asc_arange*` `asc_squeeze`/`asc_unsqueeze` `asc_pair_reduce_sum` `asc_reduce_{sum,max,min}_datablock` `asc_histogram`(4) `asc_madd`/`asc_axpy`/`asc_{addc,subc,mull}` `asc_mem_bar` `SimdVF`(147) `AscendSimdVFLowerParallel` `LegalizeSimdMerging` | 模板层 `simd_inst.h`（90 处 `__simd_callee__`，1050 行全部是这层的 wrapper）+ `nd2nz_copy.h`；codegen `:906`（发 `simd_inst.h` include）、`:922`（`__simd_vf__ inline void`）、`:1770`（ICHECK "simd op only inside SimdVF"）、`:3179`（`vector_bool v = asc_create_mask_b<bits>(…)`）、`:2626,3598`（`::from_bits`）；pass：`tilelang/ascend/pipeline.py` 的 `AscendSimdVFLowerParallel`、`LegalizeSimdMerging` | **两条候选路，P0-2 后二选一**：<br>**路 B（重）**：把 950 的"寄存器值语义 + 谓词寄存器"方言翻译成 910B AscendC 的"**UB 内存 + count 长度**"模型——`vec_t<T>` 建模为 UB scratch 段，`asc_add(dst,a,b,mask,mode)` → `AscendC::Adds(dst_p,a_p,b_p,count)`；谓词面（`vector_bool` 141 处、`PAT_*` 14 个 pattern、`MODE_{ZEROING,MERGING}`）**在 910B 无对应硬件概念**，只能用"全 VL 计算 + 事后掩码写回"模拟，`LegalizeSimdMerging` 的 merging 语义整体退化。<br>**路 C（轻，建议先做）**：910B 目标下**禁用 `T.SimdVF`/`T.SimtVF` 两条向量通路**，vector 侧交给 torch_npu 混合栈（C1 已有的回退先例），TileLang 只保 AIC/cube 件。工作量：`pipeline.py` 在 2201 档不挂 SIMD 降级 pass + `frame.py` 拒绝进入 + `codegen_ascend.cc:1770` 那条 ICHECK 前加 arch 判定给出可读报错。 |
| **C3-SIMD-TAG** | 27 | `PAT_VL/PAT_VL1..128/PAT_ALL/PAT_ALLF/PAT_H/M/Q`(14) `MODE_ZEROING`(64)`MODE_MERGING`(29) `ROUND_R` `RS_DISABLE` `PART_{EVEN,ODD}` `POST_UPDATE` `HIGHER` `LOWER` `POS_LOWEST` `INC_ORDER` `Bin_N0` `::vcvt` `::vpack` `::vsstb` `::asc_mem_bar` | `codegen_pto.cc:1709,4915-4916,5181,5351,5522`；`simd_inst.h:86-104,790-845,977-1044`；`nd2nz_copy.h:69-71,100-101,108-111,153` | 这些 tag 是 **950 SIMD 指令的分布/掩码模式参数**，随 C2 一起决定：走路 C 则连同 `simd_inst.h` 一并不编译（已被 A5 罩住，无需改造）；走路 B 则须为每个 tag 找 910B 常量或退化（`PAT_*`→无对应，`MODE_MERGING`→掩码写回模拟）。`::vcvt/::vpack/::vsstb` 是**全局作用域 legacy CCE 内建**，910B 大概率可用（同族 legacy intrinsic），但签名（4 个控制参数）不同 → 探针 P4 一次编过即知。 |
| **C4-MX-FP8** | 25 | `asc_mmad_mx` `asc_copy_l12l0{a,b}_mx` `blockscaled`(120) `RewriteFp4ToFp4x2` `ascend_mad_mx`(18) `fp8_e{4,5,8}_[2,4,8]_t` `float8_e{4m3x2,5m2x2,8m0}_t` `float4_e{2m1,1m2}x2_t` `vector_f8*`/`vector_f4*` `__ASC_{E4M3,E5M2,SATFINITE}` `__asc_cvt_float2_to_fp8x2` `cast_{float_to,fp8_}_fp8*` | `gemm.h:137-226`（`ascend_blockscaled_gemm_l1`）、`simd_inst.h:17-39`（**`#if (__NPU_ARCH__ == 3510)` 已自动关掉 bf16/fp8/fp4 的 `vec<T>` 特化**）、`ascend_fp8.h:17-146`、codegen `:245,527,1586-1588,3442-3520`、op `builtin.cc:401,411`、pass `pipeline.py` `RewriteFp4ToFp4x2`、语言 `gemm_mad_blockscaled.py` | **关特性，不桥接**：910B cube 无 fp8/fp4/mx 缩放 MMA（硬件级不支持），桥出软件 fp8 也无人可用。落地：① dtype 白名单——`target_utils.cc`/`codegen_ascend.cc:239-251,374-378,1588` 在 2201 档 ICHECK 拒绝 fp8/fp4 dtype；② `ascend_blockscaled_gemm_l1` / `tl.ascend_mad_mx` op 注册点在 2201 档不可选（`builtin.cc:401,411`）；③ `RewriteFp4ToFp4x2` pass 不挂载。收益：**C4 全部随 SIMT/SIMD 罩自动消失**（因为 `fp8/fp4` 只在 950 面有消费者）。 |
| **C5-AIC-MODE** | 16 | `asc_set_atomic_{add,max,min,none}`(+dtype 后缀) `asc_{enable,disable}_hf32` `asc_set_hf32_round_mode` `asc_hf32_round_mode::{NEAREST_AWAY,NEAREST_EVEN}` `asc_set_mmad_direction_{m,n}` `asc_{unit_flag,quant,relu_pre,dual_dst,load_l2_cache,store_l2_cache}_mode` `set_atomic`(41) `hf32`(61) | codegen `:1877-1899,1935,2527`、`:1483-1486`（`EmitL0cToUbufCopy_` 的 4 个 mode 实参）、`:1415,1430`（L2 cache mode）；op `builtin.cc:436,441,446,451`；语言 `mode.py:57-79`（`_ATOMIC_DTYPES` 注释点名 "the dav_3510 store-mode atomic"） | **逐项降级**：① `set_atomic`（bf16/f16/f32/i8/i16/i32 的 store-mode 原子）→ 910B 无等价 store-mode API，**关闭** `T.set_atomic`（`mode.py` 加 arch 判据），dW 类累加改由 host 侧或 `AtomicAdd` 系列（若探针 P3 证实存在）；② `hf32`（TF32 类高精度浮点模式）910B 无 → 关闭；③ `mmad_direction_m/n`、`unit_flag`、`dual_dst`、`quant`、`relu_pre`、L2 cache mode 属 950 MMA 扩展控制 → 关闭并把 codegen 的实参位收敛（`:1483-1486` 与 `EmitMad_` 的 `uf_ctrl`）。 |
| **C6-ND2NZ-DUAL** | 12 | `asc_copy_gm2l1_{nd2nz,dn2nz}` `asc_set_gm2l1_nz_para` `asc_set_l0c_copy_nz_para` `asc_set_copy_pad_val` `asc_fill_l1` `asc_copy_ub2l1` `asc_copy_l0c2gm` `ascend_nd2nz_scatter_callee` `nd2nz`(86) `Nd2Nz`(45) `InsertNd2Nz` `dual_copy`(20) `DATA_BLOCK_COPY` | pass：`pipeline.py` 的 `RewriteDualCopy`、`InsertNd2Nz`、`RewriteFlagToBuf`；op `copy.cc:388`（注释："**dav-3510** exposes GM->L1 ND2NZ for 8/16/32-bit storage types"）、`:589-602,702-714`（传 `unit_flag_ctl`/`sub_blockid`）、`builtin.cc:394-411`；codegen `:1443,1458,1474,1499,1530,1549,1602,1619`；模板 `nd2nz_copy.h`（`:126,170` 两处 **`static_assert(COLS == -1, "…scatter: TODO")`** = 部分 VL 分支上游本就未实现） | ① `InsertNd2Nz`：910B 上 GM→L1 直转 ND2NZ **不可假定**（`copy.cc:388` 注释只承诺 dav-3510）→ 降级为 `GM→UB(MTE2) → UB→L1(ND2NZ by Fixpipe/DataCopyExt)` 或直接让 cube 走 NZ 输入约定；② `dual_copy`/`RewriteDualCopy`：950 双写特性，2201 档不挂载该 pass；③ `asc_fill_l1`/`asc_set_copy_pad_val`（OOB padding 自动补零，`AscendInsertOOBPadding` pass 依赖）→ 910B 无 pad-val 硬件语义时改**显式 memset/补零 kernel**；④ 注意 `nd2nz_copy.h` 的 TODO 分支在 910B 上不会被绕开（`COLS != -1` 的部分 VL 场景）。 |
| **C7-MIXKERN** | 7 | `asc_get_sub_block_id`(5) `__mix__` `__cube__` `__global__` `__simd_vf__` `MixedKernel` `ASC_IS_AIC` | codegen `:342`（`__global__ __mix__(1, 2)`）、`:348`（`__global__ __cube__`）、`:869,881`（`if ASC_IS_AIC` / `if ASC_IS_AIV`）、`:887`（`if (asc_get_sub_block_id() == 0)`）、`:340` 注释（"**dav-3510**'s direct AIC-to-AIV data paths require the physical **1:2 group**"）、`:341 mix_aiv_count_`；`ir.cc:57`（Frame 1 sid = "cthread" binding） | 属性宏本身 910B 可用（`__mix__/__cube__` 在实勘清单内），**但 `__mix__(1, 2)` 的 1:2 AIC:AIV 物理分组是 950 事实**：910B 的 AIC/AIV 配比与"sub-block"概念不同 → `EmitGmToL1`/`mix_aiv_count` 相关发射与 `asc_get_sub_block_id()` 需按 910B 拓扑重定（探针 P5：在 2201 上打印 `GetSubBlockNum/GetBlockNum` 语义）。短期：混合核退化为"纯 cube 核 + host/torch 做向量"。 |
| **C8-EVENT** | 5 | `asc_sync_notify` `asc_sync_wait` `asc_sync_*_{arrive,wait}`（`inter/subblock/block/intra`）`event_t` `kIntraCoreFlagLimit` | codegen `:1349-1364`（`asc_sync_notify(PIPE_x, PIPE_y, static_cast<event_t>(id))`）、`:1366-1381`（`GetCrossCoreSyncScope` @`:274-289`，mode 0/1/2/4 → inter/subblock/block/intra）、`:1343-1346`（`asc_sync()` / `asc_sync_pipe(PIPE_x)`）；pass `insert_sync.cc:1943,2013,2018,2135`（`kIntraCoreFlagLimit = 8`，注释"exceeding the **dav-3510** limit"）、`rewrite_flag_to_buf.cc:20-23` | ① `static_cast<event_t>(int)` 这种**运行期整数→事件枚举**在 910B 老 AscendC 里不存在（那边是编译期模板参数 `SetFlag<HardEvent::MTE1_M>(id)`）→ codegen 需改为把 `GetPipePair()` 的字符串在**编译期**拼成 `HardEvent::X_Y` 模板参数（改 `EmitHardEventSync_`，站点 `:1349`）；② `subblock` scope 随 C7 的 1:2 分组一起裁决；③ flag 上限 8 是 950 数，910B 需换成 2201 的事件数（`insert_sync.cc:2135` 常量化 → 由 target 参数注入）。 |
| **C9-RNG950** | 9 | `philox_rng.h`/`random_kernel_base.h` 全篇（`__simt_callee__ __aicore__` 各 8/13 处）、`::BoxMullerFloat` `::ALG_{KEY,COUNTER}_SIZE` `IDX_2/IDX_3` `RAND_2POW32_INV(_HALF)` | 头注释 `philox_rng.h:7-9`：**vendored from CANN 9.1.T560，`__NPU_ARCH__ == 3510`，路径 `opp/.../random_common/arch35/`**；codegen `:2540-2550`（`tl.rng_init` 要求 `IsInsideSimtVF()`）；include 发射 `:622` | 关：910B 无 arch35 random 件，且 `tl.rng_init` 硬绑 SimtVF（`:2544` ICHECK）。本域七算子不用 RNG（`random.py` 无消费者）→ 2201 档直接禁 `rng_*` op，随机面若需要改由 torch 侧生成。 |

### D 类（44）— 静态不可判，交给 P0-2 探针

| 组 | 规模 | 符号 | 为什么静态判不了 | 探针 |
|---|---|---|---|---|
| **D2-AIC-API** | 14 | `asc_init`(1) `asc_copy_gm2ub_align` `asc_copy_ub2gm_align` `asc_copy_l12l0a`(+`_transpose`) `asc_copy_l12l0b`(+`_transpose`) `asc_copy_l0c2ub` `asc_mmad` `asc_lock` `asc_unlock` `asc_sync` `asc_sync_pipe` `asc_load_dev`/`asc_store_dev` | 这些符号**没有任何本地声明**：`src/ascend/` 全文不含 `c_api/` 字样（已 grep 证实），唯一入口是模板层 `#include "c_api/asc_simd.h"`，而该头在 910B 镜像**缺失**（B1 第 1 层墙），桩头是空的。**design.md 判"属 910B 兼容面"的证据只是"C1 崩溃源码里出现了这些调用"——那只能证明 codegen 发了它们，不能证明 bisheng 有声明**（详见 §4-墙0） | P1/P2/P3 |
| **D1-REGTYPE** | 14 | `vector_f32/f16/bool/u8/u16/u32/u64/s8/s16/s32/s64/bf16` + `vector_uint16_t`/`vector_uint32_t` | 同 D2（声明只在缺失的头里）。附加风险：`vector_uint16_t`(1) 与 `vector_uint32_t`(1) 仅出现在 `simd_inst.h:1011,1018` 的 `reinterpret_cast`，与全文件其余 `vector_u16/u32` 命名不一致——**疑似上游非正式别名**（若 950 头没这两个 typedef，950 上也编不过；说明更可能是别名存在，但必须实测） | P3 |
| **D3-PIPEENUM** | 11 | `PIPE_ALL/V/M/MTE1/MTE2/MTE3/S/FIX/H` + `__NPU_ARCH__`(2) | 管道枚举与 arch 宏都是 bisheng 预定义，需实测存在性与取值 | P1/P6 |
| **D4-LAYOUT** | 5 | `C0`(82) `Fractal`/`fractal`(185) `TRANS_B`(6) `nBlk` `mix_aiv_count` | 分形/块粒度是**数值前提**而非符号存在性：`gemm.h:45-47` 假定 `C0 = 32B / sizeof(InT)`（bf16→16），`gemm.h:34` 注释 "**TRANS_B must be true on dav-3510**"、`:78` "transpose=true 只对 b16 有效"；910B 的 L1↔L0A/L0B 分形（16×16 / 行步长）与转置能力**必须重测**，否则 MMA 结果会静默错位 | P7 |

---

## §4 五道结构性墙（符号层面之上）

| # | 墙 | 证据 | 影响 |
|---|---|---|---|
| **墙0** | **design.md 的"卡点=头而非通路"判断，证据链偏弱** | design.md:6 称 `asc_copy_gm2l1_nd2nz/ascend_gemm_l1<…>` "在 C1 崩溃源码中完整出现且属 910B 兼容面"；但 run 7793 config 记录的第 1 层墙是 `c_api/asc_simd.h` **在 910B 镜像缺失**，而该头是**唯一**的 `asc_*` 声明入口（`src/ascend/` 不含 `c_api/`；模板层 `common.h/debug.h/gemm.h/simd_inst.h` 都靠它）。"源码里出现" ≠ "编译器有声明" | 若 P1/P2/P3 探针失败（头与内建都没有），则 AIC 通路也要重写为 legacy AscendC（`DataCopy`/`Matmul` 模板）——工作量级别从"数天"跳到"数周"。**这是本域最大的战略风险，必须在 P0-2 第一件事就裁决** |
| **墙1** | **SIMT 无对应硬件模型**（C1 组） | `reduce.h` 29 处 `__simt_callee__` + `threadIdx`/`asc_syncthreads`；codegen `:978,986` 发 `__simt_vf__`+`asc_vf_call`；七件 kernel 全用 `T.SimtVF`（§5） | 向量侧必须换载体（SIMD 化 / 标量化 / torch_npu 混合），不是兼容层可解 |
| **墙2** | **谓词寄存器与 zeroing/merging 语义**（C2/C3） | `vector_bool` 命中 141；`asc_create_mask_b{8,16,32}` + `PAT_*`(14) + `MODE_ZEROING`(64)/`MODE_MERGING`(29)；`LegalizeSimdMerging` pass 专为 merging 语义存在；`codegen_ascend.cc:417,430` 注释要求 "SimdVF predicates must be boolx256 / 2048-bit register" | 910B 的 vector 编程是 count 长度 + 掩码内存，无 2048-bit 谓词寄存器；`LegalizeSimdMerging` 在 2201 档要么退化成"全宽计算+掩码写回"（性能损失）要么不挂载（限制写法） |
| **墙3** | **fp8/fp4/mx/hf32/atomic store-mode 全线缺失**（C4/C5） | `gemm.h:137-226` blockscaled；`builtin.cc:401,411`；`mode.py:57-79` 注释点名 dav_3510；`simd_inst.h:17` `#if (__NPU_ARCH__ == 3510)` | 好消息：**模板层已有唯一的代际开关 `__NPU_ARCH__`**，bf16/fp8/fp4 的 `vec<T>` 特化在 2201 下自动消失；坏消息：**C++ 侧完全没有 arch gate**——`src/ascend/target.cc:37-38` 只声明了 `mcpu`/`arch` 两个属性选项，全仓 C++ 无一处读取 arch 值（`grep 3510` 只命中 4 条注释），`bisheng.py:93` 的 `ASCEND_NPU_ARCH`（默认 `dav-3510`）没有回灌给 codegen。**必须新建"C++ 侧 arch 感知通道"，这是所有 C 类处置的公共前置件** |
| **墙4** | **CANN 版本线冲突** | `philox_rng.h:7-9`：模板 vendored 自 **CANN 9.1.T560 / `__NPU_ARCH__==3510`**；而 910B 实勘镜像是 **CANN 8.5.2**（run de15 config） | 950 方言头在 8.5.x 线上大概率整体不存在（不只 `c_api/asc_simd.h`，还有 `pipe_*.h` 的 SIMD intrinsic 家族）。若 P1 证实，则"等 CANN 版本线"成为路线 A 的一部分，910B 侧只能用 legacy AscendC API 重写 tilelang 后端（与 tilelang 主仓上游的 Ascend 支持方向冲突，需上报） |

---

## §5 七算子 × 依赖面矩阵（本仓 `release/ascend/kernels/*_asc.py`）

| kernel | 向量载体 | 关键 950 依赖 | 受困于 | 首波可移植性 |
|---|---|---|---|---|
| `gemm_asc.py:134` | `T.SimtVF(threads=VEC_THREADS)` | `asc_mmad`/`asc_copy_l12l0*`（`gemm.h:67-105`）+ SIMT 面 | D2 + C1 | **中**：AIC 主路只需 D2 探针点头；向量 epilogue 需换载体 |
| `gemm_bwd_dw_asc.py:117` | `T.SimtVF` | 同上 + 累加（atomic？`mode.py`） | D2 + C1 + C5 | 中；`set_atomic` 关闭后 dW 累加策略需重设计 |
| `letter_readout_asc.py:161` | `T.SimtVF` | elementwise + reduce | C1 | 中低 |
| `add_ln_asc.py:140,289` | `T.SimtVF` ×2 + `T.reduce_sum(h_fr, s_fr, dim=1)` | **warp reduce → `asc_reduce_add`/`asc_shfl_xor`**（`reduce.h:32-88`） | C1（硬） | **低**：LN 归约正撞 SIMT 墙，须走"单流 UB 归约"新实现 |
| `rope_asc.py:107` | `T.SimtVF` | `float2/float4` + `make_float*` + 逐元素 | B1 + C1 | 中：数值面 B 类可解，载体需换 |
| `attn_sw_asc.py:189,201,242` | `T.SimtVF` ×3 | softmax → exp/max/reduce | B2 + C1 | 低（依赖 reduce + SFU 精度wrapper `vexp_1ulp/vln_1ulp`） |
| `gdn_asc.py:145,157`（旗舰） | `T.SimtVF` ×2 | 短卷积 + delta rule：滑窗 + reduce + （原 Conv2D 墙见 run 7793 R7） | C1 + C2 + 外部 torch_npu Conv2D 墙 | **最低**：既是本项目旗舰，又是 SIMT 墙最厚处 |

**判读**：七件 kernel **全部**用 `T.SimtVF`，没有一件是"纯 AIC 且零向量"的。故 design.md 里"gemm/dW/readout/add_ln 四件纯 AIC → 首批移植"的分层判断需要修正为：**"AIC 主体 + SIMT 向量尾巴"**——移植顺序上先解决 C1（向量载体降级方案），而不是先解决 AIC。P1-1 的前置条件应从 D2 探针改为 C1 决策。

---

## §6 P0-2 开局：探针清单（`probe910b.sh`，本目录）

| 探针 | 裁决对象 | 方法（离线即可准备，上卡执行） | 若失败的后果 |
|---|---|---|---|
| **P1 头存在性** | 墙0 / 墙4 | `find $ASCEND_HOME_PATH -name 'asc_*.h' -path '*c_api*'` + 列 `simt_api/` `pipe_*.h`；记录 CANN 版本线 | AIC/SIMD 全部 D 组转"必须重写"，工作量重估 |
| **P2 最小 TU 空编** | D2 | 只写 `extern "C" __global__ __attribute__((aic)) void k(){}` + `#include <tl_templates/ascend/common.h>`（**注意 reduce.h 未罩的缺口**）→ 看第一个错误落在哪个符号 | 直接暴露 §3-A 那个残留缺口，最小修补 = 把 `reduce.h` 也收进 `TL_ASCEND_SIMT` |
| **P3 intrinsic 点名编** | D1/D2/D3 | 单 TU 逐个引用：`asc_init; asc_copy_gm2ub_align; asc_mmad; asc_lock; vector_f32; vector_bool; PIPE_ALL; asc_loadalign; asc_add; asc_select`（每个包在 `#if __has_include`/模板 SFINAE 或直接报错看日志） | 决定向量侧走路 B 还是路 C（§3-C2） |
| **P4 legacy CCE** | C3 | `::vcvt/::vpack/::vsstb` 在 2201 上签名核对（`nd2nz_copy.h:108-111` 的 4 参调用形） | nd2nz 件必须换实现 |
| **P5 核拓扑** | C7 | `GetBlockNum/GetSubBlockNum/GetSubBlockID` 打印 + `__mix__(1,2)` 是否被 2201 接受 | 混合核模型重定，`asc_get_sub_block_id()` 发射点改写 |
| **P6 arch 宏与选项** | 墙3 | `__NPU_ARCH__` 在 `--npu-arch=dav-2201` 下取值；`bisheng --help` 的 arch 支持集 | 决定 `TL_ASCEND_SIMT` 是否可由 `__NPU_ARCH__` 自动推导（从而免掉 `-D` 注入链） |
| **P7 分形数值** | D4 | 16×16 bf16 小矩阵 `asc_mmad` 与 CPU 对拍；`TRANS_B=true/false` 两跑 | `gemm.h` 的 `C0`/`TRANS_B` 假定需改，静默错值风险 |
| **P8 事件枚举** | C8 | `asc_sync_notify(..., static_cast<event_t>(3))` 是否编译 + 事件数上限（对比 `kIntraCoreFlagLimit=8`） | codegen `EmitHardEventSync_` 改写为模板参数 |
| **P9 compat 头自检** | B 类 | 把 `port910b_compat.h` 单独 + `numeric_limits.h`/`debug.h` 编一个 TU | B 类判决作废，重设计 |
| **P10 vecadd 端到端** | P0-2 里程碑 | 最小 `T.copy`+`T.Kernel` bf16 vecadd，实编实跑数值对 | 甲路重估（回退混合栈为唯一主线） |

> 探针脚本已给出（`probe910b.sh`），**本机零执行**，全部命令需上卡手跑；建议顺序 P1→P2→P3 一次性拿到 90% 裁决，其余按需。

---

## §7 完备性论证与已知盲区（研究任务的诚实交代）

**已做的四道自证（原文，均可 `cd release/ascend/port910b` 复跑；同一份留档 `audit.log`）**

```
$ ./checklist.sh                       # audit：族穷举 → 反查 INVENTORY.md + 附表
# audit: grep 穷举 → 反查 INVENTORY.md + inventory_symbols.md   REPO=$WORKSPACE_ROOT/tilelang
  F1   950 代际头 include（c_api/simt_api）      uniq=6     unlisted=0
  F2   device 属性与函数限定宏                uniq=16    unlisted=0
  F3   SIMT 内建变量与核身份宏               uniq=11    unlisted=0
  F4   SIMD 寄存器类型 vector_*                  uniq=19    unlisted=0
  F5   类型代理（bf16/fp16/fp8/fp4/宽向量元组） uniq=29    unlisted=0
  F6   asc_* C API（950 MicroAPI 命名空间）     uniq=160   unlisted=0
  F7   双下划线内建（转换/存储/常量）   uniq=22    unlisted=0
  F8   硬件枚举/tag/mode 类型与事件          uniq=47    unlisted=0
  F9   C++ 标准库依赖（std::bit_cast / <bit>） uniq=3     unlisted=0
  F10  950 专属语言构造与 pass 名             uniq=15    unlisted=0
  F11  全局作用域 CCE 内建与 950 运行时件 uniq=19    unlisted=0
  F12  分形/布局与同步容量前提             uniq=7     unlisted=0
# 分类计数（附表逐符号）:
   A = 35
   B = 20
   C = 250
   D = 44
AUDIT PASS: 穷举 354 个符号 100% 见于清单（unlisted=0）

$ ./checklist.sh gap                   # 族外残差体检（独立于附表生成）
  RESIDUE × 80 行
  # 残差候选=159 族内已覆盖=79         → 80 条为人工复核队列（结论见下）

$ ./checklist.sh symbols               # 附表重生成（幂等：统计与上面 counts 一致）
written: .../inventory_symbols.md
A = 35
B = 20
C = 250
D = 44
总符号行数 = 349

$ clang++ -std=c++17 -I. host_selfcheck.cc -o /tmp/sc && /tmp/sc   # B 类实现的数值/形状实证
== port910b_compat.h host self-check ==
   backend: bf16=0 half=1
f16 sweep: 200000 samples + 7 edges, mismatch=0
SELFCHECK PASS: 0 failure (B 类 20 符号的等价性在主机成立)
   # 另两种后端配置（-DTL_PORT910B_{BF16,HALF}_BUILTIN=0/0 与 =1/1）同样 PASS；
   # -DTL_ASCEND_SIMT=1 时本头整体空转（已在主机验证 0 输出 0 错误）。

**gap 残差的 80 条人工复核结论**（逐条扫过，三类）：
① tilelang/模板自有宏与模板参数——`TL_*`、`SIMD_INST_DEFINE_{LOAD,BINARY,UNARY,COMPARE}`、`TL_SIMD_PINTLV_IMPL`、`TILELANG_CHECK*`、`C0/ROWS/COLS/SUB_K/STRIDE/TILE_K_SUB/VEC_4/WIDTH/VL_T/ATTR/BLOCK_IDX/RHS/Bin/…`（宏形参、`constexpr` 局部名）；
② C/C++/ACL 代际无关面——`std` `bit` `cstdint` `cstdint` `type_traits` `stdint` `assert` `printf` `__FILE__` `__LINE__` `__attribute__` `__builtin_inf{f}` `__builtin_huge_valf` `HUGE_VALF` `int{8,16,32,64}_t` `uint{16,32,64}_t` `enable_if_t` `ACL_SUCCESS` `ACL_RT_THREAD_LEVEL` `aclError` `aclrtGetLastError`；
③ **真实 950 面且已在本清单补录**——`DATA_BLOCK_COPY`（→C6）、`float8_e{4m3x2,5m2x2,8m0}_t`/`half_t`（→C4/B1）。
→ 复核后无新增未分类外部符号。

**本次写兼容层时产生的一次改判（记入审计链）**：`::print_var`/`::print_buffer` 原判 B4（以为需要 compat 提供全局打印件），
取证 `debug.h:24,30,260,266` 后确认它们是 `TL_DEFINE_ASCEND_DEBUG` 宏体内 `PrintTraits<T>::` 的**静态成员**，
F11 的 `::` 前缀把它们误捕成全局件 → 移入 **A6-SCAN-ARTIFACT**。改的是 `checklist.sh` 的 `classify()`（判决即代码），
重跑后 **B 22→20、A 33→35，总数 349 与 audit 的 354 不变**。

**已知盲区（明说，不藏）**

1. **族集合的完备性只覆盖"模板层 + codegen 发射面"**。python 面（`tilelang/ascend/language/simd.py` 1488 行里的 op 名）与 C++ pass 内部的 IR 属性串（如 `unit_flag_ctrl`/`sub_blockid` 注解名，`copy.cc:1002-1008`、`copy.h:39-40`）未单列成族——它们最终都会落到 F6/F10 的 `asc_*`/`nd2nz` 命中上，但**若某特性只存在于 python 层而 codegen 不发 `asc_*`，本清单看不见**。补法：加 F13 = `tl\.ascend_[a-z_]+` op 名族（`builtin.cc:39-451` 有完整注册表）。
2. **`grep` 是词法手段，看不见"语义新增"**：例如 910B 上 `asc_mmad` 存在但操作数顺序/分形约束不同（D4 型风险），清单会归 D 而不会归 C。这类"同名不同义"只能靠 P7 类数值对拍发现。
3. **codegen_pto.cc（另一套 PTO 后端发射器，61 处 hf32/45 处 Nd2Nz 命中在其内）未单独切族**，只在 F6/F8/F10 的计数里出现。若走 PTO 路径，依赖面要重新数。
4. **910B 侧任何"等价物"名字都是推断级**（本机无 CANN 头，已核实无 `/usr/local/Ascend`、无 vendored `c_api/`）。§3 表里所有"910B 等价物"字样在 P0-2 前不得当作事实引用。
5. `classify()` 是**规则而非真值**：兜底组 `D0-UNRESOLVED` 现已为 0（全部符号都命中显式规则），但规则本身可争议——例如 `vector_uint16_t` 我判 D（疑上游笔误），若上游头确实定义了它，应改 A7。改判流程：改 `classify()` → 重跑 `symbols` → 重跑 audit。

**给 P0-2 的开局建议（一句话版）**：带 `probe910b.sh` 上卡，先花 15 分钟把 P1/P2/P3 三个探针跑完，**再决定要不要动任何一行移植代码**；这三个探针的结果直接决定本域后续所有孙任务（P1-1/P1-2/P1-3）的工作量数量级。
