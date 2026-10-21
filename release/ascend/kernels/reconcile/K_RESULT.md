# K_RESULT — P1-4 接口 home 910B 编译合流 · 代理 K（cube/矩阵三件）

变更：p2-13-ascend-runtime（甲路）· 波次：P1-4 接口 home 910B 编译合流 · 容器：`cann910b-k`
判决方式：monkeypatch `tilelang.compile` 真编后换 no-op/skip 发射替身（无设备、发射隔离）；
绕设备门 patch `backend_available/_npu_present/active_backend`；判据 = `ascend_env.compiled_keys()`
含 `ascend|…` 且无 `ascend|` blocker。schema：spec-driven（嵌套子变更，OpenSpec CLI 不索引，直接读
design.md D-int1/D-num2 + tasks.md P1-4）。

## 判决总览

| 件 | ascend 编译 | 凭据 | cpu 回归 |
|---|---|---|---|
| gemm_asc.py | ✅ **PASS** | `K-GEMM ASC-COMPILE-PASS`；keys=`ascend|gemm[ascend]|n64k64b16x64x64arelu/anone`，blockers={}，simt_in_src=False，.o~2182B | ✅ `K-GEMM CPU-SEMANTIC-PASS`（relu 3.8e-6 / none 5.7e-6 allclose） |
| gemm_bwd_dw_asc.py（l0tr 主案） | ❌ **BLOCKED(§12 asc_fill_l1)** | 接口动态 m：blocker=`ascend|gemm_dw[ascend]|n64k64tc16b64x64`="Ascend device compilation failed"，原文 `error: use of undeclared identifier 'asc_fill_l1'`；静态同型旁证 `K-GEMM-BWD-DW STRUCTURE-SANITY-PASS`（.o~1425B, asc_fill_l1=0, mix=0） | cpu 正文逐字未动（dw_cpu_impl 未改） |
| lora_asc.py | ⚠ **VERIFY-ONLY / PARTIAL** | 无自有方言正文；前向腿(gemm)✅编译(keys `ascend|gemm[...n32k64...]`/`ascend|gemm[...n64k32...]`)；反向 dW 腿❌承 §12（2× `ascend|gemm_dw` blocker）；整件全绿待 dW §12 补齐 | lora 为纯 torch 组合，无方言回归面 |

## ① 完成情况（逐条附凭据）

- [x] **gemm_asc：ascend 路径 910B 编译合流** —— 重写 `gemm_asc_impl` 为已验证的 Q1 结构（Cube 单工
  主乘加 + alloc_shared(UB) 累加 + 去 `T.clear` 改 `clear_accum=ko==0` + 标量 `T.serial` 收尾 bias/relu
  + 动态 m + fp32），去除 950 载体的 `T.SimtVF/T.Parallel`。验证：`python3 ascend/kernels/reconcile/verify_gemm.py`
  → `K-GEMM ASC-COMPILE-PASS`（4 例 f16/f32×relu/none 全驱动，2 独立 key 出 .o，无 SimtVF）。
- [x] **gemm_asc：cpu 语义不回归** —— `gemm_cpu_impl`/`plan`/`run` 逐字保留；`verify_gemm.py --cpu`
  → `K-GEMM CPU-SEMANTIC-PASS`，relu/none 与 `gemm_kernel._eager` 对拍 max_abs_diff < 6e-6。
- [x] **gemm_bwd_dw_asc：ascend 路径重写为 l0tr 主案** —— 转置下移到 L1→L0 装填
  （`asc_copy_l12l0a/b_transpose`），mad 恒吃 NT，L0C→GM 直出（`asc_copy_l0c2gm`），出口自然朝向。
  DSL 形态正确（静态同型旁证可编，`.o~1425B` 无 asc_fill_l1、无 mix）。**接口动态 m 编译受阻于
  §12 缺件 asc_fill_l1 → 如实报红，未假绿。**
- [x] **lora_asc：verify-only 结论** —— 本件无方言正文（全组合 gemm/dW），不改文件；判决拆层：
  前向腿可编、dW 腿承 §12 阻。`verify_lora.py`（no-op 发射让 backward 真正跑到 dW 腿取证）。
- [ ] dW **接口动态 m 编译**：未完成——被 §12 `asc_fill_l1` 缺件挡住（外部阻塞，见 ②），非 DSL 形态问题；
  补件后应即编过。
- [ ] P1-4（九件全绿）：本代理仅覆盖 cube/矩阵三件，另六件由 H/其他代理覆盖，**不代勾**（tasks.md 亦不在本代理白名单）。

## ② 错误与阻塞（含已解决的）

- **已解决**：gemm ascend 原 SimtVF 版编译失败（D-int1 复现）。根因两条：(a) 910B 方言无 `T.SimtVF`/`T.Parallel`；
  (b) 累加若放 L0C 则搬不回 UB——910B 硬件 `FixpipeL0C2UBImpl = assert(false)`（port910b 卷宗 b10）。
  → 处置：**已修复**（操作数与累加全落 UB(shared)，主乘加走 `T.Cube()` 单工，收尾标量 `T.serial`，
  fp32 的 `T.clear` 在 UB 生成坏 float2 store（`no viable overloaded '='`）→ 去 clear 改 `clear_accum=ko==0`）。
- **外部阻塞（需 §12 owner 裁）**：dW l0tr 接口动态 m → `CCE error: use of undeclared identifier 'asc_fill_l1'`
  @ 生成码 `tl_kernel.asc:12`。codegen 在每 `asc_copy_gm2l1_nd2nz` 之后对本 token 块未填满的 L1 尾区发射
  `asc_fill_l1(__cbuf__ uint32_t*, (uint32_t)value, {.repeat, .blk_num, .dst_gap})`（3 参 DMA-fill），trunk
  `port910b_compat.h`（971 行）无此件。标量 for 填充不可行（命中 `only __ubuf__/__gm__/local can be dereferenced`，
  `__cbuf__` 不可标量写，同 madta G-D2）。→ 处置：**未自行改 trunk**，写 `reconcile/compat_patch_K.h` 精确上报，编排者合。
- **harness 噪声（非缺陷）**：lora `merge` no-op 发射后残留 `RuntimeError: expected m1 and m2 to have the same
  dtype, float != ...` —— 系 no-op 发射留未初始化缓冲触发的下游 torch dtype 检查，**非编译失败**（merge 的 gemm
  腿已在 keys 中编出）。gemm f16 走前向时 kernel 实为 fp32 声明（impl 内层硬编码 fp32，plan 门 C.dtype==A.dtype），
  f16 与 f32 共用同一份 fp32 产物，编译判决与位宽无关。
- **环境**：容器默认 shell 未 source CANN → "Cannot find the bisheng compiler"。已固定每条命令前置
  `source /usr/local/Ascend/cann-8.5.0/set_env.sh; export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler
  ASCEND_NPU_ARCH=dav-2201`。@T.prim_func 不可经 stdin(-c/heredoc) 现场定义（inspect.getsourcelines 取不到源码）——
  故判决脚本一律落 .py 文件（reconcile/ 下），随 `/work` bind-mount 进容器，无需 docker cp。

## ③ 疑惑点与自行决策（凡"没问但自己定了"的事）

- **gemm 存储层落点**：任务参照给 e2e_cube(direct mad, UB) 与 a2_probe(shared.l1) 两路。我选 **Q1=UB-direct**
  （操作数/累加 alloc_shared + T.Cube），非 shared.l1(L1/L0C) 路——因 910B L0C→UB 硬件不支持，收尾标量化
  只能在 UB 做。据 D-num2 "L0C→UB 致命缺陷" 判 shared.l1 路收尾无路 → UB-direct 是唯一能编又留标量收尾的形。
- **fp32 判决形状**：受 plan 门 `C.dtype==A.dtype` 约束，接口 impl 内层硬编码 fp32；故判决以 fp32 出口自证，
  另叠 f16 入口例（fp16→fp32 kernel）验证白名单不炸。未改 plan/run 门（契约层禁改）。
- **dW 主案取舍**：波次令"主案 l0tr；合不了如实报、勿假绿"。故 ship l0tr（结构正确、唯 §12 挡编译），
  **未**采纳可编的 variant-D（UB-direct transpose_A，非主案、数值属 backlog 族）来凑绿；D 变体仅作对照留记录。
- **lora 不改文件**：无方言正文，任务令"先验证是否已可编"。结论=其可编性完全承接 gemm/dW 两件，本波如实
  分层报，未新增任何 lora 代码。
- **verify dummy 用 no-op（非抛 Skip）**：为使 backward 的多腿链跑到底、真正触发 dW 腿取证；无设备故 no-op
  发射不影响编译判决。

## ④ 偏离记录（与计划/spec 不一致之处）

- **gemm ascend impl 偏离原设计表述**：模块 docstring 原称 ascend 那份"L1 进操作数、L0C 做累加、UB 收尾"。
  实际落地改 UB-direct（L0C/L1 收尾搬不动）。已同步改 docstring 记录原因（b10 L0C→UB 不支持 + 无 SimtVF）。
  属"被硬件约束逼出的实现细化"，未违反 910B 方言铁律。
- **dW impl 偏离原 SimtVF 版**：原 `dw_asc_impl` 用 `T.Pipelined`+`T.SimtVF`+UB 转置写回（950 载体，910B 编不出）。
  按 D-int1 重写为 attempts/D 的 l0tr 主案。cpu 正文未动。
- **本波只判编译**：cube 数值 rel、V1/V2/V3 单位、清账口径数值 → 待卡（波次边界允许）。

## ⑤ 产物清单

- 修改（白名单内）：
  - `release/ascend/kernels/gemm_asc.py`（重写 `gemm_asc_impl` + docstring；cpu 正文/plan/run 未动）
  - `release/ascend/kernels/gemm_bwd_dw_asc.py`（重写 `dw_asc_impl` 为 l0tr + docstring；cpu 正文/plan/run 未动）
  - `release/ascend/kernels/reconcile/compat_patch_K.h`（§12 asc_fill_l1 缺件精确上报）
- 新增（白名单内）：
  - `release/ascend/kernels/reconcile/K_RESULT.md`（本文件）
  - `release/ascend/kernels/reconcile/verify_gemm.py`（含 `--cpu` 回归）
  - `release/ascend/kernels/reconcile/verify_dw.py`（接口动态 m 判决 + 静态同型结构旁证）
  - `release/ascend/kernels/reconcile/verify_lora.py`（no-op 发射取三入口全腿证据）
- 未改（尊重边界）：`lora_asc.py`（无方言正文）、`gemm_kernel.py`/`gemm_bwd_dw_kernel.py`/`ascend_env.py`（契约层）、
  trunk `compat.h`/`gemm.h`/`common.h`/`debug.h`、`compile_all_asc.py`、其他 op `_asc.py`、`sys1/`、`attempts/`。
- 已清理：临时探测件 `_probe_K_*.py`（14 个）全部删除，白名单外无残留。

## 复跑命令（容器内，`/work`=release bind-mount）

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201
cd /work
TILELANG_CACHE_DIR=$(mktemp -d) python3 ascend/kernels/reconcile/verify_gemm.py            # K-GEMM ASC-COMPILE-PASS
TILELANG_CACHE_DIR=$(mktemp -d) python3 ascend/kernels/reconcile/verify_gemm.py --cpu       # K-GEMM CPU-SEMANTIC-PASS
TILELANG_CACHE_DIR=$(mktemp -d) python3 ascend/kernels/reconcile/verify_dw.py               # FAIL(asc_fill_l1)+STRUCTURE-SANITY-PASS
TILELANG_CACHE_DIR=$(mktemp -d) python3 ascend/kernels/reconcile/verify_lora.py            # PARTIAL/VERIFY-ONLY
```

## 给编排者的一句话

gemm 前向（cube 单工 + UB 收尾）已 910B 编译合流、cpu 不回归；dW l0tr 主案 DSL 正确但**唯一**卡在
§12 缺 `asc_fill_l1`（DMA-fill 尾填充件，精确签名/现象见 compat_patch_K.h）——补齐即应编过；lora 无自有
方言、可编性承接上述两件，dW 未补齐前整件不得判全绿。本波未假绿、未越界改 trunk。
