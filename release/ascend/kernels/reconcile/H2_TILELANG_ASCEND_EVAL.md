# H2 — 底座对齐评估：tilelang-ascend 是否应取代"私 fork tilelang main"

> 任务：p2-13 / 执行代理 H2。**不占卡、不改产品件、不 git 写我们的仓**。
> 结论一句话：**tilelang-ascend 是"另一条 Ascend 方言"，不是我们 fork 的替代品**。它治不了我们 §12 cube 路 / GDN 反向；但它把 GDN chunk 前向族以正统 cube 路（`T.gemm_v0`）实现好了，可作为**逐算子移植源**。建议 **B（仅采 GDN chunk 族实现、底座不动）+ C（佐证回馈）**，不建议 A（全面切底座）。

三态标注：**✅实测**（本次亲跑，命令与输出见文末 §R）；**⚠️提及未取**（README/上游/他人转述，本次未复现）；**❓推断**（由已测事实推出，未直接验证）。

---

## 0. 本评估的被评对象（一手锚点）

| 项 | 值 | 来源 |
|---|---|---|
| 仓库 | `tile-ai/tilelang-ascend`（非 fork，描述 "Ascend TileLang adapter "） | ✅ `gh api repos/tile-ai/tilelang-ascend` |
| 默认分支 | `ascendc_pto` | ✅ 同上 |
| 本次 clone 版本 | HEAD `83b0ece6c2575f483f1364c5c431bd8ccf0b9819`（2026-10-09 10:50 +0800, `[Example] Organize xAttention examples and add Expert v2 (#1864)`） | ✅ `git rev-parse HEAD` |
| 版本号 | `VERSION` = `0.1.4` | ✅ `cat VERSION` |
| 最近 push | `2026-10-09T02:50:37Z` | ✅ `gh api` |
| 包内 ascend 体量 | `src/transform/` 17 个 `ascend*` pass；`src/target/` `codegen_ascend{,_pto}.{cc,h}`、`rt_mod_ascend{,_pto}.cc`；`tilelang/language/ascend.py`+`ascend_tile.py`(114KB)+`reduce_ascend.py`；examples 204 py / examples_experiment 37 py | ✅ `ls`/`find` |
| TVM | **vendored submodule**：`3rdparty/tvm` → `github.com/TileLang/tvm`，pin `c2921fdaf795b1103d21abc962e83a209c7258d7` | ✅ `.gitmodules` + `git -C 3rdparty/tvm rev-parse HEAD` |
| 我方对照 | 私 fork `tilelang` main，HEAD `994b44ec`(2026-09-30, `[Ascend] Add PTO backend for Ascend950 (#3310)`)，VERSION `0.1.15`，**自有 `tilelang/ascend/` 包**，`port910b_compat.h` 1248 行 | ✅ 本地仓 |

**核心定位差异（贯穿全篇）**：
- 我方 fork = **在 tilelang main 上给 910B 加算子**（`import tilelang.ascend.language as T`；950 内联函数用 `port910b_compat.h` 模拟打到 910B）。
- tilelang-ascend = **另一套 DSL**（`from tilelang import language as T`，`T.gemm_v0` / `alloc_L1,L0A/B/C,ub` / pass `TL_ASCEND_*`），**没有 `tilelang/ascend/` 包、没有 `tilelang/cpu/` 包**（✅ `ls tilelang/`）。二者**包名相同但内里不同**，不能原地替换。

---

## 1. 构建可行性（问题一）——✅ 可构建，17 min < 40 min 时间盒

### 1.1 结论
在本机 `cann910b-l` 容器（镜像同 `cann910b-*`，aarch64，CANN 8.5.0，Python 3.10.12，gcc 11.4，cmake 3.22.1，nproc=4，RAM 9G，**无 `/dev/davinci*` 设备**）里，**从零 clone → 完整构建成功，`MAKE_EXIT=0`**，产出 `libtvm.so`(74MB) / `libtilelang.so` / `libtilelang_module.so`。墙钟 ≈ **1027s（17.1 min）**（含一次并行 make 被 OOM kill 后重跑，见 §1.3）。**在 40 min 时间盒内。**

### 1.2 依赖与构建假设（读 `install_ascend.sh`）
- ✅ **TVM 是 vendored submodule**，不是系统依赖：`.gitmodules` 指 `3rdparty/tvm`=`github.com/TileLang/tvm` pin `c2921fd`。`install_ascend.sh:174` `git submodule update --init --recursive`；`:189` `bash 3rdparty/patches/apply_tvm_patches.sh`（打 TVM 补丁）。
- ✅ **CMake 路径**：`install_ascend.sh:197/204` 拷 `3rdparty/tvm/cmake/config.cmake` 进 `build/`，`:210` 追加 `set(USE_ASCEND ON)`，`:219` `cmake ..`，`:231-236` `CORES=$(nproc); MAKE_JOBS=$(( CORES*50/100 )); make -j${MAKE_JOBS}`（4 核 → 默认 **-j2**）。
- ✅ **bisheng 路径假设**：构建**不需要** bisheng；bisheng 只在**编 kernel**时由 `libgen.py` 触发，靠 `ASCEND_HOME_PATH`（或 `ASCEND_HOME`）定位（`libgen.py:36-49`；缺失时报 "ASCEND_HOME_PATH or ASCEND_HOME is not set"）。容器 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 后 `bisheng` = `/usr/local/Ascend/cann-8.5.0/bin/bisheng`（clang 15.0.5）。✅
- ✅ **`--enable-incremental` 存在**（`install_ascend.sh:21/31`），另支持 `--enable-llvm/--enable-shmem/--enable-coverage`。
- ✅ **CANN 8.5.0 ≥ README 要求 8.3.RC1**。

### 1.3 实测踩坑（原文 + 归因）
1. **链接失败：`collect2: fatal error: cannot find 'ld'`**（归因：**环境**）
   - 根因：TVM `cmake/utils/Linker.cmake` 在 `USE_ALTERNATIVE_LINKER=AUTO` 下自动加 `-fuse-ld=lld`，而容器**未装 lld**。
   - 处置：向 `build/config.cmake` 追加 `set(USE_ALTERNATIVE_LINKER OFF)` 后重跑 cmake + make → 通过。
2. **首次并行 make 被 OOM kill**（归因：**环境**，RAM 9G）
   - 原文：`c++: fatal error: Killed signal terminated program cc1plus`。
   - 处置：降并行度重跑 → `[100%] Built target tilelang_module` / `MAKE_EXIT=0`。
3. **submodule 递归克隆过大**（归因：**仓库设计**）
   - `.gitmodules` 含 NVIDIA `cutlass`、ROCm `composable_kernel` 等与本后端无关的大件。改为**按需 init**：`3rdparty/{dmlc-core,dlpack,rang,libbacktrace}` + `tvm` + `catlass` + `shmem` + `pto-isa`。
   - ❓ 提示：完整 `--recursive` 会很慢/占盘；构建 AscendC 路只需 `catlass`+`shmem`，PTO 路还需 `pto-isa`。

### 1.4 产物
```
/tlasc/build/tvm/libtvm.so           74371224 B
/tlasc/build/libtilelang.so           8443360 B
/tlasc/build/libtilelang_module.so    8440576 B
```
（`/tlasc` = 宿主 `refs/tilelang-ascend` 的容器挂载点；构建在宿主盘，未污染我们仓。）

---

## 2. dav-2201 编译面（问题二）——✅ 能编；**#1830 编译侧根因复现**；dav-3510 覆盖在本工具链**不可用**

### 2.1 关键源码事实（决定"能不能编"的一行）
- **AscendC 路**（`target=="ascendc"` 或 `"auto"`）：`libgen.py:152-183` **硬编码 `--npu-arch=dav-2201`**，`-xasc`，链 `-lruntime -lascendcl -ltiling_api ...`，`-I{ASCEND_HOME_PATH}/include` + `-I{TL_ROOT}/3rdparty/catlass/include` + `-I{TL_ROOT}/3rdparty/shmem/include`。**该 arch 与 `platform` 无关**——传 A5(950) 也一样 dav-2201。
- **PTO 路**（`target=="pto"`）：`libgen.py:184-186` `--cce-aicore-arch = dav-c310(A5) / dav-c220(else)`，`-xcce`。
- `compile_flags` **追加在命令末尾**（`libgen.py:258`，"bisheng 是 last-wins"）→ 可覆盖 arch（但见 §2.3）。

### 2.2 compile-only 实测（容器无设备）
cython adapter 在 `__init__` 里就调 `lib_generator.compile_lib()`（`adapter.py:269-271`），**拿到张量前就触发 bisheng**，故可 compile-only 验证。

> ⚠️ **compile-only 并非"真无设备可跑"**：`adapter.py:433` **无守卫地** `torch.device("npu")`，`cython_wrapper.pyx:43`(`__cinit__`) 无守卫访问 `torch.npu`。本容器 torch=2.14.1+**cu130**、**无 torch_npu**（pip 最高 `torch-npu 2.13.0rc1`，与本 torch ABI 不匹配→不可装），故探针加了最小 shim（`torch.device("npu")→cpu` + 假 `torch.npu`）才到达 bisheng。**这条对 CI 化是硬约束**：要么容器装匹配的 torch_npu，要么保留同类 shim。

| 用例（`ASCEND_NPU_ARCH=dav-2201`, `TL_PLATFORM=A2`, fresh `TILELANG_CACHE_DIR`） | 结果 | 证据 |
|---|---|---|
| `examples/gemm/example_gemm.py` 的 `matmul(256,256,256,128,256,64)` | ✅ **COMPILE-OK 3.4s**，kernel src 2346 B，**rc=0** | bisheng 命令见下 |
| `examples/linear_attention_and_rnn/gdn/` **六件全部**（cumsum / kkt / solve_tril / wy_fast / chunk_h / chunk_o） | ✅ **6/6 OK**，各 2.5–2.9s，src 2594–12328 B | §R 命令 |

实测 bisheng 命令行（gemm，节选关键段）：
```
bisheng --npu-arch=dav-2201 -std=c++17 -xasc
  -I.../ascend-toolkit/latest/include ... -I/tlasc/3rdparty/catlass/include
  -I/tlasc/3rdparty/shmem/include -I/tlasc/3rdparty/shmem/src/device
  -DBACKEND_HYBM -I/tlasc/tilelang/../src -L.../lib64 ... -fPIC --shared <tmp>.cpp -O2 -o <tmp>.so
[probe][so] /tmp/xxxx.so rc=0
```

**特别记录 —— 是否撞 #1830 类 dav-2201 问题？**
- **A2(910B) 正常路：不撞**（dav-2201 正是 910B 代际，rc=0）。✅
- **A5(950) 模拟：编译侧根因复现**。把 `platform=A5`（`TL_PLATFORM=A5`）后，AscendC 路**仍**用 `--npu-arch=dav-2201`（`libgen.py:155` 不看 platform），rc=0 但**产物是 910B 代际 arch**。这正是 **#1830 标题**所述：
  > `#1830 [open] [Bug] [A5] Device path compiles with --npu-arch=dav-2201 (910B-generation arch): device fault 507015 on real Ascend 950 (dav-3510) silicon; ...`（✅ `gh api .../issues/1830`，2026-09-22 建，**仍未闭**）
  即：**编译能过，错在产物 arch 与硅片代际不匹配**。本容器无 950 硅片，**device fault 507015 无法在此复现**（⚠️ 仅上游转述）。
- **"dav-3510 覆盖"缓解路：本工具链不可用**。给 `compile_flags=["--npu-arch=dav-3510"]` 后命令尾部变成 `... -O2 --npu-arch=dav-3510 -o ...`（last-wins），**bisheng 直接拒**：
  ```
  bisheng: error: Unsupported NPU architecture or soc: NPU arch is not supported!!!
  ```
  隔离直测（`-xasc` 只喂 arch）：
  ```
  bisheng --npu-arch=dav-3510 -xasc ... -c u.cpp   → rc=1  (Unsupported NPU architecture or soc)
  bisheng --npu-arch=dav-2201 -xasc ... -c u.cpp   → rc=0
  ```
  → **CANN 8.5.0 / bisheng clang 15.0.5 下 `-xasc` 仅认 dav-2201**。**#1830 里"dav-3510 override 能编但算错值"的说法，在本工具链连"能编"都做不到**（⚠️ 上游环境 CANN 版本可能不同）。**结论：#1830 在 950 上未有可用绕过，950 路 = 当前不可用。**
- **PTO 路(A5) 编译失败**（归因：**依赖/头文件**）。`target=pto, platform=A5` → `--cce-aicore-arch=dav-c310`，但 `3rdparty/pto-isa` 头与本 CANN 不兼容：
  ```
  /tlasc/3rdparty/pto-isa/include/pto/npu/a5/TCvt.hpp:66:7: error: no member named 'RoundAType' in namespace '__cce_simd'; did you mean '::RoundAType'?
  ... (RoundCType/RoundFType/RoundOType/RoundRType/RoundZType) 6 errors generated.
  ```
  → PTO 路在本容器**编不过**（pto-isa 版本 ↔ bisheng/`__cce_simd` API 失配）。

### 2.3 小结（问题二）
- ✅ AscendC/dav-2201 路在本容器**能编**（gemm + GDN 六件全过），是**当前唯一可用路**。
- ✅ **#1830 编译侧根因坐实**：A5 也走 dav-2201；且本工具链**无 dav-3510 后路**（`-xasc` 拒收）→ 950 目标**当前无解**。
- ✅ PTO 路在本容器因 pto-isa/CANN 失配编不过。
- ❓ 我们若走 910B（A2）目标，dav-2201 路是**可用**的；§12 cube 507015 的坑在**我们 910B 标量路 vs cube 路**之间，与 #1830（950）不是同一问题。

---

## 3. GDN chunk 族在树内的真实形态（问题三）——✅ 树内有、**examples 级、fwd-only、fp16**

### 3.1 落点（`grep -r` 全树）
- **主实现**：`examples/linear_attention_and_rnn/gdn/`（六件 + 装配）：
  | 文件 | 内核函数 (`@tilelang.jit`) | 关键行 |
  |---|---|---|
  | `gdn_chunk_cumsum.py` | `cumsum_ker(B,H,L,C, accum_dtype="float")` | `:14-15`；wrapper `chunk_cumsum` `:44` |
  | `gdn_chunk_scaled_dot_kkt.py` | `kkt_ker(B,H,L,DK,C,BK, dtype="float16", accum_dtype="float")` | `:16-17`；wrapper `kkt` `:82` |
  | `gdn_solve_tril.py` | `solve_tril_ker(B,H,L,C, dtype="float16", accum_dtype="float")` | `:18-19`；wrapper `solve_tril` `:65` |
  | `gdn_wy_fast.py` | `wy_fast_ker(B,H,L,DK,DV,C,BK,BV, dtype="float16", accum_dtype="float")` | `:15-16`；**cube 路** `T.gemm_v0(...)` `:83,:88`；wrapper `wy_fast` `:93` |
  | `gdn_chunk_h.py` | `chunk_h_ker(B,H,L,DK,DV,C,BK,BV, dtype="float16", accum_dtype="float")` | `:15-16`；wrapper `chunk_h` `:129` |
  | `gdn_chunk_o.py` | `chunk_o_ker(B,H,L,DK,DV,C,BK,BV, dtype="float16", accum_dtype="float")` | `:15-16`；wrapper `chunk_o` `:120` |
  - 装配：`examples/linear_attention_and_rnn/gdn_full.py`（`kernel_chunk_gdn` `:47`）；优化版 `opt_gdn_full.py` + `opt_gdn/opt_gdn_*.py`。
- **融合/变长/GQA 实验**：`examples_experiment/chunk_gated_delta_rule/{chunk_gated_delta_rule.py, expert_chunk_gated_delta_rule.py, design.md}`（`chunk_gated_delta_rule(...)` `:36`，`workspace_idx=[9,10,11,12]`，含 `cu_seqlens` 变长 + GQA）。
- **是 examples，不是库件**：✅ 六件均位于 `examples*/` 下，未进 `tilelang/` 包（无法 `import` 复用，只能拷源码）。

### 3.2 fwd-only（✅ 实测）
`grep -rniE "bwd|backward|grad|reverse" examples/linear_attention_and_rnn/gdn/` → **零命中**。
`examples/linear_attention_and_rnn/README.md:11`：**"the forward pass of GDN is divided into six processes"**。README 仅在前向列表里出现 `wy_fast` 等；**无反向**。
反向在**上游 issue 里"迁移中"**（不是已合代码）：
- ✅ `#1191`（`feat: update chunk_gated_delta_rule`）**已关闭**（2026-07-18）。
- ✅ `#1799`【open】`[Feat] Migrate add Chunk Delta H operator`（2026-09-15）
- ✅ `#1827`【open】`[Feat] Migrate and add new WY Fast operator`（2026-09-22）
- ✅ `#1866`【open】`[Feat] Migrate add Chunk O Forward operator`（2026-10-09，**前向**还在迁）
- ✅ `#1802`【open】`[Feat] Migrate add chunk_o_bwd operator for Ascend NPU`（2026-09-15，**反向**）
→ **结论：树内 GDN chunk 族 = 纯前向；反向（含 `chunk_o_bwd`）仍在 issue 阶段。**

### 3.3 dtype（✅ 实测）
六件默认 `dtype="float16"`、`accum_dtype="float"`(fp32)（§3.1 表内 `:行`）。**未见 fp32/bf16 分支**。⚠️ `dtype` 是形参，理论上可传 `bfloat16`，但**树内未提供 bf16 用例、也未验证**（且上游 `#1762` bf16 标量转换本身还 open，见 §5 风险）。

### 3.4 API 形态（✅ 实测）
- 纯 `@tilelang.jit(out_idx=..., workspace_idx=[...], pass_configs={TL_ASCEND_AUTO_SYNC:AUTO_CV_COMBINE:AUTO_CV_SYNC:MEMORY_PLANNING})` 装饰的**设备内核**，签名 `*_ker(shape 参数, dtype=...)`。
- **非 torch 集成算子**：带一个手写 `def 名(...)` wrapper（如 `chunk_o(q,k,v,s,g,C,BK,BV)`）负责 `torch.empty` 分配 + 调内核；`chunk_gated_delta_rule.py` 里带 `cu_seqlens` 但仍是**函数级调用**，非 `torch.nn.Module` / 非 autograd。
- kernel 全用 `T.Kernel(n, is_npu=True) as (cid, vid)`（`kernel.py:215` 起，`is_npu` 参数默认 False）。

---

## 4. 我们资产的迁移成本（问题四）——**九件需"半重写"**；判决件**部分可迁**

### 4.1 两套 DSL 的硬差异（✅ 实测对照）

| 维度 | 我方（fork main + port910b_compat） | tilelang-ascend | 迁移影响 |
|---|---|---|---|
| 方言导入 | `import tilelang.ascend.language as T`（`*_asc.py` 多行；`ascend_env.py:87`） | `from tilelang import language as T`（**无 `tilelang/ascend/` 包**，✅ `ls tilelang/`） | **改 import** |
| 编译 target | `target="ascend"`（`ascend_env.py:75`） | 合法值 `{auto,cuda,hip,webgpu,c,llvm}`（`target.py:13-20`）；Ascend 走 `target in ["ascendc","pto"]→"llvm --keys=ascend"`（`target.py:99`） | **改 target** `"ascend"→"ascendc"` |
| Kernel 帧 | `with T.Kernel(N) as bx:`（九件 12+ 处实测） | `with T.Kernel(N, is_npu=True) as (cid, vid):`（`kernel.py:215`；examples 均此法） | **改帧形态**（解包 1→2 元组） |
| 本地/标量缓冲 | `T.alloc_local`（九件 **28** 处） | **已移除**：`T.alloc_local` 触发 `__getattr__` 报错（`language/__init__.py:206-214`），改用 `T.alloc_var`/`T.alloc_shared`/`T.alloc_ub` | **28 处需改写** |
| 立方（cube）原语 | `T.gemm`（**5** 处）；`T.Vector`(**14**)、`T.Cube`(**4**)、`T.SimtVF`(**2**) | `T.gemm_v0(A_L1,B_L1,C_L0,...)`（`language/ascend.py:418`；A/B 在 L1、C 在 L0C）；`T.alloc_L1/L0A/L0B/L0C/ub`（`language/__init__.py:47-51`） | **分歧原语必须重写** |
| pass 属性 | 无（我方非此套） | `TL_ASCEND_AUTO_SYNC`(`tl.ascend_auto_sync`)、`TL_ASCEND_MEMORY_PLANNING`、`TL_ASCEND_AUTO_CV_COMBINE`、`TL_ASCEND_AUTO_SYNC_VS`…（`transform/pass_config.py:38-73`） | 需**新增**属性声明 |
| 通用原语 | `T.serial/T.copy/T.alloc_shared/T.Parallel/T.Pipelined/T.Kernel` | **同名同义均存在**（`language/{copy_op,allocate,parallel,pipeline,kernel}.py`） | **这部分可平移** |

我方九件 `T.*` 用量（✅ 全量 grep）：`T.serial 155`、`T.copy 45`、`T.Kernel 31`、`T.alloc_local 28`、`T.alloc_shared 27`、`T.Vector 14`、`T.gemm 5`、`T.Parallel 5`、`T.Cube 4`、`T.Pipelined 2`、`T.SimtVF 2`、`T.Persistent 1`。

### 4.2 "语义可平移 vs 要重写"逐件判断（❓ 以 §4.1 差集推）
- **标量主导件**（`add_ln_asc` / `letter_readout_asc` / `rope_asc` / `gdn_conv_asc`）：正文主要在 `T.serial`+`T.copy`+`T.alloc_shared`，**通用原语可平移**；但 `T.alloc_local` 需替 `T.alloc_var`、`T.Kernel(N) as bx` 需改帧、import/target 需改。→ **半重写（机械改，风险低）**。
- **含分歧原语件**（`attn_sw_asc`、`gdn_asc`、`gemm_asc`、`gemm_bwd_dw_asc`）：用到 `T.Vector/T.Cube/T.SimtVF/T.gemm`——**这些在 asc 仓语义不等价**（asc 的 cube 是 `gemm_v0` 走 L1→L0A/L0B→L0C，SIMT/VF 是 950 概念，910B 无）。→ **必须重写内核正文**，且 `T.SimtVF/T.Vector`（950 SIMT 载体）在 910B 上本就不该用（我方注释亦证：SimtVF/Parallel 在 910B 实测编不出）。

> **一句话**：**通用原语（serial/copy/alloc_shared/Parallel）可平移；分歧原语（Vector/Cube/SimtVF/gemm/alloc_local）必须重写**。九件整体＝**"换 import+target+帧"是机械活，但 cube/vector 正文要按 asc 的 `gemm_v0`/`alloc_L1/L0C` 重写**。

### 4.3 判决件迁移（逐个）
| 资产 | 新底座上 | 依据 |
|---|---|---|
| `TILELANG_CACHE_DIR` | ✅ **直接可用** | `kernel_cache.py:46` `Path(TILELANG_CACHE_DIR)`；`TILELANG_CACHE_DIR` 是同名 env（✅ 实测用它跑了各次 fresh cache） |
| `tilelang.compile(target="ascend")` 语义 | ❌ **不同形** | 我方 `target="ascend"`（`ascend_env.py:75`）；asc 合法 target 无 `"ascend"`（`target.py:13`），要用 `"ascendc"`/`"pto"`。**判决件里所有 target 开关要改** |
| `ascend_env._compile_probe`（探针：`T.Kernel(1)`+`T.copy`+`T.serial`+`alloc_shared`） | ⚠️ **要适配** | 探针体（`ascend_env.py:171-188`）用的是**通用原语**，**原则上可平移**；但要 (a) `target="ascend"→"ascendc"`；(b) `T.Kernel(1) as bx`→`as (cid,vid)`；(c) 仍需 `is_npu=True`；(d) 仍受 §2.2 的 `torch.device("npu")` 约束 |
| `backend_available` / `_npu_present`（`torch.npu.is_available`） | ⚠️ **要适配** | 我方按 `torch` 判 `torch.npu`（`:101`）；asc 走 `torch.npu` 且 cython adapter 亦硬依赖 `torch_npu`（§2.2） |
| `make_key`/`get_compiled`/`blockers`/`compile_count`（缓存与回退账） | ✅ **框架无关，可迁** | 纯 Python 进程态，不含方言/后端假设（`ascend_env.py:225-294`） |
| gradcheck_card / run_delta_card / dtype 矩阵 / KERNEL-USED 三证 / window 脚本 | ⚠️ **要适配** | 这些是**门面/编排**，与后端解耦部分可留；凡是"触到 `target="ascend"`、`import tilelang.ascend.language`、`T.Vector/T.Cube`"的点都要改。**`import tilelang` 本身仍成立**（两仓包名都是 `tilelang`） |

---

## 5. 裁决建议（问题五）

### 5.1 两个已知风险的量化（✅ 实测）
- **#1830 未闭**：✅ `open`，标题即"950/dav-2201 代际错配 device fault 507015；dav-3510 override 算错值"。**且本评估追加**：本工具链 `-xasc` **根本不接受 dav-3510**（§2.2）→ 950 目**当前无可用 arch 后路**。
- **成熟度**：✅ `gh api` → `open_issues_count=534`；拆分 `is:issue is:open=302` + `is:pr is:open=230`（=532，与 534 差 2，搜索 API 时滞）。**GDN 反向尚在 issue（#1799/#1802/#1827/#1866 open），前向也仍在迁（#1866）**。bf16 标量转换（#1762 **open**，= 我们 C2 发现）在上游**未修**。
- ⚠️ 补充：README 承诺（`T.gemm_v0` 正统 cube 路 / Flash / Sparse / Linear Attention & RNN / `bash install_ascend.sh`）**与实测一致**（我们实测了 `gemm_v0` 六件编译 ✅），但 **README 未承诺反向、未承诺 950 可用**——不要把它读成"生产就绪"。

### 5.2 三选项（依据 / 代价 / 命令级动作）
**A) 全面切底座** — ❌ 不建议
- 依据反对：① 两套 DSL 不同源，九件**半重写**且 950 SIMT 原语在 910B 无对应（§4.1/4.2）；② **950 目标当前编不出**（#1830 + dav-3510 被拒，§2.2）；③ PTO 路本容器编不过（pto-isa 失配，§2.3）；④ 丢了我们 `port910b_compat.h` 的 910B 标量路与**已真机全绿的判决链（标量路）**，重打一遍代价极高。
- 下一步（若仍想验证可行性，任何一条都只是"探路"）：在**装了匹配 torch_npu** 的容器内 `bash install_ascend.sh --enable-incremental`，再 `ASCEND_NPU_ARCH=dav-3510 TL_RUN_MODE=sim` 之类的 sim 路复跑 §2 —— 但 §2.2 已证 `-xasc` 拒 dav-3510。

**B) 仅采 GDN chunk 族实现、底座不动** — ✅ **建议主选**
- 依据：① asc 仓六件**编译全过**（§2.2 6/6），且用**正统 cube 路 `T.gemm_v0`**（`ascend.py:418`）——正对我们 **§12 cube 路 507015** 的痛点；② fwd-only 正好覆盖我们"GDN 反向三缺陷"**之外**的前向族；③ 不碰我们已绿的标量底座与判决链。
- 代价：**逐件把 asc 的 cube 正文翻译到我方 `tilelang.ascend.language` 方言**（`T.gemm_v0→`我方 cube 原语；`alloc_L1/L0C→?`；`is_npu` 帧），工作量 = 6 件内核 + 1 装配；需复跑 dtype 矩阵（fp16 起步）。
- 命令级动作：
  1. `cp` 六件到 `refs/tilelang-ascend/…/gdn/` 作对照（已在 refs，gitignored）；
  2. 以 `gdn_wy_fast.py`（最小 cube 件，`:83/:88` 两处 `gemm_v0`）为**首移**，逐行对照翻译；
  3. 每次翻译后用 **§R 的 compile-only 探针**在我方底座编一遍（`TILELANG_CACHE_DIR` fresh），再上真机跑 `gradcheck_card`。

**C) 维持现状 + 佐证回馈** — ✅ **建议并行（低成本）**
- 依据：上游 #1762（= 我们 C2 发现）、#1802（chunk_o_bwd）、#1830 都还 open；把我们的**真机 defect 复现**当证据回馈上游，成本极低、收益高（可能换到上游修或绕过）。
- 代价：几乎为零（不改代码，只提 issue/评论 + 附最小复现）。
- 命令级动作：以我们 §12 507015 与 GDN 反向三缺陷的最小复现，向 `tile-ai/tilelang-ascend` 提 issue / 到 #1830、#1802 留复现（**只在 GitHub 侧，不写我们仓**）。

### 5.3 "若 go（= 选 B），窗五须跑的三条腿"
1. **编译腿**：在我方底座（或目标底座）用 fixed `TILELANG_CACHE_DIR` 把**六件全部 compile-only 编过**（对照 §2.2 我方探针），贴 bisheng `rc=0`。
2. **数值腿**：真机跑 `gdn_full` 全链，与**逐块 torch 参考实现**对拍（`gdn_full.py` 自带 `ref_seq_gdn/ref_chunk_gdn`），fp16 容差内 `assert_close` 过。
3. **判决腿**：把窗口脚本 + KERNEL-USED 三证套上——确认**真走了内核**（`compiled_keys` 非空、`compile_count` 基线不变），并把结果写回 `reconcile/`。

三条腿全绿才判 go；任一红即回落到"仅存对照、不切"。

---

## §R 复跑命令（全部可复现）

**R0 被评对象锚点**
```bash
gh api repos/tile-ai/tilelang-ascend --jq '{full_name,default_branch,open_issues_count,pushed_at}'
cd refs/tilelang-ascend && git rev-parse HEAD && git -C 3rdparty/tvm rev-parse HEAD
```
**R1 构建（容器 cann910b-l，宿主挂 refs/tilelang-ascend → /tlasc）**
```bash
# 依赖：apt 需能装；无 lld 时向 build/config.cmake 追加 set(USE_ALTERNATIVE_LINKER OFF)
docker exec cann910b-l bash -lc 'cd /tlasc && rm -rf build && mkdir -p build && \
  cp 3rdparty/tvm/cmake/config.cmake build/ && cd build && \
  printf "set(USE_ASCEND ON)\nset(USE_GTEST OFF)\nset(USE_ALTERNATIVE_LINKER OFF)\n" >> config.cmake && \
  cmake .. && make -j2; echo MAKE_EXIT=$?'
```
**R2 编译探针（无设备 compile-only；含 torch-npu shim）**
```bash
docker exec cann910b-l bash -lc '
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
 export PYTHONPATH=/tlasc:/tlasc/3rdparty/tvm/python
 export ASCEND_NPU_ARCH=dav-2201 TL_PLATFORM=A2
 rm -rf /root/h2c && export TILELANG_CACHE_DIR=/root/h2c
 cd /tlasc && python3 /root/h2_probe_gemm_compile.py        # 单件 gemm
 cd /tlasc && python3 /root/h2_probe_gdn_all_compile.py     # GDN 六件 6/6
 # A5 复现 #1830 编译侧：H2_PLATFORM=A5 TL_PLATFORM=A5 …；dav-3510 覆盖：H2_EXTRA_FLAGS=--npu-arch=dav-3510'
```
**R3 issue 取证**
```bash
R=tile-ai/tilelang-ascend
gh api "search/issues?q=repo:$R+is:issue+is:open" --jq .total_count   # 302
gh api "search/issues?q=repo:$R+is:pr+is:open"    --jq .total_count   # 230
for n in 1830 1762 1758 1191 1799 1827 1866 1802; do gh api repos/$R/issues/$n --jq '{number,state,title}'; done
```
**探针脚本**（`refs/`，gitignored）：`h2_probe_gemm_compile.py`、`h2_probe_gdn_compile.py`、`h2_probe_gdn_all_compile.py`。

---

## §X 边界与未竟
- 未在**真 950/A5 硅片**上复现 #1830 的 device fault 507015（本容器无设备；compile-only 只能到"产物 arch 错"层）。
- 未验证 asc 仓 GDN 六件的**数值正确性**（编译过 ≠ 算得对；需真机 + 参考对拍）。
- `TILELANG_CACHE_DIR`、`import tilelang` 的**可迁移性**为同符号实测；`ascend_env._compile_probe` 平移后的**具体报错**未逐一取（§4.3 标 ⚠️）。
- 未测 910B 上以 asc 底座编**我们的** cube 件（那属"选 A/B 后"的动作，本轮未做）。
