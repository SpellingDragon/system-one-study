# S_RESULT — P1-1f cube 双件 correctness-first 改道（D-cube1 甲路）

- **变更**：`teacher-p2-production-full` / `p2-13-ascend-runtime`（嵌套子变更，openspec CLI 不索引，直读 design.md/tasks.md）
- **孙任务**：**P1-1f**（tasks.md:15）——`gemm_asc.py` / `gemm_bwd_dw_asc.py` 的 ascend 实现改**可移植标量体**；K 波 Cube 形态存档为 performance-track 参照
- **代理**：S（本波最重件，解锁 P2 训练步数值里程碑）
- **日期**：2026-10-10
- **一句话结论**：两件 ascend 正文已改为 cpu/ascend **同一份标量正文生成器**；**本地 cpu 轨数值全绿（13 形最大 rel=5.218e-07 < 2e-5，cube 数值首次本地闭环）**、**容器 ascend 轨 13 形 + 接口真链全 COMPILE-PASS（生成码零 cube/零 SIMT/零 §12 依赖）**、host 无回归（gradcheck 20 + train_step 8 + 全量 332）。真机 rel 随下窗；性能属 performance-track。

---

## ① 完成情况（逐条附可复跑凭据）

| P1-1f 子项 | 状态 | 凭据（可复跑） |
|---|---|---|
| K 波 Cube 形态**改前**存档 | ✅ | `reconcile/S_perf_track_cube.py`（185 行，`gemm_cube_impl` Q1=UB-direct Cube / `dw_l0tr_cube_impl` l0tr 主案 逐字存档 + V1–V6 待证清单 + 被否方案；不被产品代码 import）。自检：容器跑 → `S-PERF-TRACK ARCHIVE-COMPILE-PASS[gemm-cube]` + `[dw-l0tr]` rc=0（**存档形态至今仍可编**，坏的只是数值）；host 跑 → `ARCHIVE-SKIP(环境)`×2 rc=0（host 无 ascend tileop，与『编不出』分开报，不冒充 PASS） |
| 单一正文生成器 + `dialect(target)` 注入 | ✅ | `gemm_asc.gemm_scalar_body(T,…)` / `gemm_bwd_dw_asc.dw_scalar_body(T,…)`；`gemm_asc_impl`/`dw_asc_impl` 签名不变、改委托 `ascend_env.dialect(TARGET_ASCEND)` |
| cpu 轨数值 oracle rel<2e-5 | ✅ | `cd release && .venv/bin/python ascend/kernels/reconcile/S_cpu_oracle.py --cpu` → `S-GEMM-CPU-ORACLE-PASS` + `S-DW-CPU-ORACLE-PASS`，13 形全 PASS，最大 rel **5.218e-07** |
| 形状覆盖（64³ / 非方 / relu-none / 尾块 / 方向判别） | ✅ | 同上：gemm 7 形（64³、32×128×64、m=8/33/1、n96k48）；dw 6 形（64³、n64k128、m=8/33/129、tiles=9>NUM_BLOCKS=8 摊开）；方向判别 `wrong(A@W 不翻)=1.351~1.397`、`wrong(Aᵀ@dY)=1.074~1.331`（须 >1e-2 ⇒ 通过不是碰巧对称） |
| 与 day-1 尺子件交叉对拍 | ✅ | `[gemm 交叉 day-1 gemm_cpu_impl m33 n64 k32] rel=0.000e+00 PASS`；`[dw 交叉 day-1 dw_cpu_impl m33 n128 k64] rel=1.478e-07 PASS`（`*_cpu_impl` 一行未动 ⇒ 新标量体与 20 绿资产同解） |
| 同产物多批复用（m 动态维不重编） | ✅ | `compiled_products=5`（7 形只编 5 次）；`[dw 同产物多批 m=1/8/16/17/33/64] rels=['0.00e+00','0.00e+00','0.00e+00','5.18e-08','1.75e-07','2.75e-07'] PASS` |
| ascend 轨编译判决（容器） | ✅ | `docker exec cann910b-h … python3 ascend/kernels/reconcile/S_cpu_oracle.py --ascend` → 13 行 `COMPILE-PASS`、`banned_in_src=[]`、`S-GEMM-ASC-COMPILE-PASS（7 形，.src~3117B）`、`S-DW-ASC-COMPILE-PASS（6 形，.src~2803B，§12 依赖面=0）` |
| 接口真链（plan→run→impl）ascend 判决 | ✅ | 同容器加 `--iface` → `keys=['ascend\|gemm[ascend]\|n64k64b16x64x64anone','ascend\|gemm[ascend]\|n64k64b16x64x64arelu','ascend\|gemm_dw[ascend]\|n64k64tc16b64x64']`、`blockers={} compile_count=3` → `S-GEMM-DW-IFACE-ASC-COMPILE-PASS` |
| 零 cube/零 SIMT 静态自证 | ✅ | 生成码函数集（容器实测）：gemm 3001B = `__global__/__vector__/asc_init/asc_copy_gm2ub_align/asc_sync_notify/asc_sync_wait/write_gm_bypass_dcache`，`DMA-count=1`；dw 2250B = `asc_init + write_gm_bypass_dcache`，**`DMA-count=0`**；`Cube/mad/L12L0/L0C/asc_fill_l1/SimtVF` 命中数 = 0 |
| 无回归 gradcheck（20 基线） | ✅ | `.venv/bin/python -m pytest tests/test_ascend_gradcheck.py tests/test_ascend_train_step.py -q` → **28 passed, 1 skipped**（20+8 合并跑；基线同值） |
| 无回归 train_step（8，经 autograd 走新实现） | ✅ | 同上（8 passed 计在 28 内）；单独跑 `pytest tests/test_ascend_train_step.py -q` → 8 passed |
| 无回归全量 | ✅ | `pytest tests/ -q` → **332 passed, 6 skipped** |
| 探测器无回归（K 波资产） | ✅ | 容器 `verify_gemm.py` → `K-GEMM ASC-COMPILE-PASS`（.o~3004B，`simt_in_src=False cube_in_src=False transposeB_in_src=False`）；`verify_gemm.py --cpu` → `K-GEMM CPU-SEMANTIC-PASS`（3.815e-06/5.722e-06）；`verify_lora.py` → **`K-LORA ASC-COMPILE-PASS`**（由 K 波 `PARTIAL` 升为全绿：gemm 2 keys + gemm_dw 2 keys，`blockers={}`） |
| 白名单自证（接口/尺子件不动） | ✅ | `python3 /tmp/s_whitelist_check.py`（ast 逐字比 HEAD）：`gemm_asc.py` 的 `gemm_cpu_impl(1526B)/plan(1407B)/run(750B)`、`gemm_bwd_dw_asc.py` 的 `dw_cpu_impl(1160B)/plan(1248B)/run(525B)` **与 HEAD 逐字一致=True**；新增仅 `_alloc_tile` + `gemm_scalar_body`/`dw_scalar_body`，改动仅 `*_asc_impl`（改委托） |

**P1-1f 判据四件套逐字回执**（真实输出，非转述）：
```
S-GEMM-CPU-ORACLE-PASS / S-DW-CPU-ORACLE-PASS                       # host，rel 最大 5.218e-07
S-GEMM-ASC-COMPILE-PASS / S-DW-ASC-COMPILE-PASS                     # 容器，banned=[]
S-GEMM-DW-IFACE-ASC-COMPILE-PASS                                    # 容器，plan/run 真链零 blocker
K-LORA ASC-COMPILE-PASS（forward-leg compiled=True dW-leg compiled=True dW-leg-blocked=False）
```

---

## ② 错误与阻塞（含已解决的）

1. **`verify_dw.py` 探测器崩（非我引入，已绕行）**
   `RuntimeError: kernel dw_impl input dY device_type mismatch, expected ext_dev; expected: 1, got: 12` @ `gemm_bwd_dw_asc.py:235`（launch 处）。
   根因：该桩是"真编后**原样返回** kernel"，接口路径一编成功就真去 launch cpu 张量；tasks.md **P1-4b 小遗**已登记"verify_dw 需补 no-op 发射桩，下任代理修"。
   → 处置：**绕行**——该文件不属我白名单，改在 `S_cpu_oracle.py` 新增 `--iface` 模式（真编后换 no-op 发射替身）补齐 dW **接口路径**判决，判据与 `verify_gemm/verify_lora` 同配方。**代价**：多一个探测模式，`verify_dw.py` 本体仍待下任/编排者修桩（修好后其判决行应与我的 `--iface` 一致）。
2. **`compile_all_asc.py` 的 gemm 腿 PRECOMPILE-ERR（既有工具缺陷，待裁决）**
   `ASC-gemm PRECOMPILE-ERR TypeError: plan() got an unexpected keyword argument 'n'` @ `compile_all_asc.py:74` 以 impl kwargs 调 `plan()`，而 HEAD 的 `plan()` 签名从来无 `n=`（`git show HEAD` 可验）⇒ **非我引入**（我未动 `plan`，ast 比对逐字一致），且 `compile_all_asc.py` 在**禁改**名单。
   → 处置：**待裁决**。旁证已补：同一接口链路的 `verify_gemm.py`（`K-GEMM ASC-COMPILE-PASS`）与我的 `--iface` 都编出 `ascend|gemm[...]` 产物；该 harness 的 dw 腿本身 PASS（`ASC-gemm_bwd_dw PASS .o~2250B`）。
3. **host `--trace` 报 `Check failed: (reg != nullptr): Operator tl.tileop.ascend_copy is not registered`**
   → 处置：**分类为环境缺件**（host venv 未注册 ascend tileop），在 `S_cpu_oracle.py` 里判为 `TRACE-SKIP(环境)`，不算正文缺陷、不冒充 PASS；ascend 编译判决一律以容器 `--ascend` 为准。（dw 正文零 copy ⇒ `[trace dw] TRACE-OK`。）
4. **容器 compat 与 trunk 不一致**：`port910b_compat.h` md5 `1fa50f22…` ≠ trunk `20939eae…`
   → 处置：按任务书 `docker cp /tilelang/src/tl_templates/ascend/port910b_compat.h cann910b-h:/usr/local/lib/python3.10/dist-packages/tilelang/src/tl_templates/ascend/`；现 `md5sum` 复核两边一致（1062 行 / `20939eae21990ca264fd1f10854e056f`）。
5. **`verify_lora.py` 的 `merge` 腿异常（已知噪声，不影响判决）**：`exc:RuntimeError:expected m1 and m2 to have the same dtype, but got: float !=`
   → K 波已登记：no-op 发射替身下 merge 腿在 torch 侧回退路径上的 dtype 噪声；判决看 `keys/blockers`，本趟 4 keys 齐、`blockers={}` ⇒ `K-LORA ASC-COMPILE-PASS`。
6. **tilelang eager 取源码坑（工程性，已修）**：`@T.prim_func` 写在 `python -` stdin / heredoc 里 → `OSError: could not get source code`。→ 一律落 `.py` 文件；件文件不写 `from __future__ import annotations`（PEP 563 会剥掉只在注解里出现的 n/k 闭包）。
7. **本地计时噪声（影响性能段可信度）**：同一份标量体 256³ 三次跑 = **18.82 / 19.19 / 45.51 ms**（≈2.4x 摆幅），day-1 体 22.0~24.8 ms 相对稳定。→ 性能段只用口径/理论估计表述，不给倍率结论（详见下"性能代价"）。

---

## ③ 疑惑点与自行决策（均未询问，逐条申报）

1. **存储层落点是正文里唯一的方言分叉**（`_alloc_tile`）：cpu→`T.alloc_local`（day-1 口径）、ascend→`T.alloc_shared`（UB，H 波 add_ln 口径）。
   依据：任务书说"只用两方言公共子集"，但两方言各自的公共面**不保证对方那层能落地**（c 后端无 UB 语义；910B 的 local 面未证）。我**分的是 scope 归属、不是算式**——三重乘加/夹址/守卫/出口逐字单份。若编排者要求"零分叉"，可改成 cpu 轨也用 `alloc_shared`，但需重跑 cpu oracle 且 c 后端能否落地未证（代价：可能新开一个后端坑）。
2. **尾块越界防线自己做（夹址 + 行守卫），不赌 `T.copy` 的自动边界谓词**：`rc = T.min(row, m-1)` + `T.if_then_else(row < m, A[rc,…], 0.0)`（`if_then_else` 两侧都求值 ⇒ 必须先夹地址），出口再 `if row < m` 行守卫。
   依据：卡上越界读的表现正是 `aicore exception 507015`（P1-1d 九参位序修复"必要不充分"）；而 `T.copy` 的边界谓词在 910B 面上从未证真。
3. **W 块走 `T.copy`、A/dY 块走标量填装**（分层而非一刀切）：W 的 N/K 两轴都是编译期常量且 plan 已判整除 ⇒ 永不越界 ⇒ 保住"搬运走 DMA、计算走标量"；活轴（m）才用标量填装。
   附带理由：可规避"runtime if 里包 `T.copy`"这种**卡上无先例**的写法。若编排者偏好"全标量零 DMA"，dw 已经是那样（DMA-count=0），gemm 只余 1 条 copy。
4. **`T.clear` 全撤**，累加器清零改标量逐格置 `T.float32(0.0)`。依据：K 波取证 fp32 `T.clear` 在 UB 上生成坏 `float2` store（`no viable overloaded '='`）。
5. **tasks.md P1-1f 的 `[ ]` 未勾**（本文件是唯一申报处）：勾选属"只改两文件"白名单之外（编排者的 完成凭据 行历来由编排者统一复验后落笔，见 P1-1g/P1-4 体例）。**给可直接粘贴的行**（在 15 行行首改 `[x]`，其下插入子项）：
   > `- [x] **P1-1f …** —— 完成凭据（1010 代理 S）：单一标量正文生成器 gemm_scalar_body/dw_scalar_body 由 ascend_env.dialect(target) 注入，cpu 轨 13 形 rel 最大 5.218e-07（S-GEMM-CPU-ORACLE-PASS/S-DW-CPU-ORACLE-PASS，含方向判别 1.07~1.40 + day-1 交叉 + 同产物多批）、ascend 轨 13 形 + plan/run 真链 COMPILE-PASS 零 blocker（S-GEMM-ASC/S-DW-ASC/S-GEMM-DW-IFACE-…-PASS，生成码零 Cube/零 mad/零 SimtVF/零 §12，gemm 仅 1 条 asc_copy_gm2ub_align、dW 零 DMA）；K 形态存档 reconcile/S_perf_track_cube.py；无回归 28 passed+1 skipped / 332 passed；判决书 reconcile/S_RESULT.md；真机 rel 随下窗（消费链=train_step --target ascend）`
6. **并发事实**：`ascend_env.py` 已含 T 代理 P1-1g 的 G-gate 修复（探针去 SimtVF），我的 host 全量与容器判决都是**带该改动**跑的（对 S 有利：门不开则接口判决无从取证）。`letter_readout_asc.py`/`U_oracle.py` 是 U 代理在改，我未触碰。
7. **oracle 的块宽由 `_largest_block` 自行复现**（不走 `plan`）：为了让 `--cpu/--ascend` 直喂生成器；`--iface` 才是真走 plan/run，两者互为校验（`compiled_products=5` vs `compile_count=3` 差异=oracle 多做 relu×act 组合与 m 复用）。

---

## ④ 偏离记录（与计划/spec 不一致之处）

| 偏离点 | 计划原文 | 实际做法 | 理由/代价 |
|---|---|---|---|
| 正文语言构造子集 | "T.serial + T.copy（GM↔UB）+ 标量 load/store" | gemm 完全照做；**dW 只用了 T.serial + 标量 GM 读写，`T.copy` 数=0** | 出口 (N,K) 分块本就写 GM，中间量放 UB 反而多一趟；副作用=依赖面更小（零 DMA、零 §12），判决更稳 |
| 网格形态 | 沿用 plan 的 BLOCK_LADDER 常量网格 | gemm `T.Kernel(nb)`；dW `T.Kernel(blocks)`，`blocks=min(tiles,NUM_BLOCKS)` 后用 `iters` 摊开（tiles>8 时 `tid=bid+it*blocks`） | 910B `T.Kernel` 是 1-D 核块网格且上界必须常量（动态 m 写进网格 ⇒ 一次都不发射、静默全零）。代价：`NUM_BLOCKS=8` 是并发度上限，形状大时串行摊，属性能面 |
| `verify_dw.py` 判决行 | 任务书"照 `verify_gemm.py` 既有跑法取 `S-DW-ASC-COMPILE-PASS`" | 该桩真 launch 会崩（见②-1），改由 `S_cpu_oracle.py --ascend/--iface` 出 `S-DW-ASC-COMPILE-PASS` + 接口链 PASS | 白名单外不可修桩；判决力等价（同一 `ascend_env.compiled_keys()/blockers()` 判据） |
| `compile_all_asc.py` 逐件 per-op 行 | 任务书隐含（九件 harness） | 未取到 gemm 腿（②-2，文件禁改），以 `verify_gemm.py`+`--iface` 双旁证替代 | 待裁决；不掩盖：报错原文已录 |
| 性能数字 | "性能代价（标量 vs cube 的理论差距）如实估计" | 给**理论口径 + 噪声实测**，不给倍率结论 | 本地 host c 后端把 `T.gemm` 也退化成标量循环，且噪声 2.4x ⇒ 任何本地倍率都推不出卡上差距（见下段） |
| `tasks.md` 勾选 | apply 流程应勾 `[x]` | 未勾（白名单），在③-5 给可粘贴行 | 编排者定口径 |

**性能代价（如实估计，属 performance-track，不属本波）**
- **卡上口径**：标量体每输出元素做 `bk` 次标量 `mul+add` ⇒ 全件 `~2·m·n·k` 条标量 FMA，**无 VF（无向量化）、无 Cube fractal**。910B 的 Cube 单元一次 fractal 指令吃 16×16×16（=4096 MAC），Vector 面标量循环与之相比理论差 **2~4 个数量级**（再叠加无 VF 的 8~16x）。故 correctness 轨的定位是**数值里程碑 + 冒烟**，生产形态仍待 performance-track 的 intrinsic 考古（存档件已备）。
- **本地实测（仅作"无数量级劣势/无静默死循环"的旁证，不可外推）**：256³ host，标量体 gemm 18.8/19.2/45.5 ms（0.74~1.78 GF/s）、dw 18.4/18.9/36.8 ms；day-1 `T.gemm` 体 22.0~24.8 ms（1.35~1.52 GF/s）⇒ 比值在 **0.83x~1.84x** 之间乱摆，同码三次摆幅 2.4x。**结论：本地计时不可用作性能证据**；且 c 后端把 `T.gemm` 同样降级成标量循环，两口径在 host 上本就同质，倍率对卡上毫无解释力。
- **UB 占用（编译期常量，可直接算）**：gemm 每核块 `a_ub(bm×bk)+w_ub(bn×bk)+c_ub(bm×bn)` fp32，最坏 16×64+64×64+16×64 = 6144 float ≈ **24 KB**；dw 每核块 `dy_ub+a_ub+acc_ub`（tc=16,bn=64,bk=64）≈ 24 KB ⇒ 8 核块并发约 192 KB，UB 预算内（910B UB 每核 ~192 KB? 该数值属 port 文档域，下窗真机以 profiler 复核）。
- **复跑命令**：见文末【附录 B】全码（一次性取证件，入文以免 /tmp 蒸发后不可复跑）。

---

## ⑤ 产物清单

**新建（白名单三件）**
- `release/ascend/kernels/reconcile/S_perf_track_cube.py` — K 波 Cube/l0tr 双正文逐字存档 + V1–V6 卡待清单 + 被否方案（不被产品代码 import）
- `release/ascend/kernels/reconcile/S_cpu_oracle.py` — 双方言 oracle（`--cpu` 数值 / `--ascend` 编译+禁构造静态扫 / `--iface` plan·run 真链 / `--trace` 本地追溯自证）
- `release/ascend/kernels/reconcile/S_RESULT.md` — 本判决书

**修改（白名单两件）**
- `release/ascend/kernels/gemm_asc.py`（210→267 行）— 新增 `_alloc_tile` + `gemm_scalar_body`；`gemm_asc_impl` 改委托；`gemm_cpu_impl/plan/run/_largest_block` 未动
- `release/ascend/kernels/gemm_bwd_dw_asc.py`（189→236 行）— 新增 `_alloc_tile` + `dw_scalar_body`；`dw_asc_impl` 改委托；`dw_cpu_impl/plan/run/_largest_block` 未动；模块 docstring 改述 D-cube1 改道理由

**未动（禁改名单自证）**：`ascend_env.py`(T)、`letter_readout_asc.py`(U)、trunk `port910b_compat.h`（只做 `docker cp` 同步，无编辑）、`compile_all_asc.py`、`tests/` 全部、`verify_dw.py`/`verify_gemm.py`/`verify_lora.py`；无 git 操作。

---

## 交接：下窗真机收数路径（关键，别踩）

1. **消费链（本波解锁点）**：`python ascend/train_step.py --target ascend` → `autograd_asc._LinearFn` → `gemm_kernel/gemm_bwd_dw_kernel` → `*_asc_impl` → **本次新标量正文**。`tests/test_ascend_train_step.py` 已在 host 侧经此路走绿（8 passed）。
2. **陷阱**：`run_numerics.py` 的 `gemm_l1` / `dw` target 指向 **attempts 内联 DSL 件（`d_dw_910b` 等 cube 形态）**，**不消费接口件** ⇒ 若用 `run_numerics` 收 gemm/dW 数值，测到的仍是 cube 轨，不是本波改道结果。下窗收数须走 `train_step --target ascend`，或由编排者授权把这两个 target 改指接口件。
3. **判据已备**：`S_cpu_oracle.py --cpu` 的 13 形 rel 即真机同式尺子（正文逐字同一份，差异只在 `_alloc_tile` 的 scope），预期真机 rel 与 add_ln 同量级（≤1e-5）；若真机崩 `507015`，优先查：① `asc_copy_gm2ub_align` 的 W 块地址对齐（gemm 唯一 DMA）② 出口 `write_gm_bypass_dcache` 的行守卫是否被优化掉 ③ dw 零 DMA 则与本病灶无关（可反向定位为 vector store 面）。
4. **未收数项**：真机 rel、UB/L1 实际占用、`compile_all_asc.py` gemm 腿 harness 缺陷、`verify_dw.py` no-op 桩。

---

## 附：一键复跑（证据链）

```bash
# 0) 基线（host）
cd $REPO_ROOT/release
.venv/bin/python ascend/kernels/reconcile/S_cpu_oracle.py --cpu --trace   # S-GEMM/S-DW-CPU-ORACLE-PASS
.venv/bin/python -m pytest tests/test_ascend_gradcheck.py tests/test_ascend_train_step.py -q   # 28 passed 1 skipped
.venv/bin/python -m pytest tests/ -q                                       # 332 passed 6 skipped

# 1) 容器 ascend 编译判决
docker cp /tilelang/src/tl_templates/ascend/port910b_compat.h \
  cann910b-h:/usr/local/lib/python3.10/dist-packages/tilelang/src/tl_templates/ascend/
docker exec cann910b-h bash -lc 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \
  export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201; \
  cd /work && TILELANG_CACHE_DIR=$(mktemp -d) timeout 900 python3 \
    ascend/kernels/reconcile/S_cpu_oracle.py --ascend --iface'
docker exec cann910b-h bash -lc '… cd /work && python3 ascend/kernels/reconcile/verify_gemm.py; \
  python3 ascend/kernels/reconcile/verify_lora.py'

# 2) 白名单自证（接口/尺子件逐字未动）：全码见【附录 A】，落盘后 python3 /tmp/s_whitelist_check.py

# 3) 性能（噪声，仅旁证）：全码见【附录 B】，落盘后 .venv/bin/python /tmp/s_bench.py
```

---

## 附录 A：白名单自证脚本（/tmp/s_whitelist_check.py 全文）

```python
import ast, subprocess

def head_src(path):
    return subprocess.run(["git", "show", f"HEAD:release/ascend/kernels/{path}"],
                          capture_output=True, text=True, check=True).stdout

def funcs(src):
    tree = ast.parse(src)
    return {n.name: ast.unparse(n) for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

for fname in ("gemm_asc.py", "gemm_bwd_dw_asc.py"):
    old = funcs(head_src(fname))
    new = funcs(open(f"release/ascend/kernels/{fname}").read())
    print(f"== {fname}")
    print(f"   HEAD funcs: {sorted(old)}")
    print(f"   NOW funcs : {sorted(new)}")
    for name in sorted(set(old) & set(new)):
        if name.endswith("_cpu_impl") or name in ("plan", "run"):
            print(f"   {name:14s} 与 HEAD 逐字一致={old[name] == new[name]} ({len(new[name])}B)")
```

只读 `git show`，无任何 git 写操作。实测输出：`gemm_cpu_impl(1526B)/plan(1407B)/run(750B)` 与
`dw_cpu_impl(1160B)/plan(1248B)/run(525B)` 全部 `逐字一致=True`；NOW 仅多出 `_alloc_tile` +
`gemm_scalar_body`/`dw_scalar_body`，`*_asc_impl` 是有意改委托。

## 附录 B：性能取证脚本（/tmp/s_bench.py 全文）

```python
import time, torch, tilelang, sys
sys.path.insert(0, "$REPO_ROOT/release")
from ascend.kernels import ascend_env, gemm_asc, gemm_bwd_dw_asc as dwd
T = ascend_env.dialect("cpu")
m = n = k = 256
A = torch.randn(m, k); W = torch.randn(n, k); b = torch.randn(n); C = torch.zeros(m, n)
kw = {"n": n, "k": k, "bm": 16, "bn": 64, "bk": 64, "act_mode": 0}
ck = ascend_env.compile_kwargs("cpu")

def bench(build, args):
    kern = build()
    kern(*args)
    t = time.time()
    for _ in range(3):
        kern(*args)
    return (time.time() - t) / 3

t_new = bench(lambda: tilelang.compile(gemm_asc.gemm_scalar_body(T, A, W, b, C, **kw),
                                       out_idx=[], **ck), (A, W, b, C))
t_old = bench(lambda: tilelang.compile(gemm_asc.gemm_cpu_impl(A, W, b, C, **kw),
                                       out_idx=[], **ck), (A, W, b, C))
dY = torch.randn(m, n); A2 = torch.randn(m, k); GW = torch.zeros(n, k)
kwd = {"n": n, "k": k, "tc": 16, "bn": 64, "bk": 64}
t_dn = bench(lambda: tilelang.compile(dwd.dw_scalar_body(T, dY, A2, GW, **kwd),
                                      out_idx=[], **ck), (dY, A2, GW))
t_do = bench(lambda: tilelang.compile(dwd.dw_cpu_impl(dY, A2, GW, **kwd),
                                      out_idx=[], **ck), (dY, A2, GW))
fl = 2 * m * n * k / 1e9
print(f"gemm 256^3: scalar={t_new*1000:.2f}ms ({fl/t_new:.2f} GF/s) | day1={t_old*1000:.2f}ms | ratio={t_new/t_old:.2f}x")
print(f"dw   256^3: scalar={t_dn*1000:.2f}ms ({fl/t_dn:.2f} GF/s) | day1={t_do*1000:.2f}ms | ratio={t_dn/t_do:.2f}x")
```

**读数纪律**：三次连跑 gemm ratio ∈ {1.53, 0.83, 1.84}（同码 18.8/19.2/45.5 ms，摆幅 2.4x）、
dw ratio ∈ {0.85, 1.61, 0.84}；且 c 后端把 day-1 的 `T.gemm` 同样降级为标量循环 ⇒ 两个口径在 host 上
**同质**，任何本地倍率对卡上『标量 vs Cube』都无解释力。此段只作『无数量级劣势、无静默死循环』的旁证。

## 附录 C：生成码函数集（容器实测，非推断）

```
== GEMM 3001 B
  funcs: [__global__, __vector__, asc_copy_gm2ub_align, asc_init,
          asc_sync_notify, asc_sync_wait, write_gm_bypass_dcache]
  DMA-count: 1   memset/clear: 0
== DW 2250 B
  funcs: [__global__, __vector__, asc_init, write_gm_bypass_dcache]
  DMA-count: 0   memset/clear: 0
```

⇒ op 家族是**真机绿件 add_ln（wave2 rel_y=1.69e-05）所用构造的子集**；dW 更窄（连 DMA 都没有）。
