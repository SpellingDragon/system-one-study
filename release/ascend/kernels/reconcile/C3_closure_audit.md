# C3_closure_audit.md — p2-13 收口条件复核（证据映射，只读）

> 执行代理 **C3**（收口条件复核）。**只做证据映射，不写产品码；不改 tasks.md**（归编排者统笔，历史被并发覆写）。
> 任务：把 `openspec/.../p2-13-ascend-runtime/tasks.md` 中仍未勾选的 5 条，逐条把「验收原文」与**已有真机/本地证据**对照，产出
> `可勾 / 不可勾 / 缺什么` 三态清单。三态标注：✅已满足（挂原文位置）/ ⏳缺证据 / ❌被证伪。
> 证据源全部已入库：`release/runs/1011-p2-13-oncard-accept-2002/`、`release/runs/1011-p2-13-oncard-window2-throughput/`、
> `release/ascend/kernels/reconcile/*.md`、`release/ascend/port910b/attempts/*`、`release/runs/1010-p2-13-oncard-wave2-cube-and-gate-e079/notes.md`。

---

## 0. 三态速览（5 条）

| # | tasks.md 行 | 任务 | 三态结论 | 一句话 |
|---|---|---|---|---|
| 1 | `tasks.md:10` | P1-1 readout/gemm/dW/add_ln 四件改道移植 +对拍 | **不可勾（缺 2 项）** | 验证判据「gradcheck NPU 档」**从未落地**（全库无卡上 gradcheck）；add_ln 只有 `addln_intrinsic`、**无产品件真机数值** |
| 2 | `tasks.md:31` | P1-3 GDN 件移植（本项目旗舰） | **可勾（字面判据已满足）**（附覆盖边界） | 验证判据「`NUM-delta_fwd` 真机 rel」双窗 ✅；但**反向真机 rel 缺**（仅 CPU golden），且真机件走 `attempts/F` 执行器 |
| 3 | `tasks.md:13` | P1-1d cube-V12 位段修复 | **不可勾（缺 1 项，且被证伪）** | 验证判据「gemm_l1 + dW 数值 rel」**从未取得**——两窗真机仍 `507015`；位序修复自评「必要不充分」 |
| 4 | `tasks.md:37` | P2 三栈联调（GDN 件替换 Conv2D 路） | **不可勾（缺 2 项）+ 建议改写验收原文** | 跑的是**最小混合栈一步**（d=32），**非 0.8B**；训练步**不含 Conv2D 位点**，"替换 Conv2D"未实证 |
| 5 | `tasks.md:39` | R6/R7 成本记账与 gate 报价 | **不可勾（缺 1 项）** | 有 8 条标量 `BENCH-QUOTE`，但**熔断判定所需的对照/消融侧无任何可引用数据**（`linear_intrinsic` 死代码、`dW_intrinsic` 坏件） |

**用户已知两坑的落点**：坑①（P1-1p/q 勾=避让+反事实，非根治）不改变本核结论——P1-3 的 delta 判决**已单独核过**（见 §2）；坑②（三栈消融无可引用数据）坐实——凡验收含「吞吐/报价/消融」者（本条即 R6/R7，及未列入本核的 P1-1t）= 不可勾。

---

## 1. `tasks.md:10` P1-1 readout/gemm/dW/add_ln 四件改道移植 + 对拍

**验收原文（逐字）**：
> `- [ ] P1-1 readout/gemm/dW/add_ln 四件**改道移植**（向量面最薄可标量化；P0-1 修正：无纯 AIC 件）+对拍 —— 验证：gradcheck NPU 档`

**拆项与判据**：

| # | 判据 | 证据 | 状态 |
|---|---|---|---|
| 1a | **readout 改道移植**（真机数值） | `ASC-PROD-readout_prod PASS rel=0.00e+00 fwd=0.00e+00 bwd=0.00e+00 bitexact=True`（`accept-2002/artifacts/card_run.log:99`；`window2/artifacts/w2_all.log:10,24`） | ✅ |
| 1b | **gemm 改道移植**（真机数值） | `ASC-PROD-linear_prod PASS rel=1.59e-07 dir_disc=1.43e+00 … shapes=5`（`window2/artifacts/w2_all.log:8,22`）。⚠ 首窗同腿曾 FAIL `rel=1.807e+01`（`accept-2002/artifacts/card_run.log:97`），第二窗才绿 | ✅（最新窗） |
| 1c | **dW 改道移植**（真机数值） | `ASC-PROD-dw_prod PASS rel=1.36e-07 dir_disc=1.39e+00 reuse=4/4 shapes=4`（`accept-2002/artifacts/card_run.log:98`；`window2/artifacts/w2_all.log:9,23`） | ✅ |
| 1d | **add_ln 改道移植**（真机数值·**产品件**） | **缺**：`run_numerics.py:277-278` 自陈「这条 case 与产品件 `add_ln_asc.add_ln_asc_impl` **不是同一份正文** … 本波未接产品件」。只有 `NUM-addln_intrinsic-PASS rel_y=1.66e-05 rel_h=0.00e+00 unb_disc=7.48e-04`（`accept-2002/artifacts/card_run.log:78`，`track=performance`）——那条是 **attempts/B 内联 DSL**，非四件之一的「改道移植」正文；**无 `addln_prod` 判决行**（全库 grep `addln_prod` = 0） | ⏳缺证据（需开卡：把 add_ln 产品件接 `_prod_setup` 立 `addln_prod` 腿） |
| 1e | **+对拍** | 三件 `*_prod` 均带方向判别 `dir_disc`（1b/1c）与 `bitexact=True`（1a），对拍成立；add_ln 仅 intrinsic 有「有偏/无偏判别」`unb_disc=7.48e-04` | ✅（3/4 件） |
| 1f | **验证判据：gradcheck NPU 档** | **缺**：全库 gradcheck 只在 host 出现（`tests/test_ascend_train_step.py`、`tests/test_ascend_gradcheck.py`，走 fp64 同口径）；`ascend/port910b/card_run.sh` 的腿清单（`card_run.sh:229/236/261/290`）**无 gradcheck 腿**；`runs/**` 内 grep `gradcheck` 仅命中 1005 期的旧 CPU 日志（`runs/1005-p213-ascend-cpu-probe/o40.log:*`）。**卡上 gradcheck 从未运行** | ⏳缺证据（需开卡：新增 NPU gradcheck 腿） |

**三态结论：不可勾（缺 2 项）**
1. `1f` 验证判据「gradcheck NPU 档」——**从未落地**（这是本条验收的**唯一显式判据**，却无任何卡上证据）；
2. `1d` add_ln 的「改道移植」——**只有 intrinsic 形态**，无产品件真机数值。

**建议改写验收原文**（不改文件，仅提案）：
- 把「验证：gradcheck NPU 档」拆为两条可判据：① `addln_prod` 产品件真机 rel（比照 `linear_prod/dw_prod/readout_prod` 口径，形状含桶内 m 尾块）；② 一条 `*_gradcheck` 卡上腿（否则本行与 P1-1m/n 一样会「四项判据只落三项」）。
- 明确「四件」是否含 add_ln 的**产品件**口径——现状里 add_ln 的 `intrinsic` 与产品件是两份正文（`run_numerics.py:277`）。

---

## 2. `tasks.md:31` P1-3 GDN 件移植（本项目旗舰）

**验收原文（逐字，长条目全段）**：
> `- [ ] **P1-3 GDN 件移植（本项目旗舰）**：短卷积 ✓ 真机数值绿（6.35e-08，P1-1c）→ delta rule 前后向 **编译+CPU 对拍已收（F 代理 2026-10-10）**：fwd/bwd kernel ub+pure 双档 compile-PASS，六项梯度全实现（未动用 partial 例外条），CPU fp32 golden rel_o=1.29e-07/bwd 5.76e-07/adjoint-vs-FD 7.28e-10；**decay 的 log 缺件判定=口径(a) 仅需 trunk expf、零新增 compat**（`compat_patch_F.h` 仅口径(b)备用，本波不合并 trunk）。真机首执行件 `run_delta_card.py` 就绪（显式 9 张量分配，本地 compile-only PASS）—— 验证：同 P1-1d 单窗 `NUM-delta_fwd` 真机 rel`

**拆项与判据**：

| # | 判据 | 证据 | 状态 |
|---|---|---|---|
| 2a | **短卷积真机数值绿 6.35e-08** | `NUM-gdn-PASS rel=6.35e-08`（`accept-2002/artifacts/card_run.log:81` 及 `accept-2002/artifacts/sys1_legs/PERF-gdn.log:11`）。注：**第二窗未重跑该 `NUM-gdn` 腿**（window2 把 gdn 腿换成 `BENCH-gdn_conv`，`window2/artifacts/w2_all.log:54`），故此为**首窗**真机值 | ✅（首窗真机） |
| 2b | **delta 前向真机 rel**（=本条唯一显式验证判据） | `NUM-delta_fwd PASS rel_o=7.51e-08 rel_s=6.03e-08`（`window2/artifacts/w2_all.log:7` 及 `sys1_legs/delta.log:12`；`accept-2002/artifacts/sys1_legs/delta.log:12` 同值） | ✅ |
| 2c | **delta 反向 · 编译+CPU 对拍** | `F-GDN-DELTA-BWD-COMPILE-PASS`×3（ub/pure/dstate-gm，`attempts/F/RESULT.md:45-47`）+ CPU `[B] BWD-ADJOINT OK worst_rel_vs_FD=7.281e-10`、`[C] … G 5.757e-07`（`attempts/F/RESULT.md:65-71`） | ✅（CPU） |
| 2d | **fwd/bwd kernel ub+pure 双档 compile-PASS** | 8 格矩阵 + 回归 2 + 探针 1 = **11 个 `COMPILE-PASS`**，`MATRIX_EXIT=0`（`attempts/F/RESULT.md:41-52`） | ✅ |
| 2e | **六项梯度全实现（未动用 partial）** | `dQ/dK/dV/dBETA/dG/dH0` 逐项 ✅，双独立锚（FD + torch autograd）（`attempts/F/RESULT.md:99-108`） | ✅ |
| 2f | **CPU fp32 golden rel_o=1.29e-07 / bwd 5.76e-07 / adjoint-vs-FD 7.28e-10** | `[A] FWD fp32(expf) vs fp64 : rel_O=1.286e-07`；`F-GDN-GOLDEN rel=5.757e-07`；`[B] … 7.281e-10`（`attempts/F/RESULT.md:62-71`） | ✅ |
| 2g | **decay 口径(a) 仅需 trunk expf、零新增 compat** | 本波件只发 `expf`、`has_mte_copy:False`（`attempts/F/RESULT.md:43-50,85-93`）；`compat_patch_F.h` 仅服务口径(b)（`attempts/F/RESULT.md:86`） | ✅（判定成立；GAP-F **未合并**是既定边界，非本条判据） |
| 2h | **真机首执行件 `run_delta_card.py` 就绪** | `ascend/port910b/run_delta_card.py`（89 行）存在，`--compile-only` 分支在（`:42-44`），判据行 `NUM-delta_fwd PASS rel_o=… rel_s=…`（`:10,78`） | ✅ |
| 2i | **delta 反向真机 rel**（原文标题含「前后向」，但显式验证只点名 fwd） | **缺**：`sys1_legs/delta.log`（两窗）**只有 `NUM-delta_fwd` 一行，无 `NUM-delta_bwd`**；反向只有 CPU golden（2c） | ⏳缺证据（需开卡；`run_delta_card.py:1` 标题自陈「**前向**的…收数件」） |

**三态结论：可勾（就任务书字面判据）**
- 本条**唯一显式验证判据**「同 P1-1d 单窗 `NUM-delta_fwd` 真机 rel」（2b）**双窗均 ✅**；且 F 波的编译/CPU 三段（2c/2d/2e/2f）逐项有原文可 grep。
- **覆盖边界（须随勾记入，防假绿）**：
  ① 反向（2i）**无真机 rel**——只有 CPU golden，`tasks.md:31` 标题写「前后向」但验证只点名 fwd；
  ② 该真机 rel 出自 `run_delta_card.py:37` 导入的 **`attempts/F/f_gdn_delta_fwd_910b.py`**（执行件），**非接口 home `gdn_asc.py`**；接口 home 的 delta 在卡上只被 `train_step` 的「loss 下降」弱判据覆盖（无 rel）。
- 若编排者按「旗舰件=接口 home 前向+反向真机 rel」的**扩张口径**读本条，则为 **不可勾（缺 2 项：反向真机 rel、接口 home 前向真机 rel）**。建议在 `tasks.md` 原文补一句口径限定（不改文件）。
- **对坑①的回应**：已单独核过 delta——`gdn_asc.py:161-163` 明确「**不碰** `T.copy`」且 `has_mte_copy:False`（`attempts/F/RESULT.md:39`），故 P1-1q/M2 的**跨步 DMA 根因不触及本件**，delta 真机绿不是被避让「捎带」的假绿。

---

## 3. `tasks.md:13` P1-1d cube-V12 位段修复

**验收原文（逐字，长条目全段）**：
> `- [ ] **P1-1d cube-V12 位段修复（关键路径·结构修复已落地 2026-10-10）**：根因锁定=trunk `asc_copy_l12l0a/b` 发 **7 参**而官方 `LoadData2DL12L0ACal`（`mm_impl.h:32/35`）发 **9 参** `(dst,src,startIndex,repeatTimes,srcStride,dstGap,sid,transpose,inc)`——`sid`（codegen 本传入）被丢、`addrCalMode` 顶进 `sid` 位、漏 `transpose` 位 = DMA 描述符字段错位 → 507015。**已按官方 9 参位序补齐+回穿 sid**（`port910b_compat.h` 971 行，sync 入 `patches/`）；本地编译判决 `A2-COMPILE-PASS`（gemm_l1）+ `D-DW-COMPILE-PASS` + `E2E-CUBE-ONLY PASS`（主链无回归）。**transpose 路 arity 本就 8 参正确、非病灶**。单位（V2 元素/块）仍待卡定 —— 验证：单窗 bundle 复验 gemm_l1 + dW 数值 rel（≤8min，R20 禁卡上试错）`

**拆项与判据**：

| # | 判据 | 证据 | 状态 |
|---|---|---|---|
| 3a | 根因锁定（7 参 vs 官方 9 参位序） | 结构修复已落地（本条自陈）；`X_FRACTAL_EVIDENCE` §0 裁决③ 复核槽位（`tasks.md:20` 转记） | ✅（结构面） |
| 3b | 按官方 9 参位序补齐 + 回穿 sid（编译面无回归） | `A2-COMPILE-PASS`(gemm_l1) + `D-DW-COMPILE-PASS` + `E2E-CUBE-ONLY PASS`（本条自陈；`1010 …e079/notes.md:3-5`） | ✅（本地编译面） |
| 3c | **验证判据：单窗 bundle 复验 gemm_l1 + dW 数值 rel** | **被证伪**：`NUM-gemm_l1_intrinsic-FAIL … 507015`（`accept-2002/artifacts/card_run.log:66`）；`NUM-dw_intrinsic-FAIL … 507015`（`:70`）；第二窗 `BENCH-dW_intrinsic FAIL … 507015`（`window2/artifacts/w2_all.log:58`）。**两窗均未取得 rel 值**（只有崩溃码） | ❌被证伪 |
| 3d | 单位（V2 元素/块）卡定 | 本条自陈「仍待卡定」；`X_FRACTAL_EVIDENCE §0` 定论为「GM/L1 侧 stride=len 一律 32B 块」，但**跨步件单位**后续被 U2/S3 转到 gap/pitch 轴（见下） | ⏳缺证据 |
| 3e | 位序修复是否足够 | **否**：`1010 …e079/notes.md:4` 「位序修复『必要不充分』」；`accept-2002/notes.md:29` 「P1-1i（ND2NZ+sid 钉 0）的修复『必要但不充分』结论再次成立」 | ❌被证伪 |

**三态结论：不可勾（缺 1 项，实为被证伪）**
- 显式验证判据（3c）**从未满足**：两窗真机 `gemm_l1`/`dW` 均 `507015`，无 rel。
- 已定位第二层根因（新轴，非本条位段）：`U2_K48_EVIDENCE.md:17-19,263-284` 与 `S3_STRIDED_SECOND_FACTOR.md:11-13` 收敛到 **GM↔UB 跨步件的 `srcStride/dstStride` = gap 还是 pitch**；`S3` 用一手 ISA 判定「**是 gap，我们按 pitch 喂参才是错**」（`S3_STRIDED_SECOND_FACTOR.md:60-86`。注：第二窗 `w2_pitchgap.log` 的「硬件忽略 dstStride」裁决已被 `S3` 判为**度量伪影**并撤回，`window2/notes.md:41`）。
- ⇒ 本条要么维持 `[ ]`（正确），要么**改写标题为「部分结构修复」**并新开一条 `P1-1s`（已存在，`tasks.md:56` 仍未勾）。**不得因 3a/3b 的编译面绿而勾 3c**。

---

## 4. `tasks.md:37` P2 三栈联调（torch_npu + 自研件混合栈，GDN 件替换 Conv2D 路径）

**验收原文（逐字）**：
> `- [ ] P2 三栈联调（torch_npu+自研件混合栈，GDN 件替换Conv2D路径）—— **硬前置=P1-4 接口 home 九件 target=ascend 全绿✓**（否则训练步首个 SimtVF 件即编不出）—— 验证：0.8B 训练步跑通`

**拆项与判据**：

| # | 判据 | 证据 | 状态 |
|---|---|---|---|
| 4a | **硬前置：P1-4 九件 target=ascend 全绿** | `ASC-gemm/gemm_bwd_dw/add_ln/rope/attn_sw/gdn/gdn_conv/letter_readout/lora PASS` **9/9**（`accept-2002/artifacts/card_run.log:87-95`）；`tasks.md:32` 已勾 | ✅ |
| 4b | **三栈联调（torch_npu+自研件混合栈）跑通** | 最小混合栈一步真跑：`P2-STEP-ascend loss0=1.077804 loss1=0.969059 reduced=True params_grad_ok=True` + `KERNEL-USED-PASS kernels=13 blockers=0 fallbacks=0`（`window2/artifacts/w2_all.log:12-13`）；fp16 档 `0.975384→0.896910`（`:15-17`）；`RUNNER-DONE … exit=0`（`:31`） | ✅（**最小**栈） |
| 4c | **GDN 件替换 Conv2D 路径** | **缺**：`gdn_conv_asc.py:16` 自陈位点「原本…落到 torch 的 Conv2D 路径上」且本入口替换它，但 `train_step.py:42-44` **明确不跑 gdn_conv**（「`gdn_conv` 门面不在本件里跑」）⇒ 训练步里**无 Conv2D 位点**，「替换」未被任何**联调**实证（只在接口层编译 `ASC-gdn_conv PASS`） | ⏳缺证据 |
| 4d | **验证判据：0.8B 训练步跑通** | **缺/不符**：`train_step.py:402` 硬编码 `seq, d, heads, dim = 16, 32, 2, 16`（文档头 `:9` 自陈「参数量任意小（d=32, seq=16, heads=2, dim=16）」）——**跑的不是 0.8B**（0.8B = 学生基座 Qwen3.5-0.8B，`README.md:20`）。全库无 0.8B 训练步产物 | ⏳缺证据 |

**三态结论：不可勾（缺 2 项：0.8B 训练步、Conv2D 替换实证）+ 建议改写验收原文**
- 硬前置（4a）✅、**最小**混合栈一步（4b）✅——但 `tasks.md:38` 的「卡窗唯余：`--target ascend` 翻 device…即跑 **0.8B** 一步」与出厂件 `train_step.py` 的 **d=32 最小冒烟**不符：`train_step.py:6-7` 自陈「本波只验**接线正确性**…**不验内核数值 rel/V\***」。
- **建议改写验收原文**：把「验证：0.8B 训练步跑通」改为可判据二选一——①「最小混合栈一步 `P2-STEP-ascend` PASS + `KERNEL-USED-PASS` 三证」（**已满足，可勾**）；② 另立「0.8B 规模一步」（需接线到真模型 + 开卡，当前未做）。同时显式登记「GDN 件替换 Conv2D」=**接口层已替换（`gdn_conv_asc.py`）+ 联调层未纳入训练步**（`train_step.py:42-44`）。

---

## 5. `tasks.md:39` R6/R7 成本记账与 gate 报价

**验收原文（逐字）**：
> `- [ ] R6/R7 成本记账与 gate 报价（恢复 C5 的前置）—— 验证：报价行+熔断判定`

**拆项与判据**：

| # | 判据 | 证据 | 状态 |
|---|---|---|---|
| 5a | **成本记账（报价行）** | 8 条标量 `BENCH-QUOTE` 存在且**可用**（`window2/artifacts/w2_all.log:34-41`：linear/dW/add_ln/rope/attn_sw/gdn_conv/gdn_delta_fwd/gdn_delta_bwd） | ✅（8 条标量侧） |
| 5b | **gate 报价（对照侧）** | **缺**：`BENCH-QUOTE linear_intrinsic`（`w2_all.log:42`）与 `BENCH-ABLATE ratio=3690.3x`（`:43`）**一律作废**——`V2_THROUGHPUT_TRUTH.md:8-17,129-137` 判定该 intrinsic 件是 `__mix__(1,2)`+`if(ASC_IS_AIV[=0])` **死代码**（空发射）、`dW_intrinsic` 真执行但 `507015` **坏件** | ⏳缺证据 |
| 5c | **熔断判定** | **缺**：`accept-2002/notes.md:38` 「未跑 `SYS1_BENCH`…吞吐/报价数据仍待下一次；**R6/R7/C5 未闭**」；`V2_THROUGHPUT_TRUTH.md:204-209` 「**三栈消融无可引用数据**」⇒ 无法据对照侧出熔断结论（D12 熔断线 ¥600 的空比较） | ⏳缺证据 |

**三态结论：不可勾（缺 1 项：熔断判定所需的对照/消融侧数据）**
- 5a 的 8 条标量报价行**存在**，但 5b/5c（本条验证判据的「熔断判定」半边）**无任何可引用数据**：第三栈两条腿一死一崩（`V2_THROUGHPUT_TRUTH.md:129-137`），故「报价行+熔断判定」中的**熔断判定不可产出**。
- 与用户坑②一致：凡验收原文含「吞吐/报价/消融」者，现在**不可勾**。（本条的「报价」虽部分落地，但「熔断判定」为其与门条件，故整条不可勾。）

---

## 附录 A · 关键判决行原文索引（均可 grep）

```
accept-2002/artifacts/card_run.log:66   NUM-gemm_l1_intrinsic-FAIL … 507015
accept-2002/artifacts/card_run.log:70   NUM-dw_intrinsic-FAIL … 507015
accept-2002/artifacts/card_run.log:78   NUM-addln_intrinsic-PASS rel_y=1.66e-05 rel_h=0.00e+00 unb_disc=7.48e-04 track=performance
accept-2002/artifacts/card_run.log:81   NUM-gdn-PASS rel=6.35e-08
accept-2002/artifacts/card_run.log:84   NUM-delta_fwd PASS rel_o=7.51e-08 rel_s=6.03e-08
accept-2002/artifacts/card_run.log:87-95 ASC-{gemm,gemm_bwd_dw,add_ln,rope,attn_sw,gdn,gdn_conv,letter_readout,lora} PASS
accept-2002/artifacts/card_run.log:97   ASC-PROD-linear_prod FAIL … rel=1.807e+01（首窗）
accept-2002/artifacts/card_run.log:98-99 ASC-PROD-dw_prod PASS rel=1.36e-07 / ASC-PROD-readout_prod PASS rel=0.00 bitexact=True
window2/artifacts/w2_all.log:7          NUM-delta_fwd PASS rel_o=7.51e-08 rel_s=6.03e-08
window2/artifacts/w2_all.log:8,22       ASC-PROD-linear_prod PASS rel=1.59e-07
window2/artifacts/w2_all.log:12-17      P2-STEP-ascend fp32/fp16 + KERNEL-USED-PASS kernels=13
window2/artifacts/w2_all.log:31         RUNNER-DONE SUMMARY pass=10 fail=0 … exit=0
window2/artifacts/w2_all.log:34-41      8 条标量 BENCH-QUOTE
window2/artifacts/w2_all.log:42-43      BENCH-QUOTE linear_intrinsic / BENCH-ABLATE ratio=3690.3x（作废）
window2/artifacts/w2_all.log:58         BENCH-dW_intrinsic FAIL … 507015
```

## 附录 B · 本次实际读过的文件清单

**任务书/契约**
- `openspec/changes/teacher-p2-production-full/changes/p2-13-ascend-runtime/tasks.md`（全文 65 行）
- `openspec/changes/teacher-p2-production-full/changes/p2-13-ascend-runtime/design.md`（grep D12/熔断/cube2）

**run 台账与腿原文**
- `release/runs/1011-p2-13-oncard-accept-2002/notes.md`
- `release/runs/1011-p2-13-oncard-accept-2002/artifacts/card_run.log`
- `release/runs/1011-p2-13-oncard-accept-2002/artifacts/sys1_legs/delta.log`
- `release/runs/1011-p2-13-oncard-window2-throughput/notes.md`
- `release/runs/1011-p2-13-oncard-window2-throughput/artifacts/w2_all.log`
- `release/runs/1011-p2-13-oncard-window2-throughput/artifacts/sys1_legs/delta.log`
- `release/runs/1010-p2-13-oncard-wave2-cube-and-gate-e079/notes.md`

**reconcile / attempts 证据文档**
- `release/ascend/kernels/reconcile/V2_THROUGHPUT_TRUTH.md`
- `release/ascend/kernels/reconcile/S3_STRIDED_SECOND_FACTOR.md`
- `release/ascend/kernels/reconcile/U2_K48_EVIDENCE.md`
- `release/ascend/kernels/reconcile/M2_FP16_NAN_EVIDENCE.md`（1-70 行）
- `release/ascend/kernels/reconcile/C2_dtype_matrix.md`（1-70 行）
- `release/ascend/port910b/attempts/D2_variants.md`
- `release/ascend/port910b/attempts/F/RESULT.md`

**产品件/工装**
- `release/ascend/port910b/run_delta_card.py`
- `release/ascend/port910b/run_numerics.py`（grep 目标/case）
- `release/ascend/port910b/card_run.sh`（grep 腿清单）
- `release/ascend/train_step.py`
- `release/ascend/kernels/gdn_asc.py`、`gdn_conv_asc.py`（grep 结构/Conv2D）
- `release/ascend/port910b/INVENTORY.md`（grep Conv2D，旁证）

## 附录 C · 纪律自陈

- 本文件为**唯一新增物**（`release/ascend/kernels/reconcile/C3_closure_audit.md`）；**未改** `tasks.md`、**未改**任何产品件/工装；**未做** git 写操作。
- 全程「只认证据不认叙事」：每条判据均挂 `文件:行` 或判决行原文；找不到证据即标「缺证据」，无「应该满足」。
- 已知两坑落点：坑①（P1-1p/q=避让非根治）→ §2 已单独核 P1-3；坑②（三栈消融无可引用数据）→ §5 直接定为不可勾。
