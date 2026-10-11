# 1012 00:02 · 窗三「钉死窗」五段（repo `7c7a8a1`；runner `window3_card.sh`；墙钟 990s ≈16.5 分钟）

产物：`artifacts/`（w3 五段日志 + `card_run.log` + 14 腿原文 + **sys1_kernels.tgz 574,211B / 48 kernels**）
`W3-A-EXIT=0 / B=1 / C=1 / D=0 / E=0`；A 段 `RUNNER-DONE pass=10 fail=0 exit=0`（连续第二窗全绿）

## ① A 段：第四件产品件真机闭环
| 判决 | 值 |
|---|---|
| **`ASC-PROD-addln_prod`（本窗新增）** | **PASS** `rel_y=1.98e-05 rel_h=0.00e+00 unb_disc=7.46e-04 reuse=1/1 shapes=2 track=card` |
| `linear_prod`/`dw_prod`/`readout_prod` | PASS `1.59e-07` / `1.36e-07` / `0.00 bitexact`（与窗二逐字同＝**跨窗可复现**） |
| delta fwd | PASS `rel_o=7.51e-08 rel_s=6.03e-08`（逐字同） |
| fp32 / fp16 训练步 | `1.084437→0.941315` / `0.975380→0.896960`，两档均 `KERNEL-USED-PASS kernels=13 blockers=0 fallbacks=0`，`DTYPE-TRACE fp16 fp32_accum=true` |
⇒ **“判决件量的是产品件”纪律在四件上全部闭合**（P1-1x 真机达成）。

## ② C 段：**真机首现 GDN 反向数值缺陷**（本窗最大发现）
```
NUM-delta_fwd PASS rel_o=7.51e-08 rel_s=6.03e-08
NUM-delta_bwd FAIL first=dbeta rel=5.997e-01 absdiff=4.435e+00 thresh=2e-04
```
- **同一条腿 host cpu-alias 六项全 `1e-7` 量级**（编排者本窗复验过），负例 ×1.01 能逐项指认 ⇒ 判决件本身可信；
- ⇒ **`dbeta` 是 device-only 缺陷**（rel 0.60、绝对差 4.4，非噪声量级）；其余五项在本窗未显示（FAIL 早退，**不知道后面几项是否也坏**——下窗需 `--per-item` 或容错模式逐判）。
- **P1-3 保持未勾**（诚实：反向数值真机判负）。**loss 下降不是反向正确的证据**——本窗正是这句话的实证（训练步 PASS 与 dbeta 错 60% 同窗共存）。

## ③ B 段：卡上 gradcheck 6/8，2 件 FAIL **须先分诊再定责**
```
GRADCHECK-gdn_delta FAIL max_rel=nan eps=0.01
GRADCHECK-gdn_conv  FAIL max_rel=n/a eps=0.01
GRADCHECK-SUMMARY ops=8 pass=6 fail=2 invalid=0 kernel_used=8/8
```
`linear/add_ln/rope/attn_sw/readout/lora_apply` 六件真机 `1e-5…2e-5`（阈值 1e-4）PASS。
两件 FAIL **不得直接算产品缺陷**：V2 已实证“`randn` 当递推门会让**参考自身**爆 nan”，B3 的 host 轨 8/8 PASS 而卡上挂 ⇒ 两种可能（探针门输入在本窗分布下失良态 / 真梯度缺陷，与 C 段 `dbeta` 同源）。**归因动作＝本地对照 B3 的门输入与 C4 的 seed 口径**，先判探针再判产品。

## ④ D 段：§12 排除性结论 + 一处工装缺陷
- 机械面：`[default] so=BASE`、`[A] so=DIFF(dW_l0tr,main)`、`[C] so=DIFF(main)`、`valid=3/3`、注入行齐 ⇒ **变体确实改机器码**（工具可信）。
- 颜色面：`+A` 与 `+C` 两趟 `gemm_l1_intrinsic`/`dw_intrinsic` **均仍 507015**；default 趟出的是 **`_LaunchSkip`**（**未真执行**）——病灶＝default combo 把 `compile_all` 腿与 intrinsic 腿放**同一进程**，compile_all 的发射桩传染了 intrinsic 腿（**A2 当年同款问题的重演**，R17/一案一进程纪律）。
- 但 default 的红由 **A 段独立进程**证得（`advisory_fail=3` 含 gemm_l1/dw `507015`）⇒ 三态可拼：`default=R、A=R、C=R` ⇒ 按 `D2_variants.md §5.2`，**A 覆盖的 GM→L1 nd2nz 选择、C 覆盖的读侧拆趟/折指针均非 507015 触发者**（排除性收敛，比窗二更进一步）。

## ⑤ E 段：**H-b 实测为真，且只在跨步档**（第二层因素锁定）
```
k=48/none, k=48/relu, k=96/none, k=96/relu （dma=strided）⇒ STALE-DIFF 与 none 不同的 prefill=['zero','perm'] ⇒ H-b 真
k=32/none, k=32/relu （dma=single）        ⇒ prefill=无 ⇒ H-b 未触发
所有候选 mm+nn>0（含 sym=0 最优仍>0）⇒ 非单一几何 ⇒ 看 STALE-DIFF
STALE-DIFF>0 ⇒ H-b 真：残留含 stale ⇒ 修复须同时处理 UB 预清零/落趟拆解
```
⇒ 跨步装填根因模型由“单因子（gap/pitch）”升级为**双因子**：`F1 gap 语义误用`（一手 ISA）＋ `F2 UB 残留跨发射污染`（**本窗实测**，且**只在 `blockCount>1` 出现**，与单发档恒不触发形成对照）。这解释了 S3 的 30.9 万组穷举零命中。
**可检验推论（标为假设，非事实）**：GDN 反向若也走多突发跨步装填，则 C 段 `dbeta` 缺陷与 E 段机制同源——下窗用「UB 预清零后 dbeta 是否改善」一枪判。

## ⑥ 本窗结论与下一步
- 主线**未被阻断**：正确性链、四产品件、两档训练步、gradcheck 六件、delta fwd 全绿；新缺陷定位到 **GDN 反向 `dbeta`（device-only）**。
- 成本：16.5 分钟 ≈ ¥5.5。累计开卡约 8 窗 ≈ 3.5–4 卡时（¥70–80），距 ¥600 熔断线仍远。
- 下一步（**全部不占卡**）：①`dbeta` 缺陷离线取证（用本窗 48 枚生成码归档 + 双因子假设）②B 段 2 FAIL 的探针/产品分诊 ③D2 default 趟拆进程（一案一进程纪律回归）④compat 双因子修复提案更新（`S3_compat_proposal.diff` + UB 预清零）⑤真形档 0.8B 窗（等你，R6/R7 前置）。


## ⑦ 离线分诊与取证（D3/E3 两路，编排者判读）
### A. **我的同源假设被证伪（如实撤回）**
我在 §⑤ 写过"可检验推论：若 GDN 反向也走多突发跨步装填，则 `dbeta` 与 E 段机制同源"。D3 生成码取证（三份产物现场复现，md5 `8b11e54e/5393f4be/7092e74e`）：
- `gdnb[ascend]` **0 条 DMA**；`db_acc/dq_acc/dk_acc/da_acc` **皆寄存器**、与 `dg/dh0` 不共享缓冲；UB 八槽**先写后读**（`firstW<firstR`）、`lam_ub` 入口清零
⇒ **该内核内没有 stale 落点**，`dbeta` 与跨步/stale **结构上不同面**。**假设撤回**（它当时已标 ❓，未污染结论；教训＝"同窗共存"不等于"同源"）。

### B. `dbeta` 责任面收窄（D3）
- 工装改造：`run_delta_card.py` 加**容错逐项判**（`ITEM <name> rel= verdict=` 七条，聚合行与 fwd 行**逐字不变**；host 复跑六项仍 `1e-7`、负例 ×1.01 逐项可指认）⇒ 下窗可见"只坏 dbeta 还是多项坏"。
- **关键发现＝variant 混淆**：host 腿走 `pure`、C 段真机腿走 `ub`；而 `--variant ub` 在 cpu 后端**编译即报错**（`Only one dynamic shared memory allocation is allowed`）⇒ 我此前的"host 过 / device 挂"对比**跨了两个不同档**，"设备侧共享代码写错"前提**不成立**。
- 生成码层面 `dbeta` 在 `ub/pure/cpu` 三方**代数同构**（`db_acc += du*(v−p)`，无独立写法、无缺 `float32` 累加）⇒ 责任面收窄到 **`ub` 档的设备执行态**，最后一层必须占卡（D3 已备 `D3_NEXTSHOT_dbeta.md` S1/S2/S3 一枪判三事）。

### C. gradcheck 两件 FAIL 均判**真缺陷**（E3 证清探针）
| 件 | 卡上原文 | 分诊 |
|---|---|---|
| `gdn_delta` | `FAIL max_rel=nan` | **真缺陷**：无异常⇒算完含 nan；门分布三处口径实测**均良态**（gradcheck `max=-0.0022≤0, seq=4`；bench V2 `_gate3`；C4 seed=11 `max=-0.093`），fp64 参考有限 `max_abs=8.23`、host fp32-vs-fp64 `1.68e-07` ⇒ **探针无罪**，坏点在 device 侧内核 |
| `gdn_conv` | `FAIL max_rel=n/a` | **真缺陷**：`n/a`＝**异常被吞**（`run_one:271-280` 见 exc⇒rel=None⇒`_line:330` 打 `n/a`）；异常原文 `RuntimeError: SetPrecisionMode…500001`，抛点＝产品反向闭式 **`autograd_asc.py:434 → F.conv1d`**（torch_npu/ACL 对该 depthwise-conv 反传不接受）⇒ 与输入分布无关 |
⇒ **窗三 B 段 `fail=2` 保持指控力**，探针输入分布一字未改（改了会把真缺陷洗成噪声——这是正确的取舍）。
另：E3 报一处既有隐患（`B3_probe_negatives.py` L82 `None` 格式化崩，仅 no-kernel 分支触发）；容器 py3.10 下 cpu 后端不可用（`Forward references must evaluate to types`），host 8/8 基线只在 mac py3.12 成立——**跨环境复现须认这个差**。

### D. 窗三后缺陷台账（全部集中在 **GDN 反向**）
1. `dbeta` 真机 `rel=0.5997`（`ub` 档设备执行态）
2. `gdn_delta` gradcheck `nan`（device 侧解析反传）
3. `gdn_conv` 反传 `SetPrecisionMode 500001`（产品 fallback 依赖 `F.conv1d`，NPU 栈不接受）
⇒ **P1-3（GDN 件移植，旗舰）的真实缺口就是"反向在真机上不正确"**；前向与其余七门面均已真机绿。这不是"任务无法完成"，而是**缺口被精确到 3 条**——其中 3 号可在本地修（去 `F.conv1d` 依赖，改自研内核），1/2 号需一枪占卡判（`ub/pure` 差分 + 逐项判决 + gradcheck 逐路）。


## ⑧ 本地修复与工装修缮（F4/G3，编排者独立复验）
- **F4（P1-1zb 本地修完）**：`gdn_conv` 反向**去 `F.conv1d`**，新增昇腾件 `conv_bwd_asc_impl`（复用前向同族 scalar 机制，`ascend|gdnconvb[ascend]|c16k4b2|c8scalar`，容器真编 `.o~4898B`/`aibin 6416B`/`so 33392B`）。**编排者复验**：`grep -c conv1d(` 两文件 **0 命中**；host `GRADCHECK-gdn_conv PASS max_rel=2.653e-05`（**与旧值逐字同**，未调阈值）、`SUMMARY ops=8 pass=8 kernel_used=8/8`；`Y-linear_prod-PROD-PASS rel=2.54e-07…` 判决行未漂移；`ITEM verdict= PASS` 七条；`pytest 340 passed`。负例两条（抽头方向反 `7.479e-01`、漏末抽头 `4.663e-01`）均当场抓红。F4 自曝一次回归并自愈（`_conv_silu_torch` 原假设 3D，gradcheck 传 2D → `IndexError`，改内收形后 340 passed）。
  - 决策记录（备选与未采用）：①手工重算前向借 torch 反传＝旧路（NPU 拒，`500001`）；②反向塞进前向模具＝pre-activation 未留痕（撞 G-F3 或退回①）；③**采用**独立反向件 + UB 累加器。
- **G3(a)（工装修缮）**：`D2_probe_legs.py` 改**每腿独立子进程**（`compile_all` 与 intrinsic 腿永不同进程）。**编排者容器实跑**：`--combos default,A,C` → `valid=3/3`、`LEG-VERDICT D2-LEG-OK ×3`、九数 `.o~2850/2250/3625/2106/2911/4003/1864/922/2842B` **逐字同**；`_LaunchSkip` 命中数见本节首行（应为 0）。
- **G3(b)（双因子提案 v2，不落地）**：`attempts/S3_compat_proposal_v2.diff`（v1 gap 归一 ＋ 因子二 **UB 预清零** `vector_dup`，`blockCount==1` 跳过⇒对 SINGLE 无回归）。**两因子确实进码**由预处理原文证实：`srcStrideB/32→srcGapB/32`、`vector_dup` 三处（v1/trunk 均 0）。trunk 三处 md5 复验未变 `ca023c2e`。
  - **诚实边界**：v2 仍是提案，**真机判决未做**；且 G3 明确 **不声称顺带修 `dbeta`**（D3 已证 `gdnb` 无 DMA，不同面）。
- **编排者本轮两次自犯同坑（记档防再犯）**：① `grep -c` 返 0 时 exit 1 断 `&&` 链（第二次犯）；② 把 `2>/dev/null` 加在 docker exec 上，吞掉自己写错路径的报错（`/work/port910b` 应为 `/work/ascend/port910b`）——**与窗二 scp 同类错误**。结论：**任何验证命令都不得静默 stderr**。


## ⑩ 重大上游事实更正（2026-10-12，决策人提供的 CANN org 链接引出；按"先核对已有结论是否被推翻"纪律）
**此前结论"PMC 不维护 910 卡"只对 tilelang MAIN 仓成立，整体判断被推翻**：
1. **`tile-ai/tilelang-ascend` 是活跃维护的官方 Ascend 适配仓**（API 元信息实测：`fork:false`、描述 "Ascend TileLang adapter"、**默认分支＝`ascendc_pto`**、404 星/175 fork/**534 开放 issue**、created 2025-09-25、**pushed 2026-10-09**）；CANN 官方 org 页（gitcode.com/cann）"相关链接"直接挂它。后端双轨：AscendC & PTO / AscendNPU IR；README 含正统 cube 路（`T.gemm_v0(A_L1,B_L1,C_L0, init=…)`）、Flash/Sparse/Linear Attention & RNN 示例、ACLGraph、torch 集成；要求 CANN ≥8.3.RC1（我们的 8.5.0/8.5.2 满足）。
2. **我们的一手取证被上游独立佐证**：#1762 `[Bug] Scalar BF16-to-FP32 casts fail in the BiSheng backend`（open, 09-08）＝我们 C2 的 `not support bf16 type cast`；#1758 `[Bugfix] Lower scalar BF16-to-FP32 casts through AscendC` —— 修复路线已存在。
3. **旗舰 P1-3（GDN delta rule）正被整族上游化**：#1191 `chunk_gated_delta_rule`（closed＝已合）、#1799 `chunk_delta_h`、#1827 `WY Fast`、#1866/#1802 `chunk_o fwd/bwd`（2026-09/10 密集提交）。
4. 另：#1830 open（dav-2201 编译路径 bug，2026-09-22）说明官方适配仓在 910B 代系上也有未闭问题——**采用前必须实测，README 承诺≠我们环境的判决**。

### 对既有决策的影响（待决策人确认方向调整）
- **§9 决策记录的"上游回馈目标"改判**：未来整包 PR 的对象应从 tilelang main 改为 **tilelang-ascend**；我们的 compat/取证（bf16 复现、跨步语义）在该仓是**佐证材料**而非自用补丁。
- **§12 考古（P1-1d/P1-1s）的战略价值降级**：私 fork 在 main 仓上模拟 950 内联函数打到 910B，本质是在复刻适配仓已原生解决的层次；若适配仓实测可用，§12 轨可转为"对齐适配仓"而非继续私养。
- **P1-3（GDN 旗舰）出现第二条完成路径**：上游 chunk 族（含 bwd）实测 + 我们的可信判决件（gradcheck/delta_bwd/dtype 矩阵）做验收——比继续调我们手写反向在 `ub` 档的三个真机缺陷可能更快更稳。
- **不变**：窗四仍值得跑（15–20 分钟，把三个 GDN 反向缺陷的责任面钉死，无论底座换不换都是知识）；标量正确性链与判决件体系是**可迁移资产**。
