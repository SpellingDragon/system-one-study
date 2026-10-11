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
