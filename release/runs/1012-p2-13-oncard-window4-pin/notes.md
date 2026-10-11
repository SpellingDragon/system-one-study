# 1012 02:2x–02:5x · 三窗连跑会话（窗四钉死 + 窗五 tilelang-ascend + 窗六真形档·被实例停止中断）

repo `caf824e`；实例资源盘点（决策人要求）：**910B4 单卡、HBM 32GB（闲时余 ~29.9GB）**、Kunpeng-920 aarch64 ×192 核、内存 1.5TiB、盘 50G（全新实例仅用 478M）、`python3.12 + torch 2.7.1 + torch_npu 2.7.1.post2`（py3.12 顺带避开 E3 发现的 3.10 前向引用坑）。

## 窗四（首跑 135s 全灭 → 修复引导段 → 重跑 450s 全段跑通）
- **首跑教训（我的预检盲区）**：全新实例 pip 无 tilelang（旧窗由 card_run [2] 装）⇒ `TLPKG` 空连坐全灭。修复＝窗四借 `CARDRUN_STOP_AFTER=2` 引导（装 tilelang+overlay）+ 我 V5 段相对路径坑（`cd 空变量` 落到 HOME）。**已被实例停止打断的 ssh 探测又犯两次 compinit 内联坑——教训再次固化：远端操作一律脚本文件。**
- **V1 卡上 gradcheck 7/8**：六件 PASS（`1e-6…2.5e-5`）；**`gdn_conv FAIL rel=5.538e+04`**（F4 新反向件 host 对/容器编过/**卡上错**——device-only 缺陷第 4 例）；**`gdn_delta "PASS max_rel=nan"` 判据异常**（同窗 V2 却 `rel=2.21e5 pass=False`）⇒ **探针 nan 比较路径缺陷，须修**（窗三同件同输入判 FAIL，本轮判 PASS，不可信）。
- **V2 E3 逐路分诊**：`gdn_delta FWD finite` 全输入方向 `ana`（8–42）与 `num`（816–1.75e4）差 3 个量级 ⇒ **前向敏感度或反向内核二选一坏**（`VERDICT=clean`＝无异常无 nan，纯数值差）。
- **V3 delta_bwd 逐项**：**只有 `dbeta` 坏**（`rel=8.002e-01`），其余五项 `6.2e-8…1.0e-7` 全 PASS、`adjoint 6.0e-10`——隔离干净。
- **V4 D3 双变体差分（本窗最关键）**：
  - `variant=ub`：**六项全 PASS（dbeta `1.018e-07`）**；但 `DBETA-FP` 三跑指纹 `e609a6ab/2a507f3e/45266aef` **随 prefill 变** ⇒ ub 数值对但**残留 stale 敏感**（隐患）。
  - `variant=pure`：`dbeta FAIL 8.002e-01`（与 V3 同值）；指纹三跑恒定 `e609a6ab` ⇒ **pure 确定性但错**。
  - ⇒ **P1-1z 定位收敛：`dbeta` 缺陷＝`pure` 变体的设备执行；设备侧换 `ub` 即六项全对**（修法＝变体选择，一行级；stale 隐患另立）。V3 腿失败说明 `run_delta_card` 设备腿实际走 pure（D3 早前"设备=ub"的说法反了）。
- **V5 双因子提案 v2 判决**：补丁 `mode=p1` 应用成功（md5 `dabd4751`，与容器编译版一致）、头文件已还原（`ca023c2e` 复验）；**打补丁后 nextshot `507035 vector core exception`** ⇒ **方案 A（`vector_dup` 预清零）在卡上运行时崩**——按 S3 真值表**退方案 B（每突发单发拆趟）**。BASE 趟 4 个 strided 格 STALE-DIFF 仍非零（H-b 复证）。

## 窗五（tilelang-ascend cube 栈，225s，里程碑）
- **CANN 8.5.2 ABI 金丝雀过**（容器构建包 `d5607ec6` 解包即用，免卡上编译）。
- **六件 GDN 全部真机 PASS**：`cumsum/kkt/solve_tril/wy_fast/chunk_h/chunk_o`，`rel=0`（solve_tril `5.96e-08`），**且全部满足上游自测契约 `upstream1e-5=ok`**；`TA-KERNELPROOF bisheng_calls=8 cache_so=9 verdict=real-kernel`（真 JIT 内核非 eager）。
- **gemm**：fp16 `512³ rel=0 p50=0.410ms`、`1024³ rel=3.78e-4 p50=0.400ms`；fp32（改 block 64/128/64）`FAIL rel=nan`（该配置腿坏，登记）。
- **三栈消融的 cube 栈补齐**：`linear 512³ fp16` 我们标量路 `294.3ms` vs tilelang-ascend `0.41ms` ⇒ **~718×**。V2 留下的空档正式填上；D-cube2（性能路唯有 cube）获真机定量佐证。

## 窗六（真形档，被实例停止中断）
- 8192 档：拓扑行已出（`d=1024 L=24 gdn=18 full=6 heads=8 kv=2 hd=256 inter=3584 vocab=248320 conv_k=4 gdn_h=16x128 … dry=false src=builtin-literals（config.json 缺——拓扑兜底口径，数值同源））；**25 分钟超时 EXIT=124** ⇒ **真形档一步在标量路上 >25 分钟**——本身就是 R6/R7 的硬数据：**标量路训练不可经济**（>¥8/步），性能路必须走 tilelang-ascend（与窗五 718× 互证）。
- 4096 降级档跑到 ~16 分钟处实例连接断（应系决策人停止实例）；结果未落。若盘保留，下次启动先 `cat /tmp/w6_all.log`。

## 会话结论
1. **正确性**：四产品件+delta fwd+fp32/fp16 训练步维持全绿；GDN 反向缺口收敛为 **2 个 device-only 缺陷**（`pure` 变体 dbeta〔修法已明：设备侧选 ub〕、F4 conv 反向件）+1 个探针 bug（gradcheck nan 判据）。
2. **性能**：cube 栈真机数据到手（GDN 六件 rel=0 + gemm 718×）——**底座对齐（B 路线）由 README 承诺升级为真机判决**。
3. 待本地（不占卡）：①设备腿变体选择修 dbeta（连带 gdn_delta 2.2e5 复验）②gradcheck nan 判据修 ③conv 反向卡上缺陷归因 ④v2→方案 B ⑤R6/R7 报价按"tilelang-ascend 路径"重算口径。


## ⑧ 勘误（K4 三重证据推翻本账 §V4 的解读——第三次被代理证据纠回，机制有效）
我写的是："V3 腿失败说明 run_delta_card 设备腿实际走 pure（D3 早前'设备=ub'反了）"。**错**。K4 证据：
1. **代码**：`run_delta_card.py:60-63` `_variant()`：ascend→**ub**，且早于窗二即如此（D3 未动它）；
2. **指纹**：V3 的 `dq=6.999e-08/dk=9.322e-08/dg=1.016e-07` 与 **V4-ub 逐字同**、与 V4-pure 全异；
3. **键**：`c2ubz`。
**真正的机理**（K4 点破）：`ub` 的 **fresh（首次发射）指纹 `e609a6ab` 恰好 == pure 的指纹**——V3 的 dbeta 错不是因为走了 pure，而是 **`ub` 在 UB 未初始化（fresh）时表现与 pure 相同；`prefill=zero` 后 dbeta 才对（`1.018e-07`）**（D3 探针的 ITEM 读的是 `runs["zero"]`，`D3_probe_dbeta_order.py:160`）。
⇒ **dbeta 根因改判：`gdnb` 内核首次发射时读取了未初始化的 UB**（D3 早前"八槽先写后读无 stale 落点"的静态结论被顺序实验推翻——存在某条首迭代读-before-写路径）；**修法＝UB 预清零**（非变体选择）。
⇒ **K4 对台账必答项的回答：生产训练步（train_step/train_step_real → autograd_asc.gdn_delta）在设备上走 `ub`** ⇒ dbeta 缺陷**在生产路径上**（首次发射即错）⇒ 高危。
⇒ **统一假说（升级为待验主线）**：GDN 族 device-only 缺陷群（dbeta 0.80 / gdn_conv 5.5e4 / gdn_delta 2.2e5）可能同为"**首次发射未初始化 UB 读**"一类——与窗三 E 段跨步 stale（H-b）同族但不同落点。**一枪可判**：每 op 发射前做一次 UB 清零发射（或预热发射丢弃首跑），三缺陷同窗复判。
- 修正 tasks：P1-1z 修法从"变体选择"改为"**P1-1zg UB 预清零**"；变体选择无需改动（已是 ub，K4 零码改、负例 `--variant pure→c2purez` 证明开关为真）。
