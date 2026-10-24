# 第四批：Ascend/GDN 的技术证据与误判复盘

## 结论

GDN v3 是有历史目标形精度读数的研究候选，不是已接入 P2 正式训练的交付件。当前优先事项是**修正验证覆盖和消费链认知，再执行已有收紧精度裁决**；“换成少量 kernel 就能达到某个速度”没有足够证据。[A01–A09]

本批没有上卡。所有设备数字是历史日志；CPU 复验和静态分析的边界分别标注。

## 1. 路线与命名边界

| 路线 | 当时解决的问题 | 已有资产 | 不能据此推出 |
|---|---|---|---|
| tilelang 主仓 910B 适配／标量路径 | 代际方言、intrinsic 与设备兼容 | CPU 对拍、编译件、部分设备记录 | 所有目标算子均数值与性能可用 |
| TA v2 chunked 五核链 | 避免逐 token 路线的计算／搬运问题 | cumsum、solve_a、scan1、scan2a/2b | v2 可无条件当高精度 oracle |
| TA v3 分解候选 | 隔离复杂内核与多梯度问题 | 小核、host 正反向循环、golden、探针 | 已接入系统训练；尾块/fp32 均支持 |

主仓 `tilelang.ascend.language` 与 tilelang-ascend 示例中的 `tilelang.language` 属不同方言入口／版本；不能把一个仓的 `T.Kernel` 参数规则直接套给另一个仓。本教材只解释实际被测版本，不拼接不同方言的 API。[G05、A01]

## 2. A-F1：正式消费者仍是 gdn_asc，不是 v3

```text
release/ascend/train_step.py → ops.gdn_delta
→ autograd_asc._GdnFn → kernels/gdn_kernel.py
→ gdn_asc.forward/backward
```

`gdn_kernel.py:29-34` re-export 产品 gdn_asc；`autograd_asc.py:378-384` 按 `(dq,dk,dv,dg,dbeta)` 的顺序返回输入梯度。它没有 h0 输入，不能直接把 v3 的 `(dq,dk,dv,dbeta,dg,dh0)` 六元组接过去。

v3 的 q/k/v 等带 batch/head 维，状态布局与产品层也需明确转换。对 `release` Python 生产路径检索未找到 v3 relay 的消费调用。因此下一轮接入项需要包括输入布局、梯度顺序、h0 语义、dtype、自动微分及端到端负向锚，不能仅导入一个新函数。[A08]

## 3. A-F2：原生路径、oracle 与替换实验是三件事

v3 只复用 v2 的 `cumsum/solve_a/scan1`，不调用 scan2a/2b；rsa 的分母来自 example 的 NumPy chunk golden，而非 v2 五核链。v2 的独立正确性风险不自动推翻 rsa，但如果另一实验把 v2 当分母，就需补证。[A01–A02]

当前 `test_gdn_bwd_v3.py:93` 的 `GDN_V3_GOLDEN_H` 默认已是 0。置 1 会用参考 A/H_in 覆盖后续消费的中间量，并打印警告；它只隔离下游，不证明被替换的前置件。原生前置计算仍可能被执行，不能把替换输入和“完全不运行该算子”混为一谈。

六项输出中 `dk` 是最终写回，不参与下一块递推；矩阵递推是 `delta_f32`。`dg` 还存在 `carry` 的跨块后缀累计，因此“唯一跨块量”应限定为矩阵伴随状态，不能忽略 carry。[A01]

## 4. A-F3：BK 绕行证据成立，微观根因与候选真修仍未闭环

历史 wsc：DK128/BK64 的 A/H_in rel 为 `1.388e-1 / 4.307e-2`；BK128 对照为 `5.961e-5 / 3.992e-4`。[A05]

这支持“问题与被绕开的路径有关”，不是“所有分块累加都有错”。notes [155] 的直接搬运和动态 init 最小形式能通过，而组合形式产生异常；fill 顺序、UB→L1 与源 stride 的作用没有全部独立分离。

当前 v2 solve_a 已改为 GM→L1 直接拷贝，v3 仍保留 BK=DK。`301bfde0` 是**待上卡验证的修复候选**，不是已经成立的真修。只有 T3a 的 BK64 对照通过，才能讨论撤绕行；也不能据此自动宣布 scan2a/2b 所有路径正确。[A02]

## 5. A-F4：同步检查绿，但单文件前置件仍有漂移

本轮 `check_example_sync.py` 输出 `SYNC-OK kernel=776 example=776`。然而该脚本从 `gdn_v3_ab` 装饰行开始比较，**不覆盖内联的三个 v2 前置件**。

当前 `example_gdn_bwd.py:124,137-140` 的 solve_a 仍保留 `k_half`、fill、GM→UB→L1；而 `gdn_bwd_v2.py:254-262` 已是直接 GM→L1。这是源码事实，不是 NPU 推断。

影响：T3a 验证的 v2 版本与单文件示例不是同一份前置实现。即使 v3 段同步绿，也不能声称整个示例已同步。当前 BK=DK 可能掩盖差异；撤绕行时尤其需要把两个载体一起对账。本轮只报告，不修改 kernel 或同步工具。

## 6. A-F5：历史目标行通过，不是整批、全形状或当前版本验收

A03 批次 `20261023T2353Z`，B1/H16/T8192/DK128/DV128/C64 的日志：

| dq | dk | dv | dbeta | dg | dh0 |
|---|---|---|---|---|---|
| 5.897e-4 | 4.289e-3 | 1.043e-3 | 4.555e-3 | 4.718e-3 | 4.642e-4 |

相对 5e-3 的余量，dk 约 14.2%、dbeta 8.9%、dg 5.6%；不能把六项最小余量笼统写成 9%–14%。同一批 H16/T512/DK64 CONTROL 多项 FAIL，最终总行是 `SOME_FAIL`，两件事实同时保留。

A04 批次 `20261024T0033Z`：H1/T128 的 dk=`8.049e-3`，H1/T8192 的 dbeta=`6.116e-3`。T 扫描非单调只反驳简单的单调解释，不能排除累积误差、相消或归一化分母变化。

当前 `_realshape_acc.py` 用 seed 7，q/k 归一化，`g=-abs(randn)*0.5-0.05`，beta=sigmoid(randn)，h0=randn×0.1；g/beta 为 fp32，其余主体输入 fp16。该数值分布不同于计时脚本的 logsigmoid g，不能混成同一数值实验。[A03–A06]

## 7. A-F6：当前验收脚本还存在覆盖漏洞

- **目标形缺席**：当前 rsa 的 main 只列 H=1 的 T 扫描；直接发 rsa 不再重跑 H16/T8192 的目标形。HANDOFF 中 T5 的“目标形＋弱形”需恢复到实际用例表，不是现成命令已经覆盖。
- **NaN 可能被聚合吞掉**：`_realshape_acc.py:72` 使用 `max(previous, float(diff.max()))`，Python 的 `max(0.0, nan)` 返回 0.0；单个 head 的 NaN 可不进入聚合最大误差。它不是仅靠末尾 `rel < threshold` 就能防住的情况。需在聚合前独立检验有限性。
- **特殊值对照不完整**：`test_gdn_bwd_v3.py:252-261` 修改 dO 后仍使用修改前的 golden；boundary 又是 WARN-only。它不能证明当前 NaN/Inf 传播语义通过。
- **尾块拒绝≠尾块实现**：relay `:699` 拒绝 T%C≠0；L2 测试的成功是拒绝成功，不是原先 tail 功能已实现。fp32、DK≠DV 同样不能因入口分支存在就算验收。
- **分项诊断存在舍入差异**：dk_parts 按 dt 保存，而完整 dk 的 q3 以 fp32 加和后再 cast；`sum_vs_full` 非零不必然说明漏项，应与对应舍入预算比较。

上述脚本漏洞不证明历史每个数都错；它们说明**日志 PASS 不能独立证明未被检测的性质**。下一轮先修测量面，再取得新的验收证据。[A01、A03–A04]

## 8. A-F7：#1802 是局部反向，不能作完整六梯度替代

本地参考快照输入包含 h、dh、dv，输出为 dq/dk/dw/dg，没有 dbeta、dv 输出或 dh0。它把跨块状态及部分梯度作为输入，适用于局部 `chunk_o_bwd`，并没有在一个核里完成整个 GDN 伴随链。[A07]

快照还包含：host 的 dO scale、dv 取负、G 转置；dg 有分块维，涉及外部合并；入口拒绝尾块。其自带 main 主要验 shape 与部分 NaN，不能用 `Test Passed!` 证明全部精度。函数的 dtype 参数化与 bf16 示例也不能混说成所有接口硬编码 bf16-only。

因此：适配层否决与借鉴结构不天然矛盾；但用局部反向的“单核、零 workspace”给完整六梯度设上限，也缺少同工作量可比依据。`≤2 kernel` 若保留，应声明统计边界、是否含前置和 host 运算，再论证它服务于什么独立需求。没有这一步，不应选“按 #1802 整体照写”。

## 9. A-F8：性能数字有价值，因果与收益预测不足

`_realshape_time2.py` 对两条链前后同步、默认各跑 3 次、打印 min/median。单独运行时首轮仍可能含编译；取 min/median 不等于严格把 warmup 排除。bwd 包含非零 h0 和完整六梯度路径；fwd 不接 h0、且计时前把 beta 转 fp16。倍率是这两条脚本路径的比较，不是等价工作量或算力比。

历史约 16.3–17.5ms/chunk、目标 bwd 2.22s 与 fwd 打印 0.04s，支持“当前分解路径代价很高”。打印倍率用未舍入浮点算得，所以不能用 2.22/0.04 要求精确等于日志倍率；51–74× 只是按舍入构造的范围，不是统计误差棒。

按当前 relay 源码的调用表达式逐条计数（非 profiler）：

| 部分 | TL callable 调用次数 |
|---|---:|
| 三个前置件 | 每条序列链 3 |
| pass 1 | 每 chunk 10（dP 的 hi/lo 两次分别算） |
| pass 2，一般 chunk | 25 |
| pass 2，最后 chunk | 21（省 kt/tb 两次与 dkt 两次） |

NC≥1 且正常路径下，合计 `3+10NC+25(NC−1)+21=35NC−1`。这里不包含 factory 构造，也不把每次 TL callable 等同一个物理设备 launch；另有 ATen exp/cast/reduction/copy、内存分配、同步与 host 工作。

所以“全部是 launch”“重写可到 0.3ms/chunk”“唯一收益是 launch”都超出证据。融合可能改变数据复用、精度和并行度，需要 profiler、容量账和原型 A/B；本轮不作性能承诺。[A06]

## 10. A-F9：CPU 数学复验不是设备精度证明

本轮 host 公式检查 worst rel=`6.504e-15`；其默认用例为 B=H=1、T256、C=DK=DV=64、g=0、h0=0。两套 golden 在 H1/T512/DK=DV=64/C64 的交叉核对 worst=`3.135e-15`。它们说明所测数学组装一致，不能扩为所有形状与编译路径的证明。[A09]

host 脚本的 `quantize=True` 还把 delta_next 量化为 fp16，而当前 relay 保留 fp32 delta；因此其 `--fp16` 结果不是当前 NPU 实现的逐操作精度模拟，本轮不拿它定误差底限。

## 11. 教材保全与勘误清单

| 旧教材内容 | 本轮处理 |
|---|---|
| 六梯度、状态链、三把尺子、三段 CPU 实验、八条历史误判、自测和术语 | 保留主题并逐项检查定义和适用范围 |
| Δ' 方程省略转置、g_last 含义不清 | 补状态维度、Kt×Δ'.T、块内累计 g 与 exp |
| golden 默认 1、“分母借来” | 改为历史状态；当前默认 0；解释替换的是被测中间量 |
| toy 漏累加＝真实 bug 机理 | 降为覆盖演示，真实微观机制待证 |
| 无 CONTROL 信息量为零／控制臂证明探针完全正确 | 改为归因边界与对照本身的验证义务 |
| max_abs＝恒定底噪、ratio 永远不可达 | 撤回；提供同 max_abs、不同 ratio 反例及不等式方向 |
| NaN 所有比较 False | 改为顺序比较 False、!= True，并增加聚合吞 NaN 的问题 |
| 数字全是 NPU 实测、“唯一真 bug” | 分开 CPU/设备/推断，去掉排他性结论 |

历史 notes、HANDOFF、旧工程复盘不改写；这里是它们的后续勘误入口。旧结论仅保留为学习材料，不再作为下一轮验收依据。
