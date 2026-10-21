# ascend-runtime · spec

## Purpose
在 910B+CANN 8.5.2（snt9b 镜像）上原地移植 TileLang 昇腾算子（头桩→AIC 优先七件），探索脱离 CUDA 生态；三栈以 torch_npu 底座运行、算子成熟一个换一个（渐进替换），验收 G9；成本受控（D12）。

## Requirements
- 环境脚本可复现（版本 pin + dry-run 自检）；探针结论（方言可用性/数值/吞吐）必须实测入 run notes，禁止 mock 推断。
- 七类算子接口与 `sys1/kernels/` 同名同签名；训练侧切换零改动。
- 梯度对拍 fp32 互锁：cpu target（本地）与 NPU（云端）双跑全绿；不过不合入。
- 回退路径在册：torch_npu 底座（同超参），决策原因如实记录。

## Scenarios
#### Scenario: 本地 CPU 对拍
- **WHEN** 本地跑 `test_ascend_gradcheck.py -k cpu` **THEN** 七算子前向/反向对 fp32 参考全绿 exit 0。**例外（2026-10-05 回写）**：GDN 反向允许 partial（昇腾方言无先例，design 风险条授权同栈混合临时 torch 实现）——其余六件必全绿，partial 须入阻塞账与 notes 原因行。
#### Scenario: 910B 探针
- **WHEN** C1 短租跑 selfcheck **THEN** run notes 出方言可用性/数值行为/吞吐初值三行结论
#### Scenario: NPU 梯度对拍
- **WHEN** C2 跑全量 gradcheck（NPU）**THEN** exit 0 且与 cpu 对拍结果一致（容差内）
#### Scenario: 基准与护栏
- **WHEN** C2/C4 结束 **THEN** 基准表（自研 vs torch_npu）与 cost ledger 入 run notes；C4 报价超 ¥600 触发上报
#### Scenario: 回退决策
- **WHEN** 探针失败 **THEN** torch_npu 底座路径可用（SFT tiny 冒烟 exit 0）且决策+原因入 notes
