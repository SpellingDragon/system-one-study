## ADDED Requirements

### Requirement: HTTP 服务
`serving/server.py` SHALL 提供 `/v1/systemone`（POST decide）与 `/health`：请求/响应体遵 decision-program 契约，支持单条与批量；评测 `--endpoint` 模式 MUST 能打此服务。

#### Scenario: 端到端服务评测
- **WHEN** 起服务后以 `sys1/eval/predict.py --endpoint` 跑 10 样本
- **THEN** 预测行与本地 `--model` 模式 argmax 一致（容差内概率同）

#### Scenario: 批量接口
- **WHEN** POST 含 5 个请求的 batch
- **THEN** 返回 5 组 answers，ms 字段齐全

### Requirement: 延迟测量协议
`serving/` SHALL 提供 StartLux 协议对齐的计时：warm-up 20 + N 串行计时，报 mean/P50/P95；同款请求（1 choice + 1 noul + 1 score 三问一次前向）。

#### Scenario: 延迟报告
- **WHEN** 运行延迟脚本（N=50）
- **THEN** 报告含 warm-up 豁免说明、三延迟统计与设备/精度标注

### Requirement: 服务内前缀缓存联动
server SHALL 接入 prefix_cache（长上下文域产出），同 state 多问请求自动复用；缓存行为可观测（命中计数端点或日志）。

#### Scenario: 命中可观测
- **WHEN** 同 state 连发 3 问后查询命中统计
- **THEN** 命中计数 ≥2
