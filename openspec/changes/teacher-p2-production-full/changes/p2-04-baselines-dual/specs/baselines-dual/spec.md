## ADDED Requirements

### Requirement: laya 基线亲跑
`sys1/eval/baselines/` SHALL 在同一 harness、同一 split、同一采样参数（T=1, top_p=1）下亲跑 laya（起 `/v1/systemone` 或本地 API 等价路径），产出对照行；不接受转抄卡面数字。

#### Scenario: laya 对照行
- **WHEN** 跑 laya 于 typed-decisions test
- **THEN** 产出 {model: laya, acc/ECE/ms, 采样参数, commit} 行并入对照表

### Requirement: StartLux-0.8B 基线亲跑
同 harness 亲跑 StartLux-Decision-0.8B（本地权重，CC BY-NC 课程用途）；不可得时降级为"卡面参考列（非 gate）"并显式标注。

#### Scenario: StartLux 对照行
- **WHEN** 载 StartLux-0.8B 于同 harness
- **THEN** 对照行含与 DML 同口径的 acc/ECE/ms

#### Scenario: 降级路径标注
- **WHEN** 权重不可得
- **THEN** 对照表该行标 `source: card, gate: false`，不冒充实测

### Requirement: 对照表格式
对照表 SHALL 为 `{metric, DML, laya, startlux-0.8b, Δ, (jev ref)}` 且每行带 run-id；DML 未跑的格标 `—` 不留空。

#### Scenario: 表格完整性
- **WHEN** 生成对照表 markdown
- **THEN** 每行六个字段齐全，DML 空缺处为 `—`
