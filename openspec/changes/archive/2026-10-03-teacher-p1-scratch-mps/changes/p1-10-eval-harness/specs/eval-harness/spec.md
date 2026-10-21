## ADDED Requirements

### Requirement: 统一 predict 接口
`dmlaya/eval/predict.py` SHALL 提供 `--model <dir>`（本地）与 `--endpoint <url>`（`/v1/systemone`）两种模式输出统一预测行 `{"id", "answers": {qid: {opt: p}}, "ms"}`；键约定遵 decision-program。

#### Scenario: 本地预测行
- **WHEN** 对 SFT 产物跑 predict（10 样本）
- **THEN** 输出 jsonl 每行含 id/answers/ms，answers 内概率和=1

### Requirement: 四轴指标
`dmlaya/eval/` SHALL 输出一阶段四轴：quality（choice acc / noul acc / score MAE+within-1，按 qtype×k 分桶对照随机基线）、calibration（ECE before/after + Δ）、parity（kernel 路径 vs torch 路径 argmax 一致率）、speed（单请求延迟 P50/P95 与 tok/s，标注设备与精度）。

#### Scenario: 分桶质量报告
- **WHEN** 评分一个含 k=3 与 k=20 的 choice 预测集
- **THEN** 报告分两桶各列基线（0.333/0.05）与实测值及 Δpp

#### Scenario: parity 全绿
- **WHEN** 同一模型分别经 backends.py（tilelang_mps 或回退）与纯 torch 前向预测同一集
- **THEN** argmax 一致率 100%，报告含路径模式标注

### Requirement: run-id 溯源
评测报告 SHALL 自动附 run-id（被测模型目录的 config 来源 run）与 git commit；无溯源信息的报告 MUST 被拒绝输出（`--allow-missing-run-id` 显式豁免仅限冒烟）。

#### Scenario: 无溯源被拒
- **WHEN** 对无 config 来源的裸权重目录跑评测
- **THEN** 评测拒绝并提示补 run 信息

### Requirement: 一键复现
`examples/repro_p1.sh` SHALL 串跑 s0→s3（tiny 档）+ M3 四轴报告；退出码为变更级 DoD 唯一判定（参考时长 <20 分钟 CPU，慢机器不因时长判负，仅作回归监控信号）。

#### Scenario: 一键复现
- **WHEN** 在干净 checkout 上 `bash examples/repro_p1.sh`
- **THEN** 退出码 0，runs/ 新增链式 run-id，终端打印四轴摘要表
