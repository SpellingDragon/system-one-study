## ADDED Requirements

### Requirement: run 目录三件套
`dmlaya/runs/`（工具模块）SHALL 提供 `new_run(name, config) -> RunContext`：创建 `runs/<MMDD>-<exp-id>/`，落盘 `config.yaml`（全量超参+git commit+数据 revision+硬件）、`system.json`（torch/tilelang/OS/芯片）、空 `metrics.jsonl` 与 `notes.md`（三行骨架：假设→观察→结论）。

#### Scenario: 创建 run
- **WHEN** 调用 `new_run("s1-pretrain", cfg)`
- **THEN** 目录含四个文件；config.yaml 中 `commit` 字段等于当前 `git rev-parse HEAD`；返回的 RunContext 暴露 `run_id`

### Requirement: metrics 追加与曲线真源
RunContext SHALL 提供 `log_metrics(step, **kv)` 追加 JSON 行（含 step 与 wall-clock）；MUST 拒绝重复写同一 step 的同一指标名（防曲线重写）。

#### Scenario: 定时记录与防重
- **WHEN** 先 `log_metrics(100, loss=3.2)` 再 `log_metrics(100, loss=3.1)`
- **THEN** 第二次抛出 `MetricsConflictError`，metrics.jsonl 中 step=100 仅一行

### Requirement: notes 结论行
RunContext SHALL 提供 `conclude(text)`：向 notes.md 追加以 `结论：` 起头的行；`finish()` 时 MUST 校验 notes.md 已有结论行，否则抛错（失败实验也要记结论）。

#### Scenario: 无结论收尾被拦
- **WHEN** run 从未调用 `conclude` 即调用 `finish()`
- **THEN** 抛出 `MissingConclusionError`
