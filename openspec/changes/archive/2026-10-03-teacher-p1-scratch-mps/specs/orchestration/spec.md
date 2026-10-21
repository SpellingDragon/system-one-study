## ADDED Requirements

### Requirement: 二级子变更结构
一级变更 SHALL 以 `changes/p1-NN-<domain>/` 承载全部二级子变更，每个子变更 MUST 自包含 `{proposal.md, design.md, tasks.md, specs/<capability>/spec.md}` 四件套；一级 `specs/` 仅保留本编排契约（orchestration）。

#### Scenario: 子变更自包含
- **WHEN** 检查任一 `changes/p1-*/` 目录
- **THEN** 四件套齐备，且其 spec 能力名与一级 design D2 表一一对应

### Requirement: 一级任务只编排不实现
一级 tasks.md SHALL 仅含：编排跟踪图（mermaid，随进度更新）、子变更级 checkbox（完成 = 其孙任务全过 + spec 全过 + 按 D4 合并）、一级直管收尾任务；MUST NOT 含实现级孙任务（孙任务一律下沉子变更）。

#### Scenario: 编排与实现分离
- **WHEN** 读一级 tasks.md
- **THEN** 无 `- [ ]` 项涉及具体代码文件编写；每子变更项附 `changes/<name>/` 路径

### Requirement: 波次合并前置
子变更合并 SHALL 满足波次前置（W1 须其 W0 依赖已入 main，依此类推）；同波次子变更 MAY 任意次序合并；跨波次抢跑合并 MUST 被拒。

#### Scenario: 抢跑被拒
- **WHEN** p1-06（W1）在 p1-03（W0）未入 main 时提合并
- **THEN** review 依据本契约拒绝，直至 p1-03 合并

### Requirement: 变更级 DoD 收尾
一级收尾 SHALL 核验 design D5 六条（一键串跑/超基线/ECE 改善/对拍一致/CI 全绿/注释门），结果记入最终 run notes 后方可 `openspec archive`。

#### Scenario: DoD 未过不归档
- **WHEN** F2 未完成（六条有未过）
- **THEN** F4（归档）不得执行；归档即视为变更完成
