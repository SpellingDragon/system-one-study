## ADDED Requirements

### Requirement: 中文评测决策化
`sys1/eval/` SHALL 将 CMMLU 子集与 CLUE（tnews/ocnli）决策化为 typed 评测（choice/noul），同 harness 同打分器；registry pin 其版本。

#### Scenario: CMMLU 子集跑分
- **WHEN** 对 DML 模型跑 CMMLU 子集（≥200 题）
- **THEN** 报告 choice acc 分桶呈现 + run-id

### Requirement: 中文数据配比消融
SFT 数据 SHALL 支持中文配比档（30% / 50%）消融，产出两 run 对照中文集增益。

#### Scenario: 配比对照
- **WHEN** 30% 与 50% 两档各跑 tiny SFT
- **THEN** 中文集 acc 对照入 runs，Δ 如实记录

### Requirement: laya 中文对照（G3）
中文对照表 SHALL 含 laya 列（同 harness 亲跑）；laya 中文预期显著劣化，MUST 如实记录为"崩溃域证据"而非省略。

#### Scenario: 对照表含崩溃列
- **WHEN** 生成 G3 对照表
- **THEN** laya 中文行实测值与 DML 并列，附 ms 与采样参数
