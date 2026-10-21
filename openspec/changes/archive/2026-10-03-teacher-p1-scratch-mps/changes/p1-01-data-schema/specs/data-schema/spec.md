## ADDED Requirements

### Requirement: 样本校验与序列化
`dmlaya/data/schema.py` SHALL 提供 `validate_sample()` 与 `dump/load`：校验 `{state: str, questions: {qid: {type, instructions, criteria?}}, targets?: {qid: {opt: p}}}` 的类型/键/qtype 合法性（type ∈ {choice, noul, score}），违规 MUST 抛出含字段路径的 `SchemaError`。

#### Scenario: 合法样本通过
- **WHEN** 校验一个含 choice 问题与 soft targets（概率和=1，容差 1e-6）的样本
- **THEN** `validate_sample` 返回归一化后的样本，`dump`→`load` 往返后逐字段相等

#### Scenario: targets 概率和越界被拒
- **WHEN** 某qid 的 targets 概率和为 0.9 且未开启 `allow_unnormalized`
- **THEN** 抛出 `SchemaError`，消息含该 qid 路径与"概率和"字样

#### Scenario: 非法 qtype 被拒
- **WHEN** question type 为 `"ranking"`（不在三枚举内）
- **THEN** 抛出 `SchemaError`，消息含合法枚举列表

### Requirement: score 键序约定
score 类 targets 的选项键 SHALL 为 `"0".."n-1"` 字符串（低→高）；noul 为 `"false"/"true"`；choice 保留 criteria 原键。

#### Scenario: score 键序校验
- **WHEN** score 样本 targets 键为 `["0","1","2"]`
- **THEN** 校验通过且 `to_rank_vector()` 返回长度 3 的数值向量

#### Scenario: score 键乱序被拒
- **WHEN** score 样本 targets 键为 `["1","0"]` 而问题声明 k=3
- **THEN** 抛出 `SchemaError` 指出键集合与 k 不符
