# Tasks: p1-01-data-schema [W0 · 无依赖]

> 三级结构：`### 工作项` → 孙任务 checkbox（自带验证，exit 0 即完成）。
> 合并规则与验收见父变更 design.md D4/D5；本域 spec：`specs/data-schema/spec.md`。

### 工作项 A 校验核心

- [x] A1 实现 `dmlaya/data/schema.py`：qtype 枚举 {choice,noul,score}、`SchemaError`（消息含字段路径）、类型/键/qtype 校验 —— 验证：`python -m pytest tests/test_schema.py -k "path or enum" -q`
- [x] A2 实现 targets 归一化（概率和容差 1e-6，`allow_unnormalized` 开关） —— 验证：`python -m pytest tests/test_schema.py -k normalize -q`
- [x] A3 单测覆盖三场景：合法样本通过 / 概率和越界被拒（消息含 qid 与"概率和"）/ 非法 qtype 被拒（消息含合法枚举） —— 验证：`python -m pytest tests/test_schema.py -q`

### 工作项 B 序列化与键序

- [x] B1 实现 `dump/load`（jsonl 行式）与 `to_rank_vector()`（score → 数值向量） —— 验证：`python -m pytest tests/test_schema.py -k "roundtrip or rank" -q`
- [x] B2 单测：score 键 `["0","1","2"]` 通过、键集合与 k 不符被拒 —— 验证：`python -m pytest tests/test_schema.py -k score_key -q`
