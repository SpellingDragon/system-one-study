# Tasks: p1-02-run-notebook [W0 · 无依赖]

> 三级结构：`### 工作项` → 孙任务 checkbox。spec：`specs/run-notebook/spec.md`。

### 工作项 A RunContext 核心

- [x] A1 实现 `new_run(name, config)`：建 `runs/<MMDD>-<exp-id>/`，落 config.yaml（含 git commit）/system.json/metrics.jsonl/notes.md 骨架 —— 验证：`python -m pytest tests/test_runs.py -k new_run -q`
- [x] A2 实现 `log_metrics(step, **kv)`：JSON 行追加 + 同 step 同指标名拒绝（`MetricsConflictError`） —— 验证：`python -m pytest tests/test_runs.py -k conflict -q`
- [x] A3 实现 `conclude(text)` 与 `finish()`（无结论行抛 `MissingConclusionError`） —— 验证：`python -m pytest tests/test_runs.py -k conclude -q`

### 工作项 B run-id 规则

- [x] B1 run_id 生成（日期+slug+短哈希）与 notes 三行骨架（假设→观察→结论）固化，单测锁定格式 —— 验证：`python -m pytest tests/test_runs.py -k run_id -q`
