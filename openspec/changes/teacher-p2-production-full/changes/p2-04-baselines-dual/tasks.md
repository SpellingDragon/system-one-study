# Tasks: p2-04-baselines-dual [W0 · 依赖 p2-03；与模型开发解耦，最先动工]

> 三级结构：`### 工作项` → 孙任务 checkbox。spec：`specs/baselines-dual/spec.md`。

### 工作项 A laya 亲跑

- [x] A1 laya 服务起 + endpoint 打通：起 `/v1/systemone`，P1 predict `--endpoint` 冒烟 10 样本（服务起停与协议适配可先行；**打分前置：p2-03 的 A2 已将 typed-decisions fetch 入 bench/**） —— 验证：预测 jsonl 10 行且键约定合规
- [x] A2 laya 全量跑 typed-decisions test → 对照行（acc/ECE/ms + 采样参数 + commit） —— 验证：`sys1/eval/baselines/` 含 laya 预测 jsonl 与 run-id

### 工作项 B StartLux 亲跑

- [x] B1 StartLux-0.8B 本地权重载入亲跑 → 对照行；不可得则降级 `source: card, gate: false` —— 验证：对照行存在且 source 字段明确

### 工作项 C 对照表与入库

- [x] C1 对照表生成器：`{metric, DML, laya, startlux-0.8b, Δ, jev-ref}` 六字段 + run-id + 空缺 `—` —— 验证：`python -m pytest tests/test_baselines.py -q`
- [x] C2 基线 runs 收尾（notes 结论行：与卡面差异说明） —— 验证：两 run 目录 notes.md 均含 `结论：`
