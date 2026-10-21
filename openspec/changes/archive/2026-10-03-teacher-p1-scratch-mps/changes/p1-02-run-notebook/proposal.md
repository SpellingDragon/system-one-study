# Proposal: p1-02-run-notebook — 实验记录（二级子变更）

> 父变更：teacher-p1-scratch-mps（W0，无前置依赖）

## Why
课程红线"无 run-id 视为捏造"的落地面：一阶段起就养成 runs/ 证据链习惯。所有训练/评测域（p1-08/09/10）都消费本域产出的 RunContext，故须 W0 先行。

## What Changes
- 新增 `dmlaya/runs/`（工具模块）：`new_run(name, config) -> RunContext`、`log_metrics`（防重）、`conclude/finish`（结论行强制）
- run 目录契约：`runs/<MMDD>-<exp-id>/{config.yaml, system.json, metrics.jsonl, notes.md}`
- 新增 `tests/test_runs.py`

## 边界与依赖（不耦合声明）
- **零依赖**（仅 stdlib + pyyaml）。
- **被依赖方**：p1-08/09/10 训练评测全走本域；二阶段全部 run 沿用同一契约。
- 接口面：`new_run / RunContext.{log_metrics, conclude, finish, run_id}`。

## 验收
- spec scenarios 全过：`specs/run-notebook/spec.md`（4 场景）
- 一级 DoD 关联项：① 一键串跑产 run-id
