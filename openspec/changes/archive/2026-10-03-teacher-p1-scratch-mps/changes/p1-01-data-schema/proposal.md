# Proposal: p1-01-data-schema — 样本契约（二级子变更）

> 父变更：teacher-p1-scratch-mps（W0，无前置依赖）

## Why
两轨（learning/production）全部训练与评测数据的唯一合法入口：没有统一 schema，SFT/评测/防泄漏都无从校验。教师版须先立此契约，后续各域只管生产样本流。

## What Changes
- 新增 `dmlaya/data/schema.py`：qtype 枚举 {choice,noul,score}、`SchemaError`（含字段路径）、targets 归一化（容差 1e-6）、jsonl 行式 dump/load、score `to_rank_vector()`
- 新增 `tests/test_schema.py` 全场景单测

## 边界与依赖（不耦合声明）
- **零依赖**：不 import 本仓任何其他子域。
- **被依赖方**：p1-07 decision-program（from_systemone 校验）、p1-08/09 管线（样本入口）、p1-10 eval（预测行键约定）。
- 接口面：`validate_sample / dump / load / to_rank_vector / SchemaError`。

## 验收
- spec scenarios 全过：`specs/data-schema/spec.md`（5 场景）
- 一级 DoD 关联项：⑤ CI 全绿（`bash tools/ci.sh`）
