# Proposal: p2-04-baselines-dual — 双基线亲跑（二级子变更）

> 父变更：teacher-p2-production-full（W0；依赖 p2-03 registry 的数据 pin；与模型开发完全解耦——最先动工产出参照系）

## Why
G1 的 gate 基座：laya 与 StartLux-0.8B 必须由教师**同 harness 亲跑**（红线：不接受转抄卡面数字）。基线表一就绪，DML 三栈每一步增益都有参照。

## What Changes
- 新增 `sys1/eval/baselines/`：laya 亲跑（`/v1/systemone` 服务路径）与 StartLux-0.8B 亲跑（本地权重）
- 新增对照表生成器：`{metric, DML, laya, startlux-0.8b, Δ, jev-ref}` 六字段 + run-id + 空缺 `—`
- 新增 `tests/test_baselines.py`

## 边界与依赖（不耦合声明）
- 依赖：p2-03（registry 数据与打分器）；P1 predict 的 endpoint 模式。**权重来源已实证**：StartLux-0.8B = 魔搭 `StartLuxAI/StartLux-Decision-0.8B`（30 文件可达；**config 实测 `model_type=qwen3_5`+同构 vision_config——基线与学生同基座**，G1 因此成为纯训练配方受控实验，D10；其 `decision_config.letter_token_ids` 自 32 起，可作 p2-01 字母接缝参照）
- **被依赖方**：p2-05/06/11 的增益对照、p2-09 中文对照、p2-12 报告主表。
- 接口面：`baselines/run_baseline.py CLI + baselines/table.md 产物`。

## 验收
- spec scenarios 全过：`specs/baselines-dual/spec.md`（5 场景）
- 一级 DoD 关联项：③ 双基线对照表落 baselines/
