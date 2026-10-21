# Proposal: p2-11-rl-rlcd — RL 栈（二级子变更）

> 父变更：teacher-p2-production-full（W3；依赖 p2-06 的 on-policy 采样器与缓存设施）

## Why
三栈之末（对齐任务真值+校准）：proper_reward（严格适当得分规则）RL。教师版口径 = 实现完整 + tiny run + 发散则负结果入库（RL 不稳是真实工程坑，踩坑记录即交付物）。另含三栈串链编排（train.py）。

## What Changes
- 新增 `production/rl_rlcd.py`：proper_reward（log+spherical+RPS 按 qtype）+ advantage 归一 + 温度后置
- 新增 `production/train.py`：SFT→OPD→RL 编排（parent-run-id 链）
- 新增 `tests/{test_rl,test_train_chain}.py`

## 边界与依赖（不耦合声明）
- 依赖：p2-06（采样器）、p2-05（OPD 后模型或 SFT 模型）。
- **被依赖方**：p2-12 报告（G7 消融表）。
- 接口面：`proper_reward / rl_rlcd.py CLI / train.py 编排`。

## 验收
- spec scenarios 全过：`specs/rl-rlcd/spec.md`（5 场景）
- 一级 DoD 关联项：① RL run（reward 曲线或负结果）
