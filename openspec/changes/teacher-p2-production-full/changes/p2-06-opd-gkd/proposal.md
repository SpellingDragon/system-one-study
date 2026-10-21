# Proposal: p2-06-opd-gkd — OPD 栈（二级子变更）

> 父变更：teacher-p2-production-full（W2；依赖 p2-05 SFT 产物 + p2-02 教师；与 07/08/09/10 并发）

## Why
三栈之二（稠密蒸馏）：学生自采样决策状态，教师对同选项给稠密分布，最小化 reverse-KL/JSD。手册已声明教学假设（判别式下 OPD 可能≈soft-KD）——**消融 run 本身是交付物**（增益小或负也如实入库，教师踩坑价值所在）。

## What Changes
- 新增 `production/opd.py`：on-policy 采样器（top-k，RL 可复用）+ reverse-KL/JSD 损失 + `--ablation` 消融档
- 新增 `tests/test_opd.py`

## 边界与依赖（不耦合声明）
- 依赖：p2-05（起点模型）、p2-02（教师分布缓存）。
- **被依赖方**：p2-11 RL（采样器复用）。
- 接口面：`sample_topk(model, state, k) / opd.py CLI（train/ablation 两档）`。

## 验收
- spec scenarios 全过：`specs/opd-gkd/spec.md`（5 场景）
- 一级 DoD 关联项：① OPD 消融 run（含 Δ 结论行）
