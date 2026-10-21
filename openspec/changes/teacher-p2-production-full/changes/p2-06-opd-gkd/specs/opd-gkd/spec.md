## ADDED Requirements

### Requirement: on-policy 采样器
`production/opd.py` SHALL 提供学生自采样：学生模型对决策状态输出 top-k 选项分布（k 可配，默认 8）作为 on-policy 样本源；采样器 SHALL 可被 RL 栈复用。

#### Scenario: top-k 采样
- **WHEN** 对 20 选项状态采样
- **THEN** 返回 ≤8 个 (opt, p) 且概率和=1

### Requirement: 稠密蒸馏损失
OPD SHALL 对学生自采样状态取教师同选项归一化分布，最小化 reverse-KL（可切 JSD）；教师分布 MUST 走离线缓存；混合 ground-truth 比例可配。

#### Scenario: reverse-KL 方向
- **WHEN** 以数值单测验证损失方向
- **THEN** 优化目标是 KL(p_teacher ‖ p_student)（mode-covering），非反向

#### Scenario: 缓存命中训练
- **WHEN** 以已缓存分布跑一个 epoch
- **THEN** 教师前向计数为 0

### Requirement: 增益消融
`production/opd.py --ablation` SHALL 产出 on/off 两 run（同 seed 同数据），报告 Δ（typed-decisions acc 与 ECE）；增益为零或负也如实入库。

#### Scenario: 消融对产出
- **WHEN** 运行消融档（tiny）
- **THEN** 两个 run-id 入 runs/，notes.md 含"OPD 增益 Δ=…"结论行
