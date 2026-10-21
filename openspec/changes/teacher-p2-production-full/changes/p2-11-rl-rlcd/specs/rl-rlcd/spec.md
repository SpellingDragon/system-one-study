## ADDED Requirements

### Requirement: 适当得分奖励
`production/rl_rlcd.py` SHALL 实现 `proper_reward`：log score + spherical +（score 类 RPS），按 qtype 组合；奖励对真值分布严格适当（最优策略=真分布）。参考蓝本 `refs/laya/laya/common.py::proper_reward`。

#### Scenario: 真分布最优
- **WHEN** 以合成 logits（真分布 vs 偏移分布）计算奖励
- **THEN** 真分布奖励严格更高（单测多 qtype）

### Requirement: 策略优化循环
RL 循环 SHALL：采样决策（复用 OPD 采样器）→ 计算奖励 → advantage 归一 → LoRA 梯度更新；温度在 RL 结束后重拟合（不与奖励同时优化）。

#### Scenario: tiny RL 冒烟
- **WHEN** tiny 档跑 ≤100 step
- **THEN** run 记录 reward 曲线，退出码 0；发散时以负结果入 notes 并标注稳定域

#### Scenario: 温度后置
- **WHEN** 检查训练流程
- **THEN** 温度拟合发生在 RL 收尾（时序断言：无同 step 同时更新温度与策略）

### Requirement: 三栈串链编排
`production/train.py` SHALL 以 parent-run-id 链编排 SFT→OPD→RL，每段产物可独立加载续跑。

#### Scenario: 链式溯源
- **WHEN** 查 RL run 的 config
- **THEN** 含 parent_run_id 指向 OPD run，逐级可溯至 SFT/载重
