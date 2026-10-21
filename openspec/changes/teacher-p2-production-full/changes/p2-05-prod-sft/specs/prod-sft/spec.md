## ADDED Requirements

### Requirement: LoRA 读出位 SFT
`production/sft.py` SHALL 以载重 backbone + LoRA（r16/α32/dropout0.05/lr1e-4/warmup5%/cosine/2ep 基线档）在读出位做 CE（标注 hard + 教师伪标 soft 混合）；MPS fp16 + 梯度检查点可训。

#### Scenario: tiny SFT 冒烟
- **WHEN** 以 tiny 配置（≤500 样本、≤200 step）跑 SFT
- **THEN** run 产出 LoRA 适配器 + decision_config.json，loss 记录入 metrics.jsonl

#### Scenario: 读出位 CE 复用
- **WHEN** 检查 loss 计算
- **THEN** 复用一阶段 `decision/` 读出逻辑（无复制粘贴分叉；grep 无平行实现）

### Requirement: 教师伪标混合
SFT 数据 SHALL 支持 hard 标签与教师伪标 soft target 按配比混合（默认 50/50 可配）；伪标全走 teacher-adapters 缓存。

#### Scenario: 混合配比生效
- **WHEN** 以 30/70 配比构造 batch
- **THEN** loss 的 soft 分量权重与配置一致（单测以数值验证）

### Requirement: G8 规模点配置
`production/configs/` SHALL 提供 `gate08b.yaml`（gate 锚点）与 `scaling_06b.yaml`（第二规模点）；两档仅 backbone 与批规模不同，其余超参一致。

#### Scenario: 配置一致性
- **WHEN** diff 两配置
- **THEN** 差异仅限 backbone/批规模/步数字段
