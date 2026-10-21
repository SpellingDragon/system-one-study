## ADDED Requirements

### Requirement: 决策样本转写
`dmlaya/data/transcribe.py` SHALL 把分类/NLI/星级源数据转写为 typed 样本：label→choice、entailment→true 其余→false（noul）、等级→score；键约定遵 data-schema；转写规则 MUST 单元测试锁定。

#### Scenario: XNLI 三分类转写
- **WHEN** 输入一条 neutral 的 XNLI-zh 样本
- **THEN** 产出 noul 样本，targets = {"false":1.0}，state/premise 对齐 schema

### Requirement: 读出位 SFT
`learning/s2_decision_sft.py` SHALL 以预训练 checkpoint 为起点，在决策样本上对**读出位**做 CE（soft target：目标分布整体为监督），其余位置 loss 屏蔽；超参全入 configs；loss/metrics 走 run-notebook。

#### Scenario: smoke SFT 端到端
- **WHEN** 以 smoke_tiny 预训练产物 + 转写集 500 条跑 `s2 --config smoke`
- **THEN** 退出码 0；SFT 后在留出集 choice/noul 准确率均 > 对应分桶随机基线；产物含 decision_config.json

#### Scenario: 只在读出位计损
- **WHEN** 检查训练一个 step 的 loss 组成
- **THEN** 非 readout 位置的 logits 不参与 CE（单测以手工梯度验证）

### Requirement: SFT 数据防泄漏
SFT 训练数据 MUST 不含 typed-decisions test 集；数据来源与切分写入 run config 的 `data_revision` 字段。

#### Scenario: 泄漏自检
- **WHEN** 运行 `tools/check_leak.py`（或等价测试）比对训练样本 id 与 test 集 id
- **THEN** 交集为空
