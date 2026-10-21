# Design: p1-09-sft-pipeline

## 技术要点
- **读出位 CE**：loss 仅在样本读出位计算（`gather` 读出位 logits → CE with soft target 分布），其余位置屏蔽——手工梯度单测验证屏蔽（对非读出位 token 加扰动，loss 梯度为零）。
- soft target：targets 概率分布整体为监督（CE = −Σ p_i log q_i），多人投票份额照录不硬化。
- 数据配方（防泄漏，父 design D8）：训练 = 转写集（XNLI-zh train→noul、MASSIVE-zh 子集→choice、星级→score）+ Intern-Decision suites train 划分；typed-decisions **test 仅评测**。来源/切分入 run config `data_revision`。
- s3 校准：留出集跑 p1-05 `fit_temperature` → 温度表写 `decision_config.json.temperatures`；ECE before/after + Δ 报告入 run。
- LoRA 不用（一阶段全参微调，模型仅 40M 级；LoRA 归二阶段 0.8B）。
- 参考落点：`StartLux-Decision/finetune/finetune_lora.py`（loss 同构）、`finetune/calibrate.py`（s3 范式）。

## 文件清单
| 文件 | 职责 |
|---|---|
| `dmlaya/data/transcribe.py` | 三类源 → typed 样本 |
| `learning/s2_decision_sft.py` | SFT CLI |
| `learning/s3_calibrate.py` | 校准 CLI |
| `tools/check_leak.py` | id 交集自检 |
| `tests/` 两件 | 规则/端到端 |

## 风险与回退
- [冒烟档模型太弱、准确率贴随机] → 缩小选项数（k≤5）保信号可见；如实记 notes（父 design 风险条已裁：不设绝对线）。
- [转写规则歧义（星级 k 不齐）] → 每 src 一条显式映射表入代码常量 + 单测锁定。
