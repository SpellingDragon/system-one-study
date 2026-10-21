# Design: p1-01-data-schema

## 技术要点
- 样本形态为 `/v1/systemone` 超集：`{state, questions:{qid:{type,instructions,criteria?}}, targets?:{qid:{opt:p}}, ep_group?, ep_step?}`——`ep_*` 两字段透传不校验语义（二阶段 RL 多轮用）。
- 键约定硬编码三则：noul=`false/true`；score=`"0".."n-1"`（低→高）；choice 保留 criteria 原键。
- targets 允许 soft 分布（多人投票份额照录）；`allow_unnormalized` 仅评测侧豁免用，训练侧一律归一。
- 错误消息必须含字段路径（如 `questions.q3.type`）——这是后续域调试的生命线。
- 参考落点：`StartLux-Decision/startlux_decision/jevfmt.py::validate`（思路，非移植）。

## 文件清单
| 文件 | 职责 |
|---|---|
| `dmlaya/data/schema.py` | 校验/归一化/序列化/键序 |
| `tests/test_schema.py` | 三拒两收场景 + 往返 |

## 风险与回退
- [score 的 k 与键集合不一致的歧义] → 以 criteria 键集合为准推导 k，缺失即拒；单测锁定。
- [jsonl 序列化丢类型（键必 str）] → dump 时断言键均 str，load 后重校验。
