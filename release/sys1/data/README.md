# sys1/data/ — 样本 schema、转写与装配

> 两轨共用。数据格式即 `/v1/systemone` 的超集：`{state, questions:{qid:{type,instructions,criteria}}, targets?:{qid:{opt:p}}, ep_group?, ep_step?}`。

## 计划文件

| 文件 | 职责 | 参考落点（refs/） |
|---|---|---|
| `schema.py` | 样本校验（类型/键/分布归一）+ 序列化 | `StartLux-Decision/startlux_decision/jevfmt.py::validate`；`finetune/finetune_lora.py` 的数据读取 |
| `transcribe.py` | 分类/NLI/星级数据集 → choice/noul/score 转写（规则见 GUIDE §5：label→choice；entailment→true、其余→false；等级→score 低到高） | MASSIVE/XNLI/CLUE 等原仓库 |
| `pretrain_corpus.py` | S1 下一词预训练流（中英混排、去重、采样配比） | `refs/llms-from-scratch-cn/Codes/ch05/03_bonus_pretraining_on_gutenberg/prepare_dataset.py` |
| `longctx.py` | 合成 needle（多针、长度可配、seed 固定）；"状态截断回归"集 | PRODUCTION §5.4 |
| `mm.py` | 图文装配：image/video token 插入 state 段（二阶段，接口先留） | GLM-5.3-Flash `vision_config`（ModelScope `ZhipuAI/GLM-5.3-Flash`） |

## 规则

- **切分与 pin 归 eval/registry 管**，data/ 只管生产样本流；任何训练/评测数据必须先过 `schema.py`。
- targets 允许 soft 分布（多人投票份额直接照录，不硬化）——SFT 的 CE 与 RLCD 都吃软目标。
- 一阶段红线：语料可公开（含 benchmark 的 train 划分），**权重不可**。
