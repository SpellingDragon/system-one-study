# Design: p1-06-decoder-model

## 技术要点
- 骨架照 `llms-from-scratch-cn/Codes/ch04/gpt.py`（GPT 式）升级两点：绝对位置 → **RoPE**（长上下文同向，二阶段复用）；post-norm → **pre-norm**（深稳）。
- Mac 冒烟档规格（父 design D6）：~40M 参数（d=512, L=12, heads=8, ctx=1024）——写进 `configs/` 注释，模型本身规模无关。
- 窄接口铁律：`forward` 返回 last_hidden；`lm_head` 独立模块（weight 可被 decision/ 直接 index_select 字母行）——grep 测试锁定 decision/ 不触碰内部属性。
- save 布局与二阶段 backbone 目录对齐：`{config.json, weights.safetensors, tokenizer/, decision_config.json}`——p1-09/10 按"模型目录"消费，不感知是自训还是载重。
- seed 入 config：权重初始化可复现（无 run-id 不评测原则的模型侧镜像）。

## 文件清单
| 文件 | 职责 |
|---|---|
| `dmlaya/model.py` | ModelConfig + Decoder + save/load |
| `tests/test_model.py` | 五场景 |

## 风险与回退
- [RoPE 实现与 kernel 版 rope 数值不一致] → 与 p1-04 的 rope_ref 共用同一测试向量（fixtures 互引测试目录，非运行时耦合）。
- [MPS fp16 反向数值问题] → 默认 fp32 训练、fp16 仅前向冒烟；异常时 loss NaN 检测入训练循环（p1-08 兜底）。
