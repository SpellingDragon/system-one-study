# Design: p2-05-prod-sft

## 技术要点
- LoRA 基线超参**照抄** `StartLux-Decision/finetune/finetune_lora.py`（r16/α32/dropout0.05/lr1e-4/5% warmup/cosine/2ep/16384tok×4 accum）——同构可比（**仅超参参考，D11：StartLux 权重不进训练**）；批规模按 910B 显存折算并记录折算比。
- **执行环境（D6/D13 v3）**：训练 on 云端 910B（全 TileLang 自研栈，底座=p2-13 C1 探针裁定；torch_npu 回退在册）；本地只做代码开发+CPU 冒烟；正式 run 须待轨 A（P1）完成且 C4 报价获批。
- 读出位 CE：复用 P1 `decision/readout` 取 logits，loss = soft-CE（hard 标签 one-hot 与伪标分布按配比加权）；非读出位屏蔽同 P1 手工梯度验证。
- 伪标全走 p2-02 缓存（训练循环零教师前向）；`--prefill-pseudo` 子命令预热缓存（教师慢推理前置离线做）。
- 产物布局：`{adapter.safetensors, config.json, decision_config.json, README(来源链)}`——与 P1 模型目录形态对齐，下游 predict 不感知差异。
- tiny 冒烟档：≤500 样本/≤200 step（本地 CPU 逻辑冒烟 + 云端 NPU tiny 实跑两层）；gate08b 正式档 run 记录吞吐与显存与费用（D12）。

## 文件清单
| 文件 | 职责 |
|---|---|
| `production/sft.py` | SFT CLI（含 --prefill-pseudo） |
| `production/configs/{gate08b,scaling_06b}.yaml` | 两规模点 |
| `tests/test_prod_sft.py` | 五场景 |

## 风险与回退
- [自研栈移植受阻（p2-13 探针/对拍失败）] → torch_npu+transformers+peft 底座同超参照跑（D6 回退，决策入 notes）；[gate 报价超 ¥600] → 熔断上报降档。
- [0.6B 与 0.8B 配置漂移] → 单测 diff 白名单（仅 backbone/批规模/步数）。
