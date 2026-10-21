# Design: p1-08-pretrain-pipeline

## 技术要点
- 语料：fineweb-edu（英，`AI-ModelScope/fineweb-edu` 镜像）+ 中文教育语料（`opencsg/Fineweb-Edu-Chinese-V2.2`），流式取前 N GB；去重 = 逐 doc minhash-lite（或 URL 去重）规则显式入配置。
- token 流格式：memmap uint32（mmap 随机窗口采样，训练零拷贝）+ `index.json`（片长/配比/词表 id）。
- 训练循环：torch 原生（非 HF Trainer——教学透明优先）；梯度累积、cosine、warmup 入 yaml；NaN 检测（loss 非有限即中断并记 run notes）。
- 两档：`smoke_tiny.yaml`（~10M tok、d=128/L=2，CPU≤10min，CI `@slow`）与 `mps_main.yaml`（~40M 模型 × 1–3 亿 tok，MPS 数小时，正式冒烟）。
- 续训：checkpoint 记 step/optimizer/config hash；`--resume` 校验 config 一致后继续，metrics 无重复 step（p1-02 防重兜底）。
- 采样冒烟：温度 0.8 采 8 条 ≤40 token，原文 + 人工判定记 notes.md——"可读短句"是过程信号不是 gate，不达标如实记。
- 参考落点：`llms-from-scratch-cn ch05/{gpt_train,gpt_generate}.py`、`ch05/03_bonus.../pretraining_simple.py`。

## 文件清单
| 文件 | 职责 |
|---|---|
| `dmlaya/data/pretrain_corpus.py` | 流式装配 + 去重 + memmap |
| `learning/s1_pretrain_gpt.py` | 训练/续训/采样 CLI |
| `learning/configs/{smoke_tiny,mps_main}.yaml` | 两档配置 |
| `tests/{test_corpus,test_s1_train}.py` | 冒烟/续训/端到端 |

## 风险与回退
- [语料拉取网络不稳] → 镜像双源（ModelScope 优先 HF 兜底）+ 断点续拉；单测用内置小语料不依赖网络。
- [MPS 吞吐过低] → smoke_tiny 保 CI；正式档允许分段多 run（同 config 不同 seed 段拼接，notes 说明）。
