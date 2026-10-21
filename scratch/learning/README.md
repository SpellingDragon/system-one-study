# learning/ — 第一阶段脚本（from-scratch，禁载权重）

> 红线（GUIDE §7-0）：本目录任何脚本**不得下载/加载第三方预训练权重**（tokenizer 语料、评测数据除外）。所有运行落 `../runs/`。

## 脚本清单（对应 GUIDE §4.1）

| 脚本 | 阶段 | 做什么 | 教学主线落点（refs/llms-from-scratch-cn） |
|---|---|---|---|
| `s0_tokenizer.py` | S0 | 自训 BPE（中英+代码语料，8k–32k 词表，image/video/mask 特殊 token）；产出 tokenizer + **26 字母单 token 校验** | `Codes/ch02/01_main-chapter-code/ch02.ipynb`；校验思路 `refs/StartLux-Decision/.../jevfmt.py::check_tokenizer` |
| `s1_pretrain_gpt.py` | S1 | 随机初始化小 GPT（50–150M，因果注意力+RoPE+MLP）下一词预训练（0.5–5B tok）；采样冒烟验收 | `ch03→ch04/gpt.py→ch05/{gpt_train,gpt_generate}.py`；完整脚本 `ch05/03_bonus.../pretraining_simple.py` |
| `s1b_longctx.py`（选做 M2） | S2 | RoPE base 缩放 + 课程扩窗 + 因果滑窗；needle 召回随扩窗曲线 | `refs/Naive-N0.5-Flash`（SWA 布局）；`layers/README` |
| `s2_decision_sft.py` | S4' | 决策程序（`sys1/decision`）+ typed-decisions 类数据 CE 微调（soft target） | `refs/StartLux-Decision/finetune/finetune_lora.py`（loss 同构，只是不换权重） |
| `s3_calibrate.py` | S6' | 类型温度 NLL 网格拟合 + ECE before/after 报告 | 同仓 `finetune/calibrate.py` |

## 规约

- **注释即评分证据（CI 门）**：本目录全部 .py 必须通过 `python tools/check_comments.py`——文件头三件套【做什么/怎么做/为什么(含被否方案)】+ 公共函数 `白话：` 段落（不用技术名词）+ 密度 ≥20%，注释一律中文（规范见 GUIDE §6.1，答辩会随机抽查复述）。
- 每脚本头部注释：目的 / 输入 / 产出 run-id / 预计 GPU·h（PRODUCTION §8.1 口径）。
- 超参一律进 `configs/*.yaml`（不硬编码）；`configs/` 内给出 `scratch_small.yaml`（冒烟）与 `course_main.yaml`（正式）两档。
- 训练循环**必须**写 `metrics.jsonl`（`runs/README` 的 schema）——一阶段就养成二阶段论文证据链的习惯。
- 验收自查：`python -m sys1.eval.quality` 等能吃到本目录产物（模型目录结构一致：`config.json + weights + tokenizer + decision_config.json`）。
