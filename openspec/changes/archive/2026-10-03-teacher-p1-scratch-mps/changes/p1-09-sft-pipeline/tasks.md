# Tasks: p1-09-sft-pipeline [W2 · 依赖 p1-07+p1-06+p1-01+p1-02；与 p1-08 并发]

> 三级结构：`### 工作项` → 孙任务 checkbox。spec：`specs/sft-pipeline/spec.md`。

### 工作项 A 样本转写

- [x] A1 `dmlaya/data/transcribe.py`：XNLI-zh→noul（entailment→true 其余→false）、分类→choice、星级→score —— 验证：`python -m pytest tests/test_transcribe.py -q`
- [x] A2 转写规则锁定单测（neutral→{"false":1.0} 等） —— 验证：`python -m pytest tests/test_transcribe.py -k rules -q`

### 工作项 B SFT 循环

- [x] B1 读出位 CE（soft target 整体监督、其余位置屏蔽）+ 手工梯度单测验证屏蔽 —— 验证：`python -m pytest tests/test_s2_sft.py -k mask -q`
- [x] B2 `learning/s2_decision_sft.py` CLI：载 S1 checkpoint → SFT → 产出含 decision_config.json 的模型目录 —— 验证：`python learning/s2_decision_sft.py --help`
- [x] B3 smoke SFT 端到端：500 条转写集，留出集 choice/noul 准确率 > 分桶随机基线（**前置：p1-08 的 smoke_tiny checkpoint 已产出入 runs/；A/B/C/D 其余孙任务可与 p1-08 并发先行**） —— 验证：`python -m pytest tests/test_s2_sft.py -k e2e -q -m slow`

### 工作项 C 防泄漏

- [x] C1 `tools/check_leak.py`：训练样本 id ∩ typed-decisions test id = ∅ —— 验证：`python tools/check_leak.py --train <sft_data> --test bench/typed_decisions`

### 工作项 D 校准接线

- [x] D1 `learning/s3_calibrate.py`：温度表拟合写入 decision_config.json（ECE before/after 报告入 run） —— 验证：`python learning/s3_calibrate.py --help` + smoke 后 decision_config.json 含 temperatures 字段
