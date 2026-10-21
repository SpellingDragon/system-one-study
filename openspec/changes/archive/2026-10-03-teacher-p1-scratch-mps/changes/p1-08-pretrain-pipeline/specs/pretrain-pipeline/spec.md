## ADDED Requirements

### Requirement: 中英混排语料装配
`dmlaya/data/pretrain_corpus.py` SHALL 流式拉取并装配预训练语料（fineweb-edu 英 + 中文教育语料，配比可配，默认 7:3），产出分片 token 流（memmap 或 parquet）；去重与采样规则 MUST 显式写入配置。

#### Scenario: 冒烟语料产出
- **WHEN** 以 `--stream-limit 200MB --out runs/corpus_tiny` 装配
- **THEN** 产出分片 token 数 ≥ 配置下限，token id 全部 < vocab_size，中英 token 均非零占比

### Requirement: 下一词预训练循环
`learning/s1_pretrain_gpt.py` SHALL：随机初始化 decoder → 下一词 CE 预训练 → 定时写 metrics.jsonl（loss/lr/tok_per_s）→ checkpoint 分片保存与续训；两档配置 `smoke_tiny.yaml`（CPU≤10min）与 `mps_main.yaml`（MPS 冒烟档）。

#### Scenario: smoke_tiny 端到端
- **WHEN** `python learning/s1_pretrain_gpt.py --config smoke_tiny`
- **THEN** 退出码 0；runs/<id>/metrics.jsonl 至少 5 行且首末 loss 下降；产出 checkpoint 可被 load

#### Scenario: 断点续训
- **WHEN** 中断后以 `--resume runs/<id>` 重启
- **THEN** 从断点 step 继续，metrics 无重复 step

### Requirement: 采样冒烟验收
S1 完成后 SHALL 执行温度采样生成短句（≤40 token），样本存入 run 目录 notes.md 旁；教师版验收 = 至少出现一条人类可读句（中或英），如实记录不达标情形。

#### Scenario: 可读性抽检
- **WHEN** 对 smoke 档产物采样 8 条
- **THEN** notes.md 记录全部 8 条原文与人工判定结论
