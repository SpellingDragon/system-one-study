# Proposal: p1-08-pretrain-pipeline — 预训练管线（二级子变更）

> 父变更：teacher-p1-scratch-mps（W2；依赖 p1-03 + p1-06 + p1-02；与 p1-09 并发）

## Why
S1 从零预训练（Mac 冒烟档）：证明"能从随机权重学出语言信号"。教师版目标是流程正确 + 学习曲线可见，非模型质量（父 design D6 已裁）。

## What Changes
- 新增 `dmlaya/data/pretrain_corpus.py`：流式中英语料装配（7:3 可配）、去重、分片 token 流
- 新增 `learning/s1_pretrain_gpt.py`：随机初始化 → 下一词 CE → metrics.jsonl → checkpoint 分片 + `--resume` 续训 + 采样冒烟
- 新增 `learning/configs/{smoke_tiny,mps_main}.yaml` 两档
- 新增 `tests/test_corpus.py`、`tests/test_s1_train.py`

## 边界与依赖（不耦合声明）
- 依赖：p1-03（token 化）、p1-06（模型与 save/load）、p1-02（run 记录）。
- **被依赖方**：p1-09（SFT 载 checkpoint 起点）。
- 接口面：`pretrain_corpus.build(...) / s1_pretrain_gpt.py CLI + 产物模型目录`。
- 红线：禁载任何第三方权重（GUIDE §7-0）。

## 验收
- spec scenarios 全过：`specs/pretrain-pipeline/spec.md`（5 场景）
- 一级 DoD 关联项：① loss 稳降 + 可读短句、② SFT 超基线的前置
