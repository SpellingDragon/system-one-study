# Proposal: p1-09-sft-pipeline — SFT 管线（二级子变更）

> 父变更：teacher-p1-scratch-mps（W2；依赖 p1-07 + p1-06 + p1-01 + p1-02；与 p1-08 并发）

## Why
S2+S3 落地：把预训练"小脑"决策化——读出位 CE（soft target）微调 + 类型温度校准接线。产出物（模型目录 + decision_config.json）是 p1-10 评测的直接对象。

## What Changes
- 新增 `dmlaya/data/transcribe.py`：XNLI-zh→noul、分类→choice、星级→score 转写
- 新增 `learning/s2_decision_sft.py`（载 S1 checkpoint → 读出位 CE → 产出模型目录）
- 新增 `learning/s3_calibrate.py`（温度表写入 decision_config.json + ECE before/after）
- 新增 `tools/check_leak.py` 防泄漏自检
- 新增 `tests/{test_transcribe,test_s2_sft}.py`

## 边界与依赖（不耦合声明）
- 依赖：p1-07（渲染/读出）、p1-06（模型）、p1-01（样本校验）、p1-02（run）、p1-05（校准数学，s3 用）。
- **被依赖方**：p1-10（评测产物接口）。
- 接口面：`transcribe.* / s2/s3 CLI + 模型目录（含 decision_config.json）`。

## 验收
- spec scenarios 全过：`specs/sft-pipeline/spec.md`（5 场景）
- 一级 DoD 关联项：② SFT 后超分桶随机基线、③ ECE 改善
