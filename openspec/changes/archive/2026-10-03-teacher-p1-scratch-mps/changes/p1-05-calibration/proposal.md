# Proposal: p1-05-calibration — 校准数学（二级子变更）

> 父变更：teacher-p1-scratch-mps（W0，无前置依赖）

## Why
S3 的数学内核：类型温度拟合、ECE、分桶随机基线。纯函数域，零依赖可 W0 并行；p1-09（S3 接线）与 p1-10（calibration 轴）都消费本域。

## What Changes
- 新增 `dmlaya/calibrate.py`：`fit_temperature`（NLL 网格 [0.2,5.0]，逐 qtype，<200 池化）、`ECE`（top-label 15 bins，概率质量加权）、`bucket_baselines`（choice 1/k、noul 0.5、score MAE (k²−1)/(3k)）
- 新增 `tests/test_calibrate.py`

## 边界与依赖（不耦合声明）
- **零依赖**（torch/numpy 级纯数学）。
- **被依赖方**：p1-07 temperature.py（温度表）、p1-09 s3 接线、p1-10 calibration 轴。
- 接口面：`fit_temperature / ECE / bucket_baselines / FitReport`。

## 验收
- spec scenarios 全过：`specs/calibration/spec.md`（5 场景）
- 一级 DoD 关联项：③ ECE 校准后改善（before/after 记录）
