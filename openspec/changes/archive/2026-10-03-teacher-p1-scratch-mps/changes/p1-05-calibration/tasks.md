# Tasks: p1-05-calibration [W0 · 无依赖]

> 三级结构：`### 工作项` → 孙任务 checkbox。spec：`specs/calibration/spec.md`。

### 工作项 A 温度与 ECE

- [x] A1 `fit_temperature(logits, targets, grid=[0.2..5.0])`：逐 qtype NLL 网格 + <200 样本池化（report 标 pooled） —— 验证：`python -m pytest tests/test_calibrate.py -k fit -q`
- [x] A2 `ECE(top-label, bins=15, 支持概率质量加权)` —— 验证：`python -m pytest tests/test_calibrate.py -k ece -q`
- [x] A3 单测：温度不换 argmax（含 tie-break）/ 小样本池化 / 完美校准 ECE<1e-9 —— 验证：`python -m pytest tests/test_calibrate.py -q`

### 工作项 B 分桶基线

- [x] B1 `bucket_baselines()`：choice 1/k、noul 0.5、score MAE (k²−1)/(3k)，按 qtype×k 输出；单测 k=20 与 k=3 分列 —— 验证：`python -m pytest tests/test_calibrate.py -k bucket -q`
