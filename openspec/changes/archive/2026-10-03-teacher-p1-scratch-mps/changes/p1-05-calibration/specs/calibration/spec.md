## ADDED Requirements

### Requirement: 类型温度拟合
`dmlaya/calibrate.py` SHALL 提供 `fit_temperature(logits, targets, grid)`：在 NLL 网格（默认 [0.2, 5.0]）上为每 qtype 拟合标量温度；样本 <200 时 MUST 池化到全局温度。温度 MUST 不改变 argmax（单调变换）。

#### Scenario: 拟合不换 argmax
- **WHEN** 对任意 logits 施加拟合出的温度后 softmax
- **THEN** 每个样本的 argmax 与无温度时相同

#### Scenario: 小样本池化
- **WHEN** 某 qtype 有效样本 120 个
- **THEN** 该 qtype 使用池化温度，`fit_report` 标注 `pooled=true`

### Requirement: ECE 计算
`ECE(probs, targets, bins=15)` SHALL 按 top-label 口径计算期望校准误差；MUST 支持加权（soft target 时按概率质量加权）。

#### Scenario: 完美校准 ECE 为 0
- **WHEN** 输入频率与置信完全一致的合成数据
- **THEN** ECE < 1e-9

#### Scenario: 校准前后改善
- **WHEN** 对 SFT 产出的 logits 先后计算原始/温度后 ECE
- **THEN** 报告记录两者与 Δ，且教师版验收要求 Δ ≤ 0（改善或持平，如实记录）

### Requirement: 分桶随机基线
`bucket_baselines()` SHALL 提供 choice=`1/k`、noul=`0.5`、score 均匀 MAE=`(k²−1)/(3k)`，按 qtype×k 分桶输出。

#### Scenario: 二十选一基线
- **WHEN** 查询 k=20 的 choice 桶
- **THEN** 基线准确率 = 0.05，与三选一桶（0.333…）分列不混报
