# Design: p1-05-calibration

## 技术要点
- 温度不改 argmax（标量除 logits 的单调性）——写进单测而非注释愿望。
- NLL 网格法（非梯度法）：网格 [0.2, 5.0] 步长 0.05，教师版选网格的理由 = 可复现、无优化器超参、与 StartLux `finetune/calibrate.py` 同构可对照；clamp 思想承 `refs/laya/laya/common.py::clamp_temperature`。
- 池化规则：qtype 样本 <200 → 并入全局温度，`FitReport.pooled=true` 标注（报告不掩盖）。
- ECE top-label 口径 15 bins（typed-decisions 上游口径）；soft target 时按概率质量加权——两口径禁止混报（错误消息提醒）。
- 分桶随机基线是"防假通过"的制度化：k=3 的 0.333 与 k=20 的 0.05 分列。
- 参考落点：`StartLux-Decision/finetune/calibrate.py`（网格/ECE/池化）、`refs/laya/laya/calibrate.py::fit_temperature_map`（另一流派对照）。

## 文件清单
| 文件 | 职责 |
|---|---|
| `dmlaya/calibrate.py` | 温度/ECE/基线三函数 |
| `tests/test_calibrate.py` | 五场景 |

## 风险与回退
- [tie-break 下温度改变 argmax] → 数值上不可能（严格单调），单测以含并列 logits 的输入锁定。
- [ECE 加权与不加权口径混用] → 函数签名强制 `weighted: bool`，报告同时输出两值时显式分列。
