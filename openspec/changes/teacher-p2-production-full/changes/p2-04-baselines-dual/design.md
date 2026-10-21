# Design: p2-04-baselines-dual

## 技术要点
- laya 亲跑：`refs/laya/laya/serve.py` 起 `/v1/systemone`，P1 `predict.py --endpoint` 同路打分（DML/laya/StartLux 三者同 harness 的字面落实）。
- StartLux-0.8B：本地权重目录（工作区副本）以 P1 predict `--model` 模式直载；**不可得时降级**：该行 `source: card, gate: false` + 卡面数字仅参考列。
- 采样参数统一 T=1/top_p=1 落 run config；每行附 commit 与权重 revision。
- 对照表 DML 空缺格 `—` 显式（表完整性单测）——教师版先跑基线，DML 列由后续域逐步填。
- 耗时口径：ms 字段同协议（warm-up 豁免 + 串行）；设备公平原则——laya 速度仅其 CUDA 路径有效时标注（Mac 上跑 laya 则如实标注 Mac/CPU 路径，不冒充 CUDA 数字）。

## 文件清单
| 文件 | 职责 |
|---|---|
| `sys1/eval/baselines/run_baseline.py` | 双基线亲跑 CLI |
| `sys1/eval/baselines/table.py` | 对照表生成 |
| `tests/test_baselines.py` | 五场景 |

## 风险与回退
- [StartLux 权重不可得] → 降级路径显式标注（父 design 风险条既定），不阻塞其余行。
- [laya 服务协议与 /v1/systemone 有出入] → 适配层显式 diff 记录（键映射表入代码注释），打分器仍走统一上游口径。
- **版权边界（D11，2026-10-05）**：StartLux-0.8B 仅作对照评测推理（白名单③）——不训练、不作基座、不再发布其权重/衍生；产出对照表注明用途。
