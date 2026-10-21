# Design: p1-10-eval-harness

## 技术要点
- 预测行格式（与上游打分器对齐）：`{"id", "answers": {qid: {opt: p}}, "ms"}`；键约定承 decision-program。
- 分桶呈现：qtype×k 分桶，每桶列实测 + 分桶随机基线 + Δpp——k=3 与 k=20 永不分列合并。
- parity 轴：同模型经 `backends.py`（tilelang_mps 或回退路径，报告标注模式）与纯 torch 前向，argmax 一致率必须 100%。
- speed 轴：单请求串行、warm-up 豁免、P50/P95 + tok/s；设备与精度模式必注。
- **run-id 溯源强制**：被测模型目录无 config 来源即拒评（`--allow-missing-run-id` 仅冒烟豁免）——"无 run-id 视为捏造"的机器执行。
- repro_p1.sh：s0→s3 tiny 档 + 四轴摘要表；干净 checkout <20min CPU；退出码即变更级 DoD 判定入口。
- 参考落点：`StartLux-Decision/eval/{suites,typed_decisions,latency}.py`（协议与指标形态）。

## 文件清单
| 文件 | 职责 |
|---|---|
| `dmlaya/eval/predict.py` | 双模式预测 |
| `dmlaya/eval/{quality,calibration,parity,speed}.py` | 四轴 |
| `examples/repro_p1.sh` | 一键复现 |
| `tests/test_eval.py` | 六场景 |

## 风险与回退
- [speed 轴受机器噪声干扰] → 固定 warm-up + N≥30 + 报 P50/P95 不报 mean 为主；设备必注。
- [endpoint 模式无服务可打（CI）] → 该场景单测 mock HTTP；真实服务联动归 p2-10。
