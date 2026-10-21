# serving/ — 部署与跨后端生产化（二阶段 G6）

> 交付一个可服务的 `/v1/systemone`：同 harness 数字必须是**从这个进程里出来**的，评测与线上行为同源。

## 计划文件

| 文件 | 职责 | 参考落点（refs/） |
|---|---|---|
| `server.py` | HTTP `/v1/systemone`（health + decide）；批量经 `decide_batch` 思路 | `laya/laya/serve.py`；`StartLux-Decision/startlux_decision/server.py` |
| `mcp_tools.py`（选做） | MCP 工具面 | `laya/laya/mcp/` |
| `fp8.py` | FP8 权重（torchao per-row 动态）；**三条实测坑必须处理**：短输入可能反慢（默认仅大 batch/长上下文启用）、padding 全零行→NaN（单问无 pad 跑）、精度掉点报 before/after 决策一致率 | `StartLux-Decision/docs/inference.md` §FP8 |
| `graphs.py` | CUDA graph 捕获：pad ladder + (rows×length) 双维；启动时 eager/graph 一致性自检（>0.02 弃图） | 同仓 `model.py::_capture/self_test` |
| `prefix_cache.py` | 前缀/状态复用：同 state 多问、同 state 跨请求复用（1M 下 state 重复渲染是成本大头） | 同仓 `mlx_model.py`（思路，非移植） |
| `card.py` | 模型卡生成：backbone/teacher 来源+commit、三栈超参、六轴指标、诚实边界 | PRODUCTION §7 |

## 验收（G6 + 设备公平）

- 三后端（CUDA/MPS/昇腾）起服务 → `sys1/eval/parity.py` 全绿：argmax 100% 一致、bf16 err≤2e-2。
- 延迟报告口径对齐 StartLux 协议：warm-up 20 + N 计时、单请求串行、报 mean/P50/P95，`eval/latency.py` 同款请求（1 choice+1 noul+1 score，三问一次前向）。
- CI：跨后端矩阵 + 评测回归（gate 指标跌破即红）。
