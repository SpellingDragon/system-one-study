# Design: p2-10-serving-systemone

## 技术要点
- 框架：标准库 `http.server` 或 fastapi（MPS 单进程足够；不引入 vLLM 类重型——父 Non-Goals）。请求体 = `/v1/systemone` 原生格式，内部直调 P1 `decision/`（渲染→读出→温度）——**服务只是 decision/ 的网络皮**。
- 批量：`decide_batch` 承 P1 `decide_batch` 思路（跨请求问题合批，pad ladder 对齐）。
- 延迟协议照抄 StartLux：warm-up 20 豁免 + N=50 串行计时；请求模板固定（1 choice+1 noul+1 score 三问一次前向）——与 laya/StartLux 数字可比的前提。
- prefix_cache 挂点：请求解析后查 state 前缀 hash；`/stats` 暴露命中计数（10.4 验证用）。07 未就绪时空实现 + 接口不变。
- 并发模型：单 worker（MPS 串行前向），请求排队——延迟口径因此稳定。

## 文件清单
| 文件 | 职责 |
|---|---|
| `serving/server.py` | HTTP 服务 + 批量 + /stats |
| `serving/latency.py` | 延迟协议脚本 |
| `tests/test_serving.py` | 五场景 |

## 风险与回退
- [http 框架选择反复] → 先标准库 http.server 起步（零依赖）；需要异步再换 fastapi（接口测试不变）。
- [批量 pad 与延迟口径互相干扰] → 延迟测量固定单请求模式；批量另测吞吐，两口径分列。
- **NPU 后端（v3）**：推理优先昇腾（用户令 D6）；后端枚举 cpu/mps/**npu** 三档，npu 走 p2-13 算子或 torch_npu 推理；延迟画像含 npu 档。
