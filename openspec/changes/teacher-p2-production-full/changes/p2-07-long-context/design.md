# Design: p2-07-long-context

## 技术要点
- prefix_cache 两级：L1 进程内 KV cache（state 前缀 hash → past_key_values）；L2 同 state 多问批（问题段增量前向）。命中判定 = state 渲染前缀 token 完全一致（RENDER_VERSION 入 hash）。
- 增量前向断言：token 计数器对比复用/不复用两路径——第 2 问起前向 token 数 = 问题段长度（state 段零重算）。
- 架构策略（D9 修正）：**长文由 gate 同源 Qwen3.5 原生承载（0.8B=262K，Gated DeltaNet 3:1 混合血统）**——Qwen3-Next 仅作架构知识参照不载入（保 G1 与 StartLux 可比）；1M 走 9B 级云端配置档；本域只做工程（prefix_cache）与评测；MiniMax 警例仍有效（不自研）
- 滑窗路径：`layers/attention.py` 的窗口/层排布配置（参考 Naive hybrid 5:1 与 GLM 3:1 形态）；kernel 侧复用 P1 `attn_sw`（因果半窗）；短序列下滑窗≈全注意力（argmax 一致性冒烟）。
- needle 生成：多针（颜色-数字对）插入长干扰文；档位 8K/32K/128K/256K；seed 入 registry；指标 = 针召回率（精确串匹配）。
- **1M 双列口径**：报告 `measured`（Mac 上限如实，目标 ≥128K）与 `config ready`（云端 yaml + 运行说明）分列——防虚报的制度化。
- 参考落点：`Naive-N0.5-Flash/modeling_naive_n05_flash.py`（SWA/hybrid）、`StartLux-Decision/.../mlx_model.py`（前缀复用思路）、`tilelang/examples/flash_attention`。

## 文件清单
| 文件 | 职责 |
|---|---|
| `serving/prefix_cache.py` | 两级前缀复用 |
| `sys1/eval/longctx.py` | needle 生成 + 召回报告 |
| `sys1/layers/attention.py` | 滑窗/排布配置 |
| `production/configs/longctx_1m_cloud.yaml` | 1M 云端档 |
| `tests/test_longctx.py` | 六场景 |

## 风险与回退
- [Mac 256K OOM] → 档位自动降级 + 报告记实测上限；1M 只验管线正确性（合成小子集）不验吞吐。
- [KV cache 与 fp16 数值漂移累积] → 复用/不复用 argmax 一致性为硬断言；漂移超限则该 state 弃缓存重算。
