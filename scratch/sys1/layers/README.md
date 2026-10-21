# sys1/layers/ — 注意力与机制层

> 与 `kernels/`（怎么算）分层：这里管**机制怎么接**（掩码、窗口、稀疏选点、长上下文策略）。

## 计划文件

| 文件 | 职责 | 阶段 | 参考落点（refs/） |
|---|---|---|---|
| `attention.py` | 因果滑窗注意力（local，成本与 L 无关）+ 全注意力层混合的**层排布策略**（如 5:1） | 一阶段 M2 / 二阶段 G4 | `Naive-N0.5-Flash/modeling_naive_n05_flash.py`（SWA 128 + hybrid_layer_pattern）、GLM-5.3-Flash `layer_types`（linear+sparse 3:1） |
| `indexer.py` | 稀疏全局：Lightning-indexer 打分→Top-K 选 key（服务读出位的长程检索） | 二阶段 G4 | 同 Naive 文件 `NaiveN05FlashIndexer`（ReLU 打分+逐头权重和+fp8；注意其 top-k≤工作规模，一阶段可先只做滑窗） |
| `indexpool.py` | indexer key 的分组池化压缩（1M 下 indexer 自身的显存/延迟） | 二阶段 | GLM-5.3-Flash `index_kpool*` 配置 |
| `rope.py` | RoPE：base 设置与**缩放课程**（short→long 扩窗时调 theta） | 一阶段 S1/M2 | `llms-from-scratch-cn/Codes/ch04/gpt.py`（无 rope，用 sin/cos）→ 升级点自行实现 |
| `mhc.py` | （选做/加分）流形约束超连接：Sinkhorn 宽残差 | 二阶段加分 | `refs/TileKernels/tile_kernels/mhc/sinkhorn_*.py`（含双后端样板） |

## 原则

- 双向 vs 因果：**全程因果**（架构裁决见 GUIDE §0.4）；滑窗用因果半窗，indexer 只在因果前缀内选点。
- 机制层保持 kernel 无关的接口（`forward(hidden, mask_spec)`），kernel 由 `kernels/backends.py` 注入——这样 G8 多规模曲线不改代码。
