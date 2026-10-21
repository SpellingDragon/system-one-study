# sys1/ — 两轨共享实现层

> **契约**：一阶段（learning/）在此产出，二阶段（production/）**原样复用**，不得分叉。改接口必须同步改测试（工程红线）。各子包职责如下，细节进各子目录 README。

| 子包 | 职责 | 诞生于 | 被谁复用 |
|---|---|---|---|
| `decision/` | ★ 两阶段同构桥梁：systemone 渲染、字母读出、类型温度、>26 票选 | 一阶段 M0-S2 | 二阶段全部训练栈 + serving |
| `model.py` | 小型因果 decoder（GPT 式）封装；二阶段仅"换脑" | 一阶段 M0-S1 | 二阶段载 backbone 后同接口 |
| `layers/` | 注意力/rope/indexpool/mhc（机制层） | 一阶段 M1/M2 | 二阶段 G4 |
| `kernels/` | TileLang 三件套 + `backends.py` 分发 | 一阶段 M1 | 二阶段 G6 生产化 |
| `data/` | schema、转写、长文/图文装配 | 一阶段 M0 | 二阶段全部数据集 |
| `eval/` | ★ 唯一评测 harness + pin registry + 双基线 | 一阶段 M3 | 二阶段 G1–G8 判据 |
| `calibrate.py` | 温度拟合（NLL 网格）、ECE | 一阶段 S3 | 二阶段 P3 后重拟合 |
| `router.py` | 语言/任务/模态路由 | 二阶段 P4 | serving |
| `lang/` `vision/` | tokenizer 封装 / 视觉塔（占位，见各手册指引） | S0 / P4 | 两轨 |
| `testing/` | torch_ref 对拍 + bench | 一阶段 M1 | 持续 |

**上手顺序建议**：`decision/README` → `kernels/README` → `eval/README`。
**文件级参考地图**：GUIDE §2.1（一阶段）与 PRODUCTION §3.1（二阶段），路径基于 `refs/`。
