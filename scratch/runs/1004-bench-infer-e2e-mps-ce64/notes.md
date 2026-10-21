# run: 1004-bench-infer-e2e-mps-ce64

- 假设：待填写
- 观察：待填写

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 dmlaya.runs 校验：没有结论行就不许 finish()。
结论：推理端到端（40M、fp16、同设备同权重）：TileLang 主干前向 0.88×（11,497 vs 13,017 tok/s）——好于单算子的 0.3–0.5×，因 norm/embedding/softmax 仍走 torch 摊薄了内核劣势，但仍无净优化。决策读出的两路皆亚毫秒级、非延迟瓶颈。三组数据合并结论：本 Mac 上 TileLang 尚无性能优势（训练 -8.7% / 推理 -12% / 单算子 0.3–0.5×）；价值在功能正确（对拍全绿）与能力通路（可微内核链已通）。翻盘路径与根因分析见 1004-bench-kernel-autocograd-30step-28ba 与 1004-bench-tilelang-vs-torch-mps-1a18。
