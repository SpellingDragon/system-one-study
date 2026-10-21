# run: 1004-bench-kernel-autocograd-30step-28ba

- 假设：待填写
- 观察：待填写

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 dmlaya.runs 校验：没有结论行就不许 finish()。
结论：功能裁决：内核可微链训练 30 步真跑通——起点 loss 与 torch 逐位同（9.8506，同 seed 同权重之证），曲线重合（Δ≤0.008 fp16 预算内）；单算子梯度四路对拍全绿。性能裁决：端到端 -8.7%（3,048 vs 3,315 tok/s），未翻盘——归因：LN 主干走回退、attn 反向 torch 重算、backward 每步权重转置拷贝、gemm 热态本已 0.5×、gelu 留 torch。下一步（调优清单）：① LN/attn 的 Metal 反向件（先探针归约能力）；② 权重转置缓存；③ gemm tile 参数扫描；④ 算子融合减 launch。CUDA 端为 TileLang 优势主场，本 Mac 数据为下限样本。
