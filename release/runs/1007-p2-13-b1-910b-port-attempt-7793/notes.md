# run: 1007-p2-13-b1-910b-port-attempt-7793

- 假设：待填写
- 观察：待填写

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。
结论：B1 综合判决：910B 跑 Qwen3.5 有**两道独立墙**——①torch_npu 层：GDN conv 无执行路径（Conv2D ge 崩，CUDA-only 快库不可用）；②tilelang 层：模板深依赖 950 编译器类型/stdlib（属性语言反而全兼容，移植面收窄于类型代理与 SIMT 面，估数天离线工程）。结论：**TileLang GDN 自研件从'优化项'升格为'Qwen3.5 上 910B 的生存前提'**；战略分叉（换非 GDN backbone 求速 vs 坚守移植求道）已上报用户裁决。计时 21:39–21:59 ≈ 20min ¥6.7（含前窗合计 ≈ ¥15.7）。
