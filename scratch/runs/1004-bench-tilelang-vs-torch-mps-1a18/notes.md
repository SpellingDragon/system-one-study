# run: 1004-bench-tilelang-vs-torch-mps-1a18

- 假设：待填写
- 观察：待填写

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 dmlaya.runs 校验：没有结论行就不许 finish()。
结论：热态（预热排除编译）结论：tilelang 0.1.15 Metal 后端在 M3 Pro 上全面慢于 torch-MPS 2–6×。根因：① 16×16 固定 tile 未调优；② 本机 check_metal4_availability()=False，cooperative tensor 快路径不可用；③ torch 的 matmul 本已走 MPSGraph 硬件加速。另证：上轮 bench 之'恒 0.4–0.6s'系首次编译 2.4s 摊入均值的测量假象——热态基线必以预热为前提（教训入档）。推论：把内核直接注入模型前向当前为负优化；'TileLang 加速'的叙事成立条件在 CUDA 端与融合/调优之后。
