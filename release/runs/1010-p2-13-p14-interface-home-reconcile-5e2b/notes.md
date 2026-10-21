# run: 1010-p2-13-p14-interface-home-reconcile-5e2b

- 假设：把 attempts/ 已证 910B 形态合流进接口 home，即令 9 件 target=ascend 编得出（P2 训练步前置）。
- 观察：并发波 H/I/J/K 后编排者容器独立复验，7 接口 target=ascend 编译 PASS（gdn 出 .aibin）；gdn 反向 kernelized、conv 入口新增、rope forward 补 ascend 路由；test 20 passed。dW/lora 阻 §12 缺 asc_fill_l1——末次检索定其后端=set_l1_2d(__cbuf__*,int64 config)（SET_L1_2D 写 2D L1 区，官方 Matmul Fill 即其封装），唯 config 位打包未考。
- 结论：P1-4 七件合流达成本波可编面（全本地零卡），接口 home 从 SimtVF 迁至 910B 可编形态；dW/lora 余 P1-4b=set_l1_2d config 编码（有界 §12 活，非硬件死路），其数值本属 V1/V2/V3 卡待域，可与下窗并。

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。
