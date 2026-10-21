# run: 1009-p2-13-p11b-dw-rope-attnsw-6223

- 假设：待填写
- 观察：待填写

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。
结论：P1-1 编译面七件全绿（gemm双形态/add_ln/readout/GDN-conv/dW/rope/attn_sw）。上卡收数包：V1/V2/V2'/V3 单位证真 + 五件 npz 对拍 + tflops。dW P0 判据含转置方向判别器（relFro=1.41 可区分）。
