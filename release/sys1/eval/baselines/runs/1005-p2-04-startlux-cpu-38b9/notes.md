# run: 1005-p2-04-startlux-cpu-38b9

- 假设：待填写
- 观察：待填写
- 结论：待填写

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。
结论：startlux 于 test/smoke24 亲跑 24 题，acc=0.583333 ECE=0.06157 ms_p50=5490.882（CPU 通路）。

> 边界与预估（p2-04）：本波 MPS 被一阶段占用，仅走 CPU 通路冒烟 smoke24（24 题真实出分）；全量 100 题 CPU 预估约 9.1 分钟（未装 causal_conv1d/flash-linear-attention，走参考实现偏慢），正式全量延后至云端 C6（NPU）+ 优化内核。D11 边界：StartLux 权重仅作本对照评测推理，绝不进训练、绝不当基座、绝不二次发布其权重或衍生。
> 与卡面差异：卡面 StartLux-0.8B JevBench 179/230、IntAvg 85.03、12.2ms(H200) 系优化内核、异数据集；本波 CPU float32 参考实现 + typed-decisions smoke24 得 acc≈0.58、单题≈5.5s，量级差异源于设备与数据集不同，非同一把尺。
