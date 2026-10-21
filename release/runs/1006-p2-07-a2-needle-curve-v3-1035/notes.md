# run: 1006-p2-07-a2-needle-curve-v3-1035

- 假设：待填写
- 观察：待填写

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。
结论：A2 召回曲线（零样本基座码位口径，device=mps）：8192→0.5 / 32768→0.25 / 131072→0.0；峰值内存见逐行 metrics（上限档如实），256K/1M 归云端 C 波。
更正（编排者补录）：结论行 device 措辞有误——本曲线实际以 **CPU fp32** 跑成（MPS 在 ~118K 崩于 conv1d，见 config.device 与 v3 崩点取证）；metrics 七行与档位数字不受影响。
