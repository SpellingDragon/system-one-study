# run: 1005-p2-04-laya-cpu-fec2

- 假设：待填写
- 观察：待填写
- 结论：待填写

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。
结论：laya 于 test/smoke24 亲跑 24 题，acc=0.583333 ECE=0.079008 ms_p50=250.706（CPU 通路）。

> 边界与预估（p2-04）：本波 MPS 被一阶段占用，仅走 CPU 通路冒烟 smoke24（24 题真实出分）；全量 100 题（p1 pin typed-decisions test）CPU 预估约 33 秒，正式全量评测延后至云端 C6（NPU）执行。全量正式集规模以 p2-03 registry（bench/）为准，该域本波未就绪，故本波只读 p1 pin 数据跑通路。
> 与卡面差异：卡面 laya JevBench 130/230 系 CUDA 路径、异数据集；本波 Mac CPU + typed-decisions smoke24 得 acc≈0.58（14/24），两者数据集与设备均不同，不可直接比对，此处延迟非 CUDA 值。
