# run: 1006-p210-serving-local-half-c5d6

- 假设：p2-10 本地半场：/v1/systemone 用 decision 冻结契约驱动替身 backbone，固定模板（1 choice+1 noul+1 score）一次前向答完；CPU 档 warm-up20+N50 应能出 mean/P50/P95 三统计，endpoint 路与本地组批路 argmax 不换人，开 p2-07 前缀复用后同 state 连发三问命中增量 ≥2。
- 观察：
- - B1 延迟（device=cpu dtype=float32 backend=torch_eager，serial n=50 warmup-excluded=20）：e2e_ms mean=383.619 P50=382.662 P95=390.694 P99=392.154 min=377.534 max=392.62
- - B1 读数段单列 readout_ms（服务端自报，只包前向+读数）：mean=381.404 P50=380.481 P95=388.333
- - B1 吞吐：tokens_in=16750 tok/s(aggregate)=873.263 output_tokens=0
- - A2 双路一致：records=40 compared=40 argmax_all_same=True worst_prob_drift=2.05e-05 tolerance=0.005 passed=True
- - B2 前缀命中：requests=3 hits_delta=2 misses_delta=1 tokens_fed_delta=193 all_hits_suffix_only=True
- - 引擎：Qwen/Qwen3-0.6B loader=minimal head=[151936, 1024] render_version=dmlaya_render_v1

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。
结论：B1 三统计出数 P50=382.662ms/P95=390.694ms/mean=383.619ms；A2 双路 argmax 不换人且漂移在容差内；B2 连发 3 问命中增量 2（喂入符号 193）。CPU 档口径，NPU/MPS 画像延后（CPU / torch_eager 档（本波禁 MPS，NPU 后端画像待开卡注记））。
