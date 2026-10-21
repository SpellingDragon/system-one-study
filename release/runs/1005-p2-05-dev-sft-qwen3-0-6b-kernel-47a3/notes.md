# run: 1005-p2-05-dev-sft-qwen3-0-6b-kernel-47a3

- 假设：修完记录口径（`kernels.compiled_at_start` 与开局/终局分家、`stage` 随 run_prefix 走、
  产物 README 的来源链补 D11 红线）之后，三步短跑应能把新字段真写进记录本。
- 观察：3 步跑通，`config.yaml` 的 stage=`p2-05-prod-sft/p2-05-dev`、`compiled_at_start: []`（开局还没下笔，
  空是事实）；`metrics.jsonl` 的 step 0 行新增 `kernel_compiled: 8`（终局清点），与 50f1 那 100 步编的件数一致；
  README 来源链出现"过目 480 条，丢弃 无"与"教师权重一律不进训练"字样；21.48 s/step、17.5 tok/s。


> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。
结论：tiny SFT 跑通 3 步（qwen3-0.6b 替身 / cpu float32 / 旁路 kernel@cpu）：loss 2.285952 → 0.642264（首尾各 1 步均值，降 71.9%），吞吐 17.5 tok/s（21.476 s/step，1128 字 / 64.43 s），可训 4587520 参数 / 112 枚垫片，伪标覆盖 100.0%（StartLuxAI/StartLux-Decision-4B#scaffold-cpu 缓存口，训练循环零教师前向）；CPU 替身口径（本地开发档）；910B 正式档（gate08b/scaling 全量）待 C5。
