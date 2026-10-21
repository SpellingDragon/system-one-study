# run: 1005-p2-05-dev-rec-sft-qwen3-0-6b-kernel-4276

- 假设：账本按调用顺序落行，那么装配事实（asm_encoded/teacher_hits/soft_coverage）若在训练循环结束后
  才以 `step 0` 补记，就会带着终局墙钟排在第 100 步之后——读曲线的人无法判断该信哪条时间线，这属记录缺陷。
- 观察：改成循环前记 `step 0`、循环后把自研件终局清点挂到最后一步（`kernel_compiled`），5 行顺序即事实顺序：
  第 1 行 `{"step": 0, ...asm_encoded: 480, teacher_hits: 480, teacher_misses: 0, soft_coverage: 1.0}`（23:20:43，
  早于 step1 的 23:21:05），第 5 行 `{"step": 3, "kernel_compiled": 8}`；config 的 `kernels.compiled_at_start: []`
  与终局 8 枚分家，各自时点都诚实。3 步损失 2.285952/3.499027/0.642264 与 47a3/50f1 同种子逐点相同，
  说明这次动的是记账位置、没动数值。本 run 只验记录口径，曲线凭据仍以 7389（torch 参照）与 50f1（kernel 主路线）为准。

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。
结论：tiny SFT 跑通 3 步（qwen3-0.6b 替身 / cpu float32 / 旁路 kernel@cpu）：loss 2.285952 → 0.642264（首尾各 1 步均值，降 71.9%），吞吐 19.8 tok/s（18.972 s/step，1128 字 / 56.92 s），可训 4587520 参数 / 112 枚垫片，伪标覆盖 100.0%（StartLuxAI/StartLux-Decision-4B#scaffold-cpu 缓存口，训练循环零教师前向）；CPU 替身口径（本地开发档）；910B 正式档（gate08b/scaling 全量）待 C5。
