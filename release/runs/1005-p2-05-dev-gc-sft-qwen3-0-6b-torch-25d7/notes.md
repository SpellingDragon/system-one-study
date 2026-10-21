# run: 1005-p2-05-dev-gc-sft-qwen3-0-6b-torch-25d7

- 假设：`--gradient-checkpointing` 不是空开关——按下后真 HF 底座的旗标应翻转，且 112 枚垫片仍全收到改动。
- 观察：8 步跑通，4.288 s/step、90.3 tok/s；离线复核 `enable_gradient_checkpointing()` 交回 True、
  `Qwen3Model.gradient_checkpointing` 旗标 False→True，开着它做单次 fwd+bwd 时 112/112 枚垫片都收到改动；
  CPU 上看不出稳定成本差（8 步窗口噪声与 100 步档 5.686 s/step 同量级），故既不声称"免费"也不声称"贵多少"。


> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。
结论：tiny SFT 跑通 8 步（qwen3-0.6b 替身 / cpu float32 / 旁路 torch@cpu）：loss 2.285952 → 0.740385（首尾各 1 步均值，降 67.61%），吞吐 90.3 tok/s（4.288 s/step，3096 字 / 34.3 s），可训 4587520 参数 / 112 枚垫片，伪标覆盖 100.0%（StartLuxAI/StartLux-Decision-4B#scaffold-cpu 缓存口，训练循环零教师前向）；CPU 替身口径（本地开发档）；910B 正式档（gate08b/scaling 全量）待 C5。
