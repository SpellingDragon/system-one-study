# run: 1005-p2-05-dev-sft-qwen3-0-6b-kernel-50f1

- 假设：kernel 档（TileLang cpu 件，p2-13 交付的 `lora_kernel.apply/backward`）在真 0.6B 上跑满 100 步，
  曲线应与 torch 参照档（7389）逐点吻合，而每步耗时明显更高。
- 观察：100 步跑完，前 10 步损失均值 1.958338 → 后 10 步 1.260573（降 35.63%），与 7389 的 1.958339→1.260573
  同源；100 个点逐点损失差 max 2.6e-5、均值 2e-6 —— 两条旁路在真权重上等价到 float32 舍入量级；
  TileLang 实编 8 枚件（gemm 4 + gemm_dw 4，终局清点见 47a3 run 的 `kernel_compiled: 8`）；
  s/step 中位 28.10、均值 60.82 —— 均值被 step 81/82/88 三段拉长（step 88 记到 1247.53 s，本机在 22:05~22:41 被挂起），
  故结论行里的 6.9 tok/s 是被挂起稀释后的口径，剔除三段离群后为 28.23 s/step ≈ 15 tok/s
  （同机 torch 参照档 5.62 s/step ≈ 73 tok/s）；可训 4,587,520 / 冻 596,049,920 / 112 枚垫片 / 伪标覆盖 100.0%。


> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。
结论：tiny SFT 跑通 100 步（qwen3-0.6b 替身 / cpu float32 / 旁路 kernel@cpu）：loss 1.958338 → 1.260573（首尾各 10 步均值，降 35.63%），吞吐 6.9 tok/s（60.816 s/step，41748 字 / 6081.62 s），可训 4587520 参数 / 112 枚垫片，伪标覆盖 100.0%（StartLuxAI/StartLux-Decision-4B#scaffold-cpu 缓存口，训练循环零教师前向）；CPU 替身口径（本地开发档）；910B 正式档（gate08b/scaling 全量）待 C5。
