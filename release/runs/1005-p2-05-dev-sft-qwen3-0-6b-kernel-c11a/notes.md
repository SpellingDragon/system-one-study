# run: 1005-p2-05-dev-sft-qwen3-0-6b-kernel-c11a

- 假设：kernel 档（p2-13 自研件 cpu 路径）能在 0.6B 替身上真跑 LoRA 前向+反向，且每步耗时可接受。
- 观察：4 步真跑，loss 6.93/1.38/2.36/2.00（单桌 2 行，噪声大），step_seconds 48.4/25.1/33.5/31.9，
  tokens_per_second 9.1~15.0；TileLang 只编了 8 枚（gemm_impl/dw_impl 各 4），说明耗时是内核执行而非重复编译。
- 结论：kernel@cpu 通路真跑通（含 TileLang 编译、LoRA 注入、只读教师缓存 100% 命中、metrics 落盘），
  但本 run 在 step 4 主动中止——汇总口径里 `loss_decrease_pct` 用首末单步值、而展示的首末损失是窗口均值，
  两者不同源会写出对不上的数字；先修口径再重跑，不做第二次带病记录。CPU 替身口径（本地开发档）；
  910B 正式档（gate08b/scaling 全量）待 C5，本 run 的 s/step 不外推。

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。
