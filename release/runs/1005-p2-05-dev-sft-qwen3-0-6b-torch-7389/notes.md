# run: 1005-p2-05-dev-sft-qwen3-0-6b-torch-7389

- 假设：读点打通 + LoRA 只挂 language 侧 qkvo 之后，480 题 / 100 步的 tiny 档应能看到损失往下走；
  同种子同批的 kernel 档（50f1）与之逐点损失应吻合到 float32 求和顺序的量级。
- 观察：100 步跑完；前 10 步损失均值 1.958339 → 后 10 步 1.260573（降 35.63%），grad_norm 59.8 → 2.51；
  5.686 s/step、73.4 tok/s（CPU 4 线程 float32、0.6B 替身，一桌 512 字 / accum 1，235 桌）；
  可训 4,587,520 / 冻 596,049,920（112 枚垫片 = 28 层 × qkvo）；伪标覆盖 100.0%（480/480 命中袖珍教师
  缓存，训练循环一次教师前向都没发）；产物 adapter.safetensors + adapter_config.json + config.json +
  decision_config.json + README 五件齐。


> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。
结论：tiny SFT 跑通 100 步（qwen3-0.6b 替身 / cpu float32 / 旁路 torch@cpu）：loss 1.958339 → 1.260573（首尾各 10 步均值，降 35.63%），吞吐 73.4 tok/s（5.686 s/step，41748 字 / 568.58 s），可训 4587520 参数 / 112 枚垫片，伪标覆盖 100.0%（StartLuxAI/StartLux-Decision-4B#scaffold-cpu 缓存口，训练循环零教师前向）；CPU 替身口径（本地开发档）；910B 正式档（gate08b/scaling 全量）待 C5。
