# run: 1005-probe-08b-lora-mps-feasibility-5a64

- 假设：待填写
- 观察：待填写

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 dmlaya.runs 校验：没有结论行就不许 finish()。
结论：结论：0.6B 级 LoRA 在 M3 Pro/MPS 可训——必开梯度检查点（GC 一举双得：内存 36.6→16.2GB、且因免换页吞吐反升 226→470 tok/s）；gate 档等效 micro-batch 8×1024 在 GC 下可跑（475 tok/s，31GB 逼近上限）。预算：tiny 冒烟 ~9min；gate SFT 全量按 ~475 tok/s 外推 ≈ 20–30h 级（一夜到一天，非交互可等，宜分段 run+250 步存档同 P1 规矩）。边界：seq 更长（长上下文 needle ≥128K）本机内存不可行，与 P2 D7 的 Mac 实测档/云端配置双列口径一致；fp16 训练溢出风险需 loss-有限性哨兵（P1 s1 已有）。
