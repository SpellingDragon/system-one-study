# run: 1005-p2-05-dev-sft-qwen3-0-6b-torch-5118

- 假设：同一份 tiny 配置换成 `torch` 参照档旁路，损失应能往下走（读点通了才可能降），且每步耗时明显低于 kernel 档。
- 观察：29 步真跑（中止于 step 29）；前 5 步损失均值 2.7429 → 后 5 步 1.2262（降 55%），grad_norm 59.8 → 2.51，
  均值 6.32 s/step、68.5 tok/s（对照同机 kernel 档 c11a run 的 25~48 s/step、9~15 tok/s），
  soft_coverage 1.0（装配出的题全部命中袖珍教师缓存，训练循环零教师前向）。
- 结论：torch 参照档在 0.6B 替身 + lr1e-4 的百步档上确有下降趋势，通路成立；本 run 主动中止——
  `limit: 512` 越过 spec 冒烟档写死的「≤500 样本」上限，改 `tiny_cpu.yaml` 为 480 后重跑，不留越界记录。
  CPU 替身口径（本地开发档）；910B 正式档（gate08b/scaling 全量）待 C5；本 run 的 s/step 不外推。

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。
