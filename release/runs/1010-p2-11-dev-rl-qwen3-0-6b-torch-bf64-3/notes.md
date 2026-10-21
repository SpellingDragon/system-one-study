# run: 1010-p2-11-dev-rl-qwen3-0-6b-torch-bf64-3

- 假设：载起点 `未给起点（从 base 起步）`，用严格适当的 proper_reward（w_sph=0.5, w_rps=1.0）驱动 on-policy 采样→advantage 归一→LoRA 更新，8 题 / 上限 8 步应看到 reward 上行；温度只在 RL 收尾后拟合。
- 观察：跑了 6 步，reward -3.050061 → -4.407127；reward EMA=-3.6992802042007438；拟合温度=4.0；可训 4587520 参。
- 时序纪律：温度后置已按 spec 执行（events 仅在策略循环收尾后追加一次温度），无同 step 同时更新。

结论：RL 发散早停（稳定域=前 6 步）；本地半场 qwen3-0.6b/cpu float32/旁路 torch@cpu 跑通 6 步（负结果如实入库）。正式 RL（910B + C5 产物）待 C5。
