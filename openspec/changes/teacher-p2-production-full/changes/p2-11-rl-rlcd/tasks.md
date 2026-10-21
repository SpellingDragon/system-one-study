> **甲路挂起（2026-10-07 用户裁定）**：本域 C5 云端项全部冻结，解锁=p2-13 P2 行（三栈联调）达成；本地已完成项照旧。

# Tasks: p2-11-rl-rlcd [W3 · 依赖 p2-06 采样器]

> 三级结构：`### 工作项` → 孙任务 checkbox。spec：`specs/rl-rlcd/spec.md`。

### 工作项 A 奖励函数

- [ ] A1 `proper_reward`（log+spherical+RPS 按 qtype）+ 严格适当单测（真分布奖励严格更高） —— 验证：`python -m pytest tests/test_rl.py -k reward -q`

### 工作项 B 训练循环

- [ ] B1a 采样-奖励步：批量采样（复用 p2-06 采样器）→ proper_reward 计分（fp32 域） —— 验证：`python -m pytest tests/test_rl.py -k loop_sample -q`
- [ ] B1b 更新步：advantage 归一 + LoRA 梯度更新 + reward EMA/发散早停 —— 验证：`python -m pytest tests/test_rl.py -k loop_update -q`
- [ ] B1c 温度后置时序断言（无同 step 同时更新温度与策略） —— 验证：`python -m pytest tests/test_rl.py -k loop_temp_order -q`
- [ ] B2 tiny RL run（≤100 step；发散则负结果入 notes + 稳定域标注） —— 验证：runs/ 含 rl run-id，reward 曲线或负结果结论行

### 工作项 C 三栈串链

- [ ] C1 `production/train.py` 三栈串链（parent-run-id 链，逐级可溯；`--from` 续跑） —— 验证：`python -m pytest tests/test_train_chain.py -q`
