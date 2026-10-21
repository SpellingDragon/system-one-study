> **甲路挂起（2026-10-07 用户裁定）**：本域 C5 云端项全部冻结，解锁=p2-13 P2 行（三栈联调）达成；本地已完成项照旧。

# Tasks: p2-06-opd-gkd [W2 · 依赖 p2-05+02；与 07/08/09/10 并发]

> 三级结构：`### 工作项` → 孙任务 checkbox。spec：`specs/opd-gkd/spec.md`。

### 工作项 A 采样器

- [x] A1 on-policy 采样器：学生 top-k（默认 8）选项分布，接口可被 RL 复用 —— 验证：`python -m pytest tests/test_opd.py -k sampler -q`
      （2026-10-06 本地半场跑通：11 passed；出口 `sample_topk / topk_probs / student_topk_scores / TopKSample.to_dict`，形状契约有专测钉住）

### 工作项 B 蒸馏损失

- [x] B1 稠密蒸馏损失：reverse-KL（可切 JSD）+ 数值方向单测（KL(p_t‖p_s)） —— 验证：`python -m pytest tests/test_opd.py -k kl -q`
      （2026-10-06 本地半场跑通：13 passed；方向不对称、λ 十档单调降且终点归零、梯度=份额差 q−p、fp32 域与 keep 列纪律均入测）
- [x] B2 缓存命中训练单测（教师前向计数=0） —— 验证：`python -m pytest tests/test_opd.py -k cache -q`
      （2026-10-06 本地半场跑通：8 passed；账本取 `TeacherStats.forward_calls`，另有开在线档的对照用例证明这个 0 不是恒等式）

### 工作项 C 训练与消融

- [ ] C1 tiny OPD run（载 p2-05 的 SFT 产物续训） —— 验证：runs/ 含 opd run-id 且 config.parent_run_id 指向 sft
      **待 C5**：需要正式 SFT 产物当起点。本波只把 `train()` 的管路（分桶/调度/裁剪/run 四件套/notes）在假壳上验通，并在 `SYS1_TEACHER_INTEGRATION=1` 档用 dev 替身产物（0.6B+adapter，**替身口径、不计入 C1**）跑通一步反传与零教师前向；不在 runs/ 落替身 run，免得被当成 C1 完成。
- [ ] 【执行期回写 2026-10-06：本地 quality 轴每题仅 2–4 候选，top_k=8 截断为 no-op ⇒ on/off 会构造性 Δ≡0 不构成结论——正式消融用 --top-k 2（或 3），或待 20 选项 decision 轴装配后做；见本域 tasks 注记与 p2-06 战报】C2 消融对：on/off 同 seed 两 run，Δ（acc/ECE）结论入 notes（零/负增益如实） —— 验证：两 run-id + notes 含 `OPD 增益 Δ=` 行
      **待 C5**：单变量差集、off 档不截断、Δ 行（零/负/缺数如实）、Δ 行落进两本 notes、无评出口不臆造数字——这些编排契约已由 5 条 `-k ablation` 用例钉住；两个真 run-id 须等正式档。
