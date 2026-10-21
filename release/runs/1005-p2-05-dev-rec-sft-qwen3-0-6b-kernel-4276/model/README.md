# p2-05 SFT 产物（1005-p2-05-dev-rec-sft-qwen3-0-6b-kernel-4276）

【来源链】
- 题目：registry 轴 `quality`，数据版本 `typed-decisions@f7a2487+intern-decision@2f81580+cmmlu-subset@master+clue-subset@master`，副本 [{"set": "typed-decisions", "revision": "f7a2487e", "split": "test", "samples": 2000}, {"set": "intern-decision", "revision": "2f81580", "split": "test", "samples": 12447}, {"set": "cmmlu-subset", "revision": "master", "split": "test", "samples": 200}, {"set": "clue-subset", "revision": "master", "split": "validation", "samples": 400}]
- 编成 480 条（过目 480 条，丢弃 无），教师伪标覆盖 100.0%（命中 480/落空 0）
- 教师：`StartLuxAI/StartLux-Decision-4B#scaffold-cpu`，只读缓存 `bench/teacher_cache/p2_05_pseudo`（训练循环零教师前向；D11 红线——只吃它的分布缓存，教师权重一律不进训练）
- 底座：`Qwen/Qwen3-0.6B`（载入口 minimal）{"snapshot_path": "$REPO_ROOT/release/bench/ms_models/models/Qwen--Qwen3-0.6B/snapshots/master", "source": "modelscope", "revision": "master"}
- 旁路：LoRA r16/α32/dropout 0.05，挂 q_proj, k_proj, v_proj, o_proj；实现档 `kernel`（ascend_target=cpu）
- 训练：3 步，loss 2.285952 → 0.642264

> CPU 替身口径（本地开发档）；910B 正式档（gate08b/scaling 全量）待 C5
