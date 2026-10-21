# runs/ — 实验 lab notebook（论文证据链，唯一真源）

> 铁律（PRODUCTION §11-0）：**报告/答辩中每个数字必须对应这里一个 run 目录；无 run-id = 捏造**。过程记录，禁止事后补记。

## 目录规范

```
runs/<MMDD>-<stack>-<model>-<variant>[-seedN]/     # 例：1015-sft-q08b-typedtrain-seed3
├── config.yaml        # 全量超参 + git_commit + data_rev_pin(来自 eval/registry) + hardware + parent_run_id
├── metrics.jsonl      # 逐 step/eval 点：{step, split, loss, acc_bucketed, ece, latency_ms?, mem_peak_mb?}
├── system.json        # GPU 型号/驱动/torch/tilelang/CANN 版本
├── stdout.log
└── notes.md           # 三行必填：假设 → 观察 → 结论（失败实验同样必须入库）
```

## 强制节点（缺一按 §9.1 扣分）

三栈各 ≥2 消融或 3 seeds ｜ 温度校准前/后 ｜ G8 每规模点 ｜ 长文每档窗口/Top-K needle ｜ 多模态加图/去图 ｜ 三后端全量六轴 ｜ **双基线亲跑**（laya、StartLux-0.8B 各一条 run）。

## 约定

- `metrics.jsonl` 是曲线唯一来源——**禁止为画图重跑实验**。
- wandb/tb 只作可选镜像，本目录为权威（不依赖托管服务）。
- 大权重文件不入 git（`.gitignore` 排除 `*.safetensors`）；config 记录权重来源+commit 保证可重生成。
- `report/` 写数字时引用格式：`run:<目录名>#metrics.jsonl:step=…` 或 `run:…#summary(acc)`。
