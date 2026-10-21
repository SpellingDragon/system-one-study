# release/ — 第二阶段（P2 正式版）主场

> 变更计划：`../openspec/changes/teacher-p2-production-full/`（一级 tasks.md 是编排清单；启动门"P1 已归档"已满足）。
> 手册：`../scratch/PRODUCTION.md`；工作规则见仓库根 [AGENTS.md](../AGENTS.md)。

## 本目录结构

```
release/
├── sys1/          # 从 scratch 冻结复制的基线（P2 计划声明 decision/ 零改动；kernels/autograd 可微链随基线带入）
├── production/      # P2 训练栈（assets/sft/opd/rl_rlcd/train + teachers/ + configs/）——待实施
├── serving/         # HTTP /v1/systemone + prefix_cache ——待实施
├── tests/ configs/ examples/
├── runs/            # P2 实验证据链（run-id 规范承 scratch）
└── bench/           # P2 评测数据（typed-decisions 等可从 ../scratch/bench 按需复制或按 registry pin 重拉）
```

## 启用步骤（首个 P2 agent 会话）

```bash
cd release
$HOME/miniconda3/bin/python3.12 -m venv .venv      # 或任意 ≥3.10 解释器
.venv/bin/python -m pip install -e ".[dev]" && .venv/bin/python -m pip install modelscope safetensors
# 冻结基线自检（应全绿——若红，说明基线被动过，先查 git）
.venv/bin/python -m pytest tests -q 2>/dev/null || echo "（tests/ 为空属预期；基线自检用 ../scratch 的测试指向本目录跑）"
```

## SFT（p2-05）命令指引

本地半场（CPU 替身，只验通路，不出任何评测分）：

```bash
cd release
# ① 先用袖珍教师把伪标抄进只读账本（一次性；训练循环此后零教师前向）
.venv/bin/python -m production.sft --config production/configs/tiny_cpu.yaml --prefill-pseudo
# ② 百步 tiny 冒烟：loss 曲线落在 runs/<run-id>/metrics.jsonl，产物落在同目录 model/
.venv/bin/python -m production.sft --config production/configs/tiny_cpu.yaml
# ③ 单测（装配 / 混合配比 / 读点损失 / LoRA 生效 / kernel 档对拍 / 配置一致性 / 端到端）
.venv/bin/python -m pytest tests/test_prod_sft.py -q
```

正式档（910B 主路线，待 C5 上云触发；配置与本地只差底座/批规模/步数四类键）：

```bash
# 8B 教师 → 0.8B 学生 gate 档
python -m production.sft --config production/configs/gate08b.yaml    --run-prefix p2-05-sft
# 0.6B 规模档（同口径放大等效批）
python -m production.sft --config production/configs/scaling_06b.yaml --run-prefix p2-05-sft
# 上云前置：真伪标包（StartLux 教师分布缓存，权重绝不进训练）落 bench/teacher_cache/ 后
# 只改 teacher_cache/teacher_model_id 两键；训练侧缓存以 read_only 打开，写入会被当场拦住。
```

注意：P2 计划文档中的 `sys1/...`、`production/...` 相对路径一律以本目录为基准。

## 目录纪律（2026-10-06 规整所定，防再乱）

| 位置 | 放什么 | 不放什么 |
|---|---|---|
| `sys1/` | P1 冻结件同步副本（import 底座） | P2 新码 |
| `production/` `ascend/` `serving/` | P2 各域交付码 | 任何一次性脚本 |
| `tests/` | 各域验收测试（门） | — |
| `examples/` | **可复跑脚本**（如 a2_needle_curve.py，云端复跑同款） | — |
| `runs/` | 四件套证据档案；过程凭据入 `<run>/process_artifacts/` 随 run 归档 | — |
| `bench/`（gitignored） | 数据/缓存/**全部过程件**（`.probe_*/.patch_*/.out_*` 一律入 `bench/process/<域>/`） | — |
| **仓库根（release/）** | 只允许上表目录 + pyproject/README | **禁止任何 `.probe_/.patch_/.out_/.smoke_` 落根**（写后即入 bench） |

纪律执行归编排者：波次收尾巡检 release 根，散件即归档或清除；agent 派发包含此条。
