# System-One Study · 第一幕：从零学会造（scratch，P1 冻结）

基于 **TileLang** 的类 laya 非自回归 System-1 决策引擎课程项目，分两部分（同一仓库、同一架构：因果 decoder + 选项字母读出 + 类型温度）：

| 部分 | 手册 | 一句话 |
|---|---|---|
| 一 · 学习版 | **[GUIDE.md](GUIDE.md)** | 从零（禁载任何预训练权重）造"小脑"+决策程序+校准，TileLang 内核跑通 CUDA/MPS/昇腾 |
| 二 · 正式版 | **[PRODUCTION.md](PRODUCTION.md)** | 载 0.8B 级开源 backbone，SFT→OPD→RL 三栈；**同 harness 胜 laya 与 StartLux-0.8B**（Jev 为参考上限）；1M/多模态/中文生产化；论文级技术报告收尾 |

## 上手顺序

```bash
bash refs/clone.sh          # 0. 拉齐参考仓（pin，§2.1/§3.1 路径以 refs/<仓名>/ 解析）
pip install -e ".[dev]"     # 1. 本仓库
bash tools/ci.sh            # 2. 门禁自检（ruff→注释→compile→pytest）；hook: git config core.hooksPath .githooks
# 3. 之后照 GUIDE §8「起步五步」；二阶段 benchmark 拉取见 PRODUCTION §5.0；Actions 见 .github/workflows/ci.yml
```

## 目录地图（每个目录内有细指引 README）

```
sys1/      ★ 两轨共享实现层（decision=两阶段同构桥梁；勿分叉）
  decision/  渲染 + 字母读出 + 类型温度 + 分组票选     ← 先读这里
  layers/    注意力/rope/indexpool（混合注意力机制层）
  kernels/   TileLang 三件套 _kernel/_cuda/_mps/_asc
  data/      样本 schema / 转写 / 装配
  eval/      唯一评测 harness + pin registry + 双基线  ← 一切数字出自这里
  lang/ vision/ testing/   （空位，到对应阶段按 README 认领）
learning/    一阶段 s0–s3 脚本（禁第三方权重）
production/  二阶段 assets/sft/opd/rl + teachers/
serving/     二阶段部署：/v1/systemone + FP8 + 跨后端
runs/        实验 lab notebook —— 论文证据链唯一真源（§9.1）
report/      技术报告 LaTeX + 互审记录（§9.2/§9.3）
examples/ tests/ docs/ configs/   共用轻目录（约定见 sys1 与两手册）
refs/        参考仓克隆地（脚本在此；产物被 .gitignore）
```

## 教师版参考答案导览 · 第一部分（已完成，openspec 变更 `teacher-p1-scratch-mps`）

> main 分支已落地一阶段学习版参考实现（77 孙任务、CPU 274 + MPS 36 测试绿）；每行链接：实现 ↔ 手册出处 ↔ 证据。

| 实现 | 手册出处 | 验收证据 |
|---|---|---|
| `sys1/data/schema.py` | GUIDE §3 data/ | `tests/test_schema.py`（21 用例） |
| `sys1/runs/` | GUIDE §3 runs/ · PRODUCTION §9.1 | `tests/test_runs.py`（20 用例，commit 实取 git HEAD） |
| `sys1/lang/bpe.py` + `learning/s0_tokenizer.py` | GUIDE §4 M0-S0 · §8 起步① | run `1003-s0-bpe-16k-realedu-zh-en`（209MB 真语料实训，52 字母单 token） |
| `sys1/model.py` | GUIDE §4 M0-S1 | `tests/test_model.py`（因果性逐比特断言） |
| `sys1/decision/{render,readout,temperature,wide}.py` | GUIDE §4 M0-S2 · §0.4 架构裁决 | `tests/test_decision.py`（52 用例含渲染黄金快照） |
| `sys1/kernels/` + `backends.py` | GUIDE §4 M1 | MPS 门 36 用例；**探针结论：TileLang Metal 方言可用**，6 条实测坑见 `sys1/kernels/README.md` |
| `sys1/data/pretrain_corpus.py` + `learning/s1_pretrain_gpt.py` | GUIDE §4 M0-S1 · §5 语料 | run `1004-s1-smoke-tiny-b477`（loss 9.86→6.37，同 seed 可复现） |
| `sys1/data/transcribe.py` + `learning/s2_decision_sft.py` | GUIDE §4 M0-S2 · §5 转写规则 | run `1004-s2-decision-sft-17ac`（noul 0.64/choice 0.64 双超分桶基线） |
| `sys1/calibrate.py` + `learning/s3_calibrate.py` | GUIDE §4 M0-S3 · §6 校准轴 | run `1004-s3-calibrate-a3a2`（ECE choice -0.012 / noul -0.052） |
| `sys1/eval/`（predict/四轴/report/溯源） | GUIDE §4 M3 · §6 分桶基线 | run `1004-eval-report-225e` + `examples/repro_p1.sh`（一键串跑 exit 0） |
| 变更级 DoD 六条 | design D5 | run `1004-p1-dod-verification-5ce3` 的 notes.md |
| 诚实边界（教师版如实记录） | GUIDE §6/§7 | 采样可读句未达标（冒烟档使然）、repro tiny 档 S3 过调 ECE 为正——均入各自 run notes，非掩盖 |

### MPS 正式档训练与 TileLang 内核链（命令指引，实跑者择机触发）

```bash
V=.venv/bin/python
# ① 正式语料（已实跑：1.175 亿编号/30 片/327s，产出配比受源存量限制、en 会先耗尽——见 corpus_probe）
$V -m sys1.data.pretrain_corpus --config main            # → runs/corpus_main
# ② 纯 torch 基线（实测吞吐 ≈3.3k tok/s → 2000 步 ≈10.5h；250 步一存档，断点用 --resume）
DMLAYA_KEEP_ARTIFACTS=1 $V learning/s1_pretrain_gpt.py --config mps_main
DMLAYA_KEEP_ARTIFACTS=1 $V learning/s1_pretrain_gpt.py --config mps_main --resume runs/<s1-mps-main-...>
# ③ TileLang 可微内核链（前向 gemm/rope/attn_sw 全内核 + 反向 dW 走 gemm_bwd_dw 方言件）
DMLAYA_KEEP_ARTIFACTS=1 $V learning/s1_pretrain_gpt.py --config mps_main --kernel-backend tilelang
```

实跑数据（两档 30 步对比，run `1004-bench-kernel-autocograd-30step-28ba`）：起点 loss 与 torch 逐位相同、曲线重合（Δ≤0.008）；端到端吞吐 -8.7% **尚未翻盘**——翻盘清单（LN/attn 的 Metal 反向件、转置缓存、tile 调优、融合）已记入该 run 结论；单算子热态对比见 `1004-bench-tilelang-vs-torch-mps-1a18`。

## 三条全局红线（细则见各手册）

1. **learning/ 轨禁止加载任何第三方预训练权重**（GUIDE §7-0）；production/ 轨允许。
2. **所有跑分走 `sys1/eval/`**；laya 与 StartLux-0.8B 双基线须同 harness 亲跑（PRODUCTION §6）。
3. **数字必须可溯源**：报告/答辩中每个数字对应 `runs/` 内一个 run；无 run-id 视为捏造（PRODUCTION §11-0）。
4. **一阶段注释即评分证据**：中文注释 + 文件头三件套 + `白话：` 段落（不用技术名词讲明白），`python tools/check_comments.py` 为 CI 门（GUIDE §6.1）。
