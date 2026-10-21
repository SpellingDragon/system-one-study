# Design: p2-03-eval-registry

## 技术要点
- 注册项字段：`{id, source(git/hf/modelscope/synthetic), revision, split, seed, sampler}`——报告自动附 `export_versions()` 清单（PRODUCTION §6-2）。
- 拉取脚本照抄 `StartLux-Decision/eval/fetch_benchmarks.sh` 的 pin 实践：git clone + checkout commit；HF snapshot_download 带 revision；合成 needle 由 seed 生成（不入库）。
- **Uniform 锚点**：以均匀分布跑 typed-decisions 打分，须复现卡面 KL/TV/Brier=0.444/0.381/0.238（容差 0.01）——harness 接错的熔断器（上游 `score_task` 同款打分器）。
- 六轴 = P1 四轴 + longctx + multimodal；`--axes all` 时缺数据轴显式 `n/a`（不许静默省略）。
- 子集策略：CMMLU/MMBench-CN/OCRBench 各取固定 seed 子集（≥200 题）自持副本入 `bench/`（gitignore），registry 记来源与哈希（魔搭无镜像时的可得性兜底）。

## 文件清单
| 文件 | 职责 |
|---|---|
| `sys1/eval/registry.py` | 注册/pin/导出/拉取 |
| `sys1/eval/run.py` | 六轴调度 |
| `tests/test_registry.py` | 五场景 |

## 风险与回退
- [打分器与上游口径偏差] → 锚点自检单测化；不达标即红，禁止绕过。
- [子集自持引发可比性质疑] → registry 记原始 revision + 子集 seed + 哈希，报告脚注披露。
