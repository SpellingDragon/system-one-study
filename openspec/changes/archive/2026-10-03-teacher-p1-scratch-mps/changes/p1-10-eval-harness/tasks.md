# Tasks: p1-10-eval-harness [W3 · 依赖 p1-09 产物 + p1-05 + p1-04]

> 三级结构：`### 工作项` → 孙任务 checkbox。spec：`specs/eval-harness/spec.md`。

### 工作项 A predict 与评分

- [x] A1 `dmlaya/eval/predict.py`：`--model`/`--endpoint` 双模式，统一预测行 —— 验证：`python -m pytest tests/test_eval.py -k predict -q`
- [x] A2 quality 轴：choice acc/noul acc/score MAE+within-1，qtype×k 分桶对照基线（k=3 与 k=20 分列） —— 验证：`python -m pytest tests/test_eval.py -k quality -q`
- [x] A3 calibration 轴：ECE before/after + Δ 报告 —— 验证：`python -m pytest tests/test_eval.py -k calibration -q`
- [x] A4 parity 轴：backends 路径 vs 纯 torch 路径 argmax 一致率（`@mps`） —— 验证：`python -m pytest tests/test_eval.py -k parity -q -m mps`
- [x] A5 speed 轴：P50/P95 延迟与 tok/s（标注设备/精度） —— 验证：`python -m pytest tests/test_eval.py -k speed -q`

### 工作项 B 溯源与复现

- [x] B1 run-id 溯源强制：无 config 来源的裸权重拒评（`--allow-missing-run-id` 豁免） —— 验证：`python -m pytest tests/test_eval.py -k provenance -q`
- [x] B2 `examples/repro_p1.sh`：s0→s3 tiny 档 + 四轴摘要；退出码为判定，<20min CPU 为参考档（慢机不判负） —— 验证：`bash examples/repro_p1.sh; echo $?`

---

## 完成记录（2026-10-04，本机 macOS/MPS 可用）

| 条目 | 验证命令 | 结果 |
| --- | --- | --- |
| A1 | `pytest tests/test_eval.py -k predict -q` | exit 0（4 passed）；真实打靶 `runs/1004-s2-decision-sft-17ac` × `bench/p1-09/typed_decisions/test.jsonl` 100 条，行键恰 `{id,answers,ms}`、份额和=1 |
| A2 | `-k quality -q` | exit 0（3 passed）；真数 choice k=3 0.6500/基线 0.3333、noul k=2 0.7500/0.5000；k=3 与 k=20 分列（合成桶覆盖） |
| A3 | `-k calibration -q` | exit 0（3 passed）；真数 choice ECE 0.0374→0.0256（Δ-0.0119）、noul 0.0537→0.0014（Δ-0.0523）、pooled 0.0472→0.0111 |
| A4 | `-k parity -q -m mps` | exit 0（1 passed）；真数 12/12=100%、backend=tilelang、max\|err\|=7.6e-05、compiles=+9、blockers=0 |
| A5 | `-k speed -q` | exit 0（2 passed）；真数（17ac, mps, float32, tilelang, 串行）P50=1.46ms / P95=2.34ms，tok/s 标注为 input-only·output_tokens=0 |
| B1 | `-k provenance -q` | exit 0（8 passed）；CLI 裸权重→退出码 2 且载权重前即拒；`--allow-missing-run-id` 带 `exempt=True` 且总评判不过 |
| B2 | `bash examples/repro_p1.sh; echo $?` | exit 0，12 秒（参考上限 1200 秒）；`DEVICE=mps` 另跑一轮亦 exit 0（backend=tilelang） |

一键复现的 run-id 链（CPU 档，产物在 `runs/repro-p1/`）：
`1004-s1-smoke-tiny-ca2f-5 → 1004-s2-decision-sft-d1ec → 1004-s3-calibrate-acc2 → runs/repro-p1/runs/1004-eval-report-225e`。
评测数据来源：`bench/p1-09/typed_decisions/test.jsonl`（typed-decisions 离线转写，100 条，与 `sft_train.jsonl` 的 600 条 id 零重叠）；脚本在评测集缺失时自动退回转写留出子集并在摘要标注 `source=holdout-split`。

门与静态检查：`tools/check_comments.py dmlaya/eval` ✅（6 文件）、`ruff check dmlaya/eval tests/test_eval.py` ✅、`bash -n examples/repro_p1.sh` ✅。
