# Design: p1-02-run-notebook

## 技术要点
- config.yaml 自动注入 `commit`（`git rev-parse HEAD`）+ 调用方超参 + `data_revision` 占位——溯源三要素齐。
- metrics.jsonl 是曲线**唯一真源**：同 step 同指标名拒绝重写（`MetricsConflictError`），防止"为画图重跑"污染曲线。
- notes.md 三行骨架（假设→观察→结论）；`finish()` 强校验结论行——失败实验也必须留下结论（负结果入库的制度化）。
- run_id 规则：`<MMDD>-<slug>-<hash4>`；目录名即 run_id（与 .gitignore 的 `!runs/**/config.yaml` 等白名单对齐，大权重不入库）。

## 文件清单
| 文件 | 职责 |
|---|---|
| `dmlaya/runs/__init__.py` + `context.py` | RunContext 全部逻辑 |
| `tests/test_runs.py` | 创建/防重/结论强制四场景 |

## 风险与回退
- [并发两 run 同秒同 slug 冲突] → hash4 取 config 内容散列，冲突概率可忽略；仍撞则递增后缀。
- [metrics 半行写入（进程被杀）] → 追加后 flush；load 时容忍并剔除残行。
