# Tasks: p2-10-serving-systemone [W2 · 依赖 p2-05；与 06/07/08/09 并发]

> 三级结构：`### 工作项` → 孙任务 checkbox。spec：`specs/serving-systemone/spec.md`。

### 工作项 A HTTP 服务

- [x] A1 `serving/server.py`：POST `/v1/systemone` + `/health` + 批量 —— 验证：`python -m pytest tests/test_serving.py -k http -q`
- [x] A2 endpoint vs 本地一致单测（`--endpoint` 打服务，argmax 与 `--model` 一致） —— 验证：`python -m pytest tests/test_serving.py -k parity -q`

### 工作项 B 延迟与缓存联动

- [x] B1 延迟协议：warm-up 20 + N=50 串行，mean/P50/P95（1 choice+1 noul+1 score 同前向） —— 验证：延迟报告 json 含三统计与设备标注
- [x] B2 prefix_cache 联动：同 state 多问自动复用 + `/stats` 命中计数 —— 验证：`python -m pytest tests/test_serving.py -k hit_count -q`〔依赖注记：p2-07 前缀件已就绪，`serving/latency.py::run_all` 的 run 级 B2 证据已出（run 1006-p210-serving-local-half-c5d6，prefix_hits_delta=2/喂入符号 193）；本波缺的是 `-k hit_count` 进程内单测（需为假引擎补 kv_engine/HF past 复用面）——2026-10-06 已落：`tests/test_serving.py` 加 `FakeKVEngine`（past=已看过的编号列，末位数字回交同一个 `forward_hidden`，复用路与重算路因此共用一条公式、护栏只判等）+ `test_hit_count_service_path_counts_hits_and_never_repays_prefix`；`pytest tests/test_serving.py -k hit_count -q` 1 passed（全文件 8 passed），实测同 state 三问命中 2（spec 口径）、跨请求累计 5，前缀付费 168 只发生在冷启那一问，parity_failed=0 / quarantined=[] / prefix_recomputes=0〕
