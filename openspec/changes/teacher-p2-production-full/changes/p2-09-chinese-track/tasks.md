# Tasks: p2-09-chinese-track [W2 · 依赖 p2-03+05；与 06/07/08/10 并发]

> 三级结构：`### 工作项` → 孙任务 checkbox。spec：`specs/chinese-track/spec.md`。

### 工作项 A 转写与评测

- [x] A1 CMMLU 子集 + CLUE(tnews/ocnli) 决策化转写（choice/noul）+ registry pin —— 验证：`python -m pytest tests/test_chinese.py -k transcribe -q`
    落定（2026-10-06，release/.venv 复跑 17 passed / 整文件 24 passed）：`sys1/eval/chinese.py`
    复用 `sys1/data/transcribe` 既有件出三型折法；registry 走 `kind="derived"` 零流量派生路登记
    `cmmlu-decision`(200/choice, split=test) 与 `clue-decision`(400=choice 314+noul 86, split=validation)，
    qtype 三口一致、`--zh` 按钮在 argparse 上真存在、`fetch --zh` exit 0；D2 隔离由
    `tests/test_registry.py -k split_isolation`（2 passed）与本域
    `test_transcribe_quality_axis_carries_chinese_and_d2_isolation_holds` 双向守护。

### 工作项 B 配比消融与对照

- [ ] B1 中文配比消融：30%/50% 两档 tiny SFT 对照 —— 验证：两 run-id + 中文 acc 对照入 runs
    **待 C5**（须正式 SFT 产物才能跑两档对照）。本波已把参数化落定：`production/configs/zh_mix_30.yaml`
    与 `zh_mix_50.yaml` 两档（除 `zh_ratio` 外逐键同 tiny_cpu，`check-mix` 实测"意外差异={}"），
    两枚待接入键 `zh_ratio`/`zh_corpus` 待 p2-05 消费口在 `default_cfg()` 增键；
    `zh_corpus` 现挂显式占位 `PENDING@待 C5:zh-train`（registry 的 train 轴还没有中文语料档，
    卫生闸 `assert_mix_hygiene()` 拒绝拿评测集当语料）。
- [ ] B2 G3 对照表：DML vs laya（亲跑，崩溃域如实记录 + 失败样本 3 例） —— 验证：对照表含 laya 中文行 + ms + 采样参数
    **待 C5**（DML 侧要用正式 SFT 产物出分）。本波已交表骨架 + 挂列机制：
    `production/baselines/g3_chinese_table.md`（60 行，两集各一块 acc/ECE/ms_p50/吞吐 + 运行出处 +
    每集 3 格失败样本位），laya 列走 p2-04 打通的 `row.json` 通路（`tests/test_chinese.py::test_g3_fills_laya_cell_when_rowjson_exists`
    已证明喂进真数就会挂上）；缺数格一律显式 `待 C5`，不留空白冒充 0 分。
