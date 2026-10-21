# sys1/eval/baselines/ — 双基线亲跑产物（G1–G3 的 gate 对照）

> 纪律（PRODUCTION §6）：基线必须**同 harness、同 split、同采样参数亲跑**，不接受转抄 README 数字；每次亲跑落一个 `runs/` 目录并在此留索引。

## 目录约定

```
baselines/
├── README.md            # 本文件
├── laya/                # laya 亲跑：run-id 索引 + 对照表（注意：其 kernel 仅 CUDA，速度基线只在 CUDA 有效）
├── startlux-08b/        # StartLux-Decision-0.8B 亲跑（HF 权重 CC BY-NC，课程/科研使用允许）
└── jev_reference.md     # Jev 1.13 仅卡面数字作参考列（无权重不可亲跑；API 小额实测为可选项，须记预算）
```

## 实现参考（只借思想，不拷文件）

结构范本来自 `refs/jev-cookbook/main/06_模型评测/benchmark/`（⚠️ 该仓 **CC BY-NC-SA**，本仓 MIT：允许阅读理解后重写，禁止拷贝代码/数据文件入仓；评测数据走上游已 pin 源）：

- `adapters/` 的三类适配器：协议端点统一（`/v1/systemone` 同协议 → 一个 typesafe 风格 adapter 即可同时打 StartLux 服务与 DML 自身）；laya 走其 Python 包/server；mock 供 CI。
- `ledger.py` 的**诚实评测纪律**：无重试、无回退、按调用记账（预算写进 run config）。
- `metrics.py`：四维（准确率 / Brier / ECE top-label 10 bins / 延迟），与 `sys1/eval/registry.py` 口径一致（Uniform 基线复现卡面值作自检）。

## 上游参考数字（仅供对照，gate 以亲跑为准）

| 系统 | JevBench /231 | Intern avg | DI 0.2.1 | 3-field 延迟 |
|---|---|---|---|---|
| laya | 130 | 57.77 | 6.04 | —（仅 CUDA） |
| StartLux-0.8B | 179 | 85.03 | 38.86 | 12.2 ms（H200） |
| Jev 1.13（参考上限） | 199 | 88.74 | 57.91 | 64.0 ms（API） |
