# 双基线对照表（p2-04 · 同 harness 亲跑）

> 本波切分 `smoke24`（CPU 小样本通路，≥20 题真实出分）；全量 `full` 延后至云端 C6（NPU）。
> 已落实测的基线：2/2（未落者以 `—`/`未跑` 显式标注，不冒充）。

## 一、指标对照（同 split、同采样口径）

| 指标 | DML(自训) | laya | startlux-0.8b | Δ(laya−startlux) | jev-ref(卡面·异口径) |
|---|---|---|---|---|---|
| 准确率 | — | 0.5833 | 0.5833 | +0.0000 | 见下方脚注 |
| ECE(调温后) | — | 0.0790 | 0.0616 | +0.0174 | 见下方脚注 |
| ECE(调温前) | — | 0.0626 | 0.0956 | -0.0331 | 见下方脚注 |
| 逐题耗时中位(ms) | — | 250.7 | 5490.9 | -5240.2 | 见下方脚注 |
| 吞吐(题/秒) | — | 3.060 | 0.184 | +2.876 | 见下方脚注 |

## 二、运行元信息（出处与设备）

| 基线 | 数据集/切分 | 样本数 | 逐题耗时中位(ms) | 吞吐(题/秒) | 采样参数 | 端侧标记 | run-id | source | gate |
|---|---|---|---|---|---|---|---|---|---|
| laya | test/smoke24 | 24 | 250.7 | 3.060 | T=1/top_p=1(readout) | 端侧(CPU通路) | 1005-p2-04-laya-cpu-fec2 | harness | true |
| startlux | test/smoke24 | 24 | 5490.9 | 0.184 | T=1/top_p=1(readout) | 端侧(CPU通路) | 1005-p2-04-startlux-cpu-38b9 | harness | true |

## 三、全量档（`full`，待 C6 云端 NPU 跑）

| 指标 | DML(自训) | laya | startlux-0.8b | Δ(laya−startlux) | jev-ref(卡面·异口径) |
|---|---|---|---|---|---|
| 准确率 | 待 full | 待 full | 待 full | 待 full | 见下方脚注 |
| ECE(调温后) | 待 full | 待 full | 待 full | 待 full | 见下方脚注 |
| ECE(调温前) | 待 full | 待 full | 待 full | 待 full | 见下方脚注 |
| 逐题耗时中位(ms) | 待 full | 待 full | 待 full | 待 full | 见下方脚注 |
| 吞吐(题/秒) | 待 full | 待 full | 待 full | 待 full | 见下方脚注 |

## 脚注：外部卡面参照（非本地亲跑，仅上限参照）

- 卡面 JevBench(/230)：laya 130 · StartLux-0.8B 179 · Jev1.13 199（上限）；延迟：StartLux 12.2ms(H200)、Jev 64.0ms(API)、laya 仅 CUDA 有效。
- JevBench 与本地 typed-decisions 是不同数据集、不同口径，卡面值只作量级参照，不参与 Δ 计算，也不作 gate。
- D11 边界：StartLux-0.8B 仅作对照评测推理（白名单③），绝不训练、绝不当基座、绝不二次发布其权重或衍生。
