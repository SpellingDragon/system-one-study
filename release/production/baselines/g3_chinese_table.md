# G3 中文对照表（DML vs laya · 同 harness 亲跑）

> 波次边界：本表由 p2-09 本地半场出**骨架**；DML 行取自中文配比消融的优者（30%/50% 两档，须待正式 SFT 产物），laya 行走 p2-04 已打通的亲跑通路，两者实测格均标 `待 C5`，不以替身数字或卡面值顶替。
> 转写版号 `zh_decision_v1`；题型 choice/noul 二型，分桶口径 (qtype, k)，禁止跨 k 合并。

## cmmlu-decision（split=`test`，切分名 `zh-cmmlu`）

| 指标 | DML(配比优者) | laya | Δ(laya−DML) |
|---|---|---|---|
| 准确率 | 待 C5 | — | 待 C5 |
| ECE(调温后) | 待 C5 | — | 待 C5 |
| 逐题耗时中位(ms) | 待 C5 | — | 待 C5 |
| 吞吐(题/秒) | 待 C5 | — | 待 C5 |

### 运行出处

| 基线 | 数据集/切分 | 样本数 | 采样参数 | 设备 | run-id | source | gate |
|---|---|---|---|---|---|---|---|
| DML(配比优者) | cmmlu-decision/zh-cmmlu | 待 C5 | 待 C5 | 待 C5 | 待 C5 | 待 C5（配比消融优者） | false |
| laya | cmmlu-decision/zh-cmmlu | — | — | — | — | 未跑（待 C5） | false |

### 崩溃域证据：失败样本（至多 3 例，题面截断展示）

| # | 题号 | 题型/k | 人工 | 模型 | 题面摘录（截断） |
|---|---|---|---|---|---|
| 1 | 待 C5 | 待 C5 | 待 C5 | 待 C5 | 待 laya 中文亲跑落预测 jsonl 后挂载 |
| 2 | 待 C5 | 待 C5 | 待 C5 | 待 C5 | 待 laya 中文亲跑落预测 jsonl 后挂载 |
| 3 | 待 C5 | 待 C5 | 待 C5 | 待 C5 | 待 laya 中文亲跑落预测 jsonl 后挂载 |

## clue-decision（split=`validation`，切分名 `zh-clue`）

| 指标 | DML(配比优者) | laya | Δ(laya−DML) |
|---|---|---|---|
| 准确率 | 待 C5 | — | 待 C5 |
| ECE(调温后) | 待 C5 | — | 待 C5 |
| 逐题耗时中位(ms) | 待 C5 | — | 待 C5 |
| 吞吐(题/秒) | 待 C5 | — | 待 C5 |

### 运行出处

| 基线 | 数据集/切分 | 样本数 | 采样参数 | 设备 | run-id | source | gate |
|---|---|---|---|---|---|---|---|
| DML(配比优者) | clue-decision/zh-clue | 待 C5 | 待 C5 | 待 C5 | 待 C5 | 待 C5（配比消融优者） | false |
| laya | clue-decision/zh-clue | — | — | — | — | 未跑（待 C5） | false |

### 崩溃域证据：失败样本（至多 3 例，题面截断展示）

| # | 题号 | 题型/k | 人工 | 模型 | 题面摘录（截断） |
|---|---|---|---|---|---|
| 1 | 待 C5 | 待 C5 | 待 C5 | 待 C5 | 待 laya 中文亲跑落预测 jsonl 后挂载 |
| 2 | 待 C5 | 待 C5 | 待 C5 | 待 C5 | 待 laya 中文亲跑落预测 jsonl 后挂载 |
| 3 | 待 C5 | 待 C5 | 待 C5 | 待 C5 | 待 laya 中文亲跑落预测 jsonl 后挂载 |

## 脚注

- `cmmlu-decision`：200 题（choice=200 noul=0），真值覆盖 200，sha256 `7a4e58083ed6…`，源副本 `cmmlu-subset`
- `clue-decision`：400 题（choice=314 noul=86），真值覆盖 400，sha256 `5e723144eadb…`，源副本 `clue-subset`
- tnews 的候选文字沿用源头给的类别码（`"100".."116"`），盘上 pin 的 parquet 没带"码↔类别名"对照表，本域不凭记忆补名字——这条局限随中文 acc 一起披露。
- CLUE 取 validation 不取 test：实测 test 档 label 整列为 -1（官方隐藏答案），没有真值的题面只能当语料不能当考卷（口径由 p2-03 裁定并登记）。
- D11 边界：laya 行仅作对照评测推理，同 harness 同采样参数（`待 C5` 的格子不许拿卡面值填）。
