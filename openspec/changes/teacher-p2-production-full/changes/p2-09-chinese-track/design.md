# Design: p2-09-chinese-track

## 技术要点
- 转写：CMMLU（四选一）→ choice（k=4，criteria=选项文本）；tnews（新闻主题）→ choice；ocnli（蕴含）→ noul（entailment→true 其余→false）。规则显式常量 + 单测锁定（承 P1 transcribe 模式）。
- 配比消融：SFT 语料中文占比 30% vs 50% 两档 tiny run（其余同 seed 同超参），中文集 acc 对照——增益归因数据侧。
- G3 对照：laya 行复用 p2-04 的亲跑设施在中文集重跑（同 harness 同采样）；DML 行 = 配比消融优者。
- 崩溃域记录规范：laya 中文实测值 + 失败样本 3 例（截断 state 等）入 notes——"会崩"要有证据链不是印象。
- 获取：CMMLU=`haonan-li/cmmlu`；CLUE 走 cluebenchmark 官方仓；魔搭无镜像时走官方源（p2-03 registry 统一管理）。

## 文件清单
| 文件 | 职责 |
|---|---|
| `sys1/eval/chinese.py` | 转写 + 评测 |
| `production/configs/zh_mix_{30,50}.yaml` | 配比两档 |
| `tests/test_chinese.py` | 四场景 |

## 风险与回退
- [CMMLU/CLUE 拉取受阻] → 固定 seed 子集自持 + registry 记来源哈希（p2-03 兜底机制复用）。
- [50% 配比英文集回退] → 对照表英中双列呈现，不隐藏 trade-off。
