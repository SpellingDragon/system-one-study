# Design: p2-11-rl-rlcd

## 技术要点
- `proper_reward` 蓝本 `refs/laya/laya/common.py::proper_reward`（与 ICLR 2026 RLCD "Rewarding Doubt" 同构：log scoring rule 作 reward、最优策略=置信对齐真率；reward 专职校准，质量由 SFT/OPD 承担——报告 Related Work 可引，见 design D9）（log score + spherical + score 类 RPS）：严格适当性单测锁定（合成分布下真值分布奖励严格最优）。
- 优化循环：采样（**复用 p2-06 采样器，零复制**）→ 奖励 → batch 内 advantage 归一（(r-mean)/std，GAE-free 简化档）→ LoRA 梯度更新（REINFORCE 式策略梯度）。
- 温度后置：RL 收敛后才跑 P1 `fit_temperature` 重拟合（温度与奖励不同时优化——时序断言入测试）。
- **reward 双通道（GLM-V verifier 经验，D9）**：proper score 管校准之外，叠加分域确定性 verifier 做方向信号——choice=exact-match 硬奖励、score=within-容差、noul=布尔——两通道加权可配并单列消融（正确性方向 × 置信对齐互补）。
- 稳定性预案：reward EMA 监控 + 发散检测（EMA 连续 N 步劣化即早停）；发散 run 不删除——notes 记"稳定域"（lr/kl 上限的实测边界）。
- train.py 编排：三段各产模型目录，`parent_run_id` 链式溯源；任一段可 `--from <run-id>` 独立续跑。

## 文件清单
| 文件 | 职责 |
|---|---|
| `production/rl_rlcd.py` | 奖励 + RL 循环 CLI |
| `production/train.py` | 三栈编排 |
| `tests/{test_rl,test_train_chain}.py` | 五场景 |

## 风险与回退
- [tiny 档 reward 噪声淹没信号] → 批内归一 + 多样本平均；仍无信号则负结果入库（父 D4-5 允许）。
- [REINFORCE 方差大] → 教师分布作 baseline 的 variance reduction（可选孙任务，时间盒限定）。
- **执行环境（D6 v3）**：on 云端 910B 自研栈（依赖 p2-13 底座决策）；本地仅开发+CPU 冒烟；RL 段费用单列记 D12 ledger。
