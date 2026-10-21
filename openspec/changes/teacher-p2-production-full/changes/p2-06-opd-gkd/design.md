# Design: p2-06-opd-gkd

## 技术要点
- 采样器独立成模块（`production/opd.py::sampler`）：输入模型+状态 → top-k (opt, p)；k=8 默认；**RL 栈直接 import 复用**（接口稳定是关键，实现可换）。
- 损失方向锁定（D9 业界数学澄清，Agarwal/OPD-survey 口径）：逐 token 粒度的 `KL(p_teacher ‖ p_student)`（reverse/mode-covering——教师多峰不被抹平）；可切 JSD（β=0.5）。数值单测以手工小例验证方向。
- on-policy 流程：学生采 k 选项 → 教师对**同 k 选项**归一化分布（缓存键含选项集）→ 学生最小化散度；混 ground-truth 比例可配（默认 0.2）。
- 消融档：同 seed 同数据 on/off 两 run，报 typed-decisions acc 与 ECE 的 Δ——**零/负增益如实记 notes**（教师版把"假设不成立"也教学化）。
- TRL GKDTrainer 作对照参考（手册 P2 建议"直接用 TRL"）；教师版手写核心循环（教学透明）+ notes 记与 GKD 的差异点。

## 文件清单
| 文件 | 职责 |
|---|---|
| `production/opd.py` | 采样器 + 蒸馏损失 + CLI |
| `tests/test_opd.py` | 五场景 |

## 风险与回退
- [散度损失数值不稳（fp16）] → log-softmax 在 fp32 域计算后回投；KL 下界 clamp 防 -inf。
- [采样器接口被 RL 需求撕裂] → 变更走父变更一级评审（接口是两栈公共契约），不许单域私改。
- **执行环境（D6 v3）**：on 云端 910B 自研栈（依赖 p2-13 底座决策）；教师在线打分同卡内存账（64GB 显存，学生训练+教师推理同驻估算入 C2 对拍记录）；本地仅开发+CPU 冒烟。
