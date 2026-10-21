# production/ — 第二阶段训练栈（允许载重，锁 0.8B 级 gate）

> 目标（PRODUCTION §1）：**0.8B 级** backbone 同 harness 胜过 laya 与 StartLux-0.8B（Jev 参考）；三栈 SFT→OPD→RL 每段增益可消融（G7）。复用 `sys1/{decision,kernels,eval,data}`，禁止分叉。

## 文件清单

| 文件 | 职责 | 关键约定 |
|---|---|---|
| `assets.py` | 载开源资产：backbone（Qwen3-0.6B/0.8B 级，ModelScope 优先）、ViT（SigLIP）、teacher；**记录来源+commit 入 run config** | 接缝三校验（PRODUCTION §0 过渡）：`check_tokenizer` 字母单 token、chat template 思考关闭后缀、letter_rows 随 hidden_size 重建 |
| `sft.py` | 读出位 CE；标注数据 + 教师伪标（soft target） | 超参基线照抄 `refs/StartLux-Decision/finetune/finetune_lora.py`（r16/α32/dropout0.05/lr1e-4/warmup5%/cosine/2ep/16384tok×4）；框架可换 TRL/ms-swift |
| `opd.py` | On-Policy Distillation（≈TRL `GKDTrainer`）：学生自采样决策状态，教师对同选项给稠密分布，KL/JSD | **教学假设声明**：判别式下 OPD 可能≈soft-KD，增益幅度由 G7 消融裁决，不许未消融先宣称 |
| `rl_rlcd.py` | 严格适当得分规则 RL：`sys1` 版 `proper_reward`（log+spherical+RPS），advantage 归一；温度最后重拟合 | 奖励函数蓝本 `refs/laya/laya/common.py::proper_reward`；多轮用 `ep_group/ep_step`（`td_lambda_targets` 选做） |
| `teachers/` | 教师适配层：**接口=渲染后 prompt + 选项键 → 选项分布**（文本教师取末位字母 logprob；视觉教师产伪标） | 教师可离线缓存分布（PRODUCTION §8.1 降本要点） |
| `train.py` | 编排 SFT→OPD→RL，串 run 链（每个 run 记 parent-run-id，供 G7 消融图） | 每步产物落 `../runs/`，命名含栈名与 seed |
| `configs/` | 规模锚点配置：`gate08b.yaml`（**参与 gate**）、`scaling_*.yaml`（G8 曲线点） | gate 判定只认 `gate08b` 的 run |

## 与基线的关系（防误读，PRODUCTION §1 难度定位）

StartLux-0.8B=38.86 DI 属"载入同级开源底座后的**复现-加**"：路线适配与 backbone 起点已由课程决策免费解决；本目录的真实工作量在**数据配方**（StartLux 训练数据未公开→自采 Intern-Decision suites train + typed-decisions train + 教师伪标）与三栈调教。硬活在 G3–G6/G8，勿把工时烧在 G1 刷分上。
