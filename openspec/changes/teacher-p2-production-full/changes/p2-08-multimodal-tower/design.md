# Design: p2-08-multimodal-tower

## 技术要点

- **主路线（权重已实证）**：`Qwen3.5-0.8B model.visual.*`（153 张量随主 checkpoint）——A1 探 processor 成熟度（GLM-V 一站式范式优先：messages+url→return_dict；手写像素管线仅备选）；数据流=processor 出 pixel_values → `forward(pixel_values, input_ids)` → state 段嵌 `<|vision_start|>…<|vision_end|>`。
- **token 经济**（视觉 token 成本账）：448² 图 ÷ patch16² = 784 patch → **2×2 merge 压至 ~196 token/图**——决策场景单图 state 增量可控；评测批量走 pixel_values 堆叠。
- **微调面**：LoRA 挂 language 侧（qkvo），视觉塔+merger 冻结；若 A1 发现 merger 需调，升为小 lr 解冻单列消融。GLM-5.3-Flash 为架构语义参照（2×2 merge/token 入 vocab/temporal——Qwen3.5 已原生同构，无需自建）。
- 评测：MMBench-CN 固定 seed 子集 ≥200 + OCRBench 抽样 100；无图对照列同表呈现；加/去图对照点入 runs。
- 备选全套在册（processor 不成熟触发）：SigLIP-so400m 冻结 + 两层 MLP projector（LLaVA 式两段：预热对齐→小 lr 联合）；不做 cross-attention。
- **塔替换消融（可选，2026-10-05 议定不采为主路线）**：若加/去图对照显示视觉增益弱，可做塔容量消融（自带 12 层 vs GLM-V 24 层抽取 vs SigLIP）回答“瓶颈在塔还是在主干”——GLM-V 塔跨家族特征错位需重对齐（等于外挂路线成本）且须下完整 checkpoint 抽取；编码（自带塔，与主干同家族匹配）与知识（GLM 320B 教师伪标）分工已是最优，不替换主路线。
- 视觉伪标：**GLM-5.3-Flash API 离线包**（经 p2-02 缓存，soft 分布）；**多模态 OPD 降为可选**（在线蒸馏需教师实时打分与零在线 API 冲突；若做走批量异步——采样落盘→批量回填→续训，仍是离线语义；文本域 OPD 已消融验证，多模态 SFT(伪标)+RL(确定性 verifier) 为主，如实入消融表）。
- 参考落点：`refs/GLM-V/inference/trans_infer_cli.py`（一站式调用）、`refs/GLM-V/glmv_reward/`（verifier 分域）；Qwen3.5 vision_config（本仓 design.md D9 实证记录）。

## 文件清单

| 文件 | 职责 |
|---|---|
| `release/sys1/data/mm.py` | 图文装配（vision token 嵌入 state） |
| `release/production/vision_finetune.py`（或并入 sft 配置） | 冻结塔 + LoRA language 侧微调 |
| `release/sys1/eval/multimodal.py` | 评测 + 无图对照 |
| `release/tests/test_mm.py` | 五场景（A1 探针/装配/对照/评测/缓存） |

## 风险与回退

- [transformers 5.18 qwen3_5 processor 不成熟] → A1 探针前置裁决；备选 SigLIP+projector 全套在册，切换成本单域内。
- [视觉 token 挤占 ctx（多图场景）] → 决策场景限单图/双图；多图归后续。
- [冻结塔下视觉增益弱] → 加/去图对照如实报；merger 解冻消融裁决。
