## ADDED Requirements

### Requirement: 视觉塔载入
`production/` 多模态路径 SHALL 载入 SigLIP/CLIP ViT（ModelScope 镜像优先），两段式：冻结投影对齐 → 解冻微调；视觉特征以 image token 序列注入 state 段（`sys1/data/mm.py` 装配）。

#### Scenario: 图文样本装配
- **WHEN** 装配一个含 1 图 + 3 选项的样本
- **THEN** 序列含 image token 占位且过 schema 校验，纯文本问题不受影响

#### Scenario: 加图/去图对照
- **WHEN** 同一样本有图/无图各跑一次
- **THEN** 两次预测均入 runs（对照点，G7 多模态消融形态）

### Requirement: 图文决策评测
`sys1/eval/multimodal.py` SHALL 在 MMBench-CN 子集（≥200 题）与 OCRBench 抽样上报告决策准确率；对照表含"纯文本 backbone 无图"列以量化视觉增益。

#### Scenario: MMBench-CN 子集跑分
- **WHEN** 对多模态模型跑子集
- **THEN** 报告含 acc、样本数、run-id；无图对照列同表呈现

### Requirement: 视觉教师伪标
多模态 SFT 伪标 SHALL 经 teacher-adapters 视觉接口（在线 VLM 或离线包）产出；缓存优先。

#### Scenario: 伪标缓存复用
- **WHEN** 第二次多模态 SFT 冒烟
- **THEN** 视觉教师前向计数为 0
