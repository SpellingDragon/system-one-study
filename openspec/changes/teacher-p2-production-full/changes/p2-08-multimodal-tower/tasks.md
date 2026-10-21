# Tasks: p2-08-multimodal-tower [W2 · 依赖 p2-01+02+03；与 06/07/09/10 并发]

> 三级结构：`### 工作项` → 孙任务 checkbox。spec：`specs/multimodal-tower/spec.md`。

### 工作项 A 视觉塔与装配

- [x] A1 【探针·前置】自带塔路线成熟度实证（主路线，权重已证 153 张量）：transformers 5.18 processor 一站式（messages+image url→return_dict）+ 视觉塔前向出 token 数账（2×2 merge）+ 与  `<\|vision_start/end\|>` 三件套嵌入 state 段的渲染对齐；不成熟即切换 SigLIP+projector 备选并记录取证 —— 验证：CPU 上 1 图过 processor→塔前向→token 拼接全链 exit 0，run notes 含路线判定行
- [x] A2 `sys1/data/mm.py` 图文装配（image token 入 state 段，schema 兼容） —— 验证：`python -m pytest tests/test_mm.py -k assemble -q`

### 工作项 B 对照与评测

- [ ] B1 加图/去图对照点 run（同一样本两预测入 runs） —— 验证：runs/ 含 mm 对照 run-id
- [x] B2 `sys1/eval/multimodal.py`：MMBench-CN 子集（≥200 题）+ OCRBench 抽样，含无图对照列 —— 验证：评测报告含 acc/样本数/run-id/无图列
- [x] B3 视觉伪标缓存复用冒烟（二次冒烟教师前向=0） —— 验证：`python -m pytest tests/test_mm.py -k pseudo_cache -q`
