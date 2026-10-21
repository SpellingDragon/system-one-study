# Proposal: p2-02-teacher-adapters — 教师适配（二级子变更）

> 父变更：teacher-p2-production-full（W0，无前置依赖）

## Why
三栈与多模态的信号源：文本教师（末位字母 logprob → 选项分布）与视觉教师（图文伪标）。Mac 教师推理慢——**离线缓存先行**是父 design D3 的并发关键（慢推理只付一次成本，三栈全吃缓存）。

## What Changes
- 新增 `production/teachers/text.py`：`score_options(prompt, option_keys) -> {opt: p}`（**三教师分工（终态，D9）**：文本决策教师 = **StartLux-4B**（主力——决策专精、`score_options` 对接其原生字母读出零 prompt 工程、本地权重；教学叙事=蒸馏 4B 知识击败其 0.8B 家族）；视觉教师 = **GLM-5.3-Flash API 离线伪标**（320B 质量、§11-4 红线合规——外部仅作教师离线产标、全链零在线 API；逐选项置信 soft 分布主径 + `<answer>` 协议兜底；本地 9B 跑不动故走 API，量级 3–6k 次几十元级）；**Qwen3.5-4B 为文本备选**（同家族 tokenizer 接缝复用 p2-01 校验）。**权重已实证（2026-10-05）**：魔搭 `StartLuxAI/StartLux-Decision-4B` 可达（32 文件、权重 2 分片、**自带 decision_config.json**、含 vision/video preprocessor——或可兼视觉教师，B3 探测）；GLM-5.3-Flash API key = 环境变量 **`ZAI_API_KEY`**（已在位，长度 49，值不入库））
- 新增 `production/teachers/vision.py`：(image, prompt, keys) → 分布——**GLM-5.3-Flash API 离线伪标包**（主路线：逐选项置信 soft 分布 + answer 协议兜底，parquet 回放二次零 API）+ 本地 Qwen3.5-9B 备选
- 新增 parquet 分布缓存：(model_id, prompt_hash, keys_hash) 键，miss 才推理并回写
- 新增 `tests/test_teachers.py`

## 边界与依赖（不耦合声明）
- 依赖：P1 decision/ 的渲染输出（prompt 形态）——只依赖字符串契约。
- **被依赖方**：p2-05 SFT（伪标）、p2-06 OPD（稠密分布）、p2-08 多模态（视觉伪标）。
- 接口面：`score_options / VisionTeacher.score / DistCache`。

## 验收
- spec scenarios 全过：`specs/teacher-adapters/spec.md`（5 场景）
- 一级 DoD 关联项：三栈消融（教师缓存是成本前提）
