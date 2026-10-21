# Proposal: p2-08-multimodal-tower — 多模态（二级子变更）

> 父变更：teacher-p2-production-full（W2；依赖 p2-01 载重 + p2-02 视觉教师 + p2-03 registry；与 06/07/09/10 并发）

## Why

G5（**叙事修正，D10**：StartLux-0.8B 实测同基座原生多模态但未做视觉决策训练——本域改为**同基座多模态对照**，视觉增益在同 backbone 下可归因；vs laya 仍严格占优）：用户裁决全做。**主路线已定且有权重实证（D9 + 2026-10-05 查证）**：Qwen3.5-0.8B 自带原生视觉塔——`vision_config`（12 层 ViT-Base 系 hidden768/heads12/patch16/**2×2 spatial merge**/**temporal_patch_size=2**/out_hidden=1024 对齐主干）+ **153 个 `model.visual.*` 张量随主 checkpoint 发布** + `image/video_token_id` 与 `vision_start/end_token_id` 三件套预置——与手册参照 GLM-5.3-Flash（24 层/patch14/448/2×2 merge/视觉 token 入 vocab）同构做法的小号版，**无需自建视觉塔、无需另下权重**。

## What Changes

- A1 探针（前置首孙任务）：transformers 5.18 对 qwen3_5 多模态 processor 的成熟度——**调用范式采 GLM-V 官方 CLI 一站式**（`processor.apply_chat_template(messages 含 {type:image,url}, tokenize=True, return_dict=True)` 一次出 pixel_values+input_ids，`refs/GLM-V/trans_infer_cli.py` 实证）；不成熟即触发备选
- `sys1/data/mm.py`：图文装配——state 段嵌 `<|vision_start|>…<|vision_end|>`，image token 对由 processor 展开；schema 兼容（纯文本样本零影响回归）
- 微调面：**LoRA 挂 language 侧（qkvo），视觉塔与 spatial-merger 冻结**（0.8B 资源纪律；merger 若需调则小 lr 解冻单列消融）
- `sys1/eval/multimodal.py`：MMBench-CN 子集（≥200 题）+ OCRBench 抽样；**无图对照列**（同题去图跑纯文本）量化视觉增益；加图/去图对照点入 runs（G7 消融形态）
- 视觉教师伪标经 p2-02 缓存；备选路线（processor 不成熟时）：SigLIP-so400m + 两层 MLP projector（LLaVA 式两段），全套设计在册单域内切换

## 边界与依赖（不耦合声明）

- 依赖：p2-01（backbone 含视觉塔）、p2-02（视觉教师/伪标缓存）、p2-03（评测集）。
- 被依赖方：p2-12（G5 图表）。
- 接口面：`mm.assemble(state, images)` + processor 管线封装。
- 禁止：改动 `decision/`（图像 token 仅是 state 前缀扩展，读出仍取末位字母位——零改动契约不受影响）；不做 cross-attention adapter；不追从头联合预训练（资源边界如实声明）。

## 验收

- spec scenarios 全过（5 场景）+ 加/去图对照 run + 无图对照列入报告
- 一级 DoD 关联：② 多模态 tiny run + 无图对照
