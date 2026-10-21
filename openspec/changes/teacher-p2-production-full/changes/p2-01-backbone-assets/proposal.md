# Proposal: p2-01-backbone-assets — 载重接缝（二级子变更）

> 父变更：teacher-p2-production-full（W0，关键路径起点；P1 decision 契约已冻结于 scratch/release 基线）

## Why

"同构换脑"的落点：预训练 backbone 替换一阶段自训小 GPT。接缝三校验是二阶段一切训练的前提——校验不过则三栈全白跑。**选型已定且有实证**（D9）：Qwen3.5-0.8B 一个 backbone 全包 G1/G4/G5（与基线 StartLux 同系保可比、原生 262K、原生视觉塔）；0.6B 备（LoRA+GC 可训已实测）。

## What Changes

- `production/assets.py`：`load_backbone(name)`——**Qwen3.5-0.8B 优先 / Qwen3-0.6B 备**（ModelScope 优先/HF 兜底，fp16 适配 MPS）+ 来源 repo/revision 入 run config
- 接缝三校验（载入时强校验，`SeamCheckError` 拒载）：① 字母单 token（复用 P1 `check_tokenizer`）② letter_rows 随 backbone hidden_size 重建 ③ chat template 思考关闭快照（Qwen3.5 前缀实测后固化）
- 窄接口适配层：暴露 `forward(input_ids, attn_mask) -> last_hidden` 与 `lm_head.weight`——backbone 内部对 decision/ 完全不可见
- **B2 真实路径验证**：Qwen3.5-0.8B 真拉真载一次（【P1 教训】mock 不算完成；0.6B 已预验 251s/1.2GB，Qwen3.5 首拉预期 ~1.7GB）

## 已就绪资产（执行加速）

- `release/.venv`（transformers 5.18/peft 0.21/modelscope）已建；Qwen3-0.6B 权重已缓存 `release/bench/ms_models`（真拉真载+LoRA fwd/bwd 已通，见 run `1005-probe-08b-lora-mps-feasibility-5a64`）；Qwen3.5-0.8B config/index 已拉取（视觉塔三件套+153 张量已证）——**本域 0.6B 路径大半预验，主攻 Qwen3.5 拉取与三校验**。

## 边界与依赖（不耦合声明）

- 依赖：P1 `sys1/decision/`（冻结，仅调用）；工作目录 `release/`。
- 被依赖方：p2-05/07/08（载 backbone）。
- 接口面：`load_backbone -> (model_like, tokenizer_info)` 满足 P1 窄接口。
- 禁止：改动 `decision/`（D4-4 硬拦）。

## 验收

- spec scenarios 全过（4 场景）+ B2 真实路径实测行入 run notes
- 一级 DoD 关联：换脑冒烟即 G7 同构证据前半
