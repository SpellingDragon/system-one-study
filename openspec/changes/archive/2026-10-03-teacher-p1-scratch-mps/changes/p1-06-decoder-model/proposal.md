# Proposal: p1-06-decoder-model — 解码器（二级子变更）

> 父变更：teacher-p1-scratch-mps（W1；依赖 p1-03 的词表规格常数——开发可并行，集成需其合并）

## Why
S1 的模型载体：GPT 式因果 decoder。其窄接口（`last_hidden` + `lm_head.weight`）是"两阶段换脑"契约的另一半——二阶段载入 backbone 后 decision/ 零改动的必要条件。

## What Changes
- 新增 `dmlaya/model.py`：ModelConfig（d/L/heads/ctx/vocab/rope_theta/seed）+ pre-norm 因果注意力 + RoPE + MLP；save/load（safetensors + config.json + decision_config.json 同目录）
- 新增 `tests/test_model.py`（配置往返/因果性/窄接口/MPS）

## 边界与依赖（不耦合声明）
- 依赖：p1-03（仅词表大小常数入 config，无 import 耦合——接口常数依赖）。
- **被依赖方**：p1-08（S1 训练载体）、p1-09（SFT 起点）。
- 接口面：`ModelConfig / forward(input_ids, attn_mask) -> last_hidden / lm_head.weight / save/load`。
- 禁止：被 `decision/` 反向 import（架构不变量，父 design D1）。

## 验收
- spec scenarios 全过：`specs/decoder-model/spec.md`（5 场景）
- 一级 DoD 关联项：① 链路可跑、⑤ CI
