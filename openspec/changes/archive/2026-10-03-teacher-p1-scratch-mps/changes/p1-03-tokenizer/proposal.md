# Proposal: p1-03-tokenizer — 分词器（二级子变更）

> 父变更：teacher-p1-scratch-mps（W0，无前置依赖；3C 真实语料可延后不阻塞 W1）

## Why
S0 是从零主线的第一环：决策读出依赖"26 字母各为单 token"，此约束必须从训练词表层面保证而非事后补救。教师版分词器同时是二阶段换 backbone 前理解 tokenizer 契约的教学样本。

## What Changes
- 新增 `dmlaya/lang/`：BPE train/save/load 封装（词表 8k–32k 可配）+ 特殊 token + `check_tokenizer()`（52 字母单 token 强校验）
- 新增 `learning/s0_tokenizer.py` CLI（`--corpus --vocab-size --stream-limit --out`）
- 新增 `tests/test_tokenizer.py`（内置小语料往返 + 污染词表被拒）

## 边界与依赖（不耦合声明）
- 依赖：仅 `tokenizers` 库；**零本仓依赖**。
- **被依赖方**：p1-06（词表规格常数）、p1-07（tokenizer 接口：encode/字母 id 映射）、p1-08（语料 token 化）。
- 接口面：`train / save / load / check_tokenizer`。

## 验收
- spec scenarios 全过：`specs/tokenizer/spec.md`（4 场景）
- 一级 DoD 关联项：① tokenizer 可编解码中英且字母单 token
