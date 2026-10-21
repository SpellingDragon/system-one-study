# Proposal: p1-07-decision-program — 决策程序（二级子变更）

> 父变更：teacher-p1-scratch-mps（W1；依赖 p1-01 schema + p1-03 tokenizer 接口；stub 并行开发）

## Why
**全项目最重要的解耦点**（两阶段同构桥梁）：渲染 + 末位字母读出 + 类型温度 + 分组票选。读输出只依赖"末位 hidden × 词表 26 字母行"，与 backbone 内部无关——一阶段小 GPT 与二阶段预训练 backbone 共用本域全部代码。教师版必须把此契约做实（测试守护），二阶段"零改动复用"才有根基。

## What Changes
- 新增 `dmlaya/decision/{render,readout,temperature,wide}.py` + `RENDER_VERSION` 常量
- 新增 `tests/test_decision.py`（黄金快照/右 padding 不变性/无生成断言/温度/票选）

## 边界与依赖（不耦合声明）
- 依赖：p1-01（from_systemone 校验）、p1-03（tokenizer 接口：encode + 字母 id 映射）——开发期以 stub tokenizer 并行，集成期替换。
- **被依赖方**：p1-09（SFT 读出位）、p1-10（eval predict）、**p2 全部**（原样复用，硬契约）。
- 接口面：`from_systemone / render / readout / apply_temperature / wide_vote`。
- 禁止：import `dmlaya/model.py` 内部（父 design D1）。

## 验收
- spec scenarios 全过：`specs/decision-program/spec.md`（7 场景）
- 一级 DoD 关联项：① 链路、② 准确率（经本域读出）、⑤ CI
