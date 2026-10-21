# Proposal: p2-05-prod-sft — SFT 栈（二级子变更）

> 父变更：teacher-p2-production-full（W1；依赖 p2-01 载重 + p2-02 教师 + p2-03 registry train split）

## Why
三栈之首（冷启动）：载重 backbone + LoRA 读出位 CE + 教师伪标 soft 混合。其产物是 OPD/RL 与全部扩展域的起点模型——关键路径的第二段。

## What Changes
- 新增 `production/sft.py`：LoRA（r16/α32/dropout0.05/lr1e-4/warmup5%/cosine/2ep）读出位 CE，hard+伪标混合（50/50 可配），MPS fp16 + 梯度检查点
- 新增 `production/configs/{gate08b,scaling_06b}.yaml`（G8 两规模点）
- 新增 `tests/test_prod_sft.py`

## 边界与依赖（不耦合声明）
- 依赖：p2-01（backbone）、p2-02（伪标缓存）、p2-03（train split）、P1 decision/（读出复用）。
- **被依赖方**：p2-06 OPD、p2-07/09/10 扩展域（起点模型）。
- 接口面：`sft.py CLI + 产物（LoRA 适配器 + decision_config.json 模型目录）`。
- 铁律：读出 loss 复用 `decision/`（grep 断言无平行实现——同构证据）。

## 验收
- spec scenarios 全过：`specs/prod-sft/spec.md`（4 场景）
- 一级 DoD 关联项：① 三栈首环 run + 对照表 DML 列首填
