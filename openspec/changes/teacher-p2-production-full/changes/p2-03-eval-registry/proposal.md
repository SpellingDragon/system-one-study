# Proposal: p2-03-eval-registry — 评测注册（二级子变更）

> 父变更：teacher-p2-production-full（W0，无前置依赖；在 P1 eval-harness 四轴之上扩展）

## Why
六轴评测与数据 pin 的单一真源：三件套 benchmark pin 到 commit、Uniform 锚点自检、中文/长文/多模态集注册——双基线（p2-04）与三扩展评测全依赖本域就绪。

## What Changes
- 新增 `sys1/eval/registry.py`：注册 + pin（Intern-Decision@2f81580 / jevbench@7ce310c7 / typed-decisions@f7a2487e / CMMLU / CLUE / MMBench-CN / LongBench-zh / 合成 needle seed）+ `export_versions()` + `fetch` 子命令
- Uniform 锚点自检（KL/TV/Brier 卡面复现）
- 六轴调度入口（`--axes`，缺数据标 n/a）
- 新增 `tests/test_registry.py`

## 边界与依赖（不耦合声明）
- 依赖：P1 eval 四轴（原样保留，`--axes` 调度）。
- **被依赖方**：p2-04 双基线、p2-07/08/09 三扩展评测。
- 接口面：`registry.fetch / export_versions / run --axes`。

## 验收
- spec scenarios 全过：`specs/eval-registry/spec.md`（5 场景）
- 一级 DoD 关联项：③ 双基线对照表的 pin 前提
