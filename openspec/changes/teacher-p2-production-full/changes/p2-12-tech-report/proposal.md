# Proposal: p2-12-tech-report — 报告（二级子变更）

> 父变更：teacher-p2-production-full（W4；依赖全部域——**只读不写**）

## Why
二阶段最终交付形态：arXiv 风格技术报告骨架 + runs 索引 + 主表 + 诚实边界。教师版报告 = "踩坑地图"的文本化，学生照图施工。

## What Changes
- 新增 `report/main.tex` 八节骨架（Abstract→…→Limitations→Appendix）+ 编译链
- 新增 `report/gen_runs_index.py` + `runs_index.md`（自动对账）
- 新增主表（DML vs laya vs startlux-0.8b vs jev-ref）与诚实边界清单

## 边界与依赖（不耦合声明）
- 依赖：全部前序域的 runs 与对照表产物（只读）。
- **被依赖方**：一级收尾（DoD ⑥）。
- 接口面：`report/ 目录产物（PDF + runs_index.md）`。

## 验收
- spec scenarios 全过：`specs/tech-report/spec.md`（5 场景）
- 一级 DoD 关联项：⑥ 报告骨架 + runs 索引 + 边界 ≥3 条
