# Proposal: p2-09-chinese-track — 中文专项（二级子变更）

> 父变更：teacher-p2-production-full（W2；依赖 p2-03 registry + p2-05 SFT；与 06/07/08/10 并发）

## Why
G3（laya 中文会崩 → 崩溃域证据）：中文决策化评测 + 数据配比消融。教师版把"laya 中文崩溃列"实测出来——这是报告里最有说服力的对照之一。

## What Changes
- 新增 CMMLU 子集 + CLUE（tnews/ocnli）决策化转写（choice/noul）+ registry pin
- 新增中文 SFT 配比档（30%/50%）消融
- 新增 G3 对照表：DML vs laya（亲跑，崩溃域如实记录）

## 边界与依赖（不耦合声明）
- 依赖：p2-03（评测集注册）、p2-05（SFT 栈复用）、p2-04（laya 基线行复用）。
- **被依赖方**：p2-12 报告（G3 表）。
- 接口面：`chinese 转写函数 + 配比配置 + G3 对照表产物`。

## 验收
- spec scenarios 全过：`specs/chinese-track/spec.md`（4 场景）
- 一级 DoD 关联项：② 中文对照表（含 laya 崩溃列）
