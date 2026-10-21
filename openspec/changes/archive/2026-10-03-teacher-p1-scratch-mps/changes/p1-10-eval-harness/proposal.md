# Proposal: p1-10-eval-harness — 评测出口（二级子变更）

> 父变更：teacher-p1-scratch-mps（W3；依赖 p1-09 产物接口 + p1-05 + p1-04 parity）

## Why
一切数字的唯一出口（M3）：四轴指标 + 分桶呈现 + run-id 溯源 + 一键复现。课程红线"所有跑分走 dmlaya/eval/"的落地面；二阶段六轴在此四轴上扩展。

## What Changes
- 新增 `dmlaya/eval/predict.py`（`--model`/`--endpoint` 双模式统一预测行）
- 新增四轴：quality（分桶）/ calibration（before/after）/ parity（kernel vs torch 路径）/ speed（P50/P95）
- 新增 `examples/repro_p1.sh` 一键复现（tiny 档 <20min CPU）
- 新增 `tests/test_eval.py`

## 边界与依赖（不耦合声明）
- 依赖：p1-09（模型目录接口）、p1-05（ECE/基线）、p1-04（parity 经 backends）、p1-07（predict 内部走 decision/）。
- **被依赖方**：p2-03 eval-registry（在此之上扩 pin 与六轴）。
- 接口面：`predict.py CLI / 各轴 score 函数 / 预测行格式 {id, answers, ms}`。

## 验收
- spec scenarios 全过：`specs/eval-harness/spec.md`（6 场景）
- 一级 DoD 关联项：① 一键串跑、②③④ 各轴数字、⑤ CI
