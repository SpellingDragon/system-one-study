# Proposal: p2-10-serving-systemone — 服务化（二级子变更）

> 父变更：teacher-p2-production-full（W2；依赖 p2-05 SFT 产物；与 06/07/08/09 并发；prefix_cache 联动来自 07）

## Why
G6 的最小落地（MPS-only 裁剪版）：HTTP `/v1/systemone` 服务——评测 endpoint 模式的靶子，"同 harness 数字必须从这个进程里出来"（评测与线上同源）。FP8/CUDA graph 已裁（父 Non-Goals）。

## What Changes
- 新增 `serving/server.py`：POST `/v1/systemone`（decide + decide_batch）+ `/health`
- 新增延迟测量协议（warm-up 20 + N 串行，mean/P50/P95；1 choice+1 noul+1 score 三问一次前向）
- prefix_cache 联动（命中计数可观测）

## 边界与依赖（不耦合声明）
- 依赖：p2-05（模型产物）；p2-07 的 prefix_cache（联动项，07 晚于此域时可先空挂接口）。
- **被依赖方**：p2-04/12（endpoint 打分与延迟报告）、p2-03（`--endpoint` 模式）。
- 接口面：`HTTP /v1/systemone + /health + /stats（命中计数）`。

## 验收
- spec scenarios 全过：`specs/serving-systemone/spec.md`（5 场景）
- 一级 DoD 关联项：① endpoint 评测可用
