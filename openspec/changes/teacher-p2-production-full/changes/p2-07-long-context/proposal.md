# Proposal: p2-07-long-context — 长上下文（二级子变更）

> 父变更：teacher-p2-production-full（W2；依赖 p2-01 载重 + p2-03 registry + p2-05 SFT 产物；与 06/08/09/10 并发）

## Why
G4（laya 无此能力，严格占优域）：1M 是用户裁决"全做踩坑"的扩展之一。教师版口径 = 工程链路真做 + Mac 实测档如实 + 1M 云端配置完备——"学生照此可在有卡环境真跑 1M"。

## What Changes
- 长文能力由 gate 同源 **Qwen3.5 原生 262K 承载**（D9：与 StartLux 同系保可比，不另载 Qwen3-Next/不自研）；1M 由 9B 级扩展点/云端配置档（**0.8B@1M 属可选边界实验·三件套方案（D7 定案）**：滑窗封顶 256K + GDN 线性记忆适配器旁路 + 门控短路——KV 封顶 7GB、无 RoPE 外推、≤262K 结构性走原路径（needle 回归守护）；云端训练、双指标（262K 回归 + 1M needle）入库，不作 gate；详见父 design D7）
- 新增 `sys1/eval/longctx.py`：合成 needle（多针、8K/32K/128K/256K 档、seed 固定）+ 召回曲线报告
- 新增 `serving/prefix_cache.py`：同 state 多问 / 跨请求 KV 前缀复用
- 长文滑窗路径（复用 P1 attn_sw；W 与层排布可配）
- 1M 云端配置与运行说明（measured/config 双列口径）

## 边界与依赖（不耦合声明）
- 依赖：p2-01（backbone 长上下文能力）、p2-03（needle 注册）、p2-05（起点模型）。
- **被依赖方**：p2-10 serving（prefix_cache 联动）、p2-12 报告（G4 图表）。
- 接口面：`PrefixCache / needle 评测 CLI / longctx 报告产物`。

## 验收
- spec scenarios 全过：`specs/long-context/spec.md`（6 场景）
- 一级 DoD 关联项：② 长文 tiny run + Mac 实测档记录
