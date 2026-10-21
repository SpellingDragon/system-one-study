# Teacher P2 · System-One Study 正式版（production-full）—— 昇腾 910B 云端执行版

> 多级变更：一级司编排（本件），spec 下沉至 12+1 个二级子变更（`changes/p2-01..p2-13`）。
> 2026-10-05 v3 重写：执行环境大转向（Mac MPS → 云端昇腾 910B ¥20/h）+ StartLux 版权边界 + 双阶段并行编排。

## Why（背书链）

1. **课程两阶段目标**：阶段一（已归档 `teacher-p1-scratch-mps`）从 0 到 1 造出引擎；本变更交付阶段二——**基于 Qwen3.5-0.8B 的受控超越**（同基座纯训练配方实验，D10）。
2. **用户三条新约束（2026-10-05 定）**：
   - **版权边界（D11）**：训练基座必须 Qwen3.5（Apache 2.0）；StartLux（CC BY-NC）仅用作蒸馏教师打分、RL verifier、对照评测推理——禁止任何训练初始化/微调基座/权重再发布。
   - **昇腾优先（D6 改）**：训练与推理**优先完全使用昇腾卡**（云端 910B 单卡 ¥20/h），算子充分参考 **TileKernels**（2026-09-30 已加昇腾后端，标注 950——与 910B 兼容性 W0 探针实证）；**训练栈主路线=全 TileLang 自研**（用户决策），torch_npu 底座为在册回退。
   - **云端执行（D12/D13）**：阶段二不在本地训练；本地只做数据准备/开发/CPU 对拍与一阶段 MPS 长跑，正式训练+推理上 910B；成本探针化（先短租实测再定档）。
3. **创新定位（D10 重估）**：方法学（三栈受控消融）+ 架构增量（1M 三件套）+ 工程创新（端侧→昇腾自研算子栈，吃自己狗粮的完全体）。

## What Changes（13 域）

| 域 | 内容 |
|---|---|
| p2-01 backbone-assets | Qwen3.5-0.8B 权重获取+字母接缝校验+**权重布局转换（HF→自研栈张量布局）** |
| p2-02 teacher-adapters | 三教师：StartLux-4B（蒸馏/RL，魔搭实证）/ GLM-5.3-Flash API（视觉伪标，key=ZAI_API_KEY）/ Qwen3.5-4B 备选；parquet 缓存；**打分负载上 910B** |
| p2-03 eval-registry | 评测登记：Intern-Decision train/test + typed-decisions + MMBench/OCR；**评审 harness 适配 NPU 推理** |
| p2-04 baselines-dual | laya 本地 + StartLux-0.8B **对照评测（D11 白名单内，纯推理不训练）**；双基线亲跑同 harness |
| p2-05 prod-sft | SFT（LoRA r16+GC）**全 TileLang 自研栈 on 910B**（torch_npu 回退在册） |
| p2-06 prod-opd | on-policy 蒸馏（per-token reverse KL）；**教师同卡在线打分内存账** |
| p2-07 long-context | 原生 262K + prefix_cache + needle；1M=三件套边界实验（910B 64GB 显存富余） |
| p2-08 multimodal-tower | Qwen3.5 自带视觉塔启用（冻结）+ GLM-V 一站式 processor + API 伪标蒸馏 |
| p2-09 chinese-track | 决策化中文 + 配比消融 + laya 崩溃列 |
| p2-10 serving | 跨后端服务（**新增 NPU 后端**）；延迟/显存画像 |
| p2-11 prod-rl | RLCD（log score）+ 分域 verifier 双通道；**910B 训练** |
| p2-12 tech-report | 受控实验报告 + **成本报表（¥20/h 记账）** + 复现指南 |
| **p2-13 ascend-runtime（新增）** | **910B 运行时与自研算子栈**：环境脚本/方言探针/算子移植（linear·rope·attn·GDN·LN·读出，参考 TileKernels）/梯度对拍/性能基准/成本护栏 |

## Capabilities（对应二级 spec）

12+1 个能力域 spec 下沉至各二级变更（CLI 只跟踪本一级件）；G1 决策质量 / G2 校准 / G3 中文 / G4 长文 / G5 多模态（同基座对照）/ G6 跨后端（含 NPU）/ G7 三栈消融 / G8 scaling / G9 昇腾算子栈（新增验收面）。

## 与阶段一的并行编排（D13）

本地轨 A（P1 收官）∥ 本地轨 B（P2 数据/开发，CPU 为主避让 MPS）∥ 云端轨 C（开卡后探针→移植→正式训练；C5 gate 须待 A 完成）。详见 tasks.md mermaid 双泳道。

## 成本口径（D12）

¥20/h 单卡：探针段（~2–4h）→ 移植/对拍段（~4–8h）→ tiny 冒烟（分钟级）→ gate 正式训练（探针实测吞吐后报价，超 **¥600 预算线熔断上报**）；全程 run notes 记时长/费用。
