# Proposal: teacher-p1-scratch-mps — 一阶段教师参考答案（from-scratch + MPS-only）

## Why

仓库当前是纯规划骨架（`dmlaya/`、`learning/` 无一行实现代码），学生无从对照。需要一份**教师版参考答案**：在 main 分支上把 GUIDE 一阶段（学习版）的核心基础流程完整跑通一遍，作为评分对照与排坑先行。同时，本次实现过程（openspec 变更 → 子领域拆分 → 孙任务执行 → 可验证落地）将作为《Agent Driven Software Engineering》课程的典型参考案例，故过程本身须可追溯、每步可验证。

约束边界（教师版裁剪，对照学生版 GUIDE）：

- **不做跨后端**：TileLang 内核仅实现 MPS 方言（`_kernel`/`_mps` 两件套），不做 `_cuda`/`_asc`；`backends.py` 简化为 MPS 单后端探测分发。
- **Mac 冒烟档**：S1 从零预训练以本机 MPS 为主战场（约 30–50M 参数 × 数亿 token 级语料），目标是流程正确 + 学习曲线可见，非模型质量。
- **注释规范完整遵守并示范**：GUIDE §6.1 中文注释 + 白话段落 + 文件头三件套，`tools/check_comments.py` CI 门对教师版代码同样生效。
- M2 长上下文支线不单列（因果滑窗 attn 内核本身属 M1；RoPE 扩窗课程归二阶段 1M）。

## What Changes

- 新增 `dmlaya/lang/`（S0 自训 BPE 分词器封装 + 26 字母单 token 校验）
- 新增 `dmlaya/model.py`（小型因果 decoder：因果注意力 + RoPE + MLP，Mac 冒烟档配置）
- 新增 `dmlaya/decision/`（`render.py`/`readout.py`/`temperature.py`/`wide.py`——两阶段同构桥梁）
- 新增 `dmlaya/calibrate.py`（类型温度 NLL 网格拟合 + ECE）
- 新增 `dmlaya/data/`（`schema.py` 样本校验 / `transcribe.py` 分类→typed 转写 / `pretrain_corpus.py` 中英混排语料流）
- 新增 `dmlaya/kernels/`（TileLang **仅 MPS 方言**：`gemm`/`add_ln`/`rope`/`attn_sw`（因果滑窗）+ `backends.py` MPS 单后端分发 + `testing/torch_ref/` 逐算子对拍）
- 新增 `dmlaya/eval/`（一阶段四轴：quality / calibration / parity / speed；分桶随机基线；runs 证据链接口）
- 新增 `learning/s0_tokenizer.py`…`s3_calibrate.py` + `configs/`（冒烟档与正式档两档配置）
- 新增 `runs/` 证据链记录（config/metrics.jsonl/notes.md schema 落地）
- CI：`tests/` 增补单元与集成测试；`@mps` 标记用例接入 `tools/ci.sh --device mps` 档

## Capabilities

> **结构说明（多级 specs）**：下列能力的 spec 已全部下沉至对应二级子变更目录（`changes/p1-NN-*/specs/<capability>/spec.md`）；一级 `specs/` 仅保留编排契约（orchestration）。二级子变更各含完整四件套（proposal/design/tasks/specs），孙任务在其 tasks.md 内。

### New Capabilities

- `data-schema`: 两轨共用的 typed 样本契约——校验/序列化/soft 分布 targets（`{state, questions, targets}` 超集 `/v1/systemone`）
- `tokenizer`: S0 自训 BPE（中英混排、特殊 token）+ 26 选项字母各为单 token 的强校验
- `decoder-model`: GPT 式因果 decoder 定义（配置化层数/维度/RoPE），权重自训、接口与二阶段"换脑"兼容
- `decision-program`: `/v1/systemone` 渲染 + 末位字母 logits 读出 + 类型温度 + >26 分组票选（`output_tokens=0`，右 padding 因果不变性）
- `pretrain-pipeline`: S1 下一词预训练——语料装配、训练循环、metrics.jsonl、采样冒烟验收（Mac 冒烟档）
- `sft-pipeline`: S2 决策 SFT——读出位 CE（soft target）、typed-decisions 类数据
- `calibration`: S3 类型温度拟合（NLL 网格）+ ECE before/after + 分桶随机基线报告
- `mps-kernels`: TileLang MPS 方言算子（gemm/add_ln/rope/attn_sw 因果滑窗）+ torch 参考对拍（fp16 err≤2e-2、argmax 一致）
- `eval-harness`: 一阶段评测唯一出口——四轴指标 + 分桶呈现 + run-id 溯源
- `run-notebook`: `runs/` 实验 lab notebook——config/metrics/notes 三件套 schema 与工具

### Modified Capabilities

（无——仓库尚无既有 spec，本变更为首批能力落地。）

## Impact

- **代码**：`dmlaya/`（lang/model/calibrate/decision/data/kernels/eval/testing）、`learning/`（s0–s3 + configs）、`tests/`、`runs/`
- **CI**：`tools/check_comments.py` 首次对真实代码生效（此前扫描 0 文件）；`tools/ci.sh --device mps` 首次有真实用例；GitHub Actions macos-latest 可承载 MPS 档（标注 self-hosted 可选）
- **依赖**：`torch>=2.4`（MPS）、`tilelang>=0.1.15`（Metal 方言 `tilelang.metal.language`）、`tokenizers`、`pyarrow`、`pyyaml`——均已列于 pyproject，无需新增
- **参考仓**：`refs/llms-from-scratch-cn`（S0/S1 教学主线）、`refs/laya/tl_kernels.py`（kernel 蓝本，attn 改因果）、`refs/StartLux-Decision`（decision 程序与校准范式，URL 由课程方提供——教师版以本地工作区副本为准）
- **不做**（防跑偏）：`_cuda`/`_asc` 方言、跨后端探测分发、昇腾/MPS 之外的设备矩阵、M2 扩窗课程、任何第三方预训练权重
