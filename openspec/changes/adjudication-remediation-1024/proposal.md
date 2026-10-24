# Proposal: adjudication-remediation-1024 — 五项裁定的计划修订落地

## Why（背书链）

1. 2026-10-24 全项目技术重评（`docs/reassessment/01..05`）完成，五项取向经用户裁定（`docs/reassessment/05-plan-recommendations.md` §5 裁定记录：1-C 双轨 / 2-B 保留判据补性能锚 / 3-A 替身纪律且教师选型开放 / 4-C on-policy 选项蒸馏 / 5-C 支持面分拆）。
2. 重评定级的高优先缺口需要立即修复，否则后续任何训练或验收结论可被污染：OPD 默认读评测轴且守卫未覆盖该入口（P204/V04）；验收探针存在吞 NaN 聚合与目标形用例缺失（V07/V08）；单文件 example 内联前置件与 v2 源码漂移而同步门不覆盖（A-F4）。
3. 本变更只承载"阶段一（本地 0 卡）"的修订与设计件；上卡数值复验（T3a 队列）属阶段二，不在本变更内。

## What Changes（三工作组）

| 组 | 内容 | 主要落点 |
|---|---|---|
| A 产品代码 | R4 训练轴堵漏（OPD 默认 train + 守卫扩全入口 + 注错自证）；裁定 4-C on-policy 选项蒸馏（采样移入训练循环周期性重采 + 命名/文档统一 + design D14 修订） | `release/production/opd.py`、`release/tests/`、一级 design.md（追加节） |
| B 探针与测量面 | R2 测量面修复（rsa NaN 硬门+目标形用例、boundary 重算参考、sync 扩段对账内联前置件、example 前置件同步）；裁定 5 的 pad 不变性本地对照脚本 | tilelang-ascend `examples/gdn_bwd_*`、run `scripts/check_example_sync.py` |
| C 账目与文档 | R1 p2-13 STATUS.md；裁定 2-B §9.1 形态判据统计边界；裁定 5-C 能力边界降级声明；R7 p2-08 spec 主路线修正与 B2 勾选勘误；R9 静态设计件 | openspec 账目（只追加勘误不改原文）、`release/ascend/kernels/reconcile/` |

## 边界与依赖

- 依赖：无（全部本地、0 卡、0 训练、0 下载）。
- 被依赖方：阶段二上卡复验（T3a/T3'/T2b/T5 与 v3t/ex 回归）依赖本变更 B 组的测量面修复；C5 点火依赖 R4+A 组与 R9 设计件。
- 接口面：见上表；跨仓白名单与勾选归属见 design D6。
- 禁止：不动 `gdn_bwd_v3.py` 产品面与 `scratch/`；不改历史账原文（只允许追加勘误/注记行）；不上卡、不跑训练、不下载权重、不做 git 提交；不把上卡项伪装成本地已验。

## 验收（DoD）

1. A 组：轴守卫全绿且注错自证必红；OPD 重采样有单测（两轮 top-k 可变）且 resample_every=0 保持旧行为；design.md 出现 D14 修订节。
2. B 组：sync 两段全绿（含内联前置件）；`_pad_invariance.py` 本地实跑 PASS（worst<1e-12）；rsa/boundary 的修改以编译+结构 grep 验证（数值复验明标"待阶段二上卡"）。
3. C 组：五个账目/文档产物存在且锚 grep 命中；p2-08 B2 保持原勾选但带勘误注记，补产物任务 C6 显式挂起。
4. 全程负结果如实入库；勾选由编排者按完成度五查后统一执行（design D6）。
