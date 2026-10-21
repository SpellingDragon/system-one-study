## ADDED Requirements

### Requirement: 报告骨架
`report/main.tex` SHALL 为 arXiv 风格骨架：Abstract → Intro（claim–evidence 成对）→ Related Work（如实含 laya/Jev/StartLux/Intern-Decision，不藏竞品）→ Method（架构+三栈图）→ Setup（数据 pin 表+算力）→ Results（主表+消融）→ Limitations（必填，≥3 失败分析）→ Appendix（runs 索引+复现指南）。

#### Scenario: 骨架完整性
- **WHEN** 编译 main.tex
- **THEN** 产出 PDF 且八节齐全，Limitations 含 ≥3 条引用 runs/*/notes.md 的失败分析

### Requirement: runs 索引自动生成
`report/runs_index.md` SHALL 由脚本从 runs/ 自动生成：每图每表 ↔ run-id 对照；数字无 run-id 不得入表。

#### Scenario: 索引与数字对账
- **WHEN** 运行索引生成脚本
- **THEN** 主表每个数字行附 run-id，抽查 3 个 run-id 目录真实存在

### Requirement: 诚实边界清单
报告 SHALL 列教师版裁剪与实测边界：MPS-only、1M 实测档 vs 配置档、FP8 未做、三栈在 Mac 的耗时实测；每条附对应 runs 证据。

#### Scenario: 边界可溯
- **WHEN** 查 Limitations/边界节
- **THEN** 每条边界有 runs 链接或 `—` 说明，无无据断言
