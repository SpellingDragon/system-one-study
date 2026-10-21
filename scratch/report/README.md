# report/ — 论文级技术报告 + 同行评审（PRODUCTION §9.2/§9.3）

> 二阶段最终交付。结构、必交图表、数字规范以 **PRODUCTION §9.2 表**为准；这里只放骨架与流程资产。

## 计划内容

```
report/
├── main.tex                # arXiv tech-report 风格（8–12页+附录），英文优先
├── figures/                # 必交六图：架构图 / 三栈管线图 / G8 规模曲线 /
│                           #   校准前后可靠性图 / G7 消融表 / 三后端一致性表
├── tables/                 # 主表模板：{metric, DML, laya, startlux-0.8b, Δ, (jev ref)}
├── runs_index.md           # 每图每表 ↔ run-id 对照表（答辩抽查用，由脚本从 runs/ 生成）
└── reviews/                # 互审：审稿单(1页: 摘要/优点/缺点/问题) + rebuttal + 修订说明
```

## 写作红线

1. **claim–evidence 成对**："更好/更快/更稳"必须带 Δ、方差、run-id；均值报 3 seeds 或 bootstrap 95% CI。
2. Related Work 不藏竞品：laya / Jev(TypeSafe) / StartLux / Intern-Decision / Decision Index / Naive-SWA / GLM-多模态 如实引用并写差异。
3. **Limitations 必填**且引用 ≥3 个 `runs/*/notes.md` 失败分析；负结果不删。
4. 附录含复现指南：`examples/` 一键脚本 + `bench/` pin 清单（自动导出版本号）。

## 流程

初稿(P6) → 班内互审（每组审 2 份）→ rebuttal 一轮 → final。**审稿质量计入成绩**（PRODUCTION §9.3）。
