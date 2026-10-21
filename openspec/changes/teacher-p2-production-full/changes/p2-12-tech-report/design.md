# Design: p2-12-tech-report

## 技术要点
- 八节骨架（PRODUCTION §9.2 表）：Abstract → Intro（claim–evidence 成对）→ Related Work（laya/Jev/StartLux/Intern-Decision 如实引用不藏竞品）→ Method（架构图+三栈管线图，mermaid 出图）→ Setup（数据 pin 表 + harness + 算力）→ Results（主表+消融）→ **Limitations（必填 ≥3 失败分析，引用 runs/*/notes.md）** → Appendix（runs 索引+复现指南）。
- `gen_runs_index.py`：扫描 runs/ 生成"图/表 ↔ run-id"对照；主表每数字行附 run-id，无 run-id 数字不得入表（脚本强校验）。
- 诚实边界清单（教师版特有）：MPS-only、1M measured/config 双列、FP8 未做、三栈 Mac 耗时实测——每条附 runs 链接。
- 编译：tectonic 或 latexmk；CI 不编译 PDF（重依赖），本地脚本 `report/build.sh` 出 PDF。

## 文件清单
| 文件 | 职责 |
|---|---|
| `report/main.tex` + `figures/` | 骨架与图 |
| `report/gen_runs_index.py` | 索引对账 |
| `report/build.sh` | 编译入口 |
| `report/tables/main.md` | 主表源 |

## 风险与回退
- [LaTeX 环境缺失] → 表格与索引均先以 markdown 产出（无 LaTeX 也可验收），PDF 为增强项。
- [数字与 runs 对不上] → gen_runs_index 对账失败即红——本域存在的意义。
