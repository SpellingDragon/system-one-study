# Tasks: p2-12-tech-report [W4 · 依赖全部；只读]

> 三级结构：`### 工作项` → 孙任务 checkbox。spec：`specs/tech-report/spec.md`。

### 工作项 A 骨架与索引

- [ ] A1 `report/main.tex` 八节骨架（Limitations ≥3 失败分析引用 runs/*/notes.md）+ 编译通过 —— 验证：`bash report/build.sh`（tectonic 或 latexmk）exit 0
- [ ] A2 `runs_index.md` 自动生成脚本 + 对账（每数字行附 run-id，抽查目录存在） —— 验证：`python report/gen_runs_index.py` exit 0

### 工作项 B 表与边界

- [ ] B1 主表填充：DML vs laya vs startlux-0.8b vs jev-ref（六字段） —— 验证：主表无空格（`—` 显式）且每行 run-id
- [ ] B2 诚实边界清单：MPS-only / 1M 双列 / FP8 未做 / 三栈耗时实测 —— 验证：每条边界附 runs 链接（人工核验）
