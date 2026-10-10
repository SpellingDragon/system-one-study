# AGENTS.md（System-One Study）— 本仓的四目录布局与 agent 资产使用说明

> 面向在此仓库工作的 AI coding assistant（与人类协作者）。改代码前先读本文。

## 布局总览

```
<repo-root>/
├── AGENTS.md            # 本文
├── README.md            # 根级导航（手册与各阶段详情在 scratch/release 内）
├── .agent/              # ★ 本项目沉淀的 agent 资产（skill / sub-agent / rules 用法）
├── openspec/            # ★ 变更计划唯一真源
├── scratch/             # ★ 第一阶段（P1 学习版）冻结主场：主路径结构 + 全部产物
├── release/             # ★ 第二阶段（P2 正式版）主场：从冻结基线开工，产物落于此
├── refs/                # 参考仓克隆地（refs/clone.sh，产物 gitignored）
└── .github/ .githooks/  # CI 与本地 hook（当前指向 scratch，见下）
```

## 阶段工作规则（防两阶段互污）

| 规则 | 说明 |
|---|---|
| **scratch 冻结只读** | P1（teacher-p1-scratch-mps）已归档：`scratch/` 内代码原则上不再修改；修复性改动须在 openspec 归档件或 runs notes 中留痕 |
| **P2 在 release 工作** | `release/sys1` 是从 scratch 复制的冻结基线（P2 计划声明"decision/ 零改动"）；P2 实施的工作目录是 `release/`，产物（runs/bench）落 `release/runs`、`release/bench` |
| **openspec 计划里的相对路径** | P2 计划文档中的 `sys1/...`、`production/...` 均以 `release/` 为基准解析 |
| **venv 各自独立** | `scratch/.venv`（P1 用，已修复 editable）；`release/` 启用时自建（`python3.12 -m venv .venv && pip install -e ".[dev]"`） |
| **手册位置** | GUIDE.md / PRODUCTION.md 在 `scratch/`（属 P1 主结构）；P2 工作时以 `../scratch/PRODUCTION.md` 为手册 |

## .agent 资产清单与用法

### `.agent/skills/openspec-multilevel-planning/`（v4）

本项目全流程所沉淀的规划-执行-验收方法论，三文件：

| 文件 | 用途 | 何时读 |
|---|---|---|
| `SKILL.md` | 索引：六阶段工作流、核心原则、反模式 R 系列（编号以本 skill 为唯一源，引用处不写死区间） | 接手任何"制定/执行/评审多级变更计划"任务时 |
| `review-checklist.md` | 规划防线：覆盖性 5 查（目标-任务矩阵/消费者枚举/档位覆盖/性能主张/真实路径）+ 一致性检查 | 计划产出后、执行前 |
| `apply-orchestration.md` | 执行防线：派发八要素、**完成度四查与重派三型**、计划缺陷回写、验收双路法 | 执行任何 openspec apply 时 |

**加载方式**：新会话/其他工具（Claude Code、Codex 等）直接把 `.agent/skills/openspec-multilevel-planning/SKILL.md` 作为首读文档即可。

### sub-agent 定义与用法（本项目实证过的派发模式）

| 资产 | 位置 |
|---|---|
| `openspec-apply` 子智能体定义（144 行，含自主决策规则） | [`.agent/agents/openspec-apply.md`](.agent/agents/openspec-apply.md) |

宿主内置的 agent 类型配以下固定用法（详见 skill 的 apply-orchestration §二/§四）：

| sub-agent | 用途 | 派发要点 |
|---|---|---|
| `openspec-apply` | 实施二级子变更（每域一个） | prompt 八要素：四件套路径/venv python 绝对路径/已就绪资产情报/写入白名单/验证纪律（exit 0 才勾）/质量门自检/禁 git/探针回退分支 |
| `CodeReview` | 验收评审的代码层（唯一一个） | 给足范围界定（git status/diff）、契约清单、假绿/数值/安全重点、分级输出 |
| 编排者（主 agent） | 集成验证、跨域缺陷裁决、完成度四查、计划回写、账目勾选 | 不可下放的职责见 skill apply-orchestration §三–§六 |

**完成度四查**（每波 agent 返回后必做）：勾选真实性（复跑验证命令）/ 产物盘点（白名单文件与 spec↔测试映射）/ 数字溯源（汇报数字逐个实测）/ 遗漏检测（任务 vs 勾选 + 覆盖矩阵复查）→ 未完成者归因三型（agent 未完成/做错/计划缺任务）后按三型重派（复跑验证型/补缺型/修复型）。

### `.agent/rules/`（always-on 通用工作纪律）

| 文件 | 用途 |
|---|---|
| `common.md` | "One For All" 通用工程纪律：第〇层元规则（断言即负债 / 结构不得自证 / 边界跟随语义单元）、第一层操作纪律（问题分析/数据流/变更/抽象/回归/文档）、第二层工程实测表 |

frontmatter 标 `alwaysApply: true`，宿主自动注入、长期遵循；其他工具（Claude Code、Codex 等）把它作为会话首读的纪律底盘。内容为语言中立表述，不绑定具体仓库。

## CI 与 hook

- GitHub Actions（`.github/workflows/ci.yml`）与 `.githooks/pre-commit` 当前指向 `scratch/tools/ci.sh`；**P2 主场切换时**改为 `release/tools/...`（release 启用时同步建立自己的 tools/ 或调整路径）。
- 本地启用 hook：`git config core.hooksPath .githooks`（在仓库根执行）。

## 快速上手（新会话）

1. 读本文 + [.agent/rules/common.md](.agent/rules/common.md)（通用工程纪律） → 2. `openspec list` 看活跃变更 → 3. P2 任务读 `openspec/changes/teacher-p2-production-full/`（一级 tasks.md 是编排清单）→ 4. 按 `.agent/skills/.../apply-orchestration.md` 的协议派发执行 → 5. 工作目录 `release/`，验证命令一律 `release/.venv/bin/python`（启用后）。
