# .agent/ — 本项目沉淀的 agent 资产

| 资产 | 说明 |
|---|---|
| `skills/openspec-multilevel-planning/` | 多级变更规划-执行-验收方法论（v4，三文件）。源出本仓 P1 全流程复盘，与用户级 `~/.qoder/skills/` 保持同步副本 |
| `agents/openspec-apply.md` | 子智能体定义（源自 `openspec init --tools qoder` 生成的 `~/.qoder/agents/openspec-apply.md`）：OpenSpec 变更实施专家，含自主决策规则；本项目每域实施均由它承担 |
| `rules/common.md` | "One For All" 通用工程纪律（第〇层元规则 / 第一层操作纪律 / 第二层工程实测表）；frontmatter `alwaysApply: true`，宿主自动注入，供任何 coding agent 长期遵循；语言中立表述，不绑定具体仓库 |

用法总说明见仓库根 [AGENTS.md](../AGENTS.md)；skill 正文：
- [SKILL.md](skills/openspec-multilevel-planning/SKILL.md) — 六阶段工作流 / 核心原则 / 反模式 R 系列（编号以 SKILL.md 为唯一源，此处不写死区间）
- [review-checklist.md](skills/openspec-multilevel-planning/review-checklist.md) — 规划防线（覆盖性 5 查 + 一致性）
- [apply-orchestration.md](skills/openspec-multilevel-planning/apply-orchestration.md) — 执行防线（派发八要素 / 完成度四查与重派三型 / 计划缺陷回写 / 验收双路法）

**维护约定**：skill 与 agent 定义更新时，本目录与用户级（`~/.qoder/skills/`、`~/.qoder/agents/`）双写同步（本目录是入库的项目级真源）。派发 openspec-apply 时，prompt 八要素见 skill apply-orchestration §二——定义文件是它的行为底盘，八要素是它的任务订单。
