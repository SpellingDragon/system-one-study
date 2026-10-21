# audit-remediation-1010 · 第三方评审驱动的对账修复（独立变更，2026-10-10）

## Why
独立辩证评估（八维并行审计 + A-D 证据分级，报告全文归档于 `evidence/eval_report.md`，基准 HEAD `b4a399f`）判定：**项目账目诚实、门面系统性超前**——5 条阻断级叙事（RLCD 零实现、910B"完全体"、72 孙任务全勾、296 绿、4B 教师实为 36.6M 脚手架）全部可由一条 grep 复核失实；另有训测同集未爆雷（训练口 `axis: quality` 对着评测 test 集）、收集期硬错误（test_backends.py 双重 fixture）、零 LICENSE 等 P0 缺陷。编排者已在当前 HEAD（`dba5dc0`）逐条实证复核，全部属实。

## What Changes
按审计 §6.3 优先级清单执行"**声明与证据对齐 + 工程卫生门禁**"：
- **P0**：① README 全面"三行必填"式对账重写（按现态 `dba5dc0`，非评审旧快照）；② 训练口切 `axis: train` + 守卫测试（p2-05:49 原闸提升为当下执行）；③ 修复 test_backends.py 双重 fixture；④ 补 MIT LICENSE。
- **P1**：release CI job、[dev] 依赖补声明+钉版、R 编号口径统一（README/AGENTS/skill 三处）、卫生批（egg-info 出库、spec Purpose、run-notes 待填写清理、refs/clone.sh 钉 rev）。
- **P2**：统计补强方案记录（多种子/扩样，待正式档前置）、合规论证入档（CC BY-NC 蒸馏权利推演、GLM API ToS 分析）。

## 边界与依赖
- **依赖**：评审报告（已归档 evidence/）；当前 HEAD 现态（b4a399f→dba5dc0 增量见 design D3）；p2-05 tasks:49 既有闸（本变更是其提前执行，不重复立项）。
- **被依赖方**：无（p2-13 甲路、C5 挂起域均不被本变更阻塞；本变更也不替它们排产）。
- **接口面**：README.md / LICENSE / .github/workflows/ci.yml / release/production/{sft.py,configs} / release/tests——均为文档与工程门禁，不动 decision/、不动 scratch 代码。
- **禁止**：不补做 RLCD/910B 接入/正式训练（审计明示"正常未完成事项，非审计修复义务"）；不改 scratch/ 冻结代码（P1 层问题一律 README 措辞修正，见 design D6）；不删任何 run（负结果入库原则不变）。

## 验收（DoD）
1. README 每个数字可溯源（run-id 或命令），grep 旧失实词（296 绿/72 孙任务全勾/GQA/32..57/完全体/R1–R18）零命中；
2. `pytest tests -q -m "not integration"` 收集期 0 error（fixture 修复后全量可跑）；
3. 训练口守卫测试在（训 quality 轴即红）且 configs 训练路径零 `axis: quality`；
4. LICENSE 在库且与 refs/clone.sh 措辞一致；
5. CI 含 release job 且绿（或明示 MPS-only skip 口径）。
