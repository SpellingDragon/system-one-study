# orchestration · audit-remediation-1010

## Purpose
第三方评审（evidence/eval_report.md）驱动的对账修复编排：P0 四项并发（README 对账 / axis:train / fixture / LICENSE）→ P1 门禁与卫生 → P2 入档。范围=声明与证据对齐（design D1），不代产品域排产。

## 启动门
- P0 无前置即可发（三代理写面互斥：N=README.md；O=test_backends.py+LICENSE；P=sft.py+configs+新守卫测试）。
- P1 的 CI job 待 R-P0-3 收口（否则 job 起步即收集期红，基线失真）。
- P2 无前置，但 R-P2-1 显式标注"前置=训测分离完成"。

## 契约守护
- scratch/ 代码冻结不改（D6）；run 不删；历史 run/config 不回写（勘误走 append）。
- README 重写按现态 dba5dc0（D3 增量表），交付逐数字溯源表供编排者复核。
- 训练口守卫测试须含"注错变红"自证（R14）。

## DoD
见 proposal 验收 1–5；收尾 F1–F3 由编排者执行（集成复验 + run 台账 + 评审映射表）。
