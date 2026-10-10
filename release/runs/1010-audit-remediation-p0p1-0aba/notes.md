# run: 1010-audit-remediation-p0p1-0aba

- 假设：评审 P0/P1 修复项可并发闭合且全部机器可验。
- 观察：五代理两波全绿，编排者逐项复验（grep/pytest/CI 实跑）；ubuntu 基线 290/11/28 归因清晰。
- 结论：声明与证据对齐达成（P0 四项+P1 三项）；门面失实词清零、训测分离落地有守卫、CI 覆盖主战场（基线期）；余 R-P1-4 门化与 P2 入档，均为记录级。

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。

## F3 勘误核对：评审发现 → 现态映射（基准 b4a399f → 现 HEAD，编排者复核非转述）
| 评审条目（级） | 现态 | 凭据 |
|---|---|---|
| 2.1.1 RLCD 零实现（A） | **维持"未实现"**（审计义务=声明对齐，不代排产 D1）；README 已改"p2-11 0/6 + 3 条 CPU 旁路 dev 探索 run（发散早停，负结果入库）" | `release/runs/1010-p2-11-dev-rl-*`（已 commit）、README:23 |
| 2.1.2 910B"完全体"（A） | 门面已改**两态分开**；且实质推进：接口 home **9/9 target=ascend 编译 PASS**、GDN 反向 kernelized、§12 ND2NZ 成套落地、**cube 数值经标量路本地闭环 5.22e-07**；真机数值仍欠 | tasks P1-4/P1-1f/P1-1k；`compile_all_asc` PASS=9；run oncard-wave2 |
| 2.1.3 "72 孙任务全勾"（A） | 已改现态计数（81/57）并注明"全勾"属归档第一幕 | `grep -cE '^- \[x\]'` 复跑；N_traceability.md |
| 2.1.4 "296 绿"不可复现（A） | **收集期硬错误已修**；现 ubuntu 硬门 `0 failed`、host 340 passed；数字改"以运行为准" | CI run 38076475622 ✓；pytest 原文 |
| 2.1.5 真 4B 教师缺失（A） | README 等强度实名（36.6M 脚手架 / 9.34GB 未载入原因 / 仓库内无任何真教师分布） | README:22；b3_payload `scaffold_params_m:36.6` |
| 2.4.1 训测同集（A，最危险） | **闸已落**：sft 默认 + 5 yaml 全切 `axis: train`，7 用例守卫含注错变红 | `tests/test_train_axis_guard.py` 7 passed |
| 2.2.3/2.6 CI 只门 scratch（A） | release job 上线并转**硬门**（连两绿后删 continue-on-error）；资产/设备类口径统一 | ci.yml `continue-on-error in job: False`；run 38070929040+38075461293+38076475622 |
| 2.5.1 零 LICENSE（A） | MIT 入库；并补授权射程声明（MIT 仅自有代码，不覆盖第三方权重；教师 CC BY-NC） | `LICENSE`、README:61、docs/compliance.md |
| 2.5.2/2.5.3 合规论证缺失（B/D） | 成文＋风险分级；**GLM ToS 原文未取得**（8 项全标待核验，未编造） | docs/compliance.md §2.2/§3 |
| 2.4.2/2.4.4 统计与选择性呈现（A） | 方案入档（种子集/n≥97 门槛/区间+参照线身份入报告口径）＋负结果已上门面 | docs/stats_plan.md；README 边界段 |
| 我（编排者）自纠 3 处 | ① 评审"完全体"判词后我又发现**判决件量错对象**（readout FAIL 实非产品件）；② `sliding` 我原记"预期保留 1 红"被下一 run 证伪→改判环境非确定缺陷；③ `card_run.sh` c_api 落点错是我的工装缺陷，致 wave2 首跑全灭 | run oncard-wave2 / D_determinism.md / Y_RESULT.md |
