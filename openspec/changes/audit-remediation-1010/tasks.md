# Tasks · audit-remediation-1010（只列修复义务；波次=优先级，解锁=依赖入 main）

> 只修复：声明与证据对齐 + 工程门禁。不代 p2-11/p2-13/C5 排产（design D1）。
> 写面互斥：N=README；O=fixture+LICENSE+CI；P=axis 切换；Q=卫生批。并发派发，波后编排者集成复验。

## P0 波（并发三代理）

- [x] **R-P0-1 README 全面对账重写（代理 N）**：按"三行必填"标准重写——每个数字溯源（run-id/命令），失败同强度上门面（温度标定 ECE 恶化 0.1175→0.2267、TileLang Metal 内核慢 2–6×、训测同集自查）；删除/修正：GQA 词（实现为 MHA）、A–Z=32..57（P1 实测 38..63；32..57 属 P2 分词器并标注归属）、~40M（双口径 54.2M 总/37.8M 非嵌入）、"296 绿"（改实跑数+复现命令）、"13 子域 72 孙任务全勾"（81 条/57 勾现态）、"完全体"（改"9/9 编译绿、数值待卡"现态）、"R1–R18"（改指 skill 不写死区间）、教师实名（#scaffold-cpu 36.6M 脚手架 vs 真 4B 未载入）、GDN 反向现态=kernelized —— 验证：`grep -cE "296 绿|72 孙任务全勾|GQA|32..57|完全体|R1–R18" README.md` =0 + 交付逐数字溯源表
- [x] **R-P0-2 训练口切 axis:train + 守卫（代理 P）**：`sft.py` 默认与 5 份 yaml 训练口全切 `axis: train`（旧值留注释与理由）；新增守卫测试（训练配置 axis ∈ quality/test 轴即 fail；引用 registry 轴定义） —— 验证：`grep -h "axis:" release/production/configs/*.yaml` 无训练路径 quality + 守卫测试 passed + 故意注错变红自证（R14）
- [x] **R-P0-3 双 fixture 修复（代理 O）**：`release/tests/test_backends.py:26-29` 删重复装饰 —— 验证：`cd release && .venv/bin/python -m pytest tests -q -m "not integration"` 收集期 0 error（MPS-only 败/skip 如实计数）
- [x] **R-P0-4 LICENSE=MIT（代理 O）**：根目录 MIT 文本（2026，版权人=仓库署名者） —— 验证：`ls LICENSE` 且 `grep -n "MIT" refs/clone.sh` 措辞一致

## P1 波（P0 收口后）

- [x] R-P1-1 CI release job（代理 O 续）：ubuntu 自建 release/.venv + `[dev]` 补声明（modelscope/transformers/PIL）+ `pytest -q -m "not integration"`；首版 `continue-on-error: true` 采基线，红项三分归因（本变更引入/既有/环境）后定正式门 —— 验证：workflow 语法过（actionlint 或 push 后 run）+ 基线报告入 run
- [x] R-P1-2 R 编号口径统一（代理 Q）：AGENTS.md"R1–R14"、README（随 N 重写已消）、skill 实际最大编号三处对齐（以 skill 为源） —— 验证：三处 grep 计数一致
- [x] R-P1-3 卫生批（代理 Q）：egg-info 出库+gitignore；orchestration spec Purpose 去 TBD；refs/clone.sh 六 clone 钉 rev；run-notes"待填写"清零或显式豁免清单 —— 验证：`git ls-files | grep -c egg-info`=0；`grep -rl "待填写" scratch/runs release/runs | wc -l`=0（或豁免表行数）

## P2 波（记录/入档，不阻塞）

- [x] R-P1-4 release-gates 正式门化（首跑基线后立）：①资产缺失用例统一 skip-when-missing 口径（bench/ 按设计不入库，7 红应转 SKIP）；②test_backends 4 红在 ubuntu 归因（tilelang 探测/环境）；③归因后删 `continue-on-error` —— 验证：ubuntu run 全绿或红项全部归因入档
- [x] R-P2-1 统计补强方案入档（多种子/基线扩样/Wilson 区间/needle 扩题；前置=训测分离后） —— 验证：docs 或 design 增补节存在
- [x] R-P2-2 合规论证入档（CC BY-NC 蒸馏权利推演、GLM API ToS 分析——标注"非法律意见"） —— 验证：docs/compliance.md 存在且含两节

- [x] R-P1-5 授权面收口（代理 W 自纠所揭，新风险）：本仓 `LICENSE` 对外发 MIT（含商用），而教师为 CC BY-NC-4.0——若"模型输出非 Adapted Material"前提被推翻，等于**替第三方素材做超权限授权**。处置＝① LICENSE 不变（授的是自有代码），② README/NOTICE 一行明写"MIT 仅覆盖本仓自写代码，不授予任何第三方模型/数据集权利（教师权重为 CC BY-NC-4.0，底座为 Apache-2.0，含上游 NOTICE 血缘）" —— 验证：`grep -n "MIT.*仅覆盖\|not grant.*third-party" README.md`（或 NOTICE）命中 + 血缘声明与 registry/教师卡实际 license 一致

## 收尾

- [x] F1 编排者集成复验：DoD 1–5 逐项 + 全量 pytest 无新红 + grep 清单
- [x] F2 run 台账（修复前后对照：收集期 error→0、失实词→0、axis 守卫红→绿）
- [ ] F3 评审报告勘误核对：报告基于 b4a399f，README 修正后对"评审发现→现态"映射表入 run notes（不回写报告本身，D7）
