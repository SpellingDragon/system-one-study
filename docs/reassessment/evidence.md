# 证据索引与验证记录

## 1. 基线和口径

本轮起点（2026-10-24）：

| 仓库 | 分支 | HEAD | 原有未跟踪项 |
|---|---|---|---|
| system-one | main | `b9c1dce019952068710dd83db1efad9d7fa84cd1` | `release/ascend/port910b/ta_defect_dossier.md`、`system_one_study_eval.agent.final.md` |
| tilelang-ascend | ascendc_pto | `1bdce50e13eba4d7d135033dc928f15f68ed98b7` | `examples/gdn_bwd/` |

这些旧未跟踪项未修改，也不作为本轮结论的唯一依据。报告中的“当前”限于上述基线；后续产品变动需要重新核查。

证据等级：**S**＝当前源文件事实；**H**＝历史原始日志；**V**＝本轮执行验证；**D**＝原始计划或外部文档声明；**I**＝推断。D 不是实测，H 不是当前版本复验；两个同源实现互相对齐也不是独立数学证明。

路径约定：下表未带仓库前缀的路径以 system-one 根目录为基准；`TA/` 为同级 tilelang-ascend 仓库。W11 为 `release/runs/1017-p2-13-oncard-window11-v2-acceptance/`。使用日志时保留批次，不对不同批次做无条件拼接。

## 2. 全局计划来源

| 编号 | 等级 | 一手来源 | 可支持的命题／限制 |
|---|---|---|---|
| G01 | D | `openspec/changes/archive/2026-10-03-teacher-p1-scratch-mps/{proposal,design,tasks}.md` 及十域四件套 | P1 教师版裁剪、归档账目；不代表后续正式档重验 |
| G02 | D | `scratch/GUIDE.md`、`scratch/PRODUCTION.md` | 学生课程目标与升级契约；需结合教师版裁剪及后续裁决 |
| G03 | D | `openspec/changes/teacher-p2-production-full/{proposal,design,tasks}.md` 及十三域四件套 | P2 三栈、扩展、云端路线、预算和授权变更 |
| G04 | D | G03 design D6、D12、D13（尤其尾部 10-07、10-08 裁决） | 渐进替换、C5 挂起、p2-13 主线、本地交叉编译；不能据旧“vehicle 可跑”保证当前完整模型可跑 |
| G05 | D | `openspec/changes/teacher-p2-production-full/changes/p2-13-ascend-runtime/{proposal,design,tasks}.md`、`specs/ascend-runtime/spec.md` | 头桩→改道→标量正确性／cube 性能→TA 路线；spec 中旧 partial 例外、fp32 对拍与 v3 新边界需协调 |
| G06 | D/H | W11 `HANDOFF.md`、`w11_fork_plan.md`、`notes.md` [144][155][158][159] | 收紧 dk/dbeta 的裁决与待验队列；性能归因、不可达论证不因写入笔记而成为事实 |
| G07 | D/H | `openspec/changes/audit-remediation-1010/`、`docs/{ci_baseline_triage,stats_plan,compliance}.md` | 既有审计与整改；其历史结论不覆盖本轮之后的变化 |

## 3. P1 核查入口

| 编号 | 等级 | 来源 | 命题范围 |
|---|---|---|---|
| P101 | S | `scratch/sys1/model.py`、`kernels/{backends,autograd}.py`、`eval/{predict,parity,report}.py` | 正式 forward 与独立 parity 路径、kernel 门控、报告 backend 标签 |
| P102 | H | `scratch/runs/1004-s2-decision-sft-17ac/`、`1004-s3-calibrate-a3a2/`、`1004-eval-report-1f94/` | tiny 链的 SFT／校准／评测；校准配置的数据路径与文字说明不一致，不能当独立测试收益 |
| P103 | H | `scratch/runs/1005-dc8c/`、`1006-s2-decision-sft-1e95/`、`1006-s3-calibrate-962f/`、`1006-eval-report-aa36/` | 后续正式模型链；正式评测 choice n=40、noul n=60，ECE 不改善 |
| P104 | H | `scratch/runs/1004-bench-tilelang-vs-torch-mps-1a18/`、`1004-bench-kernel-autocograd-30step-28ba/`、`1004-bench-infer-e2e-mps-ce64/` | 当时 M3 Pro／Metal 版本的负性能结果，不外推其他硬件 |
| P105 | S | `scratch/examples/repro_p1.sh`、`scratch/sys1/data/transcribe.py`、`.gitignore` | 复现前提与数据重建入口；原工作区成功不等于裸克隆成功 |
| P106 | S | `scratch/sys1/decision/`、`release/sys1/decision/` | 两阶段共享契约的内容比较；完整 diff 结果另记 V 项 |

## 4. P2 核查入口

| 编号 | 等级 | 来源 | 命题范围 |
|---|---|---|---|
| P201 | H/S | `release/runs/1005-p202-teacher-adapters-b3-realpath-b570/`、`production/teachers/verify_b3.py`（位于 release 下） | 4B 权重获取与 tokenizer 成功；完整模型加载被守卫阻止；scaffold 不是真教师 |
| P202 | H/S | `release/runs/1005-p201-backbone-assets-realpath-5d5c/`、`release/production/{assets.py,backbone/}` | 0.8B 载重与布局测试；不自动证明布局路径被训练消费 |
| P203 | H/S | `release/runs/1005-p2-05-dev-sft-qwen3-0-6b-kernel-50f1/`、`release/production/sft.py` | CPU 0.6B 替身、LoRA 旁路、scaffold 伪标，非正式 0.8B/NPU 实验 |
| P204 | S | `release/production/opd.py`、`release/tests/test_opd.py`、`test_train_axis_guard.py` | 实际 KL 方向、确定性选项截取、默认 axis 与守卫覆盖 |
| P205 | S/H | `release/sys1/eval/{registry,run,chinese}.py`、`release/production/configs/zh_mix_30.yaml` | train split 与中文配比是否真被消费；ID 不重合不等于语义去重 |
| P206 | H/S | `release/runs/1006-p2-08-a1-multimodal-probe-0224/`、`release/production/{data/mm.py,eval/multimodal.py}` | processor 探针与组件；不同于加／去图决策收益 |
| P207 | H | `release/runs/1006-p2-07-a2-needle-curve-v3-1035/` | 零样本 MPS：8K 两题、32K 四题、128K 一题；不能证明 1M 能力 |
| P208 | H/S | `release/runs/1006-p210-serving-local-half-c5d6/`、`release/serving/{server,prefix_cache}.py` | CPU 服务端真实消费者与缓存对照；不等于 NPU 后端验收 |
| P209 | H/S | `release/production/baselines/compare_table.md`、`release/sys1/eval/baselines/` | 基线样本量、gate 字段含义、公平比较条件 |
| P210 | H/S | `release/runs/1010-p2-11-dev-rl-qwen3-0-6b-torch-bf64/`、p2-11/p2-12 tasks | 探索记录与零勾选；运行目录不能替代缺失的可复跑实现 |

## 5. Ascend/GDN 核查入口

| 编号 | 等级 | 来源 | 命题范围 |
|---|---|---|---|
| A01 | S | `TA/examples/gdn_bwd_v3/{gdn_bwd_v3,test_gdn_bwd_v3,example_gdn_bwd}.py` | 真实原生路径、dtype、tail、golden 开关与六梯度组装 |
| A02 | S/H | `TA/examples/gdn_bwd_v2/gdn_bwd_v2.py`、W11 notes [155][158][159] | BK 路径隔离证据；`301bfde0` 尚待 NPU 验证，不称真修已经成立 |
| A03 | H | W11 `artifacts/w11car/20261023T2353Z/w11car_rsa.log` | B1/H16/T8192/DK128/DV128/C64 六项 rel<5e-3；同批 CONTROL 存在 FAIL，目标行不等于整批验收通过 |
| A04 | H | W11 `artifacts/w11car/20261024T0033Z/w11car_rsa.log` | H1/T128 dk=8.049e-3；H1/T8192 dbeta=6.116e-3；对应控制形与输入见原日志 |
| A05 | H | W11 `artifacts/w11car/20261023T2343Z/w11car_wsc.log` | DK128/BK64 与 BK128 的 A/H_in 对照；不是对全部分块结构的普遍归因 |
| A06 | S/H | `TA/examples/gdn_bwd_v2/_process/_realshape_time2.py`、W11 notes [143] | 暖机 host 计时；17ms/chunk、倍率及舍入范围，不能拆出纯 launch 开销 |
| A07 | S/D | `TA/examples/gdn_bwd_v2/_process/_ref/chunk_o_bwd_pr1802.py`、`release/ascend/kernels/reconcile/Q6_CHUNKED_V2_DESIGN_HANDOFF.md` | 本地参考快照的局部反向接口与适配否决；不代表线上 PR 当前状态 |
| A08 | S | `release/ascend/{autograd_asc,train_step,train_step_real}.py`、`kernels/{gdn_kernel,gdn_asc}.py` | 正式 Ascend 接入与 v3 候选的分离 |
| A09 | S/H | `TA/examples/gdn_bwd_v3/test_gdn_bwd_v3_host.py`、`TA/examples/gdn_bwd_v2/_process/_golden_xcheck.py` | CPU 数学组装与两套 golden 一致性，非 NPU 证明 |

## 6. 本轮验证记录

- V00：两仓 `git status --short`、`rev-parse HEAD` 已核；release 解释器为 `release/.venv/bin/python`，torch `2.14.1`、pytest `9.1.1`。收尾再核：仅 docs 变更（新增 `docs/reassessment/`、`docs/learning_path/`，修订 `docs/teaching_gdn_bwd_verification.md`），scratch/release 产品与两仓历史件未动。
- V01：`git rev-parse HEAD:scratch/sys1/decision HEAD:release/sys1/decision` 同为 `d199c4c88f54bc7069c572b7a440970c7ee099d6` ⇒ 已跟踪内容一致；目录直比仅 `__pycache__` 二进制差异，不作为契约漂移证据。
- V02：release 首次收集期失败系当前目录 tilelang 源码遮蔽 venv 包（env.py 断言 build 路径不存在）；`PYTHONPATH=release` + `python -P` 后 `tests/test_train_axis_guard.py` **7 passed**。未改环境与依赖。
- V03：scratch 解释器实导入 `scratch/sys1`；`tests/test_calibrate.py` **11 passed**（1.63s）。
- V04：直接 import 核实 `production/opd.py` 默认 `axis='quality'`；`reverse_kl(p=[.9,.1], logp=log[.5,.5])=0.3680641055`，与 `Σ p(log p−log q)` 手算一致（反方向 0.510826）。
- V05：教材示例运行器 `docs/learning_path/verify_examples.py`：第 2/3 章 4 块、第 4/5 章 6 块、第 6 章 2 块、勘误后 GDN 教材 5 块，各重复 10 次输出稳定（共 17 块/170 次）；链接检查覆盖 learning_path、GDN 教材与 reassessment 目录（含 URL 解码）。
- V06：TA `test_gdn_bwd_v3_host.py` worst=6.504e-15（用例 B=H=1/T256/C=DK=DV=64/g=0/h0=0）；`_golden_xcheck.py 512 64` worst=3.135e-15。均为 CPU 数学一致性，不外推设备精度。
- V07：`check_example_sync.py` SYNC-OK 776/776；但锚点自 `gdn_v3_ab` 起，且 grep 证实 `example_gdn_bwd.py:124,137-140` 内联 solve_a 仍为 fill+GM→UB→L1 旧形态（源码事实）。
- V08：python3 实测 `max(0.0, nan)=0.0`，rsa 逐 head 聚合可被 NaN 掩蔽；`_realshape_acc.py` main 用例表仅 H=1 扫描，无 H16 目标形（源码事实）。
- V09：`scratch/runs/1005-s1-mps-main-dc8c/config.yaml` 亲验 d=512/L=12/heads=8/ctx=1024/max_steps=2000。
- V10：计划中的独立 CodeReview 子代理因配额错误（code 112）派发失败，未产生复核结果；已由主评估者直接复验补偿（清单见 05 §4），建议有额度时补一次。
- V11：2026-10-24 用户裁定五项取向（05 §5 裁定记录：1-C 双轨 / 2-B 保留判据补性能锚 / 3-A 替身纪律且教师选型开放 / 4-C on-policy 选项蒸馏 / 5-C 支持面分拆）；据此更新 05（R5/R9/R10 标记、§3 路线定向、§5 改裁定记录）与 README；新增教学点 learning_path 06 §6.9（哑 token 状态不变实验，复跑 10 次通过，第 6 章共 3 块/30 次）。
- V12：裁定落地变更 `openspec/changes/adjudication-remediation-1024/`（proposal/design/tasks 三件套，勾选归编排者）派发三个 openspec-apply 执行单（A 产品/B 测量面/C 账目，白名单互不相交）。编排者五查复跑实证：① A 组 `PYTHONPATH=$PWD/release release/.venv/bin/python -P -m pytest release/tests/test_train_axis_guard.py release/tests/test_opd.py -q` → **50 passed, 1 skipped**，`opd.default_cfg()` 实测 `axis='train'`、`resample_every=50`（负锚 `test_r14_monkeypatch_opd_default_turns_red` 在内）；② B 组 sync 双段 → `SYNC-OK 776/776` + `SYNC2-OK 3 个内联前置件逐行一致`、exit 0；`_pad_invariance.py` → `PADINV worst=9.641e-17 verdict=PASS`、exit 0（负锚 PADINV-NEGFAIL/NEGSTATE 实触发）；③ C 组锚 grep 全命中（T3a×5、统计边界/裁定向注记 L140/L142、勘误 L14、spec L4 自带+备选、R9 件 12834B、design D14 @L202）；`git diff --numstat` 证实既有账目文件仅增行（p2-13 tasks +4/0、p2-08 tasks +1/0、design +9/0；spec +1/1 为授权例外）。两仓改动面与白名单一一对应，零 git 提交。B5 实测新事实：末尾 pad+仅有效位损失时 g_pad 违约对有效位梯度零影响，约束对象是状态链/终态——已补记入 R9 设计件 §1.6。遗留待裁：rsa 探针退出码语义（需先核 w11_car.sh run_leg 是否消费退出码）；C6 挂起（资源前置）；B1/B2/B4 上卡复验归阶段二首车。
- V13：CI 基线红项归属（快照 e507822 与本次 5fe2253 的 job 结论逐项一致：release-gates 绿，cpu-gates/mac-gates 红）⇒ 红因属快照遗留、非本轮引入：① `scratch/a2_neg_selfcheck.py` ruff I001/RUF100×3；② 快照到离 `scratch/runs/corpus_tiny/*.bin` 仅留 index.json，而两处测试的 skip 判据只看目录/index 存在 → FileNotFoundError。修复：ruff --fix（3 行）；skip 判据改为“index 所列 shards 齐全”（缺片与缺目录同属资产不可用，沿用仓库既有 skip-when-missing 口径）。本地 `bash scratch/tools/ci.sh` → **278 passed, 2 skipped, 41 deselected**，ruff/注释/编译三门全绿。scratch 冻结区的修复留痕于本行与提交信息。
- 研究子代理返回的测试数字（含 release 全量 340 passed）未由主评估者复跑，不列为本轮全量验收；未取得原始结果的数字不写入验收摘要。
- 第一批只做计划、源码与历史证据盘点；全程无训练、无上卡、无下载、无付费 API、无 git 写操作。

后续批次在这里追加实际命令、退出状态、计数与限制，不覆写历史运行文件。
