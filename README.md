# System-One Study

**这是一个学习项目**。它记录的不是"一个模型有多强"，而是**一个人如何从 0 到 1 学会造一个端侧决策引擎，并学会用受控实验证明它好在哪**——两阶段，一部教材，一份全程可审计的学习账。

> 门面守则（2026-10-10 按"三行必填"标准重写）：本页每个数字都带出处（run-id 或命令）与样本量口径，负结果与进展同强度写；"设计定了"与"跑出来了"分开写——凡替身、骨架、待卡处只写状态标签与进行时，不写完成时。逐数字溯源表：`openspec/changes/audit-remediation-1010/N_traceability.md`。

## 学习叙事：一幕一题

### 第一幕 · 从零学会造（`scratch/`，P1 学习版，已完成归档）

题目苛刻得不讲情面：**禁止加载任何第三方预训练权重**。于是一切都得自己来——

- 自己训 BPE 词表（16k，中英双料）→ 自己写因果 decoder（d=512 / L=12 / heads=8 / ffn×4 / lm_head 不共享；参数量**双口径：总 54.2M（54,213,632）/ 非嵌入 37.8M**——旧门面那句"~40M"是按非嵌入口径含糊说的，低估了总账 35%）→ 自己定义决策程序（末位字母读出 + 类型温度）→ 再亲手把 gemm/rope/attn/LN 写成 **TileLang 内核**并补上全套反向算子。注意力是标准 MHA：`sys1/model.py:205` 为 `qkv = Linear(d, 3*d)`，Q/K/V 同宽同头数，**没有分组共享的 kv 头**（旧门面把一个未实现的架构名词写了进来，已删词）；
- 终点不靠感觉：四轴评测一键复现（`repro_p1.sh`。诚实口径：它要求本机已备 XNLI/MASSIVE/星级评分三类语料再 `python -m sys1.data.transcribe` 转写，`bench/` 整目录不在 git 里，裸克隆会在阶段 0 体检 **exit 2 诚实拒跑**；而 exit 0 查的是"轴非空 + 溯源未豁免 + parity 未换人"，**质量分不参与退出码**——`sys1/eval/report.py:99-109` 的 note 自述）。最终正式档（run `1006-eval-report-aa36`）：**决策质量 choice +31.67pp（0.6500 vs 随机基线 1/3，choice k=3 子桶 n=40，全轴 n=100）/ noul +25.00pp（0.7500 vs 0.5000，n=60）双双越过参考线**——那条参考线是项目自写常数 `REFERENCE_MARGIN_PP = 10.0`（`sys1/eval/quality.py:39`），不是任何竞品基线；**parity argmax 一致 20/20**（20 样本；口径是 `sys1/eval/parity.py:74` 手工装配的 `KernelForward` 前向，不是 `Decoder.forward`）；**P50 5.8ms**（`e2e_p50_ms=5.817`，MPS eager 路径）。从零造出的小脑确实会做判断——但请把这些当**小样本、单种子**的方向性读数来读：本幕正式档只有一份 config、一个 seed，全仓没有多种子扫描或置信区间代码（第二幕五份 yaml 同样写死 `seed: 20261005`，`release/production/configs/gate08b.yaml:15`），扩样登记在 audit-remediation-1010 的 P2 统计补强项。

这一幕学到的比分数更值钱，四条刻在 runs 里：**字母必须单 token**（全计划最大单点——**第一幕自训 BPE 实测 A–Z=38..63** 连续、a–z=70..95，52 枚字母各占一格，`scratch/runs/1003-s0-bpe-16k-realedu-zh-en/tokenizer/vocab.json`；第二幕 Qwen3.5 基座分词器是**另一套**字母段 32–57，钉在 `release/tests/test_assets.py:192` 的 `range(32,58)` 断言上——两幕编号不得互抄）；**CPU 全绿不代表设备路径可行**（swap-brain 用例在 MPS 一跑就漏出跨设备泄漏）；**同配置双跑会把吞吐腰斩**（编排者自己踩的雷，写进了方法论）；**内核写出来不等于走得到，更不等于快**——`Decoder.forward` 无条件构造 visible 掩码（`model.py:386`），内核门在 `visible is not None` 时直接回退 eager（`model.py:211`），于是可微四连（qkv→rope→attn_sw→proj）经公开前向走不到，只有 MLP gemm 能触发；即便接得上，TileLang 0.1.15 的 Metal 后端热态**全面慢于 torch-MPS 2–6×**（run `1004-bench-tilelang-vs-torch-mps-1a18`），注入训练端到端 **−8.7%**（`1004-bench-kernel-autocograd-30step-28ba`：3,048 vs 3,315 tok/s）、推理端到端 **−12%**（`1004-bench-infer-e2e-mps-ce64`）。本幕内核的价值眼下在功能正确（对拍全绿）与通路可微，不在速度——"TileLang 加速"的成立条件在 CUDA 端与融合调优之后。

### 第二幕 · 学会造得更好（`release/`，P2 正式版，进行中）

题目换成了科研的口味：**基线与对手用同一个 Qwen3.5-0.8B 基座**（实测 StartLux-0.8B `model_type=qwen3_5`）——架构差异清零，胜负纯看训练配方。于是 P2 的全部设计都是"向最强处学，量化每一步值多少"。下面是每条叙事的**设计 + 现态**，现态一律可在 HEAD grep/复跑：

- **三教师分治**（设计定稿，实绩分档写明）：文本教师向对手家族的 StartLux-4B 蒸馏——**至今 dev 管线 7 条 SFT run 的教师身份全是 `StartLuxAI/StartLux-Decision-4B#scaffold-cpu`，即同架构、随机初始化、36.6M 参数的脚手架替身**（`production/sft.py:1348` `scaffold_teacher`；参数量出自 run `1005-p202-teacher-adapters-b3-realpath-b570` 的 `b3_payload.json` 字段 `scaffold_params_m: 36.6`）。真 4B 权重**已真下载但未真载入**：9,344,023,187 字节在场，完整载入被内存守卫拦下（同 run notes 记本机可回收内存 4395.1MB / 次跑 3009.7MB 不足以安全载入，按 D6 延后至云端 C3）——也就是说，仓库里目前**没有任何一份来自真 4B 教师的分布或伪标**，"文本向蒸馏"跑的是替身口。脚手架身份以 `#scaffold-cpu` 后缀与真教师缓存键分家、run 配置原样记录，披露本身合规，缺的是门面等强度——本行即补上。视觉向 GLM-5.3-Flash API 离线伪标（官方口径 320B-A18B MoE）：首波 14 次调用**仅 2 行落包、颜色题答对 0/10**，次波 11 次调用 10 行入包（同 run 档）；"几十元成本"是设计估算，仓库内无费用凭证。学生基座 100% Apache 2.0（StartLux 权重实为 CC BY-NC-4.0）——D11 是**本仓自订**的版权边界，不等于对外合规结论：CC BY-NC 教师输出用于训练学生的权利推演、以及 GLM API 服务条款分析都还没成文，已登记为 audit-remediation-1010 的 P2 论证义务；
- **三栈消融**：SFT（教师 soft target）→ 蒸馏栈（实现是**选项级 forward KL**，`production/opd.py:521-546` 的 Σ p_t·(log p_t − log p_s)；旧门面写的"逐 token reverse KL / on-policy / GKD 谱系"三个词都不准——不是 per-token、方向是 forward、top-k 是分数降序确定性截取且整集只跑一遍，GKD 意义下随训练滚动重采样的 on-policy 要点缺失；散度数学本身实现正确，错的是名字）→ RLCD（log scoring rule + 分域 verifier 双通道；"与 Rewarding Doubt（ICLR 2026）同构"是**文献关系**，不是交付状态）。现态：SFT **7 条 dev run 全是 qwen3-0.6b 替身 / CPU / tiny 档**（run config `stage: p2-05-prod-sft/tiny-cpu`，notes 自述"910B 正式档待 C5"）；蒸馏栈 **零训练 run**（p2-06 勾 3/5，全部证据是单测假件验编排）；**RLCD 尚未落地**（p2-11 勾 0/6，HEAD 无 `rl_rlcd.py`、无串链 `train.py`、无 `tests/test_rl.py`，蓝本 `refs/laya` 需联网克隆；本地另有 3 条 CPU torch 旁路 dev 探索 run——6 步 reward 发散早停，负结果照入库，非正式栈）。"每一栈的增益都上消融表"是目标而非账：**消融表现在一张数字都还没有**。
- **昇腾自研算子栈**：两态分开写，不许混。**编译绿**——接口 home 九件 `*_asc.py` 已 **target=ascend 编译 PASS（9/9）**，含补上的 `asc_fill_l1`→`set_l1_2d`，bisheng 出 `kernel.aibin` + `executable.so`（`release/ascend/kernels/reconcile/J_RESULT.md` 首行结论 + p2-13 tasks P1-4b）；GDN 反向由 partial 撤为 **kernelized**（`kernels/gdn_asc.py:43 BWD_STATUS = "kernelized"`，attempts/F 六梯对拍）；训练接线已通——`ascend/autograd_asc.py`（8 个 `autograd.Function`，闭合"内核无训练消费方"的 R11 缺口）+ `ascend/train_step.py` 混合栈一步在 host CPU 绿：**loss 0.911→0.824、15 枚梯度全非 None**，`tests/test_ascend_train_step.py` 8 passed、gradcheck 基线 20 passed 无回归。**数值待卡**——真机数值与单位 V1–V6 全部欠账（无一入账），cube 面 gemm_l1/dW 曾在真机报 **aicore exception 507015**（run `1010-p2-13-oncard-wave1-cc3d`），装填位序的结构修复已落地且本地编译/回归全绿（run `1010-p2-13-p11d-cube-fix-wave-0c72`），但**"数值是否脱困 507015"仍待一次 ≤8min 卡窗复验**；本地 CPU 对拍 20 绿是语义尺子，不是设备分数。且生产训练口 `sft.py` 的 kernel 开关眼下只接管自研 LoRA 旁路，backbone 前向仍是 HF 实现（`sft.py:33-38` 被否方案四自述），910B 正式训练待 C5——所以自研内核与 910B 训练链的接入状态是"**编译通、数值欠、卡未上**"，不是一个已完成的整体。
- **边界实验**：1M 窗口三件套（滑窗封顶 + 线性记忆适配器 + 门控短路）是**设计**（D7），落地约 1/3——只有因果滑窗成件（p2-07 C1），GDN 适配器无训练 run、门控短路零实现，**1M 档零实测**（needle 实测档位只到 8K/32K/128K）。第一幕的天花板恰是第二幕的靶子，两笔负账写实：**温度标定在正式档上是恶化的**——choice ECE 0.1175→**0.2267**（Δ+0.1092）、noul 0.1562→0.1682，标定温度顶到网格上限 5.0（`sys1/calibrate.py:35 GRID_HI = 5.0`；过程 run `1006-s3-calibrate-962f` 自身口径 0.1709→0.2129，与正式评测档分属两处，不矛盾），报告"改善：否"（run `1006-eval-report-aa36/report.md:12`）；**零样本 32K 召回 0.25 的分母只有 4 题**（1/4，run `1006-p2-07-a2-needle-curve-v3-1035`），128K 档 1 题，统计功效近零。旧门面只摘了 tiny 链校准改善的正例、把这条恶化含糊带过，属选择性采摘——此处按同强度补齐。

## 学习的方法论（第三层叙事）

这个项目同时是 **agent 驱动软件工程**的活教材：多级 OpenSpec 变更（一级司编排、二级司领域、三级司执行；规划审覆盖、执行审一致、验收审证据）、子 agent 五段战报与完成度四查、计划缺陷回写闭环（执行期暴露的每个缺口都改写回计划与 skill；**R 系列反模式全部来自真实翻车，编号以 `.agent/skills/openspec-multilevel-planning/SKILL.md` 为唯一源，本页不写死区间**，`AGENTS.md` 的旧区间随 audit-remediation-1010 的 P1 卫生批对齐）、`runs/` 证据文化（四件套入库、大权重按来源重建、**run 不删**——连三个空跑的曲线壳都留着：`1006-p2-07-a2-needle-curve-b46b / -e6c8 / -v3-e4ba` 的 metrics 为空仍入库，因为诚实比好看贵）。自家规矩也照到自己身上：规范写着"三行必填、失败也要记"，实测 **93 个 run 目录里有 77 个 notes 仍残留"待填写"模板头**（`grep -rl 待填写 scratch/runs release/runs --include=notes.md | wc -l`，2026-10-10；审计在旧 HEAD 上为 76/88）——这是流程失修而非数据造假，清账列在卫生批任务里，不靠本门面遮丑。

## 你会在这里得到什么

| 想要 | 去读 |
|---|---|
| 从零造决策引擎的完整参考实现 | `scratch/sys1/` + `scratch/learning/`（中文注释三件套，白话可懂） |
| 每步为什么这么做、踩过什么坑 | `scratch/runs/` 与 `release/runs/` 的 notes（结论行制度；注意"待填写"残留，见上节自查） |
| 大项目如何拆给多个 agent 并行做 | `openspec/changes/teacher-p2-production-full/`（**13 子域，81 条孙任务 / 57 勾（2026-10-10 现态）**：未勾集中在 p2-11 0/6、p2-12 0/4、p2-09 1/3、p2-05 5/8、p2-06 3/5）+ `.agent/`（skill 与派发模板）。"全勾对账"属于**已归档的第一幕**（`openspec/changes/archive/2026-10-03-teacher-p1-scratch-mps/`，10 子域孙任务 66/66 + 一级 14/14），第二幕远未收口 |
| 端侧推理/蒸馏的诚实数字 | 本页每个数字后面的 run-id 与样本量口径 + `release/examples/a2_needle_curve.py` 复跑；RL 栈**尚无数字可给**（p2-11 未起跑），别找 |

## 快速上手

```bash
# 第一幕（scratch）
cd scratch && python -m venv .venv && .venv/bin/pip install -e ".[dev]"
PYTHON=.venv/bin/python bash tools/ci.sh      # 四门禁
bash examples/repro_p1.sh                     # 需先自备 XNLI/MASSIVE/星级评分并转写，否则阶段 0 体检 exit 2 拒跑

# 第二幕（release）——注意：这里必须自己建 release 的 venv，旧门面漏了这一步
cd ../release && python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests -q -m "not integration"
```

最后一行的验收口径**以运行输出为准**，不引用任何历史快照数：本机实跑（Apple Silicon / macOS，2026-10-10）为 **326 passed, 3 skipped, 3 deselected**（含 audit-remediation-1010 R-P0-2 新增的 `test_train_axis_guard.py` 7 项）。同时如实记下这笔账的前置条件：`tests/test_backends.py` 的双重 `@pytest.fixture` 装饰器在修复前会让该命令**收集期硬错误、0 个测试被执行**（修复见同变更 R-P0-3）；Linux/CI 上的数以新增的 release job 为准——MPS-only 用例在 Linux 口径不同（审计容器内最佳 276 passed），跨环境数字不可互抄。

## 三条红线（学习项目的品格）

1. 第一幕 `learning/` 轨禁第三方权重；第二幕产出基座必须 Apache 2.0（StartLux 权重实为 CC BY-NC-4.0，仅限蒸馏/verifier/对照评测；蒸馏产物权利推演待补，见 P2 合规论证项）；
2. 一切跑分走 `sys1/eval/` 同 harness，双基线亲跑不引用卡面；训练口只准吃 `axis: train`（本波 audit-remediation-1010 自查出 5 份 yaml 与默认值曾写 `axis: quality`，而 quality 轴主力集登记 `split="test"`——即训测同集风险；已切 train 并加守卫测试 `release/tests/test_train_axis_guard.py`，历史 run 档原样不改）；
3. `sys1/decision/` 契约零改动——程序即宪法，三层守护。

> 授权面声明（**非法律意见**，权利推演与核验清单见 `docs/compliance.md`）：本仓 `LICENSE` 的 MIT 仅覆盖**本仓自写代码**，不授予任何第三方模型权重/数据集的权利——教师权重为 CC BY-NC-4.0（卡面明示商用需另行取得许可人授权），底座为 Apache-2.0（含上游 NOTICE 血缘）；再分发产物时需一并保留上游 NOTICE/署名要求。
