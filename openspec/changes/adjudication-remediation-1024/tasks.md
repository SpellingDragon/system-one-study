# Tasks · adjudication-remediation-1024（阶段一：本地 0 卡；勾选由编排者统一执行）

> 依据：`docs/reassessment/05-plan-recommendations.md` §5 裁定记录 + 本 change design D1–D7。
> 白名单见 design D6；上卡数值项只做静态验证并明标（D7）。S1=`$REPO_ROOT`，TA=`$WORKSPACE_ROOT/tilelang-ascend`。

## 工作组 A · 产品代码（release/）

- [x] A1 OPD 训练轴切 train + 守卫扩全入口（R4）：`S1/release/production/opd.py` `default_cfg()` 的 `"axis": "quality"` 改 `"train"`（`eval_axis` 不动）；`S1/release/tests/test_train_axis_guard.py` `configured_axes()` 增加 `production/opd.py::default_cfg` 来源，并新增负锚用例（monkeypatch opd 默认档为 quality 时 `check_all()` 必须报该来源的红） —— 验证：`cd $REPO_ROOT && PYTHONPATH=$PWD/release release/.venv/bin/python -P -m pytest release/tests/test_train_axis_guard.py -q -p no:cacheprovider --basetemp=$PWD/.tmp_a1/adjudication-axis` 全绿且收集数 >7（原 7 + 新增负锚）
- [x] A2 on-policy 周期性重采（裁定 4-C / design D4）：`opd.py` 新增配置键 `resample_every`（默认 50，0=关闭保持旧一次性装配行为）；`train()` 优化器与调度器跨段连续，按段重调 `collect_onpolicy`（段种子 `seed+段号` 派生）；`S1/release/tests/test_opd.py` 新增用例：①resample_every=1 且 mock student 两轮 top-k 不同 ⇒ 第二段样本来自新策略；②resample_every=0 行为与旧一致（回归锚） —— 验证：`PYTHONPATH=$PWD/release release/.venv/bin/python -P -m pytest release/tests/test_opd.py -q -p no:cacheprovider --basetemp=$PWD/.tmp_a1/adjudication-opd` 全绿（收集数较改前增加）
- [x] A3 命名与文档统一（裁定 4-C）：`opd.py` 模块 docstring 定位改"on-policy 选项蒸馏（选项级 forward-KL）"；`reverse_kl` docstring 追加"命名为历史遗留，实算 KL(teacher‖student)"；`S1/openspec/changes/teacher-p2-production-full/design.md` 文末追加 **D14 裁定修订节**（D9 的"per-token reverse KL"表述由 D14 取代，D9 原文不改） —— 验证：`grep -n "D14" $S1/openspec/changes/teacher-p2-production-full/design.md` 命中 + `grep -n "on-policy 选项蒸馏" $S1/release/production/opd.py` 命中

## 工作组 B · 探针与测量面（TA 仓 + run scripts）

- [x] B1 rsa NaN 硬门 + 目标形用例（R2）：`TA/examples/gdn_bwd_v2/_process/_realshape_acc.py`：`one()` 对 `out[:6]` 逐项聚合前 `torch.isfinite` 检查，非有限打印 `REALSHAPE-ACC NONFINITE item=...` 并计 fail；`main()` 用例表首行加 `(8192, 128, 16)`（目标形）。数值复验待阶段二上卡 —— 验证：`python3 -m py_compile` 过 + `grep -n "8192, 128, 16" $TA/.../_realshape_acc.py` 命中 + `grep -n NONFINITE` 命中
- [x] B2 boundary 注入后重算参考（R2）：`TA/examples/gdn_bwd_v3/test_gdn_bwd_v3.py` `_inj()`：注入后的 do 参与参考计算（golden 由同一 do 派生），移除"用改前 golden"的口径错位。数值复验待阶段二上卡 —— 验证：py_compile 过 + 代码审读确认参考链含注入值（战报引用改动行号）
- [x] B3 sync 扩段对账内联前置件（R2/A-F4）：`S1/release/runs/1017-p2-13-oncard-window11-v2-acceptance/scripts/check_example_sync.py` 新增第二段比对：example 内联的 `gdn_bwd_v2_cumsum/solve_a/scan1` 三函数体 vs `TA/examples/gdn_bwd_v2/gdn_bwd_v2.py` 对应函数（提取"def 名(”到下一顶层 def，沿用现有归一化）—— 验证：改后首跑应输出 `SYNC2-FAIL`（负锚实证：solve_a 漂移被抓住）
- [x] B4 example 前置件同步到 v2 现行 idiom（R2/A-F4）：`TA/examples/gdn_bwd_v3/example_gdn_bwd.py` 内联 solve_a 改为与 `gdn_bwd_v2.py:254-262` 一致（GM→L1 直拷 + `init=(i==0)`，删除 `k_half`/`fill`/UB 中转；文件头精度说明同步修正）。上卡回归（v3t/ex）待阶段二 —— 验证：`PYTHONPATH=$TA/examples/gdn_bwd_v2 python3 $S1/release/runs/.../scripts/check_example_sync.py` → 两段 `SYNC-OK` + `python3 -m py_compile $TA/examples/gdn_bwd_v3/example_gdn_bwd.py` 过
- [x] B5 pad 不变性本地对照（裁定 5-C）：新增 `TA/examples/gdn_bwd_v2/_process/_pad_invariance.py`：T=100 与 T=130（C=64）构造有效输入，pad 至整倍数（哑 token：q 任意、k/v/do/g/β 补零且 **g 补 0**），对 `_np_chunk_fwd/_np_chunk_bwd`（fp64）比较有效位置六梯度 —— 验证：`cd $TA/examples/gdn_bwd_v2 && python3 _process/_pad_invariance.py` → `PADINV` 逐项 worst<1e-12、`verdict=PASS`（真实可跑，负锚：g 误填非零时脚本自证会 FAIL）

## 工作组 C · 账目与文档（只追加，不改原文行）

- [x] C1 p2-13 STATUS.md（R1）：新增 `S1/openspec/changes/teacher-p2-production-full/changes/p2-13-ascend-runtime/STATUS.md`：现行队列（T3a/T3'/T2b/T5 及判据）、已废/被替代（P1-1zh、fork 路线、旧 T 编号→现义映射）、参考件（HANDOFF/w11_fork_plan/05 裁定）—— 验证：`grep -c "T3a" STATUS.md` ≥1 且含"已废"节
- [x] C2 §9.1 形态判据统计边界（裁定 2-B）：`S1/.../p2-13-ascend-runtime/tasks.md` §9.1 形态行**追加注记**：判据保留；统计边界=反向扫描主体 kernel 数，不含三个前置件；host 运算与 launch 数另行入账；现状登记为待改进项（20-kernel、~17ms/chunk）；性能锚专项另行立项 —— 验证：`grep -n "统计边界" 该文件` 命中；原行未被删除（git diff 仅增行）
- [x] C3 §9.1 能力边界降级声明（裁定 5-C）：同文件能力边界行**追加注记**：fp32 未验收且无生产消费方（bf16 已 ruling 3 排除）；DK≠DV 降级（Qwen3.5 GDN 头 16V+16QK、head_dim 128 ⇒ DK=DV）；尾块以 host pad 承接（契约见 learning_path 06 §6.9，本地对照 `_pad_invariance.py`，完整前后对照入 R9 设计件）—— 验证：`grep -n "host pad" 该文件` 命中
- [x] C4 p2-08 spec 主路线修正（R7）：`S1/.../p2-08-multimodal-tower/specs/multimodal-tower/spec.md` "视觉塔载入" Requirement 改为：自带视觉塔（冻结）为主路线，SigLIP/CLIP 为备选（与 A1 探针实证及 design D8 一致；变更未归档，delta 可修订）—— 验证：`grep -n "自带" 该 spec` 命中且 SigLIP 标注备选
- [x] C5 p2-08 B2 勾选勘误（R7）：`S1/.../p2-08-multimodal-tower/tasks.md` B2 行下追加注记行：2026-10-24 复核原勾选时无报告产物；补产物任务见 adjudication-remediation-1024 C6（**不改 [x]，只附勘误**）—— 验证：`grep -n "勘误" 该文件` 命中
- [ ] C6 B2 产物补齐（挂起，资源前置：模型推理与数据）：产出 MMBench-CN ≥200 题含无图列评测报告 —— 本阶段不执行，保持未勾
- [x] C7 R9 静态设计件（裁定 1-C）：新增 `S1/release/ascend/kernels/reconcile/R9_V3_BRIDGE_DESIGN.md`：v3 六梯 ↔ 生产 `gdn_asc` 五梯的输入布局/头维映射、梯度顺序与 h0 语义、dtype、autograd 包装契约、端到端 gradcheck 与负锚清单、同形状 A/B 方案、pad 契约引用（引 B5 与教材 §6.9）；结论层写明"过评审后才决定是否实施" —— 验证：`ls` 存在 + `grep -c "h0\|autograd\|A/B" 该文件` ≥3

## 收尾（编排者执行）

- [x] F1 完成度五查 + 统一勾选本文件；R9-adjacent 数值项（B1/B2/B4 上卡复验）列入阶段二首车清单
- [x] F2 本变更产物回写 `docs/reassessment/05-plan-recommendations.md` §3（阶段一完成态）与 evidence V12
