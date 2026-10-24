# Design: adjudication-remediation-1024

## 决策（D1–D7，含被否方案；裁定原文见 `docs/reassessment/05-plan-recommendations.md` §5）

### D1 双轨取向（裁定 1-C）
算子研究按 W13 收口与训练先决（R4+R9 静态设计）并行；C5 点火条件 = R4 完成 ∧ R9 静态设计过评审 ∧ 教师选型与许可核验落定。被否：纯收口（训练先决停滞）；直上训练系统（真教师/RL/预算先决未就绪）。

### D2 性能判据口径（裁定 2-B）
`≤2 kernel` 判据保留；本变更只做"统计边界声明"（C2 任务）：指反向扫描主体，不含 cumsum/solve_a/scan1 三前置件；host 运算与 launch 数另行入账；现状（20-kernel、~17ms/chunk）登记为待改进项。profiler/容量账/同工作量 A/B 为独立专项，不在本变更排产。被否：登记放弃（失去独立锚）；按 #1802 重写（局部反向不可比）。

### D3 替身教师纪律（裁定 3-A）
scaffold 仅限流程冒烟，一切引用带 `#scaffold-cpu`；真 4B 载入与 ToS/CC 核验暂缓（用户补充：教师选型开放，后续可能换模型蒸馏）。本变更不改教师配置文件；gate08b 现值不动。被否：立即切真 4B（选型未定，投入有沉没风险）。

### D4 on-policy 选项蒸馏（裁定 4-C）
- 实现面：`collect_onpolicy` 的采样从"训练前一次性"移入训练循环，按 `resample_every` 步分段重采（默认 50；0=关闭，行为与旧版一致，作为回归锚）；优化器与调度器跨段连续不重建；每段采样种子由 `seed+段号` 派生并可复现，写入 run 记录。
- 命名/文档面：模块定位统一为"on-policy 选项蒸馏（选项级 forward-KL）"；`reverse_kl` 函数名保留（改名波及全部调用与测试，收益为负），docstring 注明"命名为历史遗留，实算 KL(teacher‖student)"；一级 design.md 以追加 D14 节取代 D9 的"per-token reverse KL"表述，D9 原文不改。
被否：仅改名（未兑现随策略更新）；序列级 on-policy（零生成产品形态下"学生生成序列"无自然对应物，将来若做需先立设计定义）。

### D5 支持面分拆（裁定 5-C）
- fp32：显式降级声明（从未验收；bf16 已被 ruling 3 排除；训练主档 fp16；数学正确性由 CPU fp64 golden 覆盖）。
- DK≠DV：显式降级声明，依据目标模型公开结构（Qwen3.5-0.8B GDN 头 16 V + 16 QK、head_dim 128 ⇒ DK=DV=128）。
- 尾块：host pad 方案承接。pad 契约 = 哑 token（q 任意、k=0、v=0、g=0、β=0、dO=0），教材依据 `docs/learning_path/06-accelerators-and-kernels.md` §6.9；本变更交付本地 pad 不变性对照脚本（B5）作为"golden 同 pad 对照"的第一步，完整前后对照入 R9 设计件。
被否：全降级（尾块是变长训练真实需求）；全补实现（fp32/DK≠DV 无消费方）。

### D6 跨仓白名单与勾选归属（编排纪律）
- 三工作组触碰集互不相交：A 组=`release/production/opd.py`、`release/tests/{test_train_axis_guard,test_opd}.py`、`openspec/.../teacher-p2-production-full/design.md`（仅追加 D14 节）；B 组=tilelang-ascend `examples/gdn_bwd_v2/_process/{_realshape_acc,_pad_invariance}.py`、`examples/gdn_bwd_v3/{test_gdn_bwd_v3,example_gdn_bwd}.py` + system-one run `scripts/check_example_sync.py`；C 组=openspec p2-13 `STATUS.md`（新）与 `tasks.md`（仅追加注记）、p2-08 `specs/.../spec.md` 与 `tasks.md`（仅追加勘误）、`release/ascend/kernels/reconcile/R9_V3_BRIDGE_DESIGN.md`（新）。
- **tasks.md 勾选单点归编排者**：执行代理只报完成与凭据，不勾账（R17 账目单点归属）；编排者完成度五查后统一勾选。

### D7 验证面声明（防假绿）
- 可本地实证的（A 组测试、B3/B4 sync、B5 pad 对照）：必须实跑并附命令与输出。
- 上卡才能出数的（B1/B2 的数值行为）：本变更验证面 = `py_compile` + 结构 grep（用例存在、注入参与参考），任务行明标"数值复验待阶段二上卡"，不得以本地绿冒充。
- release 侧跑 pytest 的已知陷阱：工作目录若在 tilelang 源码树内会遮蔽 venv 包，必须 `PYTHONPATH=<release 绝对路径>` 且 `python -P`（V02 实测）。

## 风险与回退

- [B4 改 example 前置件影响上库形态] → 改动与 gdn_bwd_v2.py 现行 idiom 逐字一致，且 relay 仍传 BK=DK（bk_num=1 两条写法均正确）；上卡回归（v3t/ex）在阶段二首车执行，任务行已注明。
- [A2 重采样改变训练动力学] → `resample_every=0` 保留旧路径作回归锚；单测先钉行为再合入。
- [C 组改活跃账] → 只追加注记/新文件，不删改原行；git 可按行回滚。
