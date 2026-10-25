# 状况交接 · p2-13 GDN 反向（2026-10-25 修订版）

> 本文路径：`release/runs/1017-p2-13-oncard-window11-v2-acceptance/HANDOFF.md`
> 目标：**让一个没有任何上下文的新会话在 10 分钟内接手**。
> 凡路径 / 提交号 / 数字均在本文件写就时核对过；`§9` 给了逐条复核命令。
> 本文件不改产品代码、不替代 `notes.md`（逐条原始记录），只做索引与当前态。
> 与上一版（2026-10-24）的差别：五项取向已裁定并落地、验收测量面已修、CI 由红转绿、实例重新可用、两仓远端结构变化。

---

## 0. 30 秒

| 项 | 内容 |
|---|---|
| 在做什么 | `system-one` **p2-13**：GDN（Gated DeltaNet）反向在昇腾 910B 上的实现与验收 |
| 目标形 | `B1 H16 T8192 DK128 DV128 C64`（128 chunk），六项 `dq/dk/dv/dbeta/dg/dh0` |
| 数值状态 | 目标形**首次六项 <5e-3**（历史批次，余量 9%~14%）；但 `dk`/`dbeta` 是**薄边界项**，换 H 或 chunk 数会翻线 |
| 根因 | `solve_a` 的 KK^T 分块路径在 `bk_num>1` 时错；上游 idiom 是 GM→L1 直拷，我们多了一层 UB 中转。`BK=DK` 是绕行，真修 `301bfde0` **尚待上卡验证** |
| 本会话新增 | 五项裁定落地（`openspec/changes/adjudication-remediation-1024`，16/17）；验收测量面修复；R9 接入静态设计；全项目重评与教材（`docs/reassessment`、`docs/learning_path`） |
| 待办 | 阶段二队列从头开始：**T3a**（`wsc` 验真修）→ T3'（dk 分解）→ T2b（dbeta）→ T5（双形收紧） |
| 卡 | **实例已恢复（`ALIVE`，端口未变）**，可立即发车 |
| 最大风险 | ① 真修未验证前任何"六项全绿"都建立在绕行上；② `scan2a/scan2b` 的同形 BK 分半仍未验（v2 自身作为 oracle 的风险） |

---

## 1. 三个仓库位置与状态

| 仓库 | 路径 | 分支 / 提交 | 说明 |
|---|---|---|---|
| system-one（计划与证据） | `$REPO_ROOT` | `main` @ `1d98dd5` | 2026-10-25 已与 `origin/main` 同步；远端曾被重建为 curated snapshot，**本地旧历史已作废** |
| tilelang-ascend（算子所在） | `$WORKSPACE_ROOT/tilelang-ascend` | 本地 `ascendc_pto` @ `5adc8753` | 相对上游 ahead 222 / behind 18 |
| tilelang-ascend 的 fork | `github.com/SpellingDragon/tilelang-ascend` | 分支 `gdn-bwd-measure-fixes` @ `5adc8753` | 本会话提交**只在这里**；fork 为 public |
| tilelang-ascend 上游 | `github.com/tile-ai/tilelang-ascend` | `origin/ascendc_pto` @ `43b0b697` | **未推送、未强推**；本地落后 18 个提交，恢复同步需 rebase 或 merge（会改写或引入历史，属被裁事项） |

> 文中 `$REPO_ROOT` / `$WORKSPACE_ROOT` 是脱敏占位符（个人路径不入库）。本机典型值：
> `$REPO_ROOT=~/Documents/codes/system-one`，`$WORKSPACE_ROOT=~/Documents/codes`；发车脚本所用的 `.env.card` 已导出这两个变量。

本会话在 tilelang-ascend 的实际改动（提交 `5adc8753`）：

| 文件 | 改动 |
|---|---|
| `examples/gdn_bwd_v2/_process/_realshape_acc.py` | 聚合前逐输出 `isfinite` 硬门（原 `max(prev, nan)` 会吞 NaN）；用例表首行恢复 `(8192,128,16)` 目标形 |
| `examples/gdn_bwd_v3/test_gdn_bwd_v3.py` | boundary 注入后**用同一 do 重算参考**（原口径是拿注入后的输出比未注入的 golden） |
| `examples/gdn_bwd_v3/example_gdn_bwd.py` | 内联 `solve_a` 回到 v2 现行 idiom（删 `fill` + GM→UB→L1 中转） |
| `examples/gdn_bwd_v2/_process/_pad_invariance.py` | 新增：哑 token pad 契约的本地对照（见 §3.6） |

---

## 2. 背景（零上下文版）

### 2.1 项目与证据目录

`system-one` 是教学项目，两阶段：`scratch/`（P1 学习版，冻结只读）、`release/`（P2 正式版）。本会话全部工作在 `release/`。主证据目录：

```
release/runs/1017-p2-13-oncard-window11-v2-acceptance/
├── notes.md          4271 行，逐条记录 [1]…[159]（只读尾部与指定编号，别通读）
├── w11_fork_plan.md  W12 波次表 / W13 收获清单 / 待裁项
├── scripts/          w11_car.sh（真机长跑）、w11_pull_logs.sh（归档）、
│                     check_example_sync.py（示例↔kernel + **内联前置件**对账）、
│                     .env.card（含凭据路径，不入库）
└── artifacts/w11car/<UTC时间>/   15 批车证据（全量日志 + 卡上运行器 + MANIFEST 字节数）
```

> **artifacts 当前不在版本库里**：2026-10-25 快照重建时被移出工作区。可从旧提交恢复（19 MB，仅工作区、不自动入库）：
> `git restore --source=7f1011b --worktree -- release/runs/1017-p2-13-oncard-window11-v2-acceptance/artifacts`
> 是否重新入库属未决项（快照清洗可能是有意剔除大件）；未恢复时下文 `§3` 的批次引用只能从旧提交读。

配套：`p2-13` 变更目录下的 `STATUS.md`（现行队列/已废项/双 T5 映射）与 `w11_fork_plan.md` 互补。

### 2.2 算子结构（读懂这四个文件就够）

| 文件 | 内容 |
|---|---|
| `tilelang-ascend/examples/gdn_bwd_v3/gdn_bwd_v3.py` | 20+ 个单职责 kernel + 2 个 host wrapper：`gdn_bwd_v3`（精简）、`gdn_bwd_v3_relay`（完整反向，**主线**） |
| 同上 `example_gdn_bwd.py` | 上库单文件示例：内联 kernel + 三个 v2 前置件 + fp64 golden + 两个 case |
| 同上 `test_gdn_bwd_v3.py` | 分层测试（l0/l1/l2/boundary/p3），`GDN_V3_GOLDEN_H` 默认 `0`（原生路径） |
| `tilelang-ascend/examples/gdn_bwd_v2/gdn_bwd_v2.py` | 三个前置件：`cumsum` / `solve_a`（求逆 A）/ `scan1`（状态扫描 H_in）；**本轮的 bug 在 `solve_a`** |

数据流（relay 内部）：`cumsum → solve_a → scan1` 得 `ws_g/ws_a/ws_h`；再按 chunk 缓存 `vnew/W/Kt/Δ'`，反向扫 `Δ'`（唯一矩阵递推量）并产出六项。`dg` 另有 `carry` 后缀累计。

**生产消费方不是 v3**：训练链走 `release/ascend/{train_step,autograd_asc}.py → kernels/gdn_asc.py` 的五梯（无 h0 入参），v3 六梯目前无生产调用。

### 2.3 三个误差尺子（不要混用）

- `rel = max|a−g| / max|g|`：本项目主判据，阈值 5e-3；
- `max_abs`：绝对误差，与量级绑定；**不是恒定底噪**；
- `matched_ratio`（`|a−g| ≤ atol + rtol·|g|`，fp16 `2^-14`/`2^-9`，要求 ≥0.99）：对深链梯度常达不到。
  **注意**：旧版曾据此断言"数学上永远不可达"，本轮已撤回——相同 `max_abs` 可对应 0.999 或 0.001 的匹配率，见 `docs/learning_path/05-experimental-evidence.md` 的反例。用 `rel` 判定时应写明理由，`ratio` 作诊断列。

---

## 3. 当前数值状态（唯一权威来源：notes + artifacts）

### 3.1 目标形（`rsa` 腿，原生路径）

> 批次 `artifacts/w11car/20261023T2353Z/w11car_rsa.log`（修复后）

```
B1 H16 T8192 DK128 DV128 C64，128 chunk
dq 5.897e-04   dk 4.289e-03   dv 1.043e-03   dbeta 4.555e-03   dg 4.718e-03   dh0 4.642e-04
→ 六项全 <5e-3（dk 余量 14.2% / dbeta 8.9% / dg 5.6%）
```

**同批 CONTROL（H16/T512/DK64）多项 FAIL，整批终判 `SOME_FAIL`**——目标行通过 ≠ 整批验收通过。

### 3.2 前置件对 golden（`wsc` 腿，带 64 形 CONTROL）

```
DK=64   BK=64   A 5.93e-05   H_in 3.62e-04   ok
DK=128  BK=64   A 1.388e-01  H_in 4.307e-02  SUSPECT   ← bk_num=2（旧 UB 中转写法）
DK=128  BK=128  A 5.961e-05  H_in 3.992e-04  ok        ← bk_num=1（绕行）
```

### 3.3 薄边界项（`rsa` T 扫描，H=1、DK=128，CONTROL = H1/DK64/T256 全 PASS）

| T | chunks | dk | 该档结论 |
|---|---|---|---|
| 128 | 2 | **8.049e-03 FAIL** | SOME_FAIL |
| 512 | 8 | 1.321e-03 | ALL_PASS |
| 2048 | 32 | 3.316e-03 | ALL_PASS |
| 8192 | 128 | 4.111e-03 | `dbeta` **6.116e-03 FAIL** |

⇒ `dk`/`dbeta` 跨配置摆动 1.3e-3~8.0e-3，**非单调**。`dq/dv/dh0` 所有档 ≤1.1e-3。

### 3.4 性能（`rs2` 腿，暖机 median）

```
T=512/8chunk   bwd 0.14s   fwd 0.01s   11.3×   ms/chunk 17.5
T=2048/32      bwd 0.52s   fwd 0.02s   32.9×   ms/chunk 16.3
T=8192/128     bwd 2.22s   fwd 0.04s   63.5×   ms/chunk 17.3
```

⇒ bwd ≈ 17 ms × chunk 数；约 35 次 TL 调用/chunk（源码计数 `3+10NC+25(NC−1)+21`）。**尚未用 profiler 分解 launch/拷贝/算力**，63× 的分母只有两位有效数字（引用需带 51~74× 的舍入范围，这不是统计误差棒）。

### 3.5 回归（`v3t` + `ex`，`BK=DK` 之后）

- `v3t` 六项 rel 与改动前**逐位相同**（T=64: dq 4.321e-04 / dk 3.669e-04 / dv 6.514e-04 / dbeta 3.400e-04 / dg 2.511e-04 / dh0 4.619e-04）；l1 T=128/192 六项 gate PASS。
- `ex` 打印 `Kernel Output Match!`（T=64 与 T=256）。
- 上述均为**旧 example（内联前置件为旧写法）**的结果；本会话已把 example 前置件同步为 v2 现行写法，**需阶段二首车重跑 `v3t/ex` 确认**。

### 3.6 本会话修的验收面（重要：旧的"全绿"要在新门下重测）

| 修复 | 旧问题 | 新行为 |
|---|---|---|
| `_realshape_acc.py` 有限性硬门 | `max(0.0, nan) == 0.0`，单 head 的 NaN 被聚合吞掉 | 逐输出 `isfinite`，非有限直接 `NONFINITE` + FAIL |
| 目标形用例 | `main()` 只剩 H=1 扫描，目标形不再自动跑 | 用例表首行恢复 `(8192,128,16)` |
| boundary 参考 | 注入 NaN 后仍比未注入 golden | 用同一注入 do 重算参考（可判定的"传播语义"观察，仍 WARN-only） |
| `check_example_sync.py` | 只比 v3 段（776 行），**内联前置件漂移看不见** | 新增 SYNC2：三函数体 vs `gdn_bwd_v2.py` 逐行；当前两段全绿 |
| `_pad_invariance.py`（新增） | 尾块 pad 契约无本地证据 | T=100/130 → 补到整倍数，六梯度 `worst 9.641e-17 PASS`；负锚实测发现：**末尾 pad + 仅有效位损失时，`g_pad≠0` 对有效位梯度零影响**，真正被影响的是状态链/终态（约束对象写入 R9 §1.6） |

---

## 4. 已作废的结论（别重走，每条都花过钱）

| 曾经的结论 | 为什么作废 |
|---|---|
| D23：循环内 `gemm_v0(init=True)` 只清一次 | 两轮最小复现均不成立 |
| `dk` 参与跨 chunk 递推，需给 dk 加 fp32 中转 | 读错代码：`dk` 只写不读，递推量是 `delta_f32`；`dg` 另有 carry |
| `copy_ub_to_gm` 不支持 fp32 | 真因是 src/dst dtype 不一致；模板本身 dtype 泛型 |
| "H>1 不支持" | H=1 的 T=512 一样坏；H 数/chunk 数/`H_in`/h0/golden 已逐个排除 |
| "K=128 的 GEMM 收缩不准" | 仓内 `notes [99]` 早有反例：`MM C=64 M=128 N=64 rel=3.330e-04 PASS` |
| "width 128 / fp16 落 GM 有问题" | `_wide_tile_probe` 14 格全 PASS（最大 3.4e-04） |
| "`solve_a` 的 BK 分块本身有 bug（上游同病）" | 上游同 idiom 是 GM→L1 直拷且其 `DK=64/BK=32`（bk_num=2）被测且绿；是**我们多加了 UB 中转** |
| "改善 7~22×" | 两把尺子混比（`max_abs` 与 `rel`），以及不同形状 |
| `ratio≥0.99` 作为主判据 | 对深链梯度常不可达；且"数学上永远不可达"的论证本身已撤回 |
| `matched_ratio` 永远到不了 0.99（旧教材） | 相同 `max_abs` 可对应 0.999/0.001 两种匹配率；旧论证把"最大值"当"逐元素底噪" |
| `#1802` 的形态可作为完整六梯度替代蓝本 | 本地参考件是**局部** `chunk_o_bwd`：无 dbeta/dv 输出/dh0，h/dh 为输入，禁尾块；只能作局部参考 |
| v3 已接生产 / 六项全绿=可交付 | 生产走 `gdn_asc` 五梯；v3 无生产调用，且换形翻线 |
| 本会话的"文档全绿"= 验收全绿 | `SYNC-OK` 只覆盖 v3 段，直到本会话才加上内联前置件（见 §3.6） |

---

## 5. 环境与发车

```
release/runs/1017-…/scripts/.env.card      # 不入库，含 key 路径
  CARD_SSH / CARD_KEY / CARD_PORT / WORKSPACE_ROOT / REPO_ROOT
```

**实例重启后会坏的三件事**（遇到过 3 次）：

1. **端口可能变** ⇒ 在控制台核对后更新 `.env.card` 的 `CARD_PORT`；
2. `/tmp` 被清空 ⇒ `/tmp/ta_unpack`（TA 包）消失；车会检测并自动 `UNPACK-OK`，失败需手工重传；
3. SSH 网关返回 `HTTP/1.1 502 Bad Gateway`（表现为 `Connection closed`）⇒ 实例侧隧道未起，不是"实例停机"。

### 发一辆车（三条命令）

```bash
cd $REPO_ROOT/release/runs/1017-p2-13-oncard-window11-v2-acceptance
set -a; . scripts/.env.card; set +a
bash scripts/w11_car.sh "rsa wsc"          # 腿名见 §6；一条车顺序跑完所有腿（单锁）
bash scripts/w11_pull_logs.sh 20261025T0100Z   # 归档到 artifacts/w11car/<tag>/
```

车会做四件事：manifest md5 校验（传错码拒跑）→ 远端 flock 单锁 → 每腿跑完打印 `$leg-CONTROL=none|FAIL|ok(n)` → 日志落 `/tmp/w11car_$leg.log`。
**`CONTROL=none` 的腿不得引作结论**（该腿没有对照臂）。

---

## 6. 待办队列

### 6.1 本会话已完成（本地，无卡）

`openspec/changes/adjudication-remediation-1024/`（16/17 任务，`C6` 挂起）：

| 组 | 内容 |
|---|---|
| A 产品代码 | OPD 训练轴切 `train` + 守卫扩全入口（含注错必红）；`resample_every`（默认 50，0=旧路径）；改名"on-policy 选项蒸馏"；design 追加 D14 |
| B 测量面 | §3.6 的五处修复 + `_pad_invariance.py`；`SYNC2` 两段全绿 |
| C 账目 | p2-13 `STATUS.md`；§9.1 两条注记（形态判据统计边界、能力边界降级）；p2-08 spec 主路线修正 + B2 勘误；`R9_V3_BRIDGE_DESIGN.md` |

另：CI 从红转绿（修 ruff 与 corpus 分片缺件两处快照遗留），P1 `278 passed`、P2 `343 passed`；README 重写为学生视角。

### 6.2 阶段二（需卡；实例已 `ALIVE`）

| 单号 | 动作 | 判据 | 腿 |
|---|---|---|---|
| **T3a** | 验 `301bfde0`（`solve_a` 回上游 idiom）：DK=128、**BK=64** 下量 A/H_in | `a_rel ≈ 6e-5`、`h_rel ≈ 4e-4` ⇒ 真修成立，可撤 `BK=DK` 绕行；否则保留绕行并记证据 | `wsc` |
| **回归** | `v3t` + `ex`（example 前置件已改，必须重跑） | 64 形六项与历史逐位相同或更好；`ex` 打印 Match | `v3t ex` |
| **T3'** | `dk` 四项分解（最差 T=128/2 chunk 与最好 T=512/8 chunk） | 看哪一项 abs_max 与 dk 同量级；`sum_vs_full≈0` 否则拆分不完备 | `dks128` |
| **T2b** | 同法看 `dbeta` 主导项 | 若与 dk 的 q2 同源，一次修两处 | 需扩 `_dk_split_128.py` |
| **T5** | 重跑 `rsa`（目标形 + 弱配置 T=128/H=1） | `dk`、`dbeta` 都 ≤2e-3 才算收紧成功 | `rsa` |

### 6.3 挂起 / 待裁

- `C6`：p2-08 B2 的评测报告产物（资源前置：模型推理与数据）；
- `rsa` 探针是否 exit-code 化（需先核 `w11_car.sh` 的 `run_leg` 是否消费退出码，未核前不动）；
- **教师选型开放**（后续可能更换蒸馏模型）——真 4B 载入与 ToS/CC 核验暂缓，选型定后重开 R6；
- 上游同步：本地 `ascendc_pto` behind 18，恢复需 rebase/merge，属未决。

### 6.4 取向已裁定（不再讨论）

见 `docs/reassessment/05-plan-recommendations.md` §5：1-C 双轨 / 2-B 保留 `≤2 kernel` 判据并补性能锚 / 3-A 替身纪律且教师选型开放 / 4-C on-policy 选项蒸馏 / 5-C 支持面分拆（fp32、DK≠DV 降级；尾块 host pad）。

---

## 7. 纪律（用代价换来的）

1. **一条结论 = 数字 + 适用范围（形/dtype/输入分布/开关）+ 对照组**。填不满不许进 DoD。
2. **目标形排在用例表第一行**；跑不动写"未验证"，不许小形顶替。
3. **探针必须带 CONTROL 臂**；无对照的 FAIL 不可归因。对照臂自身也要能红（注错必红）。
4. **一次只改一个变量，判读规则写在跑之前**。
5. **不改运行中的脚本**：bash 按字节偏移增量读取；脚本若嵌在 `ssh "…"` 双引号串里，**不许出现反引号**，内层双引号写 `\"`。
6. **`finite` 不是 `accurate`**；写"跑通"必须同句给数值或"数值未验"。
7. **先查自己的笔记再提假设**（本会话 3 个"从代码读出的因果链"全被数据推翻）。
8. **产物必须归档**（日志 + 卡上运行器 + 字节数）；只在 `/tmp` 里的东西实例一停就没了。
9. **执行代理不勾账**：勾选由编排者按完成度五查后统一执行（R17 账目单点归属）。
10. **提交前跑 `tools/scrub_sensitive.py --check-rev HEAD`**（个人路径/云登录名/key 名一律占位符化）；`--check` 扫工作区时可能命中未入库旧件，以 `--check-rev` 为准。
11. **README 面向学生，项目状态放 `docs/reassessment` 与 STATUS**；不把管理口径写进门面。
12. **上卡前核对释放表**（谁在跑、锁在哪）；不得另起一路双跑。

---

## 8. 文件索引

| 想解决 | 读这个 |
|---|---|
| 本会话的全项目评估（含裁定与建议） | `docs/reassessment/{01-global-plan,02-p1-chain,03-p2-chain,04-ascend-gdn,05-plan-recommendations}.md` + `evidence.md`（V00–V13） |
| 零基础弄懂"算子到底对不对" | `docs/learning_path/`（七章，18 个 CPU 实验）+ `docs/teaching_gdn_bwd_verification.md`（已勘误） |
| p2-13 现行队列与已废项 | `openspec/changes/teacher-p2-production-full/changes/p2-13-ascend-runtime/STATUS.md` |
| v3 → 生产接入的静态设计 | `release/ascend/kernels/reconcile/R9_V3_BRIDGE_DESIGN.md` |
| 本轮所有实测数据与逐条记录 | `…/notes.md`，重点 `[144][152]–[159]` |
| 编排与待裁项 | `…/w11_fork_plan.md` |
| 车的腿名与脚本 | `…/scripts/w11_car.sh`（`v3t ex rsa rs2 wsc attr wide bks dks128 pab unit …`） |
| 探针 | `tilelang-ascend/examples/gdn_bwd_v2/_process/*.py`（本轮新增 `_pad_invariance.py`） |
| 项目纪律 | `system-one/.agent/rules/common.md`、两仓 `AGENTS.md`、`.agent/skills/openspec-multilevel-planning/` |

---

## 9. 怎么验证本文件没写错（新会话开场可跑）

```bash
# 以下命令假定两个根已导出（发车脚本的 .env.card 也会导出它们）
export REPO_ROOT=$HOME/Documents/codes/system-one WORKSPACE_ROOT=$HOME/Documents/codes

# 1) 两仓状态与提交
cd $REPO_ROOT && git log --oneline -1 && git status --short
cd $WORKSPACE_ROOT/tilelang-ascend && git log --oneline -1 && git rev-list --left-right --count origin/ascendc_pto...HEAD

# 2) 证据与门禁
cd $REPO_ROOT
ls release/runs/1017-p2-13-oncard-window11-v2-acceptance/{notes.md,w11_fork_plan.md}
# artifacts 未随库分发；先恢复再数（见 §2.1）
git restore --source=7f1011b --worktree -- release/runs/1017-p2-13-oncard-window11-v2-acceptance/artifacts
ls release/runs/1017-p2-13-oncard-window11-v2-acceptance/artifacts/w11car | wc -l   # 期望 ≥15
python3 tools/scrub_sensitive.py --check-rev HEAD                                   # 期望 exit 0
openspec list                                                                        # 三个活跃变更 + 一个空壳

# 3) 目标形数字是否与 §3.1 一致（必须指定批次）
f=release/runs/1017-p2-13-oncard-window11-v2-acceptance/artifacts/w11car/20261023T2353Z/w11car_rsa.log
grep -aq "T=8192 H=16 DK=128" "$f" && echo BATCH-OK
sed -n "/T=8192 H=16/,/ACC dim=128/p" "$f" | tail -8

# 4) 本会话修的验收面是否在位
TA_ROOT=$WORKSPACE_ROOT/tilelang-ascend \
  release/.venv/bin/python -P release/runs/1017-p2-13-oncard-window11-v2-acceptance/scripts/check_example_sync.py
# 期望 SYNC-OK + SYNC2-OK 3/3
cd $WORKSPACE_ROOT/tilelang-ascend/examples/gdn_bwd_v2 \
  && $REPO_ROOT/release/.venv/bin/python -P _process/_pad_invariance.py
# 期望 PADINV worst≈9.6e-17 verdict=PASS

# 5) 本地门禁
cd $REPO_ROOT/scratch && PYTHON=$PWD/.venv/bin/python bash tools/ci.sh   # 278 passed
cd ../release && PYTHONPATH=$PWD .venv/bin/python -P -m pytest tests -q -m "not integration"              # 343 passed
```

`§9-4` 与 `§9-5` 的期望数字来自 2026-10-25 本地实测；示例脚本的 CPU 输出不证明 NPU 数值。

---

## 10. 不要动的东西

| 东西 | 原因 |
|---|---|
| `system-one/release/ascend/port910b/ta_defect_dossier.md` | 2026-10-17 旧件，非本轮产物（未跟踪），与当前结论可能冲突 |
| `system-one/system_one_study_eval.agent.final.md` | 根目录未跟踪旧件；含 5 处敏感性字样，永不入库 |
| `scripts/.env.card` | 含凭据路径，**不入库**；实例重启后可能需要更新 `CARD_PORT` |
| `scratch/`（system-one） | P1 学习版，冻结只读（本会话仅修了快照遗留的 ruff 与资产缺件口径，已留痕 V13） |
| 已完成但未验证的 `BK=DK` 绕行 | 撤它要以 T3a 通过为前提，不许"看着有真修了就顺手撤" |
| 上游 `origin/ascendc_pto` | 本地 behind 18，任何同步/推送需先裁定策略（rebase 会改写 222 个本地 hash） |
