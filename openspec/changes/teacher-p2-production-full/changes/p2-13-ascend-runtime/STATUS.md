# STATUS · p2-13-ascend-runtime（GDN 反向）

> 用途：p2-13 的唯一现状索引（补 R1 的"同一编号语义漂移/最新状态只在 run notes"缺口）。本页只登记
> **现行队列 / 已废与被替代 / 参考件 / 一行状态**，不改任何旧账面；新结论先入 run notes，再回写本页。
> 生成：2026-10-24（`adjudication-remediation-1024` C1）。历史账原文（tasks.md 各节、notes、HANDOFF）保持不动。

## 一行状态

当前**停等实例**（SSH 网关 `Connection closed` / 502，非停机；恢复后先核对本地未入库的实例连接配置——重启后端口可能变化）；**阶段二首车 = `wsc`（T3a）**。

## 1. 现行队列（按次序；判据先写死，出处随行）

| 单号 | 动作 | 判据 | 腿 |
|---|---|---|---|
| **T3a** | 验 `301bfde0`（`solve_a` 回上游 kkt idiom）于 **DK=128、BK=64**，量 A / H_in | `A ≈ 6e-5`、`H_in ≈ 4e-4` ⇒ 真修成立，可议撤 `BK=DK` 绕行；否则保留绕行并记证据 | `wsc` |
| **T3'** | `dk` 四项分解（`GDN_V3_DK_SPLIT=1`）：最差配置（T=128 / 2 chunk）与最好（T=512 / 8 chunk）各一次 | 看哪一项 abs_max 与 `dk` 同量级；`sum_vs_full ≈ 0`，否则拆分不完备 | `dks128` |
| **T2b** | 同法看 `dbeta` 主导项 | 若与 `dk` 的 q2 同源，一次修两处 | 需扩 `_dk_split_128.py` |
| **T5** | 重跑 `rsa`（**目标形 + 弱形 T=128/H=1**） | `dk`、`dbeta` **均 ≤2e-3** 才算收紧成功（T4 裁定 (c)，2026-10-24） | `rsa` |

补充（本节之外的当前裁定，均只追加注记、不改原行）：

- 形态判据已于 2026-10-24 裁定 **2-B**：`≤2 kernel` **保留** + 声明统计边界（反向扫描主体 kernel 数，不含 `cumsum`/`solve_a`/`scan1` 三前置件；host 运算与 launch 数另行入账）；现状 20-kernel / ~17 ms/chunk / 63×（51~74× 误差棒）登记为待改进项。见 `tasks.md §9.1` 注记。
- 能力边界按裁定 **5-C** 分拆：fp32 与 DK≠DV 显式降级；尾块以 host pad 承接（哑 token 契约见 `docs/learning_path/06-accelerators-and-kernels.md` §6.9）。见 `tasks.md §9.1` 注记。
- `ratio ≥ 0.99` 仅作诊断列，不作主判据（对深链梯度结构性不可达，见 HANDOFF §2.3）。

## 2. 已废 / 被替代（别重走）

| 旧物 | 处置 | 依据 |
|---|---|---|
| **P1-1zh「GDN 反向改走 TA（per-token）」** | **已废**：窗九真机撞 TA 编译器缺陷域（最小形即 507015），由 `P1-1zj` chunked 二版（现行 v2 五核）取代；fork 反向层三修证伪后不再修（`P1-1z`/`P1-1za`/`P1-1ze` 随之挂起，fork 路线关闭） | `tasks.md` P1-1zh / P1-1zj；`runs/1013-p2-13-oncard-window9-ta-bwd` |
| **旧 §8「G1–G9 逐项修法」** | **被替代**：改为 §9 终态倒推。G5 关闭；G1/G2/G3 完成；G4 并入 P4 分层测试；G7/G8 并入 P1；G9（fp32）移出本轮 | `tasks.md §9.4` |
| **W12 与 HANDOFF 的 T 编号混用** | **必须带出处引用**（存在**两个 `T5`**）：映射见 §3；单写 "T5" 即语义不明 | `w11_fork_plan.md §W12`、`HANDOFF.md §6` |
| W12 的 T4（两处取向裁定） | **已裁**（2026-10-24）：①数值判据统一到 `rel`、`ratio` 降为诊断列；②形态判据取 2-B（保留 + 统计边界） | `w11_fork_plan.md §T4`、`docs/reassessment/05-plan-recommendations.md §5` |

## 3. T 编号映射（W12 ↔ HANDOFF；两个 T5 之义）

| 编号 | W12（`w11_fork_plan.md` §W12 波次编排） | HANDOFF §6（2026-10-24 现行队列） |
|---|---|---|
| T1 | 回归 `v3t` + `ex`（已完成 2026-10-24 00:20，六项 rel 逐位相同） | —（已完成，数字见 HANDOFF §3.5） |
| T2 | `dk` 波动定性（H=1、T 扫描；已完成 00:33） | —（已完成，结论入 notes [156]） |
| T3 | `bks` 三点拆分（`solve_a` 分块机理） | 不单列：T3a 通过即撤绕行；机理追查按需保留 |
| T3' | `dk` 四项分解（`dks128`，已发车未取到结果） | **T3'**（同义、同腿） |
| T3a / T3a-D28 | — | 新增：T3a 验 `301bfde0`；T3a-D28 把 `_bcast_last.expand` 临时回退（D28 归因，需临时改码先报） |
| T2b | — | 新增：`dbeta` 主导项 |
| T4 | 两处取向裁定（ratio 口径 / 形态判据） | 已裁（见 §2） |
| **T5** | **收尾**：F2 全量门禁 + F3 DoD 逐项核验 + archive | **数值收紧复验**：重跑 `rsa`（目标形 + 弱形），`dk`/`dbeta` 均 ≤2e-3 |

> **两个 T5 之义**：W12 的 T5 = **收尾归档**；HANDOFF 的 T5 = **收紧判据复验**。任何引用必须写全"出处 + 编号"（如 "HANDOFF §6 T5"），禁止单写 "T5"。

## 4. 参考件（唯一引用锚，文件 + 节）

| 件 | 路径 | 用途 |
|---|---|---|
| HANDOFF | `release/runs/1017-p2-13-oncard-window11-v2-acceptance/HANDOFF.md` | 零上下文接手：结构 / 数值 / 队列 / 纪律；§6 现行队列、§9 复核命令 |
| W11 编排 | `release/runs/1017-p2-13-oncard-window11-v2-acceptance/w11_fork_plan.md` | W12 波次表（T1–T5）、W13 收获清单、T4 裁定 |
| 重评 04 | `docs/reassessment/04-ascend-gdn.md` | A-F1..F9：正式消费者仍是 `gdn_asc`（v3 未接入）、测量面漏洞、性能数字边界 |
| 重评 05 | `docs/reassessment/05-plan-recommendations.md` | §3 推进路线、§5 裁定记录（1-C / 2-B / 3-A / 4-C / 5-C） |

提交锚（写就时）：system-one `main` @ `b9c1dce`；tilelang-ascend `ascendc_pto` @ `1bdce50e`（另有 `301bfde0` 待 T3a 上卡验证）。
