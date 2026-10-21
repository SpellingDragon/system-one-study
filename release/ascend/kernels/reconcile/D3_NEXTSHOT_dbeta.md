# D3_NEXTSHOT_dbeta.md — P1-1z 下窗一枪：`dbeta` 三判决最小实验序列

> 执行代理：**D3**　｜　**本窗未执行**（不占卡）　｜　变更：p2-13 / P1-1z
> 本件是**给下窗的一枪设计**：单卡、单进程、单发射序，**同窗一次判三件事**。
> 配套驱动：`D3_probe_dbeta_order.py`（本次写好、**未在真机验证**）；判决腿复用 `run_delta_card.py`（本窗已加容错逐项判）。
>
> 标注：✅ 实测原文｜⚠️ 引用未取｜❓ 推断。**同源假设（H-b）未判决前一律 ❓。**

---

## 0. 一枪目标（三判决）

| 判号 | 要判的事 | 手段 | 判据行 |
|------|---------|------|--------|
| **①** | `dbeta` 是否**只**坏？其余五项（`dq/dk/dv/dg/dh0`）状态？ | `run_delta_card.py --target ascend --bwd`（本窗工装已改容错逐项判） | 7 条 `ITEM ... verdict=` |
| **②** | **UB 预清零/顺序**后 `dbeta` 是否改善 ⇒ 同源假设（H-b）真伪 | `D3_probe_dbeta_order.py --dev npu --variant ub` | `ORDER-VERDICT` + `DBETA-FP` |
| **③** | 若①多项坏 ⇒ 是否**共指同一 UB 区** | `run_delta_card.py --target ascend --bwd --variant pure` 与 ub 对照 | 两档 `ITEM` 对照 |

**背景锚点**（⚠️ 引用 w3 台账，本窗未复取）：`NUM-delta_bwd FAIL first=dbeta rel=5.997e-01 absdiff=4.435e+00 thresh=2e-04`。

---

## 1. 前置（卡上环境，与 w3 一致）

- 容器/兼容：`cann910b-h`，compat `ca023c2e01746ca52c87f36bcea93ee9`（⚠️ 编排者口径）。
- 单卡单任务；**三判决在同一卡窗内跑完**（S1→S2→S3，避免跨窗环境漂移）。
- 假绿守卫已内建：`run_delta_card.py` 必见 `gdnb[` 键（否则 `FAIL 假绿`）；`D3_probe_dbeta_order.py` 打印 `gdnb keys = …`。

---

## 2. 最小实验序列（每步一条命令 + 预期真值表）

### S0（不占卡，host 自证工装）
```bash
cd release/ascend/port910b && ../.venv/bin/python run_delta_card.py --target cpu --bwd | grep -E '^(ITEM|NUM-delta)'
```
**预期**：7 条 `ITEM` 全 `PASS`、`NUM-delta_bwd PASS`（本窗已实测，见 `D3_DBETA_EVIDENCE.md §1.3`）。
**门禁**：若 S0 不 PASS ⇒ 工装被改动或环境坏，**不要占卡**。

### S1（卡上，判 ①）——逐项判决
```bash
python3 ascend/port910b/run_delta_card.py --target ascend --bwd
```
**预期输出**（`ITEM` 行**全出**，不再早退）：
```
ITEM dq    rel= <v> verdict= <PASS|FAIL>
ITEM dk    rel= <v> verdict= <PASS|FAIL>
ITEM dv    rel= <v> verdict= <PASS|FAIL>
ITEM dbeta rel= <v> verdict= <PASS|FAIL>
ITEM dg    rel= <v> verdict= <PASS|FAIL>
ITEM dh0   rel= <v> verdict= <PASS|FAIL>
ITEM adjoint rel= <v> verdict= <PASS|FAIL>
NUM-delta_bwd <PASS|FAIL first=…>
```

**真值表（①）**：

| 观测 | 推断（❓ 判决后转 ✅） | 下一步 |
|------|----------------------|--------|
| **只 `dbeta` FAIL**，其余六项 PASS | `dbeta` **孤立**缺陷；且 `dv`（`=bt·du`）过 ⇒ **`du` 正确** ⇒ 缺陷**不在 `du`/`lam_ub` 链**，而在 `dbeta` 专属路径（`db_acc` 归约/落盘） | 重点看 S3 的 `dbeta` fp |
| `dbeta` **+ 多项** FAIL（如 `dg`/`dh0` 也挂） | 非孤立 ⇒ 疑**共指同一 UB 区**（`lam_ub`/`dec_ub`…） | 进 S2 判 UB 专属 |
| **全 PASS** | 与 w3 矛盾 ⇒ 疑**非确定性/窗口相关** | 连跑 S1 三次看是否闪烁；并跑 S3 |

> 注意：`dv=bt·du`，S1 中 `dv` 判决**直接约束 `du` 正确性**；这是把责任面从"整条伴随链"收窄到"`dbeta` 专属式"的关键一行。

### S2（卡上，判 ③）——`ub` vs `pure` 档对照
```bash
python3 ascend/port910b/run_delta_card.py --target ascend --bwd                 # ub（默认）
python3 ascend/port910b/run_delta_card.py --target ascend --bwd --variant pure  # pure
```
**真值表（③）**：

| `ub` | `pure`（零 `alloc_shared`、Λ 在 GM） | 结论（❓→✅） |
|------|--------------------------------------|--------------|
| FAIL | **PASS** | 缺陷 **UB 专属**（Λ/暂存在 UB）⇒ 共指 UB 面；与①"多项坏"互相印证 |
| FAIL | **FAIL** | **非** UB 专属 ⇒ 缺陷在**共享标量数学/公共 codegen**，UB 只是同现 |
| PASS | PASS | S1 的 FAIL 不复现 ⇒ **非确定性**（时间/顺序相关）⇒ S3 必跑 |

> `pure` 档设备腿**本窗从未跑过**（host 腿才跑 pure，且因 variant 混淆与设备 ub 不可比——见 `D3_DBETA_EVIDENCE.md §3.3`）。S2 是**第一次**在真机上让 `pure` 落地，直接给"UB 专属 vs 共享"的判词。

### S3（卡上，判 ②）——UB 预清零/顺序判决（同源假设 H-b）
```bash
python3 ascend/kernels/reconcile/D3_probe_dbeta_order.py --dev npu --variant ub
# 若 S2 显示 pure 也挂，再补一条：
python3 ascend/kernels/reconcile/D3_probe_dbeta_order.py --dev npu --variant pure
```
该驱动同进程依次发射：`clean(none)` → `dirty预填` → `clean(after-dirty)` → `zero预填` → `clean(after-zero)`，比对 clean 三跑的 `dbeta` 指纹。

**预期输出**：
```
gdnb keys = ['ascend|gdnb[ascend]|h2k8v8|c2ubz']
ITEM dq/dk/dv/dbeta/dg/dh0 rel= … verdict= …
DBETA-FP  clean 三跑指纹（none / after-dirty / after-zero）:
  none   dbeta-fp=<h1>
  dirty  dbeta-fp=<h2>
  zero   dbeta-fp=<h3>
ORDER-VERDICT <逐位同 ⇒ 顺序无关(H-b 证伪) | 随 prefill 变 ⇒ 存在 stale(H-b 证实)>
```
**真值表（②）**：

| 观测 | 结论（❓→✅） | 含义 |
|------|--------------|------|
| clean 三跑 `dbeta` **逐位同**，且 rel 仍 ~`6e-1` | **H-b 证伪** | 与前一趟 UB 内容**无关** ⇒ E 段"多突发跨步残留"机制**不适用** gdn（与结构证据"0 DMA、无 stale 落点"一致） |
| clean `dbeta` **随 prefill 内容变**（fp 不同） | **H-b 证实** | 存在 **stale 成分** ⇒ 同源假设成立；`ORDER-DIFF` 行会列出受影响项 ⇒ 借差异**指认 UB 落点** |
| `zero` 后**改善**、`dirty` 后恶化 | **H-b 证实 + 可修** | 修法方向：发射前对相关 UB 区**预清零**（与 S3 提案同轴） |

---

## 3. 合并判词（三判决×结果 → 责任面）

| S1 | S2(ub/pure) | S3(order) | 责任面（下窗可写死的结论） |
|----|-------------|-----------|---------------------------|
| 只 dbeta FAIL | ub FAIL / pure PASS | 顺序无关 | **`ub` 档 `dbeta` 专属缺陷**，非 stale；下窗直接审 `ub` 档 `db_acc` 归约的**真机指令**（占卡反汇编/单步） |
| 只 dbeta FAIL | ub FAIL / pure PASS | 顺序相关 | **`ub` 档 + stale**：`db_acc` 或其读源（`v_scr`/`p`）受上次发射残留污染 |
| 多项 FAIL | ub FAIL / pure PASS | 顺序相关 | **UB 区残留**共因；先按 S3 `ORDER-DIFF` 圈定区，再预清零 |
| 任意 | ub FAIL / pure FAIL | 任意 | **共享标量数学/codegen 缺陷**（与 UB 无关）⇒ 回到 `gdn_asc.py` 正文找分岔 |
| 全 PASS | — | — | **非确定性**；升为非确定性专项（重复发射统计） |

---

## 4. 耗时与中止

- **卡窗预算**：S1/S2 各 ~1 次编译+1 次发射；S3 = 5 次发射同进程。单卡一次跑完应在**分钟级**。
- **假绿中止**：任一腿未见 `gdnb[` 键 ⇒ 判 `FAIL 假绿`，**停止**（勿把 torch 回退当内核过）。
- **不可比中止**：若 S1/S2 报的环境（compat/容器）与 `ca023c2e` 不符 ⇒ 标 ⚠️ 不可比，重跑。

---

## 5. 与本窗结论的衔接

- 本窗已证（✅）：`gdnb` **0 DMA**（不同面于 E 段跨步机制）；`dbeta` 四方生成码**同构**、无共享累加缓冲、UB **先写后读无 stale 落点**；device 跑 `ub`、host 跑 `pure` ⇒ **variant 混淆**（"设备侧共享代码缺陷"前提**不成立**）。详见 `D3_DBETA_EVIDENCE.md`。
- 本窗**未能**定到具体表达式（因 `ub` 档 host 编不过、生成码无异常、复现件与 C 段真跑件缓存未对位）。
- 下窗一枪即补上这最后一层：**S1 定"坏在哪几项"、S2 定"UB 专属与否"、S3 定"是否 stale"**。

---
*D3 一枪设计件。标注：✅ 实测 / ⚠️ 引用未取 / ❓ 推断。*
