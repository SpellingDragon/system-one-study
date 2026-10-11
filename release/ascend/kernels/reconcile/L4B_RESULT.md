# L4B_RESULT — A：gradcheck 探针 nan 判据（机理+修复+负例自证+回归）；B：conv 反向件卡上缺陷归因（UB 首写/首读审计 + 下窗一枪设计）

执行代理 **L4b**（上一棒 L4 返回残缺，全量重做）｜**不占卡**（本件全部为 host 回归 + 容器真编只读审计）｜无 git。

---

## 0. 结论速览

- **A（判据机理）**：窗三 `FAIL` / 窗四 `PASS` 两次跑的是**同一份未打补丁的判据代码**（经 `.pyc` 反汇编证实），判决却相反——根因是判据的 `inf <= inf` 假绿洞 **叠加** 两次设备脏值**内容不同**（窗三 `abs_err=nan` vs 窗四 `abs_err=ana_norm=+inf`）。两窗打印都是 `max_rel=nan`（`nan/inf` 与 `inf/inf` 皆 `nan`），故"同打印不同判词"。**前提未推翻：判据确有问题。**
- **A（修复）**：非有限一律 FAIL 的补闸**已在盘上**（`ascend/gradcheck_card.py` mtime 03:59，L4 所落），本件**复核无误**并补齐负例自证（注入 nan/inf 必红、正常梯度仍绿）与三项回归（cpu 8/8、B3 三负例、pytest 340 passed）。
- **B（审计）**：`conv_bwd_asc_impl` 唯一 UB 槽 `dw_ub` **无 read-first**（源码 261→279；生成码 22→48）；前向 `gdn_asc`（ub）各槽同样 **无 read-first**（对照成立）。**但已知坏件 `gdn_bwd` 生成码也无任何 read-first** ⇒ 静态/生成码审计对本缺陷类**无判别力**，故 conv 属统一假说 **未决**（并非被证伪）。给出下窗一枪真值表与可证伪预测。

---

## A. gradcheck 探针 nan 判据

### A.0 一手事实（窗三 vs 窗四）

| 窗 | 文件:行 | 判词 |
|----|---------|------|
| 窗三 | `runs/1012-p2-13-oncard-window3-pin/artifacts/w3_gradcheck.log:39` | `GRADCHECK-gdn_delta FAIL max_rel=nan eps=0.01` |
| 窗四 | `runs/1012-p2-13-oncard-window4-pin/artifacts/w4_gradcheck.log:42` | `GRADCHECK-gdn_delta PASS max_rel=nan eps=0.01` |

两窗该件的 `fwd_keys` **完全相同**（同内核键、同路径）：

- 窗三 `w3_gradcheck.log:40` → `fwd_keys=[gdn[ascend]|h2k8v8|c8ubh]+1other`
- 窗四 `w4_gradcheck.log:43` → `fwd_keys=[gdn[ascend]|h2k8v8|c8ubh]+1other`

同窗 conv：`w4_gradcheck.log:49` `GRADCHECK-gdn_conv FAIL max_rel=5.538e+04 eps=0.01`（`fwd_keys=[gdnconv[ascend]|c8k4b2|c8scalare]+1other`，行 50）。

### A.1 机理：窗三 FAIL vs 窗四 PASS

**读到的比较路径**（`ascend/gradcheck_card.py`，现盘）：`_gradcheck_op`（165-215）→ `rel = abs_err / max(ana_norm, 1e-30)`（201）→ `ok = finite and (abs_err <= ATOL + RTOL * ana_norm)`（212）。

**未打补丁的旧式**（从 `ascend/__pycache__/gradcheck_card.cpython-310.pyc`（mtime **00:28**＝窗三时代）反汇编复原；`co_names` 无 `math`/`isfinite`，返回 dict 仅 **7** 键 `('rel','abs_err','ana_norm','eps','fwd_keys','exc','ok')`）：

```
193: abs_err, ana_norm = sq_err ** 0.5, sq_ana ** 0.5
194: rel = abs_err / max(ana_norm, 1e-30)
195: return {'rel': rel, 'abs_err': abs_err, 'ana_norm': ana_norm, 'eps': eps,
           'fwd_keys': fwd_keys, 'exc': None,
196:        'ok': abs_err <= ATOL + RTOL * ana_norm}      # COMPARE_OP 1 (<=)  —— 无 finite 闸
```

（对照：现盘 `gradcheck_card.cpython-312.pyc`（mtime **03:59**）含 `math`/`isfinite`、字符串 `nonfinite=true(abs_err=`，即已打补丁版。两窗跑的正是上面这份**未打补丁**的旧式。）

**逐候选裁定**（题目给的三个候选方向）：

1. **候选①「`nan<thr`=False 的分支方向」**：方向本身**正确**（`nan` 的比较恒 `False` ⇒ 判红）。但这恰是**窗三"看起来对"**的来源——窗三是 `abs_err=nan`，`nan <= x` = `False` ⇒ **FAIL**，属"因垃圾输入而偶然判红"，不是真比较。**非否定项，是巧合项。**
2. **候选②「max 聚合吞 nan」**：**证伪**。全函数唯一的 `max` 是 `max(ana_norm, 1e-30)`（标量下界，**不是**对误差项的聚合）；误差累加走 `sq_err += float((d**2).sum())`（**求和**）。不存在"吞 nan 的最大聚合"。
3. **候选③「两侧路径差异」**：**证伪为"代码路径无差异，差异在输入内容"**。两窗同键同路（见 A.0），差异只在**设备侧非有限值的具体形态**：

   | 窗 | 设备脏值形态 | 判据演算 | 判词 |
   |----|--------------|----------|------|
   | 窗三 | `abs_err = nan` | `nan <= 1e-5 + 1e-4·x` = **False** | **FAIL** |
   | 窗四 | `abs_err = +inf` 且 `ana_norm = +inf` | `+inf <= 1e-5 + 1e-4·(+inf)` = `+inf <= +inf` = **True** | **PASS（假绿）** |

   两者 `rel = abs_err / max(ana_norm, 1e-30)` 均 = `nan/inf` 或 `inf/inf` = **`nan`** ⇒ 故**两窗打印都是 `max_rel=nan`**，此即"同打印不同判词"的机关。

   **窗四必为"双 +inf"的排除法**：要 `rel` 打印 `nan` 且 `ok=True`——
   - `abs_err` 有限、`ana_norm=inf` → `rel=0` 且 `ok=True`（会打印 `0`，不符 nan）；
   - `abs_err=inf`、`ana_norm` 有限 → `rel=inf` 且 `ok=False`（会打印 `inf`，不符 PASS）；
   - 唯 `abs_err=ana_norm=+inf` → `rel=nan` 且 `ok=True`。**唯一解。**

   **与 V2 分诊自洽**：同件在**另一次跑**（设备脏值不同）得 `rel=2.21e5 pass=False` 且数值/解析两侧 finite（窗四 `w4_triage.log` 亦见 num 侧 `maxabs` 达 1.049e7、gdn_conv num 侧达 2.09e6）⇒ 设备脏值在 **nan / +inf / 大有限** 间**随跑不定**，正是 K4 定的 device-only 缺陷群（首次发射未初始化 UB 读）在数值层的表现，而非判据路径差异。

⇒ **机理级结论**：窗三 FAIL vs 窗四 PASS = 判据存在 **`inf<=inf` 假绿洞**（补 finite 闸即修因）**叠加** 两次设备脏值内容不同（`nan` vs `+inf`）。**题目前提「判据问题」成立、未被推翻**；同时如实指出"两次输入内容不同"是同一判据给出相反判词的直接原因。

### A.2 修复（沿 D3 口径，非有限一律 FAIL）

现盘 `ascend/gradcheck_card.py` 已在 `_gradcheck_op`（200-212）加 finite 闸（**L4 于 03:59 所落，本件复核未再改盘上文件**）：

```
200: abs_err, ana_norm = sq_err ** 0.5, sq_ana ** 0.5
201: rel = abs_err / max(ana_norm, 1e-30)
207: finite = (math.isfinite(abs_err) and math.isfinite(ana_norm) and math.isfinite(rel))
209-212: return {"rel": rel, "abs_err": abs_err, "ana_norm": ana_norm,
                 "finite": finite, "eps": eps, "fwd_keys": fwd_keys, "exc": None,
                 "ok": finite and (abs_err <= ATOL + RTOL * ana_norm)}
```

- **口径一致**：与 `ascend/port910b/run_delta_card.py:295-299` 的 D3 逐项式 `fin = bool(np.isfinite(v)); ok = fin and (v <= thr)` 同构。
- **可观测性**：`run_one`（298-300）透传 `finite/abs_err/ana_norm`；`main`（395-397）在 `finite is False` 时补打印 `nonfinite=true(abs_err=… ana_norm=… => 强制 FAIL)`；`_line`（347）仍打 `max_rel=nan`（非有限值），故坏行自带形态证据。
- **不改语义**：`RTOL, ATOL = 1e-4, 1e-5`（71）未动；`_binding`/`INVALID`（`reason=no-kernel|blockers|fallback`）路径未触；异常分支（213-215）仍返回 `ok=False`。

### A.3 负例自证（注入必红 / 正常仍绿）

`ascend/kernels/reconcile/L4_probe_nanfix.py`（L4 件，本件复核运行）→ 末行 `L4-NANFIX-DONE fail=0`：

- **F-frozen**（冻结旧式 vs 现行判据，`abs_err=ana_norm=+inf`）：旧式 `ok=True`（**复现假绿病灶**），现行 `ok=False`（**补闸生效**）。
- **N-green**（八件正常梯度）：全 `PASS` 且 `finite=True`。
- **N-inf**（八件注入 inf ＝ 窗四病灶形态）：全 `FAIL max_rel=nan finite=False ana_norm=inf`。
- **N-oneinf**（单元素 +inf）：`gdn_delta FAIL finite=False`。
- **N-nan**（注入 nan）：`gdn_delta FAIL finite=False`。

⇒ **注入 nan/inf 必红、正常梯度仍绿**，自证成立。

### A.4 回归（三项，全绿）

| 命令 | 结果 |
|------|------|
| `.venv/bin/python ascend/gradcheck_card.py --target cpu` | 8 行全 PASS；`GRADCHECK-SUMMARY ops=8 pass=8 fail=0 invalid=0 kernel_used=8/8`；exit 0 |
| `.venv/bin/python ascend/kernels/reconcile/B3_probe_negatives.py` | `B3-NEG-DONE pass=3 fail=0`；exit 0（N-a `invalid=8/8`；N-b `INVALID reason=no-kernel`；N-c `FAIL max_rel=9.993e-04`） |
| `.venv/bin/python -m pytest -q -m "not integration"` | `340 passed, 3 skipped, 3 deselected, 2 warnings`；exit 0 |

---

## B. conv 反向件卡上缺陷归因（**只读审计，不改产品件**）

### B.0 手法

容器 `cann910b-h`（CANN 8.5.0 / py3.10.12 / tilelang 0.1.15，release 挂 `/work`）用 `ascend/kernels/reconcile/L4B_probe_ubaudit.py` 真编四份模具（`bwd_conv`/`fwd_conv`/`fwd_gdn`/`bwd_gdn`），取 `get_kernel_source()` 逐行编号落盘；源码侧另标行号。**两路行号并挂**。

> 生成码说明：除 `bwd_conv` 单 UB 槽被命名为 `dw_ub` 外，其余内核的 shared 缓冲被合成为单一 `buf_dyn_shmem`（`__ubuf__ uint8_t*`，基址 0），故按**元素偏移**标注槽位。

### B.1 `conv_bwd_asc_impl`（`ascend/kernels/gdn_conv_asc.py`）— 唯一 UB 槽 `dw_ub`

| UB 槽 | 源码首写 | 源码首读 | 生成码首写 | 生成码首读 | read-first? |
|-------|----------|----------|------------|------------|-------------|
| `dw_ub`（k=4） | **261** `dw_ub[j] = T.float32(0.0)` | **279** `dw_ub[j] = dw_ub[j] + ds * xv` | **22** `dw_ub[j] = 0.0f` | **48** `dw_ub[j_2] = (dw_ub[j_2] + (ds*xv))` | **否** |

- 声明：源码 254；生成码 12 `__ubuf__ float *dw_ub = (__ubuf__ float *)0;`
- 读出口：源码 **283** `DW[ch, j] = dw_ub[j]`；生成码 **54** `write_gm_bypass_dcache(...DW..., dw_ub[j_3])`
- 前向 `conv_asc_impl`（variant=`scalar`）零 `alloc_shared`：生成码 `fwd_conv.cce` 无任何 UB 符号（37 行），**无 read-first 可言**。

⇒ **`conv_bwd_asc_impl` 的唯一 UB 槽在首读前已被显式清零，无 read-first。**

### B.2 `gdn_asc` 前向（`variant="ub"`）— **对照组**（卡上 `NUM-gdn 6.35e-08` 对）

| UB 槽 | 源码首写 | 源码首读 | 生成码首写 | 生成码首读 | read-first? |
|-------|----------|----------|------------|------------|-------------|
| `st_ub` [0,8) | **227** `st_ub[i]=H0[h,j,i]` | **243** `Hist=st_ub[i]` / **251** RMW | **25** `buf[i]=H0[...]` | **36** / **40** | 否 |
| `k_ub` [8,16) | **236** `k_ub[i]=K[h,t,i]` | 252 / 267 | **32** | **41** | 否 |
| `q_ub` [16,24) | **237** `q_ub[i]=Q[h,t,i]` | 268 | **33** | **47** | 否 |
| `dec_ub` [24,32) | **235** `dec_ub[i]=T.exp(G[h,t,i])` | 251 | **31** | **40** | 否 |

（生成码槽位＝ `fwd_gdn.cce` 的 `buf_dyn_shmem` 元素偏移：`st_ub`[0,8) / `k_ub`[8,16) / `q_ub`[16,24) / `dec_ub`[24,32)。）

⇒ 前向四槽全 write-before-read，**无 read-first**，与"前向卡上对"自洽，**对照成立**（前向干净／反向存在缺陷件，判别关系本该如此）。

### B.3 `gdn_bwd`（`variant="ub"`）— **已知 dbeta 坏件**（判别力反证）

已知缺陷：K4 定 `delta_bwd` dbeta `rel=0.80`（device-only），且 `prefill=zero` 即转对。

源码：`lam_ub` 零化 **388-394**（`lam_ub[j,i] = T.float32(0.0)`，行 **394**；`dstate="gm"` 时读宿主 `Lam`，行 392）；首读 **419** `du = du + (lam_ub[j,i] + do_scr[j]*q_ub[i])*k_ub[i]`。

| UB 槽（元素偏移） | 生成码首写 | 生成码首读 |
|-------------------|------------|------------|
| `lam_ub` [0,64) | **46**（零化循环 44-48；源码 394） | **65** / **66** |
| `dec_ub` [64,72) | **53** | **65** / **83** |
| `u_scr` [72,80) | **70** | **84** |
| `dp_scr` [80,88) | **71** | **85** |
| `k_ub` [88,96) | **54** | **65** |
| `q_ub` [96,104) | **55** | **66** / **81** |
| `v_scr` [104,112) | **58** | **68** |
| `do_scr` [112,120) | **59** | **65** / **81** |

⇒ **已知坏件（dbeta 0.80 device-only）的生成码也无任何 read-first**（全部 write-before-read）。

**这是本审计最关键的发现**：静态/生成码"首写/首读"审计**对目标缺陷类无判别力**——坏件与好件在生成码层面**同形**。因此 `conv_bwd`"无 read-first"**不能证伪**统一假说，只能把 conv 判为 **未决**（UNDECIDED），而非"支持 / 证伪"。

**推论（假说需换更强口径）**：若 K4 的"未初始化 UB 读"真存在，其发生面必在**槽粒度之上**——候选口径：跨核/跨发射的 UB 地址复用、首发射地址补丁（`(__ubuf__ ...)0` 基址在首发射前的可见性）未生效、首发射的分配/同步时序。这需另设探针，非本静态审计可判。

> **附**：已知坏件 `gdn_bwd` 与好件 `fwd_gdn` 的生成码**都经 `write_gm_bypass_dcache` 写 GM 输出**（`bwd_gdn.cce` 72/75/90-92/97；`fwd_gdn.cce` 36/49/52），故 bypass-dcache 本身**也不判别**（好件同样用）——仅作独立根因候选留档（见 B.4）。

### B.4 下窗一枪设计（**写进报告，不执行**）

**目的**：**一次发射前预热/清零**，同批复判三缺陷——`gradcheck-gdn_conv`、`gradcheck-gdn_delta`、`delta_bwd` 的 `dbeta`。

**三臂**（每缺陷各跑一遍）：

- **A0 baseline**：现口径单发射（复现当前判词）。
- **A1 UB 预清零**：被测内核发射前，把该内核用到的 UB 区段显式写 0（同尺寸哑发射或 `T.alloc_shared` 前置 zeroing 模具），再发射判件。
- **A2 预热发射**：先跑一次同形哑发射（预热），再发射判件。

**判别规则**：

- A1 转绿 ⇒ 字面"未初始化 UB 读"（统一假说 **+1**，修法＝UB 预清零，与 M4 同法）。
- A1 不转、A2 转 ⇒ 非 UB 内容问题，指向**首发射类**（地址补丁 / 分配 / DCache 一致性），修法另议。
- A1 与 A2 均不转 ⇒ **独立根因**。

**真值表（含可证伪预测）**：

| 缺陷 | 现状（窗四/窗三） | A1 UB 预清零 预测 | A2 预热 预测 | 判读 |
|------|-------------------|-------------------|--------------|------|
| `dbeta`（`bwd_gdn` / delta_bwd） | 0.80（**已证** `prefill=zero` 转对） | **转绿**（锚点，已证） | 转绿 | **支持**统一假说（锚） |
| `gradcheck-gdn_delta` | FAIL 2.21e5（曾 PASS/假绿 `nan`） | 待测 | 待测 | A1 转绿⇒**支持**；不转⇒**独立根因** |
| `gradcheck-gdn_conv` | FAIL 5.538e4 | **预测不转绿** | 待测 | 见下 |

**对 conv 的强预测（诚实标注，可证伪）**：`conv_bwd` 的唯一 UB 槽 `dw_ub` 已在源码 **261** / 生成码 **22** 显式清零，**UB 预清零对其"无可修之物"** ⇒ 预测 **A1 不转绿** ⇒ conv 指向**独立根因**。独立根因候选：输出经 `write_gm_bypass_dcache` 写 GM 后，FD 复用同缓冲读取的**一致性/可见性**（E3 分诊见 num 侧 `maxabs` 2.09e6 而 fwd 输出 finite 且小，指向 FD 数值侧非确定）。

**证伪条件**：若实测 conv 的 A1 **转绿**，则 B.3 的"静态审计无判别力"论断被推翻，须回炉重审计（说明 read-first 发生在生成码**之上**）。

**下窗产出**：`ascend/kernels/reconcile/L4B_probe_oneshot.py`（本件已备骨架，**未执行**）逐件打印三行 `L4B-ONESHOT <op> arm=<A0|A1|A2> verdict=… rel=…`，末尾汇 `L4B-ONESHOT-TABLE`。

---

## 附：产物与边界

- **新建**：`ascend/kernels/reconcile/L4B_RESULT.md`（本报告）、`ascend/kernels/reconcile/L4B_probe_ubaudit.py`（UB 首写/首读审计，已执行）、`ascend/kernels/reconcile/L4B_probe_oneshot.py`（下窗一枪骨架，未执行）。
- **只读复核（未改）**：`ascend/gradcheck_card.py`、`ascend/kernels/gdn_conv_asc.py`、`ascend/kernels/gdn_asc.py`。
- **未动**：`autograd_asc.py` / `run_delta_card.py` / `train_step*.py` / `window*_*.sh` / trunk / `runs/**`（只读）/ 任何 `tasks.md`；**无 git**。
