# D3_DBETA_EVIDENCE.md — P1-1z / D3：GDN 反向 `dbeta` device-only 缺陷取证

> 执行代理：**D3**　｜　**不占卡**（host 编译/取证腿）　｜　变更：p2-13 / P1-1z
> 本件只**钉责任面**，不下因果、不做修复、不做真机判决（留给下窗一枪）。
>
> 标注图例：**✅ 实测原文**（本会话真取到、可复跑）｜**⚠️ 提及未取**（引用他人台账，本会话未复取）｜**❓ 推断**（未经判决实验）

---

## 0. 结论速览（TL;DR）

| # | 待判问题 | 结论 | 三态 |
|---|---------|------|------|
| A | 该件是否与 S3/E 段"多突发跨步 UB 残留"**同面**？ | **不同面**。`gdnb[ascend]` 正文 **0 条 DMA**（无任何 `DataCopy`/`copy_gm_to_ubuf`），全部 GM→UB 是**标量逐元素**赋值 ⇒ 无 `blockCount` 可谈，E 段"gap/pitch 多突发"机制**结构上不适用** | ✅ 结构实测；❓"因此不是同源"仍属推断（真机未判） |
| B | `dbeta` 归约是否有独立/可疑写法？ | **无**。源码、`ascend/pure`、`ascend/ub`、`cpu/pure` **四份** `dbeta` 表达式**同构**：`db_acc += du*(v - p)`，无除法、无 beta 入分母、无缺失的 `float32` 累加、不依赖 `T.copy` | ✅ 三方生成码原文 + 源码原文 |
| C | `dbeta` 与 `dg`/`dh0` 是否共用累加缓冲/UB 区？ | **不共用**。`db_acc`/`dq_acc`/`dk_acc`/`da_acc` **四者皆为寄存器** `float`（非 UB）；`dg` 走寄存器 `da_acc`，`dh0` 读 UB `lam_ub`；UB 8 槽**全部先写后读**，`lam_ub` 入口清零一次 ⇒ **本件无 stale 落点** | ✅ UB 槽位 firstW/firstR 实测 |
| D | host 过、device 挂 ⇒ "共享代码里的设备侧缺陷"？ | **此前提被证伪（variant 混淆）**：host 腿 `_variant()`→`pure`，设备 C 段腿→`ub`；`pure` 设备腿从未跑、`ub` host 腿**根本编不过**。两条腿**跑的不是同一档** | ✅ `--target cpu --variant ub` 编译报错原文 |
| E | 缺陷到底在哪一层？ | **未能定到具体表达式**。三方生成码的 `dbeta` 代数完全一致、无 DMA、无 stale 落点；剩余可能只在 `ub` 档的**真机执行态**（含 UB 访问/同步），**必须占卡**才能判 —— 见 `D3_NEXTSHOT_dbeta.md` | ❓ |
| F | `dv/dg/dh0` 等状态？ | **未知**（改前 FAIL 早退，非"通过"）。已在本件（Task 1）把工装改成**容错逐项判**，下窗一次全出 | ✅ 工装已就绪 |

**一句话**：结构上 `dbeta` 的算式在四个产物里**一模一样**，而 device 跑的是 **host 从未验证过的 `ub` 档** —— 责任面从"某处表达式写错"收窄到"**`ub` 档设备执行态**"，但**未**能进一步钉到具体表达式。

---

## 1. 工装先行（Task 1）：FAIL 早退 → 容错逐项判

### 1.1 为什么必须改

`w3_deltabwd.log` 原文（⚠️ 引用台账，用户提供）：
```
NUM-delta_fwd PASS rel_o=7.51e-08 rel_s=6.03e-08
NUM-delta_bwd FAIL first=dbeta rel=5.997e-01 absdiff=4.435e+00 thresh=2e-04
```
改前 `_bwd_leg` **只打首个超阈项就 return** ⇒ 无法区分"**只坏 dbeta**"与"**dbeta 只是第一个坏的**"；`dv/dg/dh0` 状态**未知**（不是"通过"）。

### 1.2 改动（唯一白名单件：`release/ascend/port910b/run_delta_card.py`，md5 `2fb3e73efcb1793b73fa9d93af840428`）

- **docstring（L11-17）** 增补 `ITEM` 行说明与容错口径。
- **`_bwd_leg`（L277-309）**：把原早退块换成逐项循环（每项**独立** `try/except`），逐字关键段：

```python
277    # 容错逐项判（P1-1z / D3）：FAIL **不再早退**。改前只打首个超阈项就 return，导致
278    # "只坏 dbeta" 与 "dbeta 只是第一个坏的" 无法区分（w3 窗 dbeta FAIL 后 dv/dg/dh0 状态未知）。
279    # 现在每项**独立** try/except 求 rel 并各打一行 `ITEM <name> rel= <v> verdict= <PASS|FAIL>`，
280    # 任一项的异常/数值问题都不连坐其余项；末行 `NUM-delta_bwd` 聚合判据与阈值（2e-4）**逐字不变**。
281    vals, diff = {}, {}
282    for it in order:
283        try:
284            if it == "adjoint":
285                vals[it] = _adjoint_fd_check(gd)
286                diff[it] = None
287            else:
288                vals[it] = relerr(meas[it], ref[it])
289                diff[it] = float(np.max(np.abs(np.asarray(meas[it]) - np.asarray(ref[it]))))
290        except Exception as e:  # 单项炸不连坐：记 inf 并继续（聚合口径下 inf>thr 必判 FAIL）
291            vals[it] = float("inf")
292            diff[it] = None
293            print(f"ITEM-ERR {it} {type(e).__name__}: {str(e)[:160]}")
294        v = vals[it]
295        fin = bool(np.isfinite(v))
296        ok = fin and (v <= thr[it])          # 逐项 verdict：非有限值不算过（防"nan 当 PASS"）
297        rtxt = format(v, ".3e") if fin else "nan/inf"
298        note = "" if fin else f" (非有限;聚合口径 v>{thr[it]:.0e} 不判 FAIL)"
299        print(f"ITEM {it} rel= {rtxt} verdict= {'PASS' if ok else 'FAIL'}{note}")
300
301    first = next((it for it in order if vals[it] > thr[it]), None)   # ← 聚合口径与改前逐字同
```

- **阈值与聚合口径不动**：`thr = {it: TOL_F32(=2e-4) for ...}`、`thr["adjoint"]=TOL_ADJ(=1e-8)`（L273-275）与 `first = next(... vals[it] > thr[it] ...)`（L301）逐字保留。

> **逐项行新增口径（自申报）**：逐项 `verdict` 把**非有限值（nan/inf）判 FAIL**（防空值蒙混），这是逐项行**新增**的保守口径；**聚合行**仍是原 `v > thr` 规则（保持逐字不变）。二者在正常有限值下完全等价。

### 1.3 验证（host 复跑，✅ 实测）

命令与输出（`cd release/ascend/port910b && .venv/bin/python run_delta_card.py --target cpu --bwd`）：
```
NUM-delta_fwd PASS rel_o=7.49e-08 rel_s=4.69e-08
ITEM dq rel= 7.531e-08 verdict= PASS
ITEM dk rel= 6.718e-08 verdict= PASS
ITEM dv rel= 6.676e-08 verdict= PASS
ITEM dbeta rel= 1.007e-07 verdict= PASS
ITEM dg rel= 9.716e-08 verdict= PASS
ITEM dh0 rel= 6.176e-08 verdict= PASS
ITEM adjoint rel= 6.222e-10 verdict= PASS
NUM-delta_bwd PASS dq=7.53e-08 dk=6.72e-08 dv=6.68e-08 dbeta=1.01e-07 dg=9.72e-08 dh0=6.18e-08 adjoint=6.22e-10 rel_max=1.01e-07
```
- **7 条 `ITEM` 行全 PASS**，六项量级仍 `1e-7` ⇒ **改造不改变结论** ✅
- **聚合 PASS 行与改前逐字同**；**fwd 行与改前逐字同**（`rel_o=7.49e-08 rel_s=4.69e-08`）✅
- **六项 host 数值不变**：`dq=7.53e-08 dk=6.72e-08 dv=6.68e-08 dbeta=1.01e-07 dg=9.72e-08 dh0=6.18e-08` ✅

**负例（×1.01 逐项可指认，✅ 实测）**：对 `dq/dk/dv/dbeta/dg/dh0` 各跑 `--neg-perturb <ITEM>`，被扰动项是**唯一** FAIL，`first=` 正确指认，聚合 FAIL 行与 C4_RESULT 记录值逐字一致（如 `dbeta` → `first=dbeta`）。

### 1.4 回归红线（✅ 实测）

| 红线 | 结果 |
|------|------|
| `pytest tests -q -m "not integration"` | **340 passed**, 3 skipped, 3 deselected（RC=0） |
| fwd 判决行与改前逐字同 | ✅ `NUM-delta_fwd PASS rel_o=7.49e-08 rel_s=4.69e-08` |
| 六项 host 数值不变 | ✅ 同上 |

---

## 2. 生成码取证（Task 2）：`gdnb[ascend]` 反向内核

### 2.1 归档 `sys1_kernels.tgz` 对位（✅ 实测）

- 文件：`runs/1012-p2-13-oncard-window3-pin/artifacts/sys1_kernels.tgz`（**574,211 B**）。
- 解出 `0.1.15/linux-aarch64/kernels/` 下 **48** 枚内核目录（`ls | wc -l` = 48）✅。
- 逐一 `grep 'void gdn_bwd_impl_kernel('` ⇒ **全档仅 1 枚** gdnb：
  `…/kernels/3d931bf04c97cf56a99632d8c33f59a2a5e032af0dba9fc6bb04aa8af1325990/device_kernel.asc`

> ⚠️ **归档件不是 C 段真跑件**（重要caveat）：归档 gdnb 的循环维是 **16×16**（`j<16/i<16`，`db_acc` 用 `v_scr` 偏移 **336**），属 **A 段 trainstep 腿** 的模具；而 **C 段** `run_delta_card.py --bwd` 用的是 `H,TT,DK,DV=2,8,8,8`（**8×8**）且用**自己的 `mktemp -d` 缓存**（见 `window3_card.sh` C 行），**该缓存从未入库**。
> ⇒ 为对位 C 段真跑件，本件在容器 `cann910b-h`（compat `ca023c2e`）**重编** 8×8（见 §3）。

### 2.2 归档 gdnb（16×16）关键原文（✅ 实测，行号原样）

```
24:  float db_acc = float(0x0p+0f/*0.000000e+00*/);
42:  db_acc = float(0x0p+0f/*0.000000e+00*/);
51:    db_acc = float(0x0p+0f/*0.000000e+00*/);
73:      db_acc = (db_acc + (du * (((__ubuf__ float*)buf_dyn_shmem)[(j_2 + 336)] - p)));
75:    tl::write_gm_bypass_dcache(((__gm__ float*)DBeta + (…)), db_acc);
```

### 2.3 C 段真跑档（8×8 ub）关键原文（✅ 实测，md5 `8b11e54ed9d56b995ad7027db7946ea9`）

来源：容器内 `--target ascend --bwd --no-run` 真编复现（见 §3.1）。
```
 6:  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
44:  for (int32_t j = 0; j < 8; ++j) {
45:    for (int32_t i = 0; i < 8; ++i) {
46:      ((__ubuf__ float*)buf_dyn_shmem)[((j * 8) + i)] = float(0x0p+0f/*0.000000e+00*/);   // lam_ub 入口清零
53:      ((__ubuf__ float*)buf_dyn_shmem)[(i_1 + 64)] = expf(G[…]);   // dec_ub ← 标量逐元素
54:      ((__ubuf__ float*)buf_dyn_shmem)[(i_1 + 88)] = K[…];         // k_ub   ← 标量逐元素
55:      ((__ubuf__ float*)buf_dyn_shmem)[(i_1 + 96)] = Q[…];         // q_ub   ← 标量逐元素
58:      ((__ubuf__ float*)buf_dyn_shmem)[(j_1 + 104)] = V[…];        // v_scr  ← 标量逐元素
59:      ((__ubuf__ float*)buf_dyn_shmem)[(j_1 + 112)] = DO[…];       // do_scr ← 标量逐元素
65:      p = (p + (…buf_dyn_shmem)[(i_2 + 64)] * Hist[…] * (…buf_dyn_shmem)[(i_2 + 88)]);
66:      du = (du + (((…buf_dyn_shmem)[((j_2 * 8) + i_2)] + ((…buf_dyn_shmem)[(j_2 + 112)] * (…buf_dyn_shmem)[(i_2 + 96)])) * (…buf_dyn_shmem)[(i_2 + 88)]));
68:      uj = (bt * (((…buf_dyn_shmem)[(j_2 + 104)] - p)));
69:      dp = (float(0x0p+0f) - (bt * du));
70:      ((__ubuf__ float*)buf_dyn_shmem)[(j_2 + 72)] = uj;           // u_scr
71:      ((__ubuf__ float*)buf_dyn_shmem)[(j_2 + 80)] = dp;           // dp_scr
72:      tl::write_gm_bypass_dcache(((__gm__ float*)DV_ + …), (bt * du));
73:      db_acc = (db_acc + (du * (((__ubuf__ float*)buf_dyn_shmem)[(j_2 + 104)] - p)));   ★ dbeta 归约
75:    tl::write_gm_bypass_dcache(((__gm__ float*)DBeta + …), db_acc);   ★ dbeta 落盘
88:        ((__ubuf__ float*)buf_dyn_shmem)[((j_3 * 8) + i_3)] = (((…buf_dyn_shmem)[(i_3 + 64)] * dst);   // lam_ub 就地更新 Λ
```
UB 槽位分配（按 `alloc_shared` 次序，dk=dv=8）：`lam_ub[0,64) dec_ub[64,72) u_scr[72,80) dp_scr[80,88) k_ub[88,96) q_ub[96,104) v_scr[104,112) do_scr[112,120)`。

### 2.4 逐项核查结论

**(a) DMA / blockCount**：正文 **0 条** DMA 指令（`DataCopy`/`asc_copy_gm2ub`/`copy_gm_to_ubuf`/`copy_ubuf_to_gm` 计数 = 0，见 §3.2 表 C1 列）。GM→UB 全部是 `((__ubuf__ float*)buf_dyn_shmem)[…] = <gm 读>` 型**标量逐元素**；输出走 `tl::write_gm_bypass_dcache`。**⇒ 无多突发、无 `blockCount` 概念 ⇒ 与 E 段"多突发跨步"结构上不同面。**

**(b) dbeta 归约原文**：`db_acc = (db_acc + (du * (v_scr[j] - p)))`（L73）。特征：
- 每 token 入口 `db_acc = 0`（L51）—— 单 token 局部归约，不跨 token 累。
- `du`（L66 寄存器）、`v_scr[j]`（V 的 UB 影）、`p`（L65 寄存器）全部 `float32` 寄存器/UB，**无除法、无 beta 入分母、无 fp64 混入**。
- **与数学式 `dβ_t = Σ_j du[j]·(v_t[j]−p_t[j])`（`gdn_asc.py` L300）同构**，无额外/缺失项。

**(c) 是否与 `dg`/`dh0` 共用累加缓冲/UB 区**：**不共用**。
- 四个累加器 `db_acc / dq_acc / dk_acc / da_acc` 全部是**寄存器** `float`（ub .asc L21-24；pure/cpu 同），**不在 UB**；
- `dg` 用寄存器 `da_acc`（L87）、`dh0` 读 UB `lam_ub`（L97）——**与 dbeta 无共享缓冲**；
- UB 8 槽**全部先写后读**（§3.2 表：firstW < firstR），`lam_ub` 入口**一次性清零** ⇒ **本件无 stale 读落点**。

> **Task 2 结论**：结构上 `gdnb[ascend]` **不在** E 段"跨步/stale"同面；`dbeta` 算式忠实、无共享缓冲、无 stale 落点。（**≠ 因果**：真机 `ub` 档执行态未判。）

---

## 3. 差分定位（Task 3）：三份生成码对照

### 3.1 三份产物（✅ 全部本会话真取）

| 料号 | 生成途径 | md5 | 行数 | 对应腿 |
|------|---------|-----|------|-------|
| `ascend/ub` | 容器 `cann910b-h`：`--target ascend --bwd --no-run`（`_variant()`→`ub`） | `8b11e54ed9d56b995ad7027db7946ea9` | 101 | **设备 C 段真跑档**（本窗判决件） |
| `ascend/pure` | 容器：`--target ascend --bwd --no-run --variant pure` | `5393f4be0f55480a8ff0d6549a172c53` | 108 | 设备可编、**本窗未跑** |
| `cpu/pure` | host：`--target cpu --bwd`（`_variant()`→`pure`） | `7092e74e9ab6f21ef9619880ff57ad9b` | 124 | **host 复验腿真跑档**（bug 未复现） |

复现命令（一条）：
```
cd release && ./.venv/bin/python ascend/kernels/reconcile/D3_probe_gendiff.py   # 需 docker cann910b-h
# 或离线复用已取产物：
./.venv/bin/python ascend/kernels/reconcile/D3_probe_gendiff.py --no-docker \
  --asc-ub /tmp/d3_c_real/gdnb_ub_8x8.asc --asc-pure /tmp/d3_c_real/gdnb_pure_8x8.asc \
  --cpu /tmp/d3_c_real/gdnb_cpu_pure.cu
```

### 3.2 三方结构差分表（探针 C1/C2/C3/C4 输出，✅ 实测）

**C1 DMA 计数**（该件是否与 E 段"多突发跨步装填"同面）：

| 产物 | DMA 条数 | dbeta 归约行号 |
|------|---------|---------------|
| `ascend/ub` | **0** | 24, 42, 51, **73**, 75 |
| `ascend/pure` | **0** | 23, 41, 50, **66**, 100 |
| `cpu/pure` | **0** | 37, 56, 64, **80**, 114 |

⇒ **三份均 0 条 DMA**，`dbeta` 相关行在**同一逻辑位置**（初始化 / 每 token 清零 / 归约 / 落盘）。

**C4 dbeta 归约原文逐字对照**：

```
ascend/ub   :73  db_acc = (db_acc + (du * (((__ubuf__ float*)buf_dyn_shmem)[(j_2 + 104)] - p)));
ascend/pure :66  db_acc = (db_acc + (du * (vj - p)));
cpu/pure    :80  db_acc[0] = (db_acc[0] + (du[0] * (vj[0] - p[0])));
```
⇒ **代数完全同构**：`db_acc += du*(v − p)`。差异**仅在 `v` 的来源**：ub 走 UB 影 `v_scr[j]`，pure/cpu 走 GM `vj`。

**C2/C3 UB 槽位（首写/首读）**：

| 产物 | 结果 |
|------|------|
| `ascend/ub` | 8 槽 all **先写后读**（lam 46<65、dec 53<65、k 54<65、q 55<66、v 58<68、do 59<66、u 70<84、dp 71<85）⇒ **无 stale 落点** |
| `ascend/pure` | 无 `alloc_shared`（0 处 `buf_dyn_shmem`）⇒ 该档**无 UB 槽**（Λ 在 GM `Lam`，走 `read/write_gm_bypass_dcache`） |
| `cpu/pure` | 无 UB，Λ 在 GM `Lam`（L59 清零 / L75 读 / L111 就地写） |

**三方结构性差异（仅"存哪、怎么循环"，非"算什么"）**：

| 维度 | `ascend/ub`（设备真跑） | `ascend/pure`（未跑） | `cpu/pure`（host 真跑） |
|------|------------------------|----------------------|------------------------|
| Λ 状态存哪 | **UB** `lam_ub[0,64)`，趟 2 就地更新 | GM `Lam`（RMW） | GM `Lam`（RMW） |
| V/K/Q/DO/dec 暂存 | **UB** 各槽 | 直接读 GM | 直接读 GM |
| 循环结构 | **两趟**：趟 1(j 外)算 u/dv/dbeta；趟 2(i 外)算 dq/dk/dg | **单趟** j 外包 | **单趟** j 外包 |
| dq/dk/dg 归约 | i 外层**寄存器**累加（单存，无 RMW） | j==0 **首存** / 否则 GM RMW | 同 pure |
| dbeta 归约 | 寄存器 `db_acc`（算 v_scr） | 寄存器 `db_acc`（算 vj） | 寄存器 `db_acc` |

### 3.3 关键发现：**variant 混淆**（Task 3 最重要产出，✅ 实测）

`run_delta_card.py` L60-63：
```python
def _variant():
    if ARGS.variant:
        return ARGS.variant
    return "pure" if ARGS.target == "cpu" else "ub"
```
⇒ **`--target cpu`（host 复验腿）走 `pure`；`--target ascend`（设备 C 段腿）走 `ub`**。

而 host 上 **`ub` 档根本编不过**（✅ 实测 `--target cpu --variant ub`）：
```
InternalError: Check failed: (!dyn_shmem_size.defined()) is false:
               Only one dynamic shared memory allocation is allowed.
```
（cpu 后端只许一处 dynamic shared；`ub` 档要 8 个 `alloc_shared` ⇒ 报错。）

> **推论（这不是因果，但直接推翻一个前提）**：用户给的一手事实里"**host 六项全 1e-7 / device `dbeta` 挂 60%** ⇒ 缺陷在设备侧共享代码"——**这个前提不成立**。因为：
> - host 跑的是 `pure`，device 跑的是 `ub`；
> - `pure` 设备腿**从未跑过**，`ub` host 腿**根本编不过**。
> ⇒ "host 过 / device 挂"的对比**跨了两个不同档**，"设备侧缺陷"与"`ub` 档缺陷"在证据上**不可分**。责任面应记为"**`ub` 档设备执行态**"，而非"共享代码的设备侧 bug"。

### 3.4 `dbeta` 在 ascend 正文是否有独立写法？

**没有**（✅）。`gdn_asc.py` 反向正文中 `dbeta` 两档各**一行**、且与数学式同构：
- `ub` 档（`gdn_asc.py` L425）：`db_acc = db_acc + du * (v_scr[j] - p)`
- `pure` 档（`gdn_asc.py` L463）：`db_acc = db_acc + du * (vj - p)`
- 归约落盘 `DBeta[h,t] = db_acc`（L426/L475）两档一致。

用户设想的候选根因——"ascend 正文里 `dbeta` 少了某个 `float32` 累加 / 依赖 `T.copy` 语义"——**在本件生成码层面被否证**：四方 `dbeta` 表达式同构、均 `float32` 寄存器、均无 `T.copy`/DMA 依赖。

---

## 4. 我没能定到哪一层，为什么（诚实交代）

**结论：未能把缺陷钉到具体表达式或代码行；责任面停在"`ub` 档设备执行态"。**

原因（按重要性）：
1. **无法在 host 复现失败档**：真跑失败的 `ub` 档在 cpu 后端**编译即报错**（无论证多动态共享），⇒ host 侧**没有任何可执行的 `ub` 样例**可供对拍定位。**这是本件最大的取证盲区**，且是**工具链限制**，非本轮可通过改工装绕过。
2. **生成码层面 `dbeta` 无异常**：`ascend/ub` 与 `pure` 的 `dbeta` 代数同构、无 DMA、无共享累加缓冲、8 槽先写后读无 stale 落点 ⇒ **源码/生成码层面找不到 `dbeta` 专属缺陷**。⇒ 若缺陷存在，只可能在 \(a\) `ub` 档生成的**设备指令/UB 访问时序**在真机上的行为，或 \(b\) 我的复现件与 C 段真跑件的**环境差异**（见第 3 点）。
3. **复现件与 C 段真跑件的对位存疑**：C 段用**独立 `mktemp -d` 缓存**（未入库），本件是在 `cann910b-h`（compat `ca023c2e`）**重编** 8×8 得到。虽然键（`ascend|gdnb[ascend]|h2k8v8|c2ubz`）与维度一致，但**不能 100% 断言**与 C 段当时那枚逐字节相同（缓存未留档）。⇒ ❓ 待下窗用"同一次实验同时打印生成码摘要 + 判决"来锁死。

**⇒ 这正是 `D3_NEXTSHOT_dbeta.md` 要补的三件事**：①一次跑全六项（工装已就绪）；②`ub` 档 UB **预清零**后 `dbeta` 是否改善（判"同源/stale"）；③若多项坏，是否共指同一 UB 区。

---

## 5. 复跑命令清单（全部可溯源）

```bash
REL=/Users/pengweiye/Documents/codes/system-one/release

# (1) 工装验证：host 全六项 + fwd（应全 PASS，fwd 行逐字同）
cd $REL/ascend/port910b && $REL/.venv/bin/python run_delta_card.py --target cpu --bwd \
  | grep -E '^(ITEM|NUM-delta)'

# (2) 负例：某项 ×1.01 ⇒ 该项唯一 FAIL，first= 指认正确
$REL/.venv/bin/python run_delta_card.py --target cpu --bwd --neg-perturb dbeta | grep -E 'NUM-delta_bwd'

# (3) 回归红线
cd $REL && ./.venv/bin/python -m pytest tests -q -m "not integration"   # 期望 340 passed

# (4) 生成码取证（三份，需 docker cann910b-h）
cd $REL && ./.venv/bin/python ascend/kernels/reconcile/D3_probe_gendiff.py

# (5) host 上 ub 档不可编的实证（variant 混淆）
cd $REL/ascend/port910b && $REL/.venv/bin/python run_delta_card.py --target cpu --bwd --variant ub 2>&1 \
  | grep -i 'dyn_shmem\|Only one dynamic'
```

---
*D3 取证件。三态标注：✅ 实测 / ⚠️ 引用未取 / ❓ 推断。同源假设未判决前一律 ❓。*
