# K4_RESULT.md — P1-1z / K4：设备腿**变体选择**（把 GDN 反向 `dbeta` 缺陷关掉）

> 执行代理：**K4**　｜　**不占卡**（host 双证 + 容器只编腿）　｜　变更：p2-13 / P1-1z
> 标注：**✅ 实测原文**（本会话真取、命令可复跑）｜**⚠️ 引用**（引他人台账）｜**❓ 推断**（未判决）
> 本件**未改动任何产品码**（理由见 §5），新增 `K4_probe_variant_leg.py` 机械复算前提。

---

## 0. 一句话结论（含台账需要的"生产训练步是否受影响"明确回答）

1. **"设备腿走 pure"这个前提不成立 —— 设备腿走的是 `ub`**，且**早在本窗之前**（≥窗二 C4）就已如此。
   `run_delta_card --target ascend --bwd` 的设备腿**已经是 ub**，用户要求的那条"把设备腿统一选 ub"的修复**代码里已存在**，属**空改动**（无码可改）。✅ 三条独立证据见 §2。
2. **生产训练步（`train_step*.py` → `autograd_asc.gdn_delta`）在设备上走 `ub`**（默认；env 全库无设置）。
   **⇒ `dbeta` 缺陷**（见 §4）**在 `ub` 档上同样存在**（`ub` 的 fresh 跑 FAIL、prefill 敏感），
   与判决腿 V3 是**同一个变体**。
   ⇒ **结论：影响真实训练**（缺陷不是"只影响判决腿"）—— **升级为高危**；真机数值确认留下窗。

---

## 1. 三条调用链的**变体落点**（file:line，全部 ✅ 实测）

变体选择链的**唯一决策函数**：`_asc_variant()`（`ascend/kernels/gdn_asc.py:57-62`）
```python
raw = (variant or os.environ.get(ENV_VARIANT) or "ub").strip().lower()
#      显式参数      >   env SYS1_GDN_VARIANT   >   缺省 "ub"
```

| # | 调用链 | 变体落点（file:line） | 实际落档 |
|---|--------|----------------------|---------|
| ① | **判决腿** `run_delta_card --target ascend --bwd` | `_variant()`（`ascend/port910b/run_delta_card.py:60-63`）→ 传 `gdn_asc.backward(..., target="ascend", variant=_variant())`（`:243-244`） | **ascend→`ub`**（cpu→`pure`） |
| ② | **生产腿** `train_step*.py` → `autograd_asc.gdn_delta` → `_GdnFn` → `gdn_kernel.backward` | `_GdnFn.backward` 调 `gdn_kernel.backward(q,k,v,g,beta,dout,ctx.target)`（`ascend/autograd_asc.py:378`）**不传 variant** ⇒ `plan_backward(..., variant=None)`（`gdn_asc.py:574/581`）⇒ `_asc_variant(None)` | **`ub`**（env 未设；全库无 `SYS1_GDN_VARIANT` 设置点） |
| ③ | 产品 home 内核件 | `plan()` 设备分支 `kwargs["variant"]=_asc_variant(variant)`（`gdn_asc.py:554-560`）；`plan_backward()` `var=_asc_variant(variant)`（`:581-583`） | 同上，缺省 `ub` |

**判决腿落点原文（`run_delta_card.py:60-63`，逐字）**：
```python
def _variant():
    if ARGS.variant:
        return ARGS.variant
    return "pure" if ARGS.target == "cpu" else "ub"
```

**生产腿入口原文**：
- `ascend/train_step.py:362`：`out_g = ops.gdn_delta(qg, kg, vg, gate, beta, out_dtype=out_dtype, target=target)`
- `ascend/train_step_real.py:692`：`o = self.d.gdn_delta(q, k, v, gate3, beta)` → `:505-506` `ops.gdn_delta`
- `ascend/autograd_asc.py:590-594` `gdn_delta` → `_GdnFn.apply(...)`；`_GdnFn.forward:361` / `backward:378` 均**不传 variant**
- ⇒ 变体 = `_asc_variant(None)` = **`ub`**

> **明确回答（写台账）**：**生产训练步在设备上走 `ub`**。生产腿**不**传 variant、env 全库无设置点 ⇒ 恒 `ub`。
> 判决腿（V3）**也**走 `ub`（不是 pure）。⇒ 两者**同档** ⇒ `ub` 档的 `dbeta` 缺陷**必然在生产路径上**。

### 1.1 为什么用户"设备腿走 pure"的推断是错的（同源，一句话）
用户据 **`dbeta` 单值**（V3`s=8.002e-01` == pure`s=8.002e-01`）反推"设备腿= pure"。但：
- **`ub` 的 `none`（fresh）指纹 `e609a6ab` 本就等于 pure 的指纹 `e609a6ab`** ⇒ **fresh-`ub` 与 `pure` 的 `dbeta` 逐位同**（`ub` 的 stale 态恰好撞上 pure）；
- 其余五项是**变体指纹**：V3 的 `dq/dk/dg` 与 V4-ub **逐字同**、与 V4-pure **异**（见 §2 E2）。
⇒ `dbeta` 与 pure 同值是**"同 prefill 态同值"**，**不是**"V3 走了 pure"。D3 原记载"设备=ub"**是对的**，本轮"更正"才是误判。

---

## 2. 证据（三条独立线，全部 ✅ 实测 / 可复跑）

### E1 代码（host，秒级）
`run_delta_card.py` 全文 `md5=2fb3e73efcb1793b73fa9d93af840428`（与 D3 §1.2 记录**同**），
`_variant()` 对 ascend 返回 `"ub"`（§1 原文）。**D3 对白名单件的唯一改动是 `_bwd_leg` 容错逐项循环（D3 §1.2），未动 `_variant()`** ⇒ 该选择**早于**窗二（`window2 notes:63` 已记设备键 `c2ubz`）。

### E2 运行指纹（V3 vs V4 双变体，逐字）
| item | **V3**（设备腿真跑） | V4-ub（D3 探针） | V4-pure |
|---|---|---|---|
| dq | 6.999e-08 | **6.999e-08** | 7.551e-08 |
| dk | 9.322e-08 | **9.322e-08** | 6.437e-08 |
| dv | 6.629e-08 | 6.629e-08 | 6.629e-08 |
| dbeta | 8.002e-01 | 1.018e-07 | 8.002e-01 |
| dg | 1.016e-07 | **1.016e-07** | 9.571e-08 |
| dh0 | 6.176e-08 | 6.176e-08 | 6.176e-08 |

`V4-ub` 的 `ORDER-DIFF` **只列 `item=dbeta`**（`w4_d3_ub.log` 原文）⇒ **`dq/dk/dg/dv/dh0` 与 prefill 无关、是纯变体指纹**。
V3 = `['dq','dk','dv','dg','dh0']` 与 V4-ub **逐字同**，`dq/dk/dg` 三项与 V4-pure **全异** ⇒ **V3 只能落在 `ub`**。
（机械复算：`K4_probe_variant_leg.py`，`判词: 设备腿走 ub（结论成立）`, RC=0 ✅）

### E3 编译键（容器 `cann910b-h`，只编不跑；compat 口径见注）
```
$ docker exec cann910b-h bash -lc 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; export ASCEND_NPU_ARCH=dav-2201; \
    cd /work/ascend/port910b && TILELANG_CACHE_DIR=$(mktemp -d) python3 run_delta_card.py --target ascend --bwd --no-run'
NUM-delta_fwd PASS rel=n/a (compile-only)
NUM-delta_bwd PASS rel=n/a (compile-only) key=ascend|gdnb[ascend]|h2k8v8|c2ubz      ← 设备腿默认 = ub
# 负例（证明确有开关）：
... run_delta_card.py --target ascend --bwd --no-run --variant pure
NUM-delta_bwd PASS rel=n/a (compile-only) key=ascend|gdnb[ascend]|h2k8v8|c2purez     ← 强制 pure = pure
```
> 注：容器 `/work` 是 host `release` 的 bind mount，`/work/ascend/port910b/run_delta_card.py` md5 `2fb3e73e` == host。
> 本容器 CANN 目录为 `cann-8.5.0`（窗四台账口径为 `cann-8.5.2`）；**只编不跑**，与数值无关。

### E4 host 双证（before == after，因为本件未改任何产品码）
| 判据 | 命令 | 结果 |
|------|------|------|
| host 反向六项 | `.venv/bin/python ascend/port910b/run_delta_card.py --target cpu --bwd` | 六项 `1e-7` 全 PASS（`dq=7.531e-08 dk=6.718e-08 dv=6.676e-08 dbeta=1.007e-07 dg=9.716e-08 dh0=6.176e-08`，`adjoint=6.222e-10`），`NUM-delta_bwd PASS rel_max=1.01e-07` |
| 卡上 gradcheck（host cpu 面） | `.venv/bin/python ascend/gradcheck_card.py --target cpu` | `GRADCHECK-SUMMARY ops=8 pass=8 fail=0 invalid=0 kernel_used=8/8` |
| 回归 | `.venv/bin/python -m pytest -q` | **340 passed**, 6 skipped, 2 warnings |

host 反向行与 `D3_DBETA_EVIDENCE.md §1.3`、`window2 notes:63` **逐字同** ⇒ **cpu/host 轨行为逐字不变**（本件无改动，无需 before/after 变化）。

---

## 3. 前提证伪小结

| 命题 | 判定 | 依据 |
|------|------|------|
| "`run_delta_card` 设备腿实际走 pure" | **证伪** | E1 代码/E2 指纹/E3 键；且 D3 §1.2 证 `_variant()` 未被 D3 改过 |
| "设备侧换 ub 即六项全对（修法一行级）" | **不成立** | 设备腿**已**是 ub（E1–E3），而 **V3 仍 `dbeta` FAIL**（§4）⇒ "换 ub"**不能**关掉 dbeta 缺陷 |
| "D3 早前'设备=ub'的说法反了" | **反过来**：D3 说法**正确** | §2 三证据 |

---

## 4. 为什么"设备腿=ub"却**仍** `dbeta` FAIL（真机理，P1-1zg 的种子）

窗四 V4 双变体差分（`w4_d3_{ub,pure}.log` ✅）：
- `ub`：`dbeta` **随 prefill 变** ⇒ `ORDER-VERDICT 随 prefill 变 ⇒ 存在 stale(H-b 证实)`；
  `none`→FAIL 值（fp `e609a6ab`）、`dirty`→`2a507f3e`、`zero`→PASS（fp `45266aef`）。
- `pure`：三跑指纹恒 `e609a6ab`（确定性但错）。

⇒ **设备腿（fresh ub，无预清零）落在 `ub` 的 "none" 态 ⇒ `dbeta`=8.002e-01（错）**；
V4-ub 报 `dbeta PASS=1.018e-07` 是因为 D3 探针的 `ITEM` 取自 **`runs["zero"]`**（`D3_probe_dbeta_order.py:160`，即**预清零后**那一跑）。

**结论**：`dbeta` 缺陷的本体是 **`ub` 档的 stale 敏感**（fresh ⇒ 错、预清零 ⇒ 对），**不是**变体选错。
修法则应落在"**发射前对相关 UB 区预清零**"（= D3 已提的 P1-1zg 方向），**不在变体选择**。

---

## 5. 本件改动清单 + 为什么**无产品码改动**

- **产品码改动：无。** 白名单三件（`gdn_asc.py`/`run_delta_card.py`/`autograd_asc.py`）**一字未动**。
  理由：要求的目标态（"设备腿统一选 ub"）**代码里已存在**（§1），改动只能是**空改动**或**引入未请求的行为变更**（如删掉 `SYS1_GDN_VARIANT` 逃生口）——二者都违背"改动最小"与诚实纪律。
  经 `md5` 复核：`gdn_asc.py=676c07936547e9d897d3b2366a0317e0`、`run_delta_card.py=2fb3e73efcb1793b73fa9d93af840428`、`autograd_asc.py=92e2fd0e6f7c251bf23b81b4afee64f7`（会话前后一致）。
- **新增**：`ascend/kernels/reconcile/K4_probe_variant_leg.py`（不占卡、机械复算"设备腿=ub"；RC=0）。
  复跑：`.venv/bin/python ascend/kernels/reconcile/K4_probe_variant_leg.py`

---

## 6. 新隐患任务登记：**P1-1zg**（ub 变体 stale 敏感）

> **P1-1zg**：`ub` 变体的输出指纹**随前一趟发射/prefill 内容变化**（`w4_d3_ub.log` 已证 `dbeta` fresh⇒错、zero⇒对）。
> **要求**：下窗用 **D3 顺序探针**（`D3_probe_dbeta_order.py`）在**真机**复测，判定是否需**发射前预清零**（及清零哪块 UB）；
> 并**先真机确认生产训练步（`train_step*` 路径）是否数值复现** `dbeta` 缺陷（本件据同档给出"受影响"的**结构判定**，数值确认留卡）。
> 关联证据：`w4_d3_ub.log`（三跑指纹 `e609a6ab/2a507f3e/45266aef`）；`D3_probe_dbeta_order.py:160`（ITEM 取自 `runs["zero"]`）。

---

## 7. 未决 / 交接（需上级裁决）

1. **本件为"前提证伪"件，非"修复落地"件**：请求的修复（设备腿→ub）**已是现状**，**真正**要关的缺陷（ub stale）落在**预清零**，**超出本件写白名单**（白名单只许动"变体选择的设备分支"）。⇒ **请裁决**：是否新开一件做"发射前预清零（P1-1zg）"。
2. **生产训练步受影响的结构判定**（§0/§1）需**下窗真机**以 `train_step*` 路径或 D3 探针复测确认数值；本件不占卡，未做。
3. 本件**禁改**面（`train_step*`/`gradcheck_card.py`/`gdn_conv_asc.py`/`window*_*.sh`/trunk/`runs/**`/任何 `tasks.md`）**未触碰**；**未用 git**。
