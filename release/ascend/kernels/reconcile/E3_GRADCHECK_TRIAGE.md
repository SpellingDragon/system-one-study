# E3_GRADCHECK_TRIAGE — 窗三 B 段两件 FAIL 的「探针 vs 产品」分诊（不占卡）

> 执行代理 **E3**（p2-13 执行波，**不占卡**）。写面：新建 `ascend/kernels/reconcile/E3_GRADCHECK_TRIAGE.md`
> + `reconcile/E3_probe_{nanpath,gate,cardtriage}.py`。**未改**任何产品件/判据/提交面（见 §7 指纹）。
> 一手事实：`runs/1012-p2-13-oncard-window3-pin/artifacts/w3_gradcheck.log`（卡上原文）。
> 纪律：**先证量具可信，才可指控产品**——本件先做输入分布良态核查（§2）+ 代码路径反推（§3），再下结论（§5）。

---

## ① 结论速览（两条，均非探针缺陷）

| 件 | 判词（原文行） | 分诊结论 | 一句话依据 |
|---|---|---|---|
| `gdn_delta` | `FAIL max_rel=nan eps=0.01`（`w3_gradcheck.log:42`） | **真缺陷（须产品面立项）** | 判词**无异常**⇒算完了但梯度含 nan（§3-P-nan 复刻）；门分布**已证良态**（§2：`g` 全非正、`seq=4`、`∏exp(g)≤1`、参考有限）⇒ V2「randn 门爆 nan」先例**在此不成立**（§2-D 只在 randn 门 ∩ seq=512 时复现）。真机坏点 ⇒ device 侧内核；旁证：同窗 C 段 `gdn_bwd_impl` 的 `dbeta` device-only 缺陷。 |
| `gdn_conv` | `FAIL max_rel=n/a eps=0.01` + `err=RuntimeError: SetPrecisionMode…500001`（`w3_gradcheck.log:47-48`） | **真缺陷（须产品面立项）** | `n/a` = **异常被吞**（`run_one` 走 `exc` 分支，`rel` 恒 `None`）——§3-P-na_ok 在 host **逐字复刻**该判决行；异常是 **torch_npu/ACL 层编译选项**错误，**与输入数据无关**⇒非输入分布类探针缺陷；抛点＝产品反向闭式 `autograd_asc._GdnConvFn.backward`（`autograd_asc.py:434` → `_conv_silu_torch` → `F.conv1d`）。 |

**⇒ 窗三 B 段的 `fail=2` 保持指控力**：两件都指向**产品/device 侧**，**不应**（也无需）改 `gradcheck_card.py` 的输入分布去"洗白"它。
两条结论各有一处**标注为假设**的残余未知（§5），由 §6「下窗一枪」钉死。

---

## ② 量具良态核查：三处输入口径对照（task (2)）

复跑：`cd release && .venv/bin/python ascend/kernels/reconcile/E3_probe_gate.py` → `E3-GATE-DONE pass=7 fail=0`

| 口径 | 递推门 `g` 分布 | `β` 分布 | `seq` | 实测 `g.max` | `∏_t exp(g_t)` max |
|---|---|---|---|---|---|
| **A. `gradcheck_card::_build("gdn_delta")`**（本件） | `-rand*0.2 ∈ (-0.2,0]` | `rand*0.9+0.05 ∈ [0.05,0.95]` | **4** | **-0.002169 ≤ 0** | **0.997834 ≤ 1** |
| B. `bench_card` V2 `_gate3`（`bench_card.py:296-304`） | `-rand ∈ (-1,0]` | `rand/dk`（`bench_card.py:248`） | 512 | -0.000122 ≤ 0 | ≤1 |
| C. `run_delta_card` C4 seed=11（`run_delta_card.py:68-77`） | `G=-rng.random ∈ (-1,0]` | `rand*0.5+0.25 ∈ [0.25,0.75]` | 8 | -0.09274 ≤ 0 | ≤1 |

**读数**：三处口径**同族**（`g` 全非正 ⇒ `exp(g)≤1` ⇒ 递推**不自增长**）。**A 并未用 `randn` 门**——
即 V2 那个「`randn` 当递推门 ⇒ 参考自身爆 `inf/nan`」的坏口径，**在 gradcheck 本件里不存在**。

**H-V2 对照（坏口径长什么样）**：`g=randn` ∩ `seq=512` ⇒ `∏exp(g)≈2.85e20`，fp32 递推输出
`finite=False inf=5 nan=3266`（探针实际位宽 fp32；fp64 尚有限，但 fp32 上限 3.4e38 ⇒ 溢出）。
**这正是会让 `rel=nan` 的探针缺陷形态——但它需要「randn 门」∧「长序列」两件同时成立，A 口径两件都不满足。**

**参考有限性实测（A 口径输入）**：fp64 精确递推 `autograd_asc._gdn_torch`（`autograd_asc.py:384-401`）
输出 `shape=(2,4,8) max_abs=8.22721 finite=True`；host fp32 前向 vs fp64 参考 `rel=1.677e-07`。
⇒ **A 的参考/数值自身是良态的**，不可能因输入分布产生 `nan`。

---

## ③ 代码路径反推：`nan` 与 `n/a` 各归哪条（task (1)）

复跑：`cd release && .venv/bin/python ascend/kernels/reconcile/E3_probe_nanpath.py` → `E3-NANPATH-DONE pass=3 fail=0`

| 结论 | 证据（host 人工制造同形态） |
|---|---|
| **`max_rel=n/a` = 「异常被吞」**（不是 FD 不可得、不是解析梯度为 None） | **P-na_ok**：前向用内核成功、反向闭式抛 `RuntimeError` ⇒ 逐字产出 `GRADCHECK-gdn_conv FAIL max_rel=n/a eps=0.01` + `err=RuntimeError: SetPrecisionMode…500001`，`keys=['cpu\|gdnconv[cpu]\|c8k4b2']`（前向键仍在） |
| **`max_rel=nan` = 「算完了，但梯度含 nan/inf」**（不是异常分支） | **P-nan**：解析侧注入 `nan` ⇒ `GRADCHECK-linear FAIL max_rel=nan eps=0.05`，`err=None` |
| `n/a` 的边界：**只有「前向键已到手之后才抛」才出 `n/a`** | **P-na_early**：前向入口就抛（没吃到内核）⇒ 判 **`INVALID reason=no-kernel`**，**不是** `n/a` |

**归行**：
- `gdn_conv` 的 `n/a`：`run_one`（`gradcheck_card.py:271-280`）——`_gradcheck_op` 的宽 `except`
  （`gradcheck_card.py:198-200`）把异常收进 `result["exc"]`，`run_one` 见 `exc is not None` ⇒ 回
  `FAIL` + `rel=None`（`:278-280`），`_line` 把 `rel is None` 打成 `n/a`（`gradcheck_card.py:330`）。
- `gdn_delta` 的 `nan`：`_gradcheck_op` 计算公式 `rel = abs_err / max(ana_norm, 1e-30)`
  （`gradcheck_card.py:194`）——`num` 或 `ana` 任一含 nan/inf ⇒ `rel=nan`、`ok=False`（`:195-197`），
  全程**无异常**（否则会走 `:198` 变 `n/a`）。

---

## ④ 容器复现（不占卡；`cann910b-h`，compat `ca023c2e`，`/dev/shm` 缓存）

| 命令 | 结果 | 读法 |
|---|---|---|
| `python3 ascend/gradcheck_card.py --target ascend --no-run`（编译面） | `SUMMARY ops=8 pass=8 fail=0 invalid=0 kernel_used=8/8` | **八件在昇腾后端都能真编出前向内核**；gdn 两件的编译面**不受** FAIL 影响 |
| `python3 ascend/gradcheck_card.py --target cpu`（容器内） | `SUMMARY ops=8 pass=0 invalid=8 kernel_used=0/8`（全 `reason=no-kernel`） | **环境差异，非缺陷**：容器是 **py3.10**，`probe_cpu` 编译触发 `TypeError: Forward references must evaluate to types. Got buffer.`（`ascend_env.py:142` blocker）⇒ 容器内 **cpu 后端不可用**。design §5.1 的 host 8/8 基线在 **mac venv（py3.12）**，不在容器。 |
| **host（mac venv，task 指定）** `.venv/bin/python ascend/gradcheck_card.py --target cpu` | `SUMMARY ops=8 pass=8 fail=0 invalid=0`（exit 0）；`gdn_delta PASS 8.569e-06`、`gdn_conv PASS 2.653e-05` | **gdn 两件在当前探针实现下 host 仍 PASS** ⇒ 卡上挂是 device 侧 |
| `.venv/bin/python ascend/kernels/reconcile/B3_probe_negatives.py` | `B3-NEG-DONE pass=3 fail=0` | 负例（防假绿三证 / plan 拒形状 / ×1.001）**不动判据仍红/仍 INVALID** |
| `pytest tests -q -m "not integration"` | **`340 passed, 3 skipped, 3 deselected`** | 回归红线 |

> **未做输入分布改动**：既已判两件**非探针缺陷**，按 task (2)/(3) 的**条件**（"若判定探针缺陷"）**不改**
> `gradcheck_card.py`——改了反而会把 product-side 的 `fail=2` 洗成探针噪声。故本件只**增可观测性**
> （落在**新文件** `E3_probe_cardtriage.py`，不碰 B3 交付件）。

---

## ⑤ 分诊结论（逐件，二选一）

### 5.1 `gdn_delta` → **真缺陷（须产品面立项）**

- **探针已被排除**（三证）：① 代码路径 = 「算完但含 nan」（§3-P-nan），非异常；② 门分布**良态**
  （§2：`g∈[-0.1921,-0.0022]` 全非正、`seq=4`、`∏exp(g)≤1`、fp64 参考 `finite`）；③ V2 坏口径需要在
  **randn 门 ∩ 长序列**下才复现（§2-D），本件两件都不满足 ⇒ **`nan` 不可能来自探针输入分布**。
- **⇒ 坏点在 device 侧内核**：解析路走 `gdn_bwd_impl`（`gdn_asc.py:583` 键 `gdnb[ascend]|…`），数值路只读
  前向 `gdn_impl`（`gdn_asc.py:552/560`，键 `gdn[ascend]|h2k8v8|c8ubh` = **cores 8 ⇒ blockCount>1**）。
- **旁证（标注为假设，非事实）**：
  - 同窗 C 段已实测 `gdn_bwd_impl` 的 **`dbeta` 是 device-only 缺陷**（`w3_deltabwd.log:20`：
    `NUM-delta_bwd FAIL first=dbeta rel=5.997e-01 absdiff=4.435e+00`；host 同腿六项 `1e-7` 量级）——
    **同一内核、同窗、同 device-only 特征**。
  - E 段实测「UB 残留跨发射污染仅在 `blockCount>1` 出现」，而 gradcheck 的 fwd 键正是 `c8`（8 核）。
    ⇒ 两条独立线索都指向「**多块 UB 内核在 device 上数值不可靠**」这一族；`nan` 是其更剧烈形态。
- **残余未知（待钉）**：具体是**解析路**（反向内核）还是**数值路**（前向内核）先含 nan、坏在哪个输入——
  未在卡上观测过（§6 一枪判）。

### 5.2 `gdn_conv` → **真缺陷（须产品面立项）**

- **探针已被排除**：`n/a` = 异常分支（§3-P-na_ok 逐字复刻）；而异常是 **`SetPrecisionMode` /
  `ACL_PRECISION_MODE` 的编译选项错误（code 500001）**——**与输入数值无关**（改输入分布无用，也非
  FD/解析可导性所致）⇒ **不是输入分布类探针缺陷**。
- **抛点在产品反向闭式**：`_GdnConvFn.backward`（`autograd_asc.py:423-439`）用 `torch.nn.functional`
  现场重算前向再回传；`_conv_silu_torch`（`autograd_asc.py:442-456`）里的 **`F.conv1d`（:455，depthwise）**
  在 **NPU 张量**上执行 ⇒ torch_npu 在该 op 上报 `SetPrecisionMode …500001`。
  前向 tilelang 内核 `conv_impl` **编译成功**（`w3_gradcheck.log:45-46`），其余 7 件（含同样走 torch 闭式的
  `attn_sw`）在卡上**都 PASS** ⇒ 失败**特定于 conv 这条 NPU op**，不是 tilelang 发射、也不是设备整体坏。
- **结论**：产品 `gdn_conv` 的**可微路径在 NPU 上不可执行**（前向内核 OK、反向闭式落 NPU torch op 即崩）
  ⇒ 产品面须处理（要么补 kernelized conv 反向，要么避开该 NPU op）。**须产品面立项**。
- **标注为假设（非事实）**：`500001` 也可能源于 **torch_npu/CANN 环境不支持该 depthwise-conv 反传**
  （而非产品逻辑写错）。但无论根因在谁，**修复责任在产品侧**（产品必须让 `gdn_conv` 反向能在设备上跑通）
  ——故仍归"真缺陷（须产品面立项）"，唯**归因细节**待 §6 的 traceback 帧钉死。

---

## ⑥ 下窗一枪（一条命令；判两件）

```bash
source /usr/local/Ascend/cann-8.5.2/set_env.sh; export ASCEND_NPU_ARCH=dav-2201; \
export BISHENG_HOME=$(dirname $(dirname $(command -v bisheng))); export TILELANG_CACHE_DIR=$(mktemp -d); \
cd /work && timeout 900 python3 ascend/kernels/reconcile/E3_probe_cardtriage.py --target ascend
```

**它判什么**（新件 `reconcile/E3_probe_cardtriage.py`，判据阈值/rtol/atol/INVALID 语义**一字不动**，只加可观测性）：

- `gdn_delta`：逐路（**数值差分路**＝只读前向内核 vs **解析反传路**＝唯一吃反向内核）× 逐输入
  `q/k/v/g/β` 报「是否含 nan/inf + 最大绝对值」，打一行
  `E3-TRIAGE-gdn_delta VERDICT=nan-in-analytic(反向内核) | nan-in-numeric(前向内核) | clean`。
  ⇒ 把 §5.1 的"残余未知"钉成"前向还是反向、哪个输入"。
- `gdn_conv`：把异常**完整 traceback 的仓内帧 `file:line`** 打出来，打一行
  `E3-TRIAGE-gdn_conv VERDICT=exception-at-{forward,analytic,numeric}`。
  ⇒ 看 `raise-frame` 是否落在 `ascend/autograd_asc.py:434`（`_conv_silu_torch` 闭式）——是则坐实
  `F.conv1d` 抛；若落在 tilelang 发射帧则改判。

**该枪真机前的 host 自证**（`--selftest`，人造坏形态证诊断分支有效）：
`cd release && .venv/bin/python ascend/kernels/reconcile/E3_probe_cardtriage.py --selftest`
→ `VERDICT=nan-in-analytic(反向内核) rel=nan`（且逐输入报出坏在 `q`）
＋ `VERDICT=exception-at-analytic`，`raise-frame ascend/autograd_asc.py:434 in backward`（＝抛点帧已现）。

---

## ⑦ 复跑命令与未改佐证

复跑（全部 host，不占卡）：
```bash
cd release
.venv/bin/python ascend/gradcheck_card.py --target cpu                          # 8/8 PASS（gdn 两件 PASS）
.venv/bin/python ascend/kernels/reconcile/B3_probe_negatives.py                 # pass=3 fail=0
.venv/bin/python ascend/kernels/reconcile/E3_probe_nanpath.py                   # pass=3 fail=0（nan/n/a 归路）
.venv/bin/python ascend/kernels/reconcile/E3_probe_gate.py                      # pass=7 fail=0（量具良态）
.venv/bin/python ascend/kernels/reconcile/E3_probe_cardtriage.py --target cpu --dirs 2   # 两件 VERDICT=clean
.venv/bin/python ascend/kernels/reconcile/E3_probe_cardtriage.py --selftest     # 诊断分支自证
.venv/bin/python -m pytest tests -q -m "not integration"                        # 340 passed
```

**未改佐证**（md5 指纹，本件未动）：
`gradcheck_card.py=d5b8c6c9…`、`port910b/run_delta_card.py=2fb3e73e…`、`bench_card.py=adbc0b5a…`、
`autograd_asc.py=26d9fd1a…`；`runs/**`、任何 `tasks.md`、`attempts/D2_*`、`reconcile/{S3,U2,V2,B3,C4,D3}_*` 均未动。

**产物清单**：新建 `reconcile/E3_GRADCHECK_TRIAGE.md`（本件）、`reconcile/E3_probe_nanpath.py`、
`reconcile/E3_probe_gate.py`、`reconcile/E3_probe_cardtriage.py`；无修改件；未动 git。
