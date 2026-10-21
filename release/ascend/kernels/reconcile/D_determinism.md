# D_determinism — `test_sliding_routes_agree_with_mask_reference` 红→绿无归因翻转调查

调查人：agent D（determinism）｜日期：2026-10-11｜输入凭据：ubuntu runs
`38052186621` `38052628797` `38061623756` `38065583566` `38068744512` `38070929040`（全部只读取自
`gh run view --log --job <id>`，未重跑、未 push）＋ 宿主 arm64 实测（`.venv`，torch 2.14.1，与 CI 同版本）。

三态标注：**[实测]** = 本档附可复跑命令与真实输出；**[文档]** = 引自仓内文件原文；**[推断]** = 机制解释，
未被直接观测，需在结论中打折使用。

---

## 0. 结论速览

| 问题 | 结论 | 三态 |
|---|---|---|
| 那条红变绿了吗 | **变了**，但**与 V 的 skip 门无关、与 S/T/U 的 gemm_asc 改动无关** | [实测] |
| 翻转是否需要归因于某次代码改动 | **不需要**：同一 sha 集合内已出现两次「代码零差异下的红↔绿翻转」 | [实测] |
| 该断言在 ubuntu 上是否可复现 | **否**。6 个 ubuntu 数据点里 2 红 4 绿（33% 红率），红绿与 commit 内容无线性对应 | [实测] |
| 该断言在宿主 arm64 上是否可复现 | **是**。28 次 pytest + 线程 1–16 + 堆扰动 10 次 + 20 组形状，全部位等成立，0 红 | [实测] |
| 是不是产品数值缺陷 | **不是**：ubuntu 红值 `1.7881393432617188e-07` = fp32 **3 ULP**（2⁻²²），且 `kernel_allclose=True` | [实测] |
| 是不是判据缺陷 | **是**：`== 0.0` 位等既无 spec 支撑，又与同用例对姊妹路的 `< 1e-5` 容差自相矛盾，且在构造上不可保证 | [实测]+[文档] |
| 附带发现（真缺陷） | `attention_masked` 出口形状违背自身契约：3 维入→4 维出；4 维入→**当场 RuntimeError** | [实测] |
| 裁决建议 | **(b) 先修非确定（改判据 + 修出口形状缺陷）再翻门禁**；现在删 `continue-on-error` 会引入 ~1/3 概率的伪红 | [实测] |

**不得采用的三条路**（纪律：非确定按缺陷处置，不许写成"已知限制"）：① 单纯把 `== 0.0` 换成大容差当止血（无替代判据）；
② `xfail(strict)` 挂账（= 把非确定登记为已知限制）；③ 加 skip 门（= 掩盖）。
§3.3-A 里出现的容差不是止血：它只在**结构位等判据（非零位集合逐格）+ argmax 判据 + 同进程自比判据**
三条同时立起来之后才附列，且量级取在仓内已承诺的口径上。

---

## 1. 机制读码：这条断言到底在比什么 [文档]+[实测]

被测：`release/tests/test_longctx.py:640-644`

```python
def test_sliding_routes_agree_with_mask_reference():
    rep = LA.compare_routes(200, 32, chunk=48)
    assert rep["kernel_allclose"] and rep["local_allclose"]
    assert rep["kernel_vs_masked_max_abs"] == 0.0, "内核带状路与掩码路必须逐位相同"
    assert rep["local_vs_masked_max_abs"] < 1e-5
```

被测体：`release/sys1/layers/attention.py:394-432`（`compare_routes`）→ 三方对齐：

| 路名 | 实现体 | 是否经 tilelang 内核 | 归约形状 |
|---|---|---|---|
| 掩码路（ref） | `attention_masked` (`attention.py:303-326`)，纯 torch eager | 否 | matmul `q@kᵀ` = (H,200,200)，K=16；softmax over 200；`P@V` K=200，**batch 维带广播**（见 §4 缺陷） |
| 「带状 kernel 路」 | `attention_band(route="kernel")` (`:359-374`) → `sys1.kernels.attn_sw_kernel.forward` → **实际落到 `_eager()`** (`attn_sw_kernel.py:53,96-100`) | **否（关键）** | 每块 matmul (H,79,79) K=16；softmax over 79；`P@V` K=79 |
| 带状 local 路 | `attention_band(route="local")` (`:375-381`)，就地 torch eager | 否 | matmul (H,48,79)；softmax over 79；`P@V` K=79 |

### 1.1 决定性读码结果：所谓「内核带状路」在本用例里**从不调用内核** [实测]

```
$ cd release && .venv/bin/python ascend/kernels/reconcile/D_probe_routes.py route
[env] torch=2.14.1 threads=6 cpu_cap=DEFAULT os_cpu=12 device=cpu
[gate1] backends.active_backend(cpu)==TILELANG ? False (actual='torch_eager')
[gate2] plan(fp32 输入, band_len=79) -> None # plan 第一道门槛要求 q3/k3/v3 是 float16（attn_sw_mps.py:165），带状路喂的是 fp32
[gate2'] 即便换成 fp16：plan -> None # seq=79 不被 BLOCK_Q=16 整除 → 仍 None
[result] kernel_vs_masked_max_abs=0.0 kernel_allclose=True local_vs_masked_max_abs=1.7881393432617188e-07
[conclusion] 本用例的带状『kernel 路』从未触达 tilelang/MSL 内核，实际执行体是 attn_sw_kernel._eager()
```

三道闸任一道即否掉内核路：① `active_backend(cpu) != TILELANG`（ubuntu 更甚——无 MPS，`backends.py`
判据直接否）；② `attn_sw_mps.plan()` 第一门槛要 fp16，`compare_routes` 造的是 `torch.randn` fp32；
③ `band_len=79` 不被 `BLOCK_Q=16` 整除。**含义**：这条断言不是"内核 vs 掩码"的位等，
而是**两份 torch eager 实现之间、不同归约形状下的位等**——V 在 `docs/ci_baseline_triage.md §3`
里"host 走内核、ubuntu 走 OpenBLAS"的图景在**两侧都不成立**，两侧都是 CPU eager torch，
差别只在 BLAS 实现与 runner VM。

### 1.2 位等为何可能成立（机制，[推断]）

`_eager` 与掩码路对同一 (i,j) 数学同值，但：带状路 softmax 只扫 79 列、掩码路扫 200 列（窗外
`exp(-inf-max)=0`）；`P@V` 的 K 一为 79 一为 200。零加数不改变和值，**当且仅当非零项在各 lane/分块
里的相对次序不变**时两者才逐位相同——这依赖 BLAS 的 K 分块/lane 映射/打包与 ISA 分派，
与被测语义无关。host（Accelerate/vecLib）恰好保持该次序；x86 wheel（MKL/oneDNN，按核数分线程）
在部分 VM 形态上不保持。

### 1.3 该口径在 spec 层无授权 [实测]

```
$ grep -rn "逐位\|位等\|bit-exact" openspec/ docs/ --include="*.md" | grep -v "ci_baseline_triage\|D_determinism"
（命中 9 行，全部与滑窗路由无关：p2-13 的 ascend 内核对拍用 bit-exact、eval 报告的「数字链逐位吻合」）
```

- spec 对本条的唯一验收（`openspec/changes/teacher-p2-production-full/changes/p2-07-long-context/specs/long-context/spec.md:17-19`，`grep -n "滑窗正确性" -A2` 复核）：
  > Scenario: 滑窗正确性 —— THEN **选项 argmax 一致**，长序列下显存峰值低于全注意力（实测记录）
  **[文档]** 无"逐位相等"字样。
- 仓内既有数值契约（`openspec/changes/archive/.../p1-04-mps-kernels/proposal.md:40`）：
  `fp16 err≤2e-2、argmax 一致` —— 容差制，非位等制。**[文档]**
- 辨析（避免被读成"仓内不许位等"）：`bit-exact` 在本仓**是**合法口径，但只用在两侧同形状/同 dispatch 的地方——
  p2-13 `readout_prod` 0.0 bit-exact（`openspec/.../p2-13-ascend-runtime/tasks.md:19-20`）、
  以及同一测试文件里的姊妹用例 `test_sliding_short_sequence_equals_full_attention`（`test_longctx.py:653` 的 `torch.equal` 比的是同一个 `attention_masked` 的两次调用，seq=40、window=128≥40 时两张掩码逐格同值 → 形状与分派完全一致，位等有构造保证）。
  `attention_masked` 的两次调用，形状与分派完全一致，位等有构造保证）。**而本断言比的是 T² 物化参考 vs
  `chunk+window` 带状参考，两侧归约长度天然不同 → 位等在此无构造保证，只有偶然。**[实测]
- "必须逐位相等"的出处只有两处：测试自身（`test_longctx.py:643`）与 `sys1/layers/attention.py`
  模块 docstring 的自我陈述（`sys1/layers/attention.py:17`，"换来的是『内核口径』与『掩码口径』必须逐位相等"）。**[文档]**
  → 属**实现自设口径**，不是需求；且同一用例对仅差一个 M 形状（79 行 vs 48 行）的姊妹路
  就改用 `< 1e-5` 容差——**用例内部自相矛盾**。

---

## 2. 翻绿归因：三个候选逐个证/否 [实测]

### 2.1 六个 ubuntu 数据点（release-gates 同 job、同 workflow）

| # | ubuntu run | commit | 时间(UTC) | 汇总行（原样取自日志） | sliding 用例 | 该 commit 在 `release/sys1`+`release/tests`+`ci.yml` 的改动 |
|---|---|---|---|---|---|---|
| 1 | 38052186621 | f96f9b3 | 10-10 12:30 | `11 failed, 290 passed, 28 skipped, 3 deselected, 1 warning in 40.23s` | **RED** | ci.yml 新增 job；test_backends −7 行 |
| 2 | 38052628797 | 4b5aa70 | 10-10 12:37 | `10 failed, 291 passed, 28 skipped, 3 deselected, 1 warning in 38.87s` | **GREEN** | **无**（`git show --stat` 仅 openspec 文档 + runs 台账） |
| 3 | 38061623756 | 329710c | 10-10 14:56 | `10 failed, 291 passed, 28 skipped, 3 deselected, 1 warning in 40.58s` | GREEN | **无**（仅 runs/ 台账） |
| 4 | 38065583566 | 5128309 | 10-10 15:55 | `11 failed, 296 passed, 28 skipped, 3 deselected, 2 warnings in 40.25s` | **RED** | 新增 `tests/test_ascend_env_probe.py`（6 用例）；`ascend/kernels/gemm_asc.py` 改标量体 |
| 5 | 38068744512 | 5841c8f | 10-10 16:42 | `10 failed, 297 passed, 28 skipped, 3 deselected, 2 warnings in 33.65s` | **GREEN** | 仅 `release/ascend/{compile_all_asc.py,port910b,kernels/reconcile}`，**非 test_longctx 导入图** |
| 6 | 38070929040 | d57ef8a | 10-10 17:14 | `297 passed, 38 skipped, 3 deselected, 2 warnings in 36.89s` | GREEN（计入 passed） | V 的 10 个 skip 门 + KNOWN-ENV 注释（**未给 sliding 加任何门**） |

复跑取数命令（每条均可重放）：

```bash
gh run view <run-id> --json jobs --jq '.jobs[]|select(.name=="release-gates")|.databaseId'
gh run view --log --job <job-id> | grep -oE "[0-9]+ (passed|failed)[^Z]*in [0-9.]+s" | tail -1
gh run view --log --job <job-id> | grep -oE "FAILED tests/[a-z_]+\.py::[a-z_]+" | sort -u
# 本表数据即由这三条命令逐一产出（run1 断言原文见 §3.1）
```

两侧 torch/依赖**完全同版本**（排除"依赖漂移"这一路）：

```
$ gh run view --log --job 114213373619 | grep -oE "torch-2\.[0-9.]+\+cpu" | head -1   # run1 → torch-2.14.1+cpu
$ gh run view --log --job 114268005017 | grep -oE "torch-2\.[0-9.]+\+cpu" | head -1   # run6 → torch-2.14.1+cpu
```

### 2.2 候选①：V 的 skip 门把该用例变成 SKIP（计数误判）？—— **否**

- 门只挂在两条 needle 用例上：`git diff f96f9b3..d57ef8a -- release/tests/test_longctx.py` 显示
  sliding 用例上方**只加了 5 行注释**，无 decorator。[实测]
- 计数闭合：run5 `10 failed, 297 passed, 28 skipped`（总 335）→ run6 `0 failed, 297 passed, 38 skipped`
  （总 335）。**38 = 28（既有）+ 10（恰为 run5 那 10 条 FAILED）**，`297 passed` 一字未动。
  → 这 10 条红转 SKIP，sliding 始终在 passed 侧；若它转 SKIP，skipped 会是 39 而非 38。[实测]
- 题面给的算术链（`290 + 6 新测 + 1 = 297`）不是唯一解，也不是归因：`297 passed` 早在 run5
  （16:42，V 的门入库前）就已出现；`+6` 来自 run4 的 `test_ascend_env_probe.py`（本机
  `grep -c "^def test_" = 6`，`--collect-only` 本地 335/338 吻合）。[实测]

### 2.3 候选②：S 改 `gemm_asc.py` 标量体影响内核路位等？—— **否**

- 路不通：sliding 走 `sys1/layers/attention.py` + `sys1/kernels/attn_sw_kernel.py`；
  `grep -rn "^from ascend\|port910b\|compile_all_asc" tests/*.py` 显示 test_longctx **不导入** ascend 侧任何件，
  且 §1.1 已证该路连 tilelang 内核都没进。[实测]
- 时间线不通：`gemm_asc.py` 改于 5128309（run4，RED），到 5841c8f（run5，GREEN）**一字未动**
  （`git show --stat 5841c8f` 不含 gemm_asc.py）——同码而色变。[实测]
- 更硬的一条：`git diff --name-only f96f9b3..d57ef8a -- release/sys1` → **空**。六个 run 之间
  被测体 `sys1/` **一字节都没变**。[实测]

### 2.4 候选③：runner 环境（核数/ISA/BLAS 线程划分）—— **成立，且是唯一幸存因子**

- run1↔run2：`release/` 树**逐字节相同**、collection 集相同、pytest 顺序相同（同 CI 命令），
  结果由 RED→GREEN；run4↔run5 同理。**判别变量必然在仓库之外。**[实测]
- 具体是 VM 的哪一维（物理宿主 ISA → MKL/oneDNN 运行时分派；或 vCPU 数 → intra-op 线程划分），
  Actions 日志**不暴露** CPU 型号与核数（`gh run view --log --job 114213373619 | grep -icE "processor|nproc|cpu count|model name"` → `5`，但 5 条命中全部来自用例名 `test_real_processor_pad_expansion`，无一是 CPU 信息；run6 同查为 `0`）。**[实测]**
  故只能收敛到"runner 环境差异"这一层，**[推断]**。
- 宿主侧排除项（见 §3）：线程数 1–16 不影响位等、堆扰动不影响、20 组形状不影响 →
  在本机这条判据是稳的；不稳的是 x86 wheel 那侧的 BLAS 分派。**[实测]**（宿主）/ **[推断]**（ubuntu 侧机制）

---

## 3. 缺陷定位（非确定按缺陷处置）

### 3.1 ubuntu 红值的量级定性 [实测]

run1 断言原文（`gh run view --log --job 114213373619 | sed -n '559,570p'`）：

```
________________ test_sliding_routes_agree_with_mask_reference _________________
    assert rep["kernel_allclose"] and rep["local_allclose"]
>   assert rep["kernel_vs_masked_max_abs"] == 0.0, "内核带状路与掩码路必须逐位相同"
E   AssertionError: 内核带状路与掩码路必须逐位相同
E   assert 1.7881393432617188e-07 == 0.0
tests/test_longctx.py:626: AssertionError
```

- `1.7881393432617188e-07` 精确等于 `3 × 2⁻²⁴`（`2**-22`），即量级 <1 处的 **3 个 fp32 ULP**。[实测]
- **同一个数**在宿主上出现在姊妹断言里：`local_vs_masked_max_abs = 1.7881393432617188e-07`
  （`D_probe_routes.py route` 输出，§1.1）。宿主该差值来源元素的实测两值：
  `-0.8762607574462891` vs `-0.8762609362602234`，间距 `2⁻²⁴=5.960464477539063e-08` → 恰 3 ULP。[实测]
- 结论：ubuntu 的红与宿主 local 路的差**是同一类末位归约差**，而用例对后者给 `< 1e-5` 容差、
  对前者要求 `== 0.0`。**判据自相矛盾，且位等口径在构造上不可保证**——带状路的全部意义就是
  把归约长度从 T 变成 `chunk+window`；要在位等上"保证"，只能让带状路也按 T² 物化，
  等于废掉该路（与 `attention.py` 模块 docstring「内存与总长解耦」的立项目的冲突）。[实测]+[文档]

### 3.2 附带发现：`attention_masked` 出口形状违背自身契约（真缺陷，与抖动同源）[实测]

```
$ cd release && .venv/bin/python ascend/kernels/reconcile/D_probe_routes.py shapebug
  attention_masked.out   = (1, 2, 200, 16)   # AttnOut 声明 (heads, seq, dim)；lead_shape 注释："出口形状与入口严格一致"
  attention_band.out     = (2, 200, 16)
  相减广播后             = (1, 2, 200, 16)             # 用例实际在拿 4 维 vs 3 维做广播相减
  4 维入参               = RuntimeError: shape '[1, 2, 16]' is invalid for input of size 6400  # _flatten_heads 明写支持 (B,heads,seq,dim)
```

根因：`attention.py:320` `score.masked_fill(~window_band(seq, window, ...), NEG_INF)`，
`window_band` 返回 **(1,1,T,T)**（`attention.py:195`），而 `score` 是 **(H,T,T)**；
`masked_fill` 按广播出形，结果升成 `(1,H,T,T)`。后果三条：

1. `attention_masked` 对 `(heads,seq,dim)` 入参返回 4 维，违反 `AttnOut` 声明与 `lead_shape` 契约；
2. 对 `(B,heads,seq,dim)`（`_flatten_heads` 明写支持的口径）**直接崩溃** → 长文批推理路径若有人调它即炸；
3. 掩码路后半程 `P@V` 走的是**带 batch 广播的 matmul**（`(1,2,200,200) @ (2,200,16)`），
   带状路走的是普通 3 维 bmm——两条路的 dispatch 形态本就不同，位等更是偶然。

这三条里 (3) 与 §2.4 的抖动直接同源（偶然性被放大），(1)(2) 是独立的接口缺陷。

### 3.3 修法（给编排者落任务用；本 agent 未动任何产品件/测试件）

**A. 判据（缺陷本体，tests 侧 + 口径入 spec）**——用"平台无关且更强"的结构判据替换位等：

1. **可见性集合逐格位等**（真不变量）：`forward_weights`/带状路的非零掩码与 `window_band(seq,window)`
   逐格 `torch.equal`，并断言未来位严格 0 —— 这恰好是 `attention.py:406-407` docstring 自称要守护的
   "滑窗方向被悄悄写反"的错误类，**位等做不到它做得到**（写反方向会改变非零位集合，位等只比较数值）。
2. **argmax 一致**：spec 原生验收（§1.3），平台无关。
3. **数值容差**：`kernel_vs_masked_max_abs <= 4 * torch.finfo(torch.float32).eps * 量级因子`
   （或复用姊妹断言的 `< 1e-5`），并在断言旁注明"末位归约差随 BLAS/ISA 而变，位等非产品不变量"。
   容差不是放宽验收，而是把验收放在**产品真正承诺的口径**上；仓内先例 `fp16 err≤2e-2`。
4. **真确定性守卫**（新增，取代位等的存在感）：同进程内 `compare_routes` 连跑两次必须逐位相同
   （`torch.equal` 自比）——这才是可平台无关地钉住的"确定性"契约。
   本机已实测成立（§4 `reps` 10 次、`threads` 8 档、全绿）。
5. 同时把 `docs/ci_baseline_triage.md §3/§5` 的 KNOWN-ENV 叙述改写为**缺陷叙述**：
   "该断言在 ubuntu 不可复现（6 run 2 红 4 绿），按缺陷处置，修复=改判据"；
   §5 line 115 的"预期稳态 `1 failed …`"已被 run 38052628797 当场证伪，须订正。

**B. `attention_masked` 出口形状（sys1 侧缺陷）**：`masked_fill` 的掩码降到与 `score` 同秩
（`window_band(...).squeeze(0)` 或返回 (1,T,T) 的构造函数），并对 3 维/4 维入参各加一条出口形状断言。
修复前，本仓 `test_longctx.py` 的相减实际依赖广播，属未声明行为。

---

## 4. 宿主确定性实证（≥10 次，逐条原始输出）

环境：macOS arm64，`release/.venv`，Python 3.12.8，**torch 2.14.1（与 ubuntu 两 run 同版本号）**，
`threads=6`、`cpu_cap=DEFAULT`、`os_cpu=12`。全部命令在 `release/` 目录执行。

### 4.1 pytest 单用例连跑 10 次（默认线程）

```
for i in $(seq 1 10); do .venv/bin/python -m pytest \
  "tests/test_longctx.py::test_sliding_routes_agree_with_mask_reference" -q -p no:cacheprovider; done
```

| # | 结果（真实汇总行） |
|---|---|
| A1 | `1 passed in 0.99s` |
| A2 | `1 passed in 0.83s` |
| A3 | `1 passed in 0.83s` |
| A4 | `1 passed in 0.83s` |
| A5 | `1 passed in 0.83s` |
| A6 | `1 passed in 0.84s` |
| A7 | `1 passed in 0.84s` |
| A8 | `1 passed in 0.84s` |
| A9 | `1 passed in 0.84s` |
| A10 | `1 passed in 0.85s` |

### 4.2 `OMP_NUM_THREADS=1` 档 5 次

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest "tests/test_longctx.py::test_sliding..." -q -p no:cacheprovider
```

| # | 结果 |
|---|---|
| B1 | `1 passed in 0.84s` |
| B2 | `1 passed in 0.85s` |
| B3 | `1 passed in 0.87s` |
| B4 | `1 passed in 0.85s` |
| B5 | `1 passed in 0.85s` |

### 4.3 `OMP_NUM_THREADS=4` 档 5 次

| # | 结果 |
|---|---|
| C1 | `1 passed in 0.87s` |
| C2 | `1 passed in 0.86s` |
| C3 | `1 passed in 0.87s` |
| C4 | `1 passed in 0.92s` |
| C5 | `1 passed in 0.91s` |

### 4.4 同文件全跑（执行顺序/堆历史扰动）5 次

```
.venv/bin/python -m pytest tests/test_longctx.py -q -p no:cacheprovider
```

| # | 结果 |
|---|---|
| D1 | `45 passed, 1 skipped in 0.93s` |
| D2 | `45 passed, 1 skipped in 0.89s` |
| D3 | `45 passed, 1 skipped in 0.93s` |
| D4 | `45 passed, 1 skipped in 0.90s` |
| D5 | `45 passed, 1 skipped in 0.89s` |

### 4.5 前置模块真跑过再执行该用例（分配历史扰动，3 次）

```
.venv/bin/python -m pytest tests/test_prod_sft.py tests/test_longctx.py::test_sliding_routes_agree_with_mask_reference -q -p no:cacheprovider
```

| # | 结果 |
|---|---|
| F1 | `36 passed in 3.85s` |
| F2 | `36 passed in 2.93s` |
| F3 | `36 passed in 2.86s` |

### 4.6 进程内重复 10 次 + 无关大 matmul 扰动（`D_probe_routes.py reps 10`）

```
  baseline max_abs=0.0
  rep0  perturbed max_abs=0.0 eq=True
  rep1  perturbed max_abs=0.0 eq=True
  rep2  perturbed max_abs=0.0 eq=True
  rep3  perturbed max_abs=0.0 eq=True
  rep4  perturbed max_abs=0.0 eq=True
  rep5  perturbed max_abs=0.0 eq=True
  rep6  perturbed max_abs=0.0 eq=True
  rep7  perturbed max_abs=0.0 eq=True
  rep8  perturbed max_abs=0.0 eq=True
  rep9  perturbed max_abs=0.0 eq=True
```

### 4.7 intra-op 线程数扫描 1–16（`D_probe_routes.py threads`）

```
  [env] torch=2.14.1 cpu_cap=DEFAULT os_cpu=12
  threads= 1 kernel_vs_masked_max_abs=         0.0 bitwise_eq=True allclose=True
  threads= 2 kernel_vs_masked_max_abs=         0.0 bitwise_eq=True allclose=True
  threads= 3 kernel_vs_masked_max_abs=         0.0 bitwise_eq=True allclose=True
  threads= 4 kernel_vs_masked_max_abs=         0.0 bitwise_eq=True allclose=True
  threads= 6 kernel_vs_masked_max_abs=         0.0 bitwise_eq=True allclose=True
  threads= 8 kernel_vs_masked_max_abs=         0.0 bitwise_eq=True allclose=True
  threads=12 kernel_vs_masked_max_abs=         0.0 bitwise_eq=True allclose=True
  threads=16 kernel_vs_masked_max_abs=         0.0 bitwise_eq=True allclose=True
```

### 4.8 形状扫描 20 组（同机同 seed，只改 seq/window/chunk；`D_probe_routes.py shapes`）

```
  [env] torch=2.14.1 threads=6 cpu_cap=DEFAULT  # 同机、同 seed、只换形状
  seq=200 win= 32 chunk= 48 | kernel_vs_masked=  0.0 eq=True  | local_vs_masked=1.7881393432617188e-07
  seq=200 win= 32 chunk= 16 | kernel_vs_masked=  0.0 eq=True  | local_vs_masked=1.7881393432617188e-07
  seq=200 win= 32 chunk= 64 | kernel_vs_masked=  0.0 eq=True  | local_vs_masked=1.7881393432617188e-07
  seq=200 win= 16 chunk= 48 | kernel_vs_masked=  0.0 eq=True  | local_vs_masked=2.384185791015625e-07
  seq=200 win= 64 chunk= 48 | kernel_vs_masked=  0.0 eq=True  | local_vs_masked=5.960464477539062e-07
  seq=200 win=128 chunk= 48 | kernel_vs_masked=  0.0 eq=True  | local_vs_masked=1.7881393432617188e-07
  seq= 96 win= 32 chunk= 48 | kernel_vs_masked=  0.0 eq=True  | local_vs_masked=  0.0
  seq=128 win= 32 chunk= 48 | kernel_vs_masked=  0.0 eq=True  | local_vs_masked=  0.0
  seq=160 win= 32 chunk= 48 | kernel_vs_masked=  0.0 eq=True  | local_vs_masked=  0.0
  seq=240 win= 32 chunk= 48 | kernel_vs_masked=  0.0 eq=True  | local_vs_masked=  0.0
  seq=256 win= 32 chunk= 48 | kernel_vs_masked=  0.0 eq=True  | local_vs_masked=  0.0
  seq=300 win= 32 chunk= 48 | kernel_vs_masked=  0.0 eq=True  | local_vs_masked=2.384185791015625e-07
  seq=200 win= 32 chunk= 24 | kernel_vs_masked=  0.0 eq=True  | local_vs_masked=1.7881393432617188e-07
  seq=200 win= 32 chunk= 32 | kernel_vs_masked=  0.0 eq=True  | local_vs_masked=1.7881393432617188e-07
  seq=200 win= 32 chunk= 50 | kernel_vs_masked=  0.0 eq=True  | local_vs_masked=  0.0
  seq=200 win= 33 chunk= 48 | kernel_vs_masked=  0.0 eq=True  | local_vs_masked=1.4901161193847656e-07
  seq=128 win= 16 chunk= 32 | kernel_vs_masked=  0.0 eq=True  | local_vs_masked=  0.0
  seq=144 win= 24 chunk= 40 | kernel_vs_masked=  0.0 eq=True  | local_vs_masked=  0.0
  seq=210 win= 40 chunk= 56 | kernel_vs_masked=  0.0 eq=True  | local_vs_masked=  0.0
  seq=256 win= 16 chunk= 64 | kernel_vs_masked=  0.0 eq=True  | local_vs_masked=  0.0
  [summary] 0/20 组形状上『带状 kernel 路 vs 掩码路』位等不成立（本机）；local 路 10/20 组出现末位差
```

读法：位等在宿主侧对**带状路**稳、对**local 路**已经不稳（10/20 组出末位差，用例仅 `<1e-5` 兜住）。
同一条数学、只差一个 M 形状，一路判"必须位等"、一路判"容差"——判据的偶然性在本机就能看见，
不必等到 ubuntu。**[实测]**

### 4.9 逐算子拆解（`D_probe_routes.py split`，宿主全绿时的位等来源）

```
[out] max_abs=0.0 ndiff=0/6400 ulp=0
  [chunk start=0 band_len=48] 同一批(i,j)分数：位不等=0/4608 max_abs=0.0 ulp=0
    概率 P：位不等=0/4608 max_abs=0.0
    同一概率 @V：位不等=0/1536 max_abs=0.0
  [chunk start=48 band_len=79] 同一批(i,j)分数：位不等=0/7584 max_abs=0.0 ulp=0
    概率 P：位不等=0/7584 max_abs=0.0
    同一概率 @V：位不等=0/1536 max_abs=0.0
  [chunk start=96 band_len=79] 同一批(i,j)分数：位不等=0/7584 max_abs=0.0 ulp=0
    概率 P：位不等=0/7584 max_abs=0.0
    同一概率 @V：位不等=0/1536 max_abs=0.0
```

**宿主合计**：pytest 调用 **28 次**（A10+B5+C5+D5+F3）+ 全量 1 次（`332 passed, 3 skipped, 3 deselected`）
+ 进程内位等判定 **38 次**（reps 10 + threads 8 + shapes 20）→ **宿主 0 红**。
（对照：ubuntu 6 次 **2 红**。）

---

## 5. 裁决建议

**(b) 先修非确定再翻门禁。** 依据与边界如下：

1. **(a) 立即可翻 = 不可接受**。6 个 ubuntu 数据点里 2 红（33%），红绿与 commit 内容无对应
   （§2.1 表）；删 `continue-on-error` 等于把 1/3 概率的伪红设成合入阻断，且真回归信号会被伪红淹没。[实测]
2. **(c) 再补数据点 = 不必要**。补齐 6 个点已经够了（含两次"代码零差异下的翻转"，
   这是决定性证据，比再多几个点强）。CI 侧无需新 run；**要补的是 §3.3 的判据修复**。[实测]
3. **(b) 的具体顺序建议**：
   ① 先按 §3.3-A 改判据（结构位等 + argmax + 容差 + 同进程重复自比），并把 §3.3-B 的
   `attention_masked` 出口形状缺陷登记为独立缺陷任务（4 维入参直接崩，属产品件真缺陷）；
   ② 修后在 ubuntu 上观察 **2–3 个自然 push**（不专门重跑）确认该用例恒绿且不再受 runner 形态影响；
   ③ 再删 `continue-on-error` 那一行（其余一字不动）；
   ④ 同步订正 `docs/ci_baseline_triage.md §0/§3/§5`：把"KNOWN-ENV 环境伪影、预期保留红"
   改写为"判据缺陷，已按 §3.3 修复"，并撤掉 line 115 那条已被证伪的预期稳态。
4. **纪律声明**：本调查**未**放宽任何断言、**未**加 skip、**未**写 xfail、**未**改任何产品件/测试件/
   ci.yml；`continue-on-error` 仍在原位，翻转由编排者执行。修复的正当性不来自"想变绿"，
   而来自 §1.3（位等无 spec 授权、与姊妹断言自相矛盾）+ §3.1（构造上不可保证）+ §3.3-A1/A4
   （替换判据对目标错误类的守护力严格更强）。

---

## 6. 产物与复跑索引

- 新增（本轮唯一写入面）：`release/ascend/kernels/reconcile/D_determinism.md`（本档）、
  `release/ascend/kernels/reconcile/D_probe_routes.py`（只读探针，`ruff check` 通过：
  `scratch/.venv/bin/ruff check release/ascend/kernels/reconcile/D_probe_routes.py → All checks passed!`）
- 未触碰：`release/tests/**`、`release/sys1/**`、`release/ascend/**` 产品件、`.github/workflows/ci.yml`、
  `docs/**`、openspec tasks.md、任何 git 操作。
- 宿主全量自证（确认本轮无引入）：
  `cd release && .venv/bin/python -m pytest tests -q -m "not integration" -p no:cacheprovider`
  → `332 passed, 3 skipped, 3 deselected, 2 warnings in 35.97s`（与 audit 记录的 host 基线一致）
