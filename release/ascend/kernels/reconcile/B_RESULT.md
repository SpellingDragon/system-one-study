# B_RESULT — p2-13 卡窗可信化（代理 B）

**一句话结论**：H1（torch_npu 未注册）与 P1-1l（内核启用静默假绿）两处病灶都已有**产品侧断言 + 窗侧探针 + 负例自证**；
卡窗的 `RUNNER-DONE` 已变成带计数的判决（容器干跑实证 `pass=2 fail=5 degraded=3 advisory_fail=5 exit=1`，
而旧版同景会打一行裸 `RUNNER-DONE` 被读成通过）。**卡上真值仍待下窗**（本机与容器都无 npu，本文所有
"卡上预期"都标注为待逐字对照，不含糊）。

- 改动件（白名单内）：`release/ascend/train_step.py`(+151) / `release/ascend/autograd_asc.py`(+75/-1) /
  `release/ascend/port910b/card_run.sh`(+199/-21) ；新建 3 枚探针 + 本文。
- **未动**：任何 `*_asc.py` / `*_kernel.py` / `ascend_env.py` / trunk / `compile_all_asc.py` /
  `run_numerics.py` / `bench_card.py` / tests / docs / **任何 tasks.md**；未执行 git 写操作。
- 基线守住：`bash -n card_run.sh` OK；host `pytest tests -q -m "not integration"` →
  **340 passed, 3 skipped, 3 deselected, 2 warnings in 42.89s**（0 failed）。

---

## 1 H1：`torch_npu` 设备注册（真阻塞，已实证）

### 1.1 病灶直接证据（原文，可复跑）
```
$ git show HEAD:release/ascend/train_step.py > /tmp/b_before_ts.py
$ grep -c torch_npu /tmp/b_before_ts.py ; grep -c KERNEL-USED /tmp/b_before_ts.py
0
0
```
容器里跑原始件（`cann910b-h`，origin/main 树 `/tmp/a3sys1`）复现 wave2 原文：
```
$ docker exec cann910b-h bash -lc 'cd /tmp/a3sys1/release && python3 ascend/train_step.py --target ascend'
RuntimeError: Expected one of cpu, cuda, ipu, xpu, ... device type at start of device string: npu
```

### 1.2 修法（两处）
**产品侧** `ascend/train_step.py`：
- `NPU_TARGET_ALIASES=("ascend","asc","npu")`、`npu_requested(argv)`（只看 `--target`/env `DMLAYA_ASCEND_TARGET`）。
- 模块级 `if npu_requested(): ensure_torch_npu()` —— **import 本件即完成注册**，且 **cpu 轨一次都不 import**
  （无 npu 机器上 `import torch_npu` 会 ImportError，把 cpu 轨整条砸掉）。
- `ensure_torch_npu()` 幂等、失败不抛，返回 `{status:ok|fail|skip, version, detail}`；
  `npu_reg_line()` 打成 `NPU-REG <status> <ver> target=<t> npu_avail=<..> <detail>`。
- 新增两个退码：`EXIT_DEVICE_UNAVAILABLE=4`（设备/注册不可用）、`EXIT_KERNEL_UNUSED=5`（见 §2）。

**窗侧** `card_run.sh` **[0] 段最前**（早于任何腿）的探针行：
```
NPU-REG fail torch_npu:ModuleNotFoundError:No module named 'torch_npu'
NPU-REG-CONSEQUENCE 未注册 npu 设备 ⇒ train_step 走不到 npu、产品腿只出编译判
```
（容器实测原文；卡上应出 `NPU-REG ok <ver> avail=True count=1 torch=…`。）

### 1.3 踩到并修掉的两个真陷阱（登记为判读口径）
1. **`hasattr(torch, "npu")` 不能用作探针**：它触发 torch 的**惰性后端 import**，cpu 轨也会被惊动 ⇒
   `_npu_avail_text()` 改走 `try: torch.device("npu")`，skip 态打 `npu_avail=n/a`（不把"没看"混成"看了说没有"）。
2. **`import tilelang` 本身就会触发 torch 的 npu 惰性探测**（二分定位：单独 `import ascend`→不打；
   `ascend.kernels.ascend_env`→不打；`import tilelang`→打 `An unexpected error occurred while checking
   NPU availability`）。这是上游既有行为，与本件无关，但**判读口径**要写清：卡上若装了 torch_npu，
   tilelang 的 import 可能已顺手把 npu 注册好了——**能依赖，也不能依赖**（版本/顺序一变即失效），
   所以 H1 的修法必须是**显式 import**，探针的判据也只看"本件发起的 `import torch_npu ->` 是否出现在 cpu 轨"。

### 1.4 负例（无 npu 机器必须报 fail，不得谎称卡上已验）
```
$ cd release && .venv/bin/python ascend/train_step.py --target ascend ; echo $?
NPU-REG fail none target=ascend npu_avail=False import torch_npu -> ModuleNotFoundError: No module named 'torch_npu'
P2-STEP-ascend FAIL: NPU-REG-FAIL 设备名不可用 RuntimeError: Expected one of cpu, cuda, ... npu（…）
ASC-EXIT=4          ← 旧版这里是裸 traceback；现在先自证再停，不给"假绿"留位置
```
`B_probe_h1_npu_reg.py` 另用 **PYTHONPATH 毒药假件**（`torch_npu.py` import 即抛 `POISON`）端到端验：
cpu 轨输出**不含** `import torch_npu ->`（= 没 import）且退 0；ascend 轨把 POISON 收成
`NPU-REG fail … RuntimeError: POISON` + 退 4。`ok` 分支只能在进程内桩出来（`sys.modules["torch_npu"]`
假件 + `_npu_device_usable` 桩），验的是代码路径与判据，**不是卡**。

---

## 2 P1-1l：内核启用自证（最后一道静默假绿，已闭合）

### 2.1 before（实证同一趟训练里"真回落了却打 PASS 退 0"）
```
$ cd release && PYTHONPATH=. .venv/bin/python /tmp/b_before_ts.py --target cpu --device cpu
…/ascend/kernels/gdn_asc.py:724: RuntimeWarning: 方言后端 [gdn_backward[cpu]] 不可用，已回退 torch eager：…
P2-STEP-cpu loss0=0.910992 loss1=0.823897 reduced=True params_grad_ok=True
BEFORE-EXIT=0                    ← 零个"内核真走了"的断言
```

### 2.2 after（同一条命令）
```
$ cd release && .venv/bin/python ascend/train_step.py --target cpu --device cpu ; echo $?
NPU-REG skip not-imported target=cpu npu_avail=n/a cpu 轨不 import torch_npu（无 npu 机器会当场 ImportError）
P2-STEP-cpu loss0=0.910992 loss1=0.823897 reduced=True params_grad_ok=True
KERNEL-USED-FAIL kernels=13 blockers=1 fallbacks=1 target=cpu why=blockers_present,fallback_events blocker_keys=['gdn_backward[cpu]']
KERNEL-USED-BLOCK: 未达项=['blockers_present', 'fallback_events'] ⇒ 未计入门禁（cpu 轨的 gdn_backward[cpu] 回落是已登记事实；要强验就 SYS1_KERNEL_USAGE_STRICT=1）
AFTER-EXIT=0
$ SYS1_KERNEL_USAGE_STRICT=1 .venv/bin/python ascend/train_step.py --target cpu --device cpu ; echo $?
KERNEL-USED-BLOCK: 未达项=['blockers_present', 'fallback_events'] ⇒ exit=5
STRICT-EXIT=5                    ← 同一份真账本，门一开就咬住
```

### 2.3 三路证据与实现
`autograd_asc.kernel_usage_report(target, *, fallback_events, compiled, blockers)`（**纯函数**，判决与打印分离）：
① `ascend_env.compiled_keys()` 里 `<target>|` 前缀键 ≥1；② `ascend_env.blockers()=={}`；③ `fallback_events==0`。
判决行：`KERNEL-USED-PASS kernels=<n> blockers=0 fallbacks=0 target=<t>`；不满足则 `…-FAIL … why=<未达项>`
（附 `blocker_keys=` / `foreign_keys=` 便于读回到底走了哪一路）。
- `FALLBACK_WARNING_TOKEN="已回退 torch eager"` + `count_fallback_events(messages)`：token 正身在
  `ascend_env.py:234`（禁改件，只当可观测量，**不复制不修改**）。
- `train_step.main()` 用 **`warnings.showwarning` 钩子旁听**（原样转交 stderr，不吞公告、不改过滤器 ⇒
  日志形态与旧版逐字一致），把收到的条数喂给纯函数。口径：**按 key 去重的公告次数**（`record_blocker`
  本身一次性），所以 `fallbacks=1` 就是"这一个件回落了"。
- 纯函数**绝不调** `backend_available()`/`active_backend()`——那两个函数会把"设备不在场"**登记成新 blocker**，
  判决污染账本（已写进 docstring）。
- 门（gating）= `target==ascend` 或 `SYS1_KERNEL_USAGE_STRICT=1`；cpu 轨默认不进门禁，因为
  `gdn_backward[cpu]` 回落是本波**已登记事实**（见 `tests/test_ascend_train_step.py`），把它顶成红等于
  越权改 host 行为。强验钩子留着，负例就靠它。
- **判决行落点在 `main()` 而不是 `run()`**：`tests/test_ascend_train_step.py::test_train_step_cli_shape`
  断言 `run()` 的 stdout **最后一行**以 `P2-STEP-cpu ` 开头；tests 属禁改件 ⇒ 新行只能加在 `main()`。

### 2.4 负例自证（"伪造一次回退→断言必红"）
`B_probe_kernel_usage_negative.py`（host 22 ok / 0 MISS / exit 0）：
| 档 | 打断什么 | 期望 |
|---|---|---|
| M1 | 干净合成账本 | `KERNEL-USED-PASS kernels=3 blockers=0 fallbacks=0 target=ascend`（逐字钉格式） |
| M2 | ①拔掉（无 `ascend|` 键） | `FAIL why=kernels_absent` |
| M3 | ②伪造（塞一条 blocker） | `FAIL why=blockers_present` + 带出键名 |
| **M4** | **③真发一条与 `record_blocker` 同形的回退公告**（钩子数出来） | `FAIL why=fallback_events`，且 ①② 单独保持绿 ⇒ 三路各自 load-bearing |
| M4b | 无关 warning | 计数 0（不许把噪声算成回落） |
| M5 | 前缀错位（只有 `cpu|` 键却问 ascend） | `FAIL` + `foreign_keys=` 指出来 |
| E1/E2/E3 | 端到端 cpu / cpu+STRICT / ascend | `KERNEL-USED-FAIL` 但退 0 / 退 **5** / 先撞 H1 退 **4** |

**探针自身也被证伪过**（这一步差点漏掉，值得登记）：最初 `check(label, cond, evidence)` 把非空字符串
当真，判决行写错照样报 ok = **探针自己的假绿**。改成 `check(label, got, want)`（字符串→子串、其余→逐字相等）
后，用 sabotage 证明 load-bearing：
```
$ python3 - <<'PY'   # 把 M1 的期望 kernels 改成 999
... s.replace('check("kernels=3", rep["n_kernels"], 3)','check("kernels=3", rep["n_kernels"], 999)') ...
B-KU-DONE FAIL ['kernels=3']        ← 判据确实咬得住
```

---

## 3 H2/H3/H4：卡窗读法与计数

### 3.1 腿执行器（`card_run.sh:75-127`）
`run_leg <标签> <超时秒> <判决前缀正则> <FAIL 正则> <PASS 正则> <降级正则|-> <命令…>`
- **H2 两趟**：第一趟只打判决前缀行（额度 30），第二趟 `error:|FATAL|Traceback|RuntimeError|AssertionError`
  （额度 8）。**`aicore` 已去掉**——它命中 bisheng 命令行的 `-cce-aicore-*`，每件失败件白吃 3 行
  （readout_prod 实测两件同灭时判决行被完全挤掉）。原文整份另落 `$RUNLOG/<标签>.log`，屏上截断不丢证据。
- **H4 计数**：`PASS / DEGRADED / FAIL / FAIL-NO-VERDICT / FAIL-TIMEOUT / FAIL-EXIT<rc>`；
  **没有判决行一律 FAIL**（与 `compile_all_asc` 门禁同构）；PASS 还额外要求 `rc=0`；
  `DEGRADED`（`note=无设备⇒只判编译` 这种 PASS 形状的降级）**计进 fail**。
- 对照轨（`NUM-*`）标 `track=advisory`：失败进 `advisory_fail` 单列、不顶退码——与 `compile_all_asc`
  里 `ASC-PERF-*` 默认 non-blocking 的既有口径同构（Y 口径：NUM-* 不可当正确性判决引用）。
- **H3 窗末自扫**（`runner_done`）：`OVERLAY-OK|INJECT-OK|^FATAL|CASE-|npu_avail|NPU-REG|RuntimeError|
  Traceback|KERNEL-USED|ASC-|BENCH-|KERNELS-TGZ|note=无设备`，并打
  `RUNNER-DONE SUMMARY pass=… fail=… skipped_legs=… degraded=… advisory_fail=… exit=…`，`fail>0 ⇒ exit 1`。
- `runner_done` 是**唯一出口**（`STOP_AFTER=2/3` 分支也走它，并如实报 `skipped_legs`）。
- `$RUNLOG` 起跑前 `rm -f *.log tally.txt`（不清上次的账会把这次的计数顶歪）。

### 3.2 判决矩阵（`B_probe_cardrun_verdict_matrix.py`，host 33 ok / 0 MISS / exit 0）
不修改卡窗件：把 `card_run.sh` 的执行器与 [4b]/[4c] 两段**按行原样摘录**进沙盒，喂"判决行形状逐字对齐
真件"的假命令。S1 PROD-PASS⇒PASS；**S2 `note=无设备⇒只判编译`⇒DEGRADED 且整窗 exit 1**；S3 PROD-FAIL；
S4 无判决行⇒FAIL-NO-VERDICT；S5 PASS 行+rc≠0⇒FAIL-EXIT1；S6 rc=124⇒FAIL-TIMEOUT；S7 advisory 单列；
S8/S9 增采 C 两态；S10/S11 BENCH 门；S12/S12b trainstep 腿两态。

变异体自证（`CARDRUN_SRC` 只为此留的口子）：
```
$ CARDRUN_SRC=/tmp/b_m1.sh .venv/bin/python ascend/kernels/reconcile/B_probe_cardrun_verdict_matrix.py
  MISS 降级计进 fail（旧版这里就是 pass=1 的错觉）
  MISS 整窗 exit 1: 0
B-CR-DONE FAIL 降级计进 fail（旧版这里就是 pass=1 的错觉）,整窗 exit 1
$ CARDRUN_SRC=/tmp/b_m2.sh …
  MISS FAIL-EXIT1 / MISS PASS 计数为 0
B-CR-DONE FAIL FAIL-EXIT1,PASS 计数为 0
```
（M1=把 DEGRADED 不再计 fail；M2=PASS 不校验 rc。两者一改，对应负例立刻响 ⇒ 判据不是装饰。）

---

## 4 增采 C / D + E2 契约腿

- **增采 C**（`card_run.sh` [4c]，走 `run_leg`）：
  `tar czf ${CARDRUN_TGZ:-/tmp/sys1_kernels.tgz} -C "$TILELANG_CACHE_DIR" .` 后打
  `KERNELS-TGZ <path> size=<B>B entries=<n> kernels=<device_kernel.asc 数>`；
  `entries=0`（缓存是 mktemp 全新的，不可能"本来就空"）或 tar 失败 ⇒ `KERNELS-TGZ-FAIL` ⇒ 计一条 fail。
  容器干跑实测：`KERNELS-TGZ /tmp/p213b_kernels.tgz size=351963B entries=224 kernels=32` → VERDICT-PASS。
  包内形状（已解开验过）：`./0.1.15/linux-aarch64/kernels/<key-hash>/{device_kernel.asc,host_kernel.c,
  manifest.json,params.json,executable.so}`，`device_kernel.asc` 是**真源码**（`#include <tl_templates/ascend/…>`）。
  当场就出了一条 §12 结论：这 32 件里 `grep -rl "Cube|asc_gemm|alloc_l1|asc_copy_in_dcache"` **命中 0**，
  即本窗编译产物全是可移植标量路，**与 §12 intrinsic 是不同生成码族**（与 A2/A3 判读一致）——零额外编译拿到。
- **增采 D**（[3] 末，`ASC-products` 腿）：`env DMLAYA_ASCEND_TARGET=ascend python3 ascend/compile_all_asc.py
  --target ascend --products`。判决=**`门禁失败数=0`**（`^ASC-products FAIL|^ASC-PROD-skipped FAIL|门禁失败数=[1-9]`
  为 FAIL；非门禁的 `ASC-PERF-*` FAIL 不顶红）。容器实测 20 行判决 +
  `—— 汇总 target=ascend 接口腿 PASS=9 FAIL=0 产品腿 PASS=3 FAIL=0 对照腿 PASS=8 FAIL=0 SKIP=0 —— 门禁失败数=0` → PASS。
- **E2 契约腿**（[4b]）：`SYS1_BENCH` 未设或 `$SYS1/release/ascend/bench_card.py` 不在 ⇒
  只打 `CASE-BENCH SKIP …` + `skipped_legs++`，**默认不进窗、不影响退码**。真跑时按契约只用
  `^BENCH-…PASS/FAIL` 判（未去看 E2 的实现；若它的前缀形状不符，这条腿会判 `FAIL-NO-VERDICT` 而非误绿）。

---

## 5 容器干跑（改后卡窗的完整判决，`cann910b-h`）

```
docker exec cann910b-h bash -lc 'cd /tmp/p213b/release/ascend/port910b && SYS1_DIR=/tmp/p213b \
  CARDRUN_SKIP_CLONE=1 CARDRUN_LOGDIR=/tmp/p213b_legs CARDRUN_TGZ=/tmp/p213b_kernels.tgz bash card_run.sh'
```
逐腿判（无设备 ⇒ 全是"应当红"的形态，**这正是本次要证的**：窗子不再把降级读成通过）：
```
NPU-REG fail torch_npu:ModuleNotFoundError      [0] 段最前，早于任何腿
OVERLAY-OK headers=5 tpl=12 / INJECT-OK          [2] 自证三行齐
CASE-P11i-linear_prod  VERDICT-DEGRADED  rc=0    Y-linear_prod-ASC-COMPILE-PASS … note=无设备⇒只判编译
CASE-P11i-dw_prod      VERDICT-DEGRADED  rc=0
CASE-P11i-readout_prod VERDICT-DEGRADED  rc=0
CASE-PERF-*  ×5        VERDICT-FAIL rc=1 track=advisory   （NUM-*-FAIL … device string: npu）
CASE-delta             VERDICT-FAIL rc=1   NUM-delta_fwd FAIL run: ModuleNotFoundError: No module named 'torch_npu'
CASE-ASC-products      VERDICT-PASS rc=0   —— 门禁失败数=0
CASE-trainstep         VERDICT-FAIL rc=4   P2-STEP-ascend FAIL: NPU-REG-FAIL …（wave2 原文一字不差，但先自证）
CASE-BENCH             SKIP（SYS1_BENCH 未设）
CASE-KERNELS-TGZ       VERDICT-PASS rc=0   size=351963B entries=224 kernels=32
RUNNER-DONE SUMMARY pass=2 fail=5 skipped_legs=1 degraded=3 advisory_fail=5 exit=1   → DRY-EXIT=1
SUMMARY-DEGRADED note=无设备 命中=6 行
```
旧版同一景会打 `RUNNER-DONE`（裸一行）+ 每腿 `head -6` 里挤进来的 `-cce-aicore-*` 噪声，被读成"跑完=成功"。

---

## 6 卡上预期行（下窗逐字对照表；**本文不声称已验**）

| 位置 | 卡上应当逐字出现的形 | 若不符说明什么 |
|---|---|---|
| [0] | `NPU-REG ok <torch_npu.__version__> avail=True count=1 torch=…` | 缺 torch_npu/驱动 ⇒ 数值面作废，别烧窗 |
| [3] | `Y-<case>-PROD-PASS rel=<≤阈值> dir_disc=… reuse=<n>/<n>` | 出 `ASC-COMPILE-PASS … note=无设备` ⇒ 设备门没开（本窗已按 DEGRADED 计 fail） |
| [4] | `NPU-REG ok … target=ascend npu_avail=True` → `P2-STEP-ascend … reduced=True` → **`KERNEL-USED-PASS kernels=13 blockers=0 fallbacks=0 target=ascend`** | `kernels<13` ⇒ 有件回落；`blockers`/`fallbacks` 非 0 ⇒ 发射或探针失败落了 eager（本元假绿，现在腿级就红） |
| [4c] | `KERNELS-TGZ /tmp/sys1_kernels.tgz size=…B entries=… kernels=…` | `kernels=0` ⇒ 整窗没编译任何东西 |
| [5] | `RUNNER-DONE SUMMARY pass=… fail=0 … exit=0` | 任何 `fail>0` ⇒ 整窗 exit 1 |

（`kernels=13` 的来源 = A3_PREFLIGHT §3 实测的 train_step ascend 唯一键数。）

---

## 7 环境发现与遗留风险（须上呈，非本件可修）

1. **容器 `cann910b-h` 不是 cpu 轨的等价替身**：容器里一步 cpu 训练本身就崩——
   `ascend/kernels/lora_asc.py:137 _eager_grad` → `RuntimeError: element 0 of tensors does not require
   grad and does not have a grad_fn`；**origin/main 的原始 `train_step.py` 同崩（退 1）** ⇒ 与本件改动无关
   （host `.venv` py3.12.8/torch2.14.1 同一条命令退 0）。机制推测：`_eager_grad` 依赖 backward 执行期间
   grad mode 仍开启，容器那套 torch 构建不满足。**风险**：卡上 torch/torch_npu 构建若同类，`[4]` 腿会在
   lora 门面先崩（那时看到的是 `P2-STEP-ascend FAIL`，不是 `KERNEL-USED-FAIL`）。`lora_asc.py` 属禁改件，
   留给编排者裁决（建议 R20 单独立项，或在卡窗 [4] 前加一条 lora 门面最小探针）。
2. **口径不一致（已在窗侧显形，未擅改禁改件）**：`compile_all_asc.py --products` 把
   `Y-*-ASC-COMPILE-PASS note=无设备` 计成 **PASS**（门禁 0），而 `run_numerics` 直跑的同一条判决在我新
   腿级被判 **DEGRADED（计 fail）**。同一件事两套账 ⇒ 卡上真机时前者一定给 `PASS`，读者会误以为"产品件
   数值过了"。本文与窗末 `SUMMARY-DEGRADED … 命中=N 行` 把它摆在明面；**建议**：`compile_all_asc` 的门禁
   将来把 `no_run=True` 的产品腿记 DEGRADED/SKIP（该件禁改，本轮只登记）。
3. 容器根盘曾被一次 `tar` 全量复制顶到 100%（`release/bench/ms_models` 是 GB 级缓存）。已清掉我自己
   放的那份 7.2G（`/tmp/b_sys1`），现 5.2G 可用（87%）。**后续代理别再整树 tar**——按 §8 的增量配方只送
   `ascend/{train_step,autograd_asc,compile_all_asc,kernels,port910b/{card_run.sh,run_numerics.py,run_delta_card.py,patches}}`。
4. 并行代理已在同一工作树留下改动（`git status`：`kernels/gemm_asc.py`、`gemm_bwd_dw_asc.py`、`rope_asc.py`、
   `port910b/patches/port910b_compat.h` 为 M，`ascend/bench_card.py` 等 ??）。**本文与容器干跑用的是
   origin/main + B 的 overlay**，不含他们的改动 ⇒ 合并后需重跑 §8 的三条验证（尤其 `card_run.sh` 干跑）。

---

## 8 复跑配方

```bash
# host（三条必跑）
cd $REPO_ROOT/release
bash -n ascend/port910b/card_run.sh
.venv/bin/python -m pytest tests -q -m "not integration"        # 340 passed, 3 skipped, 3 deselected
for p in B_probe_h1_npu_reg B_probe_kernel_usage_negative B_probe_cardrun_verdict_matrix; do
  .venv/bin/python ascend/kernels/reconcile/$p.py; done          # B-H1-DONE PASS / B-KU-DONE PASS / B-CR-DONE PASS
.venv/bin/python ascend/train_step.py --target cpu --device cpu   # KERNEL-USED-FAIL … target=cpu，退 0
SYS1_KERNEL_USAGE_STRICT=1 .venv/bin/python ascend/train_step.py --target cpu --device cpu; echo $?   # 5
.venv/bin/python ascend/train_step.py --target ascend; echo $?                                        # 4（NPU-REG fail）

# 容器（A3/A2 同款配方 + 干跑）
docker exec cann910b-h bash -lc 'source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh; \
  export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201'
# 增量送件（勿整树 tar，见 §7.3）：
cd $REPO_ROOT/release && tar --exclude __pycache__ -cf - \
  ascend/train_step.py ascend/autograd_asc.py ascend/compile_all_asc.py ascend/kernels \
  ascend/port910b/card_run.sh ascend/port910b/run_numerics.py ascend/port910b/run_delta_card.py \
  ascend/port910b/patches | docker exec -i cann910b-h bash -c 'tar -xf - -C /tmp/p213b/release'
# 卡窗干跑（本文 §5 的出处）：
docker exec cann910b-h bash -lc 'cd /tmp/p213b/release/ascend/port910b && SYS1_DIR=/tmp/p213b \
  CARDRUN_SKIP_CLONE=1 CARDRUN_LOGDIR=/tmp/p213b_legs CARDRUN_TGZ=/tmp/p213b_kernels.tgz bash card_run.sh'
```
