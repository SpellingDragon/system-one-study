# D2_legs_fix — P1-1r：D2 变体腿"卡上不可用"的归因与修复

**变更**：p2-13 / **P1-1r**（任务书：归因工具自身缺陷）　**执行代理**：T2　**日期**：2026-10-11
**容器**：`cann910b-h`（CANN 8.5.0；`/Users/pengweiye/Documents/codes/system-one/release → /work` 为 bind 挂载，
故容器内 `/work/ascend/port910b/attempts/*` 即仓内 `release/ascend/port910b/attempts/*`）
**唯一可改产品面**：`release/ascend/port910b/attempts/D2_probe_legs.py`
**修复后指纹**：`md5 fead1092c0976b143fe34048c610c18a`（408 行，原 263 行）
**compat 头指纹**（复现前置，未改）：`md5 ca023c2e01746ca52c87f36bcea93ee9`
**一句话**：第二窗"`--combos A,C --card` 两趟作废"的**真因是工具把 repo 根写死成容器路径
`/work/ascend`、而卡上 repo 在 `/tmp/sys1`** ⇒ 逐腿 `FileNotFoundError` ⇒ 子进程 rc=1、缓存 0 件、
`aibin … fp=e3b0c44298fc1c14`（空串 md5）、无判决行。已 1:1 复现并修好；同窗还捎带修掉三个
"让失败静默/让判据不可能" 的缺陷（`--card` 空开关、子进程输出被 pipe 吞、`so=` 行结构上不可出）。

---

## 0. 结论三行

1. **归因＝(c) 路径/工作目录**：`D2_probe_legs.py` 顶部 `REPO = os.environ.get("P910B_REPO", "/work/ascend")`
   —— `/work/ascend` 是**容器**布局（`RUNNERS` 全部指它）；**卡上 repo 根在 `/tmp/sys1`**
   （`card_run.sh:68` `SYS1=${SYS1_DIR:-/tmp/sys1}`，`window2_card.sh` 也是这套）。卡上没人 export
   `P910B_REPO` ⇒ `…/port910b/run_numerics.py` 不存在 ⇒ 每腿 `open(script)` 抛 `FileNotFoundError`。
   逐字节同型现场见 §2.2。**不是** (a) env 缺失、**不是** (b) patch 没跨进程、**不是** (d) `--card`
   引用了不存在的 CANN 路径（`--card` 分支里根本没有 CANN 路径）。
2. **另有三个"次生缺陷"必须一起修**，否则即便路径对了，判据仍会假绿/静默：
   `--card` 是**空开关**（真机数值腿从未执行）；子进程 stdout 走 pipe + 缓冲 ⇒ 硬杀即"静默 rc=1"；
   `so=DIFF/SAME` 行**只在给 `--baseline` 时才打** ⇒ 第二窗命令 `--combos A,C --card` 结构上不可能出该行。
3. **修复后容器实证**：`--combos default,A,B,C --baseline <snap>` → **四趟各出三条判决行、
   `D2-LEGS-SUMMARY valid=4/4`、rc=0**；`default` 十一核 `asc=SAME so=SAME`、九件门九数
   `2850/2250/3625/2106/2911/4003/1864/922/2842` **与 A2 基线逐字同**；`+A` so=DIFF、`+B` so=SAME、
   `+C` so=DIFF —— 与 `D2_variants.md §6.1/§6.2` 指纹**逐条吻合**。

---

## 1. 改了什么（逐条，均为 `D2_probe_legs.py` 内部）

| # | 缺陷（原行为） | 修法 | 对应第 3 步要求的哪一条 |
|---|---|---|---|
| F1 | `REPO` 写死 `/work/ascend`；repo 根不对时每腿 `FileNotFoundError`，**静默 rc=1、0 件、无 NUM 行** | `main()` 开头 **precheck**：逐腿 `os.path.exists`，缺任一 → 打 `FATAL D2-PRECHECK missing-runner leg=… path=…` + `P910B_REPO` 正确写法提示，**退 2**（不再进入假跑） | "任何一步失败都打印可归因的判决行而非静默 rc=1" |
| F2 | `--card` 空开关：父只"少传 `--compile-only`"，子 `compile_only` 默认仍是 True ⇒ 真机数值腿从不执行 | 父**显式**给子传 `--run`/`--no-run`（`spawn(..., do_run, ...)`）；子 `--run`=store_true、`--no-run`=store_false，`SEC12_LEGS and not a.run ⇒ 才加 --no-run` | 让 `--card` 名副其实（否则"卡上跑"永远拿到编译面 n/a） |
| F3 | `spawn()` 用 `capture_output=True`（pipe）；子进程被硬杀时**缓冲丢尾** ⇒ 输出为空 = 静默 | 子进程用 `sys.executable -u` + `PYTHONUNBUFFERED=1`，stdout/stderr **落文件**（非 pipe）；父**无条件**回显 + 打 `[<c>] child-rc=<n> log=… bytes=<n>`；0 字节且非 0 退码时补一句"子进程零输出即退出（硬杀/未启动）" | 子进程 stderr 单独捕获、不被父吞 |
| F4 | 子进程 `## inprocess` 注入行在 import 之后；import 死则该行也没了 | 子进程**第一行**先打 `D2-CHILD-START combo=… cache=… run=…`；import/patch 失败打 `D2-INJECT-FAIL reason=…`；腿脚本缺失打 `D2-LEG-EXEC-FAIL leg=… reason=FileNotFoundError` | 注入确认行必出 |
| F5 | `so=DIFF/SAME` 行**只**在 `--baseline` 时由 `compare()` 打；无 baseline 时②结构上永缺 | 每腿**无条件**打 `[<c>] SO-VERDICT so=<SAME|DIFF|BASE>`：有 `--baseline` 比基线；无 baseline 以**首趟 combo 为窗内参照**（参照腿记 `BASE`）；0 核腿记 `NOKERNELS`（⇒ 必判 INVALID） | 判据②必出 |
| F6 | 无"三判决行齐否"的刚性检查 ⇒ 缺行照算数据 | 每腿打 `[<c>] LEG-VERDICT D2-LEG-OK | D2-LEG-INVALID missing=inject/num/so`；收尾打 `D2-LEGS-SUMMARY valid=k/n combos=… invalid=…`；**有任一 INVALID ⇒ 进程退 1**（"不许算数据"落到退码上） | "缺任一 ⇒ 该趟自判 `D2-LEG-INVALID`" |
| F7 | （顺带）缓存根写死 `/tmp`，盘紧时无路可走 | 新增 `--cache-root`（env `D2_CACHE_ROOT`，默认 `/tmp`）⇒ 卡上可指 `/dev/shm`；`--log-dir` 可指子进程日志落点 | 支持任务书的 `TILELANG_CACHE_DIR=/dev/shm/...` 约束 |

**没改**：`SEC12` 腿定义、`COMBO_FLAGS`、指纹 `scan()`/`fp()`/`compare()` 判读口径、`--digest-only`、
`--inprocess` 语义、`compile_all` 只进 `default` 的规则。**判定逻辑一字未动**，只是把"失败"变可见、把"缺判据"变硬拦。

---

## 2. 归因结论（每条附原文证据）

### 2.1 卡上 repo 不在 `/work/ascend`（病灶根因）

卡上布局证据（**只读**，均来自仓内既有文件原文）：

* `card_run.sh:68`：`SYS1=${SYS1_DIR:-/tmp/sys1}`；`:177` `rm -rf "$SYS1"; git clone … "$SYS1"` ⇒ 卡上 repo 根 = `$SYS1` = **`/tmp/sys1`**。
* `card_run.sh:183` `cd "$SYS1/release/ascend/port910b"`；`:185` `TLROOT=…/site-packages/tilelang`
  （`w2_cardrun.log` 实测 `TLROOT=/home/ma-user/anaconda3/envs/PyTorch-2.7.1/lib/python3.12/site-packages/tilelang`）。
* 而 `D2_probe_legs.py`（改前）:39 `REPO = os.environ.get("P910B_REPO", "/work/ascend")`，
  `RUNNERS` 全部由它拼出 ⇒ 卡上（无 `P910B_REPO`）指向**不存在的** `/work/ascend/port910b/run_numerics.py`。

### 2.2 逐字节同型现场（容器 1:1 复现，命令 + 真实输出）

复现命令（容器 `cann910b-h`，走**编译面**，只需 bisheng 不需卡）：

```bash
# 旧版脚本（改前），把 REPO 指到不存在的路径 = 模拟"卡上没 export P910B_REPO"
bash -lc 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; cd /work/ascend/port910b/attempts; \
          P910B_REPO=/tmp/does_not_exist python3 D2_probe_legs.py --combos A --card'
```

真实输出（**与第二窗 D2 战报逐字节同型**）：

```
════════ combo=A  legs=gemm_l1_intrinsic,dw_intrinsic  cache=/tmp/d2_legs_A ════════
## inprocess combo=A flags=['-DTL910B_VARIANT_A'] cache=/tmp/d2_legs_A tilelang=0.1.15
   $ tmp/does_not_exist/port910b/run_numerics.py --only gemm_l1_intrinsic --track card --no-run
   -> EXCEPTION FileNotFoundError: [Errno 2] No such file or directory: '/tmp/does_not_exist/port910b/run_numerics.py'
   $ tmp/does_not_exist/port910b/run_numerics.py --only dw_intrinsic --track card --no-run
   -> EXCEPTION FileNotFoundError: [Errno 2] No such file or directory: '/tmp/does_not_exist/port910b/run_numerics.py'
   [A] 每核指纹（asc=源码面 so=机器码面）：
        aibin 0 件 总=0B fp=e3b0c44298fc1c14
   [A] spawn rc=1
```

对照第二窗战报原文：`[A] aibin 0 件 总=0B fp=e3b0c44298fc1c14`、`spawn rc=1`、缺 `NUM-*` 与
`so=DIFF/SAME` —— **`fp`/`rc`/缺行三处逐字符吻合**。
（`fp=e3b0c44298fc1c14` = `sha256("")[:16]`，即 aibin 列表为空时的指纹；容器内 `sha256("")[:16]`
实测 = `e3b0c44298fc1c14`，见 §5 命令 5。）

**为什么不是别的四类**（逐条排除，附依据）：

* (a) **子进程 env/source 缺失**：子进程 `subprocess.run` 未传 `env=` ⇒ 继承父环境；父若 source 了
  CANN，子也有。且同命令**带正确 REPO** 时编译面全绿（§5 命令 1）⇒ env 不是原因。
* (b) **monkeypatch 未跨进程**：patch 发生在**子进程内**（`inprocess_main` 自己 patch），不依赖跨进程；
  证据 = 容器内 `## inprocess combo=A flags=['-DTL910B_VARIANT_A']` 与 `+A so=DIFF` 同时出现（§5 命令 2）。
* (c) **路径/工作目录**：**✅ 命中**（§2.1/§2.2）。
* (d) **`--card` 引用了卡上不存在的 CANN 路径**：`D2_probe_legs.py` 的 `--card` 分支只有
  `if not card: cmd.append("--compile-only")` 一行，**不含任何 CANN 路径**；CANN 路径只在
  `card_run.sh:61-63` / `window2_card.sh:16` 出现（那是禁改件、非本工具）。⇒ 不成立。
* (e) 其它：`--card` 空开关（§2.3）与缓冲丢尾（§2.4）是**并存的次生缺陷**，会让"路径对了也照样拿不到真机判决"。

### 2.3 次生缺陷 1：`--card` 是空开关（真机腿从未执行）

改前语义：`spawn(card=not a.compile_only)` → 传 `--card` 时**不**给子进程 `--compile-only`；
但子进程 `ap.set_defaults(compile_only=True)` ⇒ 子进程 `compile_only` 仍为 True ⇒ **仍加 `--no-run`**。
复现（改前，`--combos A --card`，输出里赫然还是 `--no-run`）：

```
   $ work/ascend/port910b/run_numerics.py --only gemm_l1_intrinsic --track card --no-run   ← 传了 --card 却仍 --no-run
NUM-gemm_l1_intrinsic-PASS rel=n/a (compile-only) track=performance
```

⇒ 第二窗即便路径正确，也不可能拿到真机 `rel=`；只会得到编译面 `n/a`（**假绿形状**）。
改后（§5 命令 4）：子进程命令行不再有 `--no-run`，真去执行、失败也**可归因**：

```
   $ work/ascend/port910b/run_numerics.py --only gemm_l1_intrinsic --track card
NUM-gemm_l1_intrinsic-FAIL RuntimeError: … device string: npu
NUM-dw_intrinsic-FAIL        AttributeError: 'Tensor' object has no attribute 'npu'
```

### 2.4 次生缺陷 2：子进程输出被 pipe + 缓冲吞掉（硬杀即"静默 rc=1"）

改前 `spawn()`：`subprocess.run(capture_output=True, text=True)` ⇒ 子进程 stdout 走 **pipe**（非 tty ⇒ 块缓冲），
且只在 `rc!=0` 时回显**末 6 行 stderr**。若子进程被 SIGKILL/段错误硬杀，**缓冲区尾部全丢** ⇒ 父只见
`spawn rc=1`、零诊断 = "静默"。改后子进程 `-u` + 输出落**文件**（硬杀也留已 flush 的尾）+ 父无条件打
`child-rc`/`bytes`，0 字节时显式标注"硬杀/未启动"。

### 2.5 次生缺陷 3：`so=` 行结构上不可出（无 baseline）

第二窗命令是 `--combos A,C --card`（**无 `--baseline`**）。改前 `so=DIFF/SAME` 只在 `compare()` 里打，
而 `compare()` 仅当 `base` 非空才被调用 ⇒ 该命令**必然**缺②。改后每腿无条件出 `SO-VERDICT so=…`
（无 baseline 时以首趟为窗内参照、参照腿记 `BASE`）⇒ `--combos A,C` 这种"仅变体腿"命令也能满足三判据。

---

## 3. 卡上重跑该怎么做（含 env 正确写法）

**唯一必须先做对的事：把 `P910B_REPO` 指到卡上的 `release/ascend` 目录**。**注意语义：这个变量
是 "ascend 包目录"（`release/ascend`），并非 repo 根** —— `RUNNERS` 由它拼出
`…/compile_all_asc.py` 与 `…/port910b/run_numerics.py`。卡上 repo 根 = `/tmp/sys1`
（`card_run.sh:68`）⇒ 该值 = **`/tmp/sys1/release/ascend`**。最稳的写法是**让它自己算**（不写死）：

```bash
# ── 卡上开窗后，先 source CANN（卡上路径是 8.5.2，与本容器的 8.5.0 不同）──
source /usr/local/Ascend/cann-8.5.2/set_env.sh

# ── 取 release/ascend 目录：先 git 自算 repo 根，再拼 /release/ascend（card_run.sh 已 clone 到 $SYS1=/tmp/sys1）──
SYS1=${SYS1_DIR:-/tmp/sys1}
export P910B_REPO=$(git -C "$SYS1" rev-parse --show-toplevel)/release/ascend   # = /tmp/sys1/release/ascend
echo "P910B_REPO=$P910B_REPO"                                    # 期望 /tmp/sys1/release/ascend
test -f "$P910B_REPO/port910b/run_numerics.py" && echo RUNNER-OK # 期望 RUNNER-OK

# ── 前置自检（开关在场）：§3 第 0 步，别省 ──
TL_SRC=$(python3 -c "import tilelang,os;print(os.path.join(os.path.dirname(tilelang.__file__),'src','tl_templates','ascend'))")
md5sum "$TL_SRC/port910b_compat.h"     # 期望 ca023c2e01746ca52c87f36bcea93ee9

# ── 卡上：先编译面自检（不需要卡，秒级），确认注入链与三判据 ──
cd "$P910B_REPO/port910b/attempts"
python3 D2_probe_legs.py --combos default,A,B,C --cache-root /dev/shm/d2

# ── 卡上：真机四趟（--card 去桩；default 自动当窗内基线；盘紧用 /dev/shm）──
python3 D2_probe_legs.py --combos default,A,B,C --card --cache-root /dev/shm/d2 \
        2>&1 | tee /dev/shm/d2_window.log
#   有已知good 产物目录时优先显式给基线（免把"default 自比"当等价）：
# python3 D2_probe_legs.py --combos default,A,B,C --card \
#        --baseline /tmp/sys1/d2_cache_base --cache-root /dev/shm/d2
```

**看什么（每趟三行齐 = 该趟有效）**：

```
## inprocess combo=<c> flags=[…]           ← ① 注入确认行
NUM-<leg>-PASS rel=… / …-FAIL <原文>        ← ③ 数值/编译判决行（含 507015 之类错误码原文）
[<c>] SO-VERDICT so=<SAME|DIFF|BASE>        ← ② 机器码面判决
[<c>] LEG-VERDICT D2-LEG-OK | D2-LEG-INVALID missing=…
D2-LEGS-SUMMARY valid=k/n combos=… invalid=…
```

**停手条件（改后由工具自判，不再需要人肉）：**
* 见 `FATAL D2-PRECHECK missing-runner …` → **就是本文档的病**，`P910B_REPO` 没指对，退 2；别继续烧卡。
* 某腿 `D2-LEG-INVALID` → 该腿**不计入归因**（工具已自动排除），按 `missing=` 补链路后重跑该腿。
* `[<c>] SO-VERDICT so=SAME` 而 `c∈{A,C}` → 开关没进命令行（注入链断），该趟无论红绿都不采信。
* 全绿但 `+B` 变色 / dw 腿在 `+C` 下变色 → 覆盖关系地基动摇，**停手回报**（`D2_variants.md §3` 停窗条不变）。

> env 写法要点：① `P910B_REPO` 必须指 **`release/ascend` 目录**（含 `port910b/run_numerics.py` 的那层，**不是 repo 根**；卡上 = `/tmp/sys1/release/ascend`）；
> ② CANN 用卡上的 `cann-8.5.2`（不是 8.5.0）；③ 盘紧把 `--cache-root` 指 `/dev/shm`；
> ④ **不要**去设 `TL_DEVICE_COMPILE_FLAGS`（`D2_variants.md §2` 已论证该 env 路不稳，patch 与 env
> 传导会被 tvm 配置层吞），注入只走本脚本的 monkeypatch。

---

## 4. 修好后"这扇窗"能判哪几个真值表行（X1–X7）

**结论：X1–X7 全部可判**——第二轮真值表（`D2_variants.md §5.2`）的每一行只由
`基线 / +A / +C` 三趟的红绿与错值决定，**`+B` 不在表内**（只在 §5.3 作阴性对照）。所以本窗只需
`--combos default,A,C --card`（B 可选）即可覆盖全部七行。逐行对应关系：

| 行 | 判据（`gemm_l1_intrinsic` 腿） | 本窗所需腿 | 归因 |
|---|---|---|---|
| **X1** | 基线 R，**+A G** | default+A | 写侧 ND2NZ 取参族错、朴素 ND 才对 ⇒ **否决性**，P1 回退、立即停手回报 |
| **X2** | 基线 R，**+A G，+C G** | default+A+C（复核加 `+A+C`） | 读写两侧各一处真错、基线红=叠加 |
| **X3** | 基线 R，**+A R≠，+C R≠** | default+A+C（复核加 `+A+C`） | 两侧都有份但都不充分 ⇒ 三槽外还有病灶，转 U5/V3 |
| **X4** | 基线 R，**+A R≡，+C R≠** | default+A+C | E-1 结案"非主因"，主嫌读侧 U2/U3 |
| **X5** | 基线 R，**+A R≡，+C R≡** | default+A+C | 三开关槽位全无关 ⇒ 病灶在 §12 之外，申请 VARIANT-D |
| **X6** | 基线 **G**，+A R，**+C G** | default+A（+C 对照） | 现态绿、A 必要、C 不敏感 ⇒ E-2/E-4 需更大 tile |
| **X7** | 基线 **G**，+A R，+C R | default+A+C | 两侧都必要，最强结案 |

**两点如实标注（不得把降级写成满血）：**

1. **X3/X4/X5 的"R≡ vs R≠"区分度取决于错值指纹**。`run_numerics.py` 只在绿时打印 `rel=`，
   红时打 `NUM-…-FAIL <错误码原文>`（如 `507015`）。若窗内 `run_numerics` 未落张量级误差，
   `R≡/R≠` 退化为"错误码原文是否相同"，X3/X4/X5 区分度下降 —— **必须在结论里注明该降级**
   （此约束 `D2_variants.md §5.2` 末尾已立，与本修复无关）。
2. **本"窗"= 卡上开窗**。当前容器**无设备**，只能证编译面与三判据链；真机 `rel=`/错误码需上卡。
   §5 四条实证均为**容器编译面 + 合成负例**，这是本任务"不得把修好建立在编译面通过上"的纪律下
   所能给的最强证据 —— 病灶本身（路径/开关/缓冲/判据行）全部是**容器可判**的，真机数值是修好之后
   才谈得上的下一步。

---

## 5. 容器实证记录（逐条命令 + 真实输出摘要）

容器 `cann910b-h`；`/Users/pengweiye/Documents/codes/system-one/release → /work`（bind）⇒
容器里跑的就是**仓内那份** `D2_probe_legs.py`（`md5 fead1092…`）。

**命令 1 · 编译面自证（旧病灶对照，固定 REPO 正确）**
`python3 D2_probe_legs.py --combos A,C`（改前）→ `[A]/[C] spawn rc=0`，两腿各 `NUM-…-PASS rel=n/a (compile-only)`
+ 指纹。**证明环境/注入链本身没问题**，问题只在路径。

**命令 2 · 修复后四趟 + 基线（核心验收）**
```bash
# 先建基线快照（编译面，冷缓存）
python3 D2_probe_legs.py --combos default --cache-root /dev/shm/d2base
# 再四趟对照
python3 D2_probe_legs.py --combos default,A,B,C \
        --baseline /dev/shm/d2base/d2_legs_default --cache-root /dev/shm/d2run
```
真实输出（摘要）：
```
STEP2 rc=0   inject_lines=4  num_lines=8  so_lines=4
[default] SO-VERDICT so=SAME  （逐核 so 同基线（11 核））
[default] LEG-VERDICT D2-LEG-OK
[A] SO-VERDICT so=DIFF  （机器码变了：dW_l0tr_kernel#0,main_kernel#0）
[A] LEG-VERDICT D2-LEG-OK
[B] SO-VERDICT so=SAME  （逐核 so 同基线（2 核））
[B] LEG-VERDICT D2-LEG-OK
[C] SO-VERDICT so=DIFF  （机器码变了：main_kernel#0）
[C] LEG-VERDICT D2-LEG-OK
D2-LEGS-SUMMARY valid=4/4 combos=default,A,B,C invalid=（无）
--- 结论表 ---
 [default vs 基线]  11 核全 asc=SAME so=SAME；aibin 集合交集 11/11
 [+A vs default]    dW_l0tr_kernel#0 so=DIFF、main_kernel#0 so=DIFF；aibin 0/11 ⇒ 活开关
 [+B vs default]    两核 so=SAME；aibin 2/11 ⇒ 空开关
 [+C vs default]    main_kernel#0 so=DIFF、dW so=SAME；aibin 1/11 ⇒ 活开关
```
**九数逐字同基线**（`default` 腿 `compile_all` 原文）：`ASC-gemm .o~2850B / gemm_bwd_dw 2250B / add_ln 3625B /
rope 2106B / attn_sw 2911B / gdn 4003B / gdn_conv 1864B / letter_readout 922B / lora 2842B`，
`接口腿 PASS=9 FAIL=0 门禁失败数=0` —— 与 `D2_variants.md §6.1` 基线**逐字一致**。
（指纹亦逐条吻合：`main_kernel so=25192B 1250079b185df42c` = §6.2 基线值。）

**命令 3 · 无 baseline 形态（证明"仅变体腿"命令也满足三判据）**
`python3 D2_probe_legs.py --combos default,A,B,C --cache-root /dev/shm/d2nb` → `rc=0 valid=4/4`；
`default so=BASE`，A/C `so=DIFF`，B `so=SAME`。

**命令 4 · `--card` 修复自证（`--no-run` 消失、真机失败可归因）**
`python3 D2_probe_legs.py --combos A --card --cache-root /dev/shm/d2card`（容器无卡）→
子进程命令行为 `… --track card`（**无 `--no-run`**）；两腿出 `NUM-gemm_l1_intrinsic-FAIL RuntimeError: … device string: npu`
与 `NUM-dw_intrinsic-FAIL AttributeError: 'Tensor' object has no attribute 'npu'`；
`child-rc=1` 且 log 文件完整（子进程 stderr 未被吞）。**失败仍出判决行 ⇒ 不静默。**

**命令 5 · 空串 md5 校验（对齐第二窗指纹）**
`python3 -c "import hashlib;print(hashlib.sha256(b'').hexdigest()[:16])"` → `e3b0c44298fc1c14`
（与第二窗 `[A] … fp=e3b0c44298fc1c14` 一致，坐实"0 件"= 空列表指纹）。

**命令 6 · 负例自证（三判据刚性）：合成 runner**
```bash
mkdir -p /dev/shm/fakerepo/port910b
: > /dev/shm/fakerepo/compile_all_asc.py
: > /dev/shm/fakerepo/port910b/run_numerics.py       # 存在但空 ⇒ exec 成功、0 判决行
P910B_REPO=/dev/shm/fakerepo python3 D2_probe_legs.py --combos A --cache-root /dev/shm/d2neg2
```
真实输出：
```
D2-PRECHECK-OK runner=… 全部在场
[A] SO-VERDICT so=NOKERNELS  （本腿 0 核（编译没出件 ⇒ 无数据可算））
[A] LEG-VERDICT D2-LEG-INVALID missing=num(NUM-*),so(so=NOKERNELS)
D2-LEGS-SUMMARY valid=0/1 combos=A invalid=A        （rc=1）
```

**命令 7 · precheck FATAL 自证（第二窗病灶的"解药"本身）**
`P910B_REPO=/tmp/sys1_wrong python3 D2_probe_legs.py --combos A,C --card` → **rc=2**：
```
FATAL D2-PRECHECK missing-runner leg=compile_all path=/tmp/sys1_wrong/compile_all_asc.py
FATAL D2-PRECHECK missing-runner leg=gemm_l1_intrinsic path=/tmp/sys1_wrong/port910b/run_numerics.py
FATAL D2-PRECHECK missing-runner leg=dw_intrinsic path=/tmp/sys1_wrong/port910b/run_numerics.py
FATAL D2-PRECHECK 提示：P910B_REPO 指 <repo>/release/ascend 包目录（非 repo 根）；请 export P910B_REPO=$(cd <repo> && git rev-parse --show-toplevel)/release/ascend（卡上实测为 /tmp/sys1/release/ascend）后重跑。缺件腿一律不得算数据。
```
即：同样的"路径写错"场景，第二窗表现为**静默 rc=1 + 0 件**，现在表现为**一行 FATAL + 写法提示 + rc=2**。

---

## 6. 白名单遵循与产物清单

**只改 1 个白名单文件**：`release/ascend/port910b/attempts/D2_probe_legs.py`（`md5 fead1092c0976b143fe34048c610c18a`，408 行）。
**新建**：`release/ascend/port910b/attempts/D2_legs_fix.md`（本文件）。
**未新建** `D2_probe_*.py`（无需额外探针，容器命令即证据）。

**未碰（逐项自证）**：trunk / `patches/port910b_compat.h`（`md5 ca023c2e…` 未变）、`card_run.sh`、
`run_numerics.py`、`compile_all_asc.py`、`bench_card.py`（V2 写面）、`reconcile/S3_*`/`U2_*`（S3/U2 写面）、
任何 `*_asc.py`/`*_kernel.py`、`runs/**`、任何 `tasks.md`。**全程未用 git 写操作**（未 commit/add/config）。
容器内临时件只落 `/dev/shm`（`d2base`/`d2run`/`d2nb`/`d2neg2`/`fakerepo`/`d2_logs`）与 `/work/…/attempts/__pycache__`
（已清），仓内无过程件残留。

**待编排者落地（非本代理权限）**：`tasks.md` 的 P1-1r 复选框；卡上下窗按 §3 重跑；真机 `rel=`/错误码回填 `X_FRACTAL_EVIDENCE`。
