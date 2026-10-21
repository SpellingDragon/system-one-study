# A3_PREFLIGHT.md — p2-13 卡前干跑（无卡整链核验）· 代理 A3

**窗口**：下一次 910B 付费窗，入口 `release/ascend/port910b/card_run.sh`
**干跑环境**：host（macOS，`release/.venv` py3.12，无 npu）+ 容器 `cann910b-h`
（aarch64 / py3.10 / CANN 8.5.0 / pip tilelang 0.1.15 / bisheng 可编、**无设备**）
**纪律**：A3 全程只读 + 只跑，未改任何产品/配置件（`card_run.sh`、`run_numerics.py`、
`*_asc.py`、trunk 一字未动）。新建仅限本件与 `A3_probe_*.py` 三枚探针。

**一句话结论**：**链路本身无隐患，但有一个能废掉整窗的环境前提没被任何判决件覆盖（H1：
`torch_npu` 没人 import）**——它不会报错给编排者看，只会把产品件三腿**静默降级成
"ASC-COMPILE-PASS note=无设备⇒只判编译"**，并把 [4] 打成一行裸 `RuntimeError`。
处理掉 H1（一条 [0] 段探针 + 一句判读口径）就是"一次过、零归因成本"。

---

## 1 clone 一致性 —— **通过**

| 判据 | 实测 |
|---|---|
| 卡上 `git clone --depth 1 <https://github.com/SpellingDragon/system-one-study.git>` 拿到的分支/提交 | `git ls-remote --symref origin HEAD` → `ref: refs/heads/main HEAD` / `eaf25a705b33b2c4d1cc9d002abd6eba56cf10c7` |
| 本地 HEAD 与 origin/main | `git rev-parse HEAD origin/main` → 两个都是 `eaf25a7`（同一提交） |
| 本地是否有"写了但没入库"的卡窗件 | `git status --porcelain release/` → **空**（`--ignored` 只剩 `__pycache__/`、`.venv/`、`cann_cache/`、`runs/…safetensors` 之类与卡窗无关的缓存） |
| 卡窗依赖闭包（card_run.sh 字面件 + 三入口 import 闭包 + 数据引用） | `A3_probe_clone_deps.py` → **NEED_TOTAL=46 / IN_ORIGIN_MAIN=46 / MISSING=0 / A3-CLONE-DEPS-PASS** |

复跑：

```bash
cd $REPO_ROOT
git ls-remote --symref origin HEAD
git rev-parse HEAD origin/main
git status --porcelain release/
release/.venv/bin/python release/ascend/kernels/reconcile/A3_probe_clone_deps.py; echo EXIT=$?
```

已点名核验的在册件：`patches/{port910b_compat,common,debug,gemm,dcache_bypass,numeric_limits}.h`
+ `patches/c_api/asc_simd.h` + `patches/simt_api/{asc_simt,asc_fp16,asc_bf16,asc_fp8}.h`
+ `patches/patch_bisheng.py`、`run_numerics.py`、`run_delta_card.py`、
`ascend/train_step.py`、`ascend/autograd_asc.py`、`ascend/kernels/*_asc.py`(9) + `*_kernel.py`(8) +
`ascend_env.py`、`attempts/{A,B,C,D,E,F}/*`、`ascend/compile_all_asc.py`。

数据件（最容易"入库成指针"的那个）：`attempts/D/dw_pairs.npz` 无 LFS
（`git check-attr filter` → unspecified），blob 3,018,590B == 工作区 3,018,590B，
容器内 clone 快照 md5 `c35f3ddb5ef62d4e311cea2871b02d16` 与本地逐位同，
`np.load` 实测可读 keys=`['dY','X','exact','exact_T','tiled','fp16out']`。

**⚠ 口径提醒（不是隐患，是描述漂移）**：任务书写的 `[3] --only {gemm_l1,dw,addln,readout,gdn}`
是 **bundle_ondemand.sh 的旧列表**；`card_run.sh` 实跑的是产品件
`linear_prod / dw_prod / readout_prod`（`--track card`）+ 对照轨
`gemm_l1_intrinsic / dw_intrinsic / readout_intrinsic / addln_intrinsic / gdn`。
旧名在 `run_numerics.py` 里是 `DEPRECATED` 别名（stderr 打 `[note] … 弃用别名`），
所以 `bundle_ondemand.sh` 仍能跑，但它量的永远是 **performance 轨**，不出产品判决。

---

## 2 overlay 幂等与完整 —— **通过**（R14 负例已实测响）

容器 `cann910b-h` 内用 **`git archive origin/main` 铺的 clone 快照**（`/tmp/a3sys1`，
与卡上 clone 逐字节同源，零网络零 git 写）跑 `card_run.sh` 的 [0]–[2]：

```bash
cd $REPO_ROOT
git archive --format=tar origin/main release | docker exec -i cann910b-h bash -c 'mkdir -p /tmp/a3sys1 && tar -x -C /tmp/a3sys1'
docker exec cann910b-h bash -c 'SYS1_DIR=/tmp/a3sys1 CARDRUN_SKIP_CLONE=1 CARDRUN_STOP_AFTER=2 \
  bash /tmp/a3sys1/release/ascend/port910b/card_run.sh'
```

实测判决行：

```
OVERLAY-OK headers=5 tpl=12
compat §12+fill: 7
bisheng.py already patched (/usr/local/…/tilelang/contrib/bisheng.py)
INJECT-OK
STOP-AFTER-2 OVERLAY-OK        （脚本 exit 0）
```

三方 md5 一致（pip 生效件 == 卡上 clone 的 patches/ == tilelang 主仓 trunk），11 个文件全 SAME：

| 件 | pip vs clone(patches) | 值 |
|---|---|---|
| `port910b_compat.h` | SAME | `f56432b0423174216da50090a1125a24`（trunk 同值） |
| `common/debug/gemm/dcache_bypass/numeric_limits.h` | SAME | `46d60f36… / 72ac39a3… / 008c901d… / 25984215… / 3e43e542…`（trunk 同值） |
| `c_api/asc_simd.h`、`simt_api/{asc_simt,asc_fp16,asc_bf16,asc_fp8}.h` | SAME | `3f9f041a… / ea50bbed… / a8b26287… / 70c36cd4… / a36f621e…`（trunk 同值） |

锚点计数（**生效件 = pip 树里那份**，不是仓内副本）：

```
copy_gm_to_cbuf_multi_nd2nz = 7    asc_fill_l1 = 7    (uint8_t)sid = 0
```

⇒ `asc_fill_l1`/nd2nz 成套在场、**Z 的 sid 回穿已清零**（预期 0，实测 0），
`card_run.sh:80` 的 `compat §12+fill` = 7∪0 = **7**，与实跑输出一致。

幂等性：overlay 连跑 3 次（正 → 负例删件 → 正）后 pip md5 复原为 `3f9f041a…`，
`patch_bisheng.py` 第二次起报 `already patched`（幂等 + py_compile 自证）⇒ **重复开窗安全**。

R14 负例（按 Y 留的自证钩子，**故意删一个头文件再跑**）：

```bash
docker exec cann910b-h bash -c 'SYS1_DIR=/tmp/a3sys1 CARDRUN_SKIP_CLONE=1 CARDRUN_STOP_AFTER=2 \
  Y_SELFTEST_DROP_CAPI=1 bash /tmp/a3sys1/release/ascend/port910b/card_run.sh; echo SCRIPT_EXIT=$?'
→ FATAL overlay-missing: /usr/local/…/tilelang/src/c_api/asc_simd.h  ← 桩头没落到 -I 根
→ SCRIPT_EXIT=1        （真 exit 1，不是 warning）
```

**顺带取证（A3 的一次自摆乌龙，价值反而最高）**：我在负例之后忘了复原 overlay，直接跑了
[3] 三腿，于是把 wave2 的病灶**在 2026-10-11 又活了一遍**：

```
/usr/local/…/tilelang/src/tl_templates/ascend/debug.h:12:10: fatal error: 'c_api/asc_simd.h' file not found
Y-linear_prod-PROD-FAIL AssertionError: 假绿守卫：compiled_keys 无 ascend|*gemm[ascend]* …
```

⇒ 结论 ①：`-I` 根确实是 `tilelang/3rdparty/../src`（= `tilelang/src`，Y 的 ② 修对了）；
⇒ 结论 ②：**假绿守卫 load-bearing**，overlay 缺件会被产品腿判成 FAIL 而不是假 PASS；
⇒ 结论 ③（隐患 H2）：**判决行会被 `head -6` 挤掉**——见 §5。

小遗留（登记，不改）：仓根 `release/ascend/port910b/port910b_compat.h` 是一份**陈旧副本**
（`d9caa7be…` ≠ `f56432b0…`），卡窗链无人引用它（`card_run.sh`/`bundle_ondemand.sh` 都从
`patches/` 取），但人会照着它 diff 出错，建议后续派单删除或同步。

---

## 3 train_step 在 ascend 目标的静态可达性 —— **可达，但撞 H1**

判门（T 代理守卫法，`_npu_present` 桩 True；容器 py3.10）：

```bash
docker cp release/ascend/kernels/reconcile/A3_probe_trainstep_route.py cann910b-h:/root/
docker exec cann910b-h bash -c 'cd /tmp/a3sys1/release && A3_ROOT=/tmp/a3sys1/release \
  TILELANG_CACHE_DIR=$(mktemp -d) python3 /root/A3_probe_trainstep_route.py keys'
A3-GATE before: (ascend,npu)=RAISE<RuntimeError: torch.device('npu') 不被注册（torch_npu 未 import）> (ascend,cpu)=torch_eager
A3-GATE after : (ascend,npu)=tilelang   <- 卡上 npu 在场时未桩也应为 tilelang
```

**桩后 `active_backend(ascend, npu) = tilelang` 成立**（与 Y/T 的 `_prod_setup` 三量桩同结论），
即卡上只要设备真在场，九件门面就都会去要内核。

编译清单（`keys` 档 = 45 次请求 / **12 唯一键**；`compile` 档真编 = **13 唯一键**，
差的那一枚是 `gdnb`：`keys` 档把编译短路了，所以反向伴随模具没出现）：

| # | 键（形状/块宽内建在键名里） | 请求次数 |
|---|---|---|
| 1 | `ascend|gemm[ascend]|n32k32b16x32x32anone` | ×21（linear/lora 复用同一路） |
| 2 | `ascend|gemm[ascend]|n32k96b16x32x32anone` | ×1 |
| 3 | `ascend|gemm[ascend]|n96k32b16x32x32anone` | ×2 |
| 4 | `ascend|gemm_dw[ascend]|n32k32tc16b32x32` | ×7 |
| 5 | `ascend|gemm_dw[ascend]|n96k32tc16b32x32` | ×1 |
| 6 | `ascend|add_ln[ascend]|d32e1e-05b16` | ×2 |
| 7 | `ascend|ln_bwd[ascend]|d32e1e-05b16` | ×1 |
| 8 | `ascend|rope[ascend]|h2d16s2g1` | ×2 |
| 9 | `ascend|attn_sw[ascend]|h2d16w16b16x16` | ×2 |
| 10 | `ascend|gdn[ascend]|h2k16v16|c8ubh` | ×3 |
| 11 | `ascend|gdnb[ascend]|h2k16v16|c2ubz` | ×1（compile 档才现身） |
| 12 | `ascend|gather[ascend]|d32` | ×2 |
| 13 | `ascend|scatter_add[ascend]|d32` | ×1 |

预算：**13 次真编译 / 合计 19.1s，整步墙钟 20.0s**（容器冷缓存，`compile` 档实测）：

```bash
docker exec cann910b-h bash -c 'cd /tmp/a3sys1/release; source /usr/local/Ascend/cann-8.5.0/set_env.sh; \
  export BISHENG_HOME=/usr/local/Ascend/cann-8.5.0/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201; \
  A3_ROOT=/tmp/a3sys1/release TILELANG_CACHE_DIR=$(mktemp -d) timeout 900 \
  python3 /root/A3_probe_trainstep_route.py compile'
A3-STEP wall=20.0s compile_count=13 real_compiles=13 sum_sec=19.1 missing_grad=[] err=None
```

接线面同时被证真：`P2-STEP-ascend loss0=0.910992 loss1=0.823897 reduced=True params_grad_ok=True`
（**注意口径**：这一趟编译产物被 no-op 桩掉了发射，数值来自 eager 兜底，
所以它证明的是"八门面拓扑 + 反向全参非 None + 请求 13 枚键"，**不证明内核数值**）。

**但 `card_run.sh` [4] 真跑时（无桩）在容器里死在第一行设备调用之前**：

```
=== [4] mixed-stack train step on npu (P2 首步) ===
RuntimeError: Expected one of cpu, cuda, …, privateuseone device string: npu
```

病灶定位：`train_step.py:213` `dev = torch.device(args.device or dev_default)`（`dev_default="npu"`），
**在 `try:`（line 218）之外** ⇒ 既没有 `P2-STEP-ascend FAIL` 判决行，也没有 traceback 之外的可读结论。
根因与 §6 的 H1 同源：`train_step.py` 全链路（`train_step → autograd_asc → *_kernel → ascend_env`）
**没有任何一处 `import torch_npu`**。

---

## 4 时钟预算与超时合理性 —— **够用，但不是瓶颈；瓶颈是 H1**

容器 = "有 bisheng、无设备"，即卡窗的**纯编译面**（bisheng 在 host CPU 上跑，与卡无关，
所以这些秒数可直接当卡窗编译预算的下界读数）。

整链冷跑（`card_run.sh` 全步，全新 `TILELANG_CACHE_DIR`）：

```
A3-WALL-CHAIN-SEC=36 EXIT=0        （[0]→[5] 全程，含 [3] 八腿 + delta + [4]）
整链冷编产物数=20                    （缓存目录里 executable.so 计数）
```

分腿（同一份全新缓存目录，逐腿 `date +%s` 实测，容器冷缓存）：

| 腿 | 卡窗 timeout | 容器实测（冷） | 编译件数 | 富余 |
|---|---|---|---|---|
| `linear_prod --track card` | 900s | **9s** | 4 键（`n64k32…anone` 等） | ≈100× |
| `dw_prod` | 900s | **6s** | 4 键 | ≈150× |
| `readout_prod` | 900s | **7s** | 6 键（gather×3 + scatter×3，按 dim∈{64,26,32}） | ≈130× |
| `gemm_l1_intrinsic` | 500s | **2s** | 1 键（cube 路） | 250× |
| `dw_intrinsic` | 500s | **3s** | 1 键 | 167× |
| `readout_intrinsic` | 500s | **3s** | 1 键 | 167× |
| `addln_intrinsic` | 500s | **2s** | 1 键 | 250× |
| `gdn` | 500s | **3s** | 2 键（conv_fwd + delta_fwd） | 167× |
| `run_delta_card` | 500s | **3s** | 1 键 | 167× |
| `[4] train_step`（真编 13 键） | 900s | **20s**（§3 实测） | 13 键 | 45× |

单编均值 ≈ **1.5–2.0s/件**（与 Y/T 报的 4–7s 同量级，本容器偏快；卡上 CPU 更强或更弱都不致超时）。

换算到卡窗（新增项只有"NPU 上下文初始化 + 真发射"）：
卡上 15 次子进程各付一次 `torch.npu` 初始化（经验 3–15s/进程）≈ **45–225s**，
加上编译面 ≈33 件（[3] 的 20 + [4] 的 13）≈ **56–80s**，加上真发射与对拍 <30s
⇒ **整链预计 2.5–6 分钟**；`[3]` 的 900/500s 与 `[4]` 的 900s **都够**。

**最可能超时/最先该担心的腿**（按风险排序，都不是"编译太慢"）：
1. `readout_prod` —— 卡上会真发射 5 组形状 × 每 shape 多次 gather/scatter（dup-heavy 到 4 次命中），
   是 [3] 里唯一"多件多轮"的腿；aicore 一旦挂（507015 族）会连坐整腿重试，最吃时间。
2. `[4] train_step` —— 唯一要编 13 枚模具、且反向模具（`ln_bwd`/`gdnb`/`gemm_dw`）
   首次真发射的腿；`params_grad_ok` 判据在真发射下才第一次真起来。
3. `gdn`（对照轨）—— 旗舰件，UB 分段 + 递推，历史上唯一走过 `asc_get_sub_block_id` 的腿。

**建议窗口时长**：**20 分钟**（净链 6 分钟 + 一次 `fixup_card.sh` 式补跑 6 分钟 + 冷启/clone/overlay
与人工读日志 8 分钟）。低于 15 分钟不给补跑余量；给到 30 分钟则纯浪费。

---

## 5 判据行可 grep —— **有隐患（H2/H3），只报不改**

编排者目标正则实测（容器整链日志 `/root/a3_cardrun_full.log`，39 行）：

```bash
docker exec cann910b-h bash -c 'grep -E "ASC-|Y-.*PROD|NUM-|P2-STEP|RUNNER-DONE" /root/a3_cardrun_full.log'
→ 命中 10 行（3×Y-*-ASC-COMPILE-PASS、6×NUM-*-FAIL、1×RUNNER-DONE）
```

漏网面（同一份日志、同一正则**抓不到**的行）：

```
torch 2.14.1+cu130 npu_avail False          <- 整窗唯一的设备在场判据！
OVERLAY-OK headers=5 tpl=12 / compat §12+fill: 7 / INJECT-OK     <- [2] 的自证三行
CASE-PROD-* / CASE-PERF-* / CASE-delta                            <- 腿标签（无判决时靠它对齐列）
RuntimeError: Expected one of … npu                               <- [4] 整腿的唯一输出
FATAL overlay-missing: …（负例实测会出，正常跑不出）
```

⇒ **隐患 H3（正则层面）**：目标正则要补
`OVERLAY-OK|INJECT-OK|^FATAL|CASE-|npu_avail|RuntimeError|Traceback`，
且 `[4]` 的判决 `P2-STEP` 在 H1 触发时**整行不存在**，必须靠 `RuntimeError` 兜。

⇒ **隐患 H2（`card_run.sh` 内部过滤器层面，实测）**：腿内
`grep -E "Y-$t|FAIL|Error:|error:|aicore" | head -6` 的 `aicore` 会命中 bisheng 命令行里的
`-cce-aicore-*`，每件失败件白吃 3 行（RuntimeWarning + Command + fatal error）。
单件失败（linear_prod）刚好容下判决行；**两件同灭（readout_prod）判决行被完全挤掉**：

```bash
# 实测（容器，故意删 c_api/asc_simd.h 后按 card_run.sh 原样过滤）
CASE readout_prod 的 6 行 = 2×[RuntimeWarning + Command + fatal error]，
  Y-readout_prod-PROD-FAIL **不在输出里**（对比 linear_prod 4 行，判决行在第 4 行，容得下）
```

⇒ 建议（由编排者定，A3 不动手）：腿内过滤去掉 `aicore`、或 `head` 提到 20、
或改成**两趟**：先 `grep -E "^(Y-|NUM-|P2-STEP|ASC-)"`，再 `grep -E "error:|FATAL"`。

⇒ **A2 新前缀会不会被吞**：`card_run.sh` 四条腿的过滤器只认 `Y-$t` / `NUM-$t` /
`NUM-delta_fwd` / `P2-STEP` 加 `FAIL|Error|error:|aicore`。A2 在 `tasks.md` 里留的判决形是
`A2-COMPILE-PASS`（同类还有 `D-DW-COMPILE-PASS`、`T-PROBE-ASC-COMPILE-PASS`、
`E2E-CUBE-ONLY PASS`、`T-GATE-ASC-AVAILABLE-PASS`）。**结论：只要 A2 的判决行从 [3]/[4] 腿内
打印，就会被腿内 grep 直接吞掉**（除非那行同时含 `FAIL`/`Error`）；
而 `ASC-` 这一族**恰恰不在任何腿的过滤器里**——`compile_all_asc.py` 打印的
`ASC-<op> PASS|FAIL|PRECOMPILE-ERR` 若被引入窗内（我建议引入，见 §6 增采 D），
必须给 [3] 或新增段补一条 `grep -E "^ASC-"`。
另注：`card_run.sh` 现**不跑** `compile_all_asc.py`，所以窗内不会出现任何 `ASC-` 行
（任务书把它列为卡窗必需件，与实际入口不一致 —— 待编排者裁决是否加段）。

⇒ **隐患 H4（退出码无意义）**：整链容器实测 `CARDRUN-EXIT=0`，
而 [3] 八腿 + [4] 全灭。`timeout`/腿退码都被管道吞掉 ⇒
**`RUNNER-DONE` 只代表"跑到底了"，不代表"过了"**，完成度只能靠判决行逐条核。

---

## 6 失败面预案 —— 最小增采清单（窗内一次拿全归因）

先给一条**能省掉一整轮归因的静态事实**（A3 已在容器实测，非猜测）：

> **产品件与 §12 不在同一条生成码路径上。**
> 冷编 20 枚产物按载体分族：`gemm_impl / dw_impl / gather_impl / scatter_impl / gdn_*`
> 全是 `__vector__` 标量体，搬运只发 `asc_copy_gm2ub_align`（**10 参 UB-DMA**）；
> 只有 `main_kernel`×3（即 `gemm_l1_intrinsic`/`dw_intrinsic` 的 cube 路）出现
> `asc_copy_gm2l1_nd2nz / asc_copy_l12l0a_transpose / asc_copy_l0c2gm / asc_set_gm2l1_nz_para`
> 这一族 §12 战场符号。
> ⇒ **`linear_prod`/`dw_prod` 数值不对，归因面是"UB-DMA 10 参位序 + 夹址 min()/行号守卫"，
> ⇒ 与 U2/U3（§12 的 L1/ND2NZ 位段）无关**；反过来 `*_intrinsic` 数值不对才轮到 §12。
> 这一条把窗内"prod 失败要不要往 §12 上追"的分叉提前砍掉一半。

取证命令（可复跑，容器/卡同一条）：

```bash
docker exec cann910b-h bash -c 'K=$(ls -dt /tmp/tmp.*/0.1.15 | head -1)/linux-aarch64/kernels; \
  for d in $K/*/; do n=$(grep -o "void [a-z_0-9]*_kernel" $d/device_kernel.asc|head -1|cut -d" " -f2); \
  q=$(grep -oE "__vector__|__aicore__" $d/device_kernel.asc|sort -u|tr "\n" " "); \
  c=$(grep -oE "asc_(copy|fill|set)[a-z0-9_]*" $d/device_kernel.asc|sort -u|tr "\n" " "); \
  echo "$n | $q | $c"; done'
# 单件直读（含锚点计数）：release/ascend/kernels/reconcile/A3_probe_gen_code.py
docker exec cann910b-h bash -c 'cd /tmp/a3sys1/release && A3_ROOT=$PWD TILELANG_CACHE_DIR=$(mktemp -d) \
  python3 /root/A3_probe_gen_code.py gemm --out /root/a3_gem.cc'
→ A3-GENCODE gemm key=gemm[ascend]|n64k32b16x64x32anone bytes=2851 nd2nz=0 fill=0 sidcast=0 …
```

### 最小增采清单（按性价比排序；除 G 外全部是"日志/产物"级，零额外编译）

| # | 增采项 | 命令/做法 | 为什么非它不可 |
|---|---|---|---|
| **A** | **torch_npu 在场探针（H1，最高优先）** | [0] 段加一行：`python3 -c "import torch,torch_npu;print('NPU-REG',torch.__version__,torch.npu.is_available(),torch.npu.device_count())" 2>&1 \| tail -1` | 整窗唯一能把"Y-*-ASC-COMPILE-PASS note=无设备"与"Y-*-PROD-PASS rel=…"区分开的先验。**不加也行，但必须让判读口径写明：`ASC-COMPILE-PASS` ≠ 数值通过** |
| **B** | **`note=无设备⇒只判编译` 作为专用 token 单抓** | 窗末 `grep -c "note=无设备" 日志` | 这个降级行**长得像 PASS**（含 `ASC-COMPILE-PASS`），必须一眼可见有几腿被降级 |
| **C** | **整份 `$TILELANG_CACHE_DIR` 带回** | 收尾：`tar czf /tmp/a3_kernels.tgz -C "$TILELANG_CACHE_DIR" . && cat /tmp/a3_kernels.tgz > /dev/null`（scp 回本地即可） | 每键目录自带 `device_kernel.asc` + `host_kernel.c` + `manifest.json` + `params.json`——**§12 位序、T.Cube 是否在场、10 参 UB-DMA 实参**全在里面，**零额外编译**即可事后归因。这是本清单回报/成本比最高的一项 |
| **D** | **九件产品接口编译面** | [3] 前或后加一段：`timeout 600 env DMLAYA_ASCEND_TARGET=ascend python3 ascend/compile_all_asc.py --target ascend`（判据行 `ASC-<op> PASS/FAIL/PRECOMPILE-ERR`，需配 `grep -E "^ASC-"`） | prod 腿只给 rel，不给"哪件根本没达到编译器"；`compile_all_asc` 的 `PRECOMPILE-ERR` 能一眼分开"plan 拒形/入口回退"与"bisheng 拒绝"。（注意：任务书已把它列为卡窗必需件，但 `card_run.sh` 现在**不调它**） |
| **E** | **intrinsic ↔ prod 同判据成对打印（口径已对齐）** | 失败腿后紧跟：`python3 run_numerics.py --only <对应 *_intrinsic> --no-run` | ⚠ 二者**不同形**（intrinsic 是 64³ 常量形状、prod 是 plan 出的分块），所以这条只能证"同一判据的另一载体也红/也绿"，不能当同形对照。真正的同形对照由 C 的生成码承担 |
| **F** | **E-3 sid 判别打印** | 生成码里 grep `asc_copy_gm2l1_nd2nz`/`asc_set_gm2l1_nz_para` 的**实参原文**（C 已含）+ compat 头侧 `(uint8_t)sid` **应为 0**（A3 实测 0） | sid 回穿已被 Z 清掉；卡上唯一要验的是"新 9 参位序进了产物没有"，靠 C 的 `.asc` 原文，不靠再编一遍 |
| **G** | **失败腿的最小重跑配方（不试错，R20）** | 单腿重跑：`TILELANG_CACHE_DIR=$(mktemp -d) DMLAYA_ASCEND_TARGET=ascend python3 run_numerics.py --only <case> --track card 2>&1 \| grep -E "^(Y-|num)|error:|FATAL|RuntimeWarning"`（去掉 `head`） | 修 H2 之后才有意义：保证判决行一定在输出里 |
| **H** | **[4] 的"设备未注册"专用兜底行** | `timeout 120 env DMLAYA_ASCEND_TARGET=ascend python3 ascend/train_step.py --target ascend --device cpu 2>&1 \| grep -E "P2-STEP\|WARN"` 作对照 | `--device cpu` 会打出 `WARN: target=ascend 但 device=cpu`（ascend_env 判 eager）⇒ 一条命令切分"接线错"与"npu 没注册"，把 H1 从"归因成本"降为"零成本" |

### H1 的完整陈述（开窗前唯一真阻塞）

- **事实**：`train_step.py` / `autograd_asc.py` / `*_kernel.py` / `*_asc.py` / `ascend_env.py`
  / `run_numerics.py` **全链无 `import torch_npu`**；只有 `run_delta_card.py:51`、
  `run_kernel.py:11`、`selfcheck.py` 导了它。
- **后果（容器整链实测，四种形态，全都不需要卡就能复现）**：
  1. `train_step.py --target ascend` → `torch.device("npu")` **RuntimeError**（line 213，
     在 `try:` 之外）⇒ [4] 没有 `P2-STEP` 判决行；
  2. `run_numerics --only <prod> --track card`（**卡上也会这样，如果 npu 未注册**）
     → `Y-<case>-ASC-COMPILE-PASS keys=1 … note=无设备⇒只判编译` ⇒ **数值判决为零，但看着像过**；
  3. `run_numerics --only <intrinsic>` → `NUM-<case>-FAIL RuntimeError: Expected one of cpu,…npu`；
  4. `run_delta_card.py` → `NUM-delta_fwd FAIL run: ModuleNotFoundError: No module named 'torch_npu'`
     （这一行是整窗**唯一直接点名根因**的输出，可惜排在 [3] 最后）。
- **是否真会在卡上发生**：未知。若卡的 torch 是昇腾打过补丁的构建（`import torch` 即注册 npu），
  则无事；否则整窗按上面四种形态作废。`git archive`+容器无法替卡回答这一问。
- **A3 建议（任选一，都不动产品件）**：
  (a) [0] 段加增采 A 的探针行（1 行，成本 2s，**先于任何腿**）；或
  (b) 把 [3]/[4] 的调用改成
  `python3 -c "import torch_npu; import runpy,sys; sys.argv=[...]; runpy.run_path('run_numerics.py', run_name='__main__')"`
  这种"先注册再跑"的外壳；或
  (c) 在 `$SYS1/release` 放一份 `sitecustomize.py`（clone 后由 [2] 写入，解释器自动 import）
  ⇒ 三者都能把 H1 从"赌卡"变成"确定性"。**A3 未动任何一处，等编排者派单。**

---

## 附：本波产物与复跑入口

- 探针（只读，新建于 reconcile 目录）：
  `A3_probe_clone_deps.py`（入库闭包）、`A3_probe_trainstep_route.py`（门 + 13 键 + 编译耗时）、
  `A3_probe_gen_code.py`（生成码锚点/落盘）。
- 容器侧临时件（非仓内，不需清理）：`cann910b-h:/tmp/a3sys1`（origin/main 快照）、
  `/root/a3_cardrun_full.log`、`/root/a3_cold.log`、`/root/a3_ts.log`、`/root/a3_prod*.log`。
- 整链复跑（无卡，36s）：
  `docker exec cann910b-h bash -c 'SYS1_DIR=/tmp/a3sys1 CARDRUN_SKIP_CLONE=1 bash /tmp/a3sys1/release/ascend/port910b/card_run.sh'`
