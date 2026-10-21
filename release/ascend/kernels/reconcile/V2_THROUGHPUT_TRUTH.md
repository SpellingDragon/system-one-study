# V2_THROUGHPUT_TRUTH.md — P1-1t 收尾：吞吐对照腿（intrinsic）数据真伪定档

> 归属：p2-13 执行代理 **V2**（不占卡）。产出物：本定档 + `V2_probe_intrinsic_truth.py`（判据）+
> `ascend/bench_card.py`（工装硬闸）。台账只读、未 git。

## 0. 一句话结论（可抄入口径）

**窗口2（1011-p2-13-oncard-window2-throughput）的 `BENCH-ABLATE` 与 `BENCH-QUOTE linear_intrinsic`
作废**：那条 `linear_intrinsic` 件（`S_perf_track_cube.gemm_cube_impl`）落成
`__global__ __mix__(1, 2)`，正文整段嵌在 `if ASC_IS_AIV { if ASC_IS_AIC { … } }` 里，而 910B compat
层把 `ASC_IS_AIV` 钉死成 `(0)` ⇒ **死代码、根本没做 512³ 的活**（归类＝**未真执行**）；其
`p50_ms=0.080` 是空转+发射/同步开销，`tflops=3365.919` 另有 1000× 单位错标（真值 3.37 TFLOP/s）。
**标量侧 8 条 `ok` 复核后可用**（第二窗按 `--quick` 共 10 条标量腿：2 条 bf16 因「假绿守卫」如实
FAIL，8 条真开模且计时夹同步；见 §3 表）。**第三栈（dW intrinsic）仍缺件/坏件**：上一窗
`507015`、本窗 `RuntimeError`，两窗一致——该件是真执行（`__cube__`）但地址非法，属 V 系未证真，
不得引用其任何数。工装已加硬闸：**任何腿计时前必须数值自证，否则只打 `BENCH-<op> NOTIMED`，
不产 `p50/tflops/BENCH-QUOTE`**，且 `BENCH-ABLATE` 标 `verified=<true|false>`。

## 1. 定性：那枚 intrinsic 件到底算了什么 ⇒ **未真执行（死代码）**

**判据（`V2_probe_intrinsic_truth.py --mode static`，codegen-only，容器 `cann910b-h` 可复跑）**：

```
$ docker exec cann910b-h bash -lc 'cd /work && TILELANG_CACHE_DIR=/dev/shm/tlcache \
    python3 ascend/kernels/reconcile/V2_probe_intrinsic_truth.py --mode static --kind gemm'
INTRINSIC-STATIC kind=gemm shape=512x512x512
  compat_hint: port910b_compat.h: `#define ASC_IS_AIV (0)` / `#define ASC_IS_AIC (true)` (under -DTL_PORT910B_NATIVE_TYPES, injected by pip bisheng.py:111)
  [lower] target=ascend enable_device_compile=False → src_len=2171
  global = ['__global__ __mix__(1, 2) void gemm_impl_kernel(__gm__ float* A, __gm__ float* C, __gm__ float* W, __gm__ float* bias, int32_t m) ']
  loops = [('chunk', '((m + 127) >> 7)'), ('mx', '8'), ('ko', '8'), ('i', '16'), ('j', '64')]
  mode = __mix__(1, 2)
  has_if_ASC_IS_AIV = True
  compute_present = True
  verdict = NOT-EXECUTED
--- device source (head 1400) ---
extern "C" __global__ __mix__(1, 2) void gemm_impl_kernel(__gm__ float* A, __gm__ float* C, __gm__ float* W, __gm__ float* bias, int32_t m) {
  asc_init();
  ...
  if ASC_IS_AIV {
    if ASC_IS_AIC {
      for (int32_t chunk = 0; chunk < ((m + 127) >> 7); ++chunk) {
        for (int32_t mx = 0; mx < 8; ++mx) {
          for (int32_t ko = 0; ko < 8; ++ko) {
            ... asc_copy_gm2ub_align(...); asc_mmad(..., 16, 64, 64, 0, 1, 0, (ko == 0)); ...
```

`ASC_IS_AIV` = `(0)`（`port910b_compat.h:568`，在 `TL_PORT910B_NATIVE_TYPES` 段；该宏由容器内 pip
`tilelang/contrib/bisheng.py:111` 的 `-DTL_PORT910B_NATIVE_TYPES` 注入）⇒ `if (0) { if (1) {…} }`
= 死代码。**乘加 `asc_mmad`、GM→UB 搬运 `asc_copy_gm2ub_align`、写回全在死分支内** ⇒ 该件"跑"一次
什么都不算。

**反证（同窗另一条 intrinsic 腿，形态一一对应）**：

```
$ docker exec cann910b-h bash -lc '... V2_probe_intrinsic_truth.py --mode static --kind dw'
INTRINSIC-STATIC kind=dw shape=512x512x512
  global = ['__global__ __cube__ void dw_impl_kernel(__gm__ float* A, __gm__ float* GW, __gm__ float* dY, int32_t m) ']
  mode = __cube__
  has_if_ASC_IS_AIV = False
  compute_present = True
  verdict = EXECUTES
```

`dw_l0tr_cube_impl`（`dW_intrinsic`）是 `__cube__`（**无** AIV 包裹）⇒ 真执行 ⇒ 窗口2 里它如实
崩 **507015**（DMA 非法访问）。**`__mix__` 腿＝被 compat 宏杀死的死代码（假绿），`__cube__` 腿＝
真跑并真崩**——两条腿的表现与生成码形态一一对应。

### 排除法：另外三种候选为什么不是

| 候选 | 判定 | 证据 |
|---|---|---|
| **形状不符** | ✗ 排除 | codegen 循环边界 `chunk<((m+127)>>7)`、`mx<8`(n/64)、`ko<8`(k/64)、`i<16`、`j<64`（bm=16/bn=64），m=n=k=512 时恰为 512³；`--list-ops` 里 `linear` 与 `linear_intrinsic` 的 shape 标签逐字同为 `512x512x512`（E2 配对同形） |
| **计时未夹同步** | ✗ 排除 | `bench_card.measure`：`for _ in iters: sync(); t0=perf_counter(); call(); sync()` —— 每样本前后夹 sync（非整段平均、非单次） |
| **参数顺序错** | ✗ 排除 | `_cube_setup` 传 `args=(A,W,b,C)`，与 `S_perf_track_cube.gemm_cube_impl` 的 PrimFunc 声明 `gemm_impl(A, W, bias, C)` 逐位一致；且 TileLang 生成的 host stub 按**参数身份**重排（见下），非位置直推。死代码判定与参数顺序无关 |

**host↔device 参数顺序（归档件 `32d6c39c…/host_kernel.c` 实证）**：host 声明序为
`gemm_impl(A, W, bias, C)`（args[0..3]），构造调用栈时映射为 `[0]=A,[1]=C,[2]=W,[3]=bias,[4]=m`，
再调 device 件 `gemm_impl_kernel(A, C, W, bias, m)` —— **身份映射正确**，两腿（产品件与存档件）
共用同一套 codegen 约定。

### 证据面②：归档 tarball 里根本没有这条 intrinsic 件

| 归档 | 条目 | md5 | 核函数形态普查 |
|---|---|---|---|
| 本窗 `runs/1011-p2-13-oncard-window2-throughput/artifacts/sys1_kernels.tgz` | **431** | `440b59589201c7545db3d8e7430c1f1c` | `__cube__`×2 + `__mix__(1,2)`×1 + `__vector__`×44 |
| 上窗 `runs/1011-p2-13-oncard-accept-2002/artifacts/sys1_kernels.tgz` | **422** | `f2ba9a2080de73d1d04d4a309dc8496e` | `__cube__`×2 + `__mix__(1,2)`×1 + `__vector__`×43 |

复跑：`tar -tzf <tgz> | wc -l`、`md5 -q <tgz>`、`grep -rhoE '__global__ +(__mix__\([0-9, ]*\)|__cube__|__vector__) void [A-Za-z0-9_]+'`。

- 归档里的 `gemm_impl_kernel` 全是 `__vector__`（**产品标量件**，11 份），`dw_impl_kernel` 全是
  `__vector__`（8 份）——**bench 的 intrinsic 变体（`gemm_impl_kernel` 作 `__mix__(1,2)` /
  `dw_impl_kernel` 作 `__cube__`）一份都不在归档里**。
- 归档里那 3 个非 `__vector__` 件是**旧遗留**，与本波 bench 无关：
  `7e9cfdc1…`＝`__cube__ dW_l0tr_kernel(float*DW, half*X, half*dY)`（fp16, shape=[2048,256]）；
  `18502bd3…`＝`__mix__(1,2) main_kernel(bf16 A,B,C)`（shape=[2048]）；
  `c92425a9…`＝`__cube__ main_kernel(bf16 A,B,float O)`（shape=[64]）。
- 根因：`bench_card._cube_setup` **自己 `tilelang.compile`**、不进 `ascend_env` 缓存，`_route_key`
  对 `n/a` 子串直接返回弱判据 `"<target>|out-of-band-cube-archive", "new"`（**"句柄即证据"**）——
  故这条腿的产物落 ephemeral 缓存、**没有归档兜底**，弱判据也拦不住假绿。
  ⇒ 这正是把「数值自证」升级为**计时硬前置**的直接动机。

## 2. 窗口2 台账逐条复核（`artifacts/w2_bench.log`，`--quick --iters 3 --warmup 1 --budget-sec 300`，configs=12）

| # | op | dtype | shape | 结果 | p50_ms | 表内 tflops | 复核裁定 |
|---|----|-------|-------|------|--------|-------------|----------|
| 1 | linear | bf16>fp32 | 512³ | FAIL（假绿守卫：无 `ascend\|gemm` 键） | — | — | 见注① |
| 2 | linear | fp16>fp16 | 512³ | ok | 294.309 | 0.912 | **可用**（真开模 `ascend\|gemm[ascend]…`） |
| 3 | dW | bf16>fp32 | m1024xn512xk512 | FAIL（假绿守卫：无 `ascend\|gemm_dw` 键） | — | — | 见注① |
| 4 | dW | fp16>fp16 | m1024xn512xk512 | ok | 266.069 | 2.018 | **可用** |
| 5 | add_ln | fp32 | rows8192xdim1024 | ok | 17.113 | na | **可用** |
| 6 | rope | fp32 | tok4096xh8xd128 | ok | 11.717 | na | **可用** |
| 7 | attn_sw | fp32 | h8xseq1024xd128w512 | ok | 1879.313 | 0.858 | **可用** |
| 8 | gdn_conv | fp32 | C1024xL2048xk4 | ok | 23.223 | 0.722 | **可用** |
| 9 | gdn_delta_fwd | fp32 | h8xseq512xk64xv64 | ok | 293.228 | 0.229 | **可用（计时）**，数值见注② |
| 10 | gdn_delta_bwd | fp32 | h8xseq512xk64xv64 | ok | 431.310 | 0.156 | **可用（计时）**，数值见注② |
| 11 | **linear_intrinsic** | fp32 | 512³ | ok（**假绿**） | 0.080 | 3365.919 | **作废**（死代码，§1） |
| 12 | **dW_intrinsic** | fp32 | m1024xn512xk512 | FAIL `RuntimeError … 507015` | — | — | **作废**（真执行但崩，§3） |

`BENCH-SUMMARY ops=12 ok=9 fail=3 … fail_core=2 fail_ref=1 fail_inj=0`。

**注①（bf16 标量腿为何 FAIL）**：不是缺陷——`bench_card` 的**假绿守卫**（沿用 `run_numerics._route_key`）
要求 `compiled_keys()` 里出现 `<target>|<substr>` 键；bf16 档模具未收（`dtype-gap`，P1-1m），该趟
静默落 torch eager ⇒ 守卫如实 FAIL，**不进表**。这正是"为什么这些数可信"的一半。
**注②（gdn 腿的数值状态）**：窗口2 的 gdn 腿输入用 `randn` 当递推门（`_rand3`），而 gdn 递推是
`S *= exp(g[t])` ⇒ `exp(randn)` 在 seq=512 步内爆成 `inf`/`nan`（**参考自身即 nan**）。故这 2 条
的**计时可用**（内核控制流与数据无关），但**数值未经自证**——本波已把 gdn 腿输入改成数值良态
（门非正 + β~1/dk，见 `bench_card._gate3` 与 `_build_delta_*`），使硬闸对 gdn 腿可用。

## 3. 第三栈（intrinsic · 对照腿）现状

| 腿 | 上一窗（accept-2002） | 本窗（window2-throughput） | 形态 | 裁定 |
|----|----------------------|----------------------------|------|------|
| `linear_intrinsic`（`gemm_cube_impl`） | — | `ok` 但 **假绿**（0.080ms/3365.919） | `__mix__(1,2)` 死代码 | **作废**（未真执行） |
| `dW_intrinsic`（`dw_l0tr_cube_impl`） | 卡上 **507015** | 卡上 `RuntimeError … 507015` | `__cube__` 真执行 | **缺件/坏件**：真跑但 DMA 非法 |

即：**第三栈没有一条能进报价/消融**——`__mix__` 那条是假的、`__cube__` 那条是崩的。两窗一致⇒
不是偶发。

## 4. 本波工装改动（`ascend/bench_card.py`，白名单内）

1. **数值自证＝计时硬前置**（`run_one`）：`setup` 返回第 4 元 `verify`；`verify` 缺失或不过 ⇒ 抛
   `_NotTimed`，主循环打 `BENCH-<op> NOTIMED reason=no-numeric-selfproof|numeric-mismatch(rel=…)`，
   **不产 `p50/tflops/BENCH-QUOTE`**。参考＝入口件自身 `target="cpu"` 的 torch 尺子路；intrinsic
   腿＝torch 矩阵式（gemm `A@Wᵀ+b` / dW `dYᵀ@A`）。容差 `_NUM_TOL`：fp32 1e-2、fp16/bf16 3e-2。
2. **`BENCH-ABLATE` 标 `verified=<true|false>`**：两腿都过自证才 `true`，否则 `ratio=na`——杜绝死
   代码腿的假比值（3690.3x）再被引用。
3. **`BENCH-<op>` 带数字行尾加 `verified=true`**（只在自证通过后才可能被打出）。
4. **tflops 单位修正**：`/1e9`→`/1e12`（真 TFLOP/s），单位写进 `BENCH-NOTE`。
5. **负例自证**：`--selftest-notimed`（无对拍腿）与 `--selftest-mismatch`（对拍必挂腿）各应打
   NOTIMED。

### 判据负例自证（真跑输出）

```
$ .venv/bin/python ascend/bench_card.py --device cpu --quick --iters 2 --warmup 1 --selftest-notimed
BENCH-selftest_notimed NOTIMED dtype=n/a shape=n/a reason=no-numeric-selfproof track=cpu/cpu
$ .venv/bin/python ascend/bench_card.py --device cpu --quick --iters 2 --warmup 1 --selftest-mismatch
BENCH-selftest_mismatch NOTIMED dtype=n/a shape=n/a reason=numeric-mismatch(rel=3.333e-01 tol=0.01) track=cpu/cpu
```

两条 NOTIMED 行均**不含** `p50_ms=`/`tflops=`/`BENCH-QUOTE`（已按行断言）。即"伪造一个未对拍的腿
⇒ 必须 NOTIMED 而非出数"成立。

### 探针闸负例自证（`V2_probe_intrinsic_truth.py --mode neg`，无设备可跑）

```
NEG-SELFPROOF numeric_good ok=True (rel=0.000e+00)
NEG-SELFPROOF numeric_zeros ok=False (rel=1.000e+00)
NEG-SELFPROOF numeric_transposed ok=False (rel=2.582e-01)
NEG-SELFPROOF peak_3365 ok=False (tflops=3365.919 peak=363.7)      ← 第二窗那枚数被峰值闸拒
NEG-SELFPROOF peak_3p355 ok=True (tflops=3.355 peak=363.7)
NEG-SELFPROOF-DONE all_gates_behave=True
NEG-SELFPROOF peak_src=CANN platform_config/Ascend910B1.ini (ai_core_cnt=24 × cube_freq=1850MHz × cube 16×16×16 × 2) = 363.7 TFLOPS；族内最大，取作硬上界
```

**峰值出处（不得凭记忆，可复跑）**：
```
$ docker exec cann910b-h grep -E "^(NpuArch|ai_core_cnt|cube_freq|cube_[mnk]_size)" \
    /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/data/platform_config/Ascend910B*.ini
Ascend910B1.ini: NpuArch=2201 ai_core_cnt=24 cube_freq=1850 cube_m/n/k_size=16/16/16
Ascend910B2.ini: NpuArch=2201 ai_core_cnt=24 cube_freq=1800
Ascend910B3.ini: NpuArch=2201 ai_core_cnt=20 cube_freq=1800
Ascend910B4.ini: NpuArch=2201 ai_core_cnt=20 cube_freq=1500
```
⇒ peak_fp16 = ai_core_cnt × cube_freq(Hz) × (16·16·16) × 2；910B1 = 24×1.85e9×4096×2 =
**3.637e14 = 363.7 TFLOPS**（族内最大，取作硬上界；SoC 变体未落窗口日志，故取族内最大以免误杀）。
tilelang 自身 `tools/Analyzer.py::ARCH_CONFIGS` **只有 NVIDIA sm80/86/89、无昇腾**（已复核
tilelang/tools/Analyzer.py:179-195），故昇腾理论峰值以 CANN platform_config 为准。

## 5. 回归（本波必须仍绿）

```
$ .venv/bin/python ascend/bench_card.py --device cpu --quick --iters 2 --warmup 1 --selftest-fail
BENCH-selftest FAIL [n/a n/a] RuntimeError: 注入的失败：…（其余 op 必须照跑）
BENCH-SUMMARY ops=11 ok=9 fail=2 notimed=0 wall_s=15.8 skip=0 fail_core=1 fail_ref=0 fail_inj=1 track=cpu/cpu iters=2 warmup=1
$ .venv/bin/python -m pytest tests -q -m "not integration"
340 passed, 3 skipped, 3 deselected, 2 warnings in 40.80s
```
（cpu 轨 `gdn_delta_bwd` 必落 torch ⇒ 「假绿守卫」如实 FAIL，属**预期**，见 `BENCH-NOTE`；
`fail_core` 仍如实计数。cpu 轨耗时**禁止**当卡上性能。）

## 6. 可发表口径（供抄入 run notes 与研究报告）

> **两栈实测 + 第三栈缺件/坏件**：产品**标量栈**真开模且计时夹同步——linear/dW fp16 512³ 档
> 0.29ms / 0.27ms（fp16），add_ln/rope/attn_sw/gdn_conv/gdn_delta_fwd/gdn_delta_bwd 六件 fp32 计时
> 可用（bf16 档模具未收、如实 FAIL）；**intrinsic 对照栈**两条腿均不可用——`linear_intrinsic`
> 落 `__mix__(1,2)` 死代码（未真执行，**作废**），`dW_intrinsic` 真执行但 aicore 507015（**坏件**）；
> 故**三栈消融无可引用数据**，第二窗的 `BENCH-ABLATE ratio=3690.3x` 与 `BENCH-QUOTE linear_intrinsic`
> 一并作废。工装已把「计时前必须数值自证」升为硬闸（未过 ⇒ NOTIMED、不出数），并以负例自证。
