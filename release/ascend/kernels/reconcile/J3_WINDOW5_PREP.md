# J3 窗五制备报告（tilelang-ascend 卡上判决件 · B 路线）

- 角色：p2-13 执行代理 J3；**不占卡、不改本仓既有产品件**；禁 git。
- 目标：把 H2 在容器 `cann910b-l` 已构建好的 `refs/tilelang-ascend`（`ascendc_pto`）**可运行面**打包，
  上传到卡（CANN **8.5.2** / aarch64 / torch_npu 2.7.1.post2）**免编真跑**，产三件判决：
  ① GDN 六件（chunk 前向族）**数值**对拍；② **性能** p50/p95；③ **gemm** 同采（自带 1024³ + 512³ fp16/fp32）。
- 前置：H2 报告 `release/ascend/kernels/reconcile/H2_TILELANG_ASCEND_EVAL.md`（构建配方 + 环境坑）。

> 纪律：本报告所有**性能行只标 `track=ta-ascendc`，不出结论**（结论归编排者）。

---

## 1. 交付物（均在白名单内）

| 交付物 | 路径 | 状态 |
|---|---|---|
| 可迁移包 + 校验清单 | `refs/ta_pkg/ta_pkg.tar.gz`、`refs/ta_pkg/MANIFEST.txt`（gitignored） | ✅ |
| 卡侧脚本 | `release/ascend/port910b/window5_ta_card.sh`（新，内联驱动，自足） | ✅ |
| 宿主发射器 | `release/ascend/port910b/run_window5_local.sh`（新） | ✅ |
| 探针/驱动 dev 副本 | `refs/j3_probe_ta_compile.py`、`refs/j3_ta_run.py`（gitignored） | ✅ |
| 本报告 | `release/ascend/kernels/reconcile/J3_WINDOW5_PREP.md` | ✅ |

---

## 2. ① 可迁移包 `ta_pkg`

- **源**：`refs/tilelang-ascend`（@83b0ece, `ascendc_pto`），H2 配方 `install_ascend.sh` + `set(USE_ALTERNATIVE_LINKER OFF)`，`make -j2`。
  容器 `cann910b-l`：CANN 8.5.0 / aarch64 / Python 3.10.12；宿主 `refs/tilelang-ascend` → 容器 `/tlasc`。
- **运行面**（解包即用所需，缺一编不动）：
  - `tilelang/`（python 包，去 `__pycache__`）+ `src/`（`TILELANG_TEMPLATE_PATH`，含 `tl_templates/ascend`）
  - `build/{libtilelang.so, libtilelang_module.so, tvm/{libtvm.so, libtvm_runtime.so}}`（H2 所谓"三个 .so" + runtime）
  - `3rdparty/tvm/python/tvm/`（tvm python 包）、`3rdparty/{catlass/include, shmem/{include,src/device}}`（bisheng 编译头）
  - `examples/linear_attention_and_rnn/{gdn/*.py}`、`examples/gemm/example_gemm.py`、`VERSION`
  - `DEPS.txt`（运行时依赖清单：core 硬校验 / optional best-effort；源自 `refs/tilelang-ascend/requirements.txt` 裁剪）
- **体积 / 取舍**：
  - `ta_pkg.tar.gz` = **36,478,146 B (34.8 MiB)**；解包 **117,591,141 B (112 MiB)** / **1996 files**（< 300MB，无需额外取舍说明）。
  - **有意排除**：`build/**/CMakeFiles`、`*.a`（构建中间物，416M）、`3rdparty/pto-isa`（34M，PTO 路本工具链编不过、卡上不用）、examples 其余。
- **校验和清单**（`refs/ta_pkg/MANIFEST.txt`）：
  - `ta_pkg.tar.gz` sha256 = `d5607ec6f9c321c9652fa7d25f234d2e17a074cf308938fa2c4b522830b02172`（宿主 = 容器，已交叉核对）
  - `libtilelang.so` `7d00c31f…` / `libtilelang_module.so` `1df15390…` / `libtvm.so` `068365f7…` / `libtvm_runtime.so` `bbde9824…`
- **"解包即用"验证**（换前缀 ⇒ 非原仓路径）：`refs/j3_probe_ta_compile.py` 在**新前缀** `/tmp/tapkg_test/ta_pkg` 与 `/tmp/ta_unpack/ta_pkg` 各跑一次，均
  **`J3-UNPACK-PROBE six=6/6 gemm=1 fail=0`**（bisheng_calls=7 rcs 全 0）⇒ 解包到任意前缀即可编，非假绿（探针强制校验 `tilelang.__file__` 以 `$TA_ROOT` 开头）。

---

## 3. ② 卡侧脚本 `window5_ta_card.sh`

分段（每段打印 `=== [N] …` 段标记，供宿主按段判进度）：

- **[0] 解包 + env + CANN/ABI 快检**
  - 核对 tar `bytes + sha256`（≠ EXPECT_SHA 即 `TA-ABORT reason=tar-sha`）；`tar xzf` 到 `/tmp/ta_unpack/ta_pkg`。
  - CANN env：优先 `cann-8.5.2/set_env.sh`，回退 `ascend-toolkit/set_env.sh`；`source` 期间临时 `set +u`（set_env.sh 引用未定义变量）。
  - **条件 shim**：卡上有 torch_npu ⇒ **去 shim**；驱动内 `if not hasattr(torch,"npu")` 才装最小 shim（H2 配方）。卡上不会进 shim 分支。
  - 依赖：`CORE`（Cython/cloudpickle/ml_dtypes/psutil/tqdm/typing_extensions/numpy/torch）**硬校验**；
    `OPT`（attrs/decorator/scipy/tornado/yaml/pybind11）best-effort，缺只 WARN（本路径实测不吃）。
  - `PYTHONPATH=$TA_ROOT:$TA_ROOT/3rdparty/tvm/python`；`LD_LIBRARY_PATH` 兜底 `build/tvm:build:`（`libtilelang.so` RUNPATH 硬编码 `/tlasc/build/tvm`，卡上必须靠 env）。
  - 内联驱动 → `/tmp/j3_ta_run.py`（heredoc，单文件自足）；**CANN/ABI 快检＝单件 cumsum 编译（≤300s）**，见 §9。
- **[1] GDN 六件真跑对拍 + 性能；[2] gemm 同采；[3] 汇总/内核自证**：`timeout 2400 python3 $DRIVER`，产全部 `TA-*` 判决行。
- **[4] 现场日志打包** `tar czf /tmp/ta_w5_logs.tgz`，末行 `TA-DONE`。

**判决行契约**：
```
TA-GDN-<name> PASS rel=<v> ref=torch-fp32acc p50_ms=<x> p95_ms=<y> dtype=fp16 iters=20 warmup=3 src_bytes=<b> absmax=<a> upstream1e-5=<ok|over> track=ta-ascendc
TA-GDN-<name> SKIP reason=no-device compile=ok src_bytes=<b>          # 无设备形态出口
TA-GEMM dtype=<dt> shape=<m>x<n>x<k> blk=<bm>/<bn>/<kl1> PASS rel=<v> ref=torch p50_ms=<x> p95_ms=<y> tol=<t> track=ta-ascendc
TA-KERNELPROOF bisheng_calls=<n> cache_so=<m> rc_all_zero=<b> judge='…' verdict=<real-kernel|eager-suspect|compile-error>
TA-SUMMARY six=<n>/6 gemm=<n> fail=<k> skip=<j> mode=<full|compile-only> iters=20 warmup=3 rel_tol=0.002
```

**对拍阈值依据**：GDN 参考侧＝**tilelang-ascend 自带 `ref_*`**（内部 fp32 累加、末转 fp16，跑在 npu）；判据 `rel<2e-3`（**沿用本仓既有 fp16 口径**，peak-归一化 `max|Δ|/max|ref|`）+ 额外记录**上游自测契约 `assert_close(rtol=atol=1e-5)` 是否满足**（`upstream1e-5=ok|over`）——两个口径并列，不单独下结论。
**计时口径**：沿 E2 纪律——输入构造在计时区外，`warmup=3` 不计，`iters=20`，每次迭代前后 `torch.npu.synchronize()` 夹计，报 p50/p95（线性插值分位）。

**真走内核自证（判法）**：驱动包裹 `subprocess.run` 计数 `bisheng` 调用 `bisheng_calls`，并 `glob(TILELANG_CACHE_DIR/**/kernel_lib.so)` 计 `cache_so`；
判据 **`bisheng_calls>0 且 cache_so>0 ⇒ 走 JIT 内核（非 eager）`**（`verdict=real-kernel`）；全 0 ⇒ `eager-suspect`。

---

## 4. ③ 宿主发射器 `run_window5_local.sh`

- 上传 **2 件**：`window5_ta_card.sh`(≈21 KB) + `refs/ta_pkg/ta_pkg.tar.gz`(36.5 MB)；
  上传后核对 `REMOTE_BYTES` + `sha256` 前 16 位，不符即 `LOCAL-ABORT`（**不烧卡时**）。
- 上传时间估算（写入脚本头）：36.5 MB ⇒ SSH ≥1 MB/s **≤40s**；≥0.3 MB/s **≤2min**。
- 起跑：`setsid bash /tmp/window5_ta_card.sh > /tmp/ta_all.log 2>&1 &`，先探 `python3 / npu-smi / torch_npu`。
- **监控纪律（吸取窗二教训，不单看外层）**：45s/轮，同时取 outer `ta_all.log` 的 `=== [N] ` 段标记 +
  inner `ta_smoke.log`/`ta_run.log` **行数**；内层行数连续 10 轮（7.5min）不增且无 python3 ⇒ `STALLED` 如实停报。
- 回读 `TA-*` 判决行；best-effort 拉回 `/tmp/ta_w5_logs.tgz` → `refs/ta_pkg/w5_logs_remote.tgz`。

---

## 5. ④ 自证记录（不占卡）

**(a) 语法**：`bash -n window5_ta_card.sh` / `bash -n run_window5_local.sh` → **均 OK**。
内联驱动与 `refs/j3_ta_run.py` **逐字节一致**（`diff` 通过，杜绝 heredoc 走样）。

**(b) 容器"无设备形态"逐条跑（负例：必须显式 SKIP，不得静默假绿）**：

- 驱动（`TA_ROOT=/tmp/ta_unpack/ta_pkg`，容器无 npu）：
  ```
  TA-DEVICE torch=2.14.1+cu130 torch_npu=False npu_available=False
  TA-MODE compile-only
  TA-GDN-cumsum   SKIP reason=no-device compile=ok src_bytes=2595
  …（六件同形：kkt/solve_tril/wy_fast/chunk_h/chunk_o 全 SKIP reason=no-device）
  TA-GEMM dtype=float16 shape=1024x1024x1024 blk=128/256/64 SKIP reason=no-device
  TA-GEMM dtype=float16 shape=512x512x512  blk=128/256/64 SKIP reason=no-device
  TA-GEMM dtype=float32 shape=512x512x512  blk=64/128/64  SKIP reason=no-device
  TA-KERNELPROOF bisheng_calls=8 cache_so=9 rc_all_zero=True verdict=real-kernel
  TA-SUMMARY six=0/6 gemm=0 fail=0 skip=9 mode=compile-only
  → exit 4（无设备哨兵）
  ```
- 卡脚本**默认（严格）**：`bash window5_ta_card.sh` → 快检检出无设备 ⇒ **`TA-ABORT reason=no-device`** → `TA-DONE`（**显式负例，非假绿**）。
- 卡脚本**全路径**（容器自证专用 `TA_ALLOW_NODEVICE=1`，卡上禁设）：`[0]` 快检 `TA0-SMOKE-WARN reason=no-device` →
  `[1]/[2]/[3]` 全部 `SKIP reason=no-device`、`TA1-EXIT=4`、`[4] TA-TGZ`、`TA-DONE`。
- 解包即用探针：`J3-UNPACK-PROBE six=6/6 gemm=1 fail=0`（新前缀，bisheng_calls=7 全 rc=0）。

> 结论口径：容器内**每一条** window5 命令都以**显式 `SKIP reason=no-device` / `TA-ABORT reason=no-device`（exit 4/2）**收口，无一处静默假绿。

---

## 6. 关键发现与坑（卡上大概率同样适用）

1. **gemm dtype 拼写**：example `matmul(..., dtype="float16")`，**不是** `"fp16"`；`"fp16"` 会解析出非法 dtype → `TVMError`。已订正。
2. **fp32 gemm 的 L0 预算**：自带 block `(128,256,64)` 在 fp32 下 bisheng 编不过（L0A/L0B 超预算）；实测 `(64,128,64)`/`(128,64,32)`/`(64,64,32)` 可编 → fp32 腿用 `(64,128,64)` 并在行内标 `blk=`。
3. **`import tvm` 直连会加载到 CANN 自带 libtvm.so 而报 `undefined symbol: TVMGetLastPythonError`**；必须先 `import tilelang`（其 `env.py` 在 import tvm 前设好 `TVM_LIBRARY_PATH`）。⇒ 包布局必须镜像仓结构 + `build/`。
4. **`tilelang/version.py` 需要 `<pkg>/../VERSION`**（否则 `FileNotFoundError`）⇒ 包内已含 `VERSION`（=0.1.4）。
5. **`libtilelang.so` RUNPATH 硬编码 `/tlasc/build/tvm`**（不可移植）⇒ 卡上靠 `LD_LIBRARY_PATH`。
6. **CANN `set_env.sh` 引用未定义变量** ⇒ 与 `set -u` 冲突，需在 `source` 期间 `set +u`。

---

## 7. 对拍参考选择理由（自带 ref vs torch_npu）

**选 tilelang-ascend 自带 `ref_*`**，理由：
- 输入构造**逐一镜像各 example 自带 `__main__`**（同 `seed=0`、同 shape/归一化），对拍即"上游自测契约"本身，可追溯、可复跑；
- 自带 `ref_*` **内部 fp32 累加、末转 fp16**，天然满足窗五"**fp16 数值 vs fp32 参考**"；
- 其运行在 npu（torch_npu ops），与内核同设备，排除 CPU/GPU 混算的额外差异源；
- gemm 腿另有 example 自带 `rtol=atol=1e-2` 参考（`ref=torch` 的 `a@b`），一并记录。

---

## 8. 性能口径（只标 track，不出结论）

`p50_ms/p95_ms`：`warmup=3` + `iters=20`，每次迭代 `sync→计时→sync`（E2 纪律）；输入在计时区外构造。
行内显式带 `iters/warmup/dtype/src_bytes`，与标量数（`linear 512³ fp16 294.3ms`）**并排**交编排者归因。

---

## 9. CANN 8.5.2（卡） vs 8.5.0（容器）差异风险 与 10 分钟判定/止损

**风险面**：
1. **动态库加载**：预编译 `.so` 由容器 CANN 8.5.0 链出，卡上 8.5.2 加载。风险＝NEEDED/RUNPATH 解析（用 `LD_LIBRARY_PATH` 兜底）、glibc/GLIBCXX 版本（容器 glibc 2.35、最高符号 `GLIBC_2.34`；GLIBCXX 3.4.29 / libstdc++ 6.0.30）。
2. **bisheng 头文件 ABI**：编译期（`-xasc` + CANN 头）与卡上 8.5.2 头可能不一致。

**10 分钟内判定与止损**：
- `[0]` 的**单件 cumsum 编译快检**（≤300s，实测单件 ~3s）是**前置闸**：
  - `TA0-SMOKE compile-ok` ⇒ 兼容，继续全量；
  - `TA0-SMOKE-WARN/ABORT reason=cann-abi`（含 `FAIL reason=compile-error`）⇒ **立即 `TA-ABORT reason=cann-abi` 并停**（打印原始报错），不烧后续；从启动到判定 **≤10min**。
  - `reason=wrong-tilelang` ⇒ PYTHONPATH 未生效（环境问题，同样停）。
  - `reason=no-device` ⇒ 跑在了无设备形态（卡上不该出现）。
- **止损结论**：若确证 ABI 不兼容，**唯一修复＝卡上重编（≈17.1min / ≈¥6），本窗不做**——如实报 `TA-ABORT`，交编排者定夺。这正是 B 路线"可迁移包是否免编"的**负例出口**。

---

## 10. 未决/待办（交编排者）

- 卡上 CANN 8.5.2 的 `.so` 兼容性**尚未实测**（本窗只制备，不占卡）；`[0]` 快检即为其 10 分钟判据。
- fp32 gemm 因 L0 预算改 block（`64/128/64`），若编排者要求与 example 同 block，需换 target/重编，另议。
- 性能行**不含结论**；与标量数并排后的归因由编排者完成。
