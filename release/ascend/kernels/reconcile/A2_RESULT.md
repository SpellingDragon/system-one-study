# A2_RESULT — p2-13 卡前最后一里：两处探测器债清账（代理 A2，rig 抛光）

- **变更**：`teacher-p2-production-full` / 嵌套子变更 `p2-13-ascend-runtime`（openspec CLI 不索引孙层，按任务书文本直读；A1 在写 openspec 勾选，本件不碰 tasks.md）
- **本窗性质**：本地 host + 容器 `cann910b-k`（**有 patched §12 + bisheng、无设备**）；**不含真机数值判决**
- **改动件（白名单内）**：`ascend/kernels/reconcile/verify_dw.py`（固化发射桩 + 假绿守卫）、
  `ascend/compile_all_asc.py`（兼作唯一判决入口，新增 `--products`）、本件
- **一字未动**：任何 `*_asc.py` / `*_kernel.py` / `ascend_env.py` / `autograd_asc.py` / `train_step.py` /
  trunk `port910b_compat.h` / `.github/**` / tests / docs / README / tasks.md；
  **`run_numerics.py` 最终零改动**（原判据正文、阈值、case 正身全部原地保留，聚合靠外部转调完成）
- **一句话结论**：`verify_dw.py` 补桩后容器内 **3/3 稳定** 打 `K-GEMM-BWD-DW ASC-COMPILE-PASS`
  （`.o~2250B`），且**两条负例都如实变 FAIL**（桩没放进假绿）；判决入口已合并，卡窗一行
  `python3 ascend/compile_all_asc.py --target ascend --products` 即拿
  **九件接口 + 三条产品件 + 八条 intrinsic 对照 = 20 行判决 + 单一汇总（rc=0，已连跑 4 轮同判）**；
  host `--target cpu` 九件仍全绿、`pytest tests -q -m "not integration"` 仍 **340 passed / 0 failed**。

---

## 0. 结论摘要（三句话，均可复跑）

1. **债一是探测器缺桩，不是产品缺陷**——已由"补桩即得判决行 + 补桩前 traceback 落在发射门"两面证明：
   补桩前 `verify_dw.py` 在 `:59`（即 `gemm_bwd_dw_kernel.backward` 调用行）抛
   `RuntimeError: kernel dw_impl input dY device_type mismatch, expected ext_dev; expected: 1, got: 12`
   且**无任何判决行**；补桩后同一形状、同一判据（`compiled_keys()` 含 `ascend|gemm_dw[ascend]`）
   三次冷缓存连跑同判 `ASC-COMPILE-PASS`，`real_compiles=1 gen_src_bytes=2250 compile_count=1`。
2. **桩没有把假绿放过来**：判据仍是 `compiled_keys()`，另加两条 Y 式守卫（桩须真观测到 ≥1 次成功编译
   且生成码非空）。负例实测——`ascend_env.backend_available ≡ False` ⇒ `keys=[] real_compiles=0`
   ⇒ `ASC-COMPILE-FAIL`（rc=1）；`tilelang.compile ≡ 零生成码假件` ⇒ **键仍会入缓存**但
   `gen_src_bytes=0` ⇒ `ASC-COMPILE-FAIL(假绿守卫…)`（rc=1）。第二条正是"只看 compiled_keys
   会不会被桩蒙过"的答案：**会被蒙过，所以不能只看它**。
3. **入口分裂已合并**：`compile_all_asc.py` 成为唯一编排入口（`--products` 转调 `run_numerics` 的
   `PROD_TARGETS`/`PERF_TARGETS`），`run_numerics.py` 保留不删（卡窗 `run_*_card.py` /
   `run_delta_card.py` / `fixup_card.sh` / `bundle_ondemand.sh` 仍依赖其 case 与弃用别名）。
   聚合**不复制任何判据常量**：case 函数原样直呼，判决原文照抄，只加腿别标签与统一汇总。

---

## 1. 债一：`verify_dw.py` 固化发射桩

### 1.1 定性（before 原文，容器 `cann910b-k`，冷 `TILELANG_CACHE_DIR`）

```
[TileLang] begins to compile kernel `dw_impl` with `out_idx=[]`
[TileLang] completes to compile kernel `dw_impl`          ← §12 落地后编译已绿
Traceback (most recent call last):
  File ".../reconcile/verify_dw.py", line 127, in <module>  rc1 = _interface_dynamic_m()
  File ".../reconcile/verify_dw.py", line 59, in _interface_dynamic_m
    gw = gemm_bwd_dw_kernel.backward(dY, A, out_dtype=torch.float32, target="ascend")
  File ".../ascend/kernels/gemm_bwd_dw_asc.py", line 235, in run
    kernel(dY, A, GW)
RuntimeError: kernel dw_impl input dY device_type mismatch, expected ext_dev; expected: 1, got: 12
```

**判决行一条都没有**（控制流被发射门打断在判决之前）。原文件的 `_stub_compile()` 是**写了却没接上**
的死函数，且其 `_fake` 返回真产物、不隔离发射——即"只覆盖编译期抛→记 blocker 分支"的初版设计。

### 1.2 处置（复用 `verify_lora.py` 既有桩形态，不另立标准）

- `_stub_compile()`：`real = tilelang.compile` → 真编译（失败原样抛，仍由 `get_compiled` 记 blocker
  如实落 FAIL）→ 累加生成码到 `COMPILED_SRC` → 返回 `__call__` 为 **no-op 发射** 的 `_Dummy`；
  并返回 `restore` 闭包，判决主体跑完即还原 `tilelang.compile`（旁证支 `_static_structure_sanity`
  要真编真取生成码，不能被本件的桩污染）。
- 桩只吃"发射"这一段：本波**只判编译**，`backward` 拿回的 `gw` 是未写入的空表——判决行显式带
  `note=发射已 no-op 桩隔离⇒本行不判数值`，数值判决仍归 `run_numerics.py --only dw_prod`（卡）
  或 cpu-alias 轨（host）。

### 1.3 after（3/3 同判据，每次全新 `TILELANG_CACHE_DIR`）

```
docker exec cann910b-k bash -c 'source /usr/local/Ascend/cann-8.5.0/set_env.sh;
  export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201;
  cd /work && TILELANG_CACHE_DIR=$(mktemp -d) timeout 300 python3 ascend/kernels/reconcile/verify_dw.py'

RUN1 rc=0  RUN2 rc=0  RUN3 rc=0（三行逐字同）
  K-GEMM-BWD-DW ASC-COMPILE-PASS .o~2250B key=ascend|gemm_dw[ascend]|n64k64tc16b64x64 note=发射已 no-op 桩隔离⇒本行不判数值
  [interface·dynamic-m] keys=['ascend|gemm_dw[ascend]|n64k64tc16b64x64'] blockers={}
  [interface·dynamic-m] real_compiles=1 gen_src_bytes=2250 compile_count=1
  K-GEMM-BWD-DW STRUCTURE-SANITY-PASS(l0tr 形态可编…) [static-sanity] PASS .o~1425B asc_fill_l1_in_src=False
```

第 4 次同判另见 §1.5 的全量扫描（`verify_dw.py rc=0 verdict=K-GEMM-BWD-DW ASC-COMPILE-PASS`）。
`.o~2250B` 与聚合入口里 `ASC-gemm_bwd_dw PASS .o~2250B` 同尺（两处形状相同：n=k=32 是 harness、
本件喂 n=k=64——本件的 2250B 是自身三跑之间的稳定基线，harness 的 2250B 是 Z 记的九件基线，
数值巧合不等于同一件，引用时按各自 key 区分：本件 `…n64k64tc16b64x64`、harness `…n32k32tc16b32x32`）。

### 1.4 负例自证（桩没放宽判据）

```
docker cp scratch/a2_neg_selfcheck.py cann910b-k:/tmp/ && docker exec cann910b-k bash -c '
  source /usr/local/Ascend/cann-8.5.0/set_env.sh; export BISHENG_HOME=… ASCEND_NPU_ARCH=dav-2201;
  cd /work && TILELANG_CACHE_DIR=$(mktemp -d) python3 /tmp/a2_neg_selfcheck.py 1'   # 再跑 2
```

| 负例 | 打桩内容 | 实测输出 | rc |
|---|---|---|---|
| NEG-1 | `ascend_env.backend_available ≡ False`（后端缺席⇒编译根本不发生） | `keys=[] blockers={} real_compiles=0 gen_src_bytes=0 compile_count=0` ⇒ `K-GEMM-BWD-DW ASC-COMPILE-FAIL` | **1** |
| NEG-2 | `tilelang.compile ≡ 零生成码假件`（模拟"桩放过假绿"） | `keys=['ascend\|gemm_dw[ascend]\|…'] blockers={} real_compiles=1 gen_src_bytes=0` ⇒ `K-GEMM-BWD-DW ASC-COMPILE-FAIL(假绿守卫：compiled_keys 有键但桩未观测到真编译/非空生成码)` | **1** |
| 正例（对照） | 无桩 | `keys=[…] real_compiles=1 gen_src_bytes=2250` ⇒ `ASC-COMPILE-PASS` | 0 |

before/after 差分即"判决可信"的证据：正例绿、两负例红，且红的原因是**账目为空**（`real_compiles`
/ `gen_src_bytes`），不是文案改写。

### 1.5 同查：其余 attempts 侧探测器是否同病

结论：**只有 `verify_dw.py` 缺桩**。其余七支早已有发射隔离桩（`_stub_compile` / `_install_stubs`
返回 `raise _LaunchSkip` 的 `_Dummy`，或 `verify_lora` 的 no-op），容器内逐个冷跑（同一配方）：

```
docker cp scratch/a2_probe_verifies.sh cann910b-k:/tmp/ && docker exec cann910b-k bash -c 'nohup bash /tmp/a2_probe_verifies.sh > /tmp/a2_probe.log &'
```

| 脚本 | rc | Traceback | 判决行 |
|---|---|---|---|
| verify_addln.py | 0 | 0 | `H-ADDLN-ASC-COMPILE-PASS` |
| verify_attn_sw.py | 0 | 0 | `I-ATTNSW-ASC-COMPILE-PASS` ×5 形状（含 s33 尾块/h1d64w64） |
| verify_dw.py | 0 | 0 | `K-GEMM-BWD-DW ASC-COMPILE-PASS`（**本次修复后**）+ STRUCTURE-SANITY-PASS |
| verify_gdn.py | 0 | 0 | `J-GDN-FWD/BWD/CONV … ASC-COMPILE-PASS` ×9 档 |
| verify_gemm.py | 0 | 0 | `K-GEMM ASC-COMPILE-PASS` |
| verify_letter_readout.py | 0 | 0 | `H-READOUT-ASC-COMPILE-PASS` |
| verify_lora.py | 0 | 0 | `K-LORA ASC-COMPILE-PASS` |
| verify_rope.py | 0 | 0 | `I-ROPE-ASC-COMPILE-PASS`（多形状多 sign） |

**附带发现（非本任务改动，属 §12 落地的旁证）**：`verify_lora.py` 现打 `K-LORA ASC-COMPILE-PASS`
而非此前的 `ASC-COMPILE-PARTIAL(反向 dW 腿承 §12 asc_fill_l1 阻)`——lora 的 dW 腿也编过了，
即 §12 `asc_fill_l1`→`set_l1_2d` 的补件对**组合件**同样生效。此结论由脚本自身判据
（`ascend|gemm[` 与 `ascend|gemm_dw[` 两键齐 + 无 blocker）给出，本代理未动该文件。

---

## 2. 债二：判决入口合并（`compile_all_asc.py` 成唯一编排入口）

### 2.1 结构与不变量

| 面 | 处置 |
|---|---|
| 九条接口腿 | **行为原封不动**：仍逐件 import 入口件、仍 `ASC-<op> PASS|FAIL|PRECOMPILE-ERR`、仍 `--target`（缺省 cpu） |
| `run_numerics.py` | **不删、不改**：case 正身、`_route_key` 假绿守卫、阈值、`DEPRECATED` 别名、`--track`/`--no-run` 语义全原地；本件仅 `import` 后改写其 `ARGS` 的 `only/track/no_run` 三量再直呼 `TARGETS[case]()` |
| 新增 CLI | `--products`（开关）/ `--prod-cases all|prod|<逗号表>` / `--prod-track card|cpu-alias` / `--prod-run` / `--prod-no-run` / `--perf-blocking` |
| 输出 | 统一 `ASC-PROD-<case>` / `ASC-PERF-<case>` 前缀 + **末尾单一汇总**（接口腿 / 产品腿 / 对照腿三段计数 + `门禁失败数`）与单一退出码 |

### 2.2 三档门禁（防"对照腿噪声"与"把对照当判决"两个方向的错）

- **计入门禁**：九条接口腿 + 三条产品件腿（`Y-*` 才是产品结论，Y 口径）。
- **默认不计门禁、但照样打印**：八条 `performance-track` 对照腿（`NUM-*`，含 E-3 判别所需
  `gemm_l1_intrinsic` / `dw_intrinsic`）；要计入加 `--perf-blocking`。
- **PASS/FAIL 不许空转**：腿别判定取判决原文里第一个 `-PASS|-FAIL` 标记，**没有标记一律 FAIL**。
- **环境不符不许冒充**：`--target ascend --products` 但 ascend **编译面不在场**（真编最小件失败）
  ⇒ 打 `ASC-PROD-skipped FAIL …` 并计一条门禁失败，host 实测 rc=1（§3.3）。
  这里刻意**不用** `ascend_env.backend_available` 当门禁：它先要求 `_npu_present()`，容器无卡必
  False，而 `run_numerics` 的 card 轨靠自己在 `_prod_setup` 里桩掉设备门来只判编译——用它会把
  容器整条 products 腿误杀。改用 `ascend_env._compile_probe("ascend")`（只读调用，未改禁改件）。

### 2.3 两套桩不能同框（聚合的正确性要点）

`compile_all_asc` 的接口腿桩把 `tilelang.compile` 换成"调用即抛本模块 `_LaunchSkip`"的替身；
`run_numerics._install_launch_skip` 会 **capture 当前 `tilelang.compile` 作为 real**，且它只认
**自己那个** `_LaunchSkip` 类。若接口腿桩不撤，产品腿会把接口腿抛出的 `_LaunchSkip` 当未知异常
报 FAIL。故 `main()` 在 `--products` 前调 `_restore_stubs()`（还原 `backend_available` /
`_npu_present` / `active_backend` / `compile_kwargs` / `tilelang.compile` 五量原身 +
`ascend_env.reset()` 清账本），产品腿看到的是干净环境。凭据：容器与 host 的产品腿 `keys/src/rel`
数字与 Y 单跑 `run_numerics.py` 时**逐字相同**（§3.1/§3.2），说明转调没改变判决本身。

---

## 3. 判决凭据（每条附可复跑命令）

### 3.1 容器 · 卡窗主命令（要求 3）

```
docker exec cann910b-k bash -c 'source /usr/local/Ascend/cann-8.5.0/set_env.sh;
  export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201;
  cd /work && TILELANG_CACHE_DIR=$(mktemp -d) timeout 900 \
    python3 ascend/compile_all_asc.py --target ascend --products'; echo rc=$?
```

**rc=0**；接口腿 9 PASS（`.o` 与 Z 的基线 **2850/2250/3625/2106/2911/4003/1864/922/2842 逐字一致**）：

```
ASC-gemm PASS .o~2850B keys=1 first=ascend|gemm[ascend]|n32k32b16x32x32anone
ASC-gemm_bwd_dw PASS .o~2250B …ascend|gemm_dw[ascend]|n32k32tc16b32x32
ASC-add_ln PASS .o~3625B / ASC-rope PASS .o~2106B / ASC-attn_sw PASS .o~2911B
ASC-gdn PASS .o~4003B / ASC-gdn_conv PASS .o~1864B / ASC-letter_readout PASS .o~922B
ASC-lora PASS .o~2842B          —— 汇总 target=ascend 接口腿 PASS=9 FAIL=0 产品腿 PASS=3 FAIL=0 对照腿 PASS=8 FAIL=0 SKIP=0 —— 门禁失败数=0
```

产品腿 3 PASS（`keys/src` 与 `Y_RESULT.md §③` 逐字同）：

```
ASC-PROD-linear_prod  PASS Y-linear_prod-ASC-COMPILE-PASS  keys=1 src~3004B track=card(no-run)
ASC-PROD-dw_prod      PASS Y-dw_prod-ASC-COMPILE-PASS      keys=1 src~2803B track=card(no-run)
ASC-PROD-readout_prod PASS Y-readout_prod-ASC-COMPILE-PASS keys=2 src~942B  track=card(no-run)
```

对照腿 8 PASS（E-3 判别所需的 intrinsic 对照现在随同一条命令出）：

```
ASC-PERF-gemm_direct / gemm_l1_intrinsic / gdn / dw_intrinsic / rope / attnsw / addln_intrinsic / readout_intrinsic
  → 全 NUM-*-PASS rel=n/a (compile-only) [track=performance]
```

**复跑稳定性（同命令 4 轮，逐轮冷 `TILELANG_CACHE_DIR`；配方见 `scratch/a2_repeat.sh`）**：

| 轮 | rc | `ASC-*` 判决行 | PASS 行 | 汇总 |
|---|---|---|---|---|
| 首轮（终端直跑） | **0** | 20 | 20 | `PASS=9/3/8 FAIL=0/0/0 门禁失败数=0` |
| rep1 | 未捕获（echo 引号被吞）| 20 | 20 | `…FAIL=0 … 门禁失败数=0`（`/tmp/a2_rep1.txt`） |
| rep2 | **0** | 20 | 20 | 同上（`/tmp/a2_rep2.txt`） |
| rep3 | **0** | 20 | 20 | 同上（`/tmp/a2_rep3.txt`） |

rep2/rep3 的 `ASC-gemm_bwd_dw` / `ASC-PROD-dw_prod` / `ASC-PERF-dw_intrinsic` 三行与首轮**逐字节同**
（`.o~2250B` / `keys=1 src~2803B` / `compile-only track=performance`）。

子集与 blocking 开关也验过：`--perf-blocking --prod-cases dw_intrinsic gemm_bwd_dw` ⇒
`ASC-gemm_bwd_dw PASS` + `ASC-PERF-dw_intrinsic PASS`、汇总标 `[blocking]`、**rc=0**。

### 3.2 host · `--target cpu`（回归门，要求 3 后半 + 要求 4）

```
cd release && .venv/bin/python ascend/compile_all_asc.py --target cpu        → 接口腿 PASS=9 FAIL=0 rc=0
cd release && .venv/bin/python ascend/compile_all_asc.py --target cpu --products --prod-cases prod
  → 产品腿 3 PASS：rel=2.54e-07 / 2.21e-07 / 0.00e+00（bitexact=True，reuse 4/4·4/4·6/6）
    —— 与 Y_RESULT §② 的 host cpu-alias 数字逐字相同 ⇒ 聚合没动判据
cd release && .venv/bin/python ascend/compile_all_asc.py --target cpu --products
  → 对照腿 8 条 `ASC-PERF-* SKIP track=cpu-alias 不量 intrinsic 正文…`（噪声归零，非假绿），rc=0
cd release && .venv/bin/python -m pytest tests -q -m "not integration"
  → 340 passed, 3 skipped, 3 deselected, 2 warnings in 40.32s   （0 failed；写后复跑同数）
```

### 3.3 环境不符路径（不许冒充）

```
cd release && .venv/bin/python ascend/compile_all_asc.py --target ascend --products --prod-cases prod
  → ASC-PROD-skipped FAIL ascend 编译面不在场（真编最小件失败）⇒ products 腿整条未跑；…
  → —— 汇总 … 产品腿 PASS=0 FAIL=1 … —— 门禁失败数=10   rc=1
cd release && .venv/bin/python ascend/compile_all_asc.py --target cpu --products --prod-cases nope
  → ASC-products FAIL 未知 case ['nope']；可用：linear_prod,…  rc=1
cd release && .venv/bin/python ascend/compile_all_asc.py --target cpu bogus_op
  → ASC-usage FAIL 未知件 ['bogus_op']；可用：gemm,…            rc=1（原行为不变）
```

### 3.4 语法/自检 与 判决面凭据

```
cd release && .venv/bin/python -m py_compile ascend/compile_all_asc.py ascend/kernels/reconcile/verify_dw.py → OK
（venv 内无 ruff；本窗质量门 = py_compile + 判决复跑 + 全量 pytest）

md5  trunk   tilelang/src/tl_templates/ascend/port910b_compat.h  = f56432b0423174216da50090a1125a24
md5  overlay release/ascend/port910b/patches/port910b_compat.h   = f56432b0423174216da50090a1125a24
md5  容器 cann910b-k pip …/port910b_compat.h                      = f56432b0…  ← 本窗判决面（Z 同步过，未再 docker cp，也未改 compat）
md5  容器 cann910b-h pip …/port910b_compat.h                      = 20939eae…  ← 旧版（asc_fill_l1 修正未落）
```

本窗全部容器判决产在 **k**；h 的 compat 落后一版，**不可**当 p2-13 判决面（登记为环境事实，未动它）。

---

## 4. 卡窗交付面（下窗怎么用）

```bash
# 卡上（有设备）：一条命令，产品件自动升级为真数值判决（no_run 由 npu 在场自动关掉）
python3 ascend/compile_all_asc.py --target ascend --products                 # 期望 Y-*-PROD-PASS rel=…（E-3 对照同窗出）
python3 ascend/compile_all_asc.py --target ascend --products --perf-blocking # 把 intrinsic 对照也计入门禁
python3 ascend/kernels/reconcile/verify_dw.py                                # 单件 dw 编译判决（已自证可稳定出判决行）
```

- `run_numerics.py` 的 `--only <case>` 单点跑法**照旧可用**（`card_run.sh` / `run_*_card.py` /
  `run_delta_card.py` / `fixup_card.sh` / `bundle_ondemand.sh` 的调用行未受影响，弃用别名仍在）。
- 九条接口腿**始终只判编译**（其 dummy 抛 `_LaunchSkip`）：卡上的数值结论来自 products 腿，
  不来自接口腿——引用判决时按 `ASC-PROD-*` / `Y-*-PROD-*` 取，`ASC-PERF-*` 只作 §12 对照。

---

## 5. 已知限制 / 待裁决（如实登记）

1. **白名单路径与本仓实际不符**：任务书写 `release/ascend/port910b/attempts/{D,C,B,E,F}/*verify_*.py`，
   实际仓内 `attempts/{A..F}/` **没有任何 `verify_*.py`**（`find release/ascend -name 'verify_*.py'`
   只命中 `release/ascend/kernels/reconcile/`），而报错原文里的 `attempts/D/verify_dw.py:59` 对应的
   真身是 `ascend/kernels/reconcile/verify_dw.py:59`（行号与 traceback 逐字吻合）。
   ⇒ 处置：按"同一类文件、同一条纪律（仅固化桩/补判决行）"在 **实际路径** 落地，未新建 attempts 副本。
2. **`--perf-blocking` 的"对照腿 FAIL 计入门禁"分支未实测**：容器内八条对照腿全 PASS（制造 FAIL 需改
   `run_numerics.py` 判据正文＝越白名单），host 侧该腿按设计打 SKIP 不计 FAIL。该分支只是
   `gating += cf` 的一行算术，已读码并实测 `[blocking]` 标签与 PASS 路径。
3. **`--products` 在 cpu target 下不量 intrinsic 正文**（设计性 SKIP）：那批正文直编
   `tilelang.compile(target="ascend")`，host 无 ascend 后端必挂（聚合前实测 8 条
   `AssertionError: Target ascend is not supported` / `ascend_copy is not registered`），
   计入 FAIL 会把"环境事实"混进"产品结论"。要看它们请在容器/卡上跑 `--target ascend --products`。
4. **卡窗首次真数值判决仍不在本窗**：容器无设备，`Y-*-ASC-COMPILE-PASS` 是编译判；
   真机 rel / 507015 是否复现 / E-1…E-4 仍随下窗（波次边界，同 Y/Z）。
5. **接口腿与产品腿的形状族仍不同**（前者一件一形状取编译判决，后者带尾块/朝向/复用判别族）：
   本窗按裁决只做**入口聚合**，未合并形状族——合并会动判据正文（越白名单），留作后续项；
   汇总里三段计数已把两族分列，不会互相顶掉。
6. **`verify_dw.py` 的旁证支仍写 l0tr 同型件**（`alloc_l1/l0a/l0b/l0c` + `T.Cube`）：产品正文早已
   换 `dw_scalar_body`，该支只是"DSL 形态可编"的结构性旁证，不属于"判决主体"，本窗按
   "仅固化桩/补判决行"未动其正文（其 `.o~1425B asc_fill_l1_in_src=False` 照旧可编）。
7. **临时探针（非交付件，可删）**：`scratch/a2_scan_rn.sh`（run_numerics 11 案摸底）、
   `scratch/a2_probe_verifies.sh`（§1.5 同查）、`scratch/a2_repeat.sh`（§3.1 复跑）、
   `scratch/a2_neg_selfcheck.py`（§1.4 负例自证）。四者均**不在判决路径上**，配方已抄进本件，
   删掉也可按本文命令复跑。

---

## 6. 产物清单

- **修改**：
  - `release/ascend/kernels/reconcile/verify_dw.py`（+ 发射 no-op 桩接上并还原、`COMPILED_SRC`
    自证账目、`ascend|gemm_dw[ascend]` 键级判据 + 两条假绿守卫、PASS 行带 `.o`/key/note、
    docstring 改述"§12 落地后首达发射门 ⇒ 缺桩是探测器债"；`_static_structure_sanity` 正文一字未动）
  - `release/ascend/compile_all_asc.py`（+ `--products/--prod-cases/--prod-track/--prod-run/
    --prod-no-run/--perf-blocking`、`_restore_stubs`、`_ascend_compile_face`、products 腿转调与
    统一汇总/退出码；九条接口腿 driver 与 `_verdict_for`/`run_one` 判据一字未动）
- **新建**：`release/ascend/kernels/reconcile/A2_RESULT.md`（本件）
- **新建（临时，§5.7）**：`scratch/a2_scan_rn.sh`、`scratch/a2_probe_verifies.sh`、
  `scratch/a2_repeat.sh`、`scratch/a2_neg_selfcheck.py`
- **未动复核**：`ascend/port910b/run_numerics.py` **零改动**；所有 `*_asc.py`/`*_kernel.py`/
  `ascend_env.py`/`autograd_asc.py`/`train_step.py`/trunk `port910b_compat.h`/`.github/**`/tests/
  docs/README/tasks.md **一字未动**；无 git 写（未 add/commit）。
