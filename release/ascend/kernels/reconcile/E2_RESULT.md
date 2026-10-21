# E2_RESULT — P1-1o 吞吐/报价工装（`ascend/bench_card.py`）

**一句话**：卡上报价工装已就位并三轨自证完毕（静态 / CPU 轨真跑 / 容器 target=ascend 真编译），
`card_run.sh` 侧只需 `SYS1_BENCH=1` 接线即可在同一次开窗里顺手采到 R6/R7 与 C5 gate 要的报价表。

> **环境前提（不可绕过）**：本机（macOS `release/.venv`）**无 torch_npu、无昇腾设备**，容器
> `cann910b-j` **也无卡**。因此本文件里**没有任何性能结论**——本机产出的 ms/tflops 只是测量骨架
> 自证物（行尾 `track=cpu/...` + `note=cpu-track-NOT-card-perf`），**禁止抄进任何 run 判决**。
> 真报价必须在 910B 窗内跑 `python3 ascend/bench_card.py`（默认 `--device npu --target ascend`）。

---

## 0. 对应孙任务（原文映射）

- `tasks.md:25` **P1-1o 吞吐/报价工装（R6/R7 前置）**：「预热+多次统计（中位数/p50/p95）+ 每 op
  tok/s·TFLOPS + npu 内存，输出一行可直接抄入 run 的报价表」，验证口径「同窗产 `BENCH-*` 行；为 C5
  gate 报价与熔断判定供数据」⇒ 本件即其交付：`unit_per_h` 就是"每 op 每小时 token"（换算成 tok/s 除以
  3600 即可），npu 内存两列由 `torch.npu.memory_allocated()/max_memory_allocated()` 给（每配置前
  `reset_peak_memory_stats()`，所以峰值是**这一条 op 的**）。
- `tasks.md:39` **R6/R7 成本记账与 gate 报价（恢复 C5 的前置）**，验证口径「报价行+熔断判定」⇒ §3 的
  `BENCH-QUOTE` 行 + 换算式即报价行，熔断判定按 D12 原式写好（¥600 ÷ ¥20/h = 30 卡时）。
- 任务书明令「禁改**任何 tasks.md**」⇒ **我没有把 `tasks.md:25` 的方框勾成 `[x]`**，完成状态请编排者
  依本文件代记（这条同时是 apply 流程"改完即勾"纪律的一次有意偏离）。

---

## 1. 交付件与契约

| 件 | 路径 | 说明 |
|---|---|---|
| 报价工装（本任务交付） | `release/ascend/bench_card.py` | 独立可跑，7 类接口件 × 形状/位宽矩阵 |
| 模具位宽自证探针 | `release/ascend/kernels/reconcile/E2_probe_mold.py` | 只编译不计时；证"某档位宽到底有没有自己的模具" |
| 一次性补丁执行器 | `release/ascend/kernels/reconcile/E2_probe_patch.py` | E2 自用（白名单 `E2_probe_*` 命名内），**可删** |
| 未改（禁改名单） | `card_run.sh`、`*_asc.py`、`*_kernel.py`、`autograd_asc.py`、`train_step.py`、`run_numerics.py`、任何 `tasks.md` | 本任务零改动 |

**与 `card_run.sh` 的交接契约**（B 代理接线，我没有动它；现已 grep 确认 `SYS1_BENCH`/`bench_card` 在 `release/` 下 0 命中＝尚未接线）：
- 调用形（工作目录 `release/ascend/port910b` 亦可用绝对路径）：
  `python3 $SYS1/release/ascend/bench_card.py --quick` 先 3 分钟摸通，再跑全表（不带 `--quick`）。
- 契约面只有两条：**文件名 `bench_card.py`** + **打印 `BENCH-*` 行**。其余字段是本件的自证信息，多出来的
  `track= kernel= reuse= note=` 字段可整行抄进 notes（正是"判决行要能直接抄"的用途）。
- 退出码：`0`＝有成功行且 `fail_core=0`；`1`＝`fail_core>0`（`fail_ref`=对照腿、`fail_inj`=注入自检，不计入）。
- 预算：默认每 op `iters=20 warmup=3`、整表 `--budget-sec` 自动 600s（`--quick` ⇒ 240s、`iters 5 warmup 2`、只取各 op 最小档形状）；超预算的后续配置打 `SKIP budget-exhausted`，**绝不拖成小时级**。

**行格式（任务书契约 + 自证附加字段）**：
```
BENCH-<op> dtype=<dt> shape=<s> p50_ms=<x> mean_ms=<y> p95_ms=<z> iters=<n> warmup=<w> \
  tflops=<t>|na mem_alloc_mb=<m> mem_peak_mb=<p> track=<device>/<target> kernel=<编译产物键> \
  reuse=<new|cache> [note=...]
```
覆盖：`linear`（512/1024/2048³，`bf16>fp32` 与 `fp16>fp16` 两档）、`dW`（m1024/2048/4096 阶梯同两档）、
`add_ln`、`rope`、`attn_sw`、`gdn_conv`、`gdn_delta_fwd`、`gdn_delta_bwd`（各 2 档形状）＋ `--intrinsic auto`
的 `linear_intrinsic`/`dW_intrinsic` 对照腿（排在整表最后）。共 **24 条正表 + 6 条对照腿**（`--list-ops` 可核）。

---

## 2. 三轨验证凭据（真实输出，可复跑）

### ① 静态自证（host）
```bash
cd $REPO_ROOT/release
.venv/bin/python -m py_compile ascend/bench_card.py            # → PY-COMPILE-PASS
.venv/bin/python ascend/bench_card.py --help                  # → 10 个参数（含 --quick/--iters/--compile-check/--selftest-fail）
.venv/bin/python ascend/bench_card.py --list-ops              # → 24 条正表 + 6 条 intrinsic 对照腿
.venv/bin/python ascend/bench_card.py                                # 无设备的 npu 档⇒拒跑(exit=2)，不会假造卡上数
#   BENCH-ENV device=npu npu_available=False ⇒ 本机无昇腾设备
#   BENCH-ABORT no-npu：真报价请在 910B 窗内跑；本机自证用 --device cpu（骨架冒烟）或容器
#     --device cpu --target ascend --compile-check（证真件）
```

### ② CPU 轨冒烟（同一测量骨架真跑；**不是卡上性能**）
命令：`.venv/bin/python ascend/bench_card.py --device cpu --quick --iters 3 --warmup 1 --selftest-fail`
终版复跑（与当前 trunk 一致）：墙钟 **12.5s**，`ops=11 ok=9 fail=2 fail_core=1 fail_inj=1`。全量行：
```
BENCH-ENV device=cpu target=cpu torch=2.14.1 npu_available=False iters=3 warmup=1 budget_s=240.0 configs=11 mode=time
BENCH-linear dtype=bf16>fp32 shape=512x512x512 p50_ms=184.553 mean_ms=185.018 p95_ms=186.765 iters=3 warmup=1 tflops=1.455 mem_alloc_mb=na mem_peak_mb=na track=cpu/cpu kernel=cpu|gemm[cpu]|n512k512b16x64x64anonedfloat32 reuse=new note=cpu-track-NOT-card-perf
BENCH-linear dtype=fp16>fp16 shape=512x512x512 p50_ms=186.612 mean_ms=187.152 p95_ms=189.205 iters=3 warmup=1 tflops=1.438 mem_alloc_mb=na mem_peak_mb=na track=cpu/cpu kernel=cpu|gemm[cpu]|n512k512b16x64x64anonedfloat32 reuse=cache note=cpu-track-NOT-card-perf;same-mold-as=bf16>fp32
BENCH-dW dtype=bf16>fp32 shape=m1024xn512xk512 p50_ms=362.646 mean_ms=368.343 p95_ms=388.296 iters=3 warmup=1 tflops=1.480 mem_alloc_mb=na mem_peak_mb=na track=cpu/cpu kernel=cpu|gemm_dw[cpu]|n512k512tc16b64x64dfloat32 reuse=new note=cpu-track-NOT-card-perf
BENCH-selftest FAIL [n/a n/a] RuntimeError: 注入的失败：验证 BENCH 单件异常隔离（其余 op 必须照跑）
BENCH-dW dtype=fp16>fp16 shape=m1024xn512xk512 p50_ms=355.600 mean_ms=366.163 p95_ms=385.319 iters=3 warmup=1 tflops=1.510 mem_alloc_mb=na mem_peak_mb=na track=cpu/cpu kernel=cpu|gemm_dw[cpu]|n512k512tc16b64x64dfloat32 reuse=cache note=cpu-track-NOT-card-perf;same-mold-as=bf16>fp32
BENCH-add_ln dtype=fp32 shape=rows8192xdim1024 p50_ms=52.585 mean_ms=52.810 p95_ms=53.545 iters=3 warmup=1 tflops=na mem_alloc_mb=na mem_peak_mb=na track=cpu/cpu kernel=cpu|add_ln[cpu]|d1024e1e-05b16 reuse=new note=cpu-track-NOT-card-perf
BENCH-rope dtype=fp32 shape=tok4096xh8xd128 p50_ms=19.948 mean_ms=20.014 p95_ms=20.595 iters=3 warmup=1 tflops=na mem_alloc_mb=na mem_peak_mb=na track=cpu/cpu kernel=cpu|rope[cpu]|h8d128s2g1qfloat32afloat32 reuse=new note=cpu-track-NOT-card-perf
BENCH-attn_sw dtype=fp32 shape=h8xseq1024xd128w512 p50_ms=1210.331 mean_ms=1236.674 p95_ms=1289.982 iters=3 warmup=1 tflops=1.332 mem_alloc_mb=na mem_peak_mb=na track=cpu/cpu kernel=cpu|attn_sw[cpu]|h8d128w512b16x16 reuse=new note=cpu-track-NOT-card-perf
BENCH-gdn_conv dtype=fp32 shape=C1024xL2048xk4 p50_ms=24.834 mean_ms=24.826 p95_ms=24.911 iters=3 warmup=1 tflops=0.676 mem_alloc_mb=na mem_peak_mb=na track=cpu/cpu kernel=cpu|gdnconv[cpu]|c1024k4b1 reuse=new note=cpu-track-NOT-card-perf
BENCH-gdn_delta_fwd dtype=fp32 shape=h8xseq512xk64xv64 p50_ms=68.293 mean_ms=74.814 p95_ms=86.185 iters=3 warmup=1 tflops=0.983 mem_alloc_mb=na mem_peak_mb=na track=cpu/cpu kernel=cpu|gdn[cpu]|h8k64v64 reuse=new note=cpu-track-NOT-card-perf
BENCH-gdn_delta_bwd FAIL [fp32 h8xseq512xk64xv64] AssertionError: 假绿守卫：compiled_keys 无 cpu|*gdnb[* ⇒ 本趟走的是 torch eager，不是内核（plan 拒形状 / 后端不可用 / 设备-目标不匹配）；blockers=[('gdn_backward[cpu]', '本趟走 torch 自动微分（cpu target 的语义尺子路，或昇腾伴随模具未编成）；昇腾形态见 gdn_asc.g
BENCH-SUMMARY ops=11 ok=9 fail=2 wall_s=12.5 skip=0 fail_core=1 fail_ref=0 fail_inj=1 track=cpu/cpu iters=3 warmup=1
```
7 类接口件（linear/dW/add_ln/rope/attn_sw/gdn_conv/gdn_delta_fwd）都真出了完整 `BENCH-*` 行；
两条 FAIL 证明异常隔离与守卫都活着：`selftest` 是注入项（证单件异常不中断整表），`gdn_delta_bwd` 在
cpu target 必落 torch 自动微分（无伴随模具）⇒ 守卫如实拦下，属预期，不属骨架缺陷。
预算纪律另有实证：`--budget-sec 1` ⇒ 第 3 条起整排 `BENCH-<op> SKIP budget-exhausted ... wall_s=2.9`。
`same-mold-as=bf16>fp32` 是 cpu 轨的真实事实：入口件在 cpu 档位宽先 cast 成 fp32，两档位宽共用同一具
`dfloat32` 模具 ⇒ 该行不可读作"fp16 模具的速度"（详见 §5）。

### ③ 容器 `target=ascend` 真编译（证"计时包裹的是真件"）
```bash
docker exec cann910b-j bash -lc 'source /usr/local/Ascend/cann-8.5.0/set_env.sh;
  export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201 \
         TMPDIR=/dev/shm/e2t TILELANG_CACHE_DIR=/dev/shm/e2c;  # 容器 overlay 已满，只能落 tmpfs
  cd /work && python3 ascend/bench_card.py --device cpu --target ascend --compile-check --quick'
```
```
BENCH-add_ln-COMPILE-CHECK dtype=fp32 shape=rows8192xdim1024 target=ascend compiled_key=ascend|add_ln[ascend]|d1024e1e-05b8 src_bytes=3649 note=无设备⇒只证编译不计时
BENCH-rope-COMPILE-CHECK  ... compiled_key=ascend|rope[ascend]|h8d128s2g1qfloat32afloat32 src_bytes=2145
BENCH-attn_sw-COMPILE-CHECK ... compiled_key=ascend|attn_sw[ascend]|h8d128w512b16x16 src_bytes=2955
BENCH-gdn_conv-COMPILE-CHECK ... compiled_key=ascend|gdnconv[ascend]|c1024k4b1|c8scalare src_bytes=1688
BENCH-gdn_delta_fwd-COMPILE-CHECK ... compiled_key=ascend|gdn[ascend]|h8k64v64|c8ubh src_bytes=3707
BENCH-linear_intrinsic-COMPILE-CHECK ... compiled_key=ascend|out-of-band-cube-archive src_bytes=2171
BENCH-SUMMARY ops=12 ok=8 fail=4 wall_s=9.0 skip=0 fail_core=3 fail_ref=1 fail_inj=0 track=cpu/ascend
```
⇒ **每条报价行背后的模具确实编得出来**（`src_bytes` 是真产物字节数，与 `reconcile/S_RESULT.md` 记的
`S-GEMM-ASC-COMPILE-PASS .src~3117B` 同量级互证）。`add_ln` 在 ascend 是 `b8`、cpu 是 `b16` ⇒ 昇腾侧核块
预算确实按 910B 的 UB 收成 8（承 `S_RESULT.md` 的 NUM_BLOCKS/UB 口径）。

本档 4 条 FAIL 逐条有因，**都不是工装缺陷**：
- `linear/dW dtype=bf16`：`plan` 现在**拒 bf16**（P1-1m 未收）⇒ 入口件静默落 eager ⇒ compile-check 档报
  "发射没被桩掉"（＝根本没编译）。卡上同理由 `假绿守卫` 拦下 ⇒ **bf16 行在卡上预期 FAIL**。
- `gdn_delta_bwd`：复合腿在无设备档必然"先编前向模具、发射即被桩掉"，伴随模具 `gdnb[` 没轮到编译
  （现已把这层解释写进报错尾注）。卡上有设备时自行走通。
- `dW_intrinsic`：`RuntimeError: Ascend device compilation failed.` ⇒ K 波 l0tr 存档件在当前 trunk 连编都编不出
  （见 §4 消融）。

---

## 3. 报价换算口径（本件**不定价**，价格归编排者）

工装交物理量三列，直接抄 `BENCH-QUOTE` 行（**只在 `--device npu` 轨产出**；cpu 轨按纪律抑制）：
```
BENCH-QUOTE op=<op> dtype=<dt> shape=<s> unit=token work_per_call=<每次调用覆盖的 token 数> \
  calls_per_h=<每小时可完成的调用次数> unit_per_h=<每小时可完成的 token 数> gflop_per_h=<…> basis=p50 track=npu
```
- `calls_per_h = 3600 ÷ (p50_ms ÷ 1000)`；`unit_per_h = calls_per_h × work_per_call`。
- 编排者按 **¥20/h**（D12）折算：`¥/1k_token = 20 × 1000 ÷ unit_per_h`；
  `卡时 = Σ(每步该 op 调用次数 ÷ calls_per_h) × 步数`；**熔断线 ¥600 = 30 卡时**（D12 原式，非本件定价）。
- 每步 op 调用次数**由生产拓扑定**（`train_step.py` 只是 d32/heads2 的玩具尺子件，不能当拓扑源）：
  C5 gate 报价时请按实际 `层数 × {linear, add_ln, rope, attn_sw/gdn_conv, gdn_delta}` 前向 1 次＋反向 1 次
  填进 `卡时` 求和式，其余照抄本表。
- `tflops` 只在有可辩护乘加口径时给（linear/dW/attn_sw/conv/gdn）；`add_ln`/`rope` 是归约+逐元素，
  如实 `tflops=na`，**不拿假 TFLOPS 充数**。

抄表模板（一行一 op，卡上跑完直接粘进 run notes）：
```
P1-1o-BENCH <op> dtype=<dt> <shape> p50=<x>ms tflops=<t> ⇒ <calls_per_h> 次/时 ≈ <unit_per_h> token/时 ⇒ ¥20/h 下 ¥<y>/1k_token
P1-1o-QUOTA 合计 <卡时> 卡时 ⇒ ¥<z>（D12 熔断线 ¥600=30 卡时）
```

---

## 4. 三栈消融（G9 叙事）与"标量路远慢于 cube 路"的如实口径

- 正表量的是 **correctness 轨（可移植标量体）**：零 `T.Cube`/零 `T.gemm`/零 SimtVF（`gemm_asc.gemm_asc_impl`
  的 P1-1f 改道说明）。
- `--intrinsic auto` 追加 **性能轨存档形态**（`kernels/reconcile/S_perf_track_cube.py`，K 波 Cube/l0tr，
  非产品件），同名 op 加 `_intrinsic` 后缀并排，另出 `BENCH-ABLATE op=<基名> scalar_p50_ms= intrinsic_p50_ms= ratio=`。
- 容器证据：`linear_intrinsic` **可编**（src 2171B），`dW_intrinsic` **编不出**（`Ascend device compilation failed.`）；
  两者数值属 V 系未证真、真机已知 aicore `507015` ⇒ 卡上这两腿**大概率 FAIL**，如实记进 `fail_ref` 即可，
  **不得**因此判报价表失败，也**不必**为它们延长窗口（它们排在整表最后，正表已逐行 flush 落地）。
- 第三栈（真 cube 位宽化性能件）目前不存在 ⇒ 消融只能给"标量 vs 存档 intrinsic"两栏；报告里请按
  **两栈实测 + 第三栈缺件**叙述，别写成三栈齐全。
- 担心设备上下文被 cube 腿污染：`--intrinsic off`（正表完全不受影响）。

---

## 5. 模具与位宽（卡上采表前必读，已写进 `BENCH-NOTE mold-dtype=`）

`E2_probe_mold.py --target ascend` 实测（同形状 512³、逐档位宽各自冷开缓存）：

| 时刻 | linear 键 | fp32 / fp16 产物 | bf16 |
|---|---|---|---|
| 本会话早先（trunk 变更前） | `…anone`（**不含位宽**） | sha **相同** 868df185b208 / 3118B，`tok_bf16=0 tok_fp16=0` ⇒ 模具只有一具 fp32 体（`gemm_scalar_body` 里 `T.Tensor(..., "float32")` 硬声明） | 同键复用 |
| 本会话稍后（**并发变更已落**） | `…anoned<float32|float16>`（位宽入键） | sha **不同** 868df185b208/3118B vs ef703edd3cba/3138B（fp16 有自己的模具，`tok_fp16=7`） | `NOT-COMPILED`＝plan 拒 bf16，落 eager |

⇒ 采表纪律：**dtype 列必须与 `kernel=` 字段互证**。若同形状换位宽后键名不变、或行尾出现
`same-mold-as=<另一档位宽>`（cpu 轨就长这样：两档位宽共用 `dfloat32` 模具），该行量到的是**别人的模具**，
须按位宽分趟进程重采（`--only linear` 跑两趟，每趟进程自带一次冷开模）。本工装已把标注自动化。

---

## 6. 遗留与待裁决（交回编排者）

1. **容器 `cann910b-j` overlay 40G/40G 已 100% 满**：`mktemp -d` 在 `/tmp` 直接失败
   （`No space left on device`），tilelang 缓存落到空串导致 import 期崩。本文件所有容器证据都改走
   `TMPDIR/TILELANG_CACHE_DIR=/dev/shm/*`（仅 64M，够编译但不够缓存整表）。开窗前建议清理或换容器。
2. **bf16 报价行在卡上预期 FAIL**（模具层拒 bf16＝P1-1m 未收）。要么接受"bf16 行只登记缺口"，要么
   先落 P1-1m 再采 bf16 档——**这是定档决策，不在我权限内**。
3. `reconcile/E2_probe_patch.py` 是补丁执行器（我自用），若不需要留档可删；`E2_probe_mold.py` 建议留
   （开窗前自检模具归属，10 秒）。
4. 未 import `port910b/run_numerics.py` 复用其守卫/桩件：它模块级 `ARGS = AP.parse_args()` 会吃我的
   argv。做法是**逐字照抄其判据与措辞**（`_route_key`/`_install_launch_skip`/`_prod_setup` 三门桩），
   判据不另造；若后续要真正共件，需要 run_numerics 侧把 argparse 收进 `main()`（不在我的白名单）。
