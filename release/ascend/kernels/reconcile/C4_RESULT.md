# C4_RESULT.md — P1-1w：GDN delta 反向真机数值判决腿（不占卡，备件）

> 执行代理：C4（p2-13 执行波）。写面：`release/ascend/port910b/run_delta_card.py`（扩 `--bwd`）。
> 新建：本件 + `reconcile/C4_probe_*.py`。**未占卡**：真机执行面只证到"容器编译面 + host cpu-alias 数值面"。
> 第三窗一条命令出判决：`cd /work/ascend/port910b && python3 run_delta_card.py --target ascend --bwd`
> （注：`card_run.sh` 是禁改件、只调前向腿 `python3 run_delta_card.py`，故反向判决需直接点名 `--bwd`）。

---

## ① 完成情况（逐条附凭据）

### 1.1 bwd 判决腿（`--bwd`，六项梯度 + 前向 o/s 复用现腿）
**做了什么**：`run_delta_card.py` 加 `--bwd` / `--target {ascend,cpu}` / `--variant` / `--no-run`（=`--compile-only` 别名）/`--neg-perturb`。
`--bwd` 驱动 **产品 home**（`ascend/kernels/gdn_asc.py` 的 `gdn_asc.backward(..., return_dh0=True)`，
即训练步里那个 `gdnb` 模具），对拍**六项梯度**（`dq/dk/dv/dbeta/dg/dh0`，按 F 波清单）vs 同输入 fp64 golden；
前向 o/s 复用现腿（F 件 fwd，判决行原样）。

**判决行（每项都打，不只打 max）**：
```
NUM-delta_bwd PASS dq=7.53e-08 dk=6.72e-08 dv=6.68e-08 dbeta=1.01e-07 dg=9.72e-08 dh0=6.18e-08 adjoint=6.22e-10 rel_max=1.01e-07
NUM-delta_bwd FAIL first=<item> rel=<v> absdiff=<v> thresh=<v>       ← FAIL 带首个超阈项名 + 差值
```

**上卡命令（第三窗；真机执行，非本波证据）**：
```
cd /work/ascend/port910b && python3 run_delta_card.py --target ascend --bwd
```
预期（本波只证编译面，数值面在 host 已证，见 §1.2）：
`NUM-delta_fwd PASS rel_o=… rel_s=…` + `NUM-delta_bwd PASS dq=… dk=… … adjoint=… rel_max=…`。

### 1.2 host 全量自证（cpu-alias 轨，真跑）
凭据（`cmd` → 输出）：
```
$ cd /tmp && <venv> run_delta_card.py --target cpu --bwd
NUM-delta_fwd PASS rel_o=7.49e-08 rel_s=4.69e-08
NUM-delta_bwd PASS dq=7.53e-08 dk=6.72e-08 dv=6.68e-08 dbeta=1.01e-07 dg=9.72e-08 dh0=6.18e-08 adjoint=6.22e-10 rel_max=1.01e-07
rc=0
```
六项 rel **6.68e-08 … 1.01e-07**，与 F 波 CPU golden（`5.76e-07` 量级 / adjoint-vs-FD `7.28e-10`）**同量级**；`adjoint=6.22e-10` 同量级。
（venv = `$REPO_ROOT/release/.venv/bin/python`。）

**负例**（人为扰动一项梯度 ×1.01 ⇒ FAIL 且指认正确项）——六项逐项都测，`first=` 与扰动项**一一对上**：
```
[neg=dk]    NUM-delta_bwd FAIL first=dk    rel=1.000e-02 absdiff=1.051e-01 thresh=2e-04
[neg=dq]    NUM-delta_bwd FAIL first=dq    rel=1.000e-02 absdiff=1.746e-02 thresh=2e-04
[neg=dv]    NUM-delta_bwd FAIL first=dv    rel=1.000e-02 absdiff=1.181e-02 thresh=2e-04
[neg=dbeta] NUM-delta_bwd FAIL first=dbeta rel=1.000e-02 absdiff=7.342e-02 thresh=2e-04
[neg=dg]    NUM-delta_bwd FAIL first=dg    rel=1.000e-02 absdiff=2.317e-02 thresh=2e-04
[neg=dh0]   NUM-delta_bwd FAIL first=dh0   rel=1.000e-02 absdiff=2.563e-02 thresh=2e-04
```
rc：PASS 例 `rc=0`；FAIL 例 `rc=1`；项名非法（`--neg-perturb nosuch`）→ `rc=1` 且明示"项名非法"（防"空转即过"）。

### 1.3 容器编译面（`cann910b-h`，compat `ca023c2e`，`/dev/shm` 缓存）
凭据：
```
$ docker exec cann910b-h bash -lc 'source /usr/local/Ascend/cann-8.5.0/set_env.sh;
    export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201;
    export TILELANG_CACHE_DIR=$(mktemp -d /tmp/tlc_cache.XXXXXX); cd /work/ascend/port910b;
    python3 run_delta_card.py --target ascend --bwd --no-run' | grep NUM-delta
NUM-delta_fwd PASS rel=n/a (compile-only)
NUM-delta_bwd PASS rel=n/a (compile-only) key=ascend|gdnb[ascend]|h2k8v8|c2ubz   ← bwd 模具键在
rc=0
```
`ca023c2e01746ca52c87f36bcea93ee9  /tilelang/src/tl_templates/ascend/port910b_compat.h`（= 任务指定 `ca023c2e`）。

### 1.4 回归红线
- **fwd 腿判决行逐字同**：容器 `--compile-only` before（改前基线，同路径同码）vs after —— 两次均 `NUM-delta_fwd PASS rel=n/a (compile-only)`；host `--compile-only` before/after `diff` → `IDENTICAL`（同为 `NUM-delta_fwd FAIL compile: AssertionError: Target ascend is not supported…`，两处 rc 均 1）。默认无参路径（`card_run.sh` 的调用形态）未改。
- **`pytest tests -q -m "not integration"`**：
  ```
  cd release && ./.venv/bin/python -m pytest tests -q -m "not integration"
  340 passed, 3 skipped, 3 deselected, 2 warnings in 44.14s
  ```
- **`card_run.sh` 未破坏**：`card_run.sh:239-240` = `run_leg "delta" 500 "^NUM-delta_fwd" "^NUM-delta_fwd FAIL" "^NUM-delta_fwd[ -]PASS" - \ python3 run_delta_card.py`。本件默认无参调用仍产 `NUM-delta_fwd … PASS …` ⇒ 正则与命令形态均不变。`card_run.sh` md5 `5ef1e13a1546318a7c519156d4c7dacc`，mtime `10-11 19:17`（早于本波开工 23:33）⇒ **未改**。

---

## ② 错误与阻塞（含已解决的）

- **host 无 ascend target**：`tilelang.compile(fn, target="ascend")` 在 host 报 `AssertionError: Target ascend is not supported` —— 这是**预期**（host 无 ascend 后端），非缺陷；正因如此 host 数值面走 `--target cpu`（cpu-alias）。
- **容器缺 CANN env**：直接 `python3 run_delta_card.py --compile-only` 报 `RuntimeError: Cannot find the bisheng compiler` @ 容器。处置：改前先 `source /usr/local/Ascend/cann-8.5.0/set_env.sh; export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler`（沿用 F 波 `run_F.sh` 的 env 配方）⇒ PASS。
- **cpu 后端单 dynamic shared**：`ub` 档在 cpu 后端报 `InternalError: Only one dynamic shared memory allocation is allowed`（ub 有 7 个 `alloc_shared`）。处置：host cpu-alias 轨强制 `pure` 档（`--target cpu` 时 `_variant()="pure"`，并显式透传 `variant=` 给产品 `backward`）；**卡上仍首推 `ub`**（容器编译面产的就是 `c2ubz` 键）。
- **device-less 容器产品件会静默回退 torch**：`active_backend("ascend", cpu_device)` 判 `torch_eager` ⇒ 产品落 torch 自动微分、`gdnb` 键登记不上（假绿）。处置：`--no-run` 时运行期替换 `backend_available/_npu_present/active_backend`（**只替换目录，不写产品源码**）强行走内核路；并加**假绿守卫**——未见 `gdnb[` 键即判 FAIL。
- **前向发射早于反向编译**：`run_backward` 先发射前向、再编反向模具；若用"抛异常跳过发射"，反向键就登记不上。处置：`--no-run` 用 **no-op 壳**（`_NoLaunch`）包 `tilelang.compile` 产物，编但**不发射**。

（环境类：容器无 davinci 设备、`torch_npu` 未装 —— 故真机执行面本波**未跑**，见 §"未闭环"。）

---

## ③ 疑惑点与自行决策（凡"没问但自己定了"的都在此申报）

1. **梯度项命名对齐**：任务写 `dQ/dK/dV/dBeta/dDecay/dG`，F 波清单是 `dQ/dK/dV/dBETA/dG/dH0`。按任务"**按 F 波清单**"，采 **`dq/dk/dv/dbeta/dg/dh0`**（`dDecay`≡`dG` 同一个 decay 门梯度；F 波第六项为 `dH0`）。
2. **驱动产品 home 而非 F 件**：`--bwd` 走 `gdn_asc.backward`（产 `gdnb[ascend]|…` 键）——因为任务要的键只有产品路出，且"从未逐元素判过"的正是训练步里那个 `gdnb`。
3. **参考侧选择（任务要求给理由与代价）**：用 **fp64 精确解析 golden**（`fwd_exact` + `bwd_adjoint`，`dt=float64`）在**同一输入**上复算。
   - 理由：任务阈值基准是"与 CPU fp32 golden 同口径（5.76e-07 量级）"⇒ 参考须是**同一把尺子**（F 波 golden）才跨窗可比；fp64 解析是比 fp32 eager 更干净的尺子（不引入 ~1e-7 的参考侧自身误差）。
   - 代价：需 numpy 顺序复刻（F 件已备），形状小（2,8,8,8）开销可忽略；且需 golden 自身可信 —— 由 `adjoint=`（fp64 伴随 vs 中心差分）**独立锚**兜底（实测 `6.22e-10`）。
   - 未采 torch eager：会引入参考侧 fp32 误差、且与 F 波口径不同，跨窗不可比。
4. **`dg` 归约口径**：产品 `dg` 是每 token 每键通道 `(H,T,DK)`（`_gate3` 把标量门摊成通道），golden `dga` 是每 token 标量 `(H,T)`。因通道共享同一门值，`dL/dg_t = Σ_c dL/dg_c` ⇒ 判据用 `dg.sum(-1)` vs `dga`（数值佐证：PASS 时 `dg=9.72e-08`）。
5. **初状态口径**：产品反向内部恒以**零初态**重算（`fwd_workspaces` 的 H0 恒零），故 golden 也用**零初态**（保证"参考与实测同输入"）；fwd 腿那份带非零 H0 的 golden 仅用于前向对拍。
6. **输入分布**：沿用现腿 seed=11 抽序（`Q,Kt,V,BETA,G` 逐位同）；`dO` 作第 6 抽（`N(0,1)`）——现腿无 dO 口径，取自然分布，不改既有五量的分布。
7. **阈值**：`dq…dh0` 用 `TOL_F32=2e-4`、`adjoint` 用 `TOL_ADJ=1e-8`，**沿用 F 波既有阈值，不另造**。PASS 行按任务格式只打数值不打印阈值。

---

## ④ 偏离记录（与计划/spec 不一致之处）

- **无面（批次/参数口径）偏离**。所有新增均为**加性**：默认无参路径逐字不变（§1.4 证）。
- 一处**命名**偏离仅在被否项：任务列的 `dDecay` 记为 `dg`（见 §③-1，非偏离，是口径对齐）。
- `--no-run` 是实现期新增的别名（任务写"`--target ascend --no-run`（或等价）"），与既有 `--compile-only` 同 `dest`，不改变后者语义。

---

## ⑤ 产物清单

- 修改：`release/ascend/port910b/run_delta_card.py`（89 → 334 行；md5 `67705588…` → `1768a6ec…`）
- 新建：`release/ascend/kernels/reconcile/C4_RESULT.md`（本件）
- 新建（设计期探针，只读产品件）：`release/ascend/kernels/reconcile/C4_probe_cpu_alias.py`、`reconcile/C4_probe_product_bwd.py`
- **未改**（禁改件，指纹佐证）：`card_run.sh`（md5 `5ef1e13a…`，mtime 10-11 19:17）、`bundle_ondemand.sh`、`run_numerics.py`、`compile_all_asc.py`、`gradcheck_card.py`、任何 `*_asc.py`/`autograd_asc.py`/`train_step.py`/`bench_card.py`、`attempts/F/*`、`runs/**`、任何 `tasks.md`。
- 未动 git。

---

## 未闭环（诚实申报）

**在卡上真机执行 `--target ascend --bwd`（跑数、出 rel）本波未做**，因为**不占卡**：容器 `cann910b-h` 无 davinci 设备、且未装 `torch_npu`（实测 `ModuleNotFoundError: No module named 'torch_npu'`）。
- 已证：**编译面**（容器 `--target ascend --bwd --no-run` 产 `gdnb[ascend]|h2k8v8|c2ubz`，rc=0）+ **数值面**（host cpu-alias `--target cpu --bwd` 六项 rel 1e-7 量级、`adjoint=6.22e-10`）。
- 待第三窗一条命令补齐：`cd /work/ascend/port910b && python3 run_delta_card.py --target ascend --bwd` ⇒ 出 `NUM-delta_bwd PASS dq=… dk=… …`。
- 六项梯度**逐项都已数值闭环**（host）；唯一未闭环的是"**同一判决在真机上再出一次**"这一执行动作本身，原因即上述"不占卡"。
