# M4_UBPREZERO.md — P1-1zg / M4：GDN 族「首次发射 UB 未初始化读」修复落地

> ⚠️ **2026-10-12 窗七证伪并已回退**（fresh 未变、prefill 恶化）：窗七真机判决——fresh `dbeta rel=8.002e-01` 与窗四**逐字同**（修复未生效）；zero-prefill 档 `1.018e-07→4.606e-01` **恶化** ⇒ ub 设备路径非中性。**已由 N5 回退**（`gdn_asc.py` 复归 md5 `676c07936547e9d897d3b2366a0317e0`），M4 探针/文档保留作负结果。证据见 `runs/1012-p2-13-oncard-window4-pin/notes.md` §⑩ 与 `N5_WARMUP_DESIGN.md`。

> 执行代理：**M4**　|　变更：p2-13 / P1-1zg　|　日期窗：1012　|　**不占卡**（host+容器双证；真机三判据留下一枪）
> 一手事实（K4 定案）：`gdnb` fresh `dbeta rel=8.002e-01` 错；`prefill=zero` 后六项全对 `1.018e-07`。
> 本件结论一句话：**生成码层 read-first 机械审计＝5 模具全 NONE（机理未定位，依逃生口径）⇒ 落
> 「内核入口先写零、再进递推」的最小面修复；host 四腿逐字不变 + 容器编译键 `c2ubz` 不变 +
> 生成码 diff 仅新增清零。**

---

## ⓪ 状态锚点（可复跑）

| 项 | 值 | 复跑命令 |
|---|---|---|
| `gdn_asc.py` 改前 md5 | `676c07936547e9d897d3b2366a0317e0` | `md5 /tmp/gdn_asc_before.py` |
| `gdn_asc.py` 改后 md5 | `f3419883b2206667ed893f89d9276311` | `md5 ascend/kernels/gdn_asc.py` |
| 源码插入面 | `gdn_asc.py:386-403`（18 行：1 空行 + 注 7 + 码 10） | `diff /tmp/gdn_asc_before.py ascend/kernels/gdn_asc.py` → `385a386,403` |
| 生成码（gdnb-ub）行数 | 改前 101 → 改后 112（净 +11 语义行） | 见 §② |
| 容器编译键 | `ascend\|gdnb[ascend]\|h2k8v8\|c2ubz`（**不变**） | 见 §② |

---

## ① read-first 定位（任务书第 1 件）

### 1.1 手法（**机械**，非人工目测）
新探针 `ascend/kernels/reconcile/M4_probe_ubaudit.py`：真编后用 `get_kernel_source()`，
逐 UB 槽标出 `firstW`/`firstR`，回答"是否存在某槽**首访问=读**（read-first）"。

- 生成码把多处 `alloc_shared` 展平成**同一块** `buf_dyn_shmem` + **常量偏移**（缓冲名不再出现）
  ⇒ 按**偏移区间**归类（槽表由源码 `alloc_shared` 次序×shape 定）。
- 单处 `alloc_shared` 时 codegen **保留缓冲名**（如 convb 的 `dw_ub`）⇒ 探针双路覆盖
  （`_NAMED_DECL` 命名单缓冲 + `_ACC` 展平偏移）。**这是早期"名字法/初版偏移法"双双漏报的根因**。
- 写/读判据：紧跟 `=`（非 `==`）为写；**行内 RMW 按"读先于写"**（保守，宁可多报）。
- 覆盖五只模具：`gdnb-ub`（设备真跑档 `c2ubz`）/ `gdnb-pure` / `gdn-fwd-ub` /
  `gdnconvb-bwd` / `gdnconv-fwd-ubstage`。

复跑（容器 `cann910b-h`，release 挂 `/work`）：
```
docker exec cann910b-h bash -lc 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \
  export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201; \
  cd /work && TILELANG_CACHE_DIR=/dev/shm/m4 python3 ascend/kernels/reconcile/M4_probe_ubaudit.py'
```

### 1.2 结果：**五模具全 NONE**（改后 `gdnb_ub` 实跑输出，原样粘）

```
===== AUDIT gdnb_ub: src_lines=112 =====
  slot        firstW  firstR   #W   #R  verdict
  lam_ub          57      77    2    3  先写后读 OK
  dec_ub          45      76    2    4  先写后读 OK
  u_scr           50      95    2    2  先写后读 OK
  dp_scr          51      96    2    2  先写后读 OK
  k_ub            46      76    2    4  先写后读 OK
  q_ub            47      77    2    2  先写后读 OK
  v_scr           52      79    2    2  先写后读 OK
  do_scr          53      77    2    3  先写后读 OK
  => read-first slots: NONE
===== AUDIT gdn_fwd_ub: src_lines=59 =====
  st_ub 25/36 OK · k_ub 32/41 OK · q_ub 33/47 OK · dec_ub 31/40 OK   => read-first slots: NONE
===== AUDIT convb_bwd: src_lines=85 =====
  dw_ub 22/48 OK                                                     => read-first slots: NONE
===== AUDIT convf_fwd_ubstage: src_lines=41 =====
  w_ub 18/25 OK                                                      => read-first slots: NONE
===== M4-UBAUDIT SUMMARY =====
  gdnb_ub: read-first=NONE
  gdnb_pure: read-first=NONE      (pure 无 alloc_shared，0 槽，符合预期)
  gdn_fwd_ub: read-first=NONE
  convb_bwd: read-first=NONE
  convf_fwd_ubstage: read-first=NONE
M4-UBAUDIT-DONE
```

**改前（基线）同一探针**：`gdnb_ub` 101 行，八槽亦**全为先写后读**——
`lam_ub W46/R66 · dec_ub W53/R65 · u_scr W70/R84 · dp_scr W71/R85 · k_ub W54/R65 ·
q_ub W55/R66 · v_scr W58/R68 · do_scr W59/R66`。改后 `#W` 各 +1（入口零写），`firstW`
前移（如 `dec_ub` 53→45）＝修复落点可视。

### 1.3 判定（**如实报，不硬猜**）
> **机理未定位。** 生成码层（`.asc`，五模具）**确为先写后读、无任何 read-first 落点**，
> 而设备行为仍是 fresh-敏感（K4：`none=e609a6ab ≠ dirty=2a507f3e ≠ zero=45266aef`）。
> 即：**fresh 敏感的读数落在 `.asc` 之下**（编译/运行时：UB 物理页复用、dcache、
> 编译期常量池/暂存、或 AIV 侧同步窗口）——非我白名单内可静态定位的槽级面。
>
> 依任务书逃生口径：**"若 read-first 找不到 ⇒ 如实报『机理未定位，修复转预热方案』"**。

### 1.4 支撑该判定的两条旁证
1. **数学悖论**：`dbeta = Σ_j du·(v_scr[j] − p)`，其输入 `du/v_scr/p` 同时驱动
   `dq/dk/dv`（后者**不随 prefill 变**）；槽级若无 read-first，则 `dbeta` 本不应变。
   ⇒ fresh 敏感只能来自**非槽级面**（编译/运行时暂存），与 1.3 一致。
2. **dcache 假说已否证**：`write_gm_bypass_dcache`/`read_gm_bypass_dcache` 曾疑为 fresh 敏感源，
   但该路径若致错会同时污染 `dq/dk/dv`，实测仅 `dbeta` 变 ⇒ 否。

---

## ② 修复（任务书第 2 件）——最小面「内核入口先写零」

### 2.1 落点：`gdn_asc.py:386-403`（新增 18 行，源码 diff `385a386,403`）
```
>                     # ── 入口预清零（P1-1zg / M4）：所有 UB 槽**先写零、再进递推** ──
>                     # 修"首次发射读未初始化 UB"的 fresh 敏感（顺序探针 none≠dirty≠zero，
>                     # `w4_d3_ub.log`）。生成码静态审计（M4_probe_ubaudit.py：gdnb-ub/pure、
>                     # gdn-fwd-ub、conv 两件）八槽**皆先写后读**、无静态落点 ⇒ 读数疑落在 `.asc`
>                     # 之下（编译/运行时暂存面），故用"内核入口先写零"兜底，覆盖任何被下移/重排的
>                     # UB 首读。**语义中性**：这些槽随后都各自被显式写入，先写 0 不改任何数学结果
>                     # （host `pure` 腿不走此 body，数值逐字不变，见 M4_UBPREZERO.md）。
>                     if variant == "ub":
>                         for i in T.serial(dk):
>                             dec_ub[i] = T.float32(0.0)
>                             k_ub[i] = T.float32(0.0)
>                             q_ub[i] = T.float32(0.0)
>                         for j in T.serial(dv):
>                             u_scr[j] = T.float32(0.0)
>                             dp_scr[j] = T.float32(0.0)
>                             v_scr[j] = T.float32(0.0)
>                             do_scr[j] = T.float32(0.0)
```
位置：`for rr in T.serial(upc): h = core_id*upc+rr` 之后、**Λ 初值之前**，
故 7 个非 `lam` 槽都在 `tt` 递推前**先写 0**；`lam_ub` 由紧随其后的 Λ 初值入口清零
（`dstate=zero` 生产路径，`gdn_asc.py:406-412`）覆盖，未重复。

### 2.2 语义中性证明（两路）
- **数学**：这 7 槽在每个 token 迭代里都被**无条件显式写入**（`dec/k/q` @ 源码 L423-426、
  `v/do` @ L427-429；`u/dp` @ 趟 1），入口先写 0 不改变任何递推取值。
- **host 实证**：`pure` 形态不走此 `if variant=="ub"` body ⇒ host（`cpu`→`pure`）
  四腿数值**逐字不变**（见 §③）。

### 2.3 生成码 diff（改动即修复的落点证据）
**(a) 归一化 diff**（把 SSA 循环变量 `i_1→i`、`j_1→j`…

**纯新增、无删改**，合计 11 行（7 条零写 + 循环骨架）：
```
43a44,48
>   for (int32_t i = 0; i < 8; ++i) {
>     ((__ubuf__ float*)buf_dyn_shmem)[(i + 64)] = float(0x0p+0f);   // dec_ub
>     ((__ubuf__ float*)buf_dyn_shmem)[(i + 88)] = float(0x0p+0f);   // k_ub
>     ((__ubuf__ float*)buf_dyn_shmem)[(i + 96)] = float(0x0p+0f);   // q_ub
>   }
44a50,55
>     ((__ubuf__ float*)buf_dyn_shmem)[(j + 72)]  = float(0x0p+0f);  // u_scr
>     ((__ubuf__ float*)buf_dyn_shmem)[(j + 80)]  = float(0x0p+0f);  // dp_scr
>     ((__ubuf__ float*)buf_dyn_shmem)[(j + 104)] = float(0x0p+0f);  // v_scr
>     ((__ubuf__ float*)buf_dyn_shmem)[(j + 112)] = float(0x0p+0f);  // do_scr
>   }
>   for (int32_t j = 0; j < 8; ++j) {          ← 此行与 before 的 Λ 初值循环头逐字同、被 diff 对齐
```
（原始 diff 另含插入致 SSA 后缀 +1 的循环变量重命名 `i→i_1→i_2…`，**纯机械、无语义**；
归一化后即上式，只有 `>` 无 `<`。）

**(b) 生成码行数**：`gdnb_ub` 101 → 112 行。（b 与 (a) 一致：净 +11 语义行。）

### 2.4 编译键（形不变）
```
NUM-delta_bwd PASS rel=n/a (compile-only) key=ascend|gdnb[ascend]|h2k8v8|c2ubz
```
复跑：
```
docker exec cann910b-h bash -lc 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \
  export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201; \
  cd /work/ascend/port910b && TILELANG_CACHE_DIR=/dev/shm/m4key python3 run_delta_card.py --target ascend --bwd --no-run'
```
⇒ 键含 `gdnb[`（打假绿守卫通过），且 `c2ubz` **与改前逐字同**（缓存不因修复失效=面未触碰编译契约）。

---

## ③ host 四腿双证（任务书第 4 件之 host 侧）

驱动：`reconcile/M4_hostcheck.sh <tag>`（`before`/`after` 两份输出落 `/tmp/m4host_<tag>.*`）。

| 腿 | 命令 | before（改前） | after（改后） | 判定 |
|---|---|---|---|---|
| delta | `run_delta_card.py --target cpu --bwd` | `NUM-delta_bwd PASS dq=7.53e-08 dk=6.72e-08 dv=6.68e-08 dbeta=1.01e-07 dg=9.72e-08 dh0=6.18e-08 adjoint=6.22e-10 rel_max=1.01e-07` | **逐字同** | ✓ |
| grad | `gradcheck_card.py --target cpu` | `GRADCHECK-SUMMARY ops=8 pass=8 fail=0 invalid=0 kernel_used=8/8` | **逐字同** | ✓ |
| step | `train_step.py --target cpu` | `P2-STEP-cpu loss0=0.910992 loss1=0.823897 reduced=True params_grad_ok=True` | **逐字同** | ✓ |
| real | `train_step_real.py --target cpu --dry` | `P2-REAL-STEP target=cpu loss0=3.282502 loss1=3.276916 reduced=True params=48 grads_ok=48/48 swaps=4 native=515 mode=run` | **判决逐字同**（仅 `COUNTS p50_ms`/`P2-REAL-WALL sec` 计时差、RuntimeWarning 行号 724→742） | ✓ |

`ITEM` 六项（before=after）：`dq 7.531e-08 · dk 6.718e-08 · dv 6.676e-08 · dbeta 1.007e-07 ·
dg 9.716e-08 · dh0 6.176e-08 · adjoint 6.222e-10`（全 PASS）。**host 因 `cpu`→`pure` 路，
本就不走预清零 body ⇒ 数值逐字不变＝语义中性的最强作证。**

### 3.1 pytest 红线
```
cd <release> && ./.venv/bin/python -m pytest tests -q -m "not integration"
→ 340 passed, 3 skipped, 3 deselected, 2 warnings in 36.03s
```
（2 条 warning 均为"本机无 NPU 回退 torch eager"的**预期**提示；其一 `gdn_asc.py:742`
＝插入 18 行后行号位移，非新增告警。）

---

## ④ 统一假说覆盖面（任务书第 3 件）——`dbeta / gdn_conv / gdn_delta`

> 统一假说：三个 device-only 缺陷可能同属"首次发射读未初始化 UB"。

| 缺陷 | 载体 | 本次机械审计 | 处置 |
|---|---|---|---|
| `dbeta 0.80` | `gdn_asc.py` bwd ub | `gdnb_ub` 八槽全 write-first，**read-first=NONE** | **已修**（§②） |
| `gdn_conv 5.5e4` | `gdn_conv_asc.py` `conv_bwd_asc_impl` | `convb_bwd` `dw_ub` W22/R48 先写后读；源码 L259-261 **已入口 `db[i]=0`/`dw_ub[j]=0` 再累加** | **只审计报告、不修**（归 conv 件/L4b） |
| `gdn_delta 2.2e5` | `gdn_asc.py` fwd ub（`run_delta_card` 走 fwd） | `gdn_fwd_ub` 四槽 `st/k/q/dec` 全 write-first，**read-first=NONE** | fwd 未改（fwd 无 fresh 敏感一手事实；不扩面） |

**结论**：三者在**槽级生成码**上**均无 read-first**（同 §①.3 的"读数在 `.asc` 之下"）。
`gdn_conv_asc.py` 按白名单**未改**（发现其入口已显式清零，与 `gdn_asc` **非同一共享代码**，
故不触发"共享代码"例外）。`convf_fwd_ubstage` 的 `w_ub` 亦 write-first（审计项）。

---

## ⑤ 下窗（真机）一枪设计 —— 留给下一枪

修复已落（内核入口零写）。真机三判据（**不占卡**，下一枪执行）：

1. **主判据（fresh ITEM）**：真机 `--target ascend --bwd`（ub 档）**首次发射**跑，
   六项 ITEM 逐项判 `rel < 2e-4`。**期望**：`dbeta` 由 `8.002e-01` → `<2e-4`
   （其余五项本已对，须不回退）。命令形：`run_delta_card.py --target ascend --bwd`。
2. **顺序指纹判据**：复跑 D3 顺序探针三态 `none/dirty/zero`，**期望** `none` 指纹
   **收敛到 `zero` 指纹**（改前 `none=e609a6ab ≠ zero=45266aef`）；即 fresh 不再敏感。
3. **形态对照判据**：同枪跑 `ub` vs `pure`（`SYS1_*_VARIANT`），**期望** ub 与 pure 六项一致
   （改前 ub-fresh 指纹 `e609a6ab` == pure 指纹＝两者都错；清零后 ub 若转对、pure 仍错，
   则定位"pure 另有独立面"，如实记、不硬并）。

**若 1/2 未转绿**（即内核入口零写未修到真机 fresh 敏感）：按逃生口径转**次选修法**
（`autograd_asc.py` 侧"发射前预热清零/丢弃首跑"）——该面**须先向主 agent 申报并获事后追认**
（白名单要求），本件**未动** `autograd_asc.py`。

---

## ⑥ 白名单与边界自述
- **改动面**：仅 `ascend/kernels/gdn_asc.py`（+18 行，纯新增，§②）。
- **未改**：`gdn_conv_asc.py`（只审计）/`autograd_asc.py`（未选次选）/`run_delta_card.py`/
  `gradcheck_card.py`/`train_step*.py`/trunk/`runs/**`/`tasks.md`。**未用 git**。
- **本件新建**：`reconcile/M4_UBPREZERO.md`、`reconcile/M4_probe_ubaudit.py`、
  `reconcile/M4_hostcheck.sh`（后两者为 `M4_probe_*` 及配套驱动）。
