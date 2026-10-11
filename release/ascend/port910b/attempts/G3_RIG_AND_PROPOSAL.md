# G3_RIG_AND_PROPOSAL.md — p2-13 执行代理 G3（两件小活：工装修缮 + 双因子提案）

- 执行代理：**G3**（p2-13，不占卡）
- 容器：`cann910b-h`（CANN 8.5.0，dav-2201；`/work`＝`release/ascend` bind mount）
- host `.venv`：`/Users/pengweiye/Documents/codes/system-one/release/.venv/bin/python`
- 基线：tilelang 仓 `994b44eca1a83a00d19d926e5264a6c608700656`；trunk `port910b_compat.h` md5 `ca023c2e01746ca52c87f36bcea93ee9`
- 交付物：
  - **(a) 改**：`attempts/D2_probe_legs.py`（唯一产品面，仅拆进程 + 判决行装配）
  - **(b) 新**：`attempts/S3_compat_proposal_v2.diff`（双因子提案，**不落地**）
  - **(b) 新**：`attempts/G3_probe_compile_v2.py`（编译面自证探针，`--inc/--dump/--tag`）
- 禁令遵从：未改 trunk/`patches/port910b_compat.h` 本体，未改 `D2_variants.md`/`reconcile/S3_*`/`run_delta_card.py`/`gradcheck_card.py`/`gdn_conv_asc.py`/`autograd_asc.py`/`run_numerics.py`/`compile_all_asc.py`/`card_run.sh`/`window3_card.sh`/`runs/**`/任何 `tasks.md`；未用 git。

---

## (a) 修「同进程发射桩传染」工装缺陷

### 归因（根因）

窗三 D 段现场（`runs/1012-p2-13-oncard-window3-pin/artifacts/w3_d2legs.log`）：

```
## inprocess combo=default flags=（默认 OFF） … run=True legs=compile_all,gemm_l1_intrinsic,dw_intrinsic
NUM-gemm_l1_intrinsic-FAIL _LaunchSkip:
NUM-dw_intrinsic-FAIL _LaunchSkip:
```

根因链（一案一进程纪律被破坏）：

1. `D2_probe_legs.py` 旧实现 `spawn()` 把**一个 combo 的所有腿**（`compile_all` + 两条 intrinsic）
   塞进**同一个** `exec` 子进程里顺序跑；
2. `compile_all_asc.py::main()`（`:438` 调 `_install_stubs()`）会把全局
   `tilelang.compile` 换成一个**调用即抛 `_LaunchSkip`** 的替身（`_install_stubs` 定义 `:104`，
   替换赋值 `:128`），而 `_restore_stubs()` **只在 `--products` 分支**才执行（`:452`，紧邻 `run_products()`）
   ⇒ 非 `--products` 时桩**不还原**；
3. 于是紧随其后的 intrinsic 腿在发射时被前一腿留下的桩截断，现场即 `NUM-*-FAIL _LaunchSkip:`；
4. `+A`/`+C` 两趟因 legs 不含 `compile_all` 而真执行、出 `507015`（有效）——反证是"同进程传染"而非设备问题。

⇒ **该趟（default combo）颜色不可用**，但机械面 `so=BASE` 仍有效（`so=` 由缓存逐核指纹算，与发射无关）。
这是 A2/窗二同款问题重演；纪律＝**一案一进程**。

### 改动（`attempts/D2_probe_legs.py`，仅拆进程 + 判决行装配）

- `spawn(combo, cache, do_run, extra_d, log_dir)`
  → **`spawn_leg(combo, leg, cache, do_run, extra_d, log_dir)`**：只起**单腿**子进程，
  `--legs <单腿>`；日志拆到 `<log-dir>/leg_<combo>_<leg>.log`；父按腿回显 + 打
  `[<combo>/<leg>] child-rc=… log=… bytes=…`。
- `main()` combos 循环：由"每 combo 一子进程"改为**每腿一子进程**，逐腿取回原文后
  `text = "".join(texts)` 聚合（**聚合后原文口径与改前一致**，供既有三判据检查）；
  `rc = 首个非零腿退码`。
- **契约一字不动**：`inject`/`so=`/`NUM-` **三判决行**、`D2-LEG-OK|INVALID` 语义、
  退出码规则（有任一 INVALID ⇒ 退 1）、`--card` 语义、`SEC12_LEGS`/`COMBO_FLAGS`/指纹判读口径不变。
- `compile_all` 腿与 intrinsic 腿**永不**同进程（`compile_all` 装的桩随其进程死亡，不再传染）。

> 一句话：**破坏"同进程"这一唯一传染通道，其余判据逻辑一行未动。**

### 自证（容器 `cann910b-h`，compat=`ca023c2e`，`/dev/shm` 缓存）

命令（无卡趟）：

```sh
source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null 2>&1
P910B_REPO=/work/ascend python3 /work/ascend/port910b/attempts/D2_probe_legs.py \
    --combos default,A,C --cache-root /dev/shm/g3_d2
```

真实输出（节选）：

```
== D2-PRECHECK P910B_REPO=/work/ascend run=False cache_root=/dev/shm/g3_d2
D2-PRECHECK-OK runner=compile_all+gemm_l1_intrinsic+dw_intrinsic 全部在场
D2-CHILD-START combo=default cache=/dev/shm/g3_d2/d2_legs_default run=False legs=compile_all
   [default/compile_all] child-rc=0 …
D2-CHILD-START combo=default … run=False legs=gemm_l1_intrinsic
NUM-gemm_l1_intrinsic-PASS rel=n/a (compile-only) track=performance
   [default/gemm_l1_intrinsic] child-rc=0 …
D2-CHILD-START combo=default … run=False legs=dw_intrinsic
NUM-dw_intrinsic-PASS rel=n/a (compile-only) track=performance
   [default/dw_intrinsic] child-rc=0 …
   [default] spawn rc=0
…
[default] SO-VERDICT so=BASE  （… asc_fp=2e302e446203ddce so_fp=22b15022ccfd06a9）
[default] LEG-VERDICT D2-LEG-OK（三判决行齐：inject+so+num）
[A] SO-VERDICT so=DIFF  （机器码变了：dW_l0tr_kernel#0,main_kernel#0；…）
[A] LEG-VERDICT D2-LEG-OK（三判决行齐：inject+so+num）
[C] SO-VERDICT so=DIFF  （机器码变了：main_kernel#0；…）
[C] LEG-VERDICT D2-LEG-OK（三判决行齐：inject+so+num）
D2-LEGS-SUMMARY valid=3/3 combos=default,A,C invalid=（无）
```

判决（无卡趟）——**通过**：

| 检查 | 命令 | 结果 |
|---|---|---|
| `valid` | `grep D2-LEGS-SUMMARY …` | **`valid=3/3`**，`invalid=（无）` |
| 无 `_LaunchSkip` | `grep -c "_LaunchSkip" /dev/shm/g3_d2_run3.log` | **`0`** |
| 每腿独立进程 | `grep -c "D2-CHILD-START"` | **`7`**（default 3 腿 + A 2 腿 + C 2 腿） |
| 退码 | `echo $?` | **`rc=0`**（无 INVALID ⇒ 不变） |
| default 腿 intrinsic 出**编译**判决行 | `grep "NUM-gemm_l1_intrinsic"` | `NUM-gemm_l1_intrinsic-PASS rel=n/a (compile-only)`（不再是 `_LaunchSkip`） |

九数回归（default 趟，`asc .o` 字节数）——**逐字同**：

```sh
grep -nE "2850|2250|3625|2106|2911|4003|1864|922|2842" /dev/shm/g3_d2_run3.log
```

```
ASC-gemm PASS .o~2850B …           ASC-gdn PASS .o~4003B …
ASC-gemm_bwd_dw PASS .o~2250B …    ASC-gdn_conv PASS .o~1864B …
ASC-add_ln PASS .o~3625B …         ASC-letter_readout PASS .o~922B …
ASC-rope PASS .o~2106B …           ASC-lora PASS .o~2842B …
ASC-attn_sw PASS .o~2911B …
```

⇒ `2850 / 2250 / 3625 / 2106 / 2911 / 4003 / 1864 / 922 / 2842` 九数**逐字同**。

**负例**（`--card` 形态但无设备，`cann910b-h` 无 NPU）——**明码 FAIL、非静默**：

```sh
P910B_REPO=/work/ascend python3 …/D2_probe_legs.py --combos default,A,C --card --cache-root /dev/shm/g3_d2card
```

```
NUM-gemm_l1_intrinsic-FAIL RuntimeError: Expected one of cpu, cuda, … device type at start of device string: npu
   [default/gemm_l1_intrinsic] child-rc=1 …
NUM-dw_intrinsic-FAIL AttributeError: 'Tensor' object has no attribute 'npu'
   [default/dw_intrinsic] child-rc=1 …
   [default] spawn rc=1
…
   [_A_] spawn rc=1        [C] spawn rc=1
[default] LEG-VERDICT D2-LEG-OK（三判决行齐：inject+so+num）
D2-LEGS-SUMMARY valid=3/3 combos=default,A,C invalid=（无）
```

⇒ 每条 intrinsic 腿**明码 `NUM-*-FAIL` + 真实异常原文**（`RuntimeError … device string: npu` /
`AttributeError … 'npu'`），`child-rc=1`、`spawn rc=1`，**`_LaunchSkip` 计数=0**；
三判决行仍在（`LEG-VERDICT D2-LEG-OK`）、`valid=3/3`、进程退码 `rc=0`（退出码规则不变）。
**FAIL 响亮、不静默**（对比旧现场是 `_LaunchSkip` 伪装成"执行过"）。

### 下窗用法

- 有卡：`P910B_REPO=<卡上 release/ascend> python3 D2_probe_legs.py --combos default,A,C --card --baseline <基线缓存>`。
- 该修仅改**工装**：下窗 default 趟 intrinsic 腿被 `--card` 放行后会**真执行**（有卡时出 rel 数值），
  不再被 `compile_all` 的桩截断；机械面 `so=` 与数值面 `NUM-` 从此同趟可信。

---

## (b) 跨步装填修复：单因子 → **双因子** 提案 v2

### 归因（窗三 E 段实测 ⇒ 根因升为双因子）

（来源：`runs/1012-p2-13-oncard-window3-pin/artifacts/w3_s3.log`）

```
k=48/96 × none/relu（dma=strided）→ STALE-DIFF 与 none 不同的 prefill = ['zero','perm']  ⇒ H-b 真
k=32 × none/relu（dma=single）    → prefill = 无                                      ⇒ H-b 未触发
所有候选 mm+nn>0（含 sym=0 最优仍>0）                                 ⇒ 非单一几何
STALE-DIFF>0                                                          ⇒ 须同时处理 UB 预清零/落趟拆解
```

⇒ 根因**双因子、正交**：

- **因子一（几何）**：codegen 把**行步长(pitch)**当 gap 传给 910B 原生件 ⇒ 多突发跨步落点错。
  （v1 已修：`gap = pitch − blockLen`，SINGLE 恒等。）
- **因子二（状态）**：UB 目的区**未清零**，DMA 未写的空洞保留上次发射残留（`STALE-DIFF` 随 prefill 变）⇒ **只修几何不够**。

E 段 `STALE-DIFF>0` **即便挑到最优几何仍复现** ⇒ v1（单因子）**不充分** ⇒ 升 v2。

### 改动（`attempts/S3_compat_proposal_v2.diff` = v1 原文保留 + 因子二）

目标文件 `src/tl_templates/ascend/port910b_compat.h`（**不落地，仅提案**；应用：`patch -p1 < …`）。
两个 hunk：

1. `@@ -604,9 +604,26 @@`（`asc_copy_gm2ub_align`，GM→UB 装填 = k=48 病档路径）：
   - 因子一（v1 原文）：`srcGapB/dstGapB = pitch − blockLen`（<0 夹 0），DMA 第 6/7 参改传 gap；
   - **因子二（新增）**：融合 DMA **之前**把本趟**整个目的跨度**预清零：
     ```cpp
     if (blockCount > 1) {
       const int spanB = (blockCount - 1) * dstStrideB + blockLenB;
       const uint16_t zrep = (uint16_t)((spanB + 255) / 256);
       __cce_scalar::vector_dup((__ubuf__ float *)dst, 0.0f, zrep, 1, 1, 8, 0);
     }
     ```
     ⇒ DMA 未写的空洞＝**确定的 0**，而非 stale。`blockCount==1`（SINGLE）**跳过** ⇒ 与旧码逐字节等价（回归安全）。
2. `@@ -817,9 +834,14 @@`（`asc_copy_ub2gm_align`，UB→GM 写回）：因子一同款 gap 归一（**同类潜在缺陷一并收口**；
   k=48 内核未触达，出口是标量 `write_gm_bypass_dcache`。需与 gm2ub 同等上卡验证后才可断言）。

**因子二 两方案（都给出，标清取舍与代价）**：

| 方案 | 做法 | 优点 | 代价 | 本 diff |
|---|---|---|---|---|
| **A（采用）** | UB 目标区预清零（`vector_dup` 整段清零） | **保留融合 DMA 快路径**（MTE2 指令数不变）；SINGLE 跳过⇒无回归；硬件即便不忠实实现 gap，空洞也退化为确定 0 | 每条跨步装填多 1 组 V 面填充（span/256B repeat）；需保证 V 面在 DMA(MTE2) 前可见（现有 `__cce_pipe_barrier(PIPE_V)` 序列化已覆盖，仍应上卡确认） | ✅ 内联 |
| **B（备选，未内联）** | 落趟拆解：多突发融合 DMA 拆成 `blockCount` 条**单发连续** DMA（每条 blockCount=1） | **语义无关的稳妥解**：blockCount=1 ⇒ stride/gap 槽**根本不参与**（几何因子消解），每块写满⇒无空洞（状态因子亦消解），可一石解双因子 | `bn=32` ⇒ 32 条 MTE2（指令数 ×blockCount，性能需评估/由上层流水吸收）；**使因子一 gap 归一在该路径成死码** | ⬜ 备选 |

选 **A** 的理由：在 v1 之上"**增加**"而非推翻 v1 的融合 DMA 与 gap 归一，改动最小、对 SINGLE 无回归；
B 会**替换**（而非叠加）v1 的 DMA，故列为备选，仅在 A 不足时退用。

### 自证

**① `patch -p1 --dry-run` 通过**（tilelang 仓 `994b44ec`）：

```sh
$ cd /Users/pengweiye/Documents/codes/tilelang
$ git rev-parse HEAD            → 994b44eca1a83a00d19d926e5264a6c608700656
$ md5 src/tl_templates/ascend/port910b_compat.h
                                → ca023c2e01746ca52c87f36bcea93ee9
$ patch -p1 --dry-run < …/attempts/S3_compat_proposal_v2.diff
patching file 'src/tl_templates/ascend/port910b_compat.h'
rc=0
```

**② 容器内应用到临时副本 + 编译最小跨步件 + 代码级对比**（副本头放 `/dev/shm/g3_v1`、`/dev/shm/g3_v2`，
`-I` 前置覆盖，**从不改 install 的 trunk 头**）：

```sh
python3 G3_probe_compile_v2.py --inc /dev/shm/g3_v2 --tag S3-V2
python3 G3_probe_compile_v2.py --inc /dev/shm/g3_v1 --tag S3-V1
python3 G3_probe_compile_v2.py --tag S3-TRUNK
```

```
S3-V2-COMPILE-HEADER /dev/shm/g3_v2/tl_templates/ascend/port910b_compat.h md5=dabd4751ea6561ba6a583f66085ea54f
S3-V2-COMPILE PASS inc=/dev/shm/g3_v2
S3-V1-COMPILE-HEADER /dev/shm/g3_v1/tl_templates/ascend/port910b_compat.h md5=6d93fcfec49c836bfddf75dca1d48542
S3-V1-COMPILE PASS inc=/dev/shm/g3_v1
S3-TRUNK-COMPILE-HEADER …/src/tl_templates/ascend/port910b_compat.h md5=ca023c2e01746ca52c87f36bcea93ee9
S3-TRUNK-COMPILE PASS inc=<install>
```

**代码级对比**（`bisheng -E` 预处理原文，即编译器**真正**看到的码；`grep` 原文）：

```
# ── 因子一：DMA 实参（copy_gm_to_ubuf 第 6/7 参）──
TRUNK:   … (uint16_t)blockCount, ((uint16_t)(((int)(blockLenB)) / 32)),
             ((uint16_t)(((int)(srcStrideB)) / 32)), ((uint16_t)(((int)(dstStrideB)) / 32)));
V1:      … (uint16_t)blockCount, ((uint16_t)(((int)(blockLenB)) / 32)),
             ((uint16_t)(((int)(srcGapB)) / 32)), ((uint16_t)(((int)(dstGapB)) / 32)));
V2:      … (uint16_t)blockCount, ((uint16_t)(((int)(blockLenB)) / 32)),
             ((uint16_t)(((int)(srcGapB)) / 32)), ((uint16_t)(((int)(dstGapB)) / 32)));
```

```
# ── 因子二：UB 预清零（V2 新增；V1/TRUNK 无）──
V2:  3 处  __cce_scalar::vector_dup((__attribute__((cce_unif_buff)) float *)dst, 0.0f, zrep, 1, 1, 8, 0);
V1:  0 处（`grep -c "vector_dup(("`）
TRUNK: 0 处
V2 片： if (blockCount > 1) {
          const uint16_t zrep = (uint16_t)((spanB + 255) / 256);
          __cce_scalar::vector_dup((…float *)dst, 0.0f, zrep, 1, 1, 8, 0);
        }
        __cce_scalar::copy_gm_to_ubuf(… srcGapB/dstGapB …);
```

```
# ── 因子一（第二 hunk，ub2gm 写回）──
TRUNK: copy_ubuf_to_gm(dst, src, 0, burst_num, (uint16_t)(burst_len >> 5),
                          (uint16_t)(burst_src_stride >> 5), (uint16_t)(burst_dst_stride >> 5));
V1/V2: copy_ubuf_to_gm(dst, src, 0, burst_num, (uint16_t)(burst_len >> 5),
                          (uint16_t)(srcGapB >> 5), (uint16_t)(dstGapB >> 5));
```

⇒ **两处改动都进了代码**：因子一（`srcGapB/dstGapB`，hunk1+hunk2）与因子二（`vector_dup` 预清零，hunk1）。

**③ trunk md5 复验未变**（`ca023c2e01746ca52c87f36bcea93ee9`，三处）：

```sh
[1] tilelang 仓  ： md5 src/tl_templates/ascend/port910b_compat.h              → ca023c2e01746ca52c87f36bcea93ee9
[2] patches/     ： md5 release/ascend/port910b/patches/port910b_compat.h       → ca023c2e01746ca52c87f36bcea93ee9
[3] 容器 pip     ： md5sum …/dist-packages/tilelang/src/tl_templates/ascend/port910b_compat.h → ca023c2e01746ca52c87f36bcea93ee9
```

### 下窗用法 + 诚实边界

**诚实边界（不得写成已修）**：v2 属**提案**——本代理**不占卡**，**真机判决未做**；仅证 **可编译** + **patch 可应用**。
判"v2 是否把 STALE-DIFF 归零"的**探针**：`release/ascend/kernels/reconcile/S3_probe_nextshot.py`
（任务原文简写作 `reconcile/S3_*`；`--dev npu --prefills none,zero,perm`，其中 `--prefills` 即该探针默认值）
（在 v2 打上后复跑）：

- `A=0 且 U=0 且 STALE-DIFF=0` ⇒ 双因子都归零、**v2 充分**；
- `A=0 但 STALE-DIFF>0` ⇒ 因子二方案 A 未奏效 ⇒ **退方案 B（落趟拆解）**。

**⚠ dbeta 与跨步装填不是同一面**：D3 段已证 `gdnb` **无 DMA**（`w3_deltabwd.log`：
`NUM-delta_bwd FAIL first=dbeta rel=5.997e-01 absdiff=4.435e+00 thresh=2e-04`）⇒ 本提案**不**声称顺带修 dbeta，
**不要把两者混为一个收益**（dbeta 属标量面，另案处理）。
