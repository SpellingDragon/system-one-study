# M2 — P1-1p fp16 训练步反向 NaN：生成码取证 + 本地复现 + 修复候选

> 命题：`release/runs/1011-p2-13-oncard-accept-2002/notes.md` §②-2 的 fp16 反向 NaN。本窗**不占卡**，
> 只用归档真机产物 `artifacts/sys1_kernels.tgz` 离线取证 + host 复现 + 给可 host 验的修复候选。
> 一手事实（notes.md:24-26、`H_case2.log`）不推翻，只补充与**修正其中一条待验证猜测**（§2.2）。

## 0. 结论摘要

| 问 | 判决 | 凭据（可复跑命令见 §1） |
|---|---|---|
| 任务书预设「内核内 fp16 累加」是否成立 | **不成立（否证）** | 归档 9 枚含 `__gm__ half` 的件，归约面普查 `accum=fp32 / NO-REDUCE`、`fp16_reduce=NONE`、`violations=0` |
| notes.md:26 的「嫌疑 `add_ln.backward`／`rsqrt(≤0)`」 | **反证（正面排除，非证据缺口）** | LN 家族 3 件签名全 `__gm__ float*`；`rs = rsqrtf(var + 1e-5)`（eps 在根号内、除数是编译期常量）；`add_ln_asc.py:335-337/407-408` 结构上不产 fp16 LN 件 |
| 归档里 fp16 档有没有「证照量不到的那一层」的直接证据 | **有，且只有一处** | `ea1059e6…/device_kernel.asc:31` = 全部 9 枚 fp16 件里**唯一**一条多突发跨步 GM→UB DMA（`32×64B 跨距192B`）；其余 6 条 gm2ub 全 `1×N 连续`，2 件 `copies=0` |
| NaN 起点是否与训练步反向吻合 | **吻合（同位、同形、同序号）** | `H_case2.log:19` OP016 `in nan=1056`；host 同模型同位宽跑出的整步**唯一**一次落跨步档的调用正是反向 dA（`M2-E2E` OP 031，A=(16,96) W=(32,96) fp16） |
| 独立旁证（不同位宽、同 DMA 三元组） | **有** | `5d732a8f…`（fp32 n96k48）与 `ea1059e6…` **同一组参数** `32, 64, …, 192, 64`；前者在 notes.md:20 已被真机判错值 `rel=1.150e+00 / got=15.24771 ref=-15.37733` |
| host 能否复现 nan | **不能（如实 FAIL）** | `M2-E2E … nan_total=0 inf_total=0 grad_nan=0 ⇒ NOT-REPRODUCED`（改前/改后各一趟）；免疫原因两条，见 §4 |
| 修复候选（已落地、host 可验） | **可验面全绿** | plan 侧坏档 15→0；容器真编译 8/8 全 `SINGLE`；pytest 340 passed；cpu 9/9；C2 矩阵 18/18；e2e 数值逐字同改前 |
| 最终真机判决 | **未做（留下一次短窗）** | 需 fp16 训练步发射判决 + §6 三条补采 |

**一句话机制**：不是"位宽错了"，是**位宽减半把 W 块装填的突发宽度从 128B 压到 64B**，从"卡上证过清白的大突发跨步档"掉进"卡上实测出过错的小突发跨步档"；而这条档只在 `bk < k` 时出现，fp16 训练步里恰好只有**反向 dA 那一枚**命中它。

---

## 1. 取证范围与命令（每个数字都可复跑）

```bash
# 解包（422 条目 / 46 枚内核 / size=552236B）
mkdir -p /tmp/m2tgz && tar xzf release/runs/1011-p2-13-oncard-accept-2002/artifacts/sys1_kernels.tgz -C /tmp/m2tgz
cd /Users/pengweiye/Documents/codes/system-one/release        # 以下皆用 .venv（python 3.10, torch 2.14.1, tilelang 0.1.15）
V=.venv/bin/python
# ① fp16 件普查（位宽面 / 归约累加变量声明 / UB 申报对账）
$V ascend/kernels/reconcile/M2_probe_gen_index.py --root /tmp/m2tgz --dt half
# ② DMA 档普查（归档侧 + plan 侧）
$V ascend/kernels/reconcile/M2_probe_dma_cfg.py --root /tmp/m2tgz --only-half --no-plan
$V ascend/kernels/reconcile/M2_probe_dma_cfg.py --root /tmp/m2tgz \
   --shapes "32x32,96x32,32x48,96x48,32x80,96x80,32x96,96x96,32x112,96x160,64x48" --dtypes fp32,fp16
# ③ host fp16 训练步复现（cpu-alias 轨，target 仍传 ascend）
TILELANG_CACHE_DIR=$(mktemp -d) $V ascend/kernels/reconcile/M2_probe_fp16_ascend_e2e.py --tag post-fix
# ④ 容器（真 ascend 编译面）生成码直读
docker exec cann910b-i bash -lc 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \
  export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201; \
  cd /work && TILELANG_CACHE_DIR=$(mktemp -d) python3 ascend/kernels/reconcile/M2_probe_card_fp16_dma.py'
```

归档清点（真实输出）：`总件数: 46`；`grep -l "__gm__ half" */device_kernel.asc` → **9 枚**
（`0294e599 dw_impl`、`d85b9806 dw_impl`、`084263ae rope_impl`、`7e9cfdc1 dW_l0tr`、`74a119a0 main`、
`cb509fe6 main`、`8f30a21f gemm_impl`、`e07da648 gemm_impl`、`ea1059e6 gemm_impl`）。

> 偏离登记：本地实测 `md5 -q sys1_kernels.tgz` = `f2ba9a2080de73d1d04d4a309dc**8496e**`，
> notes.md:4 记的是 `…309dc**4920e**`。size/条目/内核数三项**逐字吻合**（552236B / 422 / 46），
> 故按"台账 md5 抄录差异"处理，未据此改变任何结论。

---

## 2. 取证正文（引归档原文，行号 + 表达式）

### 2.1 归约累加面：fp16 件普查 → **预设机制被否证**

`M2_probe_gen_index.py --root /tmp/m2tgz --dt half` 逐件真实输出（截三列）：

```
M2-GEN 8f30a21f8fd4 gemm_impl_kernel  gm=mixed(half+float) accum=fp32      fp16_reduce=NONE half_temps=NONE  ub=2048/6144=OK  violations=0
M2-GEN e07da648b2cc gemm_impl_kernel  gm=mixed(half+float) accum=fp32      fp16_reduce=NONE half_temps=NONE  ub=2048/6144=OK  violations=0
M2-GEN ea1059e6995f gemm_impl_kernel  gm=mixed(half+float) accum=fp32      fp16_reduce=NONE half_temps=NONE  ub=2048/6144=OK  violations=0
M2-GEN 0294e599371a dw_impl_kernel    gm=mixed(half+float) accum=fp32      fp16_reduce=NONE half_temps=NONE  ub=4096/8192=OK   violations=0
M2-GEN d85b9806a036 dw_impl_kernel    gm=mixed(half+float) accum=fp32      fp16_reduce=NONE half_temps=NONE  ub=4096/8192=OK   violations=0
M2-GEN 084263ae669c rope_impl_kernel  gm=mixed(half+float) accum=NO-REDUCE fp16_reduce=NONE half_temps=NONE  tiles=0           violations=0
M2-GEN 7e9cfdc18173 dW_l0tr_kernel    gm=mixed(half+float) accum=NO-REDUCE fp16_reduce=NONE half_temps=NONE  tiles=0           violations=0
M2-GEN 74a119a089ff main_kernel       gm=mixed(half+float) accum=fp32      fp16_reduce=NONE half_temps=NONE  ub=28672/28672=OK violations=0
M2-GEN cb509fe69f40 main_kernel       gm=mixed(half+float) accum=fp32      fp16_reduce=NONE half_temps=['half condval@10','half condval_1@21'] ub=53248/61440=OK violations=0
M2-GEN-INDEX-PASS violations=0
```

病档本体（fp16 dA）里的累加变量与归约，逐字（`ea1059e6…/device_kernel.asc`，共 59 行）：

```c
 4: extern "C" __global__ __vector__ void gemm_impl_kernel(__gm__ half* A, __gm__ half* C, __gm__ half* W, __gm__ float* bias, int32_t m) {
 7:   float acc = float(0x0p+0f);            // 累加器：fp32
 8:   float v;                               // 出口临时：fp32
15:     ((__ubuf__ float*)buf_dyn_shmem)[(((i*32)+j)+1024)] = float(0x0p+0f);   // c_ub 清零走标量置 0
23:       condval = ((float)A[min(...)*96 + ko*32 + kk]);                        // A 走标量旁路，当场升 fp32
40:         acc = (acc + (a_ub_f[i*32+kk] * ((float)w_ub_h[(j*32+kk)+1024])));   // 乘积先升 fp32 再累加
45:         c_ub[i_2,j_1] = (c_ub[i_2,j_1] + acc);                               // 跨 ko 的块间累加：fp32
53:         tl::write_gm_bypass_dcache((__gm__ half*)C + (my*512 + i_3*32 + j_2), (half)v);   // 只在交货处降位
```

⇒ **生产纪律「主干 fp16、归约全程 fp32」在生成码层面是守住了的**；`half` 只出现在 GM 指针、UB 的
W 块存储（DMA 不许顺带转类型 ⇒ 与源同宽）和最后一跳 `(half)v` 交货。**没有任何 `half` 面的归约**。
（`cb509fe6` 的两枚 `half condval` 是 gather 的**标量搬运临时**，下一行即 `= ((float)condval)` 写入 UB，
非累加器。）⇒ 交付物 4 预设的改法「归约累加变量显式 fp32」**无的放矢**，本窗换成 §5 的装填档改法。

UB 布局对账（同件）：`a_ub fp32[0,2048)` / `w_ub half[2048,4096)`（即 `buf_dyn_shmem[1024]` 起）/
`c_ub fp32[4096,6144)`；`host_kernel.c:435` 申报 `= (int64_t)6144` ⇒ **精确吻合，无越界**。
`_ub_bytes(bm,bn,bk,elem)=bm·bk·4 + bn·bk·elem + bm·bn·4`（§5 引入的预算式）在三枚归档件上**逐字节复现申报值**：

| 件 | 反推形状 | 公式算出 | host_kernel.c:435 申报 |
|---|---|---|---|
| `ea1059e6` | bm16 bn32 bk32 elem2 | 6144 | **6144** |
| `5d732a8f` | bm16 bn32 bk16 elem4 | 5120 | **5120** |
| `32d6c39c` | bm16 bn64 bk64 elem4 | 24576 | **24576** |

### 2.2 LN 家族：`rsqrt`／`var`／`eps` 写法 → **嫌疑正面排除**（且不是"缺件"）

`grep -l "add_ln_impl_kernel\|ln_bwd_impl_kernel" */device_kernel.asc` → **恰 3 枚**，签名全 fp32：

```
0c80628176bd ln_bwd_impl_kernel   sig_half=0     # extern "C" __global__ __vector__ void ln_bwd_impl_kernel(__gm__ float* DX, __gm__ float* DY, __gm__ float* Db, __gm__ float* Dg, __gm__ float* G, __gm__ float* H, int32_t rows)
bb1dc080fa8e add_ln_impl_kernel   sig_half=0
ec2b5ef907cd add_ln_impl_kernel   sig_half=0
```

`0c806281…`（**这就是 fp16 训练步实际发射的那枚 LN 反向件**，dim=32）逐字：

```c
 7:   float acc = float(0x0p+0f);
33:     acc = (acc + ((__ubuf__ float*)buf_dyn_shmem)[((i*32) + j)]);
35:     mu = (acc / float(0x1p+5f/*3.200000e+01*/));                 // ÷32，编译期常量，不可能 ÷0
39:     acc = (acc + (xh * xh));                                     // xh 是 __ubuf__ float（第 38 行视图）
41:     rs = rsqrtf(((acc / float(0x1p+5f)) + float(0x1.4f8b588e368f1p-17f/*1.000000e-05*/)));   // eps 在根号**内**
```

前向件 `ec2b5ef9…:37` 同形（÷64 + 1e-5）。该件两条 gm2ub（`:20`/`:22`）为
`1, (min(16, max(rows-blk*16,0))*128), …, 同值, 同值` ⇒ **单发连续 + 尾行夹址**，不属嫌疑档。

判读：① `rsqrtf` 的参数下界是 `0 + 1e-5 > 0`，**结构上产不出 `rsqrt(≤0)`**；② 除数是编译期常量，
**产不出 0/0**；③ 全程 `float`，**没有可降位的累加面**。⇒ notes.md:26 那句「更像 0/0 或 rsqrt(≤0)
灾难，嫌疑 `add_ln.backward`」作为**猜测**在本窗被反证；`inf=0` 也不再是指向 LN 的证据（跨步 DMA 出错
同样可以只表现为 NaN/错值而不产生 inf）。

**为何 fp16 档"没有 LN 反向件"不是证据缺口**：`add_ln_asc.py:335-337 / 360-362` 的 plan 硬要求
`X/Res/G/Bt`（及 `H/DY/Dg/Db`）`dtype == torch.float32`，`:407-408` 入口 `.float()` 物化后才交给内核
⇒ fp16 训练步**结构上不会**编出 fp16 LN 件。要补采的不是 LN，见 §6。

### 2.3 病档原文：`ea1059e6…/device_kernel.asc:31`（fp16 dA，bn=32 k=96 bk=32）

```c
31:   asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ half*)buf_dyn_shmem)[1024]))),
                          (__gm__ uint8_t*)((&(W[(ko * 32)]))),
                          32, 64, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 192, 64);
```

（原文一行，此处为可读性折行；`grep -n asc_copy_gm2ub_align` 只此一处。）

经 `port910b_compat.h:599 TL910B_BYTES_TO_BLK(x)=(uint16_t)((int)x/32)` 与 `:601-610` 十参→7 参换算
（单位＝32B 块，裁份见 `ascend/port910b/attempts/compat_gap_C.md` G-C4/G-C8、`X_FRACTAL_EVIDENCE.md`）：

| 十参实参 | blockCount | blockLen | srcStride | dstStride |
|---|---|---|---|---|
| 字节 | 32 次 | **64B** | **192B** | 64B |
| 32B 块 | 32 | **2** | **6** | **2** |

⇒ **32 突发 × 64B、跨距 192B 的多突发跨步装填**。`ko` 循环（`:18 for (int32_t ko = 0; ko < 3; ++ko)`）
每轮源起址再进 64B（`&W[ko*32]`，half ⇒ 64B），故奇数 `ko` 的突发起点落在 …+64B 而非 128B 边界。
（*候选机制，仅作算术级观察*：本窗**没有**硬件文档或 readout 判决支撑"910B MTE2 要求突发 128B 对齐"，
也不主张 pitch/gap 语义轴——那条轴 `U2_probe_k48_dma_semantics.py` 已写、尚未真机落判。§5 的 128B
门只是把两处实测坏档与四处实测好档**夹出来**的工程分界，不是硬件规范。）

### 2.4 对照件：同位宽/同三元组的四组对置（真实 grep 输出）

```
M2-DMA 8f30a21f8fd4 gemm_impl_kernel  gm2ub SINGLE        1×2048B 连续(dst_stride=2048) line=30 mult32=ok   # fp16 前向 n96k32
M2-DMA e07da648b2cc gemm_impl_kernel  gm2ub SINGLE        1×2048B 连续(dst_stride=2048) line=30 mult32=ok   # fp16 前向 n32k32
M2-DMA 74a119a089ff main_kernel       gm2ub SINGLE ×4（8192/16384/2048/2048B 连续）                            # fp16 融合件
M2-DMA ea1059e6995f gemm_impl_kernel  gm2ub STRIDED-SMALL 32×64B 跨距192B dst64B line=31 mult32=ok           # fp16 反向 dA ← 病档
M2-DMA-SUMMARY small=1 single=6 big=0 dyn=0 parse_miss=0 mult32_bad=NONE archived_small=['ea1059e6995f/gemm_impl_kernel']
```

`dw_impl`（fp16 ×2）、`rope_impl`（fp16）、`dW_l0tr`（fp16）、`main_kernel`（`cb509fe6`）**5 件 `gm2ub=0`**
（`for h in 0294e599 d85b9806 084263ae 7e9cfdc1 cb509fe6; do grep -c asc_copy_gm2ub_align $h*/device_kernel.asc; done`
⇒ 全 0；纯标量旁路，根本不发 MTE2 块搬运）⇒ 直接排除。⇒ **整个归档的 fp16 面只有一枚件用多突发跨步装填，而它正是 step0 反向的 dA。**

同三元组的**异位宽**件（`grep -n asc_copy_gm2ub_align 5d732a8f…/device_kernel.asc`）：

```c
31: ... ((__ubuf__ float*)buf_dyn_shmem)[0])), (__gm__ uint8_t*)((&(W[((int32_t)block_idx*1536) + (ko*16))]))), 32, 64, 0, 0, 0, …, 192, 64);
```

fp32 `n96k48`（bk=16 ⇒ 行宽 64B、跨距 192B）与病档**同一组 (32,64,…,192,64)**；它在 notes.md:20 的
真机判决是 **`m16n96k48` 关 relu 照样坏 `rel=1.150e+00`、`got=15.24771 ref=-15.37733`（错值/反号）**。
⇒ 这条跨步档**已有一次独立的真机错值前科**（fp32），本窗再拿到它的 fp16 现场（NaN）——两次都在
`blockLen=64B`。反过来，fp32 `n32k96/bk=32`（`a421afa3…:31` = `32, 128, …, 384, 128`，突发 128B）是
同窗 fp32 训练步 **PASS**（loss 1.082224→0.945310）里同样在跑的**多突发跨步大档** ⇒ 好/坏分界落在
`blockLen ∈ (64B, 128B]` 之外的一侧，**与位宽本身无关，只与突发宽度有关**——这解释了"为何只有 fp16 炸"。

### 2.5 时序旁证（推断，非实证）

`H_case2.log` 里 `[20:13:50] Warning: Python Z3 scheduler failed …` 夹在 OP015（step0 前向末个 op）
与 OP016（step1 首个 `linear`，`in nan=1056`）之间（`:18`/`:19` 行）⇒ 该窗内**有一次编译发生在 step0
前向结束之后**，与"反向首用 fp16 dA 件"的时间位置一致。仅记为时序吻合，不作因果证据。

---

## 3. 判读汇总：位宽 / 溢出 / 除零三类风险

| 风险类 | 判读 | 依据 |
|---|---|---|
| 归约位宽（任务书预设） | **无风险**（fp32 面已核） | §2.1 全件 `accum=fp32/NO-REDUCE`、`fp16_reduce=NONE`、`violations=0` |
| fp16 位宽溢出 | **排除** | 前向 absmax 峰 4.08 ≪ 65504（notes.md:24）；`inf=0` 全程 |
| LN 除零 / `rsqrt(≤0)` | **排除** | §2.2 `rsqrtf(var + 1e-5)`、除数编译期常量 |
| UB 越界/块重叠 | **排除** | §2.1 三件申报逐字节复现；`overlap=NONE`、`ub=…=OK` 全 46 件 |
| **MTE2 小突发跨步装填（新机制）** | **嫌疑坐实到"唯一一枚 fp16 件"** | §2.3/§2.4：`small=1`，且同三元组 fp32 件已有真机错值前科 |
| 32B 整除截断（`TL910B_BYTES_TO_BLK`） | **排除** | 病档三元组 64/192/64 全为 32 的倍数（`mult32_bad=NONE`，全表亦 NONE） |
| 静默 eager（plan 拒形状） | **不是本 NaN 的因，但是证照的洞** | §7 洞 2：`gemm_kernel.py:46-51`；host 跑出的 3 个 PLAN-REJECT 形状不产 NaN（eager 本身 fp32 累加） |

---

## 4. 交付物 2：本地复现 → **如实 FAIL（NOT-REPRODUCED）**

`M2_probe_fp16_ascend_e2e.py`：cpu-alias 双方言垫片（逐字同 C2/F2）+ `target="ascend"` 原样递进 ⇒
产品路由/键名/正文口径全真、位宽原生 fp16；每枚 gemm 调用入/出 nan-inf-absmax 逐 op 记账；步末
逐参数 `.grad` 审计（monkeypatch `train_step._build_params` 抄参数字典）。真实判决行（改后，日志
`/tmp/m2logs/e2e_fp16_post2.txt`）：

```
M2-E2E-SETUP track=cpu-alias target=ascend dtype=fp16 垫片=['T.Vector', 'T.alloc_shared->alloc_local']
M2-E2E-OPS calls=45（fwd 34 / dw 11） 带nan的调用=0 keys=13 blockers=无
M2-E2E post-fix-clean dtype=fp16 target=ascend nan_total=0 inf_total=0 grad_nan=0 loss0=0.910987 loss1=0.823871 reduced=True ⇒ NOT-REPRODUCED
M2-E2E-KEYS n32k96/n96k32 档 = ['ascend|gemm[ascend]|n32k96b16x32x96anonedfloat16', …]
```

**复现不了的结构性原因（钉死，不是"没跑到"）**：

1. **MTE2 跨步突发面在 host 产物里不存在。** cpu-alias 轨的 `T.copy` 走 c 后端 lowering，生成码里没有
   `asc_copy_gm2ub_align`（本窗病档机制的量测面），host 无论怎么喂 fp16 都喂不出那条档。
2. **`target=cpu` 轨再叠一层入口升宽**：`gemm_kernel.py:40-42` `to32` 分支先 `.to(torch.float32)`，
   连"fp16 原生入"这一层都不进。
3. 反向 dA 那一步（OP031，A=(16,96) W=(32,96) fp16）在 host 按 plan **会**落跨步档（改前
   `STRIDED-SMALL 3×64B 跨距192B`），但执行的是 c 后端正文 ⇒ 计算正确、零 nan。⇒ 这恰好把"差异必在
   ascend 正文与 cpu 正文的分岔处"**具体定位到那一条 DMA 调用**，而不是笼统的"内核不同"。

⇒ **NOT-REPRODUCED ≠ 该档已判好**。真机判决只能来自发射窗（§6）。

---

## 5. 交付物 4：修复候选（已落地，host/容器可验）

**落点**：`ascend/kernels/gemm_asc.py`（白名单内），md5 `a0e7f19e…` → **`6ef882df01a0ccd6058d6493fc9b795f`**，
`wc -l` 294 → **337**。落件器 `reconcile/M2_probe_patch_gemm.py`（`--revert` 可逐字还原，`--check` 报态：
`M2-PATCH-STATE POST-FIX md5=6ef882df`）。

**改的是什么**：只改 `plan()` 里 ascend 轨 **K 块宽 `bk` 的取法**——接口、I/O、正文算式、交货位宽
一行未动。

```python
_MIN_STRIDED_BURST_B = 128          # 两处实测坏档=64B；四处实测好档≥128B ⇒ 夹出的工程分界
_UB_BUDGET_B = 24576               # 归档中卡上实际出现过的最大申报 UB（32d6c39c）

def _ub_bytes(bm, bn, bk, elem):    # 与生成码 buf_dyn_shmem 分块逐字节对账（§2.1 表）
    return bm * bk * 4 + bn * bk * elem + bm * bn * 4

def _pick_bk_ascend(bm, bn, k, elem):
    if _ub_bytes(bm, bn, k, elem) <= _UB_BUDGET_B:
        return k                    # bk==k ⇒ copy.cc 的 plan_single_row ⇒ 单发连续（卡上证过清白）
    ladder = _largest_block(k)
    if ladder is not None and ladder * elem >= _MIN_STRIDED_BURST_B:
        return ladder               # 退一步也只允许够宽的跨步档（fp32 n32k96 那种 128B 档，同窗 PASS）
    return None                     # 两条都不满足 ⇒ 交 eager，不进坏档
```

`plan()` 内 `if name == ascend_env.TARGET_ASCEND: bk = _pick_bk_ascend(...)`（`bk is None ⇒ return None`）。
依据链（tilelang `src/ascend/op/copy.cc`）：`plan_single_row = (n_rows==1) || (src_stride==row_bytes &&
dst_stride==row_bytes)` ⇒ 只有 `bk==k` 才折叠成 `blockCount=1` 的整块连续突发；`bk<k` 必生多突发跨步。

**验收（全部本窗实跑）**：

| 判据 | 命令 | 真实输出 |
|---|---|---|
| plan 侧坏档清零 | §1 命令 ② | 改前 `STRIDED-SMALL` **15 档** → 改后 **0 档**（`grep -c STRIDED-SMALL plan_{pre,post}.txt`） |
| 真 ascend 生成码直读 | §1 命令 ④ | `M2-CARD-DMA-SUMMARY pass=8/8 strided_small=NONE fail=NONE compiled=8`；逐条全 `SINGLE`，如 `16x32x96/fp16 line=30 gm2ub:SINGLE 1×6144B 连续`、`16x96x48/fp16 … 1×3072B`、`16x32x96/fp32 … 1×12288B` |
| host fp16 e2e 不回归 | §1 命令 ③ | 45 调用零 nan；`loss0=0.910987 loss1=0.823871`；16 枚 `.grad` 全有限；与改前**逐字同值**，唯一差异 3 行（OP031 的档/键、`M2-E2E-KEYS`、tag） |
| 不污染 cpu 轨键名/语义 | `.venv/bin/python ascend/compile_all_asc.py --target cpu` | `汇总 target=cpu 接口腿 PASS=9 FAIL=0 —— 门禁失败数=0`（键名仍 `cpu|…`，无 `_ascend` 侧漂移） |
| 全量回归 | `.venv/bin/python -m pytest tests -q -m "not integration"` | **340 passed**, 3 skipped, 3 deselected in 38.20s |
| C2 dtype 矩阵不回归 | `reconcile/C2_probe_dtype_matrix.py` | `C2-SUMMARY fp16:9/9 fp32:9/9 总计 18/18` |
| 容器 compat 前提 | `md5sum …/patches/port910b_compat.h` 与主仓 `src/tl_templates/ascend/port910b_compat.h` | 均 `ca023c2e01746ca52c87f36bcea93ee9` ✓（`/work` 内 3 份同值） |
| fp16 编译判决（任务书指定判据） | `C2_probe_card_fp16.py`（容器复跑） | `C2-CARD-SUMMARY A=7/7 B(取证到手)=3/3` |
| host 侧该件按设计不参与 | `.venv/bin/python …M2_probe_card_fp16_dma.py` | `M2-CARD-SKIP 本机无 ascend 编译面` rc=3 |

**口径一致性说明（防误读）**：plan 侧模拟打印的 `1×192B` 是**每行**突发宽（`bk·elem`），容器直读
`1×6144B` 是折叠后**整块**突发（`bn 行 × 192B`）——`192 × 32 = 6144`，两数同义。

**残留风险（不粉饰）**：① `_UB_BUDGET_B=24576` 锚的是"归档里申报过的最大值"，改后新升档里最高为
fp32 `n32k96`＝**20480B**（< 24576，容器**编译面**过），但**发射面**是否真放得下要真机核一次；
② 128B 门是两点实测夹出的工程分界，硬件规范未证（§2.3 已标注）；③ fp32 `96x160` 仍保留
`STRIDED-BIG 5×128B`（设计内：UB 放不下 bk=160，退回够宽的跨步档）。

---

## 6. 证据缺口与下窗补采清单（按优先级）

| # | 缺什么 | 为什么本窗给不出 | 下窗怎么采 |
|---|---|---|---|
| G1 | **病档件 `ea1059e6` 的发射面数值判决**（fp16 dA 逐元素对拍） | 归档只有生成码 + 逐 op 计数，没有该件 elementwise 参考对照 | 一案一进程跑 `A=(16,96) W=(32,96)` fp16 的 `linear` 反向，`dA` 对拍 fp64 参考；同形 fp32 做对照 |
| G2 | **改后档（bk=k）的真机 A/B** | 容器只到编译面 | 同窗两次短跑：产品件改后跑 fp16 训练步（期望 `loss0=0.975374 → 有限 loss1`）；再 `M2_probe_patch_gemm.py --revert` 复现原 FAIL（反事实） |
| G3 | 发射面 UB（20480B 新档） | 编译面不含发射校验 | G2 顺带核：fp32 训练步若仍 PASS 则本项闭合 |
| G4 | 跨步档语义轴（pitch vs gap vs 忽略） | 本窗无卡、无 readout | `U2_probe_k48_dma_semantics.py`（reconcile 里已存在）落地判决；与 P1-1q 共用一次窗 |
| G5 | `manifest.json` 不含键名/形状 | 归档工装只记文件 sha256/size | 建议下次开窗在 `KERNELS-TGZ` 旁附 `key→hash` 对照表（不改产品件，只改采集） |

---

## 7. 证照可信度边界：为何 `DTYPE-TRACE`／`KERNEL-USED` 照绿却漏了这层

三条洞，**每条都可复跑/可指行**：

**洞 1｜逐 op 证照只包前向入口，反向的关键调用不走 wrapper。**
`H_case2.log` 的 wrapped ops = `add_ln,attn_sw,…,linear,lora_apply,readout,rope`（`:1` 行）。而
`autograd_asc.py:169` 的 dA 是 `gemm_kernel.forward(dC, W.t().contiguous(), None, "none", A.dtype, ctx.target)`
——**直调内核入口件，绕过被包的 `linear`**。⇒ "step0 前向 15 次调用零 NaN" 这句话的**覆盖面里根本没有真凶那一次**。
（旁证：改后 host e2e 的 45 次调用里，唯一落跨步档的是 **OP 031**，前 15 次没有。）

**洞 2｜plan 拒形状 ⇒ 静默 eager，不记 blocker、不动 `KERNEL-USED-PASS`。**
`gemm_kernel.py:46-51`：`spec is not None and run(...)` 不成立即 `return _eager(...)`，**无 blocker、无计数**。
本窗 host 同模型跑出 3 类 PLAN-REJECT（`W=(2,32)`、`W=(8,32)`、`W=(32,8)` ⇒ Wgb / lora_a / lora_b），
`M2-E2E-OPS` 报 `keys=13 blockers=无`——与卡上 `KERNEL-USED-PASS kernels=13 blockers=0 fallbacks=0` 同数，
说明卡上那 13 键**同样不含**这些形状。⇒ `blockers=0` 的字面真 ≠ "所有线性代数都过了内核"。
（这是"证照绿得比实际宽"的最便宜的一种洞，且与位宽无关。）

**洞 3｜`DTYPE-TRACE` 的可见面 = Python 张量位宽 + 语言层声明，不含生成码 DMA 形态与 UB 申报。**
`fp32_accum=true / fp16_acc_violations=0` 那一层**本身没错**（§2.1 独立普查证实归约确实全 fp32）。它漏的是
**"数据怎么搬进 UB"**：`blockCount / blockLen / srcStride / dstStride` 只在 `asc_copy_gm2ub_align` 的
实参里，任何 Python 层纪律钉都够不着；同理"申报 UB 6144B"只在 `host_kernel.c:435`。
⇒ **正确结论不是"DTYPE-TRACE 说谎"，而是"NaN 的因不在位宽轴，在装填轴"**——证照要绿得有意义，得按轴配。

### 可机器判的补强方案（本窗已实现，未进任何产品件；是否立项由决策人定）

| 闸 | 判什么（都可自动判、失败即 rc≠0） | 成本 | 落点建议 |
|---|---|---|---|
| **A. 生成码 DMA 形态断言** `reconcile/M2_probe_dma_cfg.py` | 归档/新生成码里每条 `asc_copy_*_align`：`STRIDED-SMALL`（多突发且突发 <128B）⇒ **FAIL**；突发/跨距非 32 倍数 ⇒ FAIL；plan 侧同判（`plan_small`） | 纯文本，秒级，不占卡 | 挂在 `KERNELS-TGZ` 采集之后做**静态后验闸**（`card_run.sh` 是禁改件，故本窗只给方案不动手） |
| **B. 累加器位宽 + UB 申报断言** `reconcile/M2_probe_gen_index.py` | 每件 `accum ∈ {fp32, NO-REDUCE}`、`fp16_reduce=NONE`、实写字节 ≤ `host_kernel.c` 申报 UB、UB 块不重叠 ⇒ `violations=0` 才 PASS（**这就是任务书说的"内核生成码里断言累加器位宽"**） | 纯文本，秒级 | 同上；另可进 CI（编译面跑，不需设备） |
| **C. 每键单发连续断言** `reconcile/M2_probe_card_fp16_dma.py` | 容器真编译，直读 `kern.get_kernel_source()`：每条 gm2ub 必须 `blockCount==1`（或 `src_stride==burst`）⇒ 否则 FAIL | 容器编译，分钟级 | 生产键表新增/改 tiling 时的**必过闸** |

三条共同点：**判的是生成码文本，不是 Python 层声明**，正好补洞 3；且都不加"必须"式注释、不改产品件
（§5 的补丁里每条注释都挂归档 hash + 行号，可复核）。

---

## 8. 本窗产物索引

取证/探针（新建）：`M2_probe_gen_index.py`、`M2_probe_dma_cfg.py`、`M2_probe_fp16_ascend_e2e.py`、
`M2_probe_card_fp16_dma.py`、`M2_probe_patch_gemm.py`；产品件改动：`ascend/kernels/gemm_asc.py`；
文档：本件 + `M2_dtype_matrix_ext.md`。日志：`/tmp/m2logs/{dma_pre,dma_post,plan_pre,plan_post,
e2e_fp16_pre,e2e_fp16_post2,card_dma_post2,gen_index_all,gen_half,c2_matrix_post}.txt`。
