# M2 补档 — C2 dtype 矩阵的两处结构性缺口（K 非对齐档 + ascend 方言 fp16 轨）

> 补 `reconcile/C2_dtype_matrix.md` §2 主矩阵（9 行 × {fp32, fp16}，18/18 PASS）的两处**覆盖缺口**，
> 并给出"哪些格本地根本判不了"的清单。**不臆造判决**：判不了的格一律标 `NON-judgable(local)` 并注明缺什么。
> 取证主体在 `reconcile/M2_FP16_NAN_EVIDENCE.md`（下称《取证》）。

## 0. 为什么要补（两条缺口都是 P1-1p/P1-1q 的成因级盲区）

| 缺口 | C2 原矩阵的实况 | 后果 |
|---|---|---|
| **① 无 K 非对齐档** | §2 里 gemm/dw 的键只有 `n64k32`、`n16k32`、`n64k16`、`dW n64k32tc16b64x32` ⇒ K ∈ {16,32}，`bk` 恒等于 `k` | `bk<k` 的多突发跨步装填档**一格未覆盖**；而 `notes.md:19-22` 的 P1-1q（真变量是 K=48）与《取证》§2.3 的 P1-1p 病档，都只在 `bk<k` 时才存在 |
| **② ascend 方言 fp16 只判到"编不编得出"** | §2 是 host cpu-alias **数值**轨；§7 是容器**编译**轨（A=7/7，看签名） | **没有"生成码里装填形态"这一列** ⇒ 同位宽同形状下"单发连续 vs 跨步突发"的差异（本案唯一实测到的失败轴）在原矩阵里不可见 |

命令（本档所有数字）：

```bash
cd $REPO_ROOT/release && V=.venv/bin/python
# plan 侧档位（改后态；改前用 `reconcile/M2_probe_patch_gemm.py --revert` 再跑，跑完不带参数即 apply 还原）
$V ascend/kernels/reconcile/M2_probe_dma_cfg.py --root /nonexistent \
   --shapes "32x32,96x32,32x48,96x48,32x80,96x80,32x96,96x96,32x112,96x160,64x48" --dtypes fp32,fp16
# 容器真 ascend 编译 + 生成码直读（8 格）
docker exec cann910b-i bash -lc 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \
  export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201; \
  cd /work && TILELANG_CACHE_DIR=$(mktemp -d) python3 ascend/kernels/reconcile/M2_probe_card_fp16_dma.py'
# C2 原矩阵复跑（确认补档不回归）
TILELANG_CACHE_DIR=$(mktemp -d) $V ascend/kernels/reconcile/C2_probe_dtype_matrix.py
```

## 1. 四轨口径（每格只由它有权判的轨填）

| 轨 | 判什么 | 手段 | 位宽形态 | 能判数值吗 | 能判装填档吗 |
|---|---|---|---|---|---|
| **T1 host cpu 方言** | 语义/算式基线（`target=cpu`） | `compile_all_asc.py --target cpu`、`verify_*.py --cpu` | **入口 `.to(float32)`**（`gemm_kernel.py:40-42`）⇒ 结构上看不到 fp16 面 | 是（fp32 参考对拍） | **否**（无 ascend lowering） |
| **T2 host cpu-alias 数值轨** | 路由/键名/正文口径 + fp16 原生入的数值 | `C2_probe_dtype_matrix.py`、`M2_probe_fp16_ascend_e2e.py` | fp16 原生入（`target` 仍 `"ascend"`），正文真跑 | 是（c 后端执行） | **否**——`T.copy` 走 c 后端，产物里没有 `asc_copy_gm2ub_align`（《取证》§4 免疫原因 1） |
| **T3 容器真 ascend 编译轨** | 正文可否编 + **生成码装填档直读** | `C2_probe_card_fp16.py`（签名）、`M2_probe_card_fp16_dma.py`（逐条 `asc_copy_*_align` 分档） | fp16 原生（`__gm__ half* A/C/W`，bias 恒 `__gm__ float*`） | **否**（无设备 ⇒ 不发射） | **是**（本档唯一能判装填档的轨） |
| **T4 真机发射数值轨** | 错值/NaN 的最终判决 | `ascend/port910b/card_run.sh` + H 案归因 driver | 同 T3 | 是 | 是（间接：档错→值错） |

⇒ **本案的失败轴只在 T3/T4 上可见；T1/T2 天然绿。** 这就是 C2 原矩阵 18/18 全 PASS 却与 P1-1p 共存的原因。

## 2. 补档表 A：gemm 的 K 非对齐档 × 位宽（plan 侧 → 容器直读 → 真机）

`n×k` 记法同 `M2_probe_dma_cfg.py`（`BLOCK_M=16`，`BLOCK_LADDER=(64,32,16)`；`bn` 由 `n` 取块）。
> **读表口径（防误读）**：plan 侧那两列打印的是 `{nk}×{burst}B`，第一个数是 **ko 轮数 `nk=k/bk`**（`M2_probe_dma_cfg.py:163-166` 把 `nk` 当作 classify 的首参），而归档侧 `M2-DMA` 行的第一个数才是 codegen 真参数 **`blockCount`（=bn 行）**。两列的**分档结论**等价（判据只看 `src_stride==burst` 即 `bk==k`，与首参是否为 1 无关），但**数字不可互相引用**：病档 `32x96/fp16` plan 侧写 `3×64B`，归档件 `ea1059e6…:31` 写 `32, 64, …, 192, 64`（32 突发 ×64B）——同一件事的两根轴。
"改后档"＝当前产品件（`gemm_asc.py` md5 `6ef882df`）；"改前档"＝`--revert` 态（md5 `a0e7f19e`）。

| n×k | 位宽 | 改前档（plan 侧） | 改后档（plan 侧） | T3 容器真编译直读 | T4 真机数值 |
|---|---|---|---|---|---|
| 32x32 | fp32 | SINGLE 1×128B | SINGLE 1×128B | 未采（8 格含同形 **fp16**，不含 fp32） | `m16n32k16/relu 9.09e-08` **PASS**（notes.md:21，形为 k=16） |
| 32x32 | fp16 | SINGLE 1×64B | SINGLE 1×64B | `SINGLE`（8/8 之一） | NON-judgable(local) → 见 G1 |
| 96x32 | fp32 | SINGLE 1×128B | SINGLE 1×128B | — （同形已判） | `m16n96k32/relu 1.16e-07` **PASS**（notes.md:21） |
| 96x32 | fp16 | SINGLE 1×64B | SINGLE 1×64B | **PASS** 键 `ascend\|gemm[ascend]\|n96k32b16x32x32anonedfloat16` | 归档同形件 `e07da648`/`8f30a21f` 单发连续；H 案前向 15 op 零 NaN（**旁证，非逐元素判决**） |
| **32x48** | fp32 | **STRIDED-SMALL** 3×64B 跨距192B | SINGLE 1×192B | **PASS** `gm2ub:SINGLE`（键 `…n32k48b16x32x48…`） | NON-judgable(local) |
| **32x48** | fp16 | **STRIDED-SMALL** 3×32B 跨距96B | SINGLE 1×96B | **PASS** `SINGLE`（键 `…n32k48b16x32x48anonedfloat16`） | NON-judgable(local) |
| **96x48** | fp32 | **STRIDED-SMALL** 3×64B 跨距192B | SINGLE 1×192B | **PASS** `line=30 SINGLE 1×…`（键 `…n96k48b16x32x48anonedfloat32`） | 改前＝P1-1q 真机 FAIL：`rel=nan(96 个 NaN)`、关 relu `rel=1.150e+00 got=15.24771 ref=-15.37733`（notes.md:20，归档件 `5d732a8f…:31` 同三元组）；**改后档 NON-judgable(local) → G2** |
| **96x48** | fp16 | **STRIDED-SMALL** 3×32B 跨距96B | SINGLE 1×96B | **PASS** `SINGLE 1×3072B` | NON-judgable(local) |
| **32x80** | fp32 | **STRIDED-SMALL** 5×64B 跨距320B | SINGLE 1×320B | 未采（容器 8 格不含；plan 侧已判好档） | NON-judgable(local) |
| **32x80** | fp16 | **STRIDED-SMALL** 5×32B | SINGLE 1×160B | 未采 | NON-judgable(local) |
| **96x80** | fp32/fp16 | **STRIDED-SMALL** 5×64B / 5×32B | SINGLE 1×320B / 1×160B | 未采 | NON-judgable(local) |
| **32x96** | fp32 | **STRIDED-BIG** 3×128B 跨距384B（`bk=32` ⇒ 突发 128B，**不是**小档） | SINGLE 1×384B | **PASS** `SINGLE 1×12288B`（键 `…n32k96b16x32x96anonedfloat32`；UB 由 8192→**20480B**，见《取证》§5 残留风险） | 同形改前档（`a421afa3…:31` = `32,128,…,384,128`）**在同窗 fp32 训练步 PASS**（loss 1.082224→0.945310）⇒ 128B 突发跨步档"未致 NaN"（注意：仅"未致 NaN"，未做逐元素对拍） |
| **32x96** | fp16 | **STRIDED-SMALL** 3×64B 跨距192B（**P1-1p 病档本体**，归档 `ea1059e6…:31`） | SINGLE 1×192B | **PASS** `SINGLE 1×6144B`（键 `…n32k96b16x32x96anonedfloat16`） | 改前＝P1-1p 真机 FAIL（`H_case2.log:19` OP016 `in nan=1056`）；**改后档 NON-judgable(local) → G2（最关键的一格）** |
| **96x96** | fp16 | **STRIDED-SMALL** 3×64B | SINGLE 1×192B | 未采 | NON-judgable(local) |
| **32x112** | fp32/fp16 | **STRIDED-SMALL** 7×64B / 7×32B | **PLAN-REJECT ⇒ eager**（`bk=k` UB 超限、ladder burst <128B） | 未采 | 见 §4 覆盖代价 |
| **96x160** | fp32 | **STRIDED-BIG** 5×128B 跨距640B | STRIDED-BIG 5×128B（**设计内保留**：`bk=k=160` 需 UB>24576） | 未采 | NON-judgable(local) |
| **96x160** | fp16 | **STRIDED-SMALL** 5×64B | SINGLE 1×320B | 未采 | NON-judgable(local) |
| **64x48**（bn=64） | fp32/fp16 | **STRIDED-SMALL** 3×64B / 3×32B | SINGLE 1×192B / 1×96B | 未采 | NON-judgable(local) |

**统计（真实计数，非估算）**：`grep -c STRIDED-SMALL` → 改前 **15**、改后 **0**（上面 22 格小表，
日志 `/tmp/m2logs/plan_{pre,post}.txt`）；宽表（n×k 含 bn=64 大 K，16 格）改前 **5** SMALL、改后 **0** SMALL
+ **3 格新落 eager**（`64x112` fp32/fp16、`64x160` fp16），日志 `/tmp/m2logs/plan_{pre,post}_wide.txt`。
T3 判决总数：`M2-CARD-DMA-SUMMARY pass=8/8 strided_small=NONE fail=NONE compiled=8`。

## 3. 补档表 B：C2 原 9 件在 **T2（fp16 原生入）× T3（真方言）** 下的"装填面暴露度"

| C2 行 | 内核 fp16 面 | fp16 件的 MTE2 块搬运（《取证》§2.4） | 该件在原矩阵里能否被坏档打到 |
|---|---|---|---|
| linear | **fp16 原生**（A/W/C `half`） | 有 ⇒ 归档 3 枚 fp16 gemm：2 单发连续 + **1 跨步小档（病档）** | **能**（P1-1p 实际就是它反向 dA 中的 `bk<k` 格） |
| dW（gemm_bwd_dw） | **fp16 原生入 / fp32 出口** | `dw_impl` fp16 两枚 **`copies=0`**（纯标量旁路） | **不能**（该件天然无跨步档可踩；原矩阵 9/9 里这格是"弱覆盖"而非"零覆盖"） |
| readout | 入口升宽（`.float()`） | `main_kernel` fp16 两枚：`74a119a0` 4 条 gm2ub 全 `SINGLE`；`cb509fe6` `copies=0`（`half condval` 只作搬运临时，下一行即 `(float)`） | 否 |
| add_ln / ln_bwd | **入口升宽**（`add_ln_asc.py:335-337/360-362/407-408`）⇒ fp16 档**不存在** LN 件 | 3 件全 fp32；gm2ub `1×N 连续` + 尾行夹址 | 否（也是《取证》§2.2 反证 notes.md:26 嫌疑的依据） |
| gdn fwd / gdn bwd | 入口升宽 | 归档 `gdn_impl`/`gdn_bwd_impl` 无 `half` | 否 |
| rope | **fp16 原生 + 混宽**（`qdt=float16, adt=float32`） | `rope_impl` fp16 **`copies=0`** | **不能**（无跨步装填面） |
| attn_sw | 入口升宽 | 无 `half` 件 | 否 |
| lora | 入口升宽 ⇒ 内含 gemm 模具恒 `dfloat32`（C2 §2 已记） | 键 `…n16k32b16x16x32anonedfloat32` ⇒ `bk==k` | 否（**且注意**：`lora_a W=(8,32)`、`lora_b W=(32,8)`、`Wgb W=(2,32)` 三形 `plan` 直接拒 ⇒ 一直走 eager，`blockers=0`，见《取证》§7 洞 2） |

⇒ **补档结论**：原 18 格里"真能压到装填轴"的只有 **linear 一行**（且其 fp16 档只覆盖了 `bk==k` 的连续档）。
其余 fp16 格要么内核不发块搬运（dw/rope），要么入口升宽根本不进 fp16 内核（readout/add_ln/gdn×2/attn_sw/lora）。
⇒ 建议把"装填档"作为 C2 的**第三根轴**（位宽 × 形状 × 档），而不是指望 18 格自己长出来。

## 4. 覆盖代价（如实登记，不粉饰）

改后纪律 `bk==k 优先 → 够宽跨步 → 否则 eager`，代价是 UB 超限且 `ladder·elem < 128B` 的形状**从内核退回 eager**：

| 形状 | 位宽 | 改前 | 改后 | 代价评估 |
|---|---|---|---|---|
| 32x112 / 64x112 | fp32/fp16 | STRIDED-SMALL（**坏档**） | eager | 无损失——原来是踩坏档 |
| 64x160 | fp16 | STRIDED-SMALL（坏档） | eager | 同上 |
| 96x160 | fp32 | STRIDED-BIG 5×128B | STRIDED-BIG 5×128B（未变） | 无变化（设计内：仍留够宽跨步档） |

生产链路实测**零覆盖损失**：host fp16 训练步改前/改后同为 `keys=13`、`calls=45`、eager 集合 3 形不变，
数值逐字同（`/tmp/m2logs/{e2e_fp16_pre,e2e_fp16_post2}.txt`，diff 仅 3 行：OP031 档/键、`M2-E2E-KEYS`、tag）。
`compile_all_asc.py --target cpu` `PASS=9 FAIL=0`、`C2-SUMMARY fp16:9/9 fp32:9/9 总计 18/18` 均未回归。

## 5. 与 P1-1q / U2 的呼应（以及本窗**没能**判的事）

- `notes.md:22` 已把 P1-1q 的判决文案从"多块组合 + relu"改记为「**K 非 32/64 对齐档的 L1 装填/多轮累加**」。
  本窗《取证》§2.4 给出该改记的**具体落点**：不是 L1、不是累加位宽（普查 `accum=fp32`），而是
  **GM→UB 的 MTE2 多突发跨步装填**；并把边界夹在 `blockLen 64B 坏 / 128B 未致坏`（两点实测，**非硬件规范**）。
- **语义轴未判**：`5d732a8f` 与 `ea1059e6` 的三元组同为 `(32, 64, …, 192, 64)`，但"第 9、10 参到底是
  pitch（行距）还是 gap（间隔）还是被忽略"这条轴，本窗无卡、无 readout，**不判**。
  `reconcile/U2_probe_k48_dma_semantics.py` 已在（探针形态就绪），但 U2 卷宗引用的
  `U2_matrix_k48.md` / `U2_K48_EVIDENCE.md` 在 `reconcile/` **不存在**（`ls` 实测）⇒ 若 U2 已开窗，其文档
  落点与命名需与主目录对齐；若未开窗，本窗不代其下结论。
- `attempts/` 实测路径是 `release/ascend/port910b/attempts/`（非 `release/attempts/`）；《取证》引用
  `compat_gap_C.md`（G-C4/G-C8 单位裁决＝32B 块）与 `X_FRACTAL_EVIDENCE.md` 均在此路径下核过。
- **本地判不了的格**（汇总，全部标 `NON-judgable(local)`）：T4 全列、以及 §2 里"容器未采"的 8 格
  （32x80/96x80/96x96/32x112/96x160 各档、64x48 两档）。补采成本＝T3 容器编译（分钟级、不占卡）；
  T4 才是占卡项。

## 6. 建议进矩阵的列（下窗可直接续表，不必重跑 T1/T2）

| 新增列 | 判据（可机器判） | 判轨 |
|---|---|---|
| `装填档` | 生成码每条 `asc_copy_gm2ub_align` 必须 `blockCount==1` 或 `src_stride==burst`；出现多突发且 `burst<128B` ⇒ FAIL | T3（`M2_probe_card_fp16_dma.py` 已实现） |
| `累加器位宽断言` | `accum ∈ {fp32, NO-REDUCE}`、`fp16_reduce=NONE`、实写字节 ≤ `host_kernel.c` 申报 UB、UB 块不重叠 | T3（`M2_probe_gen_index.py` 已实现，`violations=0`） |
| `eager 覆盖账` | plan 拒形状必须**显式计数**（现不记 blocker ⇒ 与真绿不可分） | 待立项（《取证》§7 洞 2；涉及 `gemm_kernel.py`/证照件，本窗**未动**） |
