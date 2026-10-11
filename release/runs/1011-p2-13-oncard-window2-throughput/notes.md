# 1011 21:0x · p2-13 第二窗：吞吐定档 + 跨步语义裁决 + 两缺陷真机闭环（墙钟 ≈16 分钟）

runner=`window2_card.sh`（四段各自独立进程）· repo `210f078` · **A 段首次 `RUNNER-DONE … exit=0` 全绿**
产物：`artifacts/`（w2 四段日志 + card_run 内层日志 + 14 腿原文 + **sys1_kernels.tgz 561,148B / 431 条目 / md5 `440b59589201c7545db3d8e7430c1f1c`**，未入库）

## ① A 段：验收链全绿（最新代码，两缺陷真机治好）
| 判决 | 值 |
|---|---|
| `ASC-PROD-linear_prod` | **PASS** `rel=1.59e-07 dir_disc=1.43e+00 relu_neg_clipped=787 reuse=4/4 shapes=5`（上一窗同腿 `rel=1.8e+01`/NaN） |
| `dw_prod` / `readout_prod` | PASS `1.36e-07` / `0.00 bitexact=True` |
| **fp32 训练步** | `loss0=1.077804 loss1=0.969059 reduced=True` + `KERNEL-USED-PASS kernels=13 blockers=0 fallbacks=0` |
| **fp16 训练步** | **`loss0=0.975384 loss1=0.896910 reduced=True params_grad_ok=True`** + `DTYPE-TRACE fp16 params=16 fp16=16 fp32_accum=true fallback=0` + `KERNEL-USED-PASS`（上一窗 `→ nan` rc=3） |
| 汇总 | `pass=10 fail=0 skipped_legs=1 degraded=0 advisory_fail=3 exit=0`（advisory 三条＝intrinsic 族，见 ③） |
| 打包 | `KERNELS-TGZ size=561148B entries=329 kernels=47` |

## ② D 段：反事实闭环（因果被真机证明，不是碰巧）
`--revert` 撤 bk 避让纪律（md5 `a0e7f19e`）→ fp16 训练步 **`loss0=0.975396 → nan`（rc=3，与上一窗逐字同形）** → `--apply` 还原（md5 `6ef882df`）。
⇒ **M2 的因果判定成立**：NaN 由多突发跨步装填引发，避让即消除；同窗 A 段证其充分性。

## ③ C 段：跨步装填语义裁决（P1-1s 第一枪，结论出乎所有人预设）
`U2-DMA-READOUT dev=npu m=16 n=96 ks=[48,96,32]`（`W[i][j]=i*1000+j` 门牌号法，强钉 bk 梯，**实测非推断**）：
- **k=48/bk=16 与 k=96/bk=32（STRIDED 档）**：六候选失配排序 **`Z(dstStride被忽略) mismatch=45 / 48`** ≪ `G(双gap)384` < `Gd 696~720` < `Gs 1380~1408` < `P(pitch,pitch) 1512~1536` = `C(srcStride被忽略) 1536`；`U2-DISCRIMINATE … 与 P 同解的其它候选=无 ⇒ 判别力=强`
- **k=32（SINGLE 档）**：`P=C=G mismatch=0`、`Z=48` ⇒ 该档**判别力=弱**（如探针自述，单发档不能定语义，须靠跨步档）
⇒ **裁决：多突发跨步形态下硬件忽略 `dstStride`（每突发按连续写），我们的 compat 层按 pitch 语义喂参 ⇒ 装填错位坐实。既不是 pitch 也不是 gap，而是"目的步长无效"。**
**残留未归因（诚实标注）**：最佳候选仍非 0 失配（45/48），且 k=96 两档 `UNATTRIBUTED(越界/stale)=640 / 1152`、探针自判 `unexpected=4` ⇒ **存在第二层因素**（疑末块 `blockLen` 截断或 UB 残留/stale），P1-1s 未闭。

## ④ B 段：吞吐定档——**可用一侧数据硬，对照一侧数据不可信**
可信（跑的是已过数值判决的产品件，`track=npu/ascend`、`reuse=new`）：
`linear 512³ fp16 p50=294.3ms (0.912 TFLOPS)` · `dW m1024n512k512 fp16 266.1ms (2.018)` · `add_ln rows8192xd1024 17.1ms` · `rope tok4096xh8xd128 11.7ms` · **`attn_sw h8seq1024d128w512 1879.3ms (0.858)`** · `gdn_conv C1024L2048k4 23.2ms` · `gdn_delta fwd 293.2 / bwd 431.3ms` · `BENCH-SUMMARY ops=12 ok=9 fail=3 wall_s=59.5` · bf16 两格被**假绿守卫正确拦下**（`blockers=[]` 说明是 plan 拒形状而非后端故障）
**不可信**：`linear_intrinsic p50=0.080ms ⇒ 3365.9 TFLOPS`、`ratio=3690.3x`——**物理不可能**（910B 的 fp16 cube 峰值远低于此，且该腿键为 `out-of-band-cube-archive`、守卫弱）。⇒ **`BENCH-ABLATE` 与 `BENCH-QUOTE linear_intrinsic` 一律不得引用**；须离线用新归档 431 条目核对那枚 intrinsic 件是否真做了 512³ 的活（很可能近空转/形状不符）。
外推（**标注为推断，非实测**）：若每层 ~4 个同量级 matmul、24 层，仅前向 matmul ≈ 28s/步，反向 ≈2× ⇒ **分钟级/步** ⇒ 按 ¥20/h 训练不经济。

## ⑤ 战略结论（写入 design 的 D-cube2）
1. **标量路＝正确性/研究车，不是性能车**（实测亚 1 TFLOPS 量级、attn_sw 单枚 1.9s）。
2. 我 review 时预留的分支"若慢 50× 则 cube 路降级为研究项"被实测**推翻方向**：慢 3 个数量级 ⇒ **§12/cube 路不能降级，它是唯一性能出路**，投入价值被这份数据抬高。
3. 过渡期性能兜底仍是第二栈（`torch_npu` 原生算子）——这与"探索脱离 CUDA"不冲突：底座先保吞吐，逐件替换靠 cube 路修好。
4. 下一枪已备好（**全部不需开卡**）：① 核对 intrinsic 计时真伪（离线 431 条目）② 追 45/48 残留失配的第二层因素 ③ 据 `dstStride 被忽略` 定 compat 层修法（影响面＝所有 `blockCount>1` 件）。
