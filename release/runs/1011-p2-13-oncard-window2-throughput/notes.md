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


## ⑥ 勘误（离线复核推翻本台账 §③ 与 §④ 各一条，追加不改写）
1. **§③ 的"硬件忽略 `dstStride`"裁决不成立——是探针度量的伪影**（S3 复算）：U2 打分器的 `nn` **从不统计"实测为有限值而候选预测 NaN"**，于是"只敢提交 48 格"的候选天然获胜；且台账叙述"每突发连续写"（该行为应编码成 `(k,bk)`＝P）与真正胜出的候选 `(k,0)`（目的指针**完全不前移**）**自相矛盾**。⇒ **撤回该结论**，改为：**一手 ISA 证据显示 `srcStride`/`dstStride` 与 `srcGap`/`dstGap` 是同一 union 别名**（`kernel_struct_data_copy.h:34-60`，注释 "will be deprecated, use srcGap instead"）、`srcMaxOffset = nBurst*lenBurst*unit + (nBurst-1)*srcStride*unit`（`kernel_check_data_copy_util.h:346-351`）、`gmGapValue=(gStride3-gShape4)*sizeof`（`TLoad.hpp:69-72`）⇒ **该参数语义是 gap 而非 pitch，我们按 pitch 喂参才是错**（H-c 判真；SINGLE 档下 `gap=pitch−blockLen=0` 故恒等，这也解释了为何只有跨步档出问题）。
2. **§④ 的"3366 TFLOPS 物理不可能"这句话本身是错的**（V2）：`bench_card.py` 的 `tflops` 字段除以 `1e9` 却标 `tflops=` ⇒ **单位虚高 1000×**，真实读数是 **3.37 TFLOP/s**，在 910B1 实测峰值 **363.7 TFLOPS**（`platform_config/Ascend910B1.ini`: `ai_core_cnt=24 × cube_freq=1850 × 16×16×16 × 2`）以内。⇒ **作废理由换成更硬的一条：那枚 `linear_intrinsic` 件是死代码**——`__mix__(1,2)` 下正文被 `if (ASC_IS_AIV)`（容器实测 `ASC_IS_AIV=0` @ `port910b_compat.h:568`）包成 `if(0){...}`，**从未执行**，0.080ms 量的是空发射。`dW_intrinsic` 则真执行且 507015（坏件）。⇒ **三栈消融当前无任何可引用数据**；且我的"慢 3 个数量级"外推**不成立于对照侧**，标量侧绝对数仍成立（`512³ fp16 294ms`）。
3. **计数更正**：§④ 的"标量侧 9 条 ok"应为 **8 条标量 + 1 条假绿**（`ok=9` 里含 `linear_intrinsic`）。
4. **已落工装修正**（V2）：计时前置**数值自证**（未过 ⇒ `BENCH-<op> NOTIMED reason=...`，不出 `p50/tflops/QUOTE`，负例两条自证）+ `BENCH-ABLATE verified=` + 峰值闸 + **`tflops` 单位改 `/1e12`** + gdn 门输入良态化（原用 `randn` 使参考自身 `nan`）。回归 `340 passed`、cpu 轨 `BENCH-SUMMARY` 照出。
5. **S3 的否定性结论（重要，防后续再犯）**：三维穷举 `sp∈[0,96]×dp∈[0,48]×soff∈[-32,32]`＝**308,945 组零命中**（最小 L1=167）⇒ **单一几何模型解释不了 k=48 的 `(0,1536,1353,0)`**，残留里至少含一个非几何成分（最可能 H-b：`w_ub[0,512)` 全文无初始化，归档 `:12-48` 只清了 c_ub/a_ub；**未直测，标 ❓**）。
6. **提案不下地**（S3 交付 `attempts/S3_compat_proposal.diff`，gap 归一，`patch --dry-run` 通过、trunk md5 复验未变）：**等下窗按 `S3_probe_nextshot.py` 一枪判 H-b/H-e 再落**——理由：产品 plan 现恒 `bk==k`（M2 避让），此改动对当前产品路径**无可观测收益**，而"改了但没判决"会污染基线。
