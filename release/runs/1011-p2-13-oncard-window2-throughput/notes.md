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


## ⑦ 离线收口四路（V2/S3/T2/C3，编排者复验凭据）
- **V2（P1-1t 收口）**：`linear_intrinsic` 定性＝**死代码**（`__mix__(1,2)` 正文被 `if (ASC_IS_AIV)` 包裹，容器实测 `ASC_IS_AIV=0` @ `port910b_compat.h:568`）；形状/同步/参数序三候选全部排除（循环边界恰为 512³、`measure` 每样本 `sync();t0;call();sync()`、归档 `host_kernel.c` 证 host 按**参数身份**映射）。峰值闸取一手 `Ascend910B1.ini`（24 核×1850MHz×16×16×16×2＝**363.7 TFLOPS**）。**编排者独立复跑**：`--selftest-notimed` → `BENCH-selftest_notimed NOTIMED reason=no-numeric-selfproof`、`--selftest-mismatch` → `NOTIMED reason=numeric-mismatch(rel=3.333e-01 tol=0.01)`，**两条均不出 p50/tflops**；正例仍 9 条 `verified=true`；`pytest` **340 passed**。
- **S3（P1-1s）**：见 §⑥ 勘误 1/5/6。一手 ISA 三条已由 R2 **重新打开核对**（`kernel_struct_data_copy.h:45-58` union 别名、`kernel_check_data_copy_util.h:344-352` 公式落 :349-350、`TLoad.hpp:69-72`），并纠正了 S3 记的区间偏移。
- **T2（P1-1r 修复并独立复验）**：病灶＝`REPO` 默认写死容器路径 `/work/ascend`，卡上未 export ⇒ 逐腿 `FileNotFoundError` **被 pipe 缓冲吞成静默 rc=1**；另揪出 `--card` 空开关、`so=` 行结构上不可出两处次生缺陷。修后**编排者容器实跑**：`--combos default,A` → `D2-LEGS-SUMMARY valid=2/2`、每趟三判决行齐（`inject`+`SO-VERDICT so=BASE/DIFF`+`NUM/leg-rc`）、`+A so=DIFF` 命中 `dW_l0tr_kernel#0,main_kernel#0`；缺件腿 `FATAL D2-PRECHECK` 退 2（负例已由编排者触发实证）。
  - **⚠ 编排者复验新发现（T2 未自知）**：`RUNNERS` 拼法是 `$REPO/compile_all_asc.py` 与 `$REPO/port910b/run_numerics.py` ⇒ **`REPO` 语义实为 `release/ascend/`**，而 docstring:45-46 与 `D2_legs_fix.md §3` 给的卡上值是 `/tmp/sys1`（repo 根）——照它设会在 precheck 就 FATAL。**卡上正确值＝`/tmp/sys1/release/ascend`**。已派 Z2 订正（否则下窗会重演第二窗的作废现场）。
- **C3（收口条件复核，197 行）**：`P1-3` 按字面判据可勾但**反向真机 rel 缺** ⇒ 本编排者裁**不勾**（标题含前后向，不玩字面）；`P1-1 四件改道` 缺 `addln_prod` 产品件接入 + 原文"gradcheck NPU 档"**全库从未落地**（悬空判据）；`P2 三栈联调` 原文称"卡窗即跑 **0.8B** 一步"，出厂件实为 **d=32 最小冒烟**（`train_step.py:9,402`）⇒ 口径不符；`R6/R7` 缺熔断判定（依赖真模型每步 op 次数）；`P1-1d` 判 **❌被证伪**（三处 507015 原文）。C3 亦自纠一处出处错记（`NUM-gdn 6.35e-08` 仅首窗有）。
- **R2（P1-1u）**：三份 issue 草稿 + README/STATUS 落 `docs/upstream/`（词数 1608/1621/1541）；一手引文**逐条重开核对**并纠正 S3 的区间偏移；独立复现 bf16 `error in backend: not support bf16 type cast`（exit 70），且**超出 brief 框定**——`__cube__` 面同样复现 ⇒ 触发是 cast 本身而非面别（已在正文标注我方笔记之差）。私仓差异面实测：`git diff --numstat` 五件 `+82/−26` + untracked `port910b_compat.h 1248` 行等。R2 自陈最没把握的一句（把"pitch 灌进 gap 槽"当已发生事实陈述）⇒ **已裁降级为 leading hypothesis**，交 Z2 改。


## ⑧ 第三窗就绪包（A4/B3/C4 三代理 + 编排者独立复验；`HEAD` 见下）
- **A4（addln_prod 接入）**：`run_numerics.py` 新增 `addln_prod`（照 `readout_prod` 形态走产品入口件；阈值 `r_h<1e-5/r_y<3e-4/unb_disc` 一字未动）。**编排者复验**：host `Y-addln_prod-PROD-PASS rel_y=1.84e-05 rel_h=0.00 unb_disc=7.36e-04`；容器聚合 `产品腿 PASS=4 FAIL=0`（3→4）、门禁失败数=0；既有三腿单件逐跑 `2.54e-07/2.21e-07/0.00 bitexact` **与基线逐字同**；负例（桩坏 plan）判 FAIL 不找旧 attempts 件兜底。旧 `addln` case 保留加 `[note]`（并存迁移）。
  - **裁决 1（A4 待裁 D1：cpu-alias 存储层 shim）＝接受**：`alloc_shared→alloc_local` 仅装在 cpu-alias 轨、随 `restore()` 还原、产品件零改动——与 S 代理双方言技巧同族（本项目既定做法）；**证据类型标注**：host shim 轨证"数学对"，ascend codegen 证据归容器编译判决 + 第三窗真机。若有人把它当"ascend 路已验"即属误读，已写明。
  - **裁决 2（A4 待裁 D3）＝不改**：add_ln 的 plan 键本无 dtype 段（`d{dim}e{eps}b{bm}`），系产品设计（add_ln 不做 dtype 参数化）；"位宽段"要求落空属任务措辞超前，登记即可，**不解禁产品件**。
- **B3（gradcheck 档落地，P1-1y）**：`ascend/gradcheck_card.py`（398 行）8 门面全覆；**编排者复验**：host 8/8 `GRADCHECK-SUMMARY ops=8 pass=8 fail=0 invalid=0 kernel_used=8/8`；三负例（backend off ⇒ 全 INVALID 不冒充 PASS；plan 拒形 ⇒ INVALID 不冒充 FAIL；参考 ×1.001 ⇒ FAIL）；容器 `--target ascend --no-run` 8 键齐。**B3 自纠两洞**：反向键顶前向的假绿（绑定改锚前向增量键）、回落公告键含 `]` 截断 bug。**偏离裁决＝接受**（rtol/atol 数值沿用 host gradcheck 同组，但判别式为 Frobenius 范数级——理由：fp32 中心差分对近零元素的逐元素噪声地板高于 atol，逐元素口径会把 `add_ln/gdn_conv/gdn_delta` 误报 FAIL；负例与正例相隔 ~38×，判别力保留，已在设计文档 §0 登记）。lora 边界如实（eager 反退路抛 grad 错 ⇒ INVALID，B_RESULT 已记缺陷不掩盖）。
- **C4（delta_bwd 腿，P1-1w 判决件）**：`run_delta_card.py --bwd` 六项梯度逐项打（`dq/dk/dv/dbeta/dg/dh0`+`adjoint`），参考侧＝fp64 解析 golden 复算（同输入同尺子，`adjoint=6.22e-10` FD 独立锚兜底）；**编排者复验**：host `NUM-delta_bwd PASS dq=7.53e-08 … dh0=6.18e-08 adjoint=6.22e-10 rel_max=1.01e-07`；负例 ×1.01 六项逐项指认全对；容器键 `gdnb[ascend]|h2k8v8|c2ubz` 在；fwd 腿 before/after 逐字同。
- **编排者自己的坑（记入防再犯）**：`run_numerics.py --only` **不吃逗号多值**（以 `unknown target` FAIL 明拒，行为正确）；多案聚合入口是 `compile_all_asc --products`。我复验时误用逗号形态得 0 计数，单件逐跑证伪了我自己的怀疑。
- **第三窗入口**：`release/ascend/port910b/window3_card.sh`（五段：A 验收链含 addln_prod+fp16 → B 卡上 gradcheck → C delta_bwd → D D2 变体腿 X1–X7（`P910B_REPO=/tmp/sys1/release/ascend` 已按订正口径写死）→ E S3 第二层因素 `--prefills none,zero,perm`）。预计 ≈25–30 分钟。
