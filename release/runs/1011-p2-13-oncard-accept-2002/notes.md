# 1011 20:02 · p2-13 首次真机验收窗（card_run.sh 全链，含 P1-1j 产品件 + 两档 dtype 训练步）

实例 910B4 · runner=`ascend/port910b/card_run.sh`（md5 `5ef1e13a`，repo `4b43a3f`）· 主链墙钟 720s（`RUNNER-DONE SUMMARY pass=7 fail=3 skipped_legs=1 degraded=0 advisory_fail=3 exit=1`）
**产物归档**：`artifacts/card_run.log`(19,873B) · `artifacts/sys1_legs/`(14 份腿原文) · `artifacts/sys1_kernels.tgz`(**552,236B / 422 条目 / md5 `f2ba9a2080de73d1d04d4a309dc8496e`**，**未入库**——二进制，是否入库由决策人定) · `artifacts/H_case{1,2}.{py,log}`(归因 driver 与原文)

## ① 里程碑（真机首证）
| 项 | 判决 | 凭据行 |
|---|---|---|
| **H1 设备注册** | ✅ 修复 | `NPU-REG ok 2.7.1.post2 target=ascend npu_avail=True`（镜像内 torch 已注册，我们的条件 import 为无害冗余） |
| **fp32 训练步真跑通 + 确证走内核** | ✅ **R1 gate 主体达成** | `P2-STEP-ascend loss0=1.082224 loss1=0.945310 reduced=True params_grad_ok=True` + **`KERNEL-USED-PASS kernels=13 blockers=0 fallbacks=0`** |
| `dw_prod`（dW 生产件真机数值） | ✅ | `rel=1.36e-07 dir_disc=1.39e+00 reuse=4/4 shapes=4 track=card` |
| `readout_prod` | ✅ | `rel=0.00e+00 fwd=0.00e+00 bwd=0.00e+00 bitexact=True reuse=6/6 shapes=5` |
| `addln`（含反向偏判别） | ✅ | `rel_y=1.69e-05 rel_h=0.00e+00 unb_disc=7.55e-04`（J 单进程复验同值） |
| GDN delta 前向 | ✅ | `CASE-delta VERDICT-PASS` |
| 九件接口腿 target=ascend | ✅ 9/9 | `ASC-gemm/gemm_bwd_dw/add_ln/rope/attn_sw/gdn/gdn_conv/letter_readout/lora PASS`，`.o` 与容器基线**逐字同**（2850/2250/3625/2106/2911/4003/1864/922/2842） |
| 生成码打包（增采 C） | ✅ | `KERNELS-TGZ size=552236B entries=322 kernels=46`（本地可解 422 条目） |

## ② 三条真机缺陷（本窗新暴露，均有定位）
1. **`linear_prod` FAIL：真变量是 K=48，不是 relu、不是 n 方向多块**（H 案 1 六变体二分）
   - `m16n96k48/relu rel=nan(96 个 NaN)`；同形**关 relu 照样坏** `rel=1.150e+00`，`got=15.24771 ref=-15.37733`（错值/反号形态）
   - 对照全绿：`m16n96k32/relu 1.16e-07`、`m16n64k64/relu 1.55e-07`、`m16n32k16/relu 9.09e-08`、`m1n64k64/relu 1.24e-07`；relu 版 `zero_out=747 < zero_ref=767` ⇒ **relu 只放大不致因**
   - ⇒ 原判决文案"多块组合 bn=32/bk=16 + relu 出口分支"**是误导**，应改记「K 非 32/64 对齐档的 L1 装填/多轮累加」；全程 `inf=0`、无 507015 ⇒ **非硬件路**
2. **fp16 训练步 `loss0=0.975374 → nan`（rc=3）**，炸点在 **backward**（H 案 2 逐 op）
   - step0 前向 15 次调用**零 NaN 零 inf**、absmax 最大 4.08（远低于 fp16 上限）⇒ **不是位宽溢出**
   - **第一个 BAD 是 step1 的第一个 `linear`，且入参已带 NaN**（`in nan=1056`）⇒ NaN 生在 step0 的反向+优化器；`1056=1024+32=W1+b1`（算式推断，未逐张量实测）
   - **`DTYPE-TRACE fp32_accum=true`、`fp16_acc_violations=0`、`KERNEL-USED-PASS` 三证照绿** ⇒ Python 层纪律钉**量不到内核内部 fp16 累加面**（这是"证照可信度"的边界，须写进文档）；`inf=0` 全程 ⇒ 更像 0/0 或 `rsqrt(≤0)` 灾难，嫌疑 `add_ln.backward`
3. **§12/cube 路在干净进程里仍 `507015`**（J 裁决，推翻"只是聚合污染"）
   - `NUM-gemm_l1_intrinsic-FAIL ... 507015`、`NUM-dw_intrinsic-FAIL ... 507015`（各自独立单进程、冷缓存）
   - 但**聚合窗内其后的 addln/gdn/rope/attnsw 红确有污染成分**：`addln_intrinsic` 单进程 `PASS`、`gdn` 窗内 `PASS` ⇒ 那些条不得当独立证据
   - ⇒ **P1-1i（ND2NZ + sid 钉 0）的修复"必要但不充分"结论再次成立**；且 `D2_variants.md` 的变体归因腿在卡上 **spawn rc=1、零产物**（两趟作废，未得真值表落点）⇒ **归因工具本身要先修**（本地容器即可复现，不耗卡）

## ③ 代理间冲突裁定（R26）
J 质疑 H"独立进程 gemm 零 507015"与己矛盾。裁定**不矛盾**：H 案 1 跑 `linear_prod`＝**可移植标量路**（A3 静态取证 `A3-GENCODE ... Cube=0 nd2nz=0 LoadData2D=0`，只发 UB-DMA），J 跑 `*_intrinsic`＝**cube/§12 路**。两条腿本不在同一生成码族——这正是 A3 预言的"产品件数值失败不可往 §12 追"的对偶。

## ④ 工装结论（下次开窗的直接改法）
- **聚合单进程是错的**：一案一进程（本窗 `ASC-products` 用单进程跑完全部 case，导致后半串假红）。A2 的入口需支持 `--isolate` 或改由 runner 逐案起子进程。
- `RUNNER-DONE`/`exit=1` 语义有效（fail=3 全部真实）；`SUMMARY-DEGRADED 命中=0` 证本窗**无"只判编译"降级**，判决全是真发射。
- 未跑：`SYS1_BENCH`（默认不进窗，保验收优先）→ 吞吐/报价数据仍待下一次；R6/R7/C5 未闭。


## ⑤ 勘误（离线取证推翻本台账两处猜测，追加不改写）
1. **tgz md5 我记错**（`…4920e` → 实测 `…8496e`），已就地更正；size/条目/内核数三项原本即正确。
2. **「嫌疑 `add_ln.backward` 的 `rsqrt/var/eps` 在 fp16 累加」被反证**（M2 普查归档 9 枚含 `__gm__ half` 的件）：`accum=fp32`、`fp16_reduce=NONE`、`violations=0`；LN 三件签名全 `__gm__ float*`，且 `ln_bwd:41 rs = rsqrtf((acc/32.0) + 1e-5)`（eps 在根号内、除数为编译期常量）⇒ **不存在 fp16 归约面**。
3. **两案收敛到同一根因轴（两个独立代理、两套证据）**：
   - U2（K=48）：病档 `5d732a8f`(k48) 与对照 `77a8a3c4`(k32) **仅 3 处实质差异**，出口地址表达式**两处完全同一**；差异 D3＝W 块由单发连续 `1,4096,…` 变多突发跨步 **`(32,64,…,192,64)`**；成因锁定在 `tilelang/src/ascend/op/copy.cc::plan_single_row`——`bk==k` 折叠单发、`bk<k` 一律 `blockCount=bn` 跨步 ⇒「K=48 特殊」实为 **`bk(16)<k(48)`**。全表枚举：**STRIDED 仅 3 枚产品 gemm**（k48 fp32、k96 fp32、k96 fp16）+1 枚非产品件。
   - M2（fp16 NaN）：病档 `ea1059e6`(fp16 k96) 正是**唯一 fp16 跨步件**；其 `asc_copy_gm2ub_align(..., 32, 64, 0,0,0, mode0, 192, 64)` 与 U2 的坏档同形。
   ⇒ **待决语义**＝`copy_gm_to_ubuf` 的 `srcStride/dstStride` 是 **pitch 还是 gap**（U2 标为假设级 H1，未定案）；影响面＝**所有 `blockCount>1` 件**（搬运适配层单点）。

## ⑥ 已落地的处置与其性质（如实标注，勿当根因修复）
- M2 在 `gemm_asc.py` 落 **bk 选择纪律**（`bk==k` 单发优先 → 否则只允许突发 ≥128B 的跨步档 → 都不满足落 eager），md5 `6ef882df`：plan 侧坏档 **15→0**、容器真 ascend 生成码 **8/8 全 SINGLE**、pytest **340 passed**、cpu 门 **9/9**、C2 矩阵 **fp32 9/9 + fp16 9/9 = 18/18**、e2e 数值逐字同改前。**这是"避让"不是"根治"**——跨步装填语义仍未证真。
- **可观测性代价（U2 提出，务必记住）**：改后生产 plan **再也编不出 `blockCount>1`** ⇒ 真机"跑跨步"这条路不可达，跨步是否已修**只能靠强钉探针**判（`U2_probe_k48_dma_semantics.py` 的 npu 轨已加固为强钉 bk 梯）。三形状（`64x112`、`64x160`、`96x160`）改后主动落 eager，生产链路零覆盖损失（`keys=13/calls=45` 前后同）。
- **本地不复现已定性**（U2）：绿的是 T1 `cpu` 方言与 T2 `cpu-alias`；真 `ascend` 方言在 host 结构性编不出（`ascend_copy is not registered`）⇒ **本地任何绿对跨步语义零信息量**。C2 的 9/9 漏掉此缺陷的盲区根因＝`C2_probe_dtype_matrix.py:72` 的 `LIN` 三档 **k 全是 32 ⇒ 永远 bk==k ⇒ 永远编不出跨步**。
- 判决文案已更正（`run_numerics.py` 仅 `LIN_SHAPES[3]` 说明字段，diff `+9/−1`，删的那行正是"多块组合 bn=32/bk=16 + relu 出口分支"；阈值 `2e-05` 与五档形状一字未动）。

## ⑦ 下一次短窗（≈10 分钟）要做的三件事
1. `python3 ascend/kernels/reconcile/U2_probe_k48_dma_semantics.py --dev npu`（强钉跨步梯）⇒ 一枪定 pitch/gap，据此立项修 `asc_copy_gm2ub_align`
2. fp16 训练步发射判决（期望 `loss0≈0.975` → **有限** loss1 且下降）＋ `M2_probe_patch_gemm.py --revert` 反事实对照（还原跨步档应复现 NaN）
3. **H 案之后**再打包生成码（本窗归档缺 `k=48/none` 腿，令对照混了 act 变量——U2 已用本地双 act diff 补严）
