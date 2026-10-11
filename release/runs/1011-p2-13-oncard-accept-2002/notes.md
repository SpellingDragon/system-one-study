# 1011 20:02 · p2-13 首次真机验收窗（card_run.sh 全链，含 P1-1j 产品件 + 两档 dtype 训练步）

实例 910B4 · runner=`ascend/port910b/card_run.sh`（md5 `5ef1e13a`，repo `4b43a3f`）· 主链墙钟 720s（`RUNNER-DONE SUMMARY pass=7 fail=3 skipped_legs=1 degraded=0 advisory_fail=3 exit=1`）
**产物归档**：`artifacts/card_run.log`(19,873B) · `artifacts/sys1_legs/`(14 份腿原文) · `artifacts/sys1_kernels.tgz`(**552,236B / 422 条目 / md5 `f2ba9a2080de73d1d04d4a309dc4920e`**，**未入库**——二进制，是否入库由决策人定) · `artifacts/H_case{1,2}.{py,log}`(归因 driver 与原文)

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
