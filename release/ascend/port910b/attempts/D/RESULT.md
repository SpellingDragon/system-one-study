# RESULT.md — dW（权重梯度 gemm `dW = dYᵀ @ X`）910B tilelang DSL 件

执行代理 **D**（p2-13-ascend-runtime / 甲路 P1-1 波）。R19 体例：每条结论附**可复跑命令 + 输出摘录**。

## 0. 一眼结论

| 项 | 结论 |
|---|---|
| **编译判决** | **`D-DW-COMPILE-PASS`**，主案变体 = **`l0tr`**（T2048/N1024/K1024/BN128/BK128/TT64/cores64，fp16 入 → fp32 累加 → fp32 出，target=ascend `dav-2201`） |
| 变体矩阵 | `baseline` PASS / `ntt` PASS / **`l0tr` PASS** / `madl1` PASS / `madta` **FAIL** |
| 方案裁决 | **转置必须下移到 DMA 阶段，落点选 L1→L0 装填**（`asc_copy_l12l0a_transpose` + `asc_copy_l12l0b_transpose`，官方 dav_c220 8 参件）；mad 永远吃 NT；出口按 dW 自然朝向直出 |
| direct mad 路线 | **判决性否证**：`madl1`（把转置交给布局推断 `transpose_A=True`）生成码与 `l0tr` **逐字节相同**（仅 kernel 名不同）⇒ "direct mad 的 trans_A" 在 codegen 里被**降级成同一对 L1→L0 转置件**，不是一条新通路；真正想跳过转置搬运的 `madta`（GM→L0 直装）因 **G-D2**（GM→L0 无 DMA 面，codegen 退化成解引用 `__ca__` 的标量循环）编译 FAIL |
| 前置补丁 | 主案**必须**两个 compat 缺口修复才可编：GAP-D1（新增 `asc_copy_l12l0a_transpose`）+ GAP-D1b（真源 `asc_copy_l12l0b_transpose` 体内 6 参 → 官方 8 参）。详见 `compat_gap_D.md` |
| 主仓真源 | **一字未改**：`/tilelang/src/tl_templates/ascend/port910b_compat.h` = 944 行 / md5 `8faeb7c17fca396b3e0a422872f65ad1`（容器内只读复核，见 §1 命令 G） |
| 未做（声明） | 数值对拍（本机无 NPU，只交 golden + 三级判据）；`ntt` 的 `dn2nz` 真转置搬运（G-D3）；`T.Pipelined`/L0 双缓冲/`unit_flag_ctrl` 性能面；动态 token 轴与尾块谓词（当前强制 `T % TT == 0`，不符即 `ValueError`） |

---

## 1. 复现链（容器 `cann910b-d`，全部宿主机 `/bin/bash` 脚本驱动）

```bash
cd $REPO_ROOT/release/ascend/port910b/attempts/D

# A. 装配（幂等：真源 overlay → pip 副本 → GAP-D1 → GAP-D1b → bisheng 注入）
/bin/bash run_D.sh --setup
# B. 5 变体判决矩阵 + 建材回归（A2 / e2e_cube）+ golden 自检
/bin/bash run_D.sh --matrix            # 落盘版 = verdict_log_D.txt（wc -l = 208）
# C. 主案形状/tiling 鲁棒 5 组
/bin/bash run_D.sh --shapes
# D. 生成码取回（--dump）
/bin/bash run_D.sh --dump /tmp/dw_l0tr.asc l0tr
# E. golden + 上卡对拍件
/bin/bash run_D.sh --npz
# F. 判决纪律（脚本内每条编译都带）：TILELANG_CACHE_DIR=$(mktemp -d) timeout 300 python3 ...
#    理由：cache key 不含 compat 头内容 → 不换目录就是陈旧 PASS（假绿）
# G. 主仓真源未改自证
docker exec cann910b-d bash -c 'wc -l /tilelang/src/tl_templates/ascend/port910b_compat.h; md5sum /tilelang/src/tl_templates/ascend/port910b_compat.h'
```

A 的实际输出（本波最后一次装配）：

```
[1] BISHENG_HOME=/usr/local/Ascend/cann-8.5.0/aarch64-linux/ccec_compiler arch=dav-2201
[2] pip template tree: /usr/local/lib/python3.10/dist-packages/tilelang/src/tl_templates/ascend
[2] compat lines: 944 (真源同步 md5=8faeb7c17fca)
[2b] GAP-D1 compat: 991 lines (marker=3)
D-PATCH-COMPAT-D1b APPLIED .../port910b_compat.h (991 -> 994 lines)
bisheng.py already patched (.../tilelang/contrib/bisheng.py)
SETUP-DONE
```

---

## 2. 方案论证：转置恒等式 vs direct mad（本波核心）

### 2.1 撞墙的原文（约束事实，非推测）

`tilelang/ascend/op/gemm/gemm_mad.py:162`（外层 `if self._is_l1_input():` 在 :161，判据 A/B 同属 `shared.l1` 见 :166-167）：

```python
assert not self.trans_A and self.trans_B, "Ascend L1 GEMM currently only supports trans_A=False, trans_B=True (NT)."
```

而 `dW[n,k] = Σ_t dY[t,n] · X[t,k]` 的 A 操作数是 `dY` 本身（token 轴在前），**天然 trans_A=True**（TN 形态）⇒ 正面撞墙。生产尺子 `release/sys1/kernels/gemm_bwd_dw_mps.py:48-63` 就是 `T.gemm(Ds, As, Cs, transpose_A=True)` 的写法（fp16 入 / fp32 出、token 轴动态、BLOCK_TC=16、BLOCK_LADDER=(64,32,16)），在 HPU/CUDA 侧合法，在 910B L1 路非法。

**关键物理事实**：Ascend cube 的 `mad` 只认"L0A/L0B 里 K 轴连续（K-major）"的 NT 形态，而"谁在 GM/L1 里转置"是**搬运阶段**的事。于是可选的落点只有三个：① GM→L1（`dn2nz`）② L1→L0（`l12l0a/b_transpose`）③ mad 自身（`kDirectionAlign`/`btbuf_ctrl`）。

### 2.2 四条候选 + 实测判决（逐条凭据见 §3）

| 变体 | 落点 | DSL 写法 | 编译 | 判决与理由 |
|---|---|---|---|---|
| `ntt` | ① GM→L1 `dn2nz` + `shared.l1` 融合模板 | `T.copy(dY[...], a_l1, transpose=True)`；`a_l1=(BN,TT)`、`b_l1=(BK,TT)` 直接 NT；`T.gemm(a_l1,b_l1,acc,transpose_B=True)` | **PASS** | DSL 最短、模板自带 sub-K 双缓冲（性能最优候选）。**但不作主案**：compat 里 `asc_copy_gm2l1_dn2nz` 现在只是 `nd2nz` 的**同名别名 stub**（`port910b_compat.h:878-882`，注 `VERIFY V1`）⇒ **编译绿 ≠ 数值绿**，转置实际没发生（登记 **G-D3**）。取证：`dw_ntt.asc:10` 发 `asc_copy_gm2l1_dn2nz(..., 2048, mode, 128, 64, 0, 0)`，而 compat 侧体内转调 nd2nz |
| **`l0tr`** | ② **L1→L0 转置装填** | L1 存**自然布局** `dy_l1=(TT,BN)`/`x_l1=(TT,BK)`（GM→L1 `nd2nz`，真件）；`T.copy(dy_l1, a_l0, transpose=True)` 装成 NT 的 `a_l0=(BN,TT)`、`T.copy(x_l1, b_l0, transpose=True)` 装成 `b_l0=(BK,TT)`；`T.gemm(a_l0,b_l0,acc,transpose_B=True)` | **PASS**（需 GAP-D1+D1b） | **主案**。理由三条：(a) 两个转置件都是 **910B 官方真实存在**的 8 参 Cal 件（三处独立证据，见 `compat_gap_D.md` G-D1），不是别名 stub；(b) GM→L1 用**已证真**的 `nd2nz`（A案 A-2/A-3 用例同形），把不确定性只压在"新装的转置件"一处，上卡排错面最小；(c) 数学朝向天然正确：mad 发 `m=BN, k=TT, n=BK` ⇒ L0C 就是 `(BN,BK)` 的 dW 块，**出口零重排** |
| `madl1` | ③ 交给布局推断（direct mad 正写） | L0 缓冲按自然 `(TT,BN)`/`(TT,BK)` 声明，装填**不转置**，`T.gemm(..., transpose_A=True, transpose_B=False)` | **PASS** | **判决性证据**：生成码与 `l0tr` **逐字节相同**（仅 kernel 名）。`diff dw_l0tr.asc dw_madl1.asc` → 只有第 3 行 kernel 名不同（§5）。⇒ `trans_A=True` 并没有让 mad 自己转置，而是被 codegen/布局推断**翻译成同一对 `l12l0*_transpose` 显式转置装填**。"direct mad"作为**独立通路不存在**，它与主案是同一件事的两种写法 |
| `madta` | ③ + 跳过 L1 | `T.copy(gm_slice, a_l0)` 直装 L0A（`alloc_l0a((TT,BN))`）+ `transpose_A=True, transpose_B=False` | **FAIL** | 首错 `tl_kernel.asc:11:11: error: only __ubuf__ and __gm__ and local memory pointer can be dereferenced`；生成码体把 GM→L0A 降级成 `#pragma unroll` **8192 次标量写 `a_l0[i]=dY[...]`**（`verdict_log_D.txt:62-80` 全文可查）。⇒ **G-D2**：910B 后端**没有 GM→L0 的 DMA 通路**，而 codegen 不报错、静默退化成语义非法的标量体。这是 codegen/IR 层缺陷，compat 层补不了，本波**不做硬推**，只在 `compat_gap_D.md §3` 给上游最小修法 |

### 2.3 为什么"算 dW"而不是"算 dWᵀ 再转回"（转置恒等式的用法）

题给的正点方案是恒等式 `dWᵀ = Xᵀ @ dY`。本波**取了恒等式的 NT 排布，但把朝向定为直接产出 dW**（`dW = A@Bᵀ`，`A=dYᵀ 块 (BN,TT)`、`B=Xᵀ 块 (BK,TT)`），等价性由 §2.2 的 mad 参数 `m=BN,k=TT,n=BK` 与出口直写共同保证。**理由 = G-D4**：

`asc_copy_l0c2gm`（`port910b_compat.h:923-940`，A案 A-4 件）走 `copy_matrix_cc_to_gm` **行主序直出**，`nz2nd` 只做 NZ→ND 格式还原、**不做矩阵转置**。若按"先算 `dWᵀ`、出口再转回"，就得在出口补第二次重排（L0C→L1→MTE3→GM 三段路，本波未考古）。把**全部转置压在入口 DMA**，出口零成本 ⇒ 少一个未证真环节，也就少一个上卡失败模式。

因此本波对题给两案的取舍是：**"转置恒等式"胜出，但它的执行位置在 DMA 而非出口**；"direct mad（让 mad 吃 MN-major）"被 `madl1≡l0tr` 与 `madta FAIL` 双重否证。

### 2.4 语义自证（`l0tr` → 数学）

```
DSL:  T.copy(dY[kt*TT:(kt+1)*TT, n_tile*BN:(n_tile+1)*BN], dy_l1)        # dy_l1 = dY[块]      (TT,BN)
      T.copy(dy_l1, a_l0, transpose=True)                                 # a_l0  = dy_l1ᵀ      (BN,TT)
      T.copy(x_l1,  b_l0, transpose=True)                                 # b_l0  = x_l1ᵀ       (BK,TT)
      T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=(kt==0))       # acc += a_l0 @ b_l0ᵀ
生成码: asc_mmad(acc, a_l0, b_l0, 128, 64, 128, 0, 1, 0, (kt == 0))        # m=BN=128, k=TT=64, n=BK=128
  ⇒ acc = (dY块)ᵀ @ (X块)  =  Σ_t dY[t,n]·X[t,k]  形状 (BN,BK)  ← 正是 dW 的 (n_tile,k_tile) 格
出口:  T.copy(acc, DW[n_tile*BN:..., k_tile*BK:...])  → asc_copy_l0c2gm(..., 128, 128, 1024, 128, ...)
  ⇒ 行主序直写 DW 的第 n_tile 行带 / k_tile 列带，无转置、无越界（ks/nt 由编译期整除保证）
```

归约完整性：`for kt in T.serial(KS)`，`KS = T/TT = 2048/64 = 32`（生成码 `kt<32`）⇒ 全 token 轴覆盖；`clear_accum` 仅首块 ⇒ 跨块 fp32 串行累加，与 golden 的 `tiled_dw` 同序。

---

## 3. 判决矩阵（凭据，`verdict_log_D.txt`；每条均 `TILELANG_CACHE_DIR=$(mktemp -d)`）

```
L2   ----- variant=baseline -----  L3   D-DW-COMPILE-PASS
       has_nd2nz True, has_l1tmpl True, has_mmad False, has_l0c2gm True
       emitted: asc_copy_gm2l1_nd2nz asc_copy_l0c2gm ascend_gemm_l1<128,64,128,64,true,half> ...
L19  ----- variant=ntt -----       L20  D-DW-COMPILE-PASS
       has_dn2nz True, has_l1tmpl True  → 押在 G-D3 的 stub 上（故不作主案）
L36  ----- variant=l0tr -----      L37  D-DW-COMPILE-PASS   ← 主案
       has_nd2nz True, has_mmad True, has_l0c2gm True, has_l12l0_tr True
       emitted: asc_copy_gm2l1_nd2nz asc_copy_l0c2gm asc_copy_l12l0a_transpose
                asc_copy_l12l0b_transpose asc_mmad asc_set_gm2l1_nz_para ...
L53  ----- variant=madta -----     L54  D-DW-COMPILE-FAIL
       ---- first error ----
       /tmp/tmpg_7ca30f/tl_kernel.asc:11:11: error: only __ubuf__ and __gm__ and local memory pointer can be dereferenced
L83  ----- variant=madl1 -----     L84  D-DW-COMPILE-PASS  （emitted 集合与 l0tr 完全一致）
L100 建材回归 A2 纯 NT (fp16)      L101 D-A2-BASELINE-PASS  has_l1tmpl True has_l0c2gm True has_mix False
L104 建材回归 e2e_cube 直路        L105 E2E-CUBE-ONLY PASS
L107 golden 自检                   L108 D-GOLDEN-SELFCHK（数字见 §6）
```

补丁有效性因果链（防"其实一直绿"）：`l0tr` 在**未打 GAP-D 时**首挂
`error: use of undeclared identifier 'asc_copy_l12l0a_transpose'; did you mean 'asc_copy_l12l0b_transpose'?`；
只打 GAP-D1 后二挂 `port910b_compat.h:907:29: error: parameters too many`；
GAP-D1+D1b 齐 → PASS。同形状、同 cache 纪律，唯一变量是补丁 ⇒ 两缺口**都是必要前提**（原文见 `compat_gap_D.md` G-D1/G-D1b）。
反证：`madta` 打完补丁后仍 FAIL 且错误不变 ⇒ G-D2 是**另一条**独立缺口，不在 compat 层。

## 4. 形状 / tiling 鲁棒（`run_D.sh --shapes`，`verdict_log_D.txt:122-208`）

| T | N | K | BN | BK | TT | 生成码行 | 判决 |
|---|---|---|---|---|---|---|---|
| 2048 | 1024 | 1024 | 128 | 128 | 64 | 21 | PASS |
| 512 | 512 | 1024 | 64 | 64 | 32 | 23 | PASS |
| 4096 | 2048 | 512 | 128 | 64 | 128 | 23 | PASS |
| 2048 | **512** | **1024** | 128 | 128 | 64 | 21 | PASS（**非方阵** N≠K，P0 判据形状自检友好） |
| 2048 | 1024 | 1024 | **256** | **256** | 64 | 21 | PASS（见 §7 风险 R-1：L0 容量无静态校验） |

⇒ 结论不绑死单一形状；`T%TT==0`、`N%BN==0`、`K%BK==0`、四者 16 对齐是硬前提（`build()` 里 `_require_aligned` + `raise`，`d_dw_910b.py:56-58, 296-303`）。

## 5. `madl1 ≡ l0tr` 的原始 diff（direct-mad 否证的最硬证据）

```bash
cd attempts/D && diff dw_l0tr.asc dw_madl1.asc
3c3
< extern "C" __global__ __cube__ void dW_l0tr_kernel(__gm__ float* DW, __gm__ half* X, __gm__ half* dY) {
---
> extern "C" __global__ __cube__ void dW_madl1_kernel(__gm__ float* DW, __gm__ half* X, __gm__ half* dY) {
```

只有 kernel 名一行不同；DMA/mad/出口发射完全一致（`asc_copy_l12l0a_transpose(..., 0,0,4,8,4,8)` ×2、`asc_mmad(..., 128,64,128, 0,1,0, (kt==0))`、`asc_copy_l0c2gm(..., 128,128,1024,128, ...)`）。

## 6. golden 与上卡判据（`d_dw_golden.py`，交付物②）

三路真值同框（exact fp64 / tiled 复刻分块次序 fp32 / fp16-out），自检输出：

```
D-GOLDEN-SELFCHK   T=2048 N=256 K=256 TT=64 BN=128 BK=128 seed=0  in=float16 acc=float32 out=float32
exact   : norm=1208.44 maxabs=22.1562
tiled   : relFro(vs exact)=0.000e+00   <== 纯累加次序误差
fp16out : relFro(vs tiled)=0.000e+00
transposition-discriminator: relFro(exact,exact)=0.000e+00 relFro(exact,exact.T)=1.411e+00 => 可区分(P0 有效)
einsum-vs-matmul: relFro=0.000e+00
P0 转置方向: relFro(DW,exact) < 1e-2 且 relFro(DW,exact) < relFro(DW,exact.T)
P1 累加精度: relFro(DW,tiled) <= 2e-6 ; relFro(DW,exact) <= 1.0e-05
P2 逐元素  : assert_close(DW, tiled, atol=1e-3, rtol=1e-5)（对 exact 放宽 atol=1e-2）
D-GOLDEN-NPZ /tmp/dw_pairs.npz (3018590 bytes) keys=dY,X,exact,exact_T,tiled,fp16out,T,N,K,TT,BN,BK,p1_*,atol,rtol
```

**诚实标注（重要）**：`tiled relFro=0` 与 `fp16out relFro=0` 的成因是**值域设计**——输入取 `randint(-4,5)/8`（步长 `2^-3`），乘积步长 `2^-6`，`maxabs≈22.16 < 32`，全部落在 fp16 尾数可精确表示的格点上，故 golden 两路**没有**舍入散布。⇒ 这里的 `P1=2e-6` 是**理想下界**，不是实测容差。上卡真值来自 cube 内部按 16 行的二次分块，次序差可能把 `relFro(DW,tiled)` 推到 `1e-6~1e-5`；**建议上卡首轮用 `--seed` 换 randn/大值域复标 P1**，不要因 `2e-6` 未过就判功能失败。P0（方向）与 P2（逐元素）不受此影响。

## 7. 上卡证真点与遗留风险

| # | 待证真/风险 | 出处 | 首查动作 |
|---|---|---|---|
| **V2'** | 新补两个转置件的 **8 参位序**（`indexID=mStart`、`repeatTime=mStep`，`addrmode=false`、`dstFracStride=0`） | `compat_patch_D.h` / `patch_compat_D.py` | 实测发射 `asc_copy_l12l0a_transpose(a_l0, buf, 0, 0, **4, 8**, 4, 8)`：**转置形下 mStep=4=TT/16、kStep=8=BN/16**，与直觉相反 ⇒ codegen 的 m/k 按**源（L1）朝向**计。上卡若"方向对、数值散"先查这里 |
| V1 | `asc_copy_gm2l1_nd2nz` 的 32B 单位/NZ pad（继承 A案未证真项） | `port910b_compat.h:868-882` | 主案只用 nd2nz（不用 dn2nz），P0 一过即可同时证真 nd2nz 面 |
| V3 | `asc_copy_l0c2gm` stride 单位（继承 A案） | `port910b_compat.h:923-940` | 出口实测 `128,128,1024,128`（nRows/nCols/rowStride/colStride） |
| **R-1** | **L0 容量无静态校验**：`alloc_l0a/l0b/l0c`（`tilelang/ascend/language/allocate.py:38-62`）只声明 scope，不核尺寸。`BN=BK=256/TT=64` 这组的静态占用 = L0C 256KB(fp32) + L0A 32KB + L0B 32KB + L1 64KB，编译 PASS **不等于**硬件装得下 | §4 第 5 行 | 上卡前对照 CANN 910B L0/L1 规格复核，或把 `BN/BK` 上限锁到 128（本波前 4 组） |
| R-2 | `cores=64` 与 `ITER>1` 的写法（`TILES>BLOCKS`）编得过，但多 tile 串行时 **L0C/L1 复用无 `T.Pipelined`、无 unit_flag/事件同步**，性能与真并发正确性未验证 | `verdict_log_D.txt` lines=23 两组 | 属性能波范畴，本波不裁决 |
| R-3 | 动态 token 轴（生产是动态 m）与尾块谓词未支持：当前强制整除，否则 `ValueError` | `d_dw_910b.py:302-303` | 生产接入前需加尾块处理（`TT` 递减阶梯，参照 `gemm_bwd_dw_mps.py` 的 BLOCK_LADDER） |

## 8. 缺口清单摘要（全文与补丁片段 = `compat_gap_D.md`，交付物③）

| # | 缺口 | 层 | 本波处置 |
|---|---|---|---|
| **G-D1** | `asc_copy_l12l0a_transpose` **整条未定义**（codegen `codegen_ascend.cc:1563-1567` 硬发，arity 8） | compat | **已补**（`compat_patch_D.h`，官方 8 参件；建议合并回真源 §12 末尾） |
| **G-D1b** | 真源 `asc_copy_l12l0b_transpose` 体内发 **6 参**，910B 无此形态（官方三处证据均 8 参）⇒ 该发射体**从未可编**（`if constexpr` 死分支从未实例化） | compat | **已补**（`patch_compat_D.py` 就地改体，签名不动，可整块回退） |
| **G-D2** | **GM→L0A/L0B 无 DMA 通路**，codegen 静默退化成解引用 `__ca__` 的标量体（非法） | codegen/IR | **未补**（compat 补不了）；主案一律 GM→L1→L0 两段；建议上游 `src/ascend/op/copy.cc:533` 对 `GM→kL1ToL0A/B` 显式 `ICHECK` |
| **G-D3** | `asc_copy_gm2l1_dn2nz` = `nd2nz` 的**同名别名 stub** ⇒ GM→L1 转置搬运语义未落地 | compat | **未补**（需 `copy_gm_to_cbuf` 是否带 transpose 位的考古）；`ntt` 因此降级为性能候选 |
| **G-D4** | L0C→GM **无转置直出**（`copy_matrix_cc_to_gm` 行主序；`nz2nd` 不做矩阵转置） | compat/方案 | **方案层规避**：转置全压入口，出口直写 dW；若下游要 `dWᵀ` 朝向则需另立三段路 |
| **G-D5** | `asc_mmad` 硬编码 `kDirectionAlign=false` 且丢弃 `btbuf_ctrl` ⇒ mad 吃不了 MN-major，"硬件级 trans_A"无从谈起 | compat/硬件语义 | **未补**（需手册级证据）；它是"转置必须下移到 DMA"的**根据**，不是待办 |

## 9. 需编排者裁决的问题

1. **GAP-D1 / GAP-D1b 是否合并回主仓真源**（`src/tl_templates/ascend/port910b_compat.h`）？二者是 `l0tr` 主案的**必要前提**（不打补丁则 dW 无绿路）。本波严守硬边界只改容器 pip 副本，`compat_gap_D.md §1/§2` 已给出可直接贴的片段与 diff。
2. **G-D2（GM→L0 无 DMA 面）是否立为上游缺陷单**？现状是**静默生成非法代码**而非报错，对后续波次的排错成本极高（我在这条上烧了两轮编译）。建议最小修法 = `copy.cc` 路径判定点加显式 `ICHECK`。
3. **`ntt`（融合 L1 模板，性能最优候选）要不要补 G-D3 真转置件**？取决于是否要投入 `copy_gm_to_cbuf` transpose 位考古；不投入则 dW 只能走 `l0tr`（多一次 L1→L0 显式转置搬运，模板的 sub-K 双缓冲红利拿不到）。
4. **dW 出口朝向**：生产消费方是否要求 `dWᵀ` 或 NZ 布局直出给下一层？若要求，需为 **G-D4** 立 L0C→L1→MTE3→GM 三段路（A案 A-4 同类待办）。
5. **R-1 的处置**：是否给 `alloc_l0a/b/c` 加静态容量校验（本波发现 `BN=BK=256` 编译绿但占用 256KB L0C，超出常规档位）？这属主仓语言层，不在本波白名单。

6. **方言迁移面（生产件 → 910B）**：生产 `gemm_bwd_dw_mps.py:54-63` 是 2-D 核栅格 + `threads=128` + `T.clear(Cs)` + fp32 `alloc_shared` 累加；910B 侧必须换成 1-D 核栅格 + `T.Cube()` + `alloc_l0c` + `clear_accum=(kt==0)`，且 `T.Parallel`/SIMT 面本波已知不可用（已知坑）。要不要为"一份生产件跑双后端"立迁移约定（入口层按 target 分派 or DSL 层分支），请裁决。

## 10. 产物清单（`attempts/D/`）

| 文件 | 角色 |
|---|---|
| `d_dw_910b.py` | **交付物①** 5 变体 DSL 件 + `scan_source()` 发射面体检 + `D-DW-COMPILE-PASS/FAIL` 判决与首错误提取 |
| `d_dw_golden.py` | **交付物②** CPU fp32/fp64 golden 三路真值 + P0/P1/P2 上卡判据 + `--npz` 对拍件（含 `exact_T` 判别参照） |
| `compat_gap_D.md` | **交付物③** 6 条缺口 + 装配阶段表 + 补丁片段/真源回传 diff + VERIFY V2' + 防假绿自检 |
| `RESULT.md` | **交付物④** 本文件 |
| `verdict_log_D.txt` | 判决日志（`--matrix` + `--shapes` 全量 208 行（wc -l），§3/§4 的行号出处） |
| `env_setup_D.sh` / `run_D.sh` | 容器装配（幂等）/ 宿主机驱动（判决纪律内嵌） |
| `compat_patch_D.h` / `patch_compat_D.py` | GAP-D1 追加件 / GAP-D1b 就地改写件（幂等 + 锚不到显式报错） |
| `a2_baseline_ntt.py` | 建材回归对照件（纯 NT L1 fp16，证补丁未伤既有面） |
| `dw_l0tr.asc` `dw_ntt.asc` `dw_madl1.asc` `dw_baseline.asc` | 生成码取证（§5 diff 的两造） |
| `dw_pairs.npz` | 上卡对拍件 3018590 bytes（dY/X fp16 + exact/exact_T/tiled/fp16out + 阈值） |
