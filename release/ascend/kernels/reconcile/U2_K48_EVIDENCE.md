# U2_K48_EVIDENCE — P1-1q（`linear_prod` 真机 K=48 数值错）的生成码取证

> 任务归属：p2-13 / 执行代理 U2 / 交付物 1。**不占卡**：全部结论只引用已归档的真机产物 + 本地可复跑算式。
> 一手件：`runs/1011-p2-13-oncard-accept-2002/artifacts/sys1_kernels.tgz`
> （552,236B / 422 条目 / **md5 `f2ba9a2080de73d1d04d4a309dc8496e`**，本窗实测复核同值）。

---

## 0. 一句话结论

两枚生成码**不是"结构相同只有常数不同"**——它们在 W 块的搬运形态上发生了**结构性分叉**：
`k=32` 编成 **1 条单发连续突发**（步长槽根本不参与），`k=48` 编成 **32 条多突发跨步**
（`blockCount=32, blockLen=64B, srcStride=192B, dstStride=64B`）。除此之外，**A 侧读、UB 布局、
累加索引、偏置、出口地址、行号守卫、relu、同步事件配平全部逐字同构且自洽**。
⇒ 根因**不在 codegen 的索引数学，也不在边界数学**（这两项已被逐行证伪，且被 `dw_prod` 同形状
PASS 交叉证伪），而收敛到唯一一条未被任何数值直判证明过的假设：
**910B 原生件 `copy_gm_to_ubuf` 的 `srcStride`/`dstStride` 到底是"首地址间距(pitch)"还是
"块间空隙(gap)"**——该轴在本仓只被定性过单位（32B 块），从未被证真数值
（`attempts/C/compat_gap_C.md` G-C4/G-C5 原文自陈"**定性有据，数值仍待上卡证真**"）。

---

## 1. 复跑入口（每条都可复跑）

```bash
cd $REPO_ROOT/release
# ① 解包一手件（322 个内核目录）
tar -xzf runs/1011-p2-13-oncard-accept-2002/artifacts/sys1_kernels.tgz -C scratch/U2_k48
# ② DMA 形态普查（本窗新建，产出 §5 的表）
.venv/bin/python ascend/kernels/reconcile/U2_probe_k48_gen_digest.py
# ③ 两枚主角原文
cat -n scratch/U2_k48/0.1.15/linux-aarch64/kernels/5d732a8f*/device_kernel.asc   # k=48
cat -n scratch/U2_k48/0.1.15/linux-aarch64/kernels/77a8a3c4*/device_kernel.asc   # k=32
```

| 角色 | 短哈希 | `device_kernel.asc` sha256（manifest.json 原录） | size/行数 | 身份 |
|---|---|---|---|---|
| **病档** | `5d732a8f84…` | `3b84ae22d88804da6deee0f610c28977953a92bd4e37efad3ddcecba4510fc70` | 3258B / 64 | `gemm_impl_kernel` fp32 `m` 动态、**n=96 k=48 act=relu** |
| **对照** | `77a8a3c492…` | `929a99cb31ceb7874bf9e1db10b4c9567f584c73cbd1d2f36385d164b516b0d8` | 2963B / 58 | `gemm_impl_kernel` fp32 n=96 **k=32** act=none |

形状从生成码自身的常数反推，不靠命名：`A` 行步长 `* (int64_t)48` / `* (int64_t)32` ⇒ k=48 / k=32；
出口 `C + my*1536 + i_3*96` ⇒ n=96；`block_idx*1536` 与 `block_idx*32` ⇒ bn=32、nb=3。

---

## 2. 逐行 diff：三处实质差异 + 八处"确认同构"

### 2.1 实质差异（全部集中在"K 被切成几轮 + W 块怎么搬"）

| # | 位置 | k=32（`77a8a3c4`） | k=48（`5d732a8f`） | 性质 |
|---|---|---|---|---|
| **D1** | 18 / 18-19 | `for i_1…` 直接开跑（**无 K 轮外层**，nk=1） | `for (int32_t ko = 0; ko < 3; ++ko)`（nk=3） | K 分轮，nk=k/bk |
| **D2** | 29-31 / 30-32 | 每 `my` 一趟 `wait/notify` 事件对 | 每 `my`×每 `ko` 一对 ⇒ 共 3 对 | 与 D1 同源的配平，**计数自洽**（§2.3-⑧） |
| **D3** | **30 / 31** | `asc_copy_gm2ub_align(&ub[0], &W[block_idx*1024], `**`1, 4096, 0,0,0, mode, 4096, 4096`**`)` ⇒ **单发连续 1×4096B**，`srcStride`/`dstStride` 槽**不参与** | `asc_copy_gm2ub_align(&ub[0], &W[block_idx*1536 + ko*16], `**`32, 64, 0,0,0, mode, 192, 64`**`)` ⇒ **多突发跨步 32×64B，srcStride=192B(=k·4) dstStride=64B(=bk·4)** | ★**唯一"会改变装进来的数值是谁"的差异** |

原文（行号真实）：

```
5d732a8f  k=48   18:    for (int32_t ko = 0; ko < 3; ++ko) {
                    31:      asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&((__ubuf__ float*)buf_dyn_shmem)[0])),
                             (__gm__ uint8_t*)((&(W[((((int32_t)block_idx) * 1536) + (ko * 16))]))),
                             32, 64, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 192, 64);
77a8a3c4  k=32   30:    asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&((__ubuf__ float*)buf_dyn_shmem)[0])),
                             (__gm__ uint8_t*)((&(W[(((int32_t)block_idx) * 1024)]))),
                             1, 4096, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 4096, 4096);
```

D3 不是偶然：它是 tilelang 折叠判据的必然产物
（`tilelang/src/ascend/op/copy.cc` 的 `plan_single_row`，本窗复核原文）——

```c
bool plan_single_row = analyzer->CanProveEqual(plan.n_rows, I(1)) ||
    (CanProveEqual(plan_src_stride_bytes, plan_row_bytes) &&
     CanProveEqual(plan_dst_stride_bytes, plan_row_bytes));
```

W 块是 `(bn 行 × bk 列)`、GM 上行步长 = `k·elemsize`。`bk == k` ⇒ 行步长 == 行长 ⇒ **折叠成单发**；
`bk < k` ⇒ **一律 blockCount=bn 的多突发跨步**。⇒ "K=48 为什么特殊"在 codegen 层面有了确定答案：
**特殊的不是 n=96、不是 relu，而是 `bk(16) < k(48)`**。

### 2.2 确认同构（逐项排除，附行号）

| 项 | 取证 |
|---|---|
| ① A 侧标量夹址读 | `5d73:22-27` `A[min(my*16+i_1, m-1)*48 + ko*16 + kk]`；`77a8:21-26` `A[min(...)*32 + kk]` ⇒ 行步长 = k ✓、夹址守卫 ✓、仅差 `ko*bk` 一项（D1 的正解） |
| ② **UB 三区不重叠** | k=48：`w_ub` 字节 `[0,2048)`（32×16 fp32）、`c_ub` `[2048,4096)`（index `+512`）、`a_ub` `[4096,5120)`（index `+1024`）；k=32：`w_ub[0,4096)`、`a_ub[4096,6144)`（`+1024`）、`c_ub[6144,8192)`（`+1536`）⇒ 两版各自自洽，无重叠 |
| ③ 累加索引 | `5d73:40` `a_ub[i_2*16+kk_1+1024] * w_ub[j_1*16+kk_1]` ⇄ `77a8:39` `a_ub[i_2*32+kk_1+1024] * w_ub[j_1*32+kk_1]` ⇒ 与 ② 的块宽严格一致；`w_ub[j_1*bk+kk]` 正是 `dstStride=bk·4` 的紧凑堆放 ⇒ **UB 内消费侧与搬运侧参数互洽（在 pitch 语义下）** |
| ④ 累加器清零 | `5d73:13-15` / `77a8:13-15` 同式，写 `c_ub` 区，标量置 0 ✓ |
| ⑤ K 轮累加 | `5d73:45` `c_ub[...] += acc`（每 ko 累一次）；`77a8:44` 同式只累一次 ⇒ 与 nk=3/1 匹配，**无重复累加/漏累加** |
| ⑥ 偏置 | `5d73:52` / `77a8:50` 逐字同式 `bias[block_idx*32 + j_2]` ✓ |
| ⑦ **出口地址** | `5d73:57` 与 `77a8:51` **完全同一表达式** `C + my*1536 + i_3*96 + block_idx*32 + j_2` ⇒ n 向多块组合、写回切片两处一致；relu 版与非 relu 版只差 `53-56` 那 4 行 `__cond_0` 夹负 ⇒ **"relu 出口分支"不可能是错值来源**（`none` 档 `77a8` 无夹负行，同样只有一个出口式） |
| ⑧ 同步事件配平 | k=48 每轮：`wait(S→MTE2)`(30) → `copy`(31) → `notify(MTE2→S)`(32) → 消费首点 `wait(MTE2→S)`(38) / 末点 `notify(S→MTE2)`(42)；3 轮 × 一对，**wait/notify 计数相等**，收尾 `62` 的 `wait(S,MTE2)` 与末轮 notify 配对 ⇒ 不是漏等/多等（k=32 是同一套式子的 nk=1 退化） |

**判读（任务书要的"哪一处会致反号/NaN"）**：只有 **D3**。
① ② ③ ⑤ ⑥ ⑦ ⑧ 全部只能产生"同样错误的正确算法"，改不了"装进来的元素是谁"；而真机指纹是
`got=15.24771 / ref=-15.37733`（**幅度对、符号反**＝典型的"取到了另一行 W"）与 `96 个 NaN`
（**非法位型或读到未初始化字节**）——这两类形态**只能由地址算错产生**，不可能由 ③⑤ 的索引式产生
（若索引式错，host 两轨早就复现了，见 §7）。

### 2.3 对照实验把"act"这一维干净隔离（补严一个取证缺口）

**缺口如实登记**：归档 10 枚 `gemm_impl` 里**只有 `5d732a8f`（k=48）带 relu 夹负行**
（`grep -c __cond_0` = 2，其余 9 枚 = 0）⇒ H 案 1 的 `m16n96k48/**none**` 腿的生成码**不在这份归档里**，
所以 §2.1 的对照（k=48·relu ⇄ k=32·none）在形式上混了两个变量。补一枪本地对照消歧：

```bash
cd release && .venv/bin/python - <<'EOF'   # 同形状 m16 n96 k48 bk16，只变 act
# gemm_scalar_body(cpu 方言, act_mode=none|relu) → get_kernel_source() → unified_diff
EOF
# ACT-VAR-DIFF lines=6          → 差异仅 @@ -56,0 +57,3 @@ 的 3 行：if (v[0] < 0) v[0] = 0;
# ACT-VAR-DMA-SAME: True        → 搬运/索引/出口行**逐字相同**
```

⇒ act 变量在 codegen 层**只等于那 3 行出口夹负**，与 §2.2-⑦ 的"出口地址式两处同式"合并可得：
**relu 这一维不可能进入 D3 所在的装填面**。又因真机 `k=48/none` 腿本身就以 `rel=1.150e+00` 坏掉
（notes.md ②-1，独立于任何生成码对照），"relu 出口分支致错"被**两条互不依赖的证据**同时否掉。


---

## 3. 回答任务书的二选一

> "若两枚生成码**结构相同只有常数不同**，那就说明根因不在 codegen 而在形状参数或边界数学。"

本例**落在另一支**：两枚**结构不同**（多出 `ko` 外层轮、多出每轮一对事件、**W 块从单发连续变多突发跨步**）。
但"结构不同"里要再分一层：

- **不同的部分**（D1/D2）是 `nk=3` 的**正确**展开，已逐项证自洽（§2.2 ①⑤⑧）；
- **真正新增的语义**只有 D3 引入的**两个步长槽**——它们是本例里**唯一"生成了但从未被证明过"**的东西。

⇒ 定性：**不是 codegen 索引缺陷，不是边界数学缺陷，而是"codegen 的步长假设 × compat 的步长映射 ×
原生件的真实步长语义"三者一致性从未被数值闭环**。修复面因此**大概率落在
`ascend/port910b/patches/port910b_compat.h`（不在本代理白名单）**，而不是 `gemm_asc.py`。

槽位映射本窗复核（`port910b_compat.h:601-610`，10 参 → 原生 7 参）：

```c
asc_copy_gm2ub_align(dst, src, int blockCount, int blockLenB, int rsvd0, int rsvd1,
                     int rsvd2, asc_load_l2_cache_mode mode, int srcStrideB, int dstStrideB)
  → __cce_scalar::copy_gm_to_ubuf(dst, src, /*sid*/(uint8_t)0, (uint16_t)blockCount,
        TL910B_BYTES_TO_BLK(blockLenB), TL910B_BYTES_TO_BLK(srcStrideB), TL910B_BYTES_TO_BLK(dstStrideB));
                                            //  ×2      ×6                 ×2   ← k=48 实发换算结果
```

槽位（10 参的顺序）**已被 attempts/C 修对**（G-C5：早先 §10b 形参名与 codegen 实发错位，`nBurst` 槽
收过 `srcStride`），单位（32B 块）也已被 X 取证关掉
（`attempts/X_FRACTAL_EVIDENCE.md` §0 裁决③：`DEFAULT_C0_SIZE=32` + 地址模型
`kernel_check_data_copy_overflow.h:505-520`）。⇒ **剩下唯一活口就是 gap-vs-pitch 这一轴**，
而它恰好只被 `blockCount>1` 的档位（=只有 k=48/k=96）触达。

---

## 4. 交叉证据：同形状的 K=48 数学是清白的

`gemm_bwd_dw_asc.py::dw_scalar_body` 与 `gemm_asc.gemm_scalar_body` 是**同一族可移植标量正文**，
但 **dW 件的 A/dY 侧全程夹址标量读、一条 `T.copy` 都没有**（`grep -c asc_copy_gm2ub_align` 于归档
`dw_impl_kernel` = **0**，见 §5 全表 `no-DMA`），而它在真机上：

```
dw_prod ✅ rel=1.36e-07 dir_disc=1.39e+00 reuse=4/4 shapes=4 track=card   ← 含 (8, 96, 48) 档
```

⇒ **同一个 n=96 / k=48 / bn=32 / bk=16 的形状组合，在没有跨步 DMA 的那条腿上数值全绿**。
这是"缺陷面被隔离到 DMA/装填"最硬的一条卡上旁证，且它来自**本窗之前的独立 case**，非本次推断。

---

## 5. 全局普查：46 枚产物里只有 3 枚产品内核走过跨步 DMA

凭据：`.venv/bin/python ascend/kernels/reconcile/U2_probe_k48_gen_digest.py`（本窗新建，
判据 = `plan_single_row` 的文本等价形：`blockCount==1 或 srcStride==blockLen ⇒ SINGLE`）。

```
U2-DIGEST 条目=46（gemm_impl=10，其余=36）
U2-DIGEST 走 STRIDED（多突发跨步）DMA 的内核：4 枚
  18502bd3793e  main_kernel(__mix__(1,2), bf16)   dma=[(1,524288,524288,524288), (2048,256,4096,256)]   ← 非产品件
  5d732a8f8483  gemm_impl_kernel  fp32  nk=3      dma=[(32,  64, 192,  64)]   ★ k=48（P1-1q 病档）
  a421afa3a1e6  gemm_impl_kernel  fp32  nk=3      dma=[(32, 128, 384, 128)]   ★ k=96
  ea1059e6995f  gemm_impl_kernel  fp16  nk=3      dma=[(32,  64, 192,  64)]   ★ k=96（fp16：384B/2=… 96·2）
```

`gemm_impl` 十枚全表（SINGLE 七枚、STRIDED 三枚）：

| 短哈希 | 位宽 | 反推 k | DMA |
|---|---|---|---|
| 275b6049c297 | fp32 | 32 | SINGLE `(1,4096,4096,4096)` |
| 3249db2f3b39 | fp32 | 64 | SINGLE `(1,8192,8192,8192)` |
| 32d6c39c6877 | fp32 | 64 | SINGLE `(1,16384,…)` |
| **5d732a8f8483** | **fp32** | **48** | **STRIDED (32,64,192,64)** |
| 77a8a3c492a8 | fp32 | 32 | SINGLE `(1,4096,4096,4096)` |
| 8f30a21f8fd4 | fp16 | 32 | SINGLE `(1,2048,2048,2048)` |
| **a421afa3a1e6** | **fp32** | **96** | **STRIDED (32,128,384,128)** |
| e07da648b2cc | fp16 | 32 | SINGLE `(1,2048,2048,2048)` |
| **ea1059e6995f** | **fp16** | **96** | **STRIDED (32,64,192,64)** |
| f3ab55cdbfe3 | fp32 | 16 | SINGLE `(1,2048,2048,2048)` |

其余 36 枚（`dw_* / rope / gdn* / gather / scatter / add_ln / ln_bwd / …`）**要么 no-DMA（纯标量夹址），
要么 `blockCount=1` 的整块连续**（`add_ln`/`ln_bwd` 的尾块长度是 `min(16,…)*128` 运行期表达式，但
`blockCount` 恒为 1 ⇒ 步长槽不参与）。

⇒ "真机只有 K=48 出错"这条**观察**，在生成码层面被升级为**完备枚举**：
**跨步 DMA 只出现在 3 枚 gemm 前向产物里；其中 k=48 那枚被数值直判了（FAIL），k=96 两枚从未被
逐元素直判（只被"loss 下降"弱判据覆盖）。** 这两枚正是训练步 `_LinearFn.backward` 里
`dA = gemm_kernel.forward(dC, W.t().contiguous(), …)` 在 qkv 层（W=(96,32)）的实际形状
⇒ 与 P1-1r（fp16 step0 反向生 NaN）**同源嫌疑**（假设级，见 §8）。

---

## 6. 地址核算：哪种步长语义会读越界（纯地址算术，host 可判）

凭据：`.venv/bin/python ascend/kernels/reconcile/U2_probe_k48_host.py --tracks addr`

```
U2-ADDR k=48 bk=16 dma=strided  P(pitch)     读上界=18432B  span=18432B  in-bounds
U2-ADDR k=48 bk=16 dma=strided  G(gap)       读上界=20416B  span=18432B  OOB +1984B
U2-ADDR k=80 bk=16 dma=strided  G(gap)       读上界=32704B  span=30720B  OOB +1984B
U2-ADDR k=96 bk=32 dma=strided  G(gap)       读上界=40832B  span=36864B  OOB +3968B
（k=16/32/64 三档打印 dma=SINGLE —— 步长参数不参与，无从判别）
```

- **P（pitch）语义**：末块尾址 `12416 + 31·192 + 64 = 18432` **正好等于** W 的字节跨度 ⇒ codegen 的
  假设是"恰好读满整个 W、不多不少"，**自洽且无越界**。
- **G（gap）语义**（原生件把 `srcStride` 解释为"块间空隙"⇒ 实际间距 = `blockLen + srcStride`）⇒
  读上界越出 **1984B**；`dstStride` 同被解释为 gap ⇒ 写落点变成 `r·128B`，后 16 块**写出 `w_ub` 之外、
  踩进 `c_ub` 累加区**（本窗探针把它量化成"写越界块=768"，见 §7 命令）。
- 越界 1984B 在 torch caching allocator 的同一池内 ⇒ **不必然触发 507015**，但会拿到任意位型
  （含 NaN）——与真机指纹 `inf=0 / 无 507015 / 96 个 NaN / 反号错值`**相容**。
  反之 `18502bd3` 那枚 `blockCount=2048` 的跨步件越界量是 KB~MB 级 ⇒ 与 §12/intrinsic 腿的
  **507015** 也相容（统一假设见 §8，**假设级**）。

---

## 7. 本地复现结果（判决：不复现）与 host 的结构性上限

详见 `reconcile/U2_matrix_k48.md`（k×bk 全矩阵）。此处只取判决线：

```bash
.venv/bin/python ascend/kernels/reconcile/U2_probe_k48_host.py        # exit=0
#   U2-PROBE-DONE cells_failed=0
#   U2-LOCAL-VERDICT: k=48 两轨均 PASS ⇒ 本地不复现
.venv/bin/python ascend/kernels/reconcile/U2_probe_k48_dma_semantics.py --dev cpu
#   U2-MEASURE k=48/none … OK=1536 行错位=0 列错位=0 UNATTRIBUTED=0
#   U2-CANDIDATE k=48/none P(pitch,pitch) mismatch=0 …（其余候选 mismatch=48~1488）
#   U2-DMA-READOUT-DONE unexpected=0
```

**host 判不到的那一面，是结构性限制而非偷懒**（凭据）：

```bash
.venv/bin/python -c "… tilelang.compile(gemm_scalar_body(ascend_env.dialect('ascend'), …),
                                        **ascend_env.compile_kwargs('ascend'))"
# AttributeError: Check failed: (reg != nullptr) is false:
#   Operator tl.tileop.ascend_copy is not registered
```

ascend 方言的算子注册表在 macOS host 不加载 ⇒ **`T.copy` 的 MTE2 lowering 在 host 无法触发**；
`cpu-alias` 轨的 `T.copy` 走的是 **c 后端 lowering**（另一份实现）。⇒ 本地两轨能且只能判
"**正文算式 + 产品路由 + 缓存键**"这一层（判绿），**结构上量不到步长语义**（这正是 C2/S 两轮 9/9
漏掉 P1-1q 的根因，写进 `U2_matrix_k48.md` §3）。

---

## 8. 统一假设 H1（**假设级，未证真**，留给下窗一枪判决）

> **H1**：910B `__cce_scalar::copy_gm_to_ubuf` 的 `srcStride`/`dstStride` 按
> **gap（块间空隙）** 而非 **pitch（首地址间距）** 解释；compat 与 codegen 按 pitch 喂值 ⇒
> 所有 `blockCount>1` 的多突发跨步 GM→UB 搬运都读/写错地址。
> 越界小 ⇒ 静默错值/NaN（P1-1q）；越界大 ⇒ 507015（§12/intrinsic 腿，`18502bd3` 那类件）。

支持它的三条：① 只有 `bk<k` 档失败（§5 完备枚举）；② 失败腿的错值形态是"取到别人家的数"
（反号、幅度对）而非算术漂移；③ 同形状无-DMA 腿（`dw_prod`）全绿。
反对/未排除的：`attempts/C` 引的 `matmul_client.h:2745` 里 `srcGap` 一词只出现在 **L1/L0 件**的参数表，
GM→UB 的 7 参件官方文档原文本仓没有（原生头文件只在容器里）⇒ **不能凭这个词结案**。

**一枪判决件**（一条命令，单进程，不扩案）：
`reconcile/U2_probe_k48_dma_semantics.py --dev npu` —— 它不假设任何语义：把 W 编码成
"每元素自带门牌号 `i*1000+j`"、A 取 one-hot，则出口 `C[p][n]` 就是"被装进 UB 槽 (n,p) 的那个 GM
元素的门牌号"，**一次发射直接量出硬件真实的地址映射图**，再与 6 种候选语义的预测图对拍。
host 自测已证解码器正确（§7）且**自带判别力自证**（跨步档=强，`bk==k` 档=弱）。

**若 H1 成立，修复点与代价**（都不在本代理白名单，须上级裁决）：
`port910b_compat.h::asc_copy_gm2ub_align` 把 `srcStrideB/blockLenB` 换成 `srcStrideB-blockLenB`
（并对 `dstStrideB` 同理），属**搬运适配层单点改动**、影响面 = 所有 `blockCount>1` 的件（=本窗
普查的 3 枚 gemm + intrinsic 腿）。

---

## 9. 本窗未做（如实登记）

- 未改 `gemm_asc.py`/`gemm_kernel.py`：取证不指向它们（§2.2 八项同构 + 本地两轨全绿 + `dw` 旁证），
  无根据的改动等于碰运气，且会把"未证真的假设"写进产品件。
- 未改 `port910b_compat.h`：**不在白名单**，且改它需要先证 H1（只有卡能证）。
- 未跑真机：任务书明令"不占卡"；§8 那条命令就是留给下一次短窗的最小增采。
- **归档缺口**（上报，不属本代理可修）：`sys1_kernels.tgz` 的 46 枚里**缺 k=48 的 act=none 产物**
  （以及 H 案各变体的产物——H 驱动与主链不同批打包）。下次开窗若仍用聚合/多案混跑，
  请**在 H 案之后再打包生成码**，否则"逐变体比生成码"这条取证路会一直缺件。

---

## 10. 本窗收尾期的**并发写面**与它对本取证的影响（21:13 后复核，如实登记）

`git status` 显示 `release/ascend/kernels/gemm_asc.py` 于 **21:13:53** 被改（**非本代理**——是 M2 的
P1-1p 在途修法，其注释直接引用了本取证的归档件 `5d732a8f…`/`ea1059e6…`）：新增
`_pick_bk_ascend()`，在 `plan()` 的 ascend 轨把选档从"ladder 最大整除块"改成
"**先试整行连续 `bk=k`（UB 预算 24576B 内），否则退回 ladder 且要求单突发 ≥128B**"。

本窗复跑实测（bn=32，`_pick_bk_ascend(16,32,k,elem)` 直调）：

```
k=16 pick=16 SINGLE   k=32 pick=32 SINGLE   k=48 pick=48 SINGLE
k=64 pick=64 SINGLE   k=80 pick=80 SINGLE   k=96 pick=96 SINGLE     （elem=4 与 elem=2 同）
```

⇒ 三点后果，必须写清：

1. **本取证的主证据链不受影响**：归档 `.asc`（改动前生成的既有真机产物）、§2 的逐行 diff、
   §6 的地址核算、§7 的 T1（cpu 方言、探针**强喂** `bk=16`）全部与 plan 的选档无关。
   **受影响的是 T2（cpu-alias）的缓存键**：改动后产品路由给 k=48 的键由 `…b16x32x16…` 变
   `…b16x32x48…`（k=80→`b16x32x80`、k=96→`b16x32x96`，本窗复跑原文见
   `U2_matrix_k48.md` §6）⇒ **数值仍全绿，但那一格测的已不是跨步档**。
2. **§8 那条一枪判决件原本会因此失焦**：`--dev npu` 走 `gemm_kernel.forward()` → `plan()`，
   在新选档下**再也编不出 `blockCount>1`**，探针会" trivially 全对"，把"**未证真**"悄悄变成
   "**测不到**"。本窗已把 npu 轨改为**强钉 ladder `bk`、经 `gemm_asc.run` 走同一 impl/compile
   路径**（产品件一字未动），并打印 `U2-BK-POLICY` 自证。
   复跑凭据：`.venv/bin/python ascend/kernels/reconcile/U2_probe_k48_dma_semantics.py --dev cpu
   --acts none,relu` → `U2-MEASURE k=48/… bk=16 … dma=strided`、`U2-DMA-READOUT-DONE unexpected=0`。
3. **对缺陷定性的影响（重要，但属上级裁决）**：`_pick_bk_ascend` 若被采纳，它是
   "**绕开步长槽**"而非"**证真步长语义**"。H1（§8）在两种情形下都还挂着：
   ① 其它仍会编出 `blockCount>1` 的件（本窗普查到非产品件 `18502bd3…` blockCount=2048，
   与 §12/intrinsic 腿 507015 同源嫌疑）；② `bk==k` 的大突发一旦超 UB 预算，plan 会自动落回
   跨步档（`ladder*elem ≥ 128B` 那条分支）⇒ **跨步面并未被产品封死**。
   故 §8 那一枪仍值得打（一次发射、单进程），否则修复只是"把雷绕开、没把雷拆掉"。

**写面冲突提示给编排者**：`gemm_asc.py` 同时在 U2 与 M2 的可改清单上；本窗 U2 **未写**该文件，
但它在本窗最后一次回归（21:14+）之后被 M2 改过 ⇒ 本报告 §① 的三项回归是**在含 M2 改动的工作树上
复跑**的（340 passed / PASS=9 FAIL=0 / C2 18/18 均为 21:16 的新读数），非旧读数引用。
