# U2_matrix_k48 — C2 的补档矩阵：**非对齐 K 档 × bk** 的 host 双方言判决

> 任务归属：p2-13 / U2 / 交付物 3。**只新增本文件，不改 `C2_probe_dtype_matrix.py` 的任何既有结论文字**。
> 目的：把"C2 的 9/9 为何漏掉真机 P1-1q（`linear_prod` k=48 数值错）"从推测变成**表格事实**，
> 并给出 C2/S 两套档位矩阵今后**必须固化的口径**。
> 生成码侧的取证在 `U2_K48_EVIDENCE.md`（本文件只谈数值判决矩阵与覆盖口径）。

---

## 1. 判决矩阵（m=16, n=96, bn=32；fp32；两方言轨 + 一条静态判据轨 + 一条地址轨）

**复跑**（本窗实测 `exit=0`，全文留档 `scratch/U2_k48/probe_host_final.txt`）：

```bash
cd $REPO_ROOT/release
.venv/bin/python ascend/kernels/reconcile/U2_probe_k48_host.py
#   分轨单跑：--tracks cpu | alias | pred | addr
```

| 轨 | 含义（**方言轨必须分开读**） | 判到的面 / **判不到的面** |
|---|---|---|
| **T1 `cpu` 方言** | `ascend_env.dialect("cpu")` + `gemm_scalar_body` 真编真跑，bk **由探针强喂**（不经 plan） | 判：**同一份算式的数学**（含 `bk∤k` 的表达窗外档）。判不到：产品路由/缓存键 |
| **T2 `cpu-alias`** | 产品路由（`gemm_kernel.forward(…, target="ascend")`）+ `tilelang.ascend.language` 两处指向 cpu 方言 + c 后端三件套；配方同 `run_numerics._prod_setup`；带 `_route_key` 假绿守卫 | 判：**正文 + plan + 缓存键 + 入口件**这一整套产品面。**判不到 MTE2 跨步 DMA**（`T.copy` 走 c 后端 lowering，不是 ascend 那份实现） |
| **T3 `pred`** | 纯静态：按 `src/ascend/op/copy.cc` 的 `plan_single_row` 复算 ascend 侧会编成 SINGLE 还是 STRIDED | 判：**哪些档真机上会触达步长槽**（不判数值） |
| **T4 `addr`** | 纯地址算术（生成码实参 `32,64,…,192,64` 的读上界 vs 张量跨度） | 判：**哪种步长语义会读越界**（host 可判，不需卡） |

### 1.1 主表（`rel = max|got-ref| / max|ref|`，尺子 = 产品 `REL_TOL_LIN = 2e-5`，本探针不改阈值）

| k | ladder_bk | nk | T3 ascend DMA 形态 | T1 cpu 方言 `none` / `relu` | T2 cpu-alias `none` / `relu` | 是否 IN-PLAN | 档位定性 |
|---|---|---|---|---|---|---|---|
| 16 | 16 | 1 | **SINGLE** `1×64B` | 1.185e-07 / 9.927e-08 | 1.316e-07 / 1.039e-07 | ✅ | 对齐，步长槽不参与 |
| 32 | 32 | 1 | **SINGLE** `1×128B` | 2.993e-07 / 1.682e-07 | 1.746e-07 / 1.849e-07 | ✅ | 对齐 |
| **48** | **16** | **3** | **STRIDED** `32×64B, srcStride=192B, dstStride=64B` | **9.067e-08 / 9.282e-08** | **1.239e-07 / 1.172e-07** | ✅ | ★**真机 FAIL 档；本地两轨全绿 ⇒ 不复现** |
| 64 | 64 | 1 | **SINGLE** `1×256B` | （bk=64 档 T1 未强喂）/ 见 T2 | 3.409e-07 / 2.404e-07（bk=64） | ✅ | 对齐 |
| **80** | **16** | **5** | **STRIDED** `32×64B, srcStride=320B` | 9.478e-08 / 1.035e-07 | 1.306e-07 / 1.245e-07 | ✅ | ★**同类跨步档，从未上过卡**（本窗新识别的潜在第二病档） |
| **96** | **32** | **3** | **STRIDED** `32×128B, srcStride=384B` | 1.556e-07 / 1.525e-07 | 2.040e-07 / 1.939e-07 | ✅ | ★**跨步档；卡上有产物（2 枚）但从未被逐元素直判** |

> ⚠ **本表 T2 列的读数拍于 21:05；21:13:53 M2 并发改了 `plan()` 的选档（`_pick_bk_ascend` 优先 `bk==k`）⇒ k=48/80/96 三格的 T2 缓存键与数值已换轨，最新读数见 §6。T1（强喂 bk）两版下均全绿，判决线不变。**

**表达窗外档（bk 与 plan 的 ladder 不一致，`k % bk != 0` 时静默截尾）——只登记形态，不计 PASS/FAIL**：

| k | 强喂 bk | T1 cpu 方言 rel | 说明 |
|---|---|---|---|
| 16 | 32 | 1.073e+00 / 9.669e-01 | `bk > k` ⇒ `nk = k//bk = 0`，正文一轮都不累加 ⇒ **错值**（plan 永远给不出这档，故不入判决） |
| 32 | 16 | 1.076e-07 / 1.633e-07 | 整除成立（nk=2），只是 plan 不选它 ⇒ 数值正常 |
| 48 | 32 | 7.040e-01 / 6.312e-01 | `48 % 32 ≠ 0` ⇒ **`nk` 整除截尾静默丢 16 列** ⇒ 错值。**这条量化了"plan 无 K 守卫"的后果**（`gemm_asc.plan` 只 `_largest_block()` 取整除块宽，一旦有旁路喂入非整除 bk 就无声算错） |
| 64 | 16 / 32 | 1.053e-07 / 1.664e-07 | 整除成立 |
| 80 | 32 | 3.669e-01 / 5.853e-01 | `80 % 32 ≠ 0` ⇒ 同上静默截尾 |
| 96 | 16 | 1.260e-07 / 1.291e-07 | 整除成立 |
| T2（产品路由）| — | **不出现** | plan 只会给 IN-PLAN 档 ⇒ 窗外档在产品面上不可达（这是好事，也是矩阵必须靠 T1 强喂才覆盖得到的原因） |

### 1.2 T4 地址轨（判"越界"而非判"数值"）

```
k=48 bk=16  P(pitch) 读上界=18432  span=18432  in-bounds     ← codegen 的假设自洽（恰好读满 W）
k=48 bk=16  G(gap)   读上界=20416  span=18432  OOB +1984B    ← 若原生件把 stride 当"块间空隙"
k=80 bk=16  G(gap)   读上界=32704  span=30720  OOB +1984B
k=96 bk=32  G(gap)   读上界=40832  span=36864  OOB +3968B
k=16/32/64         dma=SINGLE —— 步长参数不参与（blockCount=1）
```

### 1.3 语义直读探针的 host 自测（`--dev cpu`；它同时是"解码器正确"的证明）

```bash
.venv/bin/python ascend/kernels/reconcile/U2_probe_k48_dma_semantics.py --dev cpu --acts none,relu
```

| 档 | 测得地址映射图 | 候选失配（升序，第 1 名=硬件语义） | 该档判别力 |
|---|---|---|---|
| k=48/none & /relu | `OK=1536 行错位=0 列错位=0 UNATTRIBUTED=0` | **P mismatch=0** ＜ Z 48 ＜ G/Gd 720（写越界块 768）＜ Gs 1264（读越界 128）＜ C 1488 | **强** |
| k=96/none & /relu | `OK=1536 行错位=0 列错位=0 UNATTRIBUTED=0` | **P mismatch=0** ＜ Z 48 ＜ G/Gd 720（写越界块 1536）＜ Gs 1264（读越界 256）＜ C 1488 | **强** |
| k=32/none & /relu（ctrl） | `OK=1536 …` | P mismatch=0，**但 C、G 也 =0** | **弱**（`blockCount=1` ⇒ 步长槽不参与，探针自证"这档判不出语义"） |

`U2-DMA-READOUT-DONE unexpected=0（dev=cpu）` ⇒ c 后端在 pitch 语义下装填正确，**判不到 ascend**。

---

## 2. 关键判定（任务书要求的那一条）

> **本地不复现。**

- **方言轨标注**：不复现的两轨是 **`cpu` 方言轨（T1）** 与 **`cpu-alias` 产品路由轨（T2）**；
  **真·ascend 方言轨在 host 无法编译**，非"跑了不坏"，而是**结构上量不到**：
  ```
  AttributeError: Check failed: (reg != nullptr) is false:
      Operator tl.tileop.ascend_copy is not registered
  ```
  （凭据：`ascend_env.dialect("ascend")` + `gemm_scalar_body` + `compile_kwargs("ascend")` 直编，本窗复跑命中。）
- 合计 `cells_failed=0`（T1/T2 的 IN-PLAN 档全绿），`U2-LOCAL-VERDICT` 同调。
- ⇒ 按任务书分支：这是**纯逻辑缺陷假设被否**、**缺陷面隔离到 DMA/装填面**，
  修复与判决**必须下一次短窗**（增采清单见 §4）。
- ⇒ 因此本代理**未改** `gemm_asc.py` / `gemm_kernel.py`（取证不指向它们，见 `U2_K48_EVIDENCE.md` §2.2/§3）。

---

## 3. C2 的 9/9 为什么漏掉它（覆盖盲区根因，两层）

### 3.1 铁证：C2 的 linear 形状族里 **没有一枚 `bk < k` 的档**

`reconcile/C2_probe_dtype_matrix.py:72`（原文，本窗只读未改）：

```python
LIN = ((33, 64, 32, "none"), (8, 64, 32, "relu"), (16, 32, 32, "none"))
```

三档 **k 全是 32**，`_largest_block(32)=32` ⇒ 每档 `bk == k` ⇒ 按 `plan_single_row`
**恒折叠成单发连续 DMA**，`srcStride`/`dstStride` 两个槽**一次都没被赋过有意义的值**。
⇒ 不是"C2 判据松"，是 **C2 的档位集合与缺陷变量正交**：无论跑多少遍、tol 收到多紧都判不出来。

同理 `run_numerics.LIN_SHAPES` 里另外四档（k=32/64/64/64）也全是 `bk == k`；
**整个 host/卡上判据族里，只有新加的 (16,96,48) 一枚跨步档** —— 而它偏偏只在真机上才编得出跨步
（host 的 c 后端不体现步长语义）。这就是"9/9 与真机 FAIL 并存"的确切机制。

### 3.2 第二层盲区：双方言技巧本身有一道判不到的接缝

S/C2 依赖的 `cpu-alias` 技巧把"**路由/缓存键/正文算式**"翻成产品件、把"**执行**"翻到 c 后端。
它的闭环边界是**算式层**；而 P1-1q 的嫌疑在**MTE2 步长语义层**——两层之间没有传递性：

```
   正文算式（DSL） ──✓ 可在 host 双方言闭环──► 数值
   T.copy 的 lowering 形态（单发 vs 跨步 / 步长语义）──✗ host 结构上不可达──► 真机数值
```

⇒ 只要形状族不区分 SINGLE/STRIDED，**alias 轨的绿对真机 DMA 面零信息量**（本窗 §1.1 实测正是如此）。

### 3.3 应固化的档位口径（写死进 C2/S 的矩阵定义；本窗不动那两个文件）

1. **必含非对齐 K 档**：`k ∈ {48, 80, 96}`（对应 `bk ∈ {16, 16, 32}`，全部 `bk < k`）。
   口径不写成"某个具体形状"，而写成不变量：**至少一枚 `bk < k`（=STRIDED）的档**。
2. **必标 DMA 形态列**：矩阵每格加 T3 判据（`plan_single_row` 等价式 `blockCount==1 or srcStride==blockLen`）
   ⇒ 让"这格有没有测到步长槽"变成**表格上可见的事实**，而不是评审时的口头推断。
3. **必配"语义直读"轨而不是只配数值轨**：`bk<k` 的档在 host 只会 PASS（本窗证实），
   所以矩阵必须显式声明"host 判绿 ≠ 真机判绿"，并让该格指向一条真机命令（§4）。
4. **`bk ∤ k` 的窗外档**（如 k=48/bk=32 ⇒ rel 7.0e-01）应作为**反例档保留**：
   它量化了"`plan` 无 K 守卫 ⇒ 整除截尾静默算错"的代价，是"plan 的整除前提"唯一有数字的凭据。
5. **C2 的 9/9 结论不回改**：本文件是**补档**，不动既有结论文字；既有 9/9 的表述只在
   "fp32/fp16 双方言 dtype 面"这个范围内仍然成立（该范围本身没错，只是覆盖口径要按 1–4 扩）。

---

## 4. 必须下窗才能判的最小增采（一条命令级，不扩案）

```bash
# 一枪判清 910B GM→UB 多突发跨步 DMA 的步长语义（单进程，一次发射/档）
python3 ascend/kernels/reconcile/U2_probe_k48_dma_semantics.py --dev npu
```

它判什么、判完怎么读，全在脚本 docstring 与 `U2_K48_EVIDENCE.md` §8；要点：

- 语义无关：一次发射**量出真实地址映射图**，与 6 种候选（P/C/G/Gs/Gd/Z）的预测图对拍 ⇒ 失配最小者即硬件语义；
- 附带直接报出 `读越界/写越界块/UNATTRIBUTED`（= "没报 507015 却出 NaN" 该看的量）；
- **自带判别力自证**（§1.3）：k=32 那档会被标"弱"，避免"ctrl 档同解"被误读成"语义已证真"；
- `--dev cpu` 已在本窗自测通过（`unexpected=0`），故下窗零调参成本。

**建议同窗加的第二条（仍是"一条命令级"，判覆盖缺口而非新案）**：给 `linear_prod` 的
`LIN_SHAPES` 增补 k=96 档做一次真机数值直判。理由：§5（EVIDENCE）已证 k=96 的 fp32/fp16 两枚产物
**在卡上被训练步发射过、却从未被逐元素判过**，而它和 k=48 同走跨步 DMA；若 H1 成立，
它就是 P1-1r（fp16 step0 反向 NaN）的头号嫌疑形状。**此项须上级批准**（会动判据集合的形状表，
本代理白名单只允许动文案字符串，故未自行执行）。

---

## 5. 本文件带来的其它事实（顺手登记，均 host 可复跑）

- **k=80（bk=16, nk=5）是第二枚潜在病档**：T3 判它 STRIDED、T1/T2 判它 host 绿，
  但归档 46 枚里**没有** k=80 产物 ⇒ 它只是"没上过卡"，不是"上过卡且好"。
- `plan` 的整除前提**没有 K 守卫**：§1.1 窗外档用它自己的数字（48/32 ⇒ rel 7.0e-01）量化了
  一旦有旁路绕过 `_largest_block()` 会发生什么。产品面当前不可达 ⇒ 记为纪律性事实，非缺陷。
- relu 出口在全部档上 `neg=0`（T1/T2 两轨，含 k=48）⇒ **"relu 出口分支"致错被本地直接否**，
  与真机 `m16n96k48/none` 照样坏（`rel=1.150e+00`）互相独立地指向同一结论。

---

## 6. 收尾期复核（21:14–21:16）：plan 选档被并发改动 → **T2 那三格换轨**（本表最新读数）

`release/ascend/kernels/gemm_asc.py` 在 **21:13:53** 被 **M2**（P1-1p 在途修法，非本代理）加了
`_pick_bk_ascend()`：ascend 轨优先 `bk=k`（整行连续，UB 预算 24576B 内）。
⇒ 产品路由（T2）在此之后**不再给 k=48/80/96 选 `bk<k`**，本文件的 **T3 静态判据（ladder 口径）
与 T2 实测键从此不一致**，两边都登记，不改历史格：

```
_pick_bk_ascend(16,32,k,elem) 直调（bn=32）：k=16/32/48/64/80/96 → pick=16/32/48/64/80/96，
  全部 bk==k ⇒ DMA=SINGLE（elem=4 与 elem=2 同）
```

T2 重跑（`.venv/bin/python ascend/kernels/reconcile/U2_probe_k48_host.py`，`exit=0`，21:16）：

| k | T2 缓存键（新） | T2 none / relu rel（新） | 与 §1.1 的差异 |
|---|---|---|---|
| 48 | `ascend\|gemm[ascend]\|n96k48`**`b16x32x48`**`anonedfloat32` | 2.076e-07 / 1.611e-07 | 键由 `x16`→`x48`；**该格不再是跨步档** |
| 80 | `…n96k80`**`b16x32x80`**`…` | 2.758e-07 / 2.139e-07 | 同上（`x16`→`x80`） |
| 96 | `…n96k96`**`b16x32x96`**`…` | 3.408e-07 / 4.858e-07 | 同上（`x32`→`x96`） |

**判决线不变**：T1（cpu 方言，探针强喂 `bk=16/32`）与 T2 在新旧两版 plan 下**都全绿**
（`U2-PROBE-DONE cells_failed=0`）⇒ "**本地不复现**"这一判定不受并发改动影响。

**对下窗增采的直接影响（本窗已就地修）**：`--dev npu` 原来经 `gemm_kernel.forward()` → `plan()`，
在新选档下**编不出 `blockCount>1`**，那一条命令会退化成"平凡全对"、把 H1 从"未证真"变成"测不到"。
本窗把 npu 轨改为**强钉 ladder bk、经 `gemm_asc.run()` 走同一 impl/compile/搬运参数**
（产品件一字未动），并在发射前打印 `U2-BK-POLICY dev=npu 强钉 bk=…`。host 自测复跑：

```bash
.venv/bin/python ascend/kernels/reconcile/U2_probe_k48_dma_semantics.py --dev cpu --acts none,relu
# U2-MEASURE k=48 /none bn=32 bk=16 nb=3 nk=3 dma=strided | OK=1536 行错位=0 列错位=0 UNATTRIBUTED=0
# U2-DISCRIMINATE k=48 /none 与 P 同解的其它候选=无 ⇒ 判别力=强
# U2-DMA-READOUT-DONE unexpected=0（dev=cpu）
```

**§3.3 口径追加第 6 条**：矩阵的"STRIDED 档可达性"**必须与 plan 的选档策略绑定记录**——
选档一改，档位性质就改；因此固化口径应写成"**至少一枚探针以强钉 bk 显式构造 `bk<k`**"，
而不是"依赖 plan 恰好给一个 `bk<k` 的形状"。
