# U_readout_diag.md — P1-1h `letter_readout` readout 数值诊断（代理 U）

判决窗：`runs/1010-p2-13-oncard-wave2-cube-and-gate-e079`（真机 `NUM-readout-FAIL rel=0.1777896`）
本窗性质：**本地数值诊断**（notes.md:5 原文点名"readout 属本地数值诊断"），无卡。
工装：`reconcile/U_oracle.py`（六档尺子）。改动件：`ascend/kernels/letter_readout_asc.py`（仅两条 ascend 正文）。

---

## 0. 结论摘要（先给可复核的三句话）

1. **判决件与本件不对齐（最重要的前提事实，见 §1）**：真机那条 `NUM-readout-FAIL` 量的**不是**
   `letter_readout_asc.py`，而是 `run_numerics.py::num_readout()` 里从
   `attempts/B/b_readout_910b.py::build()` **逐字内联的 scores 投影 DSL**（含点积）。本件的
   ascend 正文在卡上从未被执行——同窗 notes.md:5 记录的 G-gate（`ascend_env` 探针用 950
   SimtVF 载体 → 判 eager → 九件内核全旁路）已把它挡在 torch 回退路上。**这条需上级裁决**（§8）。
2. **四条候选假设（ids/axis 语义、scatter 累加序与初始化、dtype 链、块切分/动态网格）全部被
   证据排除**：生产正文在 host 以 cpu 方言**真编译真发射**六个形状全 **bit-exact、rel=0.000e+00**
   （含重复行号最多 4 次命中、尾块非整除、dim 非 32B 整倍、picked=1）。数值偏差**不在算式**。
3. **落地了一件"语义逐位不变"的取号收敛（形态 B）**：`Idx[row]` 每行只向 GM 取一次进标量变量，
   守卫与偏移都读寄存器。理由：这是本案与八条真机**全绿**件之间唯一可复现的结构差异——
   只有 readout 用"拿刚载入的值现算地址、且在**最内层逐格重复三次**"的间接寻址（§3 判别表：
   `gen_readout.asc` 6 处，其余 17 份生成码 **0** 处）。**它是病灶候选的缓解，不是确证后的定点修复**，
   确证要等下窗真机 rel（§8）。

---

## 1. 前提核对：真机那条 FAIL 究竟量了谁（H5）

| 事实 | 凭据 |
|---|---|
| `num_readout()` 自带正文，不 import 本件 | `run_numerics.py:311` 注释"DSL 逐字内联自 attempts/B/b_readout_910b.py::build()"；`:315 def num_readout` → `:348 tilelang.compile(build(), target="ascend", out_idx=-1)`；全文件无 `letter_readout` 字样 |
| B 域脚本不能 import（只能内联） | `run_numerics.py:20-22`："B 域不列入 sys.path：…是 module 级脚本（末尾 `sys.exit(main_())`），importlib 整脚本执行会直接终止本进程；两件已按其 build() 逐字内联" |
| 判据阈值出处 | `run_numerics.py:371` `assert r < 1e-5, "…仿真底噪 4.0e-7；若 S 全零则 rel≈1"`；`:372` 方向判别 `assert r < r_bad - 0.5`（`r_bad`=拿首位置换 POS 的对照答案） |
| 本件 ascend 路在卡上被执行了吗 | **没有**。notes.md:4-5：`train_step 因 ascend_env 探针用 SimtVF 判 eager 全旁路(grad_fn 缺失)`；G-gate 修复（P1-1g，`ascend_env.py:158-170`）之后才有 910B 合法探针，但**卡上跑的仍是 `num_readout` 那份内联正文** |

**所以本窗的可执行射程**：把"共享同一条下射路"的本件 ascend 正文诊断清楚并收敛，同时把判决件
的病灶候选（同一结构，在 b 形态正文里）如实上报——`run_numerics.py` 在**禁改清单**内，
不能本窗动。

对照：同窗 `num_addln()`（真机绿，rel=1.69e-05）也**不是** `add_ln_asc.py`，同为内联件——
即"绿"也不给本件背书，只有 G 波宿主仿真给阈值背了书。

---

## 2. 假设账本（H1-H4 排除 / H5 前提 / H6 病灶候选）

复跑命令统一为（host，`release/.venv`）：

```bash
.venv/bin/python ascend/kernels/reconcile/U_oracle.py --shadow   # ① 影子重放
.venv/bin/python ascend/kernels/reconcile/U_oracle.py --prod     # ⑥ 生产正文真跑（本窗新增档）
.venv/bin/python ascend/kernels/reconcile/U_oracle.py --cpu      # ② 镜像正文真跑
.venv/bin/python ascend/kernels/reconcile/U_oracle.py --b-shadow # ④ b 形态同源对照
# 容器（cann910b-j，/work=release）
docker exec cann910b-j bash -c 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; export \
  BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201; cd /work && \
  TILELANG_CACHE_DIR=/tmp/u_cache python3 ascend/kernels/reconcile/U_oracle.py --dump'   # ③ 生成码体检
docker exec cann910b-j bash -c '... python3 ascend/kernels/reconcile/U_oracle.py --harden' # ⑤ 三形态判决
```

### H1 ids/axis 语义错位（int64/int32 截断、末位轴 vs 首轴、守卫边界）→ **排除**

- 入口链：`forward` 里 `idx = ids.to(torch.int32).contiguous()`；`_plan`（`letter_readout_asc.py:198-217`）
  对非 int32/非连续直接回 `None` ⇒ 走 torch 回退，**不会带截断进内核**。int64 越界值同样先被
  `_check`（`:289-298`）拦下。实测：
  `FORWARD-OOB-torch.int32 被入口契约拒绝: 行号越界：需落在 [0, 8)，实得 [-1, 9]`；
  `FORWARD-OOB-torch.int64 被入口契约拒绝: 行号越界：需落在 [0, 8)，实得 [-1, 9]`。
- 轴向：正文 `Out[row, j] = Rows[k, j]` 与 cpu 参考 `rows[ids]`（首轴 gather）、`index_add_`
  （首轴 scatter）同轴；`_plan` 还强制 `src.size(-1)==dim==dst.size(-1)`。
- 生成码逐字对照（容器 `--dump`，修复后）：
  `k = Idx[block_idx*8 + i]; condval = Rows[k*64 + j]`，界守卫 `if (((0 <= k) && (k < total)))`
  ——与 cpu 语义 `rows[ids[row]]` 一致，越界读回 `float(0)`。
- 数值凭据：`--prod` 六形状 `gather_rel=0.000e+00 gather_bitexact=True`（下表 §6.1）。

### H2 scatter_add 累加序 / 初始化（漏清零、同行命中丢账重账）→ **排除**

- 清零在入口：`backward` 先 `acc = torch.zeros((row_count, dim), ...)`（`:278`）再交内核，
  正文注释"目标表必须先清零：本件只做'加一笔'，不做'盖一笔'"。
- 累加形态（生成码，`--dump`）：RMW 两侧都在 bypass 路上，自洽无跨路读旧值——
  ```c
  if (((0 <= k) && (k < total))) condval = tl::read_gm_bypass_dcache((__gm__ float*)GRows + k*64 + j);
  else condval = float(0);
  tl::write_gm_bypass_dcache((__gm__ float*)GRows + k*64 + j, (condval + GOut[i*64 + j]));
  ```
  且 910B 下 `read/write_gm_bypass_dcache` 与普通标量访问**同一实现**
  （`tilelang/src/tl_templates/ascend/dcache_bypass.h:14-18`：`TL_PORT910B_NATIVE_TYPES` 下
  `#define TL_GM_LOAD(ptr) (*(ptr))`）⇒ "bypass 名不副实"不构成丢账机制。
- 数值凭据：三重/四重命中同行为真加多次——`--prod` 的 `odd-dim`（`[9,9,9,...]`）与 `dup-heavy`
  （行 2、5 各命中 3 次、8 命中 2 次）`scatter_bitexact=True`。单块串行 `T.Kernel(1)` 无竞争。

### H3 dtype 链（fp32 出口契约 vs 中间截位）→ **排除（量级差 60×）**

- 本件全链 fp32：`_plan` 要求 `src.dtype==dst.dtype==float32`，正文无 `T.cast`，fp32→fp32 同宽。
- 判决件（b 形态）才有 fp16 入口（`HDT="float16"`），故用 `--b-shadow` 量它的位宽下界：
  ```
  [b-shadow] 源级复刻 rel=3.893e-07   rel_firstpos=1.496e+00
  [b-shadow] 全 fp16 乘加链底噪 rel=3.030e-03   ← 把乘与加全压到 fp16 也只到这个量级
  真机 rel=1.777896e-01
  ```
  最坏 fp16 链底噪 3.03e-03 与真机 1.78e-01 差近两个数量级 ⇒ 不是舍入/截位可解释的偏差。

### H4 块切分与动态网格（`GATHER_ROWS_PER_BLOCK=8` 与动态 ids 的交互）→ **排除**

- host 发射码（`get_host_source()` 实测）：`int32_t picked = Idx_shape[0]`、
  `int32_t total = Rows_shape[0]`、grid = `(picked + 7) >> 3`、FFI 参数按签名序打包 ⇒ 尾块数与
  实参都对得上，缓存键 `f"{kind}[{name}]|d{dim}"` 不含动态行数（换批不换产物，正确）。
- 尾块激励实测：`picked=6`（网格 1 块、2 格被守卫）、`picked=12`（网格 2 块）、`picked=1`、
  `picked=8`（恰整除）全部 `gather_rel=0.000e+00`；`--shadow` 另含 `picked=0` 退化网格
  （影子层 `SOURCE-SEMANTICS-PASS`，真编译层见 §7 附带发现）。

### H5 前提错位 → **坐实**（见 §1，本窗最大发现，需裁决）

### H6 病灶候选：间接寻址在最内层逐格重复 → **结构坐实、机理待真机**（§3、§4）

---

## 3. 绿红判别表（`attempts/*/gen_*.asc` 全量扫描，17 份生成码）

复跑：

```bash
cd release/ascend/port910b/attempts && for f in */gen_*.asc; do \
  i=$(grep -o "Idx\[\|POS\[\|IDS\[" "$f" | wc -l | tr -d ' '); \
  b=$(grep -o "read_gm_bypass_dcache\|write_gm_bypass_dcache" "$f" | wc -l | tr -d ' '); \
  echo "$(basename $f) 载入值当下标文本处=$i bypass标量腿=$b"; done
```

| 生成码 | 载入值当下标 | bypass 标量腿 | 真机结果（wave1/2 记录） |
|---|---|---|---|
| `gen_readout.asc`（**红案**） | **6** | 1 | `rel=0.1777896` FAIL |
| `gen_addln.asc` | 0 | 2 | `1.69e-05` 绿 |
| `gen_rope_table.asc` / `gen_rope_rational.asc` | 0 | 3 | `3.13e-08` 绿 |
| `gen_gdn_fwd_*.asc` / `gen_gdn_bwd_*.asc`（7 份） | 0 | 3~13 | `6.35e-08` 绿 |
| `gen_gdn_scalar_*.asc`（3 份） | 0 | 1 | 绿 |
| `gen_attnsw_pure.asc` / `gen_attnsw_stage.asc` | 0 | 1 | 绿 |
| `gen_ubstage_*.asc`（2 份） | 0 | 0 | 绿 |

红案那 6 处的原文（`attempts/B/gen_readout.asc:11-12, 22-23`）——同一行号在**同一个逐格循环体里
取三次**（守卫两次 + 地址一次），且地址由刚载入的值现算：

```c
if (((0 <= POS[block_idx*4 + i]) && (POS[block_idx*4 + i] < 32))) {
  condval = HID[(block_idx*65536 + i*16384 + (int64_t)POS[block_idx*4 + i]*512) + j];
...
if (((0 <= IDS[t]) && (IDS[t] < 4096))) {
  condval_1 = W[((int64_t)IDS[t]*512) + j_1];
```

其余 17 份（含全部真机绿件）的标量 GM 读写地址**一律仿射**（只含 `block_idx`/循环变量/常量），
需要非仿射下标的地方都改走批量搬运（`TA.copy(X[bx*BM,0], x_ub)`，见 `run_numerics.py` 绿案正文）。
⇒ "标量 GM 读"本身不是红因（绿件里就有 4-5 处，第一条假设曾这样写、被这条证据推翻），
**"载入值当下标 + 最内层逐格重复"才是红案独有结构**。

机理假说（**未证实，只能由真机判**）：910B 上数据相关地址的标量载入若在某次重复取号时读到
不一致值，会出现**同一行内部分格子取错行**的图案——这与 rel=0.178 落在 (0,1) 的"部分格子错"
观测一致（守卫若整块失效会读出 0，rel→≈1；地址全错也 ≈1；都不符合）。收敛成"每行一次取号"
后，同行 64 格**共用一个寄存器值**，最坏也只整行错、不会行内分裂。

---

## 4. 三形态判决（容器 `--harden`，形状 rows=(24,64) picked=20 rpb=8，ids 含重复+尾块）

```
[harden] 形状 rows=(24,64) picked=20 rpb=8 ids 含重复/尾块
  A 现行标量散读   cpu真跑 bitexact=True rel=0.000e+00   ascend PASS len=1024 每行取号文本计数=3 MTE批量=False 标量出口=True 界守卫=True
  B 取号收敛标量   cpu真跑 bitexact=True rel=0.000e+00   ascend PASS len=920  每行取号文本计数=1 MTE批量=False 标量出口=True 界守卫=True
  C MTE 整行装填   cpu真跑 bitexact=True rel=0.000e+00   ascend PASS len=1477 每行取号文本计数=2 MTE批量=True  标量出口=True 界守卫=False
```

| 形态 | 语义 | 界守卫 | 附加约束 | 判定 |
|---|---|---|---|---|
| A 收敛前（逐格 `Rows[Idx[row], j]`） | bit-exact | 有 | 无 | 基线（红案同构） |
| **B 取号收敛（每行一次 `k = Idx[row]`）** | bit-exact | **有** | 无（不依赖 `dim%8`） | **选它**：与 A 等价、结构差异消掉、生成码更短 |
| C 整行走 `T.copy` | bit-exact | **消失** | GAP-B `asc_copy_gm2ub_align` 32B 粒度**静默截断**（fp32 ⇒ `dim%8==0`） | **否决**：守卫没了 + 撞已知缺口，越界 ids 变 OOB DMA |

选 B 的三条硬理由：① 数值等价性有 host 真跑背书（A/B/C 三者 cpu 真跑都 bit-exact）；
② 只压结构、不改语义，不新增任何被真机证伪过的形态；③ C 的"界守卫消失 + 32B 静默截断"
是已知会静默出错的组合（H 波 `compat_gap_B.md`），代价大于收益。

---

## 5. 落地改动（只动 ascend 两条正文；cpu 正文与接口零改动）

`letter_readout_asc.py:105,110,112`（gather）与 `:180,182,184`（scatter）：

```python
# gather_asc_impl
with T.Kernel(T.ceildiv(picked, GATHER_ROWS_PER_BLOCK)) as bx:
    k = T.alloc_var("int32")   # 行号寄存器：每行只向 GM 取一次（P1-1h 收敛）
    for i in T.serial(GATHER_ROWS_PER_BLOCK):
        row = bx * GATHER_ROWS_PER_BLOCK + i
        if row < picked:
            k = Idx[row]
            for j in T.serial(dim):
                Out[row, j] = Rows[k, j]

# scatter_add_asc_impl
with T.Kernel(1) as bx:
    k = T.alloc_var("int32")   # 行号寄存器：每行只向 GM 取一次（P1-1h 收敛）
    for i in T.serial(picked):
        k = Idx[i]
        for j in T.serial(dim):
            GRows[k, j] = GRows[k, j] + GOut[i, j]
```

等价性论证（为什么这不改语义）：`Idx` 在本件里是**纯读**全局缓冲（`mark_scalar_dcache_bypass.cc:44-78`
的 bypass 集合只收**有写**的全局缓冲），无任何写者，串行 `T.serial` 内读多次与读一次同值；
`k` 的活跃区间严格等于 `row`/`i` 的那一轮，不进下一轮判断。语义、越界行为、出口布局全部不变，
变的只有"向 GM 取号的次数"。

`git diff --stat` 自证：本件 `24 insertions(+), 6 deletions(-)`，其中代码行只有上面 4 处，
其余是 docstring 补注（模块【怎么做】①、`gather_asc_impl`、`scatter_add_asc_impl`）。
`gather_cpu_impl` / `scatter_add_cpu_impl` / `_plan` / `_run` / `forward` / `backward` / `_check` /
`_IMPLS` **一行未动**（`git diff -U0` 的 hunk 头核对）。

---

## 6. 验证记录

### 6.1 host 生产正文真跑（`--prod`，本窗新档：把 `sys.modules["tilelang.ascend.language"]`
指向 cpu 方言后**真编译真发射仓库里那两条生产正文**，免镜像漂移）

```
[prod] 生产正文真跑 方言=tilelang.cpu.language GATHER_ROWS_PER_BLOCK=8（② 镜像档用 rpb=8）
  OK  dup(尾块)   total= 12 dim= 16 picked=  6 gather_rel=0.000e+00 gather_bitexact=True scatter_rel=0.000e+00 scatter_bitexact=True dup=True
  OK  exact(整除) total= 16 dim= 64 picked=  8 gather_rel=0.000e+00 gather_bitexact=True scatter_rel=0.000e+00 scatter_bitexact=True dup=True
  OK  odd-dim     total= 10 dim= 26 picked=  9 gather_rel=0.000e+00 gather_bitexact=True scatter_rel=0.000e+00 scatter_bitexact=True dup=True
  OK  single      total=  8 dim= 32 picked=  1 gather_rel=0.000e+00 gather_bitexact=True scatter_rel=0.000e+00 scatter_bitexact=True dup=False
  OK  wide-tail   total=  5 dim=  8 picked= 12 gather_rel=0.000e+00 gather_bitexact=True scatter_rel=0.000e+00 scatter_bitexact=True dup=True
  SKIP empty      picked=0 ⇒ tilelang 运行期 stride 校验拒 Static stride mismatch（§7）
  OK  dup-heavy   total=  9 dim= 32 picked= 12 gather_rel=0.000e+00 gather_bitexact=True scatter_rel=0.000e+00 scatter_bitexact=True dup=True
[prod] PROD-BODY-ORACLE-PASS（生产件正文自身，非镜像复刻）
```
判据 `REL_TOL=1e-5`（与 `run_numerics` 的 readout 阈值同尺），实测 **rel=0.000e+00 且逐位同**，
覆盖"多 ids 命中同行（最多 4 次）/非整除尾块/dim 非 32B 整倍/picked=1"。越界 ids 由入口契约
拦截（§2-H1 实测两条），影子层另含 `picked=0` 全绿。

交叉验证同绿：`--shadow` 6/6 `SOURCE-SEMANTICS-PASS`（含 picked=0）；`--cpu` 镜像档 5/5
`DUAL-DIALECT-ORACLE-PASS`（picked=0 记 SKIP）。

### 6.2 容器编译判决（`cann910b-j`，手法照 `reconcile/verify_letter_readout.py`）

```
keys=['ascend|gather[ascend]|d64', 'ascend|scatter_add[ascend]|d64']
.o~942B simt_or_atomic_in_src=False bounds_guard_in_src=True
H-READOUT-ASC-COMPILE-PASS
exit=0
U-READOUT-ASC-COMPILE-PASS
```
（`U-...` 是 `exit=0` 的门控回声，不改判决件本身。）

修复后生成码（`--dump`，`/tmp/u_gather.asc`，len=922）——每行一次取号、界守卫仍在、出口 bypass：

```c
int32_t k = 0;
for (int32_t i = 0; i < 8; ++i) {
  if (((((int32_t)block_idx) * 8) + i) < picked) {
    k = Idx[(((int64_t)block_idx) * 8) + ((int64_t)i)];
    for (int32_t j = 0; j < 64; ++j) {
      float condval;
      if (((0 <= k) && (k < total))) condval = Rows[(k * 64) + j];
      else condval = float(0x0p+0f);
      tl::write_gm_bypass_dcache((__gm__ float*)Out + (block_idx*512 + i*64 + j), condval);
```

### 6.3 host 全量回归

```
cd release && .venv/bin/python -m pytest tests -q -m "not integration"
→ 332 passed, 3 skipped, 3 deselected, 2 warnings in 36.85s   (rc=0)
```
基线登记为 326；本窗实测 332——**多出的 6 条来自并行波 P1-1g（代理 T）新增的
`tests/test_ascend_env_probe.py` 守卫用例**（tasks.md 的 P1-1g 完成凭据已记同一数字），非本窗引入
（本窗未加任何 tests，禁改区）。0 failed ⇒ 无回归。

---

## 7. 附带发现（都在禁改区，只上报不修）

1. **`picked=0` 编不出可用产物**：`ValueError: Static stride mismatch for parameter GOut:
   expected 16 at index 0, got 0`（tilelang 运行期对动态维为 0 的 stride 校验）。而
   `_run` 里 `kernel(*args)` **不在 try 内**（`letter_readout_asc.py:220-236`）⇒ 空 ids 批次在
   昇腾路上会直接抛，而不是降级 eager。与本窗改动无关（收敛前后同一表现），属接口健壮性。
2. **`backward` 缺 `forward` 的值域契约**：`forward` 调 `_check`（拒越界行号），`backward` 只查
   维度/个数，实测 `backward 调用 _check: False`。越界 ids 的反向靠生成码界守卫兜（读回 0、
   不落全局），两种方言是否同兜未逐一对齐——不对称，建议与第 1 条一并归到接口波。
3. **>2D 的 lead 语义可疑**：`forward` 用 `rows.reshape(-1, dim)` 摊平前导维当一张表，返回时又
   `.reshape(*lead, idx.numel(), dim)`；`backward` 的 `acc.reshape(*lead, row_count, dim)` 在
   `lead` 非空时元素数对不上会抛。语义（多张表各抽行 vs 一张大表）需要一条明确契约。
4. **910B 的 "dcache bypass" 名不副实**：`dcache_bypass.h:14-18` 在 `TL_PORT910B_NATIVE_TYPES`
   下把 `TL_GM_LOAD/STORE` 定义为 `*(ptr)`，即与普通标量 GM 访问同一实现——凡依赖"bypass 真
   绕开 dcache"的推理在 910B 都不成立（本案用它排除了 H2 的丢账机制，方向相反但同样要紧）。

---

## 8. 遗留与下窗取证设计（需裁决项）

**待裁决（本窗射程外，任选其一或组合）**
- 判决件错位（H5）：要么 (a) 把 `run_numerics.py::num_readout()` 的正文同步做同款取号收敛
  （它在本窗**禁改清单**内，需上级授权）；要么 (b) 让 `num_readout` 改调
  `letter_readout_asc.forward/backward(target="ascend")`，把"数值验收"和"交付件"对齐；
  要么 (c) 新增一条走真接口的 case（同样越界，属 proposal 级）。
- G-gate 之后的重测窗：`train_step` / 九件内核在卡上首次真执行时，`letter_readout_asc` 的
  ascend 路才算真被测到——本件的收敛效果届时才有独立读数。

**下窗判据（沿用 `run_numerics` 阈值）**：`rel < 1e-5` 且 `rel < r_bad - 0.5`（方向判别）。
最小对照两条，能一步分离"取号结构"与"fp16 入口"两个候选：
- 对照 1：b 形态正文原样 + **仅**把 `POS[...]`/`IDS[...]` 收敛成每行一次（同 §5 手法）；
  若绿 → 病灶确证为间接寻址重复取号；若仍红 → 排除，转下条。
- 对照 2：b 形态入口 `HDT` 由 `float16` 换 `float32`（其余不动）；
  若绿 → 病灶在 fp16 装填/截位路（与 §2-H3 的 3.03e-03 底噪对照解读）。
两条各只改一处，同窗跑完即可定案；仍红则要考虑"数据相关地址的标量散读"本身在 910B 不可用，
那时光滑出路只剩"索引先行 MTE 批量装填 + 核内仿射散取"（形态 C 的安全版：先把 Idx 整册搬进
UB，再逐格从 UB 取号——守卫须手写回，因 codegen 已丢）。

**本窗状态**：本件四条假设清账、形态 B 落地并三档对拍 bit-exact + 容器编译 PASS + host 无回归；
真机 rel 待下窗（无卡，无法本地闭环）。
