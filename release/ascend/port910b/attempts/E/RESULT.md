# RESULT.md — 执行代理 E / p2-13-ascend-runtime 甲路 P1-1 波
## RoPE（旋转位置编码）与 attn_sw（因果滑窗注意力 softmax 侧件）的 910B tilelang DSL 件

**判决**：`E-ROPE-COMPILE-PASS` ×3（`table` 前向 / `table` 反向 `sign=-1` / `rational` 骨架；
每次全新 `TILELANG_CACHE_DIR`）＋ `E-ATTNSW-COMPILE-PASS` ×2（`stage` / `pure`）
＋ `E-GOLDEN-PASS`（rope/attn 双语义 golden 自检全过）＋ 基线 `E2E-CUBE-ONLY PASS` 无回归。
取证反例：`E-ROPE-COMPILE-FAIL`（`--mode sinf`，**预期 FAIL**，作为"自算路不可走"的可复跑反证，不计入主判决）。
compat 缺口登记 **4 条（G-E1…G-E4）**，**零补丁实付**（compat v3=944 行现成消费面直接编过），见 `compat_gap_E.md`。
**验收上限=compile PASS**（本机无 NPU，未上卡；数值判据 R1–R6 已固化在 `e_golden.py` 头注释与 npz 件名里）。

凭据文件：`verdict_log.txt`（全矩阵一次跑完）、`gen_rope_table.asc`、`gen_rope_rational.asc`、
`gen_attnsw_stage.asc`、`gen_attnsw_pure.asc`、`_dbg_e.py`（golden 指标伪影分离取证件）。

---

## 0. 结构结论：rope / attn_sw 的每个语义段落到了哪类件

| 语义段 | 910B（dav-2201）落点 | 实证位置 | 状态 |
|---|---|---|---|
| rope (token,head) 分块 | `T.Kernel(64)` + `u//heads, u%heads`（simplifier 折叠成 `t=block_idx,h=uc`，零运行时除法） | `gen_rope_table.asc:16-19` | ✅ |
| rope 旋转本体 x1c∓x2sn | 纯 AIV 标量面：GM 直读 + 4 则 + `write_gm_bypass_dcache` 标量写；先物化 x1/x2 再双写（无自噬） | `gen_rope_table.asc:20-25` | ✅ |
| rope 角度来源 | **预计算 cos/sin 表 GM 加载**（`Cos[t,j]/Sin[t,j]` plain deref），零 transcendental | `gen_rope_table.asc:22-23` | ✅（取舍论证 §③-1） |
| rope V 槽透传 | slot2 不转，标量读→bypass 写 | `gen_rope_table.asc:28-30` | ✅（golden R2 按位判据） |
| rope 反向 | 同一件 `sign=-1` 编译期常量（sin 取负=逆旋转，生产口径） | 判决 `mode=table sign=-1` PASS | ✅ |
| attn 可见窗 [max(0,i-w+1), i] | `lo=T.max(i-window+1,0)` + 窗条件 `j<=i`（定义域内与 `j<n` 等价，逐值核对） | `gen_attnsw_pure.asc:27` | ✅ |
| attn 分数 QKᵀ·scale | 标量内积循环，scale=1/sqrt(dim) 折成一次乘（fp32 立即数） | `gen_attnsw_pure.asc:22-28` | ✅ |
| attn softmax exp | `T.exp`→`expf`→**compat §11 软件件**（真源 724 行起，C 波定标 1.103e-06） | `gen_attnsw_pure.asc:48,64` | ✅ |
| attn 窗外零权重 | 哨兵 `±1e30` + `expf` 早退钳位（x≤-104→0）→ 结构性精确 0 | golden R5：不可见扰动该行输出**按位不变** | ✅ |
| attn 行最大/归一 | `T.max`→原生 `max`；`acc/l` 原生除法（分母 ≥exp(0)=1，无 FLOOR 需要） | `gen_attnsw_pure.asc:33,67` | ✅ |
| attn 分数暂存（stage） | 2 个被访问 `alloc_shared (8,)f32` → `buf_dyn_shmem` 偏移 [0:8)/[8:16) 不重叠 | `gen_attnsw_stage.asc:30,38-39,44` | ✅（G-C8 再证实，G-E2） |
| Cube/VF/SIMT/跨pipe同步 | **两件全不需要也未触及**：`has_mix/has_ASC_IS_/has_vf_call/has_threadIdx/has_asc_sync` 六判决全 False | `verdict_log.txt` 各段 feature | ✅（结论：rope 与滑窗 softmax 侧是纯 AIV 标量件） |

**两类可用件（本波定型）**
1. **rope=零 UB 纯标量件**（`has_ubuf:False`，33 行生成码）：语义段全落「标量 GM 直读 +
   bypass 写 + 表加载」，无任何数学件依赖 → **上卡首选**，歧义面比 C 波 GDN scalar 还小
   （连 expf 都不吃）。
2. **attn_sw 双保险件**：`stage`（UB 分数暂存，吃 expf，贴生产结构）与 `pure`（零 UB，
   分数三趟重算，expf 同吃）。两件都 PASS；上卡首批可打 `pure`（零 UB 歧义），
   `stage` 验证 UB 元素级标量读写与动态池布局。

**对 delta rule 全栈的可复用底座新增**：softmax 行归一（max/expf/除法/哨兵）四件套首次
以**生产 attn 形态**编译落地；rope 表加载形态与 attn_sw 的 Q/K 消费端可直接拼进
track-A 的 attention 主体（其分数矩阵乘部分仍属 Cube 面，划在本波范围外）。

---

## 复现链（宿主机，可直接粘贴；容器 `cann910b-e`）

```bash
P=$REPO_ROOT/release/ascend/port910b
bash $P/attempts/E/run.sh matrix          # 全矩阵判决+golden+e2e 基线（RESULT 凭据源）
bash $P/attempts/E/run.sh table           # 单件：rope 表加载（主判决）
bash $P/attempts/E/run.sh sign            # 单件：rope 反向 sign=-1
bash $P/attempts/E/run.sh sinf            # 取证：rope 自算路（预期 FAIL）
bash $P/attempts/E/run.sh attn            # attn_sw stage+pure 双判决
bash $P/attempts/E/run.sh golden          # CPU golden（R1–R6 判据打印）
docker exec cann910b-e bash -c 'cd /tmp && python3 /tmp/_dbg_e.py'   # 指标伪影分离
```
`run.sh` 每次判决注入全新 `TILELANG_CACHE_DIR=$(mktemp -d)`（缓存纪律，LOCAL_RUN.md）；
并先 `cp -f /tilelang/.../ascend/*.h` 重置 pip 副本 + `cmp` 自证 + `patch_bisheng.py`（幂等）。

---

## ① 完成情况（逐条附凭据；凭据全部可复跑）

- [x] **交付物 1a `e_rope_910b.py`：`E-ROPE-COMPILE-PASS`**
  验证：`bash attempts/E/run.sh matrix` → `verdict_log.txt` `grep -c E-ROPE-COMPILE-PASS` → **3**
  ```
  E-ROPE-COMPILE-PASS  mode=table sign=1  … has_gm_bypass:True has_ubuf:False math:(none) lines:33
  E-ROPE-COMPILE-PASS  mode=rational … lines:35（兜底骨架，数值非真值）
  E-ROPE-COMPILE-PASS  mode=table sign=-1（反向件）… lines:34
  ```
  codegen 原文存 `gen_rope_table.asc`/`gen_rope_rational.asc`；语义核对见 §0 表 + G-E4。
- [x] **交付物 1b `e_attn_sw_910b.py`：`E-ATTNSW-COMPILE-PASS`**
  ```
  E-ATTNSW-COMPILE-PASS  variant=stage … has_ubuf:True null_ubuf_handle_named:False math expf lines:52
  E-ATTNSW-COMPILE-PASS  variant=pure  … has_ubuf:False null_ubuf_handle_named:False math expf lines:71
  ```
  发射体逐段核对（窗条件/scale/expf/除法/UB 偏移）：`gen_attnsw_{stage,pure}.asc`。
- [x] **交付物 1c sinf 取证变体（预期 FAIL，反证闭环）**
  `verdict_log.txt`：`E-ROPE-COMPILE-FAIL` +
  `tl_kernel.asc:23:13: error: use of undeclared identifier 'cosf'` + 发射片段
  `c = cosf(ang); sn = sinf(ang);` ⇒ G-E1 登记；取舍论证见 §③-1。
- [x] **交付物 2 `e_golden.py`：`E-GOLDEN-PASS`（CPU 语义 golden + 上卡判据 R1–R6）**
  验证：`bash run.sh golden --npz /tmp/e_pairs.npz` →
  ```
  ROPE  sim32-vs-f64: abs=2.090e-07 rel=1.186e-06 [naive-rel=9.053e-06 伪影留痕]
        V-slot 按位透传: True ；正反旋转对合 max|Δ|=4.768e-07 (预算 2.031e-05)
  ATTN  sim32(复刻expf)-vs-f64: abs=3.343e-07 rel=5.176e-06 [naive-rel=8.166e-05 伪影留痕]
        窗外权重恰0=True / 不可见扰动该行输出按位不变=True / 行分和 l∈[1,window]=True (min 1.0000 max 5.3721)
  GOLDEN-ROPE-PASS / GOLDEN-ATTN-PASS / E-GOLDEN-PASS / GOLDEN-EMITTED /tmp/e_pairs.npz
  ```
  真值三路（fp64 精确 / fp32 逐步 / §11 expf 逐行复刻）互校；rope 语义独立锚定生产
  `rope_ref.py`（rotate_half_ref 同式），attn 语义锚 `_eager`（带掩码全量 softmax）同式。
  上卡判据 R1–R6 固化在件头注释；npz 含 `QKV/COS/SIN/ROPE_F32/ROPE_F64/AQ/AK/AV/ATTN_F32/ATTN_F64/L_ROWS`。
- [x] **交付物 3 `compat_gap_E.md`：4 条登记（G-E1 新缺口 + G-E2/E3/E4 观察），零补丁实付**
- [x] **交付物 4 `RESULT.md`（本件）**
- [x] **基线无回归**：`run.sh matrix` 末段 `e2e_cube.py` → `E2E-CUBE-ONLY PASS` /
  `HAS_MIX: True HAS_ASC_IS: True`（本波 bisheng 注入没碰坏 Cube/mix 路）。
- [x] **容器 e 环境自证**：`ENV-COMPAT-IN-SYNC (944 lines)`（pip 副本==主仓真源，`cmp` 逐字节）；
  `C-BISHENG-PATCHED`（该容器首注入；再跑幂等 `already patched`）。
- [ ] **未做（范围外）**：上卡数值对拍（本机无 NPU）；bf16/fp16 位宽形态（已知 bf16 标量
  cast 后端缺陷，本波纯 fp32）；attn 的 QKᵀ 矩阵乘主体（Cube 面）；非 32B 对齐/动态 token 轴。

---

## ② 错误与阻塞（含已解决的；环境类一并列出）

1. **run.sh patches 目录相对路径错**（`$D/../patches` 不存在，真身 `$D/../../patches`）→
   首跑 `cd: No such file or directory`。**已修**：sed 改路径，复跑通过。
2. **golden 自检 naive-rel 假阳**：attn 首跑 `GOLDEN-ATTN-FAIL`（rel=8.17e-05 超 2e-5）。
   分离取证（`_dbg_e.py` 复跑 → `E-DBG-SEPARATION-OK`）：swexp 路径 abs=2.798e-07、
   true-exp 路径 abs=2.757e-07、sw-vs-true 净贡献 3.576e-07 —— 误差是**近零输出分量的
   相对度量伪影**，非 expf/语义误差。**已修**：自检口径 rel 分母垫 1%|ref|max 地板
   （naive 值保留打印留痕）；上卡判据 R1/R4 是 `allclose(rtol=atol=2e-5)` 同精度路径对比，
   不受该伪影影响、也**不改**。
3. **`null_ubuf_handle` 假阳（体检工具坑）**：stage 首跑报 True，疑似 G-C8 复发。
   核对生成码：正则误把样板行 `buf_dyn_shmem = (__ubuf__ uint8_t*)0;`（动态池基址惯用
   写法，`gen_attnsw_stage.asc:6`）当成命名空句柄。**已修**：`(?!buf_dyn_shmem)` 排除 +
   改字段名 `null_ubuf_handle_named`；复跑两变体 False。教训登记为 G-E3。
4. **pure 变体首稿在循环体内 `T.alloc_var`**（DSL 非法形态，tilelang 要求在 kernel 块
   头部声明）→ 未及上机即在自检时发现，提到顶层 `acc2`。**如实申报**：坏形态没有留
   运行时反证（写出来就没编译），风险以静态修正消解。
5. **容器 e 的 bisheng 未注入**（任务书称"compat v3 已同步"属实，但 `bisheng.py` grep
   无 `PORT910B_NATIVE_TYPES`）→ 按 LOCAL_RUN.md 用 `patches/patch_bisheng.py` 补齐，
   幂等自证。**非阻塞**。
6. **shell 层**：宿主 zsh 吞裸 `===COUNTS===`（`echo` 未加引号被当 glob）→ 分开跑并加引号
   （C 波同类坑再验证，记录以免三踩）。
7. **无未解决阻塞**。硬限制仍是"本机无 NPU"→ 数值面待上卡（R1–R6 已备好判据与 npz 件）。

---

## ③ 疑惑点与自行决策（没问但自己定了的事）

1. **rope「表加载 vs sinf 自算」取舍（任务书点名论证）→ 定表加载，四路论证**：
   ① *硬件面*：`--mode sinf` 实测编译 FAIL（`undeclared identifier 'cosf'`，G-E1）——910B
      标量面无 SFU 入口、无 libcall（C 波已证 `__builtin_*` 编得过但 ld.lld undefined
      symbol 是死路）；且频率项 `theta^(-2i/D)` 需要的 `powf/logf` 同样缺件——**"自算"在
      910B 上连输入都没干净，补软件三角件也逃不掉预计算表**。
   ② *数值面*：表由 host fp64→fp32 一次生成，是生产契约 `rope_ref.py` 明文的
      "角度只有一处真源是对拍能收敛的前提"；核内自算会引入第二真源（角度=位置×频率的
      乘法舍入 + 软件三角误差），对拍预算翻倍。表加载件实测误差 abs=2.09e-07/rel=1.19e-06，
      在 2e-5 判据下留 20 倍余量。
   ③ *带宽面*：表 = tokens×half×2×4B（默认形状 16KB），且被 heads×2 个 q/k 行摊薄复用
      （每行 (t,·) 只读 half 个 cos+half 个 sin，摊到 heads×dim×2 次算术）——rope 是
      算术/地址生成 bound 的标量件，表读不构成瓶颈。
   ④ *生态面*：track-A 生产件（rope_asc 昇腾正文、CPU 正文、_eager、rope_ref）全部收
      `cos/sin` 分表入参——910B 件若自算角度，接口就对不上生产分发，反而要为编译波
      造一条孤立路。
   保留 `--mode rational` 骨架（只吃四则、PASS）作为"若某环境必须去表"的保编译兜底，
   其泰勒数值仅 |x|≲π 可用、**不进任何对拍判据**（rope 角 t·freq 对大 t 越界，此路
   本就不精确，如实标注）。
2. **rope 出地（O 另表）而非生产的就地**：就地语义依赖"先物化 x1/x2 再写回"的顺序纪律
   （`_eager` 注释里自噬坑生产踩过），编译波不值这个风险面；出地+V 透传反而多验一条
   标量 GM 读→bypass 写通路。代价如实申报：**上卡路若要直接接管生产 rope_asc 的就地
   接口（autograd `_RopeFn` 依赖同对象返回），需换址重编就地版或入口层吸收出地** → §⑤-3。
3. **attn_sw 取"逐行在线 softmax 的标量简化版"而非生产分块 online-softmax**：任务书口径
   是"滑窗内 score 累加/掩码（softmax 侧件）"；逐行形态去掉了 bm/bn 分块的修正因子路
   （行最大一次即得），保留了全部 910B 风险面（expf/哨兵/除法/max/UB 暂存）；
   `T.gemm` 的 QKᵀ/PV 主体属 Cube 面，明确划在范围外（与 C 波"矩阵段不在 AIV 标量波"一致）。
4. **形状常量**（rope tokens=64/heads=8/dim=64、attn h4/s64/d32/w8、cores=64）：全部静态维，
   避开动态轴×网格上界的已知雷（生产 docstring：动态维写进网格"一次都不发射"）；
   动态 token/seq 轴留待上卡波 → §⑤-4。
5. **`window > seq`、`heads*seq % cores != 0` 在 Python 层 raise**（编译期判据前置），
   不造运行期守卫——本波没有运行期。窗口默认取 ≤seq 的整窗上界。
6. **stage 变体的哨兵窗不设"重算那一步再判可见性"补丁**（生产件 docstring 强调的最易踩坑）：
   本形态逐行窗内**必有真实分数**（自见格），行最大永不是哨兵，`expf(哨兵-真实最大)`
   依软件件下溢钳位恒为 0 → 该坑在本形态不成立。golden R5 以"逐行逐头不可见扰动输出
   按位不变"锚死了这个结论。依据不同处已在代码注释与本节双向标注。
7. **fp32-only**：bf16 标量 cast 后端缺陷（任务书已知坑）+ 生产内核全程 fp32
   （升降位在入口层）→ 本波不碰半精度。

---

## ④ 偏离记录（与计划/任务书不一致之处）

| 计划原文 | 实际做法 | 理由 |
|---|---|---|
| "rope 与 attn_sw 写成可编译 kernel" 字面各 1 件 | rope×3 件（table/sign=-1/rational）+ sinf 取证 + attn×2（stage/pure）= 6 次判决 | 隔离"最小可编面/生产同构面/兜底面"，且任务书点名要"表加载 vs sinf 自算"的**论证**——反证必须可复跑 |
| attn_sw="注意力滑窗/softmax 侧件"（生产分块 online-softmax 形态） | 逐行两趟 softmax（stage/pure） | §③-3；保留全部 910B 风险面、剥离 Cube 依赖 |
| rope 生产语义=就地改 QKV | 出地 O 另表 + V 透传 | §③-2；上卡接管路需一次裁决（§⑤-3） |
| 缺口清单"无则 NONE" | 4 条（G-E1 新登记 + 3 条观察/工具坑） | sinf/cosf 此前无人登记也不可绕开地编过；G-E3 不登记等于埋体检假阳雷 |
| 交付 4 件 | 另附 `run.sh`、`verdict_log.txt`、4 份 `gen_*.asc`、`_dbg_e.py` | 无这些便不可复跑，"PASS"就只是自述（C 波同纪律） |

---

## ⑤ 需编排者裁决

1. **G-E1 是否补软件 sinf/cosf**：本波结论是**不补**（表加载是生产契约+零成本）；仅当出现
   "表都放不下"的流式位置编码需求时才立项，且必须连带范围规约（θ 无上界，纯泰勒会爆）。
2. **上卡首批打哪件**：建议 `rope table`（零数学件零 UB，歧义最小）→ `attn pure`
   （吃 expf 但零 UB）→ `attn stage`（UB 布局）三批次；判据 R1–R6 在 `e_golden.py`，
   输入件 `/tmp/e_pairs.npz`（种子=7 可重生）。C 波的 G-C4/G-C7 证真与本波无关（零 MTE）。
3. **rope 就地 vs 出地**：若 910B 件未来直接接管 track-A `rope_asc.forward`（autograd
   依赖同对象返回），需要现在就裁决"就地版换址重编"还是"入口层 memcpy 出地结果"。
4. **动态 token/seq 轴**：生产件 tokens/seq 是 `T.dynamic`；本波全静态。上卡波要不要在
   910B 标量面恢复动态轴（涉及网格上界常量化的既有纪律）需排期。
5. **compat v3 的 `§10b/GAP 合并态`本波受益确认**：rope/attn 两件在 fresh `cp -f` 真源 +
   零 GAP 补丁下直接编过——**建议把"scalar 消费面（expf/max/除法/bypass 写/小 shared 数组
   动态池）已稳定"上报给后续 RMSNorm/decay 波**作为可依赖底座（B 的 rsqrtf shim 已在真源）。

---

## ⑥ 产物清单（全部落在 `release/ascend/port910b/attempts/E/`，白名单内）

**新增**：`e_rope_910b.py`、`e_attn_sw_910b.py`、`e_golden.py`、`compat_gap_E.md`、`RESULT.md`、
`run.sh`、`verdict_log.txt`、`gen_rope_table.asc`、`gen_rope_rational.asc`、
`gen_attnsw_stage.asc`、`gen_attnsw_pure.asc`、`_dbg_e.py`

**修改（本波内迭代）**：`e_attn_sw_910b.py`（acc2 上提 + 正则排除 buf_dyn_shmem）、
`e_golden.py`（rel 地板口径 + 扰动不变性逐行逐头化）、`run.sh`（patches 路径 + sinf 预期 FAIL 旁路）。

**容器内改动**（属"自己容器"白名单，主仓零改动）：
`/usr/local/lib/python3.10/dist-packages/tilelang/contrib/bisheng.py`（注入 -D/-I，幂等）、
`/usr/local/.../ascend/*.h`（每次 `cp -f` 重置为真源后与真源逐字节一致）、`/tmp/*`。

## ⑦ 硬边界自证

```bash
wc -l $WORKSPACE_ROOT/tilelang/src/tl_templates/ascend/port910b_compat.h   # 944（v3）
docker exec cann910b-e wc -l /tilelang/src/tl_templates/ascend/port910b_compat.h            # 944
docker exec cann910b-e cmp <真源> <pip副本>                                                  # IDENTICAL
cd system-one && git status --porcelain release/ascend/port910b/attempts/                   # 仅 ?? attempts/E/（+既有 D）
cd tilelang && git status --porcelain src/tl_templates/ascend/                               # M common.h 等为会话前既有状态
```
主仓 compat 真源 mtime `Oct 9 01:41`（编排层 v3 合并时刻，早于本会话全部动作 01:52+）；
本波未对 `/tilelang` 与 system-one 的 openspec/runs/git 执行任何写操作，未 commit。
`patches/patch_bisheng.py` 只读复用（C 波件），未改其内容。
