# RESULT.md — 执行代理 C / p2-13-ascend-runtime 甲路 P1-1 波
## GDN（Gated DeltaNet）短卷积前向（causal depthwise conv1d, K=4, + SiLU）的 910B tilelang DSL 件

**判决**：`C-GDN-CONV-COMPILE-PASS` **4/4**（`scalar|ubstage` × `exp|rational`，每次全新
`TILELANG_CACHE_DIR`）＋ `GOLDEN-PASS`（CPU 语义四路真值互校）＋ e2e 基线 `E2E-CUBE-ONLY PASS`
无回归。compat 缺口登记 **9 条（G-C0…G-C9）**，见 `compat_gap_C.md`。
**验收上限=compile PASS**（本机无 NPU，未上卡；G-C4/G-C7 数值未证真）。

凭据文件：`attempts/C/verdict_log.txt`（178 行，一次跑完 5 段）、`dma_probe_log.txt`、
`math_probe_log.txt`、`gen_gdn_*.asc`（5 份 codegen 产物）。

---

## 0. 结构结论：GDN 短卷积的每个语义段落到了哪类件

| GDN 短卷积语义段 | 910B（dav-2201）落点 | 实证位置 | 状态 |
|---|---|---|---|
| 通道分块（C=2048 → 64 核 × CB=32） | `T.Kernel(cores)` 1-D AI 核栅格 + `int32_t sid = asc_get_sub_block_id()` | `gen_gdn_scalar_exp.asc:16-17` | ✅ 编译面 |
| 4 抽头因果窗乘加 | **纯 AIV 标量面**：`T.Vector()` 区域内 `T.serial` 三重循环 + `T.alloc_var` 寄存器累加（**无** `T.Parallel`/SimtVF/SimdVF） | `gen_gdn_scalar_exp.asc:19-31` | ✅ |
| 因果左补零（K-1=3） | scalar 变体：`T.if_then_else(idx>=0, X[ch,max(idx,0)], 0)` → codegen 展开成真 `if/else` + 地址夹到 0，不出负下标 | `gen_gdn_scalar_exp.asc:22-27` | ✅ |
| 同上（UB 路） | ubstage 变体：`row_ub[0:pad]=0` 标量填 + `T.copy(X[ch,0:L], row_ub[pad:pad+L])` 区间搬 | `gen_gdn_ubstage_exp.asc:21-24` | ✅ |
| 逐通道权重 W / bias 读 | 标量 GM **1-index 直读**（`TL_PORT910B_NATIVE_TYPES` 下 `read_gm_bypass_dcache` 退化为 plain deref） | `gen_gdn_scalar_exp.asc:18`（`bi = BIAS[...]`）、`:29`（`wt = W[...]`） | ✅ |
| 输出 Y 写（标量） | `MarkScalarDcacheBypass` pass → `tl::write_gm_bypass_dcache((__gm__ float*)Y + off, v)` | `gen_gdn_scalar_exp.asc:33` | ✅ |
| 输出 Y 写（整行） | UB→GM MTE3：`asc_copy_ub2gm_align(..., static_cast<asc_store_l2_cache_mode>(4), 2048, 2048)` | `gen_gdn_ubstage_exp.asc:33`（`asc_copy_ub2gm_align`）| ⚠️ 编译面 OK，数值待证真（G-C4/G-C7） |
| `+ bias` 与除法 | 原生标量四则（codegen 直发 `/`） | `gen_gdn_scalar_exp.asc:32-33` | ✅ |
| **SiLU 的 `exp(-z)`** | **compat GAP-C 软件 `expf`**（范围规约 + 6 阶 Taylor + 指数域拼 `2^n`）；910B 标量面**既无 SFU 也无 libcall**，`T.exp`→`expf` 整条原生不存在 | `compat_patch_C.h` 块1 + 发射点 `gen_gdn_ubstage_exp.asc:31`（`expf(...)`）；精度 `max_rel_err=1.103e-06` | ✅（G-C1） |
| SiLU 兜底（无 exp 面时保编译） | `sigmoid(u)≈0.5*(u/(1+|u|)+1)` → `T.abs`→`fabsf`→`__builtin_fabsf`（纯指令） | `--silu rational` 两件 PASS | ✅ 但**非对拍真值**（golden 实测尾部误差 ~1.5e+01） |
| GM↔UB 搬运（MTE2/MTE3） | compat §10b 改写 + GAP-C 重载 → 910B 原生 7 参 `copy_gm_to_ubuf` / `copy_ubuf_to_gm` | `patch_compat_10b.py`、`compat_patch_C.h` 块2 | ⚠️ 同上 |
| Cube / `__mix__` / VF 发射 / SIMT / 跨 pipe 同步 | **本件全不需要也未触及**：`has_mix False / has_ASC_IS_ False / has_vf_call False / has_threadIdx False / has_asc_sync False`（四个变体皆然） | `verdict_log.txt` 每块 feature 段 | ✅（结论：短卷积是纯 AIV 件） |

**两类可用件（本波定型）**
1. **scalar 件（首推）**：零 UB、零 MTE、`has_ubuf: False`，语义段全落「标量 GM 直读 +
   `write_gm_bypass_dcache` + compat 数学面」。歧义面最少，**上卡第一枪应打这一件**。
2. **ubstage 件（骨架）**：把行数据经 MTE2 进 UB、滑窗读 UB、MTE3 回写，验证 `asc_copy_*_align`
   通路。依赖 GAP-C（G-C2/G-C3）+ §10b 就地改写（G-C5）+ 单位换算（G-C4），后两条**未上卡证真**。

**对 delta rule 全栈的可复用底座**：通道分块 + `T.serial` 滑窗累加 + GM-bypass 标量写 +
compat 数学面 = 四件公共骨架；仍需另立的是 `rsqrtf`（RMSNorm 侧，B 已给 shim）、
`log/exp10f` decay（G-C1 剩余）、以及需要 Cube/`asc_sync_*` 的矩阵段（本波明确划在范围外）。

---

## 复现链（宿主机，可直接粘贴；容器 `cann910b-c`）

```bash
# 一条命令跑完 5 段取证（矩阵判决 + 三类探针 + golden）
bash $REPO_ROOT/release/ascend/port910b/attempts/C/run.sh --matrix

# 单件判决（示例）
bash .../attempts/C/run.sh --variant scalar  --silu exp      --dump /tmp/gdn_scalar_exp.asc
bash .../attempts/C/run.sh --variant ubstage --silu rational
bash .../attempts/C/run.sh --ub            # G-C8 UB 句柄形态 4 case
bash .../attempts/C/run.sh --dma           # 搬运面 19 例（含 arity 负形对照）
bash .../attempts/C/run.sh --probe         # 数学面 23 例
bash .../attempts/C/run.sh --golden --npz /tmp/gdn_pairs.npz   # CPU 真值 + 上卡对拍件
```
`run.sh` 每次注入 `TILELANG_CACHE_DIR=$(mktemp -d /tmp/tlc_cache.XXXXXX)`——**判决纪律**：
tilelang 按 kernel 源码哈希命中缓存，改 compat 头不失效 → 不换 cache 就是陈旧 PASS（本波踩过，见 §②-4）。

---

## ① 完成情况（逐条附凭据；凭据全部可复跑）

- [x] **交付物 1 `c_gdn_conv_910b.py`：4 变体 compile PASS**
  验证：`bash run.sh --matrix` → `verdict_log.txt`
  ```
  C-GDN-CONV-COMPILE-PASS  variant=scalar  silu=exp       lines:37 has_gm_bypass:True  has_ubuf:False math expf
  C-GDN-CONV-COMPILE-PASS  variant=scalar  silu=rational  lines:37 has_gm_bypass:True  has_ubuf:False math fabsf
  C-GDN-CONV-COMPILE-PASS  variant=ubstage silu=exp       lines:37 has_gm_bypass:False has_ubuf:True
                          asc_* = asc_copy_gm2ub_align asc_copy_ub2gm_align asc_get_sub_block_id asc_init
                                  asc_load_l2_cache_mode asc_store_l2_cache_mode
  C-GDN-CONV-COMPILE-PASS  variant=ubstage silu=rational  同上，math fabsf
  ```
  `grep -c C-GDN-CONV-COMPILE-PASS verdict_log.txt` → **4**。codegen 原文存 `gen_gdn_*.asc`。
- [x] **交付物 2 `c_gdn_golden.py`：CPU 语义 golden（上卡验收件）**
  验证：`bash run.sh --golden --npz /tmp/gdn_pairs.npz` →
  ```
  EXP-SWEEP |x|<=30: max_rel_err=1.103e-06 OK
  EXP-EDGE  expf(-200)=0 expf(200)=3.40282e+38 expf(-104)=0 expf(88)=3.40282e+38 expf(0)=1
  typical(C=8,L=64,K=4) fp32:abs 7.20e-07/rel 2.11e-07 | compatexp:abs 7.20e-07/rel 2.11e-07 | torch:abs 7.89e-07/rel 1.83e-07
  L<K(C=3,L=2,K=4)      fp32:abs 9.21e-08/rel 5.29e-08 | compatexp:… | torch:abs 2.72e-08
  long(C=5,L=512,K=4)   fp32:abs 1.05e-06/rel 2.45e-07 | torch:abs 8.63e-07
  GOLDEN-SELFCHK OK / GOLDEN-PASS
  GOLDEN-EMITTED /tmp/gdn_pairs.npz X(2048,512) W(2048,4) BIAS(2048,) -> Y(2048,512) + Y_FP32/Y_COMPAT_EXP/Y_COMPAT_RATIONAL
  ```
  四路真值（float64 exact / fp32 逐步 / compat 逐行复刻 / `torch.nn.functional.conv1d`）互校一致
  ⇒ **窗方向、groups=C、左补零 K-1、SiLU 语义**四处约定被独立锚定；建议上卡判据
  `allclose(Y_kernel, Y_COMPAT_EXP, rtol=2e-5, atol=2e-5)`（exp 扫描实测的 2 倍余量）。
- [x] **交付物 3 `compat_gap_C.md`：缺口 9 条（非 NONE）**
  G-C0 bit_cast 无 `__aicore__` 入口｜G-C1 标量数学面整条缺失（23 例实测 `fails=18`，
  仅 `expf fabsf sqrt __builtin_fabsf arith` 5 例 PASS）｜G-C2 `asc_store_l2_cache_mode` 未定义｜
  G-C3 ub2gm arity 与 codegen 实发不符｜G-C4 MTE 单位 bytes vs 32B 块｜G-C5 §10b 槽位+位段错 arch｜
  G-C6 非 32B 对齐/padding 未覆盖｜G-C7 stride gap vs absolute 未定｜G-C8 单 shared 落 `(__ubuf__ T*)0`｜
  G-C9 `copy_data*` 备选路不通。每条附探针例名或主仓行号。
- [x] **交付物 4 `RESULT.md`（本件）**
- [x] **compat 补丁实付件**：`compat_patch_C.h`（块1 数学 + 块2 搬运，自带守卫）+
  `patch_compat_10b.py`（§10b 就地改写，签名不动）；注入生效凭据：
  `C-COMPAT-10B-REWRITTEN …port910b_compat.h (778 -> 799 lines)`，
  副本行位 `674 #ifndef TL_PORT910B_COMPAT_GAP_C_H` / `764 …GAP_C_DMA_H` / `584 TL910B-10B-REWRITTEN`。
- [x] **搬运面取证（19 例）**：`bash run.sh --dma` → `PROBE-DMA-SUMMARY pass=13 fail=6`
  PASS 含 `gm2ub_{cfg3,7arg}`、`ub2gm_{cfg3,7arg,f32ptr}`、`compat_*_align_10arg`、
  `codegen_ub2gm_7arg_as_emitted`、`asc_store_l2_cache_mode_exists`；
  FAIL 含 `copy_data*`（`no matching function`）与 **arity 负形对照**三例
  （`parameters too few/many`）→ 反证 CCE 真校验参数形态，上面的 PASS 是强证据。
- [x] **UB 句柄形态取证（4 case）**：`bash run.sh --ub` →
  `CASE A/B/D refs=5~6 命名空句柄=(none)`、`CASE C refs=0 命名空句柄=['sa']`、
  `PROBE-UBHANDLE-VERDICT: NULL-HANDLE-SEEN cases ['C']` → G-C8 触发条件钉死为
  「**被访问的 alloc_shared 恰好 1 个**」。
- [x] **基线无回归**：`run.sh --matrix` 末段 `python3 /tmp/e2e_cube.py` →
  `E2E-CUBE-ONLY PASS` / `HAS_MIX: True  HAS_ASC_IS: True`（我的 compat/bisheng 注入没碰坏 Cube/mix 路）。
- [ ] **未做（本波范围外）**：上卡 fp32 对拍（本机无 NPU）；delta rule 其余件；非 32B 对齐通路（G-C6）。

---

## ② 错误与阻塞（含已解决的；环境类一并列出）

1. `unknown type name 'asc_store_l2_cache_mode'` + `asc_copy_ub2gm_align` arity 不符 @ 生成的
   `tl_kernel.asc` store 行 → compat §10b 从未定义 store 枚举、且把 ub2gm 写成 10 参而 codegen 发 7 参。
   **已修**：GAP-C 块2（`enum : unsigned char` + 7 参重载）。判别关键：**必须有 fixed underlying type**，
   否则 `static_cast<…>(4)` 不合法。
2. `T.copy(X[ch, 0], row_ub[pad])` **发射退化成单个标量 store**，`asc_copy_gm2ub_align` 根本不出现
   （看着"PASS"其实空搬）→ **已修**：改区间式 `T.copy(X[ch, 0:L], row_ub[pad:pad+L])` /
   `T.copy(out_ub[0:L], Y[ch, 0:L])`，两条 MTE 才真发射（`gen_gdn_ubstage_exp.asc:24/33`（行号即 MTE2/MTE3 两条发射））。
3. `ASCEND_910B_ASSERT_ALIGN32 undeclared`，探针 **19/19 全挂**在 compat.h:591 → 注入块里的
   `#define` 写在函数**之后**，而这段代码在 **header 里**（先用后定义）→ 毒化所有 TU。
   **已修**：`patch_compat_10b.py` 改为 `GUARD + NEW_FN` 前置。
4. **陈旧 PASS（方法论级，最危险）**：修完宏顺序前 ubstage 曾报 PASS —— tilelang 按 **kernel 源码哈希**
   命中编译缓存（`TILELANG_CACHE_DIR=~/.tilelang/cache`），**改 compat 头不失效**。
   **已修**：`run.sh` 每次判决 `mktemp -d` 新 cache；重跑后才是真 PASS。此纪律已写进 `run.sh` 头注释。
5. **compat 软件 `expf` 真 bug（指数域双重加 127）**：`exp(0)→1.7014e+38`、`exp(1)→nan`、
   `exp(20)→-7.13e-31`（取证件 `_dbg.py`）。由 `c_gdn_golden.py` 的**逐行 numpy 复刻**在无卡条件下抓到。
   **已修**：`mant ± n<<23` + `n∈[-126,127]` 钳位；修后 `EXP-SWEEP max_rel_err=1.103e-06`、`expf(0)=1`。
6. golden 自身 4 处 bug（逐一修掉，各附实测数）：① rational sigmoid 误差 `1.57e+01` → 定性为
   "近似件仅保编译"，不设数值门槛；② torch 交叉检查忘套 SiLU（`abs 9.99e-01`）→ 补
   `z*torch.sigmoid(z)`；③ t=0 边界用 fp32 乘积比 fp64 参考（`rtol=1e-12` 必假阳）→ 两侧同转 float64；
   ④ `zerow` 用例广播维度错（`[None,:]`→`[:,None]`）。
7. 环境类：`bundle_ondemand.sh [3]` 的 bisheng 注入片段把 `bisheng.py` 打成 **SyntaxError**，且原定位
   方式 `__import__("tilelang")` 在坏态下自我死锁 → **已修**：`patch_bisheng.py`（锚
   `'"-O2", "-fPIC", "-std=c++20"'`、幂等、`py_compile` 自检）。**同缺陷也在 `attempts/B/env_setup.sh`
   内嵌片段里**（容器 b 有同样风险），见 §⑤-3。
8. shell 层踩坑（记录以免后人重踩）：宿主 zsh 吞 `echo ===`；未加引号的 heredoc 里反引号被执行
   （曾把 `compat_patch_C.h` 一行注释掏空）；`TILELANG_CACHE_DIR=$(mktemp …)` 三层引号转义炸出
   `syntax error near unexpected token '('` → 改为先 `printf` 生成 wrap 脚本再 `docker exec -i cat >`。
9. **无未解决阻塞**。唯一硬限制是"本机无 NPU"→ 数值面（G-C4/G-C7）只能待上卡。

---

## ③ 疑惑点与自行决策（没问但自己定了的事）

1. **GDN 短卷积语义取自约定而非参照代码**：任务书指路的
   `Naive-N0.5-Flash/modeling_naive_n05_flash.py` 实测**不含 conv/GDN 代码**。→ 按 Qwen3-Next/GDN
   通行约定建模：`conv1d(channels_first, depthwise, K=4, causal)` + bias + SiLU，
   `Y[c,t]=silu(bias[c]+Σ_k W[c,k]X[c,t-3+k])`，左补零宽 3。用 `torch.nn.functional.conv1d(groups=C)`
   作独立语义锚点交叉验证（`torch:abs ≤1.03e-06`）⇒ 窗方向/groups 约定不是我拍脑袋，有第三方对拍。
2. **kernel 形态偏离任务书建议**：建议 `T.Kernel(1, channels/块, threads=1)`；ascend 方言的
   `T.Kernel` 是 **1-D AI 核栅格且无 `threads=` 形参**（`threads=` 属通用 `tilelang.language`）。
   → 定 `T.Kernel(cores=64)` + `T.Vector()` 区域 + 核内 `T.serial(CB)` 通道块循环，语义等价且贴 910B 面。
3. **两变体 + 两 SiLU 都做**（超出"写一个 kernel"的字面要求）：为把「最小可编面」与「MTE 通路骨架」
   分开，并把 `expf` 缺失的影响面隔离（rational 只吃四则+fabs）。
4. **§10b 只改容器 pip 副本，主仓真源不动**（守硬边界）。代价如实申报：**真源仍是已知不可编状态**
   （10 参签名 + 错 arch 位段 + 单位错），任何 fresh 装的树跑 ubstage 必挂。→ §⑤-1。
5. **MTE 单位按 `/32`**：证据链完整（`matmul_client.h:2745` vs `copy.cc:240-245`），且与代理 B 的
   `TL910B_BYTES_TO_BLK(x)=x/32` **独立同结论**；但无卡不证真 → 记 G-C4，上卡首查项。
6. **`l2_cache_ctl` 丢弃**（910B 原生 7 参形态不吃此参数，与 §10b 原注释一致）；`sid` 传 0
   （codegen `EmitUbufToGmCopy_` 主动不发 sid）。属性能提示而非正确性。
7. **scalar 变体删掉全部死缓冲/只写不读的 alloc_shared**：为把 `has_ubuf` 做成可判别信号，并避开
   G-C8 的 `(__ubuf__ T*)0` 形态。这是**计划内的必要改动**（非顺手重构），删后 4 变体仍 4/4 PASS。
8. **rational SiLU 不进对拍判据**：它只保证"能编"，数值明显偏（实测 `abs ~1.54e+01`）。已在 docstring
   和 golden 里双向标注，避免将来有人拿它对 fp32 真值。

---

## ④ 偏离记录（与计划/spec 不一致之处）

| 计划原文 | 实际做法 | 理由 |
|---|---|---|
| `T.Kernel(1, channels/块, threads=1)` | `T.Kernel(cores=64)` + `T.Vector()` + `T.serial(CB)` | ascend 方言 `T.Kernel` 无 `threads=`，2-D 核栅格非 910B 形态 |
| 写"一个"前向 kernel | `scalar` / `ubstage` × `exp` / `rational` 共 4 件 | 隔离"最小可编面"与"MTE 通路"、以及 `expf` 缺失影响面 |
| compat 缺口"无则 NONE" | 9 条（G-C0…G-C9），含 2 条非 compat（G-C8 UB 句柄、G-C9 备选路） | 实测就是缺；G-C8/G-C9 会静默吃掉语义，不登记等于埋雷 |
| 只交付 4 件 | 另附 `probe_math/probe_dma/probe_ubhandle.py`、`env_setup.sh`、`patch_bisheng.py`、`patch_compat_10b.py`、`run.sh`、`verdict_log.txt`、`*_probe_log.txt`、`gen_gdn_*.asc`、`_dbg.py` | 无这些便不可复跑，"compile PASS"就只是自述；`_dbg.py` 是 §②-5 的取证件（保留不删，避免误删风险） |
| 建模建议"先做前向" | 同 | **未偏离**，如实记录 |

---

## ⑤ 需编排者裁决

1. **§10b 修复是否回写主仓真源**（`/tilelang/src/tl_templates/ascend/port910b_compat.h`，本波禁改）。
   不回写 → ubstage 路在每个新容器都要重放 `patch_compat_10b.py`；回写 → 要连带定 G-C4 单位与 G-C7 stride 语义。
2. **G-C4 / G-C7 上卡证真**（一次 `allclose(Y, Y_COMPAT_EXP)` 即可同时定二者）。裁决建议：**上卡首跑
   scalar 变体**（零 MTE 零歧义），ubstage 作第二批。
3. **A/B/C 三份 compat 补丁合并冲突**：B 与 C 都定义 `asc_store_l2_cache_mode`（**同名枚举重定义 →
   同树必炸**），落法也不同（B 整段替换 10b 两适配器 / C 保签名改体内 + ub2gm 加重载）。
   见 `compat_gap_C.md` §3 对照表，需指定唯一版本。单位换算 `/32` 两家独立一致，可直接采信。
4. **bisheng 注入片段统一**：`bundle_ondemand.sh [3]` 与 `attempts/B/env_setup.sh` 内嵌同款片段有
   打坏 `bisheng.py` + 自我死锁的缺陷；建议 A/B/C 统一切 `attempts/C/patch_bisheng.py`。
5. **910B 标量数学面要不要立专章**（G-C1 剩余 `sqrtf/rsqrtf/logf/log2f/tanhf/powf/floorf`）：
   GDN delta rule、softmax、RMSNorm 全卡在这。本波的"软件件 + golden 逐行复刻"打法可复制。
6. **G-C8（`(__ubuf__ T*)0`）是否报上游**为 codegen/UB 布局缺陷。

---

## ⑥ 产物清单（全部落在 `release/ascend/port910b/attempts/C/`，白名单内）

**新增**：`c_gdn_conv_910b.py`、`c_gdn_golden.py`、`compat_gap_C.md`、`RESULT.md`、
`compat_patch_C.h`、`patch_compat_10b.py`、`patch_bisheng.py`、`env_setup.sh`、`run.sh`、
`probe_math.py`、`probe_dma.py`、`probe_ubhandle.py`、`_dbg.py`、
`gen_gdn_scalar_exp.asc`、`gen_gdn_scalar_rational.asc`、`gen_gdn_ubstage_exp.asc`、
`gen_gdn_ubstage_rational.asc`、`gen_gdn_scalar_pre_cleanup.asc`（G-C8 反例存档）、
`verdict_log.txt`、`dma_probe_log.txt`、`math_probe_log.txt`

**修改（本波内迭代）**：`c_gdn_conv_910b.py`（T.copy 区间化 + 死缓冲清除 + `has_ubuf` 体检）、
`compat_patch_C.h`（GAP-C 块2 + `expf` 指数域修正）、`run.sh`（`--ub/--matrix` + 每次新 cache）、
`env_setup.sh`（`[2b]/[2c]` marker 双检）、`patch_compat_10b.py`（GUARD 前置）、`probe_dma.py`（+负形对照）。

**容器内改动**（属"自己容器"白名单，主仓零改动）：
`/usr/local/lib/python3.10/dist-packages/tilelang/src/tl_templates/ascend/port910b_compat.h`
（623 → 799 行）、`.../contrib/bisheng.py`（注入 -D/-I）、`/tmp/*`。

## ⑦ 硬边界自证

```bash
wc -l  $WORKSPACE_ROOT/tilelang/src/tl_templates/ascend/port910b_compat.h   # 623
ls -la $WORKSPACE_ROOT/tilelang/src/tl_templates/ascend/port910b_compat.h   # 26895 Oct 8 11:38
docker exec cann910b-c wc -l /tilelang/src/tl_templates/ascend/port910b_compat.h            # 623
```
主仓 mtime `Oct 8 11:38`（早于本会话所有动作），容器挂载内同一文件同为 623 行/26895 字节
⇒ **真源未被本波写入**。`git status --porcelain src/tl_templates/ascend/` 的
`M common.h/dcache_bypass.h/debug.h/numeric_limits.h` 与 `?? port910b_compat.h` 是**会话前既有状态**
（本波未跑过任何写主仓的命令，也未执行任何 git 写操作；system-one 的 openspec/runs/git 未触碰，无 commit）。
