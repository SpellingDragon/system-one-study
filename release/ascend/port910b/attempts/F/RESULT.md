# RESULT.md — p2-13 甲路 P1-3 波 · 执行代理 F

**交付物**：GDN（Gated DeltaNet）**delta rule 前向递推 + 反向伴随递推**的 910B tilelang DSL 件（AIV 标量面）。
**容器**：`cann910b-f`（CANN 8.5.0 / tilelang 0.1.15 / trunk compat 快照 965 行，**未 cp trunk**）。
**波次边界**：本波只做 **编译面 + CPU 分步对拍**；上卡（aicore 运行时）证真由编排者攒 bundle 单窗统一收（R20）。
**凭据源**：`attempts/F/verdict_log_F.txt`（339 行，含 14 个 `COMPILE-PASS` + 全部 golden/定标判决行原文）。
复跑：`bash attempts/F/run_F.sh --matrix` 与 `bash attempts/F/run_F.sh --golden --npz /tmp/f_gdn_pairs.npz`。

---

## 1. 件清单

| # | 文件 | kernel 名 | 形态 | 入参数 | 状态 |
|---|---|---|---|---|---|
| F1 | `f_gdn_delta_fwd_910b.py` | `gdn_delta_fwd_kernel` | **ub**：状态列常驻 UB（3 个都被访问的 shared），零 `T.copy` | 9（Q,K,V,BETA,G,H0,HIST,SOUT,O），`out_idx=-1` | compile PASS |
| F2 | 同上 | 同上 | **pure**：零 `alloc_shared`，状态就地 GM（`SOUT` 即工作缓冲，宿主预置 =H0） | 同上 | compile PASS |
| F3 | `f_gdn_delta_bwd_910b.py` | `gdn_delta_bwd_kernel` | **ub**：伴随态 `(DV,DK)`+暂存全在 UB（7 个都被访问的 shared），`dstate=zero` | 14（Q,K,V,DO,BETA,G,HIST,DSOUT → DQ,DKG,DVV,DBETA,DG,DH0） | compile PASS |
| F4 | 同上 | 同上 | **pure**：零 UB，Λ 就地存 `DSOUT`，dq/dk 走 GM 读改写累加（`j==0` 首存 ⇒ **免宿主预清零**） | 同上 | compile PASS |
| F5 | 同上 | 同上 | **ub + `dstate=gm`**：末状态梯度非零入口（截断 BPTT 之外的全链反传口径） | 同上 | compile PASS |
| F6/F7 | 两件 | 同上 | **生产形状外推** `H16 T128 DK64 DV64 cores16`（fwd ub + bwd ub） | — | compile PASS ×2 |
| G | `f_gdn_golden.py` | —（CPU numpy，无卡可跑） | 三段独立判决 [A] 前向 [B] 反向公式 [C] fp32+软件 expf 漂移；并产出上卡对拍 npz | — | `F-GDN-GOLDEN rel=5.757e-07` |
| P | `f_probe_logf.py` | `probe`（最小 `T.log`+`T.exp` 件） | GAP-F 取证件：编译差分（正/负形）+ 数值定标 + GDN 往返 | — | `DIFF-NEGCONFIRMED` + `F-LOGF-PROBE-COMPILE-PASS` |
| H | `compat_patch_F.h` + `patch_compat_F.py` | — | 缺口补丁（软件 `logf`，87 行，锚 `TL_PORT910B_COMPAT_GAP_F_H`）+ 幂等装配器 | — | 已验证可注入（965→1052 行） |

支撑件：`run_F.sh`（一键判决，含 md5 PUSH-VERIFY）、7 份 codegen 产物 `gen_gdn_*.asc`。

**语义与布局约定**（细节在两件 docstring，这里只记结论）
- state 一律**转置存储** `Sd[h,j,i] ≡ S_std[i,j]`（j=值维 DV，i=键维 DK）⇒ 第 j 列是 DK 个连续元素，标量面顺址。
- 前向工作单元 = `(head, 值维列 j)`，`H*DV` 个单元均分给 cores，**零跨核归约**；
  反向工作单元 = `head`（`dq/dk/dβ/dg` 都是对 j 的归约，不可按列切）⇒ 强制 `H % cores == 0`。登记 G-F2。
- 融合次序：`S=S*a; p+=S*k` 同趟（p 吃**已衰减**的 S）、`S+=k*u; o+=S*q` 同趟（o 吃**已更新**的 S）。
  golden 复刻体**必须同序**，否则 rel 数字无解释力。
- decay 只吃 `exp`（g 以 log 域入参）⇒ **本波件零新增 compat 依赖**（详见 gap_F.md G-F0）。

## 2. 逐件 compile 判决（原文摘要，全部 `TILELANG_CACHE_DIR=$(mktemp -d)` 防陈旧 PASS）

验证形状 `H8 T32 DK16 DV16 cores8`；特征面统一为 `__global__ __vector__ void …`、
`has_mix/has_ASC_IS_/has_vf_call/has_threadIdx/has_asc_sync : False`（纯 AIV 标量面，不吃 SIMT 头链）、
`has_mte_copy : False`（零 `T.copy`）、`named_null_ubuf : []`（**G-C8 死缓冲空句柄未触发**）。

| 件 | 判决行 | 关键特征 | asc |
|---|---|---|---|
| F1 fwd ub | `F-GDN-DELTA-FWD-COMPILE-PASS` | `has_ubuf:True gm_rmw:False lines:59` `tl::write_gm_bypass_dcache` `expf` | `gen_gdn_fwd_ub.asc` |
| F2 fwd pure | `F-GDN-DELTA-FWD-COMPILE-PASS` | `has_ubuf:False gm_rmw:True lines:50` `read+write_gm_bypass_dcache` `expf` | `gen_gdn_fwd_pure.asc` |
| F3 bwd ub | `F-GDN-DELTA-BWD-COMPILE-PASS` | `dstate=zero has_ubuf:True gm_rmw:False lines:101` `expf` | `gen_gdn_bwd_ub.asc` |
| F4 bwd pure | `F-GDN-DELTA-BWD-COMPILE-PASS` | `dstate=zero has_ubuf:False gm_rmw:True lines:104` `expf` | `gen_gdn_bwd_pure.asc` |
| F5 bwd ub dstate=gm | `F-GDN-DELTA-BWD-COMPILE-PASS` | `dstate=gm lines:101` | `gen_gdn_bwd_ub_gmstate.asc` |
| F6 fwd 生产形状 | `F-GDN-DELTA-FWD-COMPILE-PASS` | `H=16 T=128 DK=64 DV=64 cores=16 lines:59` | `gen_gdn_fwd_prod.asc` |
| F7 bwd 生产形状 | `F-GDN-DELTA-BWD-COMPILE-PASS` | `H=16 T=128 DK=64 DV=64 cores=16 lines:101` | `gen_gdn_bwd_prod.asc` |
| F1'/F3' 注入 GAP-F 后回归 | `COMPILE-PASS` ×2 | 补丁不影响本波件（仍只发 `expf`） | — |

合计 **11 个 `COMPILE-PASS`**（8 格矩阵 + 回归 2 + logf 探针 1），`MATRIX_EXIT=0`。
UB 布局复核（bwd ub 7 区，`buf_dyn_shmem` 偏移）：`lam(0,256) u(+256) dp(+272) k(+288) q(+304) v(+320) do(+336)`。
Λ^tot 修复**真落码**凭据（`gen_gdn_bwd_ub.asc`）：
```
lam = (((__ubuf__ float*)buf_dyn_shmem)[((j_3*16)+i_3)] + (((__ubuf__ float*)buf_dyn_shmem)[(j_3+336)] * ((__ubuf__ float*)buf_dyn_shmem)[(i_3+304)]));
```

## 3. 逐件 CPU 分步对拍判决（`f_gdn_golden.py`，fp32/fp64 双轨 + 三独立锚）

```
[A] FWD  fp32(exp)  vs fp64 : rel_O=1.123e-07 rel_HIST=8.217e-08 rel_Sfin=8.124e-08
[A] FWD  fp32(expf) vs fp64 : rel_O=1.286e-07 rel_HIST=9.369e-08 rel_Sfin=1.339e-07
[A] FWD  fp32(expf) vs fp32(exp) : rel_O=9.355e-08 rel_Sfin=1.063e-07   ← 纯 §11 软件 expf 贡献
[B] ADJ-FD    Q 1.380e-10  K 2.770e-10  V 3.731e-10  BETA 3.194e-10  G 7.281e-10  H0 4.415e-10
[B] ADJ-TORCH Q 8.776e-17  K 2.659e-16  V 1.542e-16  BETA 2.584e-16  G 5.383e-16  H0 5.921e-17
[B] BWD-ADJOINT OK worst_rel_vs_FD=7.281e-10 (tol 1e-08)
[C] fp32+expf vs fp64 : Q 1.365e-07  K 1.654e-07  V 1.410e-07  BETA 1.807e-07  G 5.757e-07  H0 1.187e-07
[C] fp32 发射次序 spread(ub vs pure) = 4.528e-07
[C] EXPF-DRIFT a_t 单点 max_rel=1.427e-07；沿 T=32 连乘后进 O 的 rel=1.286e-07
F-GDN-GOLDEN rel=5.757e-07 (fwd 1.286e-07 / bwd_fp32 worst 5.757e-07 / adjoint-vs-FD 7.281e-10) TOL_F32=2e-04
GOLDEN-EMITTED /tmp/f_gdn_pairs.npz (fp32 同序件 + fp64 真值，供上卡对拍)
```

读法：
- **[B] 才是反向公式的证真**——fp64 伴随递推 vs **中心差分有限梯度**（独立锚 1）vs **torch autograd**（独立锚 2），
  两套全梯度（含初状态 `dH0`）worst 7.28e-10 / ~1e-16 量级 ⇒ **公式无漏项**。
- **[C] 是上卡期望值**：kernel 真实精度档（fp32 + §11 软件 `expf` + 与 DSL 同序的累加）vs fp64 真值，
  各项 `1.2e-07 ~ 5.8e-07`，远低于 `TOL_F32=2e-04`，且与 C 波 conv 真机 6.35e-08、P1-1c 三件 1e-7 级同量族 ⇒ **上卡 rel 落在 1e-7 级即符合预期**；
  显著大于 1e-5 就要按 [A]/[B] 逐段定位（[B] 已排除公式错，故先查存储序 G-F3 / MTE 面）。
- **[A] 的 `expf vs exp` 差值（9.4e-08）单列出软件件的净贡献**，防"把 §11 件的误差算进递推的账"。

## 4. 缺口补丁说明

- 本波件（F1–F7）**不需要任何 compat 新增件**：只吃 trunk 已并的 §11 `tl910b_expf`。
- 唯一交付的补丁是 **GAP-F 软件 `logf`**（`compat_patch_F.h`，87 行），服务 **decay 口径 (b)**
  （线性域给 a、件内 `g = log(a)`，chunked 形态要 log 域累加和时用）。纯标量、无 double、无 libcall，
  与 §11 `expf` 同风格：位视图取 `m∈[1,2)` → `m>√2` 折半 → `z=(m−1)/(m+1)` 的
  `ln(m)=2z(1+z²/3+z⁴/5+z⁶/7+z⁸/9)` → 拼 `e·ln2`；`x<=0` 钳位 `-FLT_MAX`。
  退让开关 `-DTL_PORT910B_SKIP_logf`。合并纪律见 gap_F.md **G-F1**（位视图 helper 与 GAP-C 同名会重定义，
  本块已 `#ifdef TL_PORT910B_COMPAT_GAP_C_H` 复用否则自带 `tl910b_gapF_*`）。
- 已在容器 pip 副本验证可注入（965→1052 行，注入后 FWD/BWD 回归仍 PASS），
  **未写 trunk 一行**（`port910b_compat.h`/`gemm.h`/`common.h`/`run_numerics.py` 均未触碰）。

## 5. partial 范围（结论：**本波未动用 partial 例外条**）

任务书与 tasks P1-3 允许"反向 partial 入账"。实测结果好于此：

| 梯度项 | 实现 | fp64 公式证真（[B]） | fp32 上卡期望（[C]） |
|---|---|---|---|
| `dQ` | ✅ | 1.380e-10 | 1.365e-07 |
| `dK` | ✅ | 2.770e-10 | 1.654e-07 |
| `dV` | ✅ | 3.731e-10 | 1.410e-07 |
| `dBETA` | ✅ | 3.194e-10 | 1.807e-07 |
| `dG`（decay 门） | ✅ | 7.281e-10 | 5.757e-07 |
| `dH0`（初状态梯度） | ✅ | 4.415e-10 | 1.187e-07 |

⇒ **q/k/v/beta/g + 初状态 六项全实现、双独立锚证真**，无延后项。

**真正的 partial 在"验证面"而非"公式面"**（诚实口径）：
1. 只证了 **compile + CPU 同序数值**，未证 **卡上执行**（本波边界，R20）。
2. 反向 **无 T.copy/无 MTE** ⇒ C 波 G-C4/G-C7 的搬运单位/stride 面本波**没走到**，
   不等于它们被证真；一旦为带宽改成 `T.copy` 通路，风险重新引入。
3. **未做 chunk 形态**（tilelang 主仓 `examples/gdn` 是分块口径）；本波是 recurrent 逐 token，
   `T=128` 时单核内层循环 `T*DV*DK` = 52 万次标量 FMA，**性能未评估**（P1 只要求可编 + 数值）。
4. **未做 head 数 < 可用核数**时的二段并行（G-F2 的可扩展性尾巴）。

## 6. 后续上卡证真点（供编排者攒 bundle 单窗）

| 优先级 | 证真点 | 建议验法（禁卡上试错，形数已压到最小） |
|---|---|---|
| **P0** | F1 `fwd ub`（验证形状）数值 | npz 已在容器 `/tmp/f_gdn_pairs.npz`（fp32 同序 + fp64 真值）。期望 rel ~1e-7；>1e-5 即非数值问题 |
| **P0** | F3 `bwd ub`（`dstate=zero`）数值 | 同上 npz；六项梯度逐个看，`dG` 最大（5.8e-07）作为阈值参考 |
| P1 | F5 `bwd ub --dstate gm` | 末状态梯度非零入口是否被正确读入（`DSOUT` 由宿主提供） |
| P2 | F2/F4 `pure` 变体 | **G-F3**：GM 就地 RMW 读后写可见性。验法 `T=1 vs T=2` 两形对比，T=2 若 rel 突升即坐实不可见 ⇒ pure 档作废，留 ub 档 |
| P2 | UB 预算 | 生产形状 `DK=DV=64` 时 bwd 单核 Λ = 16KB；多核并行的 UB 占用与 `buf_dyn_shmem` 分块需上卡实测（编译面已 PASS） |
| P3 | HIST 带宽 | `(H,T,DV,DK)` fp32 @ 生产形状 ≈ 33.5MB/卡（G-F4），决定 P2 是否必须走 chunk 形态 |

## 7. 给编排者的 tasks.md P1-3 入账行（**我未自行改 tasks.md**，白名单只覆盖 `attempts/F/`）

建议粘贴为 P1-3 的子条目（`[ ]` 保持，真机 rel 未收）：

```
  - **P1-3a delta rule 编译面+CPU 对拍已收（2026-10-10 波 F）**：GDN recurrent delta rule 前向（ub/pure）
    + 反向伴随（ub/pure/dstate-gm，dq/dk/dv/dβ/dg/dH0 **六项全实现**）910B 编译 ×11 PASS（含 H16/DK64/DV64/T128
    生产形状外推）；CPU 分步对拍 `F-GDN-GOLDEN rel=5.757e-07`，反向公式经 **有限差分 + torch autograd 双独立锚**
    fp64 证真 worst 7.28e-10 ⇒ **partial 例外条未动用**。decay 走 log 域入参 ⇒ **只吃 §11 已并的 expf，零新增 compat**
    （"log 缺件"仅在线性域口径 (b) 成立，已备 GAP-F 软件 logf 补丁 `attempts/F/compat_patch_F.h`，
    编译差分 DIFF-NEGCONFIRMED + 定标 worst_rel=1.844e-07）。余：真机 rel（首推 ub 档，期望 1e-7 级）、
    G-F3 pure 档 GM 就地 RMW 读后写可见性、GAP-F 合并、chunk 形态与 HIST 带宽（P2）。结构性结论 G-F2：
    前向可按值维切核、**反向必须 head 并行**（dq/dk/dβ/dg 是对值维的归约）
```

另需拉一条**环境类**跨波项（gap_F.md **G-F6**）：各波 run 脚本若用 `cat >` 推送，容器 exec 侧 root 对
`docker cp` 落下的 uid=501/644 文件**无覆写权**（DAC_OVERRIDE 被剥），`Permission denied` 不触发 `set -e`
⇒ **会静默编译旧版本**。本波已踩并改 `docker cp` + md5 `PUSH-VERIFY`；建议 A/B/C/E 各波同补。

## 8. 产物清单（行数取 wc -l 口径；全部新增于 `attempts/F/`，未触碰白名单外任何文件）

kernel/对拍/探针：`f_gdn_delta_fwd_910b.py`(236) `f_gdn_delta_bwd_910b.py`(309) `f_gdn_golden.py`(370) `f_probe_logf.py`(170)
缺口补丁：`compat_patch_F.h`(87) `patch_compat_F.py`(75) `gap_F.md`(122)
判决/存证：`run_F.sh`(82) `verdict_log_F.txt`(339) `gen_gdn_fwd_ub.asc`(59) `gen_gdn_fwd_pure.asc`(50)
`gen_gdn_bwd_ub.asc`(101) `gen_gdn_bwd_pure.asc`(104) `gen_gdn_bwd_ub_gmstate.asc`(101)
`gen_gdn_fwd_prod.asc`(59) `gen_gdn_bwd_prod.asc`(101) `RESULT.md`（本文件）

## 9. 白名单自检（写后三连 + 越界排查）

- `wc -l` / `python3 -m py_compile`（5 件全 `PY_COMPILE_OK`）/ `bash -n run_F.sh`（`BASH_SYNTAX_OK`）
  / 锚 `grep -c TL_PORT910B_COMPAT_GAP_F_H compat_patch_F.h = 3`（ifndef/define/endif 齐全）——见本波末尾命令。
- **trunk 未被我写**：tilelang 仓内禁改件 mtime 全部早于本波工作窗（我的件最早 14:33）——
  `port910b_compat.h 2026-10-09 02:22`、`gemm.h 10-09 01:24`、`common.h 10-08 00:40`、
  `dcache_bypass.h 10-08 01:46`、`numeric_limits.h 10-07 21:50`、`intrin_rule_ascend.cc 10-02 16:06`。
  我的 compat 改动**只落在容器 pip 副本**（`/usr/local/.../port910b_compat.h`，965→1052 行，可 `--restore` 还原），
  上报形态是 `compat_patch_F.h`。
- ⚠️ **需编排者认领的一处非我改动**：`git status` 显示 `M release/ascend/port910b/run_numerics.py`
  （mtime `2026-10-10 14:29:52`，落在并发窗内）。我**没有写过该文件**；核对内容后判定属 **B 域 target 补入**
  （docstring 改"六 target"→"八 target"、新增 §7 add_ln/§8 readout 内联、`sys.path` 排除 B 域的注释），
  且 `grep -n "gdn_delta|attempts/F|f_gdn|GAP_F"` 在该文件内 **0 命中** ⇒ 与 F 波无关，
  应是 P1-1e 那条 rig 修复的并行改动，请编排者按 B/编排者侧入账。
- 未做任何 git 写操作（仅 `status/diff/check-ignore` 只读）。

## 10. 定稿复核（文档落定后重跑，凭据见 verdict_log_F.txt **[C] 段**）

先 `bash attempts/F/run_F.sh --restore` 把容器 compat 还原成 **965 行未注入快照**
（`F-PATCH-STATUS after-restore: lines=965 GAP_F_present=False bak_exists=True`），
再重验三格 + 复现 golden：

| 重验项 | 判决 | 特征 |
|---|---|---|
| `--cell fwd --variant ub` | `F-GDN-DELTA-FWD-COMPILE-PASS` | `gm_rmw: False lines: 59 math/intrin: expf` |
| `--cell bwd --variant ub` | `F-GDN-DELTA-BWD-COMPILE-PASS` | `gm_rmw: False lines: 101 expf` |
| `--cell bwd --variant pure` | `F-GDN-DELTA-BWD-COMPILE-PASS` | `gm_rmw: True lines: 104 expf` |
| `--golden --npz …` | `F-GDN-GOLDEN rel=5.757e-07` | [A]/[B]/[C] 三段数字与 §3 **逐位相同** |

结论两条：
1. **本波交付件对 GAP-F（`logf`）零依赖** —— 在未注入快照下三格仍全 PASS，`math/intrin` 只出现 `expf`。
   `compat_patch_F.h` 纯粹服务 decay 口径 (b)，合并与否不阻塞 F1–F7 上卡。
2. golden 数字**可复现**（非缓存/非偶然）；容器 compat 已留在**干净快照态**（965 行），
   我离场时不给后续波留副作用。全日志累计 14 个 `COMPILE-PASS` + `PUSH-VERIFY SAME ×36`。
