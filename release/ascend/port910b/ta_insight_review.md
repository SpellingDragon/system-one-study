# 认知复盘：被忽略的已知信息（2026-10-18）

> 目的：回答"是否因认知不足而陷泥潭、哪些已知信息本可点破"。只记事实与出处，不写推测成结论。

## 一、被忽略的文档承诺（本地 skill / 官方教程）

| # | 出处 | 原文要点 | 与我们的关系 |
|---|---|---|---|
| 1 | `tilelang-custom-skill/tilelang-api-best-practices/references/api-compute.md:231` | 「广播运算 `c[i,j] = a[i,j] * b[j]` — 自动广播处理（仅支持 1D→2D，索引必须是简单变量）」 | **我们的 `g_ub[j]` 完全符合**（简单变量 j）——按文档写、卡上退化 ⇒ **D15 是实现缺陷，不是用法错误** |
| 2 | `api-compute.md:200` | 示例 `c_ub[i,j] = b_ub[j] + 5  # b_ub 是 1D，c_ub 是 2D` | **与 `aux_ub[i,j] = g_ub[i] - g_ub[j]` 同构** |
| 3 | `tilelang-ascend/docs/tutorials/t_parallel.md` §3.5 | 「Mismatched Operand-Result Dimensions … dimension broadcasting on the right side … indx needs to be a simple variable or expression」+ 同一示例 | **官方教程（我们从未读过）明确承诺列广播** |
| 4 | `tilelang-op-develop/references/troubleshooting.md:52-65`、`coding-conventions.md:113` | 「broadcast 源必须是 2D `[M,1]` 或 `[1,N]`，**1D 源是错误**」+ `T.tile.broadcast` 示例 | **底层 API 的真实约束**。上层 T.Parallel 承诺"自动处理 1D 源"，底层 API 却要求 2D 源 ⇒ **D15 的机理正是自动分解没把 1D 转成 `[1,N]`** |
| 5 | `api-compute.md:233` | 「离散索引（如 `a[idx[i]]`）⇒ 编译器退回 `T.serial`」 | 一个可用的绕法（我们最终用双 serial 达到同样效果） |
| 6 | `api-compute.md` §3.1 「列切分模式」 | `for i in range(M): for j in T.Parallel(N)` — **Python `range` 外层** | **官方推荐的"串行外层 + 向量内层"是 `range`，不是 `T.serial`**；我们用 `T.serial` 外层 + Parallel 内层直接 507015（实测与文档一致） |

**核心矛盾（D15 的根因链）**：
`T.Parallel` 承诺 1D→2D 自动广播 → 底层 `T.tile.broadcast` 要求 2D 源 → 自动分解在 1D 源上退化为"只取 j=0" → 所有列广播（`g_ub[j]`）静默出错。

**为什么长期没暴露**：
- solve_a 唯一的验证用例是 g≡0（任何索引都等于 0）
- scan1 的 H 链只在 g≡0（mode1）验过
- 行广播（`b_ub[i]`，第一维）在卡上是好的 ⇒ 我们误以为"广播都对"

## 二、被忽略的社区信息

| # | 出处 | 内容 | 意义 |
|---|---|---|---|
| 1 | GitHub Issue [#1194](https://github.com/tile-ai/tilelang-ascend/issues/1194)（2026-06-16，已关闭） | 「Incorrect `T.Parallel` lowering for compact-to-aligned UB store」：2D T.Parallel 被错误重写为连续 `copy_ub_to_ub`；作者最后的稳定绕法也是 **serial** | **同族 lowering 缺陷**，官方承认并修复 |
| 2 | PR [#1219](https://github.com/tile-ai/tilelang-ascend/pull/1219)（2026-06-24 合入 `ascendc_pto`） | 「**Lower un-vectorizable T.Parallel to serial**」+「Refuse to vectorize UB->UB stride-mismatched copies」 | **官方修复思路正是"不可向量化就退 serial"**——能兜住 D15 这一族 |
| 3 | 我们的运行包 | `refs/ta_pkg/ta_pkg.tar.gz` 的 `VERSION = 0.1.4`，`DEPS.txt` 注明"**J3 窗五制备**" | **实测：ta_pkg 内的 `src/transform/ascend_lower_parallel_to_vector.cc` 与本地源码逐行相同，已含 #1219 的 "lower to serial" 修复** ⇒ **不是缺修复**；但 D15 仍发生 ⇒ **该修复的判定条件没覆盖我们的 pattern**（我们被判"可向量化"，然后在向量化时列索引退化） |
| 4 | 本地 `docs/tutorials/t_parallel.md` 存在但从未读过 | 官方对 T.Parallel 语义的完整定义 | 排查时未先读官方语义文档 |

## 二·五、环境差异：官方以 950（A5）为主，我们在 910B（A2A3）

| 资源 | **910B / A2A3** | 950 / A5 | 差异 | 我们的占用 |
|---|---|---|---|---|
| L0A | 64 KiB | 64 KiB | 1× | — |
| L0B | 64 KiB | 64 KiB | 1× | — |
| L1 | 512 KiB | 512 KiB | 1× | scan2a 122 KiB |
| **L0C** | **128 KiB** | **256 KiB** | **2×** | scan2a 实测峰值 **116 KiB（91%）** |
| **UB** | **191.75 KiB（196352）** | **256 KiB** | **1.34×** | scan1 fp32 静态 **209.5 KiB（109%，靠 liveness 复用才编过）** |

**⇒ 我们在两个资源上分别处于 91% 与 109% 的悬崖边**：任何"再加一个 buffer"的改动都可能改变行为，这让"改动无效果/行为异常"的归因变得困难（D13 的 range 无效虽与容量无关，但同类乱象的土壤在这里）。

**⇒ 后端路径也不同**：910B 走 `codegen_ascend.cc`（AscendC）+ `ascend_lower_parallel_to_vector.cc`；950 走 `codegen_ascend_pto.cc`（PTO ISA）。**两条 lowering 路径的成熟度不同**，而 `t_parallel.md` 的示例（`block_M // VEC_NUM`）虽用 910B 的概念书写，**但列广播的验收显然缺失**（否则 #1194 不会在 910B2 上才被发现）。

**⇒ 关键推论**：Issue #1194 的复现平台明确写着 **"Platform: Ascend 910B2"** ⇒ 官方已知 910B 的 T.Parallel 有缺陷；但 #1219 只覆盖了"compact→aligned 重写"与"不可向量化→serial"，**没有覆盖"列广播时第二维索引退化为 0"**——这正是我们在 910B 上踩到的那一条。

## 三、认知不足的具体表现（按代价排序）

1. **把"文档承诺"当作"已验证的能力"**：`api-compute.md` 与 `t_parallel.md` 都承诺列广播，我们从没写过一个"第二维索引"的 5 行微探针来验它。**若在开工第一小时写这个探针，D15/D8 之前的所有推理都可省。**
2. **验证用例的盲区**：solve_a 只用 g≡0、scan1 只用 g≡0。**"常数输入"能掩盖所有索引类缺陷**——因为它让不同索引取到相同值。这是本次最大的方法论教训。
3. **"源码改了 = 行为改了"**：`ta_pkg` 是预编译旧包（0.1.4/窗五），`range` 展开没改变 realTailN（D13）可能与此有关（新前端 + 旧后端）。
4. **对"部分为零"的模式过度联想**：DQ 只 1 列 → 猜是累积 GEMM（D10 否）→ 猜是 L0C 超配（D11 否）→ 才回到"copy 参数"。**应先读生成码的实参，再猜机制。**
5. **未先读官方语义文档**：`docs/tutorials/t_parallel.md` 一直在仓库里。

## 四、行动项

| 优先 | 动作 | 验收 |
|---|---|---|
| **P0** | **上游报 issue（带 5 行探针）**：`c[i,j] = b[j] + 5` 在 910B2 + ta_pkg 0.1.4 上 j 退化为 0；附 Issue #1194 的修复（#1219/`f6b0b921`）已含但未覆盖此形态的证据 | issue 草稿入 `docs/upstream/`，含探针与期望/实际 |
| **P0** | **列广播回归探针**（`_w11_bcast_probe.py`，已写）：`b=arange(64)`，退化时 C 全为 5 | 纳入每次上卡首腿，永久防回归 |
| **P0** | 容量纪律：910B 的 L0C=128KiB / UB=191.75KiB（官方示例常按 950 的 256KiB 写）——新增 buffer 前必跑 `_s7f_ub_budget.py` | 两个 kernel 的 L0C/UB 峰值 < 90% |
| P1 | **fwd 复用复核**：`gdn_chunk_scaled_dot_kkt.py` 的 `coeff_ub[i,j] = g_ub[j] - g_ub[i]` 同构受影响；其"窗五真机 rel=0"需用 g≠0 数据复验 | replay 路径的 P 与 fp64 对拍 < 1e-3 |
| P1 | **skill 文档修订**：在 `api-compute.md` §广播 与 `t_parallel.md` §3.5 加实测注记（910B + ta_pkg 0.1.4 上第二维索引退化，须双 serial 或 `range` 外层） | skill PR 或 journal 记录 |
| P2 | **验证纪律**：所有"索引类"数值验证必须至少一档 **g≠0 / 非常数输入** | 写进 checklist |

## 五、对既有结论的连带影响

- **"scan1 数学已对"（mode1，3e-5）**：只在 g≡0 成立；g≠0 的 H 链此前**从未验过**——本轮新增 `_w11_scan1_g.py` 就是补这个洞
- **"fwd 六件真机全绿"**：其 coeff 写法同构，**结论需在 g≠0 数据上复验**
- **"solve_a 干净"（ws_a absmax=1.0）**：只说明量级有界，不说明 A 正确
