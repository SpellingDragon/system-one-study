# O5_GDN_BWD_DESIGN_HANDOFF.md — O5：GDN 反向算子 TileLang-Ascend 设计交接

> 执行代理：**tilelang-op-design**（技能驱动）　|　变更：O5 / GDN bwd 迁移　|　日期窗：1012
> 上游输入：`gdn_asc.py`（fork golden，host cpu-alias 六项 `1e-7` 对 fp64；设备层 `ub` 档不可修）
> 目标仓：`tilelang-ascend` @ `ascendc_pto` / `87e6d1c2`
> 本件一句话：**`gdn_bwd` 设计文档两件已落地（design.md 11 节 + proto.yaml），本文是给 system-one 仓的交接面——设计要点、复用/新件清单、判决契约对接、op-develop 入口。**

---

## ⓪ 状态锚点（可复跑）

| 项 | 值 | 复跑命令 |
|---|---|---|
| 设计文档 | `tilelang-ascend/examples/gdn_bwd/design.md` | `wc -l examples/gdn_bwd/design.md` |
| proto.yaml | `tilelang-ascend/examples/gdn_bwd/proto.yaml` | `python3 -c "import yaml;print(yaml.safe_load(open('examples/gdn_bwd/proto.yaml'))['operator']['name'])"` |
| 目标仓 HEAD | `87e6d1c2`（`ascendc_pto`） | `git -C tilelang-ascend rev-parse --short HEAD` |
| fork golden md5（**未改**） | `676c07936547e9d897d3b2366a0317e0` | `md5 ascend/kernels/gdn_asc.py` |
| 判决契约 | `port910b/run_delta_card.py` `BWD_ITEMS=("dq","dk","dv","dbeta","dg","dh0")`，`TOL_F32=2e-4`，`TOL_ADJ=1e-8` | 见 `run_delta_card.py:1-40` |
| 梯度检查契约 | `gradcheck_card.py` `RTOL,ATOL=1e-4,1e-5`，`GRADCHECK-<op> PASS/FAIL/INVALID` | 见 `gradcheck_card.py:1-40` |

> **只读约束遵守**：本波**只新建** `examples/gdn_bwd/design.md`、`examples/gdn_bwd/proto.yaml`、
> 本文件；**未改**两仓任何既有文件；未做 git 写；不占卡；未在容器做任何写操作。

---

## ① 设计要点（5 条）

1. **算子粒度＝单 op 六梯度**：`gdn_bwd(q,k,v,g,beta,h0,do) -> (dq,dk,dv,dbeta,dg,dh0)`，对齐 `gdn_asc.backward` 与 `run_delta_card` 的七行判决（六项 + `adjoint`）。**不是** fwd-adjoint 的公开组合，而是单 kernel 融合反向（对齐 NSA Bwd 先例 `nsa_bwd_single`）。
2. **数学口径＝chunk 内伴随 + chunk 间 reverse scan**：逐 token 伴随式（§1.3.2）与 fork golden 逐字一致（`gdn_asc.py:296-307` docstring / `f_gdn_golden.py:123 bwd_adjoint`）；chunk 实现参照 fla `chunk_bwd` 六件伴随分解。**`HIST=S_{t-1}` 是 `dg` 的必要历史**。
3. **编程模式＝混合**：新增件走 **Developer + combineCV**（`TL_ASCEND_AUTO_*` 五 pass 全 True，`threads=1`，`alloc_shared/alloc_fragment`），仅复用件内部保留 Expert 手动交叉同步。对齐 NSA Bwd `_hybrid_pass_configs`。
4. **同步＝第一硬约束**（fork 翻车根因候选）：新增件**禁用**手写 cross flag，全走自动同步 pass（`TL_ASCEND_AUTO_SYNC` / `TL_ASCEND_AUTO_SYNC_VS`）；**跨 chunk 状态必须经 GM workspace `DS_ws` 中转，禁止同一 UB buffer 跨 trip 就地读写**（G-F3 约束）。引 commit `8100a3a6`（#1851 `Preserve Ascend sync dependencies across access and loop lifecycles`）+ `TL_ASCEND_AUTO_SYNC_VS` 的 `preserving #1346`（`tilelang/transform/pass_config.py:204-207`）。
5. **形状契约＝`dg/dbeta` 严格 `(B,H,T)`**：修掉 fork 的 `_gate3` 恒摊 3D 缺陷（`I3_REALSHAPE_SPEC.md §4`）。

---

## ② 复用 / 新件清单

### 2.1 复用（fwd 六件，只读其数学与接口，不改代码）

| 复用件（tilelang-ascend） | 行号 | 本算子用途 |
|---|---|---|
| `examples/linear_attention_and_rnn/gdn/gdn_chunk_cumsum.py` | `:15 cumsum_ker` `:32 T.Scope("V")` | 取 `γ=cumsum(g)`（及其伴随把 `dγ` 折回 `dg`） |
| `.../gdn_chunk_scaled_dot_kkt.py` | `:17 kkt_ker` `:51 T.gemm_v0` `:55-68` | 取 `KKᵀ` 与门控 mask（其伴随产 `dk/dβ/dg` 部分） |
| `.../gdn_solve_tril.py` | `:19 solve_tril_ker` `:46-57` `:72 solve_triangular` | 取 `A=(I+L)^{-1}`（伴随需转置前代） |
| `.../gdn_wy_fast.py` | `:16 wy_fast_ker` `:83/88 T.gemm_v0` | 取 `W=A diag(e^γβ)K`、`U=A diag(β)V`（伴随产 `dW/dU`） |
| `.../gdn_chunk_h.py` | `:16 chunk_h_ker` `:68-77/100-123` | 取 chunk 间状态递推（**其伴随即 reverse scan**） |
| `.../gdn_chunk_o.py` | `:16 chunk_o_ker` `:67/68/77` | 取 chunk 内输出 `O`（其伴随产 `dq/dk/dv` 主体） |
| `examples_experiment/deepseek_nsa/example_tilelang_nsa_bwd/...py` | `:51-56` `:334` `:397` `:410-446` | **形态模板**：单 kernel 融合反向 / `threads=1` / 五 pass / 片上直连 |

### 2.2 复用（system-one 侧，只读）

| 复用件 | 路径 | 用途 |
|---|---|---|
| fork golden（算法源） | `ascend/kernels/gdn_asc.py`（`:286 gdn_bwd_asc_impl`，`:296-307` 推导，`:332-491` 实现） | **导数公式照用**；设备层重写 |
| fp64 解析 golden | `port910b/attempts/F/f_gdn_golden.py`（`:69 fwd_exact` `:123 bwd_adjoint` `:205 gradcheck_fd`） | 搬作 TA 侧 golden |
| 判决契约 | `port910b/run_delta_card.py`（`BWD_ITEMS` / `TOL_F32` / `TOL_ADJ` / `gdnb[` 假绿守卫） | 六项逐项判对接 |
| 梯度检查契约 | `kernels/gradcheck_card.py`（`RTOL,ATOL=1e-4,1e-5` / `GRADCHECK-<op>` / 非有限值 FAIL） | 八门面对接 |

### 2.3 新件（在 tilelang-ascend 新建，须给数学定义与形状流）

| 新件 | 数学定义 | 形状流 |
|---|---|---|
| `gdn_bwd` 主 kernel | chunk 内伴随（§1.3.2）+ reverse scan | in: `q/k/v(B,H,T,DK)`, `g/β(B,H,T)`, `h0(B,H,DK,DK)`, `do(B,H,T,DV)` → out: `dq/dk(B,H,T,DK)`, `dv(B,H,T,DV)`, `dbeta/dg(B,H,T)`, `dh0(B,H,DK,DK)` |
| reverse-scan 段 | `Λ_{t-1}=a(Λ_tot+k⊗dp)` 的 chunk 级聚合 | `dS_i (DK,DV)` per chunk → workspace `DS_ws (B,H,NT,DK,DV)` → `dh0` |
| 转置前代（solve_tril 伴随） | `A^{-T} = (I+L)^{-T}` | `(C,C) → (C,C)` |
| `dg` 归约 | `dg_t = a_t Σ_j S_{t-1}[j,i] dS'_t[j,i]` | `(C,DV)×(C,DV) → (C,)` |
| workspace `DS_ws` | reverse-scan per-chunk `dS_i` | `(B,H,NT,DK,DV)` fp32 |

---

## ③ 与判决契约的对接面

### 3.1 `run_delta_card.py`（六项 + adjoint，七行）

| 对接点 | 契约要求 | TA 侧落地 |
|---|---|---|
| 输出顺序 | `BWD_ITEMS=("dq","dk","dv","dbeta","dg","dh0")` + `"adjoint"` | kernel `out_idx` 按此序（design §4.2/§8.2） |
| 阈值 | `TOL_F32=2e-4`（六项）/ `TOL_ADJ=1e-8`（adjoint-vs-FD） | design §9.2 表 |
| 逐项不早退 | `ITEM <it> rel= .. verdict= ..` 逐行 | design §9.2「六项逐项判」 |
| 假绿守卫 | `bkeys=[k for k in keys if "gdnb[" in k]` 必须非空 | 编译键须含 `gdnb[`；design §9.2 |
| 输出预清零 | `_bwd_inputs` 第六抽作 DO；输出由 caller 预清零 | design §8.6 |
| 形状 | 六梯度同输入 shape；`dg/dbeta` 严格 `(B,H,T)` | design §4.2 |

### 3.2 `gradcheck_card.py`（八门面）

| 对接点 | 契约要求 | TA 侧落地 |
|---|---|---|
| 门面 | `OPS` 含 `gdn_delta`，`RTOL,ATOL=1e-4,1e-5` | design §9.3 附加门 |
| 判据行 | `GRADCHECK-gdn_delta PASS/FAIL/INVALID max_rel=` | design §9.1/§9.3 |
| 非有限值 | 一律 FAIL（`inf<=inf` 假绿防护，`:202-212`） | design §9.2 NaN/Inf 位置敏感比较 |
| 已知 blocker | `DECLARED_BLOCKERS={("gdn_delta","cpu"):{"gdn_backward[cpu]"}}` | TA 侧 `cpu` 轨仍走 torch（预期），卡轨走 kernel |

### 3.3 与 fork 的设备层缺陷的隔离

| fork 缺陷（reconcile 证据） | 迁移侧规避（design 节） |
|---|---|
| `gdnb` fresh `dbeta rel=8.002e-01` 错；`prefill=zero` 后六项全对 `1.018e-07`（K4_RESULT.md） | 迁 TA 后**重写设备层**；`dbeta` 为 L0 重点盯防项（design §10.2） |
| M4「入口 UB 预清零」修复**被窗七证伪并回退**（`gdn_asc.py` 复归 md5 `676c...`） | **不用预清零**；改 GM workspace 边界强制可见性（design §7.2） |
| `dg` 恒摊 3D（`I3_REALSHAPE_SPEC.md §4`） | 输出严格 `(B,H,T)`（design §4.2/§10.1） |
| G-F3 跨 token UB 就地读写可见性赌注 | 禁跨 trip UB 就地读写（design §7.2 硬约束） |

---

## ④ 下一步 op-develop 的入口

### 4.1 触发与输入

- **skill**：`tilelang-op-develop`
- **输入**：`tilelang-ascend/examples/gdn_bwd/design.md`（11 节）+ `proto.yaml`
- **目标产物**：`examples/gdn_bwd/gdn_bwd.py`（纯 `@tilelang.jit`）+ `examples/gdn_bwd/test_gdn_bwd.py`（golden + L0 + main）

### 4.2 建议实现顺序（design §11.4）

1. **`l0_basic` 单 chunk 路径**（无 reverse scan，`T=C=64`）：先锁 chunk 内伴随的数学正确性。
2. 六项 + `adjoint` 逐项判通过（fp32 输入，denoise）。
3. 加 reverse scan 跨 chunk 路径（`l0_typical`，T=256）。
4. 开 fp16 主干（design §10.3 两阶段去噪）。
5. 性能哨兵（design §9.4）：`perf_large (16,16,16384,128,128)` 对齐窗五/README 大 shape。

### 4.3 交接清单（给 op-develop 的必读）

| 项 | 位置 | 要点 |
|---|---|---|
| 数学真值 | design §1.3.2 / `f_gdn_golden.py:123` | 逐 token 伴随式，`Λ_tot=Λ_{t+1}+do⊗q` |
| chunk 分解 | design §1.3.3 / §3.1 | fla `chunk_bwd` 六件伴随 |
| 同步硬约束 | design §7 | 自动同步 + GM workspace 边界；**禁手写 cross flag** |
| 技术约束 | design §3.5.1 | 一维 grid / `threads=1` / `NT_MAX` 静态 / `T.gemm_v0` / `valid_*` extent / L0C `BV=64` |
| 精度表 | design §9.3 | fp16/fp32 两行 |
| 哨兵 | design §9.4 | 三 case，超时门禁 |

### 4.4 待裁决 / 待标定（❓，交编排者）

1. **前向 workspace 复用 vs 重算**：`A/W/γ` 是否复用 TA 前向 workspace（省算力，但需跨 op 接口）还是本 kernel 重算（自足，多算力）——建议 op-develop 先**重算**保自足，性能不达标再引复用。
2. **bwd 性能预算区间标定**：design §9.4 给 `0.9–16ms`（窗五量级）/ `150–240ms`（大 shape），依据是 fwd 的 2–3.5× 外推（❓），真机标定后回填。
3. **`bfloat16` 是否纳入支持**：design §9.3 声明 fp16/fp32 两行；若上游生产需 bf16，须补精度表行 + proto.yaml dtype。
4. **变长序列 `seq_lens` 是否首版支持**：design §4.1 列为可选；首版可只支持等长 `T`，变长在 L1 扩。

---

## ⑤ 证据与三态标注

| 结论 | 三态 | 凭据 |
|---|---|---|
| fwd 六件行号 | ✅ | 源码直读（`grep -n` 见 design §3.5.3） |
| NSA Bwd 形态 | ✅ | `example_tilelang_nsa_bwd.py:51-56/334/397/410-446` |
| 同步 pass 名 | ✅ | `tilelang/transform/pass_config.py:38/41/204-207/220-221` |
| #1346/#1384/#1851 | ✅ | `git log`：`8100a3a6` / `f3a3e1b2` |
| fork 缺陷机理 | ✅（负结果） | `M4_UBPREZERO.md` 头部 / `K4_RESULT.md` / `I3_REALSHAPE_SPEC.md §4` |
| fwd 性能 | ✅ | `examples/linear_attention_and_rnn/README.md:54-59` |
| 窗五 `0.4–4.5ms` 口径 | ⚠️ 引用（任务书） | 与 README 大 shape 并排；窗五真机数据 `J3_WINDOW5_PREP.md` 为 `no-device` |
| bwd 预算外推 | ❓ 推断 | 待真机标定 |
