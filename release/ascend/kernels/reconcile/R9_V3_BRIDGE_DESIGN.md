# R9 · v3 六梯 ↔ 生产 `gdn_asc` 桥接静态设计

> 性质：**静态设计件，不含实现**。依据 `docs/reassessment/05-plan-recommendations.md` R9（2026-10-24 裁定 **1-C**：静态设计先行）。
> 本件交评审；**评审通过后才决定是否实施**。实施并上卡验收前，不得宣称“P2 训练已用 v3”。
> 仓锚：S1=`$REPO_ROOT`，TA=`$WORKSPACE_ROOT/tilelang-ascend`。

## 0. 结论层

- **唯一结论**：本件只回答“桥怎么接、怎么验、风险在哪”，**不产生代码改动**；实施与否由评审裁定（§4）。
- **桥目标**：在既有生产消费链 `train_step.py → ops.gdn_delta → autograd_asc._GdnFn` 上，增加一个**可选**后端——反向六梯由 TA `gdn_bwd_v3_relay` 提供；首期**前向仍走 `gdn_asc.forward`**，开关默认关。
- **当前最大风险**：v3 **不是**生产消费者（重评 04 A-F1），且 `solve_a` 的 BK 真修（`301bfde0`）尚未上卡（T3a 待跑）——桥若在此时撤绕行或默认启用，等于把未验收路径接进训练。
- **一句话**：桥的价值是“让 v3 六梯**可被生产 autograd 选装**”，不是“把 v3 变成生产默认”；前者本件设计，后者须过评审 + 上卡验收。

## 1. 契约层

### 1.0 消费链与边界

```text
train_step.py → ops.gdn_delta → autograd_asc._GdnFn
                                  ├─ forward : gdn_kernel.forward → gdn_asc.forward（首期保持不变）
                                  └─ backward: [新] gdn_v3_bridge → gdn_bwd_v3_relay（可选，默认关）
```

- 生产侧的输入/输出契约以 `autograd_asc._GdnFn` 为准（`release/ascend/autograd_asc.py:345-384`；`gdn_kernel.py:29-34` 为薄壳 re-export）。
- v3 relay 在内部**自行重建前向缓存**（`gdn_bwd_v3.py` pass 1：cumsum/solve_a/scan1 → vnew/W/Kt/Δ'），因此桥**不消费**生产前向的中间量；代价是前向算两遍（§2 A/B 必须计入）。
- 多 batch 不在本桥范围：生产 `_GdnFn` 的 q/k/v 为 `[H,T,D]`（无 batch 维），桥内恒按 `B=1` 映射。

### 1.1 ① 输入布局映射

| 量 | 生产 `_GdnFn` / `gdn_asc`（现状） | v3 `gdn_bwd_v3_relay` | 桥内映射 |
|---|---|---|---|
| q | `[H,T,DK]`（无 batch） | `[B,H,T,DK]` | `unsqueeze(0)` |
| k | `[H,T,DK]` | `[B,H,T,DK]` | `unsqueeze(0)` |
| v | `[H,T,DV]` | `[B,H,T,DV]` | `unsqueeze(0)` |
| g | `[H,T]`（fp32） | `[B,H,T]` | `unsqueeze(0)` |
| beta | `[H,T]`（fp32） | `[B,H,T]` | `unsqueeze(0)` |
| dO | `[H,T,DV]` | `[B,H,T,DV]` | `unsqueeze(0)` |
| h0 | **无入参**（`gdn_asc` 内部 zeros，见 1.3） | `[B,H,DV,DK]`（转置朝向） | 见 1.3（首期 zeros） |
| 头维 | `H=heads`，`DK=q.size(-1)`，`DV=v.size(-1)` | 同 | 直传；**DK≠DV 不支持**（裁定 5-C） |

- 所有入参在过桥前 `.contiguous()`（v3 relay 内部对切片走 `_contig`，桥不依赖其兜底）。
- 生产侧形状校验（`gdn_asc._check`）先行；桥不重复实现校验逻辑，只在入口断言 `q.dim()==3`、`B==1`。

### 1.2 ② 梯度顺序（必须重排）

| 来源 | 顺序 |
|---|---|
| 生产 | `gdn_asc.BWD_GRADES = ("dq","dk","dv","dg","dbeta")`（`gdn_asc.py:45`）；`return_dh0=True` 时末尾追加 `dH0`；`_GdnFn.backward` 对 7 个 forward 入参返回 `(dq,dk,dv,dg,dbeta,None,None)` |
| v3 | `(dq, dk, dv, dbeta, dg, dh0)`（`gdn_bwd_v3.py:937-938`） |

- 桥内**显式重排**：`dg ← v3[4]`、`dbeta ← v3[3]`，其余同位；`v3[5]=dh0` 首期丢弃（1.3 选 A）。
- **禁止**直接 `return v3_out[:5] + (None, None)`——那会把 dg/dbeta 对调且六项对照未必全红；该错法就是负锚 N1 的靶子。

### 1.3 ③ h0 语义（二选一，取舍写死）

事实基线：

- 生产 `_GdnFn` **无 h0 入参**；`gdn_asc.forward` 在 `fwd_workspaces` 建 **zeros** `h0=(heads,dv,dk)`（`gdn_asc.py:513`，**转置朝向 [DV,DK]**）；`gdn_asc.backward` 默认 `return_dh0=False`，docstring 明写“本接口初状态恒零”。
- v3 relay 显式收 `h0` `[B,H,DV,DK]`（同转置朝向；`gdn_bwd_v2.scan1` 的 `H0: [B,H,DV,DK]`），并交回 `dh0` `[B,H,DV,DK]`。

**选 A（首期）：桥内传 zeros h0**。取舍：

- 收益：①生产对外语义就是零初态，选 A **不改变任何调用方行为**；②不动 `ops.gdn_delta` / `train_step` 的签名；③`dh0` 无消费者（`_GdnFn` 不返回它），算而不交不构成契约缺口。
- 代价：跨段续推（非零 h0）不支持；`dh0` 被丢弃；若将来出现“状态续推”需求，须先立需求再走 B。

**选 B（备选，不在本期）**：给 `_GdnFn` 增加可选 h0 入参。代价：改 `ops.gdn_delta` / `train_step` 调用面；h0 需 `[H,DK,DV] → [1,H,DV,DK]` **转置 + cast**；backward 返回值数量变化（追加 `dh0`），所有调用方与测试同步。收益仅在跨段续推成立时兑现。

朝向纪律：内核侧状态是 **[DV,DK] 转置朝向**（生产 h0 workspace 与 v3 H0 一致）；torch 尺子 `_gdn_torch` 是 `[DK,DV]`。桥若做任何 h0 搬运，必须显式 transpose，禁止按名字猜。

### 1.4 ④ dtype / 精度

- 生产主档 **fp16**（P1-1m：fp16 是生产位宽；fp32 训练步未端到端验）；**g / β 为 fp32**；q/k/v/dO 随主 dtype。
- 生产内核出口 **恒 fp32**，交回 autograd 前按各输入原件位宽交货——`autograd_asc.py:379-384` 的 **F2 位宽纪律**；fp16 档不依赖引擎静默折算。
- v3：`dt = q.dtype`；**`dbeta`/`dg` 是 fp32 输出**，`dq/dk/dv/dh0` 随主 dtype（`test_gdn_bwd_v3.py:176` 注释）。净契约与生产“逐输入交货”一致，但桥必须**自己 `.to()`**：dq/dk/dv → q/k/v dtype，dg → g dtype，dbeta → beta dtype。
- **fp64 仅 gradcheck 用**：桥在 `q.dtype==torch.float64` 时必须**绕开 v3**，走 `_GdnFn` 现有 torch 尺子分支（TL kernel 不承担 fp64 语义）。
- bf16 不在本桥范围（ruling 3 已排除；v3 的 bf16 GEMM 路线另立）。

### 1.5 ⑤ autograd 包装

- 形态：新增 `torch.autograd.Function`（建议名 `_GdnV3Fn`，建议落点 `release/ascend/gdn_v3_bridge.py`；落点以评审为准）。主选方案 = 新 Function + `ops.gdn_delta` 里**env 门控路由**（默认 `gdn_asc`）；备选 = 直接分支在 `_GdnFn.backward` 内（零调用面改动，但两后端混居、A/B 隔离差）。两者都要求：可一键回退、失败登记 blocker。
- `ctx.save_for_backward(q,k,v,g,beta)`（首期不含 h0）；`ctx.target` 同 `_GdnFn`；按需存 `out_dtype`。
- **`enable_grad`**：`Function.backward` 默认在 `no_grad` 里跑。v3 路径只调 TL JIT kernel + host 循环，不需要 autograd 图；但桥的**回落路径**（fp64/cpu/target 不可用）若走 `_gdn_torch` + `torch.autograd.grad`，必须像 `_GdnFn.backward` 注释（`autograd_asc.py:367-369`）那样包 `torch.enable_grad()`，否则报 “element 0 of tensors does not require grad”。这是已踩过的真环境差异，桥内必须保留。
- 路由开关：建议 env（如 `SYS1_GDN_V3_BRIDGE=0|1`，默认 0）；v3 不可用（无 torch_npu / target≠ascend / TA 包缺失）时**回落 gdn_asc 并登记 blocker**，不假绿。
- 同步：v3 relay 内部已有 `torch.npu.synchronize()`；桥不重复加同步，避免改变时序语义（实现期复核）。

### 1.5b 外部参照：AIC/AIV tile 级 flag 流水（得物 910C 融合算子，2026-10-27 补记）

得物在 910C 上自研 `matmul_allreduce_add_rmsnorm_910c` 融合算子（源码 `csrc/mc2/matmul_allreduce_add_rmsnorm910c/`）展示了多阶段片上流水的具体 AscendC 写法：
- **AIC 侧**：细粒度 matmul，每算完一个 tile 就在 GM 上置位 tile-ready flag；
- **AIV 侧**：轮询 AIC 的 flag，tile 就绪即启动跨 rank 归约 + RMSNorm，不回 host；
- **流水协议**：flag 分三段（AIC tile 就绪 / 跨 rank 协议状态 / RMS 就绪），AIC 越往后算，AIV 的归约越能与之重叠，形成细粒度流水；
- **双层回退**：Python 侧形状守卫（不满足返回 None 走未融合路径）+ 算子内部 Dispatcher 回退到标准算子组合，保证正确性不依赖融合算子的形状覆盖度。

对本桥的启发：① v3 前置件（cumsum→solve_a→scan1）若未来做片上融合，可参照此 flag 协议设计 AIC/AIV 流水，替代当前的 GM workspace 中转；② 双层回退（形状守卫 + 算子内回退）是融合算子不改正确性语义的结构保证，桥的 env 路由 + blocker 登记同属此模式。

来源：得物技术《昇腾 910C 大模型推理优化（下）》§二.3（2026-10-21）；彼文在 910C/CANN 9.x 上实现，本项目在 910B/CANN 8.5.2 上实现，具体指令与 flag 地址布局不可直接搬。

### 1.6 ⑥ pad 契约（T%C≠0 由 host pad）

- 现状：v3 relay 入口 `assert T_len % C == 0`（`gdn_bwd_v3.py:699`）——**拒绝**而非实现尾块。
- 桥策略：**host pad 到整倍数**，`NC=ceil(T/C)`；哑 token 契约（教材 `docs/learning_path/06-accelerators-and-kernels.md` §6.9）：q 任意（建议 0）、**k=0、v=0、g=0、β=0、dO=0**；算完按有效长度切片。
- 三条硬点：① **g 的 pad 值必须为 0**（g≠0 会衰减旧状态，§6.9 实验 C 已示）；② dO pad=0 ⇒ 不贡献损失，pad 位梯度丢弃；③ 只有有效位参与验收比较。
- 参照与验证：本地对照 `_pad_invariance.py`（TA `examples/gdn_bwd_v2/_process/`，由 `adjudication-remediation-1024` B5 交付）先证“golden 同 pad”不变；**完整前后对照（前向 + 反向、有效位六梯度逐项）在本桥实现期执行**（V3）。pad 会增加 chunk 数，多 chunk 残差须计入容差说明。
- **实测边界（B5 交付后补记，2026-10-24）**：在“pad 位于序列末尾且损失仅覆盖有效 token（无终态损失）”的前提下，实测 g_pad 违约（-0.5）对**有效位六梯度零影响**（因果性：末尾 pad 不回传到更早位置；Δ'_last=0 时 glast_pad 乘的是零）——梯度面负锚不会触发，B5 的负锚由 pad 契约检查器 + 终态衰减（rel=1.0）承担。因此 g=0 硬点的真正约束对象是**状态链**：① 前向终态 H_T（被 g_pad 错误衰减）；② 跨段训练把 H_T 递给下一段时的下游损失。若桥启用状态传递（segment-level state passing），g_pad 违约将通过终态泄入下一段梯度，此时 N4 必须含终态对照项；若桥只服务单段、丢弃终态梯度（现状 Δ'_last=0），g_pad 违约不可观测但契约仍应保持（防未来启用状态传递）。

## 2. 验证层（实施后全绿才可默认启用）

### V1 端到端 gradcheck（fp64 CPU golden + 有限差分）

- 在 autograd 边界做：`loss = (out * dO).sum()`，`torch.autograd.grad(loss, (q,k,v,g,beta))`；与 CPU fp64 golden（`_np_chunk_bwd` / `_gdn_torch`）逐项比 rel，并用有限差分抽点复核。
- 形状（首期）：T=64/128/256，H=1/2，DK=DV=64/128，C=64；目标形（B1 H16 T8192 DK128 DV128 C64）上卡单列。
- 通过线：六项 rel ≤5e-3（项目主判据；`ratio` 仅诊断列）。

### V2 负锚清单（具名，“故意错 ⇒ 必红”）

| 编号 | 注入 | 期望 |
|---|---|---|
| N1 错序 | 不重排，直接按 v3 顺序 `(dq,dk,dv,dbeta,dg)` 交回 | 六项对照必红（dg/dbeta 对调不得被容差吞掉） |
| N2 NaN | 在 q 或 dO 注入 NaN（或令某梯度输出 NaN） | 门必红；禁止 `max()` 聚合吞 NaN（重评 04 §7 的教训） |
| N3 尾块拒绝 | T%C≠0 且关闭 pad | 必须显式拒绝（ValueError/AssertionError），禁止静默截断 |
| N4 尾块 pad | T%C≠0 + pad 开启 | 有效位六梯度与未 pad 运行一致（≤容差）；pad 位不参与比较 |

### V3 同形状 A/B（与 gdn_asc）

- **正确性先**：同输入、同 h0（零）、同 dtype、同 target，对 `gdn_asc` 六梯逐项 rel；不一致先按 R23 三分（产品缺陷 / 骨架语义错位 / 环境噪声）归因，不改产品。
- **性能后**：warmup + median，同形状同工作量；**必须计入“前向算两遍”的成本**（生产 forward + v3 relay 内部 pass 1）；倍率带误差棒，不给绝对值承诺（重评 04 §9 / 裁定 2-B）。
- 回归：`release/tests/test_ascend_gradcheck.py`（基线 20 passed / 1 skipped）与 `test_ascend_train_step.py` 不回归；上卡用 `gradcheck_card.py` 与 `run_delta_card` 同口径。A/B 的对照组必须同车同窗，禁止跨窗拼数。

## 3. 风险层

| 风险 | 说明 | 处置 |
|---|---|---|
| **BK 真修未上卡前桥不撤绕行** | v3 relay 目前硬编码 `BK=DK`（`solve_a(..., DK, C, DK)`）；`301bfde0` 仍是待验候选（T3a） | 桥**不撤绕行、不启用 BK64**；T3a 通过后再由独立变更裁定是否撤 |
| **C5 点火条件（引 D1）** | C5 点火 = R4 完成 ∧ **R9 静态设计过评审** ∧ 教师选型与许可核验落定 | 三者缺一不点火；本件只补第二项的前置 |
| v3 非生产消费者 | 重评 04 A-F1：生产链仍是 `gdn_asc`，v3 六梯无生产调用 | 桥默认关；A/B + 上卡验收前不宣称“P2 用 v3” |
| 薄边界项 | `dk`/`dbeta` 在 8~16 chunk 摆动（1.3e-3~8.0e-3），多 chunk 残差未闭环 | A/B 按形报告；不把“形特定”写成“普遍 <5e-3” |
| v2 oracle 自身风险 | `scan2a/scan2b` 同形 BK 分半未验（重评 04 §3/§6） | 对拍分母同时给 fp64 golden，不只信 v2 |
| 文档漂移 | v3 relay docstring 仍写“dg 尚未接入 / 初版未上卡”，与代码现状不符 | 实施期先修 docstring（TA 仓，另行授权） |
| 双前向成本 | 首期前向仍走 `gdn_asc`，v3 relay 内部再跑一遍前向缓存 | A/B 如实计入；若成本不可接受，再议“v3 全量替换前向”（另立设计） |

## 4. 评审出口（本件 DoD）

1. 本文件过评审，结论记入 run notes；
2. 结论二选一：**实施**（立实现任务，按 §1 契约 + §2 验证面）或**不实施**（登记理由，v3 保持研究件）；
3. 无论哪种，生产默认路径不变（`gdn_asc`），直至另行验收。

## 5. 引用锚（文件 + 节/行）

- `release/ascend/autograd_asc.py`：`_GdnFn`（345-384，含 F2 位宽纪律 379-384、`enable_grad` 注释 367-369）。
- `release/ascend/kernels/gdn_kernel.py`：re-export 面（29-34）。
- `release/ascend/kernels/gdn_asc.py`：`BWD_GRADES`（45）、zeros h0（513）、forward/backward 签名（652/681）。
- TA `examples/gdn_bwd_v3/gdn_bwd_v3.py`：relay（668）、尾块断言（699）、六元组返回（937-938）。
- `docs/reassessment/04-ascend-gdn.md` §2（消费者）/§3（A-F2）/§6（A-F5）/§9（A-F8）。
- `docs/reassessment/05-plan-recommendations.md` R9、§5 裁定 1-C。
- `docs/learning_path/06-accelerators-and-kernels.md` §6.9（哑 token 契约与实验 C）。
- `adjudication-remediation-1024` B5：`_pad_invariance.py`（TA `examples/gdn_bwd_v2/_process/`）。
- 现状索引：`openspec/changes/teacher-p2-production-full/changes/p2-13-ascend-runtime/STATUS.md`。
