# I3_REALSHAPE_SPEC — P2「真形档」拓扑定义、判决件口径、预算与窗六一条命令

> 作者：p2-13 代理 **I3**（真形档判决件制备）。**不占卡、不改既有产品件语义。**
> 交付物：本文件 + `ascend/train_step_real.py`（判决件）+ `ascend/kernels/reconcile/I3_probe_gdn_bwd_shape.py`（只读探针）。
> 需求出处：`openspec/changes/teacher-p2-production-full/changes/p2-13-ascend-runtime/tasks.md:37,72`
> （口径改写：`P2 三栈联调` 原文"卡窗即跑 0.8B 一步" ⇒ 拆 **接线档**（已过）/ **真形档**（本件）=0.8B 生产拓扑一步）。

**一句话**：把 0.8B 学生的一步拆成可数、可复跑的 op 清单，在**混合栈**上跑一步 fwd+bwd+opt，
产出 `R6/R7` 熔断判定要的**每 op 调用次数 / 时延 / 显存**记账，并把卡上那一枪写成一条命令。

---

## §0 口径与边界（谁要什么、本件刻意不做什么）

| 项 | 内容 |
|---|---|
| 消费者 | `tasks.md:39` **R6/R7 成本记账与 gate 报价**（恢复 C5 的前置）；`tasks.md:37` **P2 三栈联调**（验收＝"0.8B 训练步跑通"） |
| 熔断线 | `design.md:183` **¥600**（超线即停上报）；单价 `design.md:174` **910B ¥20/h** |
| 为什么要"每步 op 次数" | 报价与熔断要的是"生产拓扑每步各算子被调多少次、每次多贵、峰值多少显存"，不是"接线对不对" |
| 接线档（既有） | `ascend/train_step.py`：`d=32, seq=16, heads=2, dim=16`（`train_step.py:402`），自陈"本波**只验接线正确性**……**不验内核数值 rel/V\***"（`train_step.py:6-7`）；`train_step.py:42-44` 明确**不含 `gdn_conv`** ⇒ 它证明的是八门面接线，**不证明 0.8B 形状能落到内核上** |
| 本件不做（如实登记） | ① 不做 HF→自研栈坐标置换（`production/backbone/layout.py::rope_pair_permutation`）——影响数值口径，不影响 op 结构/形状/次数/显存；② 不实现 gradient_checkpointing（`gate08b.yaml:35` 开着）⇒ 本件记的是**不重算**的峰值，是**上界**；③ 不做 `gdn_delta` 的 qk-l2norm（门面契约里没有这一步）；④ 输入是随机张量，不产评测分 |
| 既有件零改动 | `train_step.py` / `autograd_asc.py` / `*_asc.py` / `card_run.sh` / `window*_card.sh` / `gradcheck_card.py` / `run_delta_card.py` / `runs/**` / 任何 `tasks.md` —— 全未动（本件只 import 既有实现） |

---

## §1 拓扑真源（数字全部引自配置原文，`文件:行` 即凭据）

模型真源：`bench/ms_models/models/Qwen--Qwen3.5-0.8B/snapshots/master/config.json` 的 `text_config`。

| 量 | 值 | 凭据 | 落到的 op 面 |
|---|---|---|---|
| `hidden_size` | 1024 | `config.json:16` | 全局 d |
| `num_hidden_layers` | 24 | `config.json:56` | 层数 L |
| `num_attention_heads` | 8 | `config.json:55` | 全注意力 q 头 |
| `num_key_value_heads` | 2 | `config.json:57` | GQA：`group = 8/2 = 4` |
| `head_dim` | 256 | `config.json:14` | 全注意力头维 |
| `intermediate_size` | 3584 | `config.json:18` | MLP 隐宽 |
| `vocab_size` | 248320 | `config.json:61` | 词嵌入（`tie_word_embeddings=true`，`config.json:59`） |
| `rms_norm_eps` | 1e-6 | `config.json:58` | 两型 RMSNorm |
| `attn_output_gate` | true | `config.json:10` | q_proj 出口 = 每头 `2×head_dim`（前半查询 / 后半门） |
| `full_attention_interval` | 4 | `config.json:13` | 每 4 层 1 个全注意力 |
| `layer_types` | 24 项 | `config.json:19-44` | 全注意力落 **idx 3/7/11/15/19/23** ⇒ **18 `linear_attention` + 6 `full_attention`** |
| `linear_conv_kernel_dim` | 4 | `config.json:45` | GDN 短卷积核宽 |
| `linear_num_key_heads` / `linear_key_head_dim` | 16 / 128 | `config.json:47` / `:46` | GDN 键宽 = 2048 |
| `linear_num_value_heads` / `linear_value_head_dim` | 16 / 128 | `config.json:48` / `:49` | GDN 值宽 = 2048 |
| `partial_rotary_factor` | 0.25 | `config.json:72` | 旋转宽 = `256×0.25 = 64` |
| `rope_theta` | 1e7 | `config.json:71` | rope 角表 |
| `dtype` | bfloat16 | `config.json:11` | 权重原生位宽（训练面 `gate08b.yaml:13` 同） |
| LoRA `r` / `alpha` | 16 / 32 | `gate08b.yaml:32` / `:33`（同 `production/sft.py:140-142`） | `lora_scaling = 32/16 = 2.0` |
| LoRA 挂载面 | `q_proj,k_proj,v_proj,o_proj` | `production/sft.py:146` `LORA_TARGET_LEAVES` | **只有 6 个全注意力层有 LoRA** ⇒ 可训参数 = 6×4×2 = **48** |
| 读点位字母表 | 26 | `sys1/decision/render.py:51` `LETTERS = string.ascii_uppercase` | 读点位头 `n = 26` |
| 损失 | 读点位软交叉熵 | `production/sft.py:533`（`option_scores`，同 `sys1/decision/readout.py:90`） | `(B, k)` 分数 |

**派生量（纯算术，不含新数字）**

| 派生量 | 算式 | 值 |
|---|---|---|
| `q_gate_n` | `heads×head_dim×2` = 8×256×2 | **4096** |
| `kv_n` | `kv_heads×head_dim` = 2×256 | **512** |
| `o_in` | `heads×head_dim` = 8×256 | **2048** |
| `gdn_key_dim` | 16×128 | **2048** |
| `gdn_val_dim` | 16×128 | **2048** |
| `conv_dim` | `2×key + value` = 2048×2+2048 | **6144** |
| `rotary_dim` | `256×0.25` | **64** |
| `group` | 8/2 | **4** |
| 主干权重元素数（实测） | — | **753,474,368**（≈0.75B；`P2-REAL-PARAMS total=`） |

> 逐层 op 顺序真源：`transformers/models/qwen3_5/modeling_qwen3_5.py`
> `Qwen3_5GatedDeltaNet` `:503-662`、`Qwen3_5Attention` `:748-820`、`Qwen3_5MLP` `:823-836`、
> `Qwen3_5RMSNorm` 零中心 `(1+w)` `:840-854`、`Qwen3_5RMSNormGated` 逐头 + silu 门 `:217-233`、
> `Qwen3_5DecoderLayer` 残差/前后置 norm 次序 `:860-913`。

### §1.1 ⚠️ 一处**缺规格**（如实报，不替决策人定）

**"一桌"→ `(seq, batch)` 的映射未在计划/配置里钉死。**

- `gate08b.yaml:21` `max_length: 8192` —— **单样本** token 上限（"超长直接丢"）；
- `gate08b.yaml:43` `max_tokens: 16384` —— **一桌补齐后**的字数上限；`:44` `accum: 4`；
- `:55` 折算口径："等效字数 = 一桌补齐后字数 × 攒桌数 = 16384 × 4 = 65536 字 / 次下笔"；
- `:58` **本身就把这条标成未定项**："910B 单卡真会不会 OOM、要不要把 `max_tokens` 再对折 —— 只有 C5 首跑能给数"。

⇒ **一步（一桌）的实际 token 数候选**（**均未定，需决策人钉死**）：

| 候选 | seq | batch(B) | 一桌 token | 依据 |
|---|---|---|---|---|
| A | 8192 | 2 | 16384 | 单样本吃满 `max_length`，一桌拼两样本吃满 `max_tokens` |
| B | 8192 | 1 | 8192 | 一桌一样本（`max_tokens` 未用满） |
| C | 16384 | 1 | 16384 | 一桌即一样本（与 `:21` 的"超长直接丢"冲突，风险最高） |
| D | 4096 | 4 | 16384 | 若 `:58` 选"对折"降级（`max_tokens` 8192 / `max_length` 4096 一类） |

**本件的处置**：`--seq`（缺省真档 8192）与 `--batch`（缺省 1）都是 CLI 旋钮，**不替决策人定形状**；
判决行与 `P2-REAL-PARAMS` 行**如实打印实际用的 seq/batch**（`seq=<n> batch=<B>`）。
窗六应至少跑 **候选 A（`--seq 8192 --batch 2`）** 与 **候选 B（`--seq 8192 --batch 1`）** 两档，
好让 C5 的 OOM 判定有 A/B 对照。

> 另注：`seq` 影响 `attn_sw` 的编译键（键里带 `w<seq>`，见 §5.1），而 op **调用次数与 seq 无关**
> ⇒ 记账表的"次数"列对 A/B/C/D 四候选**通用**，只有"时延/显存"列随 seq 变。

---

## §2 一步的 op 清单（逐层型 → 逐 op → 每步调用次数）

一步 = **1 次前向**；判决件跑 **2 次前向**（`scores0` / `scores1`，用于判"更新后 loss 是否下降"）+ 1 次反向。

### §2.1 线性注意力层（`linear_attention` × **18**；真源 `modeling:503-662`）

| # | op 面 | 门面 | 形状要点 | 每层次数 |
|---|---|---|---|---|
| 1 | `rmsnorm(in_norm)` 零中心 | 无（`rms_norm` 直写） | (T,1024) | 1 |
| 2 | `linear` ×4 | **`autograd_asc.linear`** | `in_qkv`→6144 / `in_z`→2048 / `in_b`→16 / `in_a`→16 | 4 |
| 3 | `gdn_conv`（短卷积 + 内含 silu） | **`autograd_asc.gdn_conv`** | 通道优先 `(1, 6144, T)`，k=4 | 1 |
| 4 | 切 q/k/v 三段 | 无（`layout_split`） | `[key,key,value]` = 2048/2048/2048 | 2 |
| 5 | `beta=sigmoid(b)`、`g=-exp(A_log)*softplus(a+dt_bias)` | 无（`gdn_gate`） | (16, T) | 1 |
| 6 | **门摊 2D→3D** | 无（`gdn_g_spread`，见 §4） | (16,T)→(16,T,128) | 1 |
| 7 | `gdn_delta` 递推 | **`autograd_asc.gdn_delta`** | fp32 面唯一；q/k (16,T,128) v (16,T,128) | 1 |
| 8 | `rmsnorm_gated`（逐头 + silu 门） | 无（`rms_norm_gated`） | (T,2048) | 1 |
| 9 | `linear(out_proj)` | **`autograd_asc.linear`** | (T,2048)→1024 | 1 |
| 10 | 残差加 | 无（`residual_add`） | (T,1024) | 1 |
| 11 | `_mlp`：`rmsnorm` + `linear`×3 + `silu_mul` + `residual_add` | 前两个 `linear` 走门面 | gate/up 3584、down 1024 | rmsnorm 1 / linear 3 / silu 1 / res 1 |

⇒ 每层：`linear` **8**、`gdn_conv` 1、`gdn_delta` 1、`rmsnorm` 2、`rmsnorm_gated` 1、
`layout_split` 2、`gdn_gate` 1、`gdn_g_spread` 1、`silu_mul` 1、`residual_add` 2。

### §2.2 全注意力层（`full_attention` × **6**，idx 3/7/11/15/19/23；真源 `modeling:748-820`）

| # | op 面 | 门面 | 形状要点 | 每层次数 |
|---|---|---|---|---|
| 1 | `rmsnorm(in_norm)` | 无 | (T,1024) | 1 |
| 2 | `linear` ×3 + `lora` ×3 | **`.linear` + `.lora`** | `q_proj`→4096（每头前半查询后半门）/ `k_proj`→512 / `v_proj`→512 | linear 3 / lora 3 |
| 3 | 拆 q / 门 | 无（`layout_split`） | (T,8,2,256) → q(T,8,256) g(T,2048) | 1 |
| 4 | `rmsnorm(q_norm)` + `rmsnorm(k_norm)` | 无 | 逐头 | 2 |
| 5 | `repeat_kv`（GQA 补齐） | 无 | kv 2→8 头，×2（k、v） | 2 |
| 6 | `layout_pack`（q,k,v → (T,3,8,256)） | 无 | — | 1 |
| 7 | `rope` | **`autograd_asc.rope`** | 只转 q/k 两槽；rotary 64 / 头宽 256 | 1 |
| 8 | `attn_sw` | **`autograd_asc.attn_sw`** | 全因果（window=seq） | 1 |
| 9 | 出口门 `sigmoid(g)` | 无（`attn_out_gate`） | (T,2048) | 1 |
| 10 | `linear(o_proj)` + `lora(o_proj)` | `.linear` + `.lora` | 2048→1024 | linear 1 / lora 1 |
| 11 | 残差加 | 无 | — | 1 |
| 12 | `_mlp`（同上） | 门面 | — | rmsnorm 1 / linear 3 / silu 1 / res 1 |

⇒ 每层：`linear` **7**、`lora_apply` **4**、`rope` 1、`attn_sw` 1、`rmsnorm` **4**、
`layout_split` 1、`repeat_kv` 2、`layout_pack` 1、`attn_out_gate` 1、`silu_mul` 1、`residual_add` 2。

### §2.3 读点位头（真源 `production/sft.py:533` → `sys1/decision/readout.py:90-114`）

| # | op 面 | 门面 | 形状要点 | 每次前向 |
|---|---|---|---|---|
| 1 | `readout(embed, ids)` token 嵌入 gather | **`autograd_asc.readout`** | (T,) → (T,1024) | 1 |
| 2 | `rmsnorm(final_norm)` | 无 | (T,1024) | 1 |
| 3 | `readout(h, pos_ids)` 取末位 | **`autograd_asc.readout`** | (T,1024) → (B,1024) | 1 |
| 4 | `letter_rows = embed[letter_ids]`（冻结表取样） | 无（`letter_rows_gather`） | (26,1024) | 1 |
| 5 | `linear(rows, letter_rows)` 字母打分 | **`autograd_asc.linear`** | (B,1024)×(26,1024)ᵀ → **(B,26)** | 1 |
| 6 | 软交叉熵 | 无（`loss_ce`） | 每次**跑**只 1 次（非每前向） | — |

### §2.4 每步调用次数表（**结构量，对 §1.1 四候选通用**）

前向 1 次：`linear` = 18×8 + 6×7 + 1 = **187**；两前向 ⇒ 374。

| op | 每层/头 ×次数 | 1 前向 | 2 前向（判决件实测） |
|---|---|---|---|
| `linear` | 18×8 + 6×7 + 1 | 187 | **374** |
| `gdn_conv` | 18×1 | 18 | **36** |
| `gdn_delta` | 18×1 | 18 | **36** |
| `attn_sw` | 6×1 | 6 | **12** |
| `rope` | 6×1 | 6 | **12** |
| `lora_apply` | 6×4 | 24 | **48** |
| `readout` | 头 ×2 | 2 | **4** |
| `rmsnorm`（无门面） | 18×2 + 6×4 + 1 | 61 | **122** |
| `rmsnorm_gated`（无门面） | 18×1 | 18 | **36** |
| `residual_add`（无门面） | 24×2 | 48 | **96** |
| `silu_mul`（无门面） | 24×1 | 24 | **48** |
| `layout_split`（无门面） | 18×2 + 6×1 | 42 | **84** |
| `repeat_kv`（无门面） | 6×2 | 12 | **24** |
| `layout_pack` / `layout_embed` / `letter_rows_gather`（无门面） | — | 6 / 1 / 1 | **12 / 2 / 2** |
| `attn_out_gate` / `gdn_gate` / `gdn_g_spread`（无门面） | — | 6 / 18 / 18 | **12 / 36 / 36** |
| `loss_ce`（无门面） | 每次跑 1 次 | — | **1** |

**可复跑命令**（判"结构不变"）：
```bash
# host（秒级；dry 缩宽度不缩层数 ⇒ 次数结构与真档逐项相同）
<V>/python release/ascend/train_step_real.py --target cpu --dry | grep -E '^COUNTS '
# 容器（真形状；桩发射）
docker exec <ctr> bash -c 'cd <release>; source /usr/local/Ascend/cann-8.5.0/set_env.sh; \
  export BISHENG_HOME=/usr/local/Ascend/cann-8.5.0/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201; \
  TILELANG_CACHE_DIR=$(mktemp -d) python3 ascend/train_step_real.py \
    --target ascend --stub-run --device cpu --seq 512' | grep -E '^COUNTS '
```
> `<V>` = `/Users/pengweiye/Documents/codes/system-one/release/.venv/bin/python`。

**实测对照组**（两档 `COUNTS` 逐项相同 ⇒ dry 的"层数/调用次数结构不变"判据成立）：

| op | host dry（seq=16, d=64, fp32） | 容器真形（seq=512, d=1024, fp16） |
|---|---|---|
| `attn_sw` | 12 | 12 |
| `gdn_conv` | 36 | 36 |
| `gdn_delta` | 36 | 36 |
| `linear` | 374 | 374 |
| `lora_apply` | 48 | 48 |
| `readout` | 4 | 4 |
| `rope` | 12 | 12 |

---

## §3 混合栈分流与 SWAP（**不许静默**）

**分流机制**：`autograd_asc` 八门面内部按形状/位宽路由到自研件，自研件的 `plan()` **拒形状即静默落
eager**（`gemm_kernel.py:46-51`，C2 已判的盲点）。本件在**不改既有件**的前提下给 **9 个 plan 入口**
套一层**只读观察器**（`install_plan_watch`，`train_step_real.py:293-303,314-347`）：

```
linear · linear_dW · gdn_delta · gdn_delta_bwd · gdn_conv · gdn_conv_bwd · attn_sw · rope · readout
```

`plan` 回 `None` ⇒ **立刻打一行并计数**（`<族> <首张量形状> dtype=<dt> -> native`），绝不静默。

**实测（真形档容器轨）唯一拒绝点**：

```
SWAP linear (1, 1024) dtype=float16 -> native      # scores=linear(rows,letter_rows) 前向 ×2
SWAP linear (1, 26)   dtype=float16 -> native      # 同一条腿的 dgrad
SWAP linear_dW (1, 26) dtype=float16 -> native     # 同一条腿的 dW
PLAN linear requested=702 rejected=3
PLAN linear_dW requested=208 rejected=1            ⇒ swaps = 4
```

⇒ **每步 2 次字母头 matmul `(B,1024)×(1024,26)ᵀ → (B,26)` 落 `torch_npu` 原生**（出口 `n=26`
不在 gemm 块阶梯 64/32/16 上）；其余 8 族 **rejected=0**。该数（4）**对 seq/batch 不敏感**，是
R7 报价里"底座上那两笔小账"的确定来源。

> 观察器**零行为改动**：调原函数、看返回值、不改参数、不包 `get_compiled`。

---

## §4 既有件缺陷：`gdn_asc.backward` 对 2D `g` 回报 3D `dg`（**真形档才现形**）

### §4.1 现象（容器真形档首跑，EXIT=1）
```
RuntimeError: Function _GdnFnBackward returned an invalid gradient at index 3 -
got [16, 512, 128] but expected shape compatible with [16, 512]
```
`index 3 = g`（`_GdnFn.forward(ctx, q, k, v, g, beta, …)` 第 4 个输入）。真形档 `g` 是
**每 token 一个标量门** (heads, seq)，而回报的 `dg` 是 (heads, seq, dk)。

### §4.2 定位（只读探针 `I3_probe_gdn_bwd_shape.py`，本白名单内）
| 轨 | `g_form` | 结论 |
|---|---|---|
| host cpu | 3d（每通道门） | **SAME-SHAPE**（五梯全同形） |
| host cpu | 2d（每 token 门） | **SHAPE-MISMATCH**（`dg` 回报 3D） |
| host cpu dry 档 (16,16,16,16) | 2d + 门面/autograd | **OK**（`heads==seq==dk==16` 巧合掩盖） |
| host cpu 真形状 (16,512,128,128) | 2d + 门面/autograd | **AUTOGRAD-RAISE**（同容器报错） |

**根因**：`gdn_asc.py` 的 `backward` docstring `:694` 自陈"五个与输入同形的 fp32 梯度"，
但实现里 `dg` **恒按 `_gate3` 摊开后的 3D 形**回报（昇腾伴随路 `:716,720-722` 与 torch 尺子路
`:734` 都是 `gg/g3` 的 3D 形）⇒ **违约自陈**。`torch.autograd` **只在回报形状不能
`is_expandable_to` 输入形状时才拒**：`(16,16)`→`(1,16,16)` 与 `(16,16,16)` 对齐要求
`heads==seq or seq==dk`，dry 档三者皆 16 故放行；真档 16≠512 故拒。

**为什么接线档/dry 看不到**：全注意力层之前（idx 0-2）的 GDN 层输入全来自**冻结权重** ⇒ 全不
`requires_grad` ⇒ `_GdnFn.backward` 根本不被调用；真形档里 LoRA 使 `h` 在 idx3 之后带梯度 ⇒
idx4 起的 GDN 层输入带梯度 ⇒ **首次被调用 ⇒ 缺陷现形**。

### §4.3 处置（既有件**禁改** ⇒ 本件内等价桥 + 显式公告）
`train_step_real.py` 的 `spread_gate(gate, dk)`：把 `(heads,seq)` **显式**摊成 `(heads,seq,dk)`
再交门面（广播是等价的——数学上每键通道同门），反向由 `ExpandBackward` **折回 2D**，与输入的
`(heads,seq)` 形状对齐。**不静默**：只喊一次 `GDN-G-SPREAD gform=2d->3d dk=<dk> why=…`，
并计入 `NOFACADE gdn_g_spread calls=36`。

### §4.4 建议修点（**登记给决策人，本件不动**）
`ascend/kernels/gdn_asc.py:694` 的契约与 `:716,720-722,734` 的回报形状二者必居其一：
① 让 `backward` 在 `g.dim()==2` 时把 `dg` 折回 `(heads,seq)`（与自陈一致）；或
② 改 `:694` 的 docstring，明确"`dg` 恒为 `_gate3` 摊开形"并把契约写进 `plan_backward` 判据。
⚠️ 该缺陷**只在真形档/带梯度输入下现形**，**任何 d=32 冒烟都测不到** ⇒ 建议后续单派一件修
（含 `I3_probe_gdn_bwd_shape.py` 作验收探针）。

---

## §5 判决件接口（`ascend/train_step_real.py`）

### §5.1 CLI 档位
| 档 | 命令 | 用途 | 数值 | 发射 |
|---|---|---|---|---|
| **host 自证** | `--target cpu --dry` | 秒级；层数/次数结构不变 | 真值（fp32） | 无 |
| **容器证可编** | `--target ascend --no-run` | A3 三量桩 + **真编译** + 发射桩 | `void`（明标） | 桩 |
| **容器键枚举** | `--target ascend --keys-only` | 只枚举编译键 | `void` | 桩 |
| **容器预算档** | `--target ascend --stub-run --device cpu` | 编正+反**全部**键 + 走完 fwd/bwd/opt | `void`（loss0==loss1==ln26） | 桩 |
| **卡窗真跑** | `--target ascend --device npu --dtype fp16` | **真形档终局** | 真值 | 真 |

旋钮：`--seq`（真档缺省 8192）· `--batch`（缺省 1）· `--layers` · `--dtype {fp16,fp32}` ·
`--head {letter,vocab}` · `--lr` · `--seed`。

### §5.2 判决行 / 记账行语法（`R6/R7` 直接抄）
```
P2-REAL-TOPO d=… L=… gdn=… full=… heads=… kv=… hd=… inter=… vocab=… conv_k=… gdn_h=…/… qkv_n=…/…/… conv_dim=… rotary=… lora_r=… lora_alpha=… dry=… src=<config 相对路径>
SWAP <族> <形状> dtype=<位宽> -> native                     # 每次拒形状一行（不许静默）
P2-REAL-STEP target=<t> loss0=<v> loss1=<v> reduced=<b> params=<n> grads_ok=<k>/<n> swaps=<s> native=<m> mode=<run|stub-run|compile-only|keys-only>
P2-REAL-PARAMS total=<N> trainable=<n> frozen=<N-n> seq=<S> batch=<B> head=<h> dtype=<dt> dry=<b>
COUNTS <op> calls=<n> p50_ms=<x>                            # 每 op（R6 的次数列 / R7 的时延列）
NOFACADE <op> calls=<n>                                     # 无门面的逐元素/布局 op（如实登记）
PLAN <族> requested=<n> rejected=<m>                        # 分流明细
COMPILE-KEY <key> requests=<n>                              # 每键请求次数（= 编译预算的输入）
P2-REAL-COMPILE count=<n> keys=<k> fails=<f>                # 编译数
P2-REAL-MEM alloc_gb=<a> peak_gb=<p>                        # 仅 npu 有读数；非 npu 明写 n/a
P2-REAL-MEM-ANALYTIC weights_gb=… opt_gb=… attn_scores_gb=… conv_gb=… total_gb=… lora_numel=…
P2-REAL-WALL sec=<t>
KERNEL-USED-PASS kernels=<n> blockers=0 fallbacks=0 target=ascend      # P1-1l 三证
DTYPE-TRACE fp16 params=<n> fp16=<n> fp32_accum=true fallback=0        # F2 纪律
```
`KERNEL-USED` / `DTYPE-TRACE` 均**沿用既有实现**（`import ascend.train_step as TS`，一字不改）。

### §5.3 实测值（本会话复跑，2026-10-12）

**host dry（`--target cpu --dry`，EXIT=0，wall=1.23s）**
```
P2-REAL-STEP target=cpu loss0=3.282502 loss1=3.276916 reduced=True params=48 grads_ok=48/48 swaps=4 native=515 mode=run
P2-REAL-PARAMS total=2324768 trainable=48 frozen=2324720 seq=16 batch=1 head=letter dtype=fp32 dry=true
KERNEL-USED-FAIL kernels=33 blockers=1 fallbacks=1 target=cpu why=blockers_present,fallback_events blocker_keys=['gdn_backward[cpu]']
DTYPE-TRACE fp32 params=48 fp16=0 fp32_accum=true fallback=1
P2-REAL-COMPILE count=33 keys=33 fails=0
P2-REAL-MEM-ANALYTIC weights_gb=0.009 opt_gb=0.001 attn_scores_gb=0.000 conv_gb=0.000 total_gb=0.009 lora_numel=49152
P2-REAL-WALL sec=1.23
```
> `KERNEL-USED-FAIL` 在 **cpu 轨是预期**（`gdn_backward[cpu]` 就是 torch 尺子路的 blocker，`gdn_asc.py:724` 的 RuntimeWarning 同源）；**该行不是缺陷**。

**容器真形档（`--target ascend --stub-run --device cpu --seq 512`，EXIT=0）**
```
P2-REAL-TOPO d=1024 L=24 gdn=18 full=6 heads=8 kv=2 hd=256 inter=3584 vocab=248320 conv_k=4 gdn_h=16x128/16x128 qkv_n=4096/512/512 conv_dim=6144 rotary=64 lora_r=16 lora_alpha=32 dry=false
P2-REAL-STEP target=ascend loss0=3.258096 loss1=3.258096 reduced=False params=48 grads_ok=48/48 swaps=4 native=515 mode=stub-run
P2-REAL-PARAMS total=753474368 trainable=48 frozen=753474320 seq=512 batch=1 head=letter dtype=fp16 dry=false
P2-REAL-COMPILE count=41 keys=41 fails=0
KERNEL-USED-PASS kernels=41 blockers=0 fallbacks=0 target=ascend
DTYPE-TRACE fp16 params=48 fp16=48 fp32_accum=true fallback=0
P2-REAL-MEM-ANALYTIC weights_gb=1.403 opt_gb=0.012 attn_scores_gb=0.047 conv_gb=0.012 total_gb=1.462 lora_numel=1081344
P2-REAL-WALL sec=61.84
```
> ⚠️ **`loss0==loss1==3.258096 = ln(26)`** 与 `reduced=False` 是**桩档的预期假值**（发射被桩掉 ⇒
> 前向输出是 `torch.empty` 未初始化 / eager 兜底），**不代表数值失败**，故判决行明标 `mode=stub-run`。
> 同理 `COUNTS <op> p50_ms=` 在桩档下是 **eager CPU 兜底读数（对 NPU 无效）**——
> 例如 `readout p50_ms=247.994`。**R7 的时延列只认卡窗真跑档。**

---

## §6 预算（引 A3 方法）

**A3 方法**：`reconcile/A3_PREFLIGHT.md:158-166` —— d=32 冒烟按"编译键数 × 单键编译耗时 + 整步墙钟"
记账：`A3-STEP wall=20.0s compile_count=13 real_compiles=13 sum_sec=19.1`。

### §6.1 编译预算（真形档，与 seq 近似无关）
| 项 | 实测（seq=512） | 说明 |
|---|---|---|
| 唯一编译键 | **41** | `P2-REAL-COMPILE count=41 keys=41 fails=0` |
| 编译耗时合计 | **51.8 s**（均值 1.26 s/键，n=41） | `awk '/^P2-REAL-COMPILE \? sec=/{…}'` 求和 |
| 整步墙钟 | **61.84 s** | `P2-REAL-WALL sec=61.84`（含 51.8 s 编译） |
| 规模对比 | 41 键 vs 接线档 13 键 ⇒ **×3.2** | 墙钟 61.84 vs 20.0 ⇒ ×3.1 |

键构成（41 枚）：`gemm` 21 · `gemm_dw` 12 · `gdn`2 · `gdnconv`2 · `attn_sw`1 · `rope`1 · `gather`1 · `scatter_add`1
（`grep -c '^COMPILE-KEY' /tmp/i3_ctr_real.log` 得 41）。**大部分键的 `n/k` 是权重宽、`b` 是块宽，均与 seq 无关。**

**⚠️ seq 敏感键（唯一）**：`attn_sw[ascend]|h8d256w<seq>b16x16` —— `w` = window = seq。
seq 8192 ⇒ 该键变 `w8192`，需**新增 1 次编译**；其**编译耗时与 UB 占用未知**，是本预算最大的不确定项。
（其余 40 键在 seq=8192 下**复用**，冷缓存仍要全编 41 枚 ⇒ 预算 ≈ 52-70 s。）

### §6.2 显存预算（解析上界 `_analytic_mem_gb`，A3 方法同口径）
```
weights_gb      = 权重元素数 × 位宽字节 / 2^30                 # fp16=2B、fp32=4B
opt_gb          = lora_numel × 4 × 3 / 2^30                    # master + m + v 三份 fp32
attn_scores_gb  = n_full × heads × seq² × 4 / 2^30             # 反向重算 scores（fp32）
conv_gb         = conv_dim × seq × 4 / 2^30
total_gb        = weights_gb + opt_gb + max(attn_scores_gb, conv_gb)
```

| seq | weights | opt | attn_scores | conv | **total（解析）** |
|---|---|---|---|---|---|
| 512（实测） | 1.403 | 0.012 | 0.047 | 0.012 | **1.462 GB** ✔实测一致 |
| **8192（生产候选）** | 1.403 | 0.012 | **12.0** | 0.188 | **≈13.42 GB** |
| 4096（对折降级） | 1.403 | 0.012 | 3.0 | 0.094 | ≈4.42 GB |
| 16384（候选 C） | 1.403 | 0.012 | 48.0 | 0.375 | ≈49.4 GB ⚠️ |

**口径与上界声明（重要）**：
- `attn_scores` 按 `n_full=6` 层**同时**计（最坏情形），且是 `seq²` 主导 ⇒ **seq 8192 时约 12 GB，
  是本预算的唯一大头**；`seq 16384`（候选 C）解析值 ≈49 GB ⇒ **在 32 GB 卡上必 OOM**，这是
  `gate08b.yaml:58` 那句"要不要把 `max_tokens` 再对折"的定量依据。
- 该式为**解析上界**，**只算三项可闭式复算的大头**；**不含** 24 层激活的累加、`readout`/`gather`
  的中间量、workspace 碎片 ⇒ **真峰值只认卡上 `P2-REAL-MEM alloc_gb/peak_gb`**。
- **不实现 gradient_checkpointing**（`gate08b.yaml:35` 开着）⇒ 本件记的是**不重算**峰值 ⇒ **是上界**。

### §6.3 时延预算
- **本件不产可信时延**：桩档 `p50_ms` 是 eager CPU 兜底读数（§5.3）；host dry 是 fp32 CPU。
- 卡上时延由 `P2-REAL-WALL` + `COUNTS * p50_ms`（真跑档）给。**A3 已给标量件卡上基线可作量级锚**：
  P1-1t 记录 `linear 512³ 294ms`、`attn_sw 1879ms`、`gdn fwd/bwd 293/431ms`（`tasks.md:64`，标量正确性车，
  非 cube 性能路）⇒ 真形档一次前向若全走标量路，量级为**数十秒/步**，即**报价的悲观界**。

---

## §7 窗六一条命令

约定沿用 `window5_ta_card.sh` / `run_window5_local.sh`（`set -u` / `/tmp` 暂存 / 判决行前缀可直接 grep /
`tar` 回传 / 退出码入日志）。**本件不新建 `window6_*.sh`（白名单只 3 个文件）**，故把命令内联如下，
由窗六操作者照着跑或据此 1:1 落成脚本。

### §7.1 首枪（生产口径）
```bash
# ① 宿主机（沿用 run_window5_local.sh 的 ssh 约定：K=KeyPair-19e0.pem / H=ma-user@… / -p 31955）
#    仅上传 release 子树的必要部分（ascend/ + production/ 配置 + config.json）
tar czf /tmp/w6_src.tgz -C /Users/pengweiye/Documents/codes/system-one \
    release/ascend release/production/configs release/bench/ms_models/models/Qwen--Qwen3.5-0.8B
cat /tmp/w6_src.tgz | ssh -T $S $H 'cat > /tmp/w6_src.tgz; echo W6-UP bytes=$(wc -c < /tmp/w6_src.tgz)'

# ② 卡上：一条命令（setsid 后台；整段日志落 /tmp/w6_real.log）
ssh -T $S $H 'rm -rf /tmp/sys1 /tmp/w6_real.log /tmp/w6_cache; mkdir -p /tmp/sys1 && \
  tar xzf /tmp/w6_src.tgz -C /tmp/sys1 && \
  cd /tmp/sys1/release && source /usr/local/Ascend/cann-8.5.2/set_env.sh && \
  export BISHENG_HOME=/usr/local/Ascend/cann-8.5.2/aarch64-linux/ccec_compiler \
         ASCEND_NPU_ARCH=dav-2201 TILELANG_CACHE_DIR=/tmp/w6_cache && \
  (setsid bash -c "timeout 2400 python3 ascend/train_step_real.py \
       --target ascend --device npu --dtype fp16 --seq 8192 --batch 2 \
       > /tmp/w6_real.log 2>&1; echo W6-V1-EXIT=\$? >> /tmp/w6_real.log") \
   >/dev/null 2>&1 < /dev/null &; sleep 25; head -8 /tmp/w6_real.log'

# ③ 回读判决行（窗末）
ssh -T $S $H 'grep -E "P2-REAL-(STEP|PARAMS|COMPILE|MEM|WALL)|KERNEL-USED|DTYPE-TRACE|^SWAP |W6-V1-EXIT" /tmp/w6_real.log'
ssh -T $S $H 'cd /tmp && tar czf /tmp/w6_artifacts.tgz /tmp/w6_real.log /tmp/w6_cache && \
  echo W6-TGZ bytes=$(wc -c < /tmp/w6_artifacts.tgz)'
```
> `--batch 2` 对应 §1.1 **候选 A**；要跑候选 B 只改 `--batch 1`（同命令、第二枪）。

### §7.2 回读判定（窗六"过"的门槛）
- `KERNEL-USED-PASS kernels=41 blockers=0 fallbacks=0 target=ascend`（P1-1l 三证）；
- `DTYPE-TRACE fp16 params=48 fp16=48 fp32_accum=true fallback=0`（F2 纪律）；
- `P2-REAL-STEP … grads_ok=48/48` **且** `loss1 < loss0`（真值，非 stub 的 `ln26` 假值）；
- `P2-REAL-MEM alloc_gb=<a> peak_gb=<p>` **有真读数**（决定 §1.1 候选是否需降级）；
- `W6-V1-EXIT=0`。

---

## §8 降级档（超显存 / 超时 / 编不过）

| 档 | 触发 | 命令改动 | 记账口径变化 | 代价 |
|---|---|---|---|---|
| **D0 首选** | — | `--seq 8192 --batch 2 --dtype fp16` | 全量 | — |
| **D1 seq 对折** | `--seq 8192` OOM（解析 ≈13.4 GB 且真峰值超卡） | `--seq 4096` | **op 次数结构不变**（§2.4 通用）；`attn_scores` 12.0→3.0 GB | 形状降档，时延/显存**不可**直接外推到 8192（`seq²` 项） |
| **D2 只编不跑** | 整步超时（编译 41 键 × w8192 未知键） | `--no-run`（或 `--keys-only`） | 只出 `P2-REAL-COMPILE count/keys/fails` + `KERNEL-USED`；`loss=void` | **不产时延/显存/次数**；只证"可编" |
| **D3 逐层分跑** | D2 也不够（编译超时） | `--layers 6` | 只测**每层每 op 的斜率** | ⚠️ **counts 不可线性外推**：`n=6` ⇒ full 落 idx3 ⇒ 5 GDN:1 full ≠ 真档 18:6；**只能当斜率样本**，次数以全 24 层为准 |
| **D4 位宽降档** | 不属本件（F2 硬纪律：不支持 bf16；`gate08b.yaml:13` bf16 是训练面，本件 fp16 主干 + fp32 归约） | — | — | 需另派单 |

**每档都必须保留**：`KERNEL-USED` 三证 + `DTYPE-TRACE` + `P2-REAL-TOPO`（拓扑自证）。
**每档都必须在日志里明标** `mode=`，**不许把桩档/dry 档的 `p50_ms` 当卡上时延引用**。

---

## §9 自证清单（本件已做 / 待窗六做）

| # | 判据 | 命令 | 结果 |
|---|---|---|---|
| 1 | host 自证轨（拓扑+前向一步+计数行全出、层数/次数结构不变） | `<V>/python release/ascend/train_step_real.py --target cpu --dry` | ✅ **EXIT=0**；`mode=run`；`loss0=3.282502 → loss1=3.276916`；`COUNTS` 与真档逐项同（§2.4） |
| 2 | 既有测试基线不回归 | `cd release && ./.venv/bin/python -m pytest tests -q -m "not integration"` | ✅ **340 passed, 3 skipped, 3 deselected in 41.18s** |
| 3 | 容器证可编（真形状 + 真编译 + 发射桩） | §7.1 ③ 去掉 `--device npu` 改 `--device cpu --stub-run` | ✅ **EXIT=0**；`KERNEL-USED-PASS kernels=41 blockers=0 fallbacks=0`；`fails=0` |
| 4 | 真形档缺陷隔离 | `<V>/python …/I3_probe_gdn_bwd_shape.py cpu` | ✅ `dry gform=2d → OK`；`real gform=2d → AUTOGRAD-RAISE`（§4） |
| 5 | 卡窗一枪（真设备真值） | §7.1 ② | ⏳ **待窗六**（本件不占卡） |

**本件改动面**（白名单内，零既有件改动）：
- 新增 `ascend/train_step_real.py`（判决件，1028 行）
- 新增 `ascend/kernels/reconcile/I3_probe_gdn_bwd_shape.py`（只读探针，160 行）
- 新增 `ascend/kernels/reconcile/I3_REALSHAPE_SPEC.md`（本文件）
- **禁改清单逐项核对：`train_step.py` / `autograd_asc.py` / 各 `*_asc.py` / `card_run.sh` / `window*_card.sh` /
  `gradcheck_card.py` / `run_delta_card.py` / `runs/**` / 全部 `tasks.md` —— 均未改动；全程未用 git。**
