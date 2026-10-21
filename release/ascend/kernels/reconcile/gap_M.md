# gap_M：甲路 P2-prep（autograd_asc + train_step）遇上的入口件接口缺口

上报人：p2-13-ascend-runtime P2-prep 执行代理 M
消费方：`release/ascend/autograd_asc.py`（8 件 autograd.Function 门面）+ `release/ascend/train_step.py`
本波边界：**不改任何 `*_kernel.py`/`*_asc.py`/`ascend_env.py`**——所有缺口都在**门面侧绕行**，
并把绕行代价如实登记，供后续波次（P2 卡窗 / P3 内核化补齐）评估是否入口件层需要新增接口。

## G-1 `attn_sw_kernel` 无 backward 入口

- **现象**：`release/ascend/kernels/attn_sw_kernel.py` 只 re-export 了 `forward` 与
  `forward_weights`（`__all__` 无 `backward`）；`attn_sw_asc.py` 内也没登记反向实现。
- **本波处置**：`_AttnSwFn.backward` 走 **fp32 闭式重算**（标准 softmax-反：重算 scores/p，
  `ds = p·(dp − Σdp·p)`，dq/dk/dv 三路透传），与 `sys1/kernels/autograd.py::_AttnSwFn`
  同构。已在 `autograd_asc.py` 类 docstring 明写"**入口件无反向** → 本波走 fp32 闭式"。
- **代价**：反向多一次前向乘加（T² scores/p 现场重算），比内核化的 attn-bwd 慢；显存换算力。
- **建议**：P3 若在 910B 上验通 flash-bwd 的归约与原子能力，`attn_sw_asc.py` 补 `backward`
  入口 + `attn_sw_kernel.py` re-export，本波 `_AttnSwFn.backward` 换一行调用即切内核。

## G-2 `gdn_kernel` 无 `conv_backward` 入口

- **现象**：`gdn_kernel.py` 只 re-export `conv_forward / conv_plan / conv_run`；`gdn_conv_asc.py`
  有 `_eager` 回退但没有独立 `conv_backward` 函数（`__all__` 无）。
- **本波处置**：`_GdnConvFn.backward` 用 `torch.nn.functional.conv1d + silu` 现场重算前向，
  再走 torch 自动微分回传 (dx, dw, db)。已在类 docstring 明写"**入口件无 conv_backward**
  → 本波走 fp32 闭式"。
- **代价**：与 G-1 同类——反向是"另一份实现"，虽然 fp32 口径与 `_eager` 完全一致
  （pad→conv1d→silu 三步同一份代码），但内核侧一旦 conv_bwd 到位需要改门面。
- **建议**：若 conv 是训练时反向链上的热点，`gdn_conv_asc.py` 补一份 `conv_backward`
  （depthwise 4 抽头 + SiLU 反向，闭式），本波 `_GdnConvFn` 换调用即可。

## G-3 `gdn_asc.backward` 的 cpu 回退路依赖 caller 已开 grad

- **现象**：`release/ascend/kernels/gdn_asc.py:728-732` 的回退分支：
  ```python
  leaves = [t.detach().clone().float().requires_grad_(True) for t in (q3, k3, v3, g3, b3)]
  out = _eager(*leaves)
  grads = torch.autograd.grad(out, leaves, dy3.float(), retain_graph=False)
  ```
  `torch.autograd.Function.backward` **默认在 `no_grad` 上下文里跑**（PyTorch 设计如此，
  高阶微分需 caller 显式 `enable_grad`），因此从 `_GdnFn.backward` 直接调用会炸：
  `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`。
- **本波处置**：`_GdnFn.backward` 用 `with torch.enable_grad():` 把内核 backward 包住。
  已在类里加了行内说明，标明"这是本波接线踩到的第一个真环境差异"。
- **代价**：门面侧多一次 `enable_grad()` 上下文；无功能影响。
- **建议（可选）**：`gdn_asc.backward` 里也自包 `enable_grad`，让"被 Function.backward 调用"
  这个新使用场景不依赖 caller 侧——但改动落在内核层，本波不动，交由后续 P2/P3 决定是否
  收紧。**当前不视为硬缺口，只是使用口径要写清**。

## G-4 三件入口件在 fp64 输入下会截位到 fp32（不影响生产，只影响 gradcheck）

具体三处（都写死 `.float()`）：

- `letter_readout_asc.py:232` — 内核路 `r32 = r2.float().contiguous()`；
  反向 `letter_readout_asc.py:260` 里 `acc = torch.zeros((row_count, dim), dtype=torch.float32, ...)`。
- `add_ln_asc.py:413` — 回退路 `h = (x.float() + residual.float()).to(torch.float32)`；
  `backward` 的 `h.float()` / `weight.float()` / `dy.float()` 也一律升到 fp32。
- `attn_sw_asc.py`（前向内核路同样 fp32 累加，未逐行摘录）。
- `gdn_asc.py:728` 回退路的 `t.detach().clone().float()`。

**现象**：`torch.autograd.gradcheck` 用 fp64 + eps=1e-6 做数值微分；上面这些 `.float()`
把扰动直接抹平，jacobian 对不上。

**本波处置**：
- `_ReadoutFn.forward / backward` 在 `rows.dtype == torch.float64` 时**跳过内核**，走
  `index_select` / `index_add_` 的 fp64 纯 torch 分支。
- `_AddLnFn / _AttnSwFn / _GdnConvFn / _GdnFn / _LinearFn / _RopeFn / _LoraFn` 各自加了
  fp64 分支的同口径 torch 前向（如 `_rope_torch / _attn_sw_torch / _gdn_torch /
  _conv_silu_torch`），保证"入口件内核路只吃 fp32/fp16"的契约不变，同时 fp64 gradcheck
  能对反向公式独立校验。
- 生产路径（target ∈ {cpu, ascend}, dtype ∈ {fp16, fp32}）**完全不受影响**，仍走内核。

**代价**：`autograd_asc.py` 多 4 个小 helper 函数（`_rope_torch / _attn_sw_torch /
_gdn_torch / _conv_silu_torch`）；测试覆盖更宽。

**建议**：入口件层**不必**支持 fp64（生产不需要）；把 fp64 只当"门面侧的反向公式验数通道"
是最经济的分工。这条不作为待办缺口，只是本波做了绕行。

## G-5 类数量：本波 8 件而非派单文案里的 7 件

- **现象**：派单正文列了 7 个 Function 类（`_LinearFn/_AddLnFn/_RopeFn/_AttnSwFn/_GdnFn/
  _ReadoutFn/_LoraFn`）与 8 个函数式门面（多出 `gdn_conv`）。`gdn_conv` 门面若不建独立
  Function，就得把 conv 塞进 `_GdnFn` 里做模式分支——语义上两条不同的算式（递推 vs 短卷
  积）混在一个 Function 更不好读。
- **本波处置**：加了第 8 件 `_GdnConvFn`。锚点 `grep -c "torch.autograd.Function"` 在
  `autograd_asc.py` 上应 = 8。
- **代价**：与派单文案的一行差异，无功能问题。

## 汇总（每件反向处置表）

| 门面 | 前向入口件 | 反向处置 |
|---|---|---|
| `linear` → `_LinearFn` | `gemm_kernel.forward` | **内核闭式**：`gemm_kernel.forward`（喂转置权重）+ `gemm_bwd_dw_kernel.backward` |
| `add_ln` → `_AddLnFn` | `add_ln_kernel.forward` | **内核闭式**：`add_ln_kernel.backward`；fp64 分支同口径 torch |
| `rope` → `_RopeFn` | `rope_kernel.forward`（clone 后进入，就地改） | **内核闭式**：`rope_kernel.backward`（负角回转） |
| `attn_sw` → `_AttnSwFn` | `attn_sw_kernel.forward` | **fp32 闭式回退**：G-1，softmax-反 |
| `gdn_delta` → `_GdnFn` | `gdn_kernel.forward` | **内核反向**（ascend 已 kernelized；cpu 走 torch 尺子 + `enable_grad()`） |
| `gdn_conv` → `_GdnConvFn` | `gdn_kernel.conv_forward` | **fp32 闭式回退**：G-2，torch `conv1d+silu` 现场重算 |
| `readout` → `_ReadoutFn` | `letter_readout_kernel.forward` | **内核反向**：`letter_readout_kernel.backward`；fp64 分支 index_add_ |
| `lora_apply` → `_LoraFn` | `lora_kernel.apply` | **内核反向**：`lora_kernel.backward` 三条链（gemm 两件组合） |

**"None 占位"情况**：所有 Function 都对非张量参数（`out_dtype`、`target`、`scale`、
`eps`、`window`、`rotate_slots`、`scaling`）与不消费张量的辅助输入（`ids`、`return_dh0`
等）返回 `None`；训练里所有 leaf 参数都拿到了非 None 的梯度（`train_step` 的 missing
检查在测试 `test_grad_none_would_fail_train_run` 里做过反向自证）。
