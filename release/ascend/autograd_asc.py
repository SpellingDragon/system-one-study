"""ascend 九件方言内核的 **autograd.Function** 可微层（甲路 P2-prep）。

【做什么】给训练步提供 `linear / add_ln / rope / attn_sw / gdn_delta / gdn_conv / readout /
lora_apply` 八个可微门面：前向尽量落到 ascend/kernels 的入口件（按 `target` 分发），反向按
每件"能不能拿到内核反向"分档处理——能拿到就调 `*_kernel.backward(...)`，拿不到就落 fp32
闭式或 fp64 同口径普通写法。任何一档的降级都写在本文件里，不许静默替换。

【怎么做】① 结构对齐 `sys1/kernels/autograd.py`（track-A 已证的可微封装）：每件的 forward
只存最小必要输入（不存中间大表，能重算的重算），backward 按闭式或调内核反向；② 累加口径
统一走 `_work_dtype`——fp64 保留（供 gradcheck 验反向数学），其余升到 fp32；出口再降回输入
位宽，与 P1 参考同口径；③ 契约边界与 track-A 一致：linear 仅 act="none"、rope 接受 packed-
qkv、attn_sw 只服务"无 padding、整窗因果"；带任意掩码的批内补洞场景走 eager，绝不偷换。

【每件反向处置（R14 防假绿）】
- `_LinearFn`: 内核反向齐（dA 走前向件喂转置权重、dW 走 gemm_bwd_dw 件、db 列归约）。
- `_AddLnFn`: 内核反向齐（`add_ln_kernel.backward(h, weight, dy, eps, target)` → (dh, dW, db)）；
  上游若还从 h 直传 dh_next，本件把两路合起来再原样分给 x 与 residual（前向就是它俩相加）。
- `_RopeFn`: 内核反向齐（`rope_kernel.backward` = 负角回转，与 track-A 复用前向件同构）；
  forward 必须先 clone，避免把带梯度的原件就地改脏。
- `_AttnSwFn`: **入口件无反向**（`attn_sw_kernel` 只有 `forward` 与 `forward_weights`）→
  本波走 fp32 闭式重算：softmax 反向 ds=p·(dp−Σdp·p)，三路透传；Metal/昇腾 flash-bwd 留后续。
- `_GdnFn`: 内核反向齐（`gdn_kernel.backward` 在 target=ascend 上已 kernelized，target=cpu 上
  落 torch 尺子并如实登记 `gdn_backward[cpu]` blocker）。本件把 blocker 事实透传给调用方。
- `_GdnConvFn`: **入口件无 conv_backward**（`gdn_kernel.conv_forward` 只有前向）→ 本波走
  fp32 闭式（`torch.nn.functional.conv1d + silu` 现场重算，再借 torch 自动微分回传 dx/dw/db）。
- `_ReadoutFn`: 内核反向齐（`letter_readout_kernel.backward(row_count, ids, dy, target)`
  = index_add 语义）；本件在 fp64 分支改走纯 torch 反向，避免内核 fp32 累加器截位。
- `_LoraFn`: 内核反向齐（`lora_kernel.backward(dy, x, a, b, scaling, target)` → (dx, dA, dB)
  三条链，都复用 linear 两件）。base 若给（本波未启用），梯度直穿。

【为什么】此前 `ascend/` 无任何 `autograd.Function`、`*_kernel` 无模型消费方（R11 消费者枚举
缺口）。要开卡跑训练步，先把"拓扑对不对、反向闭式对不对"这条**接线正确性**在 host cpu 上验
完（本波范围），卡窗只翻 `--target ascend` 就能跑；数值 rel/V* 域不在本波。被否方案一：
不做可微封装、只在评测里用内核——"训练步"永远停在纸面；被否方案二：反向全部手写昇腾件——
LN/flash-bwd/conv-bwd 的归约与原子能力在 910B 上有不确定性（探针未过不押注），先闭式保正确。
"""
from __future__ import annotations

import torch

from ascend.kernels import (
    add_ln_kernel,
    ascend_env,
    attn_sw_kernel,
    gemm_bwd_dw_kernel,
    gemm_kernel,
    gdn_kernel,
    letter_readout_kernel,
    lora_kernel,
    rope_kernel,
)

NEG_INF = -1e30

# ------------------------------------------------------------------ 内核启用自证（P1-1l）
#: 后端回退公告的唯一 token。文案正身在 `ascend_env.record_blocker()`
#: （ascend/kernels/ascend_env.py:234：`方言后端 [<key>] 不可用，已回退 torch eager：…`）；
#: 那是禁改件，所以本件不往里加计数器，只把它当"回落确实发生过"的**可观测信号**来数。
FALLBACK_WARNING_TOKEN = "已回退 torch eager"


def count_fallback_events(messages) -> int:
    """数"内核回落公告"发生了几次（入参是 warnings 钩子收到的原始文案序列）。

    白话：后端每把自己降级成普通 torch 写法都会喊一嗓子，本函数只数这一嗓子，不改账本、
    不猜。口径注意：`record_blocker` 按 key 去重（同一阻塞点只公告一次），所以这里是
    "公告次数"而不是"调用次数"——但只要它 >= 1，"这一趟有没有回落"这件事就已经定了。
    """
    return sum(1 for m in messages if FALLBACK_WARNING_TOKEN in str(m))


def kernel_usage_report(target=None, *, fallback_events=0, compiled=None, blockers=None):
    """纯函数：把"这一步真走了内核"的三路证据合成一份判决（P1-1l，堵最后一道静默假绿）。

    【为什么需要它】探针/发射在卡上真失败时，`ascend_env.get_compiled()` 返回 None，入口层
    就落回 torch 普通写法——**train_step 照样 loss 下降、照样打 P2-STEP-... PASS**。也就是说
    "接线正确"这条判据对"内核有没有被用"完全失明。本函数把三件硬事实收成一个判决：
      ① `compiled_keys()` 里前缀为 `<target>|` 的键 >= 1（只有真编译成功才留痕，回退不留痕）；
      ② `blockers()` 为空（任何一次回退都必须在这本账上留字）；
      ③ `fallback_events == 0`（回退公告的实际发生次数，调用方用 `count_fallback_events` 数出来）。

    【刻意不做的事】绝不调用 `ascend_env.backend_available()` / `active_backend()`：它们会把
    "设备不在场"当场**登记成一条新 blocker**（ascend_env.py:133），判决自己污染账本；也绝不
    自己发 warning 或 `sys.exit`——那都是消费方（train_step.main）的活，本函数只对喂进来的
    账本求值，所以负例可以直接喂假账自证（伪造一次回退 => 必红）。

    :param target: `ascend` / `cpu`（经 `normalize_target` 归一，决定看哪个前缀的键）
    :param fallback_events: 回落公告次数（默认 0；由调用方计数后喂进来）
    :param compiled: 覆写"已编译键"账本；None 时读 `ascend_env.compiled_keys()`
    :param blockers: 覆写"阻塞账"；None 时读 `ascend_env.blockers()`
    :returns: dict(target, prefix, kernels, other_prefix_keys, n_kernels, blockers,
              n_blockers, fallback_events, checks, ok, line)
              其中 `line` 即判决行 `KERNEL-USED-PASS kernels=<n> blockers=<m> fallbacks=<k> target=<t>`
              （不满足时同位置打 `KERNEL-USED-FAIL ... why=<未达项>`）。
    """
    name = ascend_env.normalize_target(target)
    prefix = f"{name}|"
    keys = tuple(ascend_env.compiled_keys() if compiled is None else compiled)
    blk = dict(ascend_env.blockers() if blockers is None else blockers)
    used = sorted(k for k in keys if k.startswith(prefix))
    others = sorted(k for k in keys if not k.startswith(prefix))
    fb = int(fallback_events)

    checks = (
        ("kernels_absent", len(used) >= 1, f"<{name}|> 前缀键 {len(used)} 枚"),
        ("blockers_present", len(blk) == 0, f"阻塞 {len(blk)} 条"),
        ("fallback_events", fb == 0, f"回落公告 {fb} 次"),
    )
    failed = [label for label, ok_item, _ in checks if not ok_item]
    ok = not failed
    line = (f"KERNEL-USED-{'PASS' if ok else 'FAIL'} kernels={len(used)} "
            f"blockers={len(blk)} fallbacks={fb} target={name}")
    if not ok:
        line += f" why={','.join(failed)}"
        if blk:
            line += f" blocker_keys={sorted(blk)[:3]}"
        if others:
            line += f" foreign_keys={others[:3]}"
    return {
        "target": name, "prefix": prefix, "kernels": used, "n_kernels": len(used),
        "other_prefix_keys": others, "blockers": blk, "n_blockers": len(blk),
        "fallback_events": fb, "checks": {label: ok_item for label, ok_item, _ in checks},
        "failed": failed, "ok": ok, "line": line,
    }


def _work_dtype(*ts: torch.Tensor) -> torch.dtype:
    """归约与分数使用的"最小够精度"：fp64→fp64（gradcheck 用）、其余一律 fp32。

    白话：账本一律用细尺子记——粗尺子（fp16）会把两笔挨着的账抹平成同一个数，反向对拍
    就出系统偏差；但如果客户主动交来一把更细的尺子（fp64，专门给数值微分用），就别再折
    回去。规则只有一条：非 fp64 一律按 fp32 起算。
    """
    return torch.float64 if any(t.dtype == torch.float64 for t in ts) else torch.float32


# --------------------------------------------------------------------------- linear
class _LinearFn(torch.autograd.Function):
    """C = A @ Wᵀ + bias（act=none）的可微版：前向 gemm 内核，反向 dA/dW/db 三路闭式。"""

    @staticmethod
    def forward(ctx, A: torch.Tensor, W: torch.Tensor, bias: torch.Tensor | None,
                out_dtype: torch.dtype, target: str | None) -> torch.Tensor:
        """fp16/fp32 走内核；fp64 走同口径普通写法（供 gradcheck 验反向数学）。

        白话：ctx 只留两件原料（A 与 W），大表不留——回头算账时 dW 要 A、dA 要 W，够了。
        """
        ctx.save_for_backward(A, W)
        ctx.has_bias = bias is not None
        ctx.target = target
        ctx.bias_dtype = None if bias is None else bias.dtype   # F2：db 的交货位宽按输入原件
        if A.dtype == torch.float64:
            out = A @ W.t()
            if bias is not None:
                out = out + bias
            return out
        return gemm_kernel.forward(A, W, bias, "none", out_dtype, target)

    @staticmethod
    def backward(ctx, dC: torch.Tensor):
        """dA = dC @ W（喂转置权重的前向件）、dW = dCᵀ @ A（dW 方言件）、db 列归约。"""
        A, W = ctx.saved_tensors
        dC = dC.contiguous()
        if A.dtype == torch.float64:
            dA = dC @ W
            dW = dC.t() @ A
            db = dC.sum(dim=0) if ctx.has_bias else None
            return dA, dW, db, None, None
        dA = gemm_kernel.forward(dC, W.t().contiguous(), None, "none", A.dtype, ctx.target)
        dW = gemm_bwd_dw_kernel.backward(dC, A, out_dtype=W.dtype, target=ctx.target)
        db = dC.float().sum(dim=0) if ctx.has_bias else None
        if db is not None:
            # F2 位宽传导：db 的**归约在 fp32 面做**（上行 .float()），出口按原件位宽交货；
            # bias 本就是 fp32 时 `.to()` 恒等返回 self，fp32 档行为逐字不变。
            db = db.to(ctx.bias_dtype)
        return dA, dW, db, None, None


# --------------------------------------------------------------------------- add_ln
class _AddLnFn(torch.autograd.Function):
    """h = x + residual、y = LN(h)·g + b 融合的可微版：返回 (y, h)，反向调内核。"""

    @staticmethod
    def forward(ctx, x: torch.Tensor, residual: torch.Tensor,
                g: torch.Tensor, b: torch.Tensor, eps: float,
                out_dtype: torch.dtype, target: str | None) -> tuple[torch.Tensor, torch.Tensor]:
        """内核出 (y, h)；ctx 存 x/residual/g/eps，反向前向重算 h（μ、σ̂ 都在 h 上现算）。

        白话：两摞纸合起来留个副本，副本抹匀后乘缩放加偏置出货；账本里只放原件，副本
        回头再合一次，不一直抱着——但比 track-A 多存一份，因为内核反向要 fp32 那份的原件。
        """
        ctx.save_for_backward(x, residual, g, b)
        ctx.eps = eps
        ctx.h_shape = tuple(x.shape)
        ctx.w_dtype = g.dtype
        ctx.target = target
        if x.dtype == torch.float64:
            h = x + residual
            mu = h.mean(-1, keepdim=True)
            hc = h - mu
            var = hc.pow(2).mean(-1, keepdim=True)
            y = hc * torch.rsqrt(var + eps) * g + b
            return y, h
        return add_ln_kernel.forward(x, residual, g, b, eps, out_dtype, target)

    @staticmethod
    def backward(ctx, dy: torch.Tensor, dh_next: torch.Tensor | None):
        """调内核反向得 (dh_ln, dW, db)，再把上游沿 h 直传的 dh_next 合进 dh。

        白话：内核负责"整形那一笔账"（LN 反向），本层负责把上游从残差旁路递下来的
        另一笔账合并；两条源头都吃同一份总账（前向就是它俩相加）。dh_next 若上游没接
        （比如训练只用了 y），按零张处理——这是 PyTorch 未消费输出传 None 的默认口径。
        """
        x, residual, g, b = ctx.saved_tensors
        eps = ctx.eps
        if dh_next is None:
            dh_next = torch.zeros_like(x, dtype=dy.dtype if dy.dtype == torch.float64 else torch.float32)
        if x.dtype == torch.float64:
            h = x + residual
            mu = h.mean(-1, keepdim=True)
            hc = h - mu
            var = hc.pow(2).mean(-1, keepdim=True)
            rstd = torch.rsqrt(var + eps)
            xhat = hc * rstd
            gx = dy * g
            dh_ln = rstd * (gx - gx.mean(-1, keepdim=True) - xhat * (gx * xhat).mean(-1, keepdim=True))
            dh = dh_ln + dh_next
            dg = (dy * xhat).sum(dim=tuple(range(dy.dim() - 1)))
            db = dy.sum(dim=tuple(range(dy.dim() - 1)))
            return dh, dh, dg, db, None, None, None
        dh_ln, dg, db = add_ln_kernel.backward((x.float() + residual.float()), g, dy, eps, ctx.target)
        dh = dh_ln + dh_next.float()
        return dh.to(x.dtype), dh.to(residual.dtype), dg.to(g.dtype), db.to(g.dtype), \
            None, None, None


# --------------------------------------------------------------------------- rope
class _RopeFn(torch.autograd.Function):
    """packed-qkv 就地旋转的可微版：反向 = 负角回转，直接复用内核。"""

    @staticmethod
    def forward(ctx, qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                rotate_slots: tuple[int, ...], target: str | None) -> torch.Tensor:
        """内核就地写，先 clone 再进——不能把带梯度的原件改脏。

        白话：客户交来的原件得原封摆着，本层现场拓一份副本进机器；副本被转得稀碎也不
        回头污染来路，反向算账时原件照旧可用。
        """
        ctx.save_for_backward(cos, sin)
        ctx.rotate_slots = rotate_slots
        ctx.target = target
        if qkv.dtype == torch.float64:
            return _rope_torch(qkv, cos, sin, rotate_slots, sign=1)
        return rope_kernel.forward(qkv.clone(), cos, sin, rotate_slots, target)

    @staticmethod
    def backward(ctx, dqkv: torch.Tensor):
        """旋转是正交变换，倒回去就是走一遍负角；值槽（第三摞）本来就没动，照原路递回。"""
        cos, sin = ctx.saved_tensors
        if dqkv.dtype == torch.float64:
            out = _rope_torch(dqkv, cos, sin, ctx.rotate_slots, sign=-1)
            return out, None, None, None, None
        out = rope_kernel.backward(dqkv.clone().contiguous(), cos, sin, ctx.rotate_slots, ctx.target)
        return out, None, None, None, None


def _rope_torch(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                rotate_slots: tuple[int, ...], sign: int) -> torch.Tensor:
    """rope 的纯 torch 同口径写法（供 fp64 gradcheck 用）：新建输出，不就地。

    白话：内核那路是"就地改副本"，本路是"另外造一张纸抄回去"——语义（把前两摞按角度
    转一转、第三摞原样穿过）完全一致，只是不动输入对象。fp64 数值微分要这个"输入不被
    改脏"的口径。
    """
    out = qkv.clone()
    half = qkv.size(3) // 2
    c = cos.to(qkv.dtype).unsqueeze(1)
    s = (sign * sin).to(qkv.dtype).unsqueeze(1)
    for slot in rotate_slots:
        x1 = qkv[:, slot, :, :half]
        x2 = qkv[:, slot, :, half:]
        out[:, slot, :, :half] = x1 * c - x2 * s
        out[:, slot, :, half:] = x2 * c + x1 * s
    return out


# --------------------------------------------------------------------------- attn_sw
class _AttnSwFn(torch.autograd.Function):
    """因果滑窗注意力的可微版：前向内核，反向 fp32 闭式重算（**入口件无反向**）。

    内核反向处置：`attn_sw_kernel` 只暴露 forward/forward_weights，本波不新造反向件；
    与 track-A 同构走"重算 scores/p → softmax 反向 → 三路透传"的闭式路径。
    """

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int,
                scale: float | None, out_dtype: torch.dtype,
                target: str | None) -> torch.Tensor:
        """内核出份额加权结果；q/k/v 全存，T² 的 p 不存——反向重算，省一大块驻留。"""
        ctx.save_for_backward(q, k, v)
        ctx.window, ctx.scale = int(window), scale
        if q.dtype == torch.float64:
            return _attn_sw_torch(q, k, v, window, scale)
        return attn_sw_kernel.forward(q, k, v, window, scale, out_dtype, target)

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        """标准注意力反向：重算 scores/p，ds = p·(dp − Σ(dp·p))，三路透传（fp64 保 fp64）。"""
        q, k, v = ctx.saved_tensors
        w, seq = ctx.window, q.size(-2)
        scale = ctx.scale if ctx.scale is not None else 1.0 / (q.size(-1) ** 0.5)
        wd = _work_dtype(q, dout)
        q_, k_, v_ = q.to(wd), k.to(wd), v.to(wd)
        scores = (q_ @ k_.transpose(-1, -2)) * scale
        i = torch.arange(seq, device=q.device)
        vis = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < w)
        scores = scores.masked_fill(~vis, NEG_INF)
        shifted = scores - scores.max(-1, keepdim=True).values
        exp = torch.exp(shifted)
        p = exp / exp.sum(-1, keepdim=True).clamp_min(1e-30)
        dp = dout.to(wd) @ v_.transpose(-1, -2)
        ds = p * (dp - (dp * p).sum(-1, keepdim=True))
        dq = (ds @ k_) * scale
        dk = (ds.transpose(-1, -2) @ q_) * scale
        dv = p.transpose(-1, -2) @ dout.to(wd)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None


def _attn_sw_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int,
                   scale: float | None) -> torch.Tensor:
    """attn_sw 的纯 torch 同口径前向（供 fp64 gradcheck 用）：因果滑窗 + 稳定 softmax。"""
    seq = q.size(-2)
    scale = scale if scale is not None else 1.0 / (q.size(-1) ** 0.5)
    scores = (q @ k.transpose(-1, -2)) * scale
    i = torch.arange(seq, device=q.device)
    vis = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < window)
    scores = scores.masked_fill(~vis, NEG_INF)
    return torch.softmax(scores, dim=-1) @ v


# --------------------------------------------------------------------------- GDN delta rule
class _GdnFn(torch.autograd.Function):
    """Gated DeltaRule 递推的可微版：前向/反向都调 `gdn_kernel`（本域最重件）。

    反向状态透传：`gdn_kernel.BWD_STATUS="kernelized"`（ascend target 上六梯全实现）；
    cpu target 落 torch 自动微分尺子并登记 `gdn_backward[cpu]` blocker——本件如实透传，
    不隐藏也不假绿（R14）。
    """

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                g: torch.Tensor, beta: torch.Tensor, out_dtype: torch.dtype,
                target: str | None) -> torch.Tensor:
        ctx.save_for_backward(q, k, v, g, beta)
        ctx.target = target
        if q.dtype == torch.float64:
            return _gdn_torch(q, k, v, g, beta)
        return gdn_kernel.forward(q, k, v, g, beta, out_dtype, target)

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        """调内核反向拿五梯（dq/dk/dv/dg/dβ），六梯及以后本件不接（return_dh0=False）。

        注意口径：Function.backward 默认在 `no_grad` 里跑，而 `gdn_asc.backward` 的 cpu 回退
        路要现场搭一张 torch 图再 `torch.autograd.grad` — 不开 `enable_grad` 会报"element 0
        of tensors does not require grad"。这是本波接线踩到的第一个真环境差异，务必留着。
        """
        q, k, v, g, beta = ctx.saved_tensors
        with torch.enable_grad():
            if q.dtype == torch.float64:
                leaves = [t.detach().clone().to(torch.float64).requires_grad_(True)
                          for t in (q, k, v, g, beta)]
                out = _gdn_torch(*leaves)
                return torch.autograd.grad(out, leaves, dout) + (None, None)
            grads = gdn_kernel.backward(q, k, v, g, beta, dout, ctx.target)
        # F2 位宽传导：gdn 五梯的**内核出口恒 fp32**（C2 §2：归约不降位），交回 autograd 前
        # 按各自输入原件位宽交货。fp32 档 `.to()` 恒等返回 self ⇒ 行为与改前逐字同；
        # fp16 档不依赖引擎的静默折算，交货位宽在件上就钉死。
        dq_, dk_, dv_, dg_, db_ = grads[:5]
        return (dq_.to(q.dtype), dk_.to(k.dtype), dv_.to(v.dtype),
                dg_.to(g.dtype), db_.to(beta.dtype), None, None)


def _gdn_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
               beta: torch.Tensor) -> torch.Tensor:
    """GDN 递推的纯 torch 尺子（与 test_ascend_gradcheck._gdn_torch_ruler 同口径）：fp64 保 fp64。

    白话：五步一循环——黑板按比例淡掉、照键读出预期、真值减预期乘力度补进去、照查询读
    出这一步的输出。这里用 matmul 而不是逐格乘加，故意与内核不同源，专供 gradcheck 用。
    """
    heads, seq, dk = q.shape
    dv = v.size(-1)
    state = torch.zeros(heads, dk, dv, dtype=q.dtype, device=q.device)
    out = torch.empty(heads, seq, dv, dtype=q.dtype, device=q.device)
    for t in range(seq):
        state = state * g[:, t].exp().unsqueeze(-1)
        pred = torch.bmm(state.transpose(1, 2), k[:, t].unsqueeze(-1)).squeeze(-1)
        upd = (v[:, t] - pred) * beta[:, t].unsqueeze(-1)
        state = state + torch.bmm(k[:, t].unsqueeze(-1), upd.unsqueeze(1))
        out[:, t] = torch.bmm(state.transpose(1, 2), q[:, t].unsqueeze(-1)).squeeze(-1)
    return out


# --------------------------------------------------------------------------- GDN short conv
class _GdnConvFn(torch.autograd.Function):
    """GDN 短卷积（4 抽头因果 + SiLU）的可微版：**入口件无 conv_backward** → 本波走 fp32 闭式。

    反向处置：`gdn_kernel.conv_forward` 只有前向，`gdn_conv_asc` 里也没登记 conv_backward
    入口。本波反向借 `torch.nn.functional.conv1d + silu` 现场重算前向（同口径、同位宽），
    再走 torch 自动微分回传 dx/dw/db——这不是"假装内核能反"，是明写的闭式兜底。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None,
                out_dtype: torch.dtype, target: str | None) -> torch.Tensor:
        ctx.save_for_backward(x, w, bias)
        ctx.target = target
        ctx.in_dtype = x.dtype
        if x.dtype == torch.float64:
            return _conv_silu_torch(x, w, bias, x.dtype)
        return gdn_kernel.conv_forward(x, w, bias, out_dtype, target)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        """用同口径 torch 前向 + 自动微分回传 (dx, dw, db, None, None)。"""
        x, w, bias = ctx.saved_tensors
        wd = _work_dtype(x, dy)
        leaves = [x.detach().to(wd).clone().requires_grad_(True),
                  w.detach().to(wd).clone().requires_grad_(True)]
        if bias is not None:
            leaves.append(bias.detach().to(wd).clone().requires_grad_(True))
        # 同上：Function.backward 默认 no_grad；这里要现场搭 torch 图 → 显式开回来
        with torch.enable_grad():
            y = _conv_silu_torch(*leaves[:3] if bias is not None else (*leaves, None), wd)
            grads = torch.autograd.grad(y, leaves, dy.to(wd))
        dx = grads[0].to(ctx.in_dtype)
        dw = grads[1].to(w.dtype)
        db = grads[2].to(bias.dtype) if bias is not None else None
        return dx, dw, db, None, None


def _conv_silu_torch(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None,
                     dtype: torch.dtype) -> torch.Tensor:
    """GDN 短卷积的纯 torch 尺子（与 `gdn_conv_asc._eager` 同口径）：pad→conv1d→silu。"""
    import torch.nn.functional as F

    xs = x.to(dtype)
    ws = w.to(dtype)
    bs = None if bias is None else bias.to(dtype)
    chans = xs.size(-2)
    k = ws.size(-1)
    pad = k - 1
    w4 = ws.reshape(chans, 1, k)
    xs_p = F.pad(xs, (pad, 0))               # 只在左边补零（因果）
    y = F.conv1d(xs_p, w4, bs, groups=chans)  # pad+valid 已把长度带回 L，不再剪
    return F.silu(y)


# --------------------------------------------------------------------------- readout
class _ReadoutFn(torch.autograd.Function):
    """letter 读出（index_select）的可微版：前向/反向都调 `letter_readout_kernel`。

    fp64 分支：入口件反向里的 `acc = torch.zeros(..., dtype=torch.float32)` 会把 fp64
    梯度截位——本件在 fp64 输入下走同口径的 `torch.zeros(x.dtype).index_add_`，避免
    gradcheck 因内核侧的 fp32 累加器而虚警。
    """

    @staticmethod
    def forward(ctx, rows: torch.Tensor, ids: torch.Tensor,
                out_dtype: torch.dtype, target: str | None) -> torch.Tensor:
        ctx.save_for_backward(ids)
        ctx.row_count = int(rows.size(0)) if rows.dim() == 2 else int(rows.reshape(-1, rows.size(-1)).size(0))
        ctx.dim = int(rows.size(-1))
        ctx.in_dtype = rows.dtype
        ctx.target = target
        if rows.dtype == torch.float64:
            # 内核路 r32 = r2.float() 会把 fp64 截位到 fp32（gradcheck 的 eps=1e-6 立刻被抹平）；
            # 本件 fp64 分支走同口径的 index_select，只为反向公式验数，不参与生产。
            return rows.index_select(0, ids.to(torch.long)).to(out_dtype)
        return letter_readout_kernel.forward(rows, ids, out_dtype, target)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        """反向 = 按同一张行号表把梯度加回原表；fp64 分支跳过内核（避开 fp32 累加器截位）。"""
        ids, = ctx.saved_tensors
        if ctx.in_dtype == torch.float64:
            acc = torch.zeros(ctx.row_count, ctx.dim, dtype=torch.float64, device=dy.device)
            acc.index_add_(0, ids, dy.to(torch.float64))
            return acc, None, None, None
        drows = letter_readout_kernel.backward(ctx.row_count, ids, dy, ctx.target)
        return drows.to(ctx.in_dtype), None, None, None


# --------------------------------------------------------------------------- lora
class _LoraFn(torch.autograd.Function):
    """LoRA 旁路 `delta = scaling·(x@Aᵀ)@Bᵀ`（可带 base）的可微版：前向/反向都调内核。

    base 本波只支持 None（train_step 用不到），若给 base 则梯度直穿（`dbase = dy`）；
    与 `lora_asc.apply` 的口径一致（base 与 delta 相加，不参与低秩链）。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                scaling: float, base: torch.Tensor | None,
                out_dtype: torch.dtype, target: str | None) -> torch.Tensor:
        ctx.save_for_backward(x, a, b)
        ctx.scaling = float(scaling)
        ctx.has_base = base is not None
        ctx.target = target
        ctx.in_dtype = x.dtype
        if x.dtype == torch.float64:
            delta = (x @ a.t()) @ (b * scaling).t()
            return delta if base is None else base + delta
        out = lora_kernel.apply(x, a, b, scaling, base, out_dtype, target)
        return out

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        """反向走 `lora_kernel.backward`（三条链 dx/dA/dB）；base 分支梯度直穿。"""
        x, a, b = ctx.saved_tensors
        s = ctx.scaling
        if x.dtype == torch.float64:
            with torch.enable_grad():
                leaves = [x.detach().clone().requires_grad_(True),
                          a.detach().clone().requires_grad_(True),
                          b.detach().clone().requires_grad_(True)]
                xl, al, bl = leaves
                delta = (xl @ al.t()) @ (bl * s).t()
                dx, da, db = torch.autograd.grad(delta, leaves, dy)
            dbase = dy if ctx.has_base else None
            return dx, da, db, None, dbase, None, None
        dx, da, db = lora_kernel.backward(dy, x, a, b, s, ctx.target)
        dbase = dy.to(ctx.in_dtype) if ctx.has_base else None
        return dx.to(ctx.in_dtype), da.to(a.dtype), db.to(b.dtype), None, dbase, None, None


# --------------------------------------------------------------------------- 函数式门面
def linear(A: torch.Tensor, W: torch.Tensor, bias: torch.Tensor | None = None,
           *, out_dtype: torch.dtype = torch.float32,
           target: str | None = None) -> torch.Tensor:
    """可微线性层（act=none 口径）：见 `_LinearFn`。"""
    return _LinearFn.apply(A, W, bias, out_dtype, target)


def add_ln(x: torch.Tensor, residual: torch.Tensor, g: torch.Tensor, b: torch.Tensor,
           *, eps: float = 1e-5, out_dtype: torch.dtype = torch.float32,
           target: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """可微 LN+残差融合：返回 (y, h)；见 `_AddLnFn`。"""
    return _AddLnFn.apply(x, residual, g, b, eps, out_dtype, target)


def rope(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
         *, rotate_slots: tuple[int, ...] = (0, 1),
         target: str | None = None) -> torch.Tensor:
    """可微 packed-rope：见 `_RopeFn`（反向负角回转，复用内核）。"""
    return _RopeFn.apply(qkv, cos, sin, rotate_slots, target)


def attn_sw(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int,
            *, scale: float | None = None, out_dtype: torch.dtype = torch.float32,
            target: str | None = None) -> torch.Tensor:
    """可微因果滑窗注意力：见 `_AttnSwFn`（**入口件无反向 → fp32 闭式**；带掩码走 eager）。"""
    return _AttnSwFn.apply(q, k, v, window, scale, out_dtype, target)


def gdn_delta(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
              beta: torch.Tensor, *, out_dtype: torch.dtype = torch.float32,
              target: str | None = None) -> torch.Tensor:
    """可微 GDN 递推：见 `_GdnFn`（cpu target 上反向落尺子、blocker 如实登记）。"""
    return _GdnFn.apply(q, k, v, g, beta, out_dtype, target)


def gdn_conv(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None,
             *, out_dtype: torch.dtype = torch.float32,
             target: str | None = None) -> torch.Tensor:
    """可微 GDN 短卷积：见 `_GdnConvFn`（**入口件无 conv_backward → fp32 闭式兜底**）。"""
    return _GdnConvFn.apply(x, w, bias, out_dtype, target)


def readout(rows: torch.Tensor, ids: torch.Tensor, *, out_dtype: torch.dtype = torch.float32,
            target: str | None = None) -> torch.Tensor:
    """可微 letter 读出：见 `_ReadoutFn`。"""
    return _ReadoutFn.apply(rows, ids, out_dtype, target)


def lora_apply(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scaling: float = 1.0,
               base: torch.Tensor | None = None, *, out_dtype: torch.dtype = torch.float32,
               target: str | None = None) -> torch.Tensor:
    """可微 LoRA 旁路：见 `_LoraFn`。"""
    return _LoraFn.apply(x, a, b, scaling, base, out_dtype, target)


__all__ = (
    "add_ln", "attn_sw", "count_fallback_events", "gdn_conv", "gdn_delta", "kernel_usage_report",
    "linear", "lora_apply", "readout", "rope",
    "_AddLnFn", "_AttnSwFn", "_GdnConvFn", "_GdnFn", "_LinearFn", "_LoraFn", "_ReadoutFn", "_RopeFn",
)
