"""TileLang 内核的可微封装：把四算子包进自动求导图，让训练前向也吃内核、反向有账可查。

【做什么】给模型训练链提供 `linear / add_ln / rope / attn_sw` 四个可微函数：前向尽量落
到 Metal 内核，反向按数学闭式回传每一路梯度；任一算子内核不可用则该算子回退普通写法，
整条图不断。
【怎么做】① 每个算子一个 autograd.Function：forward 记输入（不记中间大表，能重算的
   重算省内存）；backward 按闭式回梯度——linear 的 dA 复用前向 gemm 模具（喂权重转置
   的连续拷贝）、dW 走 gemm_bwd_dw 方言件、bias 梯度列归约；rope 的反向 = 负角再转
   一次（旋转矩阵正交，R(θ)ᵀ=R(−θ)，直接复用前向内核）；add_ln/attn_sw 的反向用 fp32
   闭式重算（LN 三趟归约 / softmax 的 ds=p·(dp−Σdp·p)），Metal 版留待后续，本文件如实标注。
② 所有归约与分数一律 fp32 累加，出口按输入位宽交货——与 p1-04 各参考同口径，
   否则与纯 torch 路径的梯度对拍会有系统偏差。
【为什么】若只有推理前向内核化，训练时反向仍拖整张 eager 图，前向内核对冲为零甚至为负；
补上反向才有"训练也受内核加速"的可能。被否方案一：反向全部手写 Metal 件——LN 与 flash 式
attn 反向在本方言上有归约/原子能力的不确定性（探针未过之前不押注），先闭式保正确、
数据说话后再逐个内核化；被否方案二：不做可微封装、只在评测里用内核——用户要求
训练与推理同路，且那会让"加速"永远停在纸面。

契约边界（保守回退，不静默改语义）：
- linear 仅 act="none"（relu/gelu 融合的反向暂不承接）；
- rope 接受 (tokens,3,heads,dim) 打包 fp16；非打包形状由调用方预处理；
- attn_sw 仅服务"无 padding、整窗因果"（window ≥ seq 时等价全因果）；带 attn_mask 的
  批内补洞场景走 eager——窗口语义与任意掩码语义不同，绝不偷换。
"""
from __future__ import annotations

import torch

from sys1.kernels import (
    add_ln_kernel,
    attn_sw_kernel,
    gemm_bwd_dw_kernel,
    gemm_kernel,
    rope_kernel,
)

NEG_INF = -1e30


def kernels_ready(device: torch.device | str) -> bool:
    """这台设备此刻是否值得走内核路径（探测口径由 backends 统一把脉，此处不另造判据）。

    白话：动工前先去工牌处问一句——机器合不合、模具开没开出来；问不到就不硬上。
    """
    from sys1.kernels import backends
    return backends.active_backend(device) == backends.TILELANG


class _LinearFn(torch.autograd.Function):
    """C = A @ Wᵀ + bias 的可微版：前向 gemm 内核，反向 dA/dW/db 三路闭式。"""

    @staticmethod
    def forward(ctx, A: torch.Tensor, W: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        """fp16 两输入过内核；ctx 只存 A/W（dW 要原件，dA 要权重，不留中间大表）。

        白话：把一批行数字按权重的每一排对一遍，得一批新数交出去；顺手把两件原料
        留在小本上，回头算账要用它们，但中间的大结果不留，省地方。
        """
        ctx.save_for_backward(A, W)
        ctx.has_bias = bias is not None
        return gemm_kernel.forward(A, W, bias, act="none")

    @staticmethod
    def backward(ctx, dC: torch.Tensor):
        """dA=dC@W（前向模具喂转置权重）、dW=dCᵀ@A（方言件）、db=列归约，全程 fp32 累加。

        白话：往回传时算三笔账——每个输入行该改多少、每张连接表该改多少、每列加成
        该改多少；前两笔都是乘加法，只是"竖着乘"和"横着乘"的分别，最后一笔按列求和。
        """
        A, W = ctx.saved_tensors
        dC = dC.contiguous()
        dA = gemm_kernel.forward(dC, W.t().contiguous(), None, act="none").to(A.dtype)
        dW = gemm_bwd_dw_kernel.backward(dC, A, out_dtype=W.dtype)
        db = dC.float().sum(dim=0) if ctx.has_bias else None
        return dA, dW, db


class _RopeFn(torch.autograd.Function):
    """packed-qkv 就地旋转的可微版：反向=负角再转一次，直接复用前向内核。"""

    @staticmethod
    def forward(ctx, qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """内核就地写，先 clone 再进，避免把带梯度的原件改脏。"""
        ctx.save_for_backward(cos, sin)
        return rope_kernel.forward(qkv.clone(), cos, sin)

    @staticmethod
    def backward(ctx, dqkv: torch.Tensor):
        """旋转是正交变换，转回去=反向再转一次负角；值槽的梯度原样穿过（前向没动它）。

        白话：前向把前两摞按位置转了个角，回向只要往回转回来——把转角的正负调反
        再走一遍同一台机器就行，第三摞本来就没动，照原路递回去。
        """
        cos, sin = ctx.saved_tensors
        out = rope_kernel.forward(dqkv.clone().contiguous(), cos, -sin)
        return out, None, None


class _AddLnFn(torch.autograd.Function):
    """h=x+residual、y=LN(h)·g+b 融合的可微版：返回 (y, h)，反向闭式 fp32。"""

    @staticmethod
    def forward(ctx, x: torch.Tensor, residual: torch.Tensor,
                g: torch.Tensor, b: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        """内核出 (y, h)；ctx 存 x/residual/g，μ、σ̂ 反向重算（省一次大表驻留）。

        白话：先把两串数合起来，合出来的这份留着记账用；再把它抹匀成“均值零、起伏一”
        的样子并乘缩放加偏置。账本里只放原件，合出来的大数回头再合一次，不是一直抱着。
        """
        ctx.save_for_backward(x, residual, g)
        ctx.eps = eps
        return add_ln_kernel.forward(x, residual, g, b, eps=eps)

    @staticmethod
    def backward(ctx, dy: torch.Tensor, dh_next: torch.Tensor):
        """LN 反向闭式（fp32）+ 残差直通：主干梯度 = LN 路径贡献 + 上游直传。

        白话：先按"整形前那串合起来的数"重算平均和起伏，把 y 的来账摊回到整形前的
        坐标上（三项相减是归一化的标准回账法）；这条回账和上游顺残差直传的账合并，
        同样发给两路源头——因为前向就是它俩相加；缩放量与偏置各按列收一笔总账。
        """
        x, residual, g = ctx.saved_tensors
        eps = ctx.eps
        h = (x.float() + residual.float())
        mu = h.mean(dim=-1, keepdim=True)
        hc = h - mu
        var = hc.pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        xhat = hc * rstd
        dyf = dy.float()
        gx = dyf * g.float()
        d_h_ln = rstd * (gx - gx.mean(-1, keepdim=True)
                         - xhat * (gx * xhat).mean(-1, keepdim=True))
        d_h = d_h_ln + dh_next.float()
        dg = (dyf * xhat).sum(dim=tuple(range(dy.dim() - 1)))
        db = dyf.sum(dim=tuple(range(dy.dim() - 1)))
        return d_h.to(x.dtype), d_h.to(residual.dtype), dg.to(g.dtype), db.to(g.dtype), None


class _AttnSwFn(torch.autograd.Function):
    """因果滑窗注意力的可微版：前向内核，反向 fp32 闭式重算（Metal flash-bwd 留后续）。"""

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int) -> torch.Tensor:
        """内核出份额加权结果；q/k/v 全存（T² 的 p 不存，反向重算——省显存换一次前向乘加）。

        白话：每个位置只朝后看一段，把看到的按相关性分份额再合拢；分份额那张大表不留着，
        回头算账时再照原样重算一遍——拿一次重算换一大块空地，堆得深也不挤爆。
        """
        ctx.save_for_backward(q, k, v)
        ctx.window = int(window)
        return attn_sw_kernel.forward(q, k, v, window=window)

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        """标准注意力反向：重算 scores/p，ds=p·(dp−Σ(dp·p))，三路透传（fp32 全程）。

        白话：回账要先复算一遍当初的"相关性与份额"，再按份额怎么被加权汇总的倒推：
        每个值向量的来账摊回份额表，份额表的差先减去均值校正再摊回两边坐标；
        看不见的位（未来、窗外）从头到尾份额为零，回账自然也是零。
        """
        q, k, v = ctx.saved_tensors
        w, seq = ctx.window, q.size(-2)
        scale = 1.0 / (q.size(-1) ** 0.5)
        q32, k32, v32 = q.float(), k.float(), v.float()
        scores = (q32 @ k32.transpose(-1, -2)) * scale
        i = torch.arange(seq, device=q.device)
        vis = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < w)
        scores = scores.masked_fill(~vis, NEG_INF)
        shifted = scores - scores.max(-1, keepdim=True).values
        exp = torch.exp(shifted)
        p = exp / exp.sum(-1, keepdim=True).clamp_min(1e-30)
        dout32 = dout.float()
        dp = dout32 @ v32.transpose(-1, -2)
        ds = p * (dp - (dp * p).sum(-1, keepdim=True))
        dq = (ds @ k32) * scale
        dk = (ds.transpose(-1, -2) @ q32) * scale
        dv = p.transpose(-1, -2) @ dout32
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None


def linear(A, W, bias=None):
    """可微线性层（act=none 口径）：前向 gemm 内核、反向三路闭式——见 _LinearFn。"""
    return _LinearFn.apply(A, W, bias)


def rope(qkv, cos, sin):
    """可微 packed-rope：见 _RopeFn（反向复用前向内核、负角回转）。"""
    return _RopeFn.apply(qkv, cos, sin)


def add_ln(x, residual, g, b, eps=1e-5):
    """可微 LN+残差融合：返回 (y, h)，见 _AddLnFn。"""
    return _AddLnFn.apply(x, residual, g, b, eps)


def attn_sw(q, k, v, window):
    """可微因果滑窗注意力：见 _AttnSwFn（带任意掩码的场景请勿走此路，调用方回退 eager）。"""
    return _AttnSwFn.apply(q, k, v, window)
