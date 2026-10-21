"""add_ln 参考实现：残差相加 + 层内统计缩放，**残差流全程 fp32**。

【做什么】把"这一层的输出"和"进来的那条老路"逐元素加在一起（这条老路就叫残差），再对这个
和做"每行减均值除标准差、乘系数加偏移"的整形，同时把**加完但还没整形的 fp32 结果原样交出去**，
供下一层继续当残差用。
【怎么做】① h = x.to(fp32) + residual.to(fp32)（相加在 fp32 里做，h 直接以 fp32 返回）；
② 对 h 的每一行求均值 mu、方差 var（有偏方差，除 D 而不是 D-1，与主流实现一致）；
③ y = (h - mu) / sqrt(var + eps) * weight + bias，最后才把 y 降到 out_dtype（默认 fp16）。
被整形送出去的 y 可以是 fp16，但**留在残差流上的 h 必须是 fp32**——这是本域的硬约束。
【为什么】残差降位宽会被实测抓到：bf16 残差在深层堆叠后出现可见漂移（同权重、同输入，逐层
误差被放大到决策分数换人），而 fp32 残差的代价只是多一倍存储带宽，一阶段模型尺寸完全吃得下。
被否方案一：只返回整形后的 y、把残差丢弃——省一份输出，但下一层就得重算 h，重算时若用 y
的 fp16 版本，等于把误差塞回残差流；被否方案二：y 也保 fp32——数值更稳，但下游内核（gemm/attn
的输入 operand）只吃 fp16，多一次转换还平白让对拍多一个变量。
"""
from __future__ import annotations

import torch

DEFAULT_EPS = 1e-5


def add_ln_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = DEFAULT_EPS,
    out_dtype: torch.dtype | None = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (y, h)：h = x + residual 保 fp32（残差流），y = LN(h)·weight + bias 降到 out_dtype。

    形状：x / residual 同为 (..., D)（最后一维做统计），weight / bias 同为 (D,)。
    多维输入会被摊平成 (rows, D) 处理，输出形状与输入一致。

    白话：先把两串数字逐位合起来，合出来的这份一定要用"格子多"的记法留着，后面每一层都
    靠它打底；再算出这一行的平均水平和水波大小，用它们把每个数挪到"平均为零、波动为一"
    的位置，最后乘上可学的缩放、加上可学的偏移，这份才可以换成格子少的记法送下去。
    """
    if x.shape != residual.shape:
        raise ValueError(f"x 与 residual 形状必须一致，实得 {tuple(x.shape)} / {tuple(residual.shape)}")
    if x.dim() < 1:
        raise ValueError("x 至少得一维（最后一维是特征轴）")
    d = x.size(-1)
    if weight.numel() != d or bias.numel() != d:
        raise ValueError(f"weight/bias 长度需等于特征轴 {d}，实得 {weight.numel()}/{bias.numel()}")

    h = x.to(torch.float32) + residual.to(torch.float32)  # 残差流：全程 fp32，不做任何降位宽
    mu = h.mean(dim=-1, keepdim=True)
    centered = h - mu
    var = centered.pow(2).mean(dim=-1, keepdim=True)  # 有偏方差（除以 D），与 PyTorch LayerNorm 一致
    y = centered / torch.sqrt(var + eps) * weight.to(torch.float32) + bias.to(torch.float32)
    return (y if out_dtype is None else y.to(out_dtype)), h


def layer_stats_ref(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """单独取一行的均值与有偏方差，供测试直接盯统计口径（对拍时定位是整形错还是加法错）。

    白话：只报两个数——这一行的平均水平，以及它绕着平均水平波动多大，别的都不算。
    """
    h = x.to(torch.float32)
    mu = h.mean(dim=-1, keepdim=True)
    return mu, (h - mu).pow(2).mean(dim=-1, keepdim=True)
