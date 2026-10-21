"""gemm 参考实现：C = act(A @ Wᵀ + b)，累加一律走 fp32。

【做什么】算"一批输入乘一排权重再加偏置、最后过一道激活"的常用小算式，给出**没有手写内核时
的公认正确答案**，用来检验 TileLang 内核算得对不对。
【怎么做】三步：① 把 A 与 W 先升到 fp32 再相乘（W 按行存成 (N, K)，乘的时候转置成 (K, N)）；
② 加 fp32 偏置；③ 过激活（none / relu / gelu / silu），最后才降到目标位宽（默认 fp16）落回内存。
关键点是"中间过程一律 fp32、只在出口降位宽"，内核必须复现同一条纪律，否则误差会随层数放大。
【为什么】把累加放在 fp32 是因为 fp16 只有约 10 位尾数，长 K 轴上连加会把小项直接吃掉
（实测 K=512 时 fp16 直累可达 1e-1 级偏差，远超 2e-2 验收线）。被否方案一：全程 fp16 算——
最快，但对拍时分不清"内核错了"还是"精度地板就这样"，等于把生命线交出去；被否方案二：全程 fp64——
最准，但 TileLang 后端与 MPS 都不支持 fp64 通路，参考值反而对不上真实硬件行为。本函数只依赖
torch，CPU/GPU 通吃，与 p1-04 的四算子契约一一对应（内核侧仅支持 none/relu，见 gemm_mps.py）。
"""
from __future__ import annotations

import torch

#: 支持的激活口径；内核侧目前只落地 none 与 relu（其余留 TODO，走回退路径时口径仍正确）
ACTIVATIONS = ("none", "relu", "gelu", "silu")


def activate_ref(x: torch.Tensor, act: str = "relu") -> torch.Tensor:
    """按名字做激活，输入输出同位宽（这里始终吃 fp32）。

    gelu 用 tanh 近似式而不是精确误差函数：与主流框架默认口径一致，内核侧将来要对齐的是
    这个近似式，而不是数学上的 erf 版本。

    白话：把每个数按名字过一道"拐弯"——不改就原样送出，负数压成零，或者按一条平滑的斜坡
    把负的一点点压下去；这里刻意用一条能用初等函数算出来的近似斜坡，好让两边算法对得上。
    """
    if act == "none":
        return x
    if act == "relu":
        return torch.relu(x)
    if act == "gelu":
        return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x.pow(3))))
    if act == "silu":
        return x * torch.sigmoid(x)
    raise ValueError(f"未知激活 {act!r}，可选：{ACTIVATIONS}")


def gemm_ref(
    A: torch.Tensor,
    W: torch.Tensor,
    bias: torch.Tensor | None = None,
    act: str = "relu",
    out_dtype: torch.dtype | None = torch.float16,
) -> torch.Tensor:
    """C = act(A @ Wᵀ + bias)，中间量 fp32，出口降到 out_dtype（None 表示保 fp32）。

    形状约定与内核完全一致：A 是 (M, K) 的一批行向量，W 是 (N, K) 的 N 组权重行
    （所以是"乘 W 的转置"，而不是 (K, N)），bias 是 (N,)。M 可以是任意正整数——
    这条参考实现不做任何长度对齐假设，动态 M 的可变性由内核负责复现。

    白话：一批东西各带一串数字，另一批"模板"也各带同样长的串；每样东西跟每个模板逐项
    相乘再全加起来，得到一个分数，加上该模板自带的加成，再过一道拐弯，就是答案表。
    """
    if A.dim() != 2 or W.dim() != 2:
        raise ValueError(f"A 需为 (M, K) 二维、W 需为 (N, K) 二维，实得 {tuple(A.shape)} / {tuple(W.shape)}")
    if A.size(1) != W.size(1):
        raise ValueError(f"K 轴不匹配：A 的 K={A.size(1)}，W 的 K={W.size(1)}")
    if bias is not None and (bias.numel() != W.size(0)):
        raise ValueError(f"bias 长度需等于 N={W.size(0)}，实得 {bias.numel()}")

    acc = A.to(torch.float32) @ W.to(torch.float32).transpose(0, 1)
    if bias is not None:
        acc = acc + bias.to(torch.float32).unsqueeze(0)
    acc = activate_ref(acc, act)
    return acc if out_dtype is None else acc.to(out_dtype)
