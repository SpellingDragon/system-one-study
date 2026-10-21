"""attn_sw 参考实现：**因果**滑窗——每个位置只看自己与往前 W 个，未来位置权重恒为 0。

【做什么】算"每个词只能向回看不超过 W 个词"的加权汇总：每个词给能看见的那几个词打一组
相关性分数，归一成分额（加起来等于一），再用份额把第三摞数据加权合起来。本模块还把那张
份额矩阵单独交出去，专门用来验收"看不见的地方必须严格等于零"。
【怎么做】① 分数 = 查询·键 / sqrt(dim)（点积一律 fp32 累加）；② 掩码 = causal_window_mask，
   第 i 行只保留 j ∈ (i-W, i] 的键，其余（未来的、以及窗口外更早的）整片置为 −inf；
③ 沿键轴取份额：先减每行最大值再取指数、按行和归一（减最大值是为了不溢出）；
④ 输出 = 份额 · 值（同样 fp32 累加），最后降位宽。
   ⚠️ 参考蓝本 `refs/laya/tl_kernels.py::attn_kernel` 是**双向**滑窗（前后各看 W/2），
   这里刻意改成单向：`0 <= i - j < W`。方向抄反会让模型能偷看后文，训练指标虚高、上线崩。
【为什么】单列份额矩阵接口（attn_sw_weights_ref）是为了把"未来权重=0"从隐含约定变成可断言
的事实：份额一旦在 j>i 处非零，测试立刻红，不用等到下游精度崩了再回溯。被否方案一：只用
torch.nn.functional.scaled_dot_product_attention——一行搞定且高度优化，但它不返回份额矩阵、
也没有滑窗口径，等于把验收对象藏进黑盒；被否方案二：把 −inf 换成一个大负数（如 -1e4）——
在 fp16 里更不容易出 NaN，但归一后仍留 1e-9 级残差，"恒等于零"就断言不动了。
"""
from __future__ import annotations

import math

import torch

NEG_INF = float("-inf")


def causal_window_mask(seq: int, window: int, device: torch.device | str | None = None) -> torch.Tensor:
    """生成 (seq, seq) 的布尔掩码：第 i 行第 j 列为真 ⟺ 0 <= i - j < window（只回看，不看未来）。

    window 传 None 或 >= seq 时退化为"纯因果"（回看长度不限），但仍绝不看未来。

    白话：画一张方格，横竖都是句子里的位置；只有"在我自己或我前面、而且没超过 W 格"的那些
    格子被点亮，后面（还没发生的）一律不亮——这一点是硬规矩，不是可调选项。
    """
    if seq <= 0:
        raise ValueError(f"seq 必须为正，实得 {seq}")
    idx = torch.arange(seq, device=device)
    delta = idx.unsqueeze(1) - idx.unsqueeze(0)  # i - j：正数代表 j 在 i 之前
    if window is None:
        return delta >= 0
    if window <= 0:
        raise ValueError(f"window 必须为正整数，实得 {window}")
    return (delta >= 0) & (delta < window)


def _softmax_rows_ref(scores: torch.Tensor) -> torch.Tensor:
    """沿最后一维（键轴）做份额分配：先减每行最大值再取指数、除以行和。

    全行都被屏蔽时（极端边界：某行一个可见键都没有）指数项全为零，行和加一个极小量兜底，
    结果为均匀零行而不是 NaN——因果口径下不会出现这种情况，这里只是防御性处理。

    白话：每行先找出最大的那个分数当基准，各分数减去它再变成"越高分越陡"的正数，然后按
    该行加起来等于一来分配——就像把一锅汤按各人的份量倒给对方。
    """
    shifted = scores - scores.max(dim=-1, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=-1, keepdim=True).clamp_min(1e-30)


def attn_sw_weights_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    window: int | None = 64,
    scale: float | None = None,
) -> torch.Tensor:
    """返回份额矩阵 (batch, heads, seq_q, seq_kv)，fp32；用来断言"未来与窗口外的份额==0"。

    形状：q 为 (B, H, Sq, D)，k 为 (B, H, Skv, D)。要求 Sq <= Skv 且位置对齐方式为右对齐
    （第 i 个查询对应第 Skv-Sq+i 个键，与解码时"新 token 接在已有上下文之后"的口径一致）。

    白话：把每个位置向哪些位置"分了多少注意力"整张表都摊开给你看，好让你数哪些格子必须是零。
    """
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError(f"q/k 需为 (B, H, S, D) 四维，实得 {tuple(q.shape)}/{tuple(k.shape)}")
    if q.size(1) != k.size(1) or q.size(3) != k.size(3):
        raise ValueError(f"头数与 head_dim 必须一致，实得 {q.size(1)}/{k.size(1)} 与 {q.size(3)}/{k.size(3)}")
    sq, skv = q.size(2), k.size(2)
    if sq > skv:
        raise ValueError(f"右对齐口径要求 Sq({sq}) <= Skv({skv})")

    dim = q.size(3)
    factor = scale if scale is not None else 1.0 / math.sqrt(dim)
    scores = (q.to(torch.float32) @ k.to(torch.float32).transpose(-1, -2)) * factor

    offset = skv - sq  # 右对齐偏移：查询 i 的真实位置是 i + offset
    pos_q = torch.arange(offset, skv, device=q.device).unsqueeze(1)
    pos_k = torch.arange(skv, device=q.device).unsqueeze(0)
    delta = pos_q - pos_k
    allowed = delta >= 0 if window is None else (delta >= 0) & (delta < window)
    scores = scores.masked_fill(~allowed.unsqueeze(0).unsqueeze(0), NEG_INF)
    return _softmax_rows_ref(scores)


def attn_sw_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int | None = 64,
    scale: float | None = None,
    out_dtype: torch.dtype | None = torch.float16,
) -> torch.Tensor:
    """因果滑窗汇总：份额 · 值，返回 (B, H, Sq, D)；份额由 attn_sw_weights_ref 单独可查。

    与内核的对拍口径：内核一次吃整段 Sq==Skv==seq（右 padding 到阶梯长度后调用），
    本函数在这种正方形情形退化为纯 (i-j) 口径，两者逐元素可比。

    白话：先按上一条的规则算出每个位置该向前面哪些位置分多少，再用这些份额把第三摞数字
    加权合起来——看得见的位置才参与合并，看不见的一律不掺进来。
    """
    if v.shape != k.shape:
        raise ValueError(f"v 与 k 形状必须一致，实得 {tuple(v.shape)}/{tuple(k.shape)}")
    if q.size(3) != k.size(3):
        raise ValueError(f"q 与 k 的 head_dim 必须一致，实得 {q.size(3)}/{k.size(3)}")
    weights = attn_sw_weights_ref(q, k, window=window, scale=scale)
    out = weights @ v.to(torch.float32)
    return out if out_dtype is None else out.to(out_dtype)
