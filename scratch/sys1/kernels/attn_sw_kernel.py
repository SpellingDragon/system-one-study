"""attn_sw 对外入口：整理维度 → 分发；方言可用就走滑窗内核，不可用就走普通写法。

【做什么】对上暴露 `forward(q, k, v, window)`，返回与 q 同形的注意力输出；只允许每个位置回看
"自己和前面最多 window 个"，未来位置一律不可见。接受三路 (seq, dim)、(heads, seq, dim)、
(batch, heads, seq, dim) 这几种开本，内部统一摊成 (头总数, seq, dim)。
【怎么做】① 校验：三路同形、最后一维一致、window ≥ 1、设备一致；缩放系数缺省取 1/sqrt(dim)；
② 摊平：把所有前导维当成"头"这一根轴（batch×heads），内核不需要额外一维；
③ 两级分发：先问环境（`backends.active_backend`）、再问形状（`attn_sw_mps.plan`），两级都过
才走方言并直接写出口张量（出口位宽由 `out_dtype` 决定），否则落 `_eager()`；
④ 出口还原成输入的前导维。
【为什么】摊平放在入口而不是内核里：Metal 的网格只有两维，硬编三维会让内核签名里多出一堆与
算法无关的下标运算，学生版换后端时也不好对照。回退写法留在本文件（而不是引用测试目录下的
参考实现）是刻意的：生产代码不该反向依赖 `sys1/testing/`，两边各自成立才能互相校验——对拍
的意义正是"两份独立实现互相印证"。被否方案：让入口接受任意 window 的倍数——内核里键块起点
是按块宽取整的，非整倍数窗口会让"该读的键块"漏读，实测会改变结果；这里不做那种优化，窗口
与块宽的组合交给 plan 的整除性判据把关。
TODO(学生版)：`_cuda` 到位后，本文件只需把 `active_backend` 的判定扩成多后端选择。
"""
from __future__ import annotations

import torch

from sys1.kernels import attn_sw_mps, backends


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
    scale: float | None = None,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """因果单向滑窗注意力：返回与 q 同形、位宽为 out_dtype 的输出（未来位置权重严格为 0）。

    白话：每个位置只能回头看自己前面的有限一段，把看到的内容按分数加权平均；后面的、太远的前面
    的都不参与，也绝不偷偷分走一点权重。

    :raises ValueError: 三路形状不一致、维度过少、窗口非正或设备不匹配时抛出。
    """
    _check_contract(q, k, v, window)
    dim = q.size(-1)
    scale = dim ** -0.5 if scale is None else float(scale)
    lead = q.shape[:-2]
    q3, k3, v3 = (t.reshape(-1, *t.shape[-2:]).contiguous() for t in (q, k, v))

    if backends.active_backend(q.device) == backends.TILELANG:
        out3 = torch.empty(q3.shape, dtype=out_dtype, device=q.device)
        spec = attn_sw_mps.plan(q3, k3, v3, out3, window, scale)
        if spec is not None and attn_sw_mps.run(q3, k3, v3, out3, spec):
            return out3.reshape(lead + (out3.size(1), dim))

    acc = _eager(q3, k3, v3, window, scale)
    return acc.to(out_dtype).reshape(lead + (q3.size(1), dim))


def forward_weights(
    q: torch.Tensor,
    k: torch.Tensor,
    window: int,
    scale: float | None = None,
) -> torch.Tensor:
    """只算分数并做可见性掩码后的权重（fp32），给测试与调试用，不参与内核分发。

    白话：把"每个位置分别看了谁、各看了多少"摊开给人看，用来核对窗口方向有没有抄反。
    """
    _check_contract(q, k, k, window)
    dim = q.size(-1)
    scale = dim ** -0.5 if scale is None else float(scale)
    q3, k3 = q.reshape(-1, *q.shape[-2:]), k.reshape(-1, *k.shape[-2:])
    seq = q3.size(1)
    score = q3.to(torch.float32) @ k3.to(torch.float32).transpose(-1, -2) * scale
    return torch.softmax(_mask(score, seq, window), dim=-1)


def _check_contract(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int) -> None:
    """校验三路同形、至少二维、窗口为正整数，并卡住设备一致。"""
    if q.dim() < 2:
        raise ValueError(f"q 至少要是 (seq, dim) 两维，实得 {q.dim()} 维")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v 形状必须一致，实得 {tuple(q.shape)}/{tuple(k.shape)}/{tuple(v.shape)}")
    if int(window) < 1:
        raise ValueError(f"window 必须是正整数，实得 {window}")
    if k.device != q.device or v.device != q.device:
        raise ValueError("q/k/v 必须在同一设备上")


def _mask(score: torch.Tensor, seq: int, window: int) -> torch.Tensor:
    """把不可见位置压成 -inf：只留 `0 <= i - j < window` 这条**单向**因果带状区。"""
    idx = torch.arange(seq, device=score.device)
    delta = idx[:, None] - idx[None, :]  # delta[i, j] = i - j；j > i 是未来
    allowed = (delta >= 0) & (delta < int(window))
    return score.masked_fill(~allowed, float("-inf"))


def _eager(q3: torch.Tensor, k3: torch.Tensor, v3: torch.Tensor, window: int, scale: float) -> torch.Tensor:
    """回退路径：fp32 全程、分块掩码后做 softmax 加权，口径与方言实现一致（在线算法 vs 整块）。"""
    seq = q3.size(1)
    score = _mask(q3.to(torch.float32) @ k3.to(torch.float32).transpose(-1, -2) * scale, seq, window)
    return torch.softmax(score, dim=-1) @ v3.to(torch.float32)
