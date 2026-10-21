"""add_ln 对外入口：摊平 → 补齐行数 → 分发；残差流必须是 fp32 这条纪律在这里兜住。

【做什么】对上暴露 `forward(x, residual, weight, bias)`，返回 (y, h) 两样东西：y 是整形后
可送进下一个内核的 fp16，h 是"刚加完、还没整形"的 fp32 残差，必须原封不动交给下一层。
【怎么做】① 校验两路输入同形、缩放/偏移长度等于特征轴（**两路的位宽允许不一样**：从第二层起
residual 就是上一层交出的 fp32，不许为迁就内核把它压成 fp16）；② 把前导维摊平成 (rows, dim)，
行数补齐到 ROW_BLOCK 的整数倍（内核"一块 bm 行、一行一线程"，补齐后就不存在边界问题，
补齐行算出的垃圾结果会被切片丢掉，且零填充不会造出 NaN）；③ 分发同 gemm：先问环境再问形状，
两级都过才走方言，否则落 `_eager()`；④ 出口把补齐部分切掉并还原成输入的原始形状。
【为什么】补齐放在入口而不是内核里：内核保持"行列都整块"的最简形态，学生版换后端时只需换
分发这一处。被否方案一：内核里加 `if row < rows` 逐元素守卫——实测这类语句级守卫在本方言上
会出错（结果整体偏离），且生成的 MSL 里谓词不易核对；被否方案二：h 也降成 fp16 省一半带宽——
残差流降位宽会被深层堆叠放大成决策分数换人，是本项目明确禁止的口径。
TODO(学生版)：`_cuda` 到位后，本文件只需把 `active_backend` 的判定扩成多后端选择。
"""
from __future__ import annotations

import torch

from sys1.kernels import add_ln_mps, backends

DEFAULT_EPS = add_ln_mps.DEFAULT_EPS


def forward(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = DEFAULT_EPS,
    out_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (y, h)：h = x + residual 保 fp32（残差流），y = 整形(h)·weight + bias 降到 out_dtype。

    白话：先把两串数字逐位合起来，合出来的这份要用"格子多"的记法留着，后面每层都靠它打底；
    再算出这行的平均水平与波动大小，把每个数挪到"平均为零、波动为一"的位置，最后乘缩放、
    加偏移，这一份才可以换成格子少的记法送下去。

    :raises ValueError: 两路输入形状不一致，或缩放/偏移长度对不上特征轴时抛出。
    """
    _check_contract(x, residual, weight, bias)
    shape = x.shape
    dim = x.size(-1)
    x2 = x.reshape(-1, dim).contiguous()
    r2 = residual.reshape(-1, dim).contiguous()
    rows = x2.size(0)

    weight = weight.to(torch.float32).contiguous()
    bias = bias.to(torch.float32).contiguous()

    bm = add_ln_mps.ROW_BLOCK
    rows_pad = ((rows + bm - 1) // bm) * bm
    if backends.active_backend(x.device) == backends.TILELANG:
        xp, rp = _pad_rows(x2, rows_pad), _pad_rows(r2, rows_pad)
        y_full = torch.empty((rows_pad, dim), dtype=out_dtype, device=x.device)
        h_full = torch.empty((rows_pad, dim), dtype=torch.float32, device=x.device)
        spec = add_ln_mps.plan(xp, rp, weight, bias, y_full, h_full, eps)
        if spec is not None and add_ln_mps.run(xp, rp, weight, bias, y_full, h_full, spec):
            return y_full[:rows].reshape(shape), h_full[:rows].reshape(shape)

    h = x2.to(torch.float32) + r2.to(torch.float32)
    y = _eager(h, weight, bias, eps)
    return y.to(out_dtype).reshape(shape), h.reshape(shape)


def _check_contract(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> None:
    """校验两路输入同形、至少一维、缩放与偏移长度等于特征轴。"""
    if x.shape != residual.shape:
        raise ValueError(f"x 与 residual 形状必须一致，实得 {tuple(x.shape)} / {tuple(residual.shape)}")
    if x.dim() < 1:
        raise ValueError("x 至少得一维（最后一维是特征轴）")
    if weight.device != x.device or bias.device != x.device:
        raise ValueError("weight/bias 必须与 x 同设备")
    d = x.size(-1)
    if weight.numel() != d or bias.numel() != d:
        raise ValueError(f"weight/bias 长度需等于特征轴 {d}，实得 {weight.numel()}/{bias.numel()}")


def _pad_rows(t: torch.Tensor, rows_pad: int) -> torch.Tensor:
    """把行数补到 rows_pad（零填充，位宽随原数据）；已整块则原样返回，不产生多余拷贝。"""
    rows = t.size(0)
    if rows == rows_pad:
        return t
    padded = torch.zeros((rows_pad, t.size(1)), dtype=t.dtype, device=t.device)
    padded[:rows] = t
    return padded


def _eager(h: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """回退路径的层内整形：吃已经加好的 fp32 残差，只做统计与缩放，口径与参考实现一致。"""
    mu = h.mean(dim=-1, keepdim=True)
    centered = h - mu
    var = centered.pow(2).mean(dim=-1, keepdim=True)  # 有偏方差（除以 D）
    return centered / torch.sqrt(var + eps) * weight + bias
