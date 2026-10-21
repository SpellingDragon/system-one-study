"""rope 对外入口：校验打包形状 → 分发；方言可用就就地旋转，不可用就普通写法原地旋转。

【做什么】对上暴露 `forward(qkv, cos, sin)`，把 (tokens, 3, heads, dim) 的打包数据里前两摞按
角度表旋转，**原地改写并返回同一个对象**（与 `refs/laya` 的用法一致，省一份临时大内存）。
【怎么做】① 校验：四维且第二维是 3、dim 为偶数、两张角度表是 (tokens, dim//2) 且为 fp32
（不是 fp32 就地升上来，保证乘加在 fp32 域）；② 只有"要旋转的槽位正好是前两摞"才可能走方言
（内核写死 0/1 两个槽位），其余槽位组合直接落 `_eager()`；③ 分发同 gemm：先问环境再问形状。
【为什么】把"槽位子集"这种花哨用法挡在回退路径，是为了让内核保持最简形态；一阶段决策模型
只会用默认口径，先不让少见口径污染主路。被否方案一：把槽位编号也做成编译期参数——每个不同
组合都要重开一份产物，缓存键会爆炸；被否方案二：入口一律先 clone 再旋转——语义与参考实现
完全一致、最省心，但每层都要多一份 (tokens,3,heads,dim) 的临时内存，长上下文下这份内存不便宜。
就地语义还有一条隐性前提：数据必须是连续存放的。非连续时内核拿到的是另开内存的副本，转完
原对象不会变——所以 `rope_mps.plan` 里对连续性一票否决，直接落回原地改写的普通写法。
TODO(学生版)：`_cuda` 的就地语义与本文件一致，分发处多问一次后端即可。
"""
from __future__ import annotations

import torch

from sys1.kernels import backends, rope_mps

DEFAULT_SLOTS = rope_mps.ROTATE_SLOTS


def forward(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotate_slots: tuple[int, ...] = DEFAULT_SLOTS,
) -> torch.Tensor:
    """按角度表旋转 packed-qkv 的指定槽位，**原地**改写后返回同一个对象（值槽永不参与）。

    白话：三摞材料放在同一个盒子里，第一摞和第二摞按小抄转一转、原地放回去，第三摞原封不动；
    转的时候每摞内部前后两半互相搭一手。

    :raises ValueError: 打包布局不是 (tokens,3,heads,dim)、每头宽度不是偶数，或角度表长度对不上。
    """
    tokens, dim = _check_contract(qkv, cos, sin)
    half = dim // 2
    cos = cos.to(torch.float32).contiguous()
    sin = sin.to(torch.float32).contiguous()

    if tuple(rotate_slots) == DEFAULT_SLOTS and backends.active_backend(qkv.device) == backends.TILELANG:
        spec = rope_mps.plan(qkv, cos, sin)
        if spec is not None and rope_mps.run(qkv, cos, sin, spec):
            return qkv

    snapshot = qkv.clone()
    c = cos.reshape(tokens, 1, half)
    s = sin.reshape(tokens, 1, half)
    for slot in rotate_slots:
        part = snapshot[:, slot].to(torch.float32)
        x1, x2 = part[..., :half], part[..., half:]
        qkv[:, slot, :, :half] = (x1 * c - x2 * s).to(qkv.dtype)
        qkv[:, slot, :, half:] = (x2 * c + x1 * s).to(qkv.dtype)
    return qkv


def _check_contract(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[int, int]:
    """校验打包布局与角度表形状，返回 (token 数, 每头宽度)；顺带卡住设备一致性。"""
    if qkv.dim() != 4 or qkv.size(1) != 3:
        raise ValueError(f"qkv 需为 (tokens, 3, heads, dim)，实得 {tuple(qkv.shape)}")
    tokens, _, _, dim = (int(v) for v in qkv.shape)
    if dim % 2:
        raise ValueError(f"dim 必须为偶数才能对半旋转，实得 {dim}")
    if cos.shape != (tokens, dim // 2) or sin.shape != (tokens, dim // 2):
        raise ValueError(
            f"cos/sin 需各为 (tokens={tokens}, dim//2={dim // 2})，实得 {tuple(cos.shape)}/{tuple(sin.shape)}"
        )
    if cos.device != qkv.device or sin.device != qkv.device:
        raise ValueError("cos/sin 必须与 qkv 同设备")
    return tokens, dim
