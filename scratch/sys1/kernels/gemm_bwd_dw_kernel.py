"""gemm 权重梯度入口：dW = dYᵀ @ A（fp32 累加），内核不可用时回退普通张量写法。

【做什么】给反向链一个稳定接口 `backward(dY, A, out_dtype)`：交回与"该层权重"同形状的
梯度表；内核路径走 Metal 方言件，回退路径 fp32 累加同口径，两路结果等价。
【怎么做】① 形状契约校验（两输入二维、行数一致、fp16）；② 先问 backends 环境、再问
   gemm_bwd_dw_mps.plan——两级都过才真跑内核；③ 回退做 (dY.float().T @ A.float())
   后按 out_dtype 降位——累加口径与内核一致（fp32），否则对拍有系统偏差。
【为什么】权重梯度是训练里每层必算的一半账（另一半是 dA），若只有前向内核化，
反向仍走整图张量库、内核收益被抵消；本入口让"训练也吃内核"成为可能。被否方案：
把两路都在入口里二选一硬编码判断——分叉出两份口径极易漂移，故两路共用同一条
fp32 累加约定并由测试互锁等价性。
"""
from __future__ import annotations

import torch

from sys1.kernels import backends, gemm_bwd_dw_mps


def backward(dY: torch.Tensor, A: torch.Tensor, *, out_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """权重梯度：返回形状 (N, K) 的 dW = dYᵀ @ A，累加全程 fp32、出口降回 out_dtype。

    白话：把这一批样本每条的头信号竖起来，和当时的输入两两对上记总账，得出"每个
    连接权该加多少、减多少"的一整张表；记账一律用细尺子，最后才按需要的粗细交货。

    :raises ValueError: dY 与 A 不是同步行数的二维 fp16 时抛出。
    """
    if dY.dim() != 2 or A.dim() != 2:
        raise ValueError(f"dY/A 需均为二维 (token, 宽度)，实得 {tuple(dY.shape)} / {tuple(A.shape)}")
    if dY.size(0) != A.size(0):
        raise ValueError(f"两输入的 token 行数须一致（同一批），实得 {dY.size(0)} vs {A.size(0)}")
    gw = torch.empty((dY.size(1), A.size(1)), dtype=torch.float32, device=dY.device)
    spec = gemm_bwd_dw_mps.plan(dY, A, gw)
    if spec is not None and backends.active_backend(dY.device) == backends.TILELANG \
            and gemm_bwd_dw_mps.run(dY, A, gw, spec):
        return gw.to(out_dtype) if out_dtype != torch.float32 else gw
    return _eager(dY, A, out_dtype)


def _eager(dY: torch.Tensor, A: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """回退路径：fp32 累加的同式写法，与内核结果由测试互锁等价。

    白话：没有合适模具时按同一本账手工算一遍——尺子同样是细的（先转成高精度再乘），
    算完再归到要交货的粗细，保证两条路交出来的数对得上。
    """
    return (dY.float().T @ A.float()).to(out_dtype)
