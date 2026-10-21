"""gemm 权重梯度入口：dW = dYᵀ @ A（fp32 累加），内核不可用时回退普通张量写法。

【做什么】对上暴露 `backward(dY, A, *, out_dtype=torch.float16, target=None)`，交回与"该层权重"同
形状的梯度表；函数名与参数默认值都和 sys1/kernels/gemm_bwd_dw_kernel.py 一致，反向链换后端时调用
行不用动。
【怎么做】① 契约校验：两输入都是二维、行数一致（同一批）；② 两级分发：先问 ascend_env（环境 +
   编译缓存），再问 gemm_bwd_dw_asc.plan（形状/位宽可否表达），都过才真跑；③ 内核出口固定 fp32，
   降位在这里做；④ 回退路径同样先转 fp32 再乘，保证两路口径一致。
【为什么】权重梯度是反向里每层必算的一半账，只有前向内核化的话，反向仍走整库张量写法，内核收益被
抵消一半。被否方案：把内核路径也写成 fp16 累加——省一点带宽却引入系统偏差，对拍会出现"说不清是谁
错了"的差异，故累加口径统一 fp32，由测试互锁两条路的等价性。
"""
from __future__ import annotations

import torch

from ascend.kernels import ascend_env, gemm_bwd_dw_asc


def backward(dY: torch.Tensor, A: torch.Tensor, *, out_dtype: torch.dtype = torch.float16,
             target: str | None = None) -> torch.Tensor:
    """权重梯度：返回形状 (N, K) 的 dW = dYᵀ @ A，累加全程 fp32、出口降到 out_dtype。

    白话：把这一批样本每条的头信号竖起来，和当时的输入两两对上记总账，得出"每个连接权该加多少、
    减多少"的一整张表；记账一律用细尺子，最后才按需要的粗细交货。

    :raises ValueError: dY 与 A 不是同步行数的二维数据时抛出。
    """
    if dY.dim() != 2 or A.dim() != 2:
        raise ValueError(f"dY/A 需均为二维 (token, 宽度)，实得 {tuple(dY.shape)} / {tuple(A.shape)}")
    if dY.size(0) != A.size(0):
        raise ValueError(f"两输入的 token 行数须一致（同一批），实得 {dY.size(0)} vs {A.size(0)}")
    name = ascend_env.normalize_target(target)
    to32 = name == ascend_env.TARGET_CPU
    dy = dY.to(torch.float32).contiguous() if to32 else dY.contiguous()
    aa = A.to(torch.float32).contiguous() if to32 else A.contiguous()
    gw = torch.empty((dY.size(1), A.size(1)), dtype=torch.float32, device=dY.device)

    if ascend_env.active_backend(name, dY.device) == ascend_env.TILELANG:
        spec = gemm_bwd_dw_asc.plan(dy, aa, gw, name)
        if spec is not None and gemm_bwd_dw_asc.run(dy, aa, gw, spec):
            return gw.to(out_dtype) if out_dtype != torch.float32 else gw

    return (dy.float().T @ aa.float()).to(out_dtype)


def eager_backward(dY: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """供测试与调试用的"绝对不走内核"版本：fp32 累加、直接交回 (N, K)。

    白话：拿同一本账的手工算法当尺子，专门用来核对内核那条路算得对不对，不参与任何分发。
    """
    return dY.float().T @ A.float()
