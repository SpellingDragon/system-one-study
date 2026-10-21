"""gemm 对外入口：校验契约 → 分发；方言可用就走内核，不可用就用 fp32 累加的普通写法。

【做什么】对上暴露 `forward(A, W, bias=None, act="relu", out_dtype=torch.float16, target=None)`，
交回 (M, N) 的结果；函数名、参数顺序与默认值都和 sys1/kernels/gemm_kernel.py 一致，训练侧把
import 路径从 `sys1.kernels` 换成 `ascend.kernels` 就不用改任何调用行。
【怎么做】① 契约校验：A 是 (M,K)、W 是 (N,K)、两者 K 相等、偏置长度等于 N、三者在同一设备；
② 两级分发：先问 ascend_env（环境与编译缓存），再问 gemm_asc.plan（形状与位宽能不能表达），
   两级都过才真跑内核；③ 内核全程 fp32 累加、出口 fp32，位宽转换在这里做（`.to(out_dtype)`）；
   CPU 侧内核只吃 fp32，所以入口先把输入升上来；昇腾侧按白名单直通原宽；④ 任一级不过就落
   `_eager()`：先转 fp32 乘加、按需压负为零，最后降到要交的位宽。
【为什么】把"位宽转换"和"能不能走内核"两件事都收在入口层：方言侧不收这个口径（昇腾 DMA 不许顺带
转类型，CPU 侧跨位宽整块搬运会撞生成代码的转换错误），而回退路径必须与内核同口径（都 fp32 累加），
否则对拍会出现系统偏差。被否方案一：入口直接吃 fp16 并要求内核降位——把方言缺陷写进契约；被否
方案二：给 target 做成必填参数——会破坏"调用行零改动"的迁移前提，故做成带默认值的关键字参数，
默认读环境变量 DMLAYA_ASCEND_TARGET。
"""
from __future__ import annotations

import torch

from ascend.kernels import ascend_env, gemm_asc

#: 与 P1 同名同取值的截断口径表（入口按它决定能不能走内核）
SUPPORTED_ACTS = gemm_asc.SUPPORTED_ACTS


def forward(A: torch.Tensor, W: torch.Tensor, bias: torch.Tensor | None = None,
            act: str = "relu", out_dtype: torch.dtype = torch.float16,
            target: str | None = None) -> torch.Tensor:
    """线性层前向：C = act(A @ Wᵀ + bias)，累加全程 fp32，出口降到 out_dtype。

    白话：把一批行向量挨个和一排权重对上做乘加，需要的时候再加一道偏置、把负数压成零；
    记账一律用细尺子，最后才按需要的粗细交货。

    :param target: "ascend" / "cpu"；缺省读环境变量（默认 cpu），传值不影响既有调用行。
    :raises ValueError: A/W 维度或 K 轴对不上、偏置长度不对、三者不同设备时抛出。
    """
    m, n, _k = _check_contract(A, W, bias)
    name = ascend_env.normalize_target(target)
    to32 = name == ascend_env.TARGET_CPU
    a = A.to(torch.float32).contiguous() if to32 else A.contiguous()
    w = W.to(torch.float32).contiguous() if to32 else W.contiguous()
    b = torch.zeros(n, dtype=torch.float32, device=A.device) if bias is None else bias.to(torch.float32).contiguous()
    c = torch.empty((m, n), dtype=a.dtype, device=A.device)

    if act in SUPPORTED_ACTS and ascend_env.active_backend(name, A.device) == ascend_env.TILELANG:
        spec = gemm_asc.plan(a, w, b, c, act, name)
        if spec is not None and gemm_asc.run(a, w, b, c, spec):
            return c.to(out_dtype) if out_dtype != a.dtype else c

    return _eager(a, w, b, act, out_dtype)


def _check_contract(A: torch.Tensor, W: torch.Tensor, bias: torch.Tensor | None) -> tuple[int, int, int]:
    """校验两输入都是二维、K 轴一致、偏置长度等于 N，并卡住设备一致，返回 (M, N, K)。"""
    if A.dim() != 2 or W.dim() != 2:
        raise ValueError(f"A 需 (M,K)、W 需 (N,K) 两维，实得 {tuple(A.shape)} / {tuple(W.shape)}")
    if A.size(1) != W.size(1):
        raise ValueError(f"A 与 W 的 K 轴须一致，实得 {A.size(1)} vs {W.size(1)}")
    if bias is not None and bias.numel() != W.size(0):
        raise ValueError(f"bias 长度需等于 N={W.size(0)}，实得 {bias.numel()}")
    if bias is not None and bias.device != A.device:
        raise ValueError("bias 必须与 A 同设备")
    if W.device != A.device:
        raise ValueError("A 与 W 必须在同一设备上")
    return int(A.size(0)), int(W.size(0)), int(A.size(1))


def _eager(A: torch.Tensor, W: torch.Tensor, bias: torch.Tensor, act: str, out_dtype: torch.dtype) -> torch.Tensor:
    """回退路径：与内核同口径（fp32 累加、按需截断），算完再降到要交的位宽。

    白话：没有合适模具时按同一本账手工算一遍——尺子同样是细的，算完再归到要交货的粗细，
    保证两条路交出来的数对得上。
    """
    if act not in SUPPORTED_ACTS:
        raise ValueError(f"未知 act={act!r}，只支持 {'/'.join(SUPPORTED_ACTS)}")
    acc = (A @ W.T).float() + bias.float()
    if act == "relu":
        acc = acc.clamp_min(0.0)
    return acc.to(out_dtype)
