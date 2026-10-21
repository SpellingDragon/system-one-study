"""gemm 对外入口：校验 → 分配输出 → 分发（方言优先，表达不了或方言不可用就走普通写法）。

【做什么】给上层一个稳定的 `forward(A, W, bias=..., act=...)` 接口，内部决定用 TileLang Metal
内核还是 torch 写法，两条路算出来的东西必须等价。
【怎么做】① 契约校验：A 是 (M,K)、W 是 (N,K) 且 K 对齐、bias 长度是 N；② 输出先按 out_dtype
分配好——内核一律就地写、不返回新对象，避免"到底谁持有输出"含混；③ 分发分两级：先问
backends.active_backend(设备) 判环境，再让 gemm_mps.plan() 判这组形状能不能用现有模具表达，
两级都通过才真正 run；任何一级不成立就落回本模块的 `_eager()`（累加同样走 fp32）。
【为什么】把"能不能用方言"的两级判定集中在入口，算子语义与后端选择互不污染。被否方案一：
在入口里直接 try 编译——环境判定散进四个算子，回退告警会重复弹、阻塞点也无处登记；被否方案二：
让 `_mps` 模块自己回退——那文件就该只有方言，混进 torch 写法后学生版补 `_cuda` 时无法机械对照。
TODO(学生版)：新增 `_cuda.py` 时只需在分发处多问一次后端偏好，本文件其余逻辑不变。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from sys1.kernels import backends, gemm_mps

#: 内核侧已落地的截断口径（其余口径只走回退路径，见 gemm_mps.SUPPORTED_ACTS）
KERNEL_ACTS = gemm_mps.SUPPORTED_ACTS


def forward(
    A: torch.Tensor,
    W: torch.Tensor,
    bias: torch.Tensor | None = None,
    act: str = "relu",
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """返回 C = act(A @ Wᵀ + bias)，形状 (M, N)，dtype 为 out_dtype；两条实现路径结果等价。

    白话：不管底下用机器模具还是手工算，交给你的都是同一张结果表：每行输入对每套模板打一个分，
    加上该模板的加成，需要时把负分压成零。

    :raises ValueError: 形状或长度不满足 (M,K)·(N,K)ᵀ 契约时抛出（不许静默出错形状）。
    """
    _check_contract(A, W, bias)
    A = A.contiguous()
    W = W.contiguous()
    if bias is not None:
        bias = bias.to(torch.float32).contiguous()

    O = torch.empty((A.size(0), W.size(0)), dtype=out_dtype, device=A.device)
    if act in KERNEL_ACTS and backends.active_backend(A.device) == backends.TILELANG:
        spec = gemm_mps.plan(A, W, bias, O, act)
        if spec is not None and gemm_mps.run(A, W, bias, O, spec):
            return O
    _eager(A, W, bias, O, act)
    return O


def _check_contract(A: torch.Tensor, W: torch.Tensor, bias: torch.Tensor | None) -> None:
    """校验二维形状、K 轴对齐、bias 长度等于 N，以及设备一致。"""
    if A.dim() != 2 or W.dim() != 2:
        raise ValueError(f"A 需为 (M, K) 二维、W 需为 (N, K) 二维，实得 {tuple(A.shape)} / {tuple(W.shape)}")
    if A.size(1) != W.size(1):
        raise ValueError(f"K 轴不匹配：A 的 K={A.size(1)}，W 的 K={W.size(1)}")
    if bias is not None and bias.numel() != W.size(0):
        raise ValueError(f"bias 长度需等于 N={W.size(0)}，实得 {bias.numel()}")
    if bias is not None and bias.device != A.device:
        raise ValueError(f"bias 与 A 必须同设备，实得 {bias.device} / {A.device}")


def _eager(A: torch.Tensor, W: torch.Tensor, bias: torch.Tensor | None, O: torch.Tensor, act: str) -> None:
    """回退路径：纯 torch 写法，就地写 O。累加仍在 fp32，与参考实现同一条精度纪律。

    gelu 用 tanh 近似式（与 torch_ref.activate_ref 逐字同式），silu 直接取官方实现。
    """
    acc = A.to(torch.float32) @ W.to(torch.float32).transpose(0, 1)
    if bias is not None:
        acc = acc + bias.to(torch.float32)
    if act == "relu":
        acc = acc.clamp(min=0)
    elif act == "gelu":
        acc = F.gelu(acc, approximate="tanh")
    elif act == "silu":
        acc = F.silu(acc)
    elif act != "none":
        raise ValueError(f"未知激活 {act!r}，可选：none / relu / gelu / silu")
    O.copy_(acc.to(O.dtype))
