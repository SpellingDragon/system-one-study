"""gemm 的 Metal 方言实现：C = act(A @ Wᵀ + bias)，行数 M 是动态符号、累加器放 threadgroup。

【做什么】把"一批行向量乘一排权重、加偏置、按需过一道截断"写成能在 Apple GPU 上发射的内核，
并且**一次编译服务任意批大小**——A 的行数 M 是编译期动态符号，不是常量。
【怎么做】① 网格按 (M 分块, N 分块) 铺开，块内三块 threadgroup 内存：As(bm×bk, fp16)、
   Ws(bn×bk, fp16)、Cs(bm×bn, **fp32 累加器**)；
② K 轴串行分块：每块先用 T.copy 把 A/W 的切片搬进 threadgroup，再 T.gemm(As, Ws, Cs,
   transpose_B=True) 累加进去（循环前 T.clear(Cs) 清零；T.gemm 的 clear_accum 默认就是 False，
   语义正好是"继续累加"）；
③ 收尾在**共享内存上**做逐元素：加 bias（按列广播）、按需截断为负零，最后 T.copy 回全局，
   出口位宽由编译期参数决定（fp16 或 fp32）。
【为什么】两处形状是被方言实测逼出来的，不是风格选择：
(a) 累加器必须是共享内存而**不能是 fragment**——对 gemm 的 fragment 累加器做逐元素加减会在
    代码生成阶段直接失败，原文报错
    `Check failed: (dtype == Float(16)||Float(32)||BFloat(16)) ... but got float32x4`，
    以及 MSL 侧 `invalid operands to binary expression ('metal::simdgroup_float8x8' and 'float')`；
(b) 发射必须用 execution_backend="tvm_ffi"——默认的 torch adapter 在动态尺寸下把符号化网格当
    python list 递给 C++，抛 `Unable to cast Python instance of type <class 'list'> to C++ type
    'std::vector<unsigned long long>'`。
被否方案一：把 M 做成编译期常量——少写一个符号，但每换一批大小就重编一次，直接违反"一次编译
多批复用"的验收场景；被否方案二：手写 `if by*bm+i < m` 逐元素守卫边界——实测结果错（误差到了
2e+1 量级），而 T.copy 的区域切片会被 tilelang 自动加边界谓词（生成的 MSL 里能看到
`... < arg.m[0]`，哨兵实验证实不越界），故采用切片写法。
"""
from __future__ import annotations

from typing import Any

import tilelang
import tilelang.metal.language as T
import torch

from sys1.kernels import backends

F16, F32 = T.float16, T.float32

#: B1 探针定下的发射参数：Metal 目标 + tvm_ffi 后端（换回默认 torch adapter 会在动态尺寸上崩）
JIT_KWARGS: dict[str, Any] = {"target": "metal", "execution_backend": "tvm_ffi"}

#: 内核侧真正落地的截断口径；其余口径由入口层走普通写法（见 gemm_kernel.forward 的分发）
SUPPORTED_ACTS = ("none", "relu")

#: M 轴的块高固定小值：M 可能等于 1（只要一行），块太大只会浪费并行度
BLOCK_M = 16

#: N/K 轴候选块宽，从大到小取能整除的那个；都不能整除则表达不了，交回入口层回退
BLOCK_LADDER = (64, 32, 16)


def gemm_impl(A, W, bias, Oout, n: int, k: int, bm: int, bn: int, bk: int, act_mode: int, out_fp16: int):
    """方言正文（被 tilelang 追踪，不直接调用）：M 为动态符号，其余维度是编译期常量。

    追踪期约定：n/k/bm/bn/bk/act_mode/out_fp16 必须是 python int，才能在函数体里用普通
    if 选口径、并让形状里的 M 保持符号。act_mode 0=不截断、1=负值压零；out_fp16 决定出口位宽。

    白话：把一大片乘加切成小方块，每个小方块自己算自己的，行数是多少都不影响这套切法；
    算完先在公共小格子里加上加成、压一下负数，再一次性写回大表格。
    """
    m = T.dynamic("m")
    ODT = F16 if out_fp16 else F32

    A: T.Tensor((m, k), F16)
    W: T.Tensor((n, k), F16)
    bias: T.Tensor((n,), F32)
    Oout: T.Tensor((m, n), ODT)

    with T.Kernel(T.ceildiv(m, bm), T.ceildiv(n, bn), threads=128) as (by, bx):
        As = T.alloc_shared((bm, bk), F16)
        Ws = T.alloc_shared((bn, bk), F16)
        Cs = T.alloc_shared((bm, bn), F32)
        T.clear(Cs)
        for ko in T.serial(T.ceildiv(k, bk)):
            T.copy(A[by * bm:(by + 1) * bm, ko * bk:(ko + 1) * bk], As)
            T.copy(W[bx * bn:(bx + 1) * bn, ko * bk:(ko + 1) * bk], Ws)
            T.gemm(As, Ws, Cs, transpose_B=True)
        if act_mode == 1:
            for i, j in T.Parallel(bm, bn):
                Cs[i, j] = T.max(Cs[i, j] + bias[bx * bn + j], T.cast(0, F32))
        else:
            for i, j in T.Parallel(bm, bn):
                Cs[i, j] = Cs[i, j] + bias[bx * bn + j]
        T.copy(Cs, Oout[by * bm:(by + 1) * bm, bx * bn:(bx + 1) * bn])


def _largest_block(total: int) -> int | None:
    """在候选块宽里取能整除 total 的最大值；取不到说明这个维度表达不了（返回 None）。"""
    for blk in BLOCK_LADDER:
        if total % blk == 0:
            return blk
    return None


def _zero_bias(length: int, device: torch.device) -> torch.Tensor:
    """生成长度为 length 的 fp32 零向量，用作"没有加成"的等价物（方言不接受空 operand）。

    白话：内核要求必须递上一列加成数，没有加成时就全填零，效果和不加一模一样。
    """
    return torch.zeros(length, dtype=torch.float32, device=device)


def plan(A: torch.Tensor, W: torch.Tensor, bias: torch.Tensor | None, O: torch.Tensor, act: str) -> dict[str, Any] | None:
    """判断这组形状/口径能不能交给方言内核；能则返回编译参数与缓存键，否则返回 None。

    三条硬条件：输入必须是 fp16（Metal 侧只给 fp16 操作数开了通路）、偏置必须是 fp32 或没有、
    N/K 两个轴都能被某个候选块宽整除。任一不满足就回 None，由入口层换普通写法——**不硬凑**。

    白话：先看这活儿的尺寸合不合工具箱里现成的模具；不合就不勉强开模，直接换手工做法。
    """
    if A.dtype != torch.float16 or W.dtype != torch.float16:
        return None
    if act not in SUPPORTED_ACTS:
        return None
    if bias is not None and bias.dtype != torch.float32:
        return None
    if O.dtype not in (torch.float16, torch.float32):
        return None
    n, k = int(W.size(0)), int(W.size(1))
    bn = _largest_block(n)
    bk = _largest_block(k)
    if bn is None or bk is None:
        return None
    tiles = {
        "n": n,
        "k": k,
        "bm": BLOCK_M,
        "bn": bn,
        "bk": bk,
        "act_mode": 1 if act == "relu" else 0,
        "out_fp16": 1 if O.dtype == torch.float16 else 0,
    }
    key = "gemm|" + "|".join(f"{kk}={vv}" for kk, vv in tiles.items())
    return {"key": key, "tiles": tiles}


def run(A: torch.Tensor, W: torch.Tensor, bias: torch.Tensor | None, O: torch.Tensor, spec: dict[str, Any]) -> bool:
    """按 spec 取（或首次编译）内核句柄，就地写 O；返回 False 表示方言此刻不可用，需要回退。

    缓存键里不含行数——同一份编译产物会被 1、7、16 行等不同批大小反复命中，这正是
    backends.compile_count() 能自证"一次编译多批复用"的地方。

    白话：模具只在第一次真正开；后面不管来多少活儿都拿同一个模具用，开不了模就如实说干不了。
    """
    kernel = backends.get_compiled(spec["key"], lambda: _build(A.device, spec["tiles"]))
    if kernel is None:
        return False
    b = _zero_bias(spec["tiles"]["n"], A.device) if bias is None else bias
    kernel(A, W, b, O, **spec["tiles"])
    return True


def _build(device: torch.device, tiles: dict[str, Any]) -> Any:
    """编译并预热一次：用 1 行哑数据把内核真正编出来，之后不同行数都命中同一份产物。

    预热是必须的——tilelang 的 eager 模式要到第一次调用才编译，若不在此处触发，编译异常就会
    泄漏进业务调用，回退信号也就拿不到了。
    """
    kern = tilelang.jit(**JIT_KWARGS)(gemm_impl)
    n, k = tiles["n"], tiles["k"]
    odtype = torch.float16 if tiles["out_fp16"] else torch.float32
    dummy_a = torch.zeros((1, k), dtype=torch.float16, device=device)
    dummy_w = torch.zeros((n, k), dtype=torch.float16, device=device)
    dummy_b = torch.zeros((n,), dtype=torch.float32, device=device)
    dummy_o = torch.zeros((1, n), dtype=odtype, device=device)
    kern(dummy_a, dummy_w, dummy_b, dummy_o, **tiles)
    torch.mps.synchronize()
    return kern
