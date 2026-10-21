"""add_ln 的 Metal 方言实现：残差相加 + 行内整形，**残差流以 fp32 原样落回内存**。

【做什么】一个内核同时干完两件事：把本层输出与进来的老路逐元素相加（这条老路就是残差），再对
相加结果做"减均值、除标准差、乘缩放、加偏移"的整形；**加完的 fp32 结果必须单独交出去**，
供下一层继续当残差打底。进来的老路本身就是上一层交出的 fp32，所以两路输入的位宽允许不一样。
【怎么做】① 网格按行数分块（行数是动态符号），每块 bm 行；② 每行交给**一个线程**从头到尾串行
扫一遍特征轴，三趟扫描：第一趟求和（同时把 fp32 的和写进残差出口）、第二趟用已知均值再扫一遍
求"偏差平方和"、第三趟算出缩放值并把整形结果写回；③ 每行的均值/方差/倒数标准差用 T.alloc_var
的标量变量存放，块内不需要任何跨线程合并；④ 除的标准差用 T.rsqrt(方差 + eps)；⑤ 老路那一摞的
位宽是编译期开关 res_fp32：它是 fp32 时读回来就已在高精度域，不需要额外转换。
【为什么】三趟扫描而不是"一趟同时攒和与平方和"，是为了数值口径与参考实现逐字一致：一趟攒
`E[x²]-E[x]²` 在均值很大、波动很小的行上会灾难性抵消（实测能把方差算出负数），而行内整形一旦
方差走负，rsqrt 立刻变 NaN，整层崩掉；多扫两趟只花全局读的钱，而这个内核本来就受限于"一行
一个线程"，代价可接受。被否方案一：用 T.reduce_sum/T.warp_reduce_*——Metal 后端**没有注册
归约实现**，编译期直接报 `tl.reduce requires a target-specific implementation, but no reduce
implementation is registered for {"kind":"metal"}`，只能手写线程级串行归约；被否方案二：把
每行放进 threadgroup 再合并——dim 上到 512 时 bm 行的共享内存要 128KB，远超 Apple GPU 每块
32KB 的上限。残差出口坚持 fp32：bf16/fp16 残差在深层堆叠后会把误差推进决策分数（实测换人）。
"""
from __future__ import annotations

from typing import Any

import tilelang
import tilelang.metal.language as T
import torch

from sys1.kernels import backends

F16, F32 = T.float16, T.float32

#: 与 gemm 同一套发射参数（Metal 目标 + tvm_ffi 后端），原因见 gemm_mps 模块头
JIT_KWARGS: dict[str, Any] = {"target": "metal", "execution_backend": "tvm_ffi"}

#: 每块行数 = 每块参与归约的线程数（一行一线程），块内并行度上限 128 线程
ROW_BLOCK = 16

#: 默认容差，与 torch_ref.DEFAULT_EPS 同值；两边都要改必须同时改，否则对拍会白抖
DEFAULT_EPS = 1e-5


def add_ln_impl(X, Res, G, Bt, Y, Hout, dim: int, eps: float, bm: int, res_fp32: int, out_fp16: int):
    """方言正文（被 tilelang 追踪，不直接调用）：行数为动态符号，特征轴 dim 为编译期常量。

    追踪期约定：eps 是 python float（会进编译缓存键），dim/bm/res_fp32/out_fp16 是 python int，
    这样才能在追踪阶段定下两路输入各自的位宽。

    白话：一行数据交给一个人从头看到尾，看三遍——第一遍把数加起来并顺手把原始和记下来，
    第二遍看每人和平均差了多少、把差平方加起来，第三遍才按算好的尺度把每格改到位。
    """
    rows = T.dynamic("rows")
    ODT = F16 if out_fp16 else F32
    RDT = F32 if res_fp32 else F16

    X: T.Tensor((rows, dim), F16)
    Res: T.Tensor((rows, dim), RDT)
    G: T.Tensor((dim,), F32)
    Bt: T.Tensor((dim,), F32)
    Y: T.Tensor((rows, dim), ODT)
    Hout: T.Tensor((rows, dim), F32)

    with T.Kernel(T.ceildiv(rows, bm), threads=128) as bx:
        tx = T.get_thread_binding()
        for r in T.serial(bm):
            if tx == r:
                row = bx * bm + r
                acc = T.alloc_var(F32)
                mu = T.alloc_var(F32)
                rs = T.alloc_var(F32)
                acc = 0.0
                for j in T.serial(dim):
                    v = T.cast(X[row, j], F32) + T.cast(Res[row, j], F32)
                    Hout[row, j] = v  # 残差流出口：相加结果原样以 fp32 交给下一层
                    acc = acc + v
                mu = acc / dim
                acc = 0.0
                for j in T.serial(dim):
                    dv = T.cast(X[row, j], F32) + T.cast(Res[row, j], F32) - mu
                    acc = acc + dv * dv
                rs = T.rsqrt(acc / dim + eps)  # 有偏方差（除以 dim），与主流层内整形同口径
                for j in T.serial(dim):
                    dv = T.cast(X[row, j], F32) + T.cast(Res[row, j], F32) - mu
                    Y[row, j] = T.cast(dv * rs * G[j] + Bt[j], ODT)


def plan(
    x2: torch.Tensor,
    res2: torch.Tensor,
    weight: torch.Tensor,
    beta: torch.Tensor,
    y: torch.Tensor,
    h: torch.Tensor,
    eps: float,
) -> dict[str, Any] | None:
    """判断这组形状能否交给方言；能则给出编译参数与缓存键，否则回 None 让入口走回退。

    要求：本层输出那一路是 fp16、老路那一路是 fp16 **或 fp32**（从第二层起它就是上一层交出的
    fp32 残差，不许为迁就内核而降回去）；缩放/偏移是 fp32；行数已是 ROW_BLOCK 的整数倍（补齐由
    入口负责）；特征轴能被 8 整除（否则线程内串行扫描的宽度太窄，宁可走普通写法）。

    白话：先确认尺寸配得上模具——位数对得上、行数已经凑满一整块，才值得开模。
    """
    if x2.dtype != torch.float16 or res2.dtype not in (torch.float16, torch.float32):
        return None
    if weight.dtype != torch.float32 or beta.dtype != torch.float32:
        return None
    if y.dtype not in (torch.float16, torch.float32) or h.dtype != torch.float32:
        return None
    rows, dim = int(x2.size(0)), int(x2.size(1))
    if rows % ROW_BLOCK or dim % 8:
        return None
    tiles = {
        "dim": dim,
        "eps": float(eps),
        "bm": ROW_BLOCK,
        "res_fp32": 1 if res2.dtype == torch.float32 else 0,
        "out_fp16": 1 if y.dtype == torch.float16 else 0,
    }
    key = "add_ln|" + "|".join(f"{kk}={vv}" for kk, vv in tiles.items())
    return {"key": key, "tiles": tiles}


def run(
    x2: torch.Tensor,
    res2: torch.Tensor,
    weight: torch.Tensor,
    beta: torch.Tensor,
    y: torch.Tensor,
    h: torch.Tensor,
    spec: dict[str, Any],
) -> bool:
    """取（或首次编译）内核并就地写 y/h；返回 False 表示方言此刻不可用，入口需要回退。

    缓存键只含特征轴、容差与两路位宽组合，**不含行数**：同一份产物服务任意行数。

    白话：模具只开一次，行数多少不碍事；开不了就老实说干不了，让上面换手工做法。
    """
    kernel = backends.get_compiled(spec["key"], lambda: _build(x2.device, spec["tiles"]))
    if kernel is None:
        return False
    kernel(x2, res2, weight, beta, y, h, **spec["tiles"])
    return True


def _build(device: torch.device, tiles: dict[str, Any]) -> Any:
    """用 bm 行哑数据编译并预热一次，让真正的编译异常在这里暴露、而不是漏进业务调用。"""
    kern = tilelang.jit(**JIT_KWARGS)(add_ln_impl)
    dim = tiles["dim"]
    n = tiles["bm"]
    odtype = torch.float16 if tiles["out_fp16"] else torch.float32
    rdtype = torch.float32 if tiles["res_fp32"] else torch.float16
    x = torch.zeros((n, dim), dtype=torch.float16, device=device)
    res = torch.zeros((n, dim), dtype=rdtype, device=device)
    g = torch.ones(dim, dtype=torch.float32, device=device)
    b = torch.zeros(dim, dtype=torch.float32, device=device)
    y = torch.zeros((n, dim), dtype=odtype, device=device)
    h = torch.zeros((n, dim), dtype=torch.float32, device=device)
    kern(x, res, g, b, y, h, **tiles)
    torch.mps.synchronize()
    return kern
