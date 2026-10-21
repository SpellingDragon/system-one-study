"""gemm 权重梯度的 Metal 方言件：dW = dYᵀ @ A，沿 token 轴累加（token 数是动态符号）。

【做什么】给训练反向算"权重该往哪儿挪"的那一步：把整批样本的头信号转置后乘回输入，
沿"这一批有几条样本"的轴累加成一张和权重同形状的梯度表。
【怎么做】① 网格按 (N/bn, K/bk) 铺开，每块负责梯度表的一小方格；② token 轴切成固定块
   tc 串行循环——循环上界 ceildiv(m, tc) 里 m 是动态符号，所以换批大小不重编译；
③ 每块先在 threadgroup 里攒 fp32 累加器 Cs，两切片按 (tc,bn)/(tc,bk) 载入后
   T.gemm(Ds, As, Cs, transpose_A=True) 继续累加（不 clear，循环前清一次）；
④ T.copy 的区域切片自动带边界谓词，m 不被 tc 整除时尾块不越读（探针实测 m=33 精确）。
【为什么】现成的前向 gemm 模具把动态轴放在"行数"上、归约轴必须是编译期常量，而权重梯度
偏偏要沿动态的 token 数归约——旧模具表达不了，必须另开一模。被否方案一：dW 留普通
张量库写法——放弃它等于放弃训练主链里一半的内核收益（每层的权重梯度都走它）；
被否方案二：把 token 数固定成编译期常量、换批就重编——训练里批形状多变，重编
两秒级开销直接吞掉收益（探针数据：一次编译 ≈2.4s）。方言能力先经探针验证
（transpose_A 与动态循环界、非整除尾块三项全过，err=0）才落成此件。
"""
from __future__ import annotations

from typing import Any

import tilelang
import tilelang.metal.language as T
import torch

from sys1.kernels import backends

F16, F32 = T.float16, T.float32

#: 与前向件同一发射口径：Metal 目标 + tvm_ffi（默认 torch adapter 在动态尺寸下崩，见 gemm_mps 记录）
JIT_KWARGS: dict[str, Any] = {"target": "metal", "execution_backend": "tvm_ffi"}

#: token 轴的块长：16 与前向 BLOCK_M 同量级，太小喂不饱并行、太大尾块浪费
BLOCK_TC = 16

#: N/K 轴块宽候选（沿用前向阶梯）；都不能整除则表达不了，交回入口层回退
BLOCK_LADDER = (64, 32, 16)


def gemm_dw_impl(dY, A, GW, n: int, k: int, tc: int, bn: int, bk: int):
    """方言正文（被追踪，不直接调用）：token 数 m 动态，其余维度是编译期常量。

    追踪期约定：n/k/tc/bn/bk 必须是 python int 才能定网格与切片；GW 出口固定 fp32，
    降位宽由入口层负责（保持累加精度是纪律）。

    白话：梯度表切成小方格，每格自己把"这一批所有样本对该格的贡献"一笔笔记到同一个
    账本上；样本有多少条不影响账本格子的大小，账本记完一次交账。
    """
    m = T.dynamic("m")

    dY: T.Tensor((m, n), F16)
    A: T.Tensor((m, k), F16)
    GW: T.Tensor((n, k), F32)

    with T.Kernel(T.ceildiv(n, bn), T.ceildiv(k, bk), threads=128) as (by, bx):
        Ds = T.alloc_shared((tc, bn), F16)
        As = T.alloc_shared((tc, bk), F16)
        Cs = T.alloc_shared((bn, bk), F32)
        T.clear(Cs)
        for to in T.serial(T.ceildiv(m, tc)):
            T.copy(dY[to * tc:(to + 1) * tc, by * bn:(by + 1) * bn], Ds)
            T.copy(A[to * tc:(to + 1) * tc, bx * bk:(bx + 1) * bk], As)
            T.gemm(Ds, As, Cs, transpose_A=True)
        T.copy(Cs, GW[by * bn:(by + 1) * bn, bx * bk:(bx + 1) * bk])


def _largest_block(total: int) -> int | None:
    """候选块宽里取能整除 total 的最大值；取不到即表达不了（返回 None 交回退）。"""
    for blk in BLOCK_LADDER:
        if total % blk == 0:
            return blk
    return None


def plan(dY: torch.Tensor, A: torch.Tensor, GW: torch.Tensor) -> dict[str, Any] | None:
    """这组形状能否交给方言模具：两输入 fp16、同行数、fp32 出口、N/K 可整除。

    白话：先看尺寸合不合现成模具；不合就老实说干不了，由入口换普通做法，绝不硬凑。
    """
    if dY.dtype != torch.float16 or A.dtype != torch.float16:
        return None
    if GW.dtype != torch.float32:
        return None
    if dY.dim() != 2 or A.dim() != 2 or dY.size(0) != A.size(0):
        return None
    n, k = int(GW.size(0)), int(GW.size(1))
    bn, bk = _largest_block(n), _largest_block(k)
    if bn is None or bk is None:
        return None
    if dY.size(1) != n or A.size(1) != k:
        return None
    tiles = {"n": n, "k": k, "tc": BLOCK_TC, "bn": bn, "bk": bk}
    key = "gemm_dw|" + "|".join(f"{kk}={vv}" for kk, vv in tiles.items())
    return {"key": key, "tiles": tiles}


def run(dY: torch.Tensor, A: torch.Tensor, GW: torch.Tensor, spec: dict[str, Any]) -> bool:
    """取（或首编）内核句柄就地写 GW；False 表示方言此刻不可用，须回退。

    缓存键不含行数：同一份产物服务 16、64、8192 等不同批大小——权重梯度每次训练步
    都要算，若换批重编则两秒级开销吞掉一切收益。

    白话：模具只开第一次，之后不管一批里有多少条样本都拿同一个模具用；开不了就如实报。
    """
    kernel = backends.get_compiled(spec["key"], lambda: _build(dY.device, spec["tiles"]))
    if kernel is None:
        return False
    kernel(dY, A, GW, **spec["tiles"])
    return True


def _build(device: torch.device, tiles: dict[str, Any]) -> Any:
    """编译并预热一次（1 行哑数据），把编译异常挡在业务调用之前。

    白话：先在台架上空转一遍确认模具能用，真正的活儿来了才不会当场把机器掀了。
    """
    kern = tilelang.jit(**JIT_KWARGS)(gemm_dw_impl)
    n, k, tc = tiles["n"], tiles["k"], tiles["tc"]
    dummy_dy = torch.zeros((tc, n), dtype=torch.float16, device=device)
    dummy_a = torch.zeros((tc, k), dtype=torch.float16, device=device)
    dummy_gw = torch.zeros((n, k), dtype=torch.float32, device=device)
    kern(dummy_dy, dummy_a, dummy_gw, **tiles)
    torch.mps.synchronize()
    return kern
