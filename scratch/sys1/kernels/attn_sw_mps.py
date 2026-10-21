"""attn_sw 的 Metal 方言实现：因果**单向**滑窗注意力（在线 softmax，flash 式分块）。

【做什么】对每个查询位置，只允许它看"自己和前面最多 W 个键"，把这部分加权平均成输出；
窗口之外的过去、以及一切未来位置，权重必须严格为零。
【怎么做】① 网格 (查询块, 头)；每块 bm 行查询。② 每个查询块自己算要扫哪些键块：起点
`kb_begin = (q_lo + 1 - W)` 向下取整到块、终点 `kb_end = ceil((q_lo + bm)/bn)`——只有这段
可能可见，窗口外的键块**根本不读**，这是滑窗相对全量注意力的主要收益。③ 键块内：
`T.gemm(Qs, Ks, Ss, transpose_B=True)` 出分数（fp32 共享累加器），再逐格判可见性
`0 <= q_lo + i - k_idx < W`，不可见一律压成极大负数；④ 每行交给一个线程做在线 softmax：
更新行最大值 → 用 `exp2(旧max - 新max)` 把已攒的输出整体缩小 → 把本块概率写进 Ps 并累计
行和；⑤ `T.gemm(Ps, Vs, Os, clear_accum=False)` 把本块贡献累加进共享输出；⑥ 全部键块扫完，
除以行和并按需降位宽写到全局。
【为什么】⚠️ 蓝本 `refs/laya/tl_kernels.py::attn_kernel` 是**双向**滑窗（`|i-j| < W`），照抄
方向就把"未来不可见"这条纪律抄没了；本文件写死单向 `0 <= i-j < W`，且测试用"未来权重恒为 0"
钉住。三个看似多余的细节都是被方言逼出来的：(a) 分数与输出都放共享内存而不是 fragment——对
gemm 的 fragment 累加器做逐元素运算会在代码生成期直接失败（原文见 gemm_mps 模块头）；
(b) 行最大/行和走"一行一线程串行扫"而不是 T.reduce_*——Metal 后端没注册归约实现，编译期报
`no reduce implementation is registered for {"kind":"metal"}`；(c) 底数换成 2（`exp2` + 分数
预先乘 log2e），因为 MSL 侧 `exp2` 是最自然的一条硬件指令，softmax 数值口径不受影响。
被否方案一：先算完整分数矩阵再做普通 softmax——多驻留一整块 seq×seq 的分数，长上下文下内存
与带宽都吃不消，而在线算法的中间量只有每块 bn 个；被否方案二：把不可见位置填 0 而不是极大
负数——填 0 等于"它可见且分数为零"，会稀释行和、把窗口外的键算进平均。
TODO(学生版)：`_cuda` 与本文件同源，把网格换成 (查询块, 头, 批) 并把 W 做成动态符号即可。
"""
from __future__ import annotations

from typing import Any

import tilelang
import tilelang.metal.language as T
import torch

from sys1.kernels import backends

F16, F32 = T.float16, T.float32

#: 与 gemm/add_ln 同一套发射参数（Metal 目标 + tvm_ffi 后端），原因见 gemm_mps 模块头
JIT_KWARGS: dict[str, Any] = {"target": "metal", "execution_backend": "tvm_ffi"}

#: 查询块高与键块宽：两者必须互相整除，否则块终点推算与"最后一块"的边界会错位
BLOCK_Q = 16
BLOCK_KV = 16

#: 不可见位置的分数填充值。用 -1e30 而不是 -inf：exp2(-inf - max) 在某些路径上出 NaN，
#: 而 -1e30 减去行最大后必然下溢成 0，效果与 -inf 相同且无未定义行为
NEG = -1.0e30

#: 自然对数底换成 2 底时的换算系数（log2(e)），与 softmax 的数值口径等价
LOG2E = 1.4426950408889634

#: 行和的保护下限。因果口径下每行至少看得见对角自己，正常数据永远走不到这个分支
ROWSUM_FLOOR = 1.0e-30


def attn_sw_impl(
    Qin,
    Kin,
    Vin,
    Oout,
    heads: int,
    seq: int,
    dim: int,
    window: int,
    bm: int,
    bn: int,
    out_fp16: int,
):
    """方言正文（被 tilelang 追踪，不直接调用）：头数/长度/每头宽度/窗口都是编译期常量。

    追踪期约定：以上参数必须是 python int——`seq` 参与网格与可见性判断、`window` 参与键块
    起点推算，都要求追踪阶段就是整数。出口位宽由 out_fp16 决定（写回时降精度，中间量全 fp32）。

    白话：每个小工只管自己那一行，从左往右只看"自己和前面最多 W 个"的存货；算分时先把不该
    看见的位置涂成极小的负数（一指数化就成了零），边看边把已攒的结果按新出现的更大分数整体
    缩一缩，最后一律除以自己看到的总份数。
    """
    ODT = F16 if out_fp16 else F32
    scale = (1.0 / dim) ** 0.5 * LOG2E  # 1/sqrt(dim) 与换底系数合并进一次乘法

    Qin: T.Tensor((heads, seq, dim), F16)
    Kin: T.Tensor((heads, seq, dim), F16)
    Vin: T.Tensor((heads, seq, dim), F16)
    Oout: T.Tensor((heads, seq, dim), ODT)

    with T.Kernel(T.ceildiv(seq, bm), heads, threads=128) as (bq, bh):
        Qs = T.alloc_shared((bm, dim), F16)
        Ks = T.alloc_shared((bn, dim), F16)
        Vs = T.alloc_shared((bn, dim), F16)
        Ss = T.alloc_shared((bm, bn), F32)
        Ps = T.alloc_shared((bm, bn), F16)
        Os = T.alloc_shared((bm, dim), F32)
        rowmax = T.alloc_shared((bm,), F32)
        rowsum = T.alloc_shared((bm,), F32)
        bmax = T.alloc_shared((bm,), F32)
        bsum = T.alloc_shared((bm,), F32)
        alpha = T.alloc_shared((bm,), F32)

        tx = T.get_thread_binding()
        q_lo = bq * bm
        kb_begin = T.floordiv(T.max(0, q_lo + 1 - window), bn)
        kb_end = T.floordiv(T.min(seq, q_lo + bm) + bn - 1, bn)

        T.copy(Qin[bh, q_lo:q_lo + bm, :], Qs)
        T.fill(Os, 0)
        T.fill(rowsum, 0)
        for r in T.serial(bm):
            rowmax[r] = NEG
        T.sync_threads()

        for kb in T.serial(kb_begin, kb_end):
            T.copy(Kin[bh, kb * bn:(kb + 1) * bn, :], Ks)
            T.copy(Vin[bh, kb * bn:(kb + 1) * bn, :], Vs)
            T.clear(Ss)
            T.gemm(Qs, Ks, Ss, transpose_B=True)
            for i, j in T.Parallel(bm, bn):
                k_idx = kb * bn + j
                Ss[i, j] = T.if_then_else(
                    (q_lo + i >= k_idx) & (q_lo + i - k_idx < window) & (k_idx < seq),
                    Ss[i, j] * scale,
                    NEG,
                )
            for r in T.serial(bm):
                if tx == r:
                    bmax[r] = NEG
                    for j in T.serial(bn):
                        bmax[r] = T.max(bmax[r], Ss[r, j])
                    alpha[r] = T.exp2(rowmax[r] - T.max(rowmax[r], bmax[r]))
                    rowmax[r] = T.max(rowmax[r], bmax[r])
                    for c in T.serial(dim):
                        Os[r, c] = Os[r, c] * alpha[r]
                    bsum[r] = 0.0
                    for j in T.serial(bn):
                        e = T.exp2(Ss[r, j] - rowmax[r])
                        Ps[r, j] = T.cast(e, F16)
                        bsum[r] = bsum[r] + e
                    rowsum[r] = rowsum[r] * alpha[r] + bsum[r]
            T.sync_threads()
            T.gemm(Ps, Vs, Os, clear_accum=False)
            T.sync_threads()

        for r in T.serial(bm):
            if tx == r:
                inv = 1.0 / T.max(rowsum[r], ROWSUM_FLOOR)
                for c in T.serial(dim):
                    Oout[bh, q_lo + r, c] = T.cast(Os[r, c] * inv, ODT)


def plan(
    q3: torch.Tensor,
    k3: torch.Tensor,
    v3: torch.Tensor,
    out: torch.Tensor,
    window: int,
    scale: float | None,
) -> dict[str, Any] | None:
    """判断这组形状/口径能否交给方言；能则给出编译参数与缓存键，否则回 None 让入口走回退。

    四条硬门槛：三路输入都是 (heads, seq, dim) 的 fp16 且长度一致；seq 同时被查询块高与键块宽
    整除（补齐由入口按 ladder 负责，内核里不写边界守卫）；dim 被 16 整除；缩放系数就是
    `1/sqrt(dim)`（内核把系数和换底合并成一次乘法，别的系数只能走普通写法）。

    白话：先看这三摞本子是不是标准开本、每页行数能不能正好切成整数块；不合本就不开模，
    直接换手工做法。
    """
    if q3.dtype != torch.float16 or k3.dtype != torch.float16 or v3.dtype != torch.float16:
        return None
    if out.dtype not in (torch.float16, torch.float32):
        return None
    if q3.shape != k3.shape or q3.shape != v3.shape:
        return None
    if window < 1:
        return None
    heads, seq, dim = (int(x) for x in q3.shape)
    if seq % BLOCK_Q or seq % BLOCK_KV or dim % 16:
        return None
    if scale is not None and abs(scale - dim ** -0.5) > 1e-12:
        return None
    tiles = {
        "heads": heads,
        "seq": seq,
        "dim": dim,
        "window": int(window),
        "bm": BLOCK_Q,
        "bn": BLOCK_KV,
        "out_fp16": 1 if out.dtype == torch.float16 else 0,
    }
    key = "attn_sw|" + "|".join(f"{kk}={vv}" for kk, vv in tiles.items())
    return {"key": key, "tiles": tiles}


def run(
    q3: torch.Tensor,
    k3: torch.Tensor,
    v3: torch.Tensor,
    out: torch.Tensor,
    spec: dict[str, Any],
) -> bool:
    """取（或首次编译）内核并写 out；返回 False 表示方言此刻不可用，入口需要回退。

    缓存键含 seq 与 window——本算子按"长度 ladder"逐档编译（design 约定），换档重编是设计内
    行为，与 gemm 的"一次编译多批复用"口径不同。

    白话：这一档的开本配这一档的模具；模具打不开就如实说干不了，让上面换手工做法。
    """
    kernel = backends.get_compiled(spec["key"], lambda: _build(q3.device, spec["tiles"]))
    if kernel is None:
        return False
    kernel(q3, k3, v3, out, **spec["tiles"])
    return True


def _build(device: torch.device, tiles: dict[str, Any]) -> Any:
    """按 tiles 编译一次并真正发射一发哑数据，把编译期异常挡在业务调用之前。"""
    kern = tilelang.jit(**JIT_KWARGS)(attn_sw_impl)
    heads, seq, dim = tiles["heads"], tiles["seq"], tiles["dim"]
    odtype = torch.float16 if tiles["out_fp16"] else torch.float32
    dummy_q = torch.zeros((heads, seq, dim), dtype=torch.float16, device=device)
    dummy_o = torch.zeros((heads, seq, dim), dtype=odtype, device=device)
    kern(dummy_q, dummy_q, dummy_q, dummy_o, **tiles)
    torch.mps.synchronize()
    return kern
