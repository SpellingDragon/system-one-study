"""rope 的 Metal 方言实现：packed-qkv 就地旋转（rotate-half），只动前两摞、第三摞不碰。

【做什么】对上暴露一份能发射到 Apple GPU 的方言实现：把打包存放的三份数据（查询、键、值）里
前两摞按位置转个角度——越靠后的位置转得越多，两个位置一减就能看出相隔多远；第三摞必须逐位
保持原样。
【怎么做】① 网格按 token 数铺开——**一个 token 占一个块**（块数就是动态的 tokens），这样每个
块的下标天然小于总长，根本不存在"最后一块越界"的问题，也就不需要在内核里写边界守卫；
② 每块内先对槽位 0、1 串行两轮，每轮用 T.Parallel 覆盖 (头, 前半下标)：从全局读回同一对的
前后两个数、升到 fp32、查 fp32 的 cos/sin 分表、算出新值后**就地写回原槽位**（写回时才降位宽）；
③ 角度表由外部预计算好递进来，内核里不出现任何三角函数或幂运算。
【为什么】"一块一 token"牺牲了块内并行度（每块只有 heads×half 个格子要做），换来的是零边界
处理：实测在本方言里对动态长度写 `if tok < tokens:` 这类语句级守卫会得到错误结果，而 T.copy 的
自动谓词又帮不上"就地读写全局"这种非搬运算。被否方案一：把 tokens 做成编译期常量——省掉动态
符号，但每换一句长度就重编一次，与"一次编译多批复用"的验收口径相背；被否方案二：内核里现算
角度——Metal 侧的超越函数通路在本域未验证（B1 探针未覆盖），一旦不可用整个算子就得回退，
不如把三角函数留在 torch 侧，两边共用同一张表，对拍才有唯一真源。就地写回是刻意的：与
`refs/laya/tl_kernels.py::rope_kernel` 的用法一致，省下一份 (tokens,3,heads,dim) 的临时内存。
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

#: 打包布局里被旋转的槽位：0=查询、1=键；2=值 永远不动
ROTATE_SLOTS = (0, 1)


def rope_impl(QKV, Cos, Sin, heads: int, dim: int):
    """方言正文（被 tilelang 追踪，不直接调用）：token 数为动态符号，头数与每头宽度是编译期常量。

    追踪期约定：heads/dim 必须是 python int，`half = dim // 2` 才能在追踪阶段就是整数，
    从而让前后半的下标偏移在生成的 MSL 里是常量。

    白话：每格材料自己占一个小工；工头把前两摞按小抄上的角度转一转——前半减去后半乘纵向抄，
    后半加上前半乘横向抄，算完就原地放回；第三摞连碰都不碰。
    """
    tokens = T.dynamic("tokens")
    half = dim // 2

    QKV: T.Tensor((tokens, 3, heads, dim), F16)
    Cos: T.Tensor((tokens, half), F32)
    Sin: T.Tensor((tokens, half), F32)

    with T.Kernel(T.ceildiv(tokens, 1), threads=128) as bx:
        for s in T.serial(ROTATE_SLOTS[0], ROTATE_SLOTS[0] + len(ROTATE_SLOTS)):
            for hh, j in T.Parallel(heads, half):
                x1 = T.cast(QKV[bx, s, hh, j], F32)
                x2 = T.cast(QKV[bx, s, hh, j + half], F32)
                c = Cos[bx, j]
                sn = Sin[bx, j]
                QKV[bx, s, hh, j] = T.cast(x1 * c - x2 * sn, F16)
                QKV[bx, s, hh, j + half] = T.cast(x2 * c + x1 * sn, F16)


def plan(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> dict[str, Any] | None:
    """判断这组打包数据能否交给方言；能则给出编译参数与缓存键，否则回 None 让入口走回退。

    要求：qkv 是 (tokens, 3, heads, dim) 的 fp16 且**内存连续**、两张角度表是 (tokens, dim//2)
    的 fp32、dim 为偶数。连续性这条不能松：内核是就地改写的，一旦对方递的是 `contiguous()`
    临时复制出来的另一块内存，转完原对象纹丝不动，"就地"语义就静默失效了。

    缓存键只含 heads 与 dim，**不含 token 数**——换长度不换产物。

    白话：先核对盒子是不是"三摞、每摞一行格、格子连着摆"的标准形状，抄表也没换成省格子的记法；
    形状正、位数对、东西也确实摊在一整张桌上，才值得动用模具。
    """
    if qkv.dtype != torch.float16 or qkv.dim() != 4 or qkv.size(1) != 3:
        return None
    if not (qkv.is_contiguous() and cos.is_contiguous() and sin.is_contiguous()):
        return None
    tokens, _, heads, dim = (int(v) for v in qkv.shape)
    if dim % 2 or cos.dtype != torch.float32 or sin.dtype != torch.float32:
        return None
    if cos.shape != (tokens, dim // 2) or sin.shape != (tokens, dim // 2):
        return None
    tiles = {"heads": heads, "dim": dim}
    key = "rope|" + "|".join(f"{kk}={vv}" for kk, vv in tiles.items())
    return {"key": key, "tiles": tiles}


def run(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, spec: dict[str, Any]) -> bool:
    """取（或首次编译）内核并**就地**旋转 qkv；返回 False 表示方言此刻不可用，入口需要回退。

    白话：把盒子原样递进模具、原地就转好了拿回来；模具开不了就说一声，让外面手工转。
    """
    kernel = backends.get_compiled(spec["key"], lambda: _build(qkv.device, spec["tiles"]))
    if kernel is None:
        return False
    kernel(qkv, cos, sin, **spec["tiles"])
    return True


def _build(device: torch.device, tiles: dict[str, Any]) -> Any:
    """用 1 个 token 的哑数据编译并预热一次，把编译期异常挡在业务调用之外。"""
    kern = tilelang.jit(**JIT_KWARGS)(rope_impl)
    heads, dim = tiles["heads"], tiles["dim"]
    dummy_qkv = torch.zeros((1, 3, heads, dim), dtype=torch.float16, device=device)
    dummy_cos = torch.ones((1, dim // 2), dtype=torch.float32, device=device)
    dummy_sin = torch.zeros((1, dim // 2), dtype=torch.float32, device=device)
    kern(dummy_qkv, dummy_cos, dummy_sin, **tiles)
    torch.mps.synchronize()
    return kern
