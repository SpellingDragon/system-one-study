"""torch 参考实现出口（对拍生命线：内核未编译成功时也以这些函数为准绳）。

四算子契约各自成文件，这里只做统一出口，避免测试侧写四条深路径：
- gemm_ref / activate_ref     —— C = act(A @ Wᵀ + b)，fp32 累加
- add_ln_ref / layer_stats_ref —— 残差相加 + 层内整形，残差流保 fp32
- rope_ref / rope_angle_tables —— packed-qkv 就地旋转，cos/sin 分表 fp32
- attn_sw_ref / attn_sw_weights_ref / causal_window_mask —— 因果滑窗（只回看不看未来）

【做什么】给 tests/ 与内核对拍提供一个稳定的导入点。
【怎么做】纯 re-export，不做任何计算，也不导入 tilelang/torch 之外的依赖。
【为什么】集中出口能让"参考实现"与"内核实现"的对应关系一目了然；被否方案：让测试各自
按深路径导入——一旦文件名调整，散落的导入会同时崩，排查成本更高。
"""
from __future__ import annotations

from sys1.testing.torch_ref.add_ln_ref import DEFAULT_EPS, add_ln_ref, layer_stats_ref
from sys1.testing.torch_ref.attn_sw_ref import (
    NEG_INF,
    attn_sw_ref,
    attn_sw_weights_ref,
    causal_window_mask,
)
from sys1.testing.torch_ref.gemm_ref import ACTIVATIONS, activate_ref, gemm_ref
from sys1.testing.torch_ref.rope_ref import DEFAULT_THETA, rope_angle_tables, rope_ref, rotate_half_ref

__all__ = [
    "ACTIVATIONS",
    "DEFAULT_EPS",
    "DEFAULT_THETA",
    "NEG_INF",
    "activate_ref",
    "add_ln_ref",
    "attn_sw_ref",
    "attn_sw_weights_ref",
    "causal_window_mask",
    "gemm_ref",
    "layer_stats_ref",
    "rope_angle_tables",
    "rope_ref",
    "rotate_half_ref",
]
