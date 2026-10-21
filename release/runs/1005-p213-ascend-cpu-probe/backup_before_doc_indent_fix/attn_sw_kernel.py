"""attn_sw 对外入口件：把 ascend.kernels 的名字对齐 sys1/kernels/attn_sw_kernel.py 的调用面。

【做什么】只暴露与 P1 同名同签（末尾多挂一个 target 开关）的 `forward(q, k, v, window, scale,
out_dtype)` 与调试用的 `forward_weights(q, k, window, scale)`，实现全在 attn_sw_asc.py。
【怎么做】直接 re-export（薄壳），并把块宽常量指向方言件，让上游与 P1 用同一个取值来源。
【为什么】分层理由同 rope_kernel/add_ln 入口件：入口管交货口径（形状还原、升降位、设备），
方言层只管算式；把算式写进入口会让回退路径被方言细节污染。
"""
from ascend.kernels import ascend_env, attn_sw_asc

#: 查询块高/键块宽与哨兵、地板值：与方言件同源，避免两处各写一份而漂移
DEFAULT_BLOCK_Q = attn_sw_asc.BLOCK_Q
DEFAULT_BLOCK_KV = attn_sw_asc.BLOCK_KV

forward = attn_sw_asc.forward
forward_weights = attn_sw_asc.forward_weights

__all__ = ["DEFAULT_BLOCK_KV", "DEFAULT_BLOCK_Q", "ascend_env", "forward", "forward_weights"]
