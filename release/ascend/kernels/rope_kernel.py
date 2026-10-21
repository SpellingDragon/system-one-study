"""rope 对外入口件：把 ascend.kernels 的名字对齐到 sys1/kernels/rope_kernel.py 的调用面。

【做什么】只暴露与 P1 同名同签数的 `forward(qkv, cos, sin, rotate_slots=DEFAULT_SLOTS)` 与
`backward(...)`，实现全在 rope_asc.py；本文件不含任何算式。
【怎么做】直接 re-export（薄壳），另把 DEFAULT_SLOTS 常量指向方言件的 ROTATE_SLOTS，让上游
和 P1 用同一个取值来源，避免两处各写一份 (0,1) 而漂移。
【为什么】被否方案：把算式写在本文件里——那会让"入口层/方言层"分层失效，回退口径被方言细节
污染（分层理由见包 __init__）。薄壳的代价是多一个文件，换来的是"P1 的调用行原样搬过来就能跑"。
"""
from ascend.kernels import ascend_env, rope_asc

#: 被旋转的槽位；与 sys1/kernels/rope_kernel.DEFAULT_SLOTS 同源同值
DEFAULT_SLOTS = rope_asc.ROTATE_SLOTS

forward = rope_asc.forward
backward = rope_asc.backward

__all__ = ["DEFAULT_SLOTS", "ascend_env", "backward", "forward"]
