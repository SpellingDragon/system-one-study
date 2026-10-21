"""LoRA 注入件对外入口：把 ascend.kernels 的名字对齐到本域设计文档的旁路调用面。

【做什么】只暴露 `apply(x, a, b, scaling, base, out_dtype, target)`、
`backward(dy, x, a, b, scaling, target)`、`merge(a, b, scaling, out_dtype, target)` 三个入口，
实现全在 lora_asc.py；本文件不含任何算式。
【怎么做】直接 re-export（薄壳）。本件与其余六件不同：**它没有自己的方言正文**，三条链全部
由已验证的 linear 两件（gemm 前向件 + gemm_bwd_dw 反向件）组合而成，因此方言层的"两文件"在此
退化为"一个组合件 + 一个壳"，仍保持 `_kernel` 结尾=可被业务直接 import 的这一层。
【为什么】被否方案一：让上游（p2-05/06/11 三栈与 p2-10 推理）自己串两次 gemm——那样"训练期走
旁路、推理期才合并"这条口径会在四处各写一遍，缩放折进 B 这个关键决定也会被写歪；被否方案二：
在本壳里补一份 torch 回退——回退口径必须只有一个来源（lora_asc），否则两处会漂移。
P1 没有对应的 LoRA 件，故本件的签名是本域按 design.md 自定的，不受"同名同签名"红线约束。
"""
from ascend.kernels import ascend_env, lora_asc

#: 交付位宽默认值；与其余入口件一致
DEFAULT_OUT_DTYPE = lora_asc.DEFAULT_OUT_DTYPE

apply = lora_asc.apply
backward = lora_asc.backward
merge = lora_asc.merge

__all__ = ["DEFAULT_OUT_DTYPE", "apply", "ascend_env", "backward", "merge"]
