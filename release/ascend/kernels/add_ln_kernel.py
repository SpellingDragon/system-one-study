"""add_ln 对外入口件：把 ascend.kernels 的名字对齐到 sys1/kernels/add_ln_kernel.py 的调用面。

【做什么】只暴露与 P1 同名同签数的 `forward(x, residual, weight, bias, eps, out_dtype, target)`
与闭式反向 `backward(h, weight, dy, eps, target)`，实现全在 add_ln_asc.py；本文件不含任何算式。
【怎么做】直接 re-export（薄壳），并把 DEFAULT_EPS 指向方言件常量，让上游与 P1 共用同一个取值
来源，避免两处各写一份 1e-5 而漂移。
【为什么】本包的契约是"每件两文件"（方言正文 + 对外入口），分层理由见包 __init__；被否方案是
把算式写在本文件里——那会让 torch 回退口径被方言细节污染。add_ln 是本域唯一**返回两个张量**
的入口（y 可降位宽、h 是必须原封下传的 fp32 残差流），这条纪律由方言件守住，壳层不重复实现。
"""
from ascend.kernels import add_ln_asc, ascend_env

#: 层内 eps；与 sys1/kernels/add_ln_mps.DEFAULT_EPS 同源同值
DEFAULT_EPS = add_ln_asc.DEFAULT_EPS

forward = add_ln_asc.forward
backward = add_ln_asc.backward

__all__ = ["DEFAULT_EPS", "ascend_env", "backward", "forward"]
