"""letter_readout 对外入口件：把本包的名字对齐设计文档的"读出 letter_rows（index_select）"调用面。

【做什么】只暴露 `forward(rows, ids, out_dtype, target)`（按行号表抽行）与
`backward(row_count, ids, dy, target)`（按同一张行号表把梯度加回原表），实现全在
letter_readout_asc.py；本文件不含任何算式。
【怎么做】直接 re-export（薄壳）。P1 的 sys1/kernels/ 里没有这一件（letter 读出是 P2 才上的），
所以接口由本域按设计文档定名，签名风格与其余入口件保持一致（末尾统一挂 target 开关）。
【为什么】分层理由同 rope_kernel：入口层管交货口径（位宽、形状还原、设备），方言层只管算式；
把算式写进入口会让"回退路径"和"方言细节"搅在一处。
"""
from ascend.kernels import ascend_env, letter_readout_asc

forward = letter_readout_asc.forward
backward = letter_readout_asc.backward

__all__ = ["ascend_env", "backward", "forward"]
