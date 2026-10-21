"""GDN 对外入口件：把本包的名字对齐设计文档的"Gated DeltaRule 前向（+反向 partial）"调用面。

【做什么】只暴露 `forward(q, k, v, g, beta, out_dtype, target)`、`backward(...)` 与状态标记
`BWD_STATUS`，实现全在 gdn_asc.py；本文件不含任何算式。
【怎么做】直接 re-export（薄壳）。`BWD_STATUS` 一并导出，让上层与测试能读到"反向未内核化"这个
边界，而不是靠猜——上游若要临时走混合栈（该层用 torch、其余算子用自研），按这个标记分支。
【为什么】分层理由同其余入口件；额外一条：GDN 是本域最重件，反向的边界必须写在接口面上，
避免下游误以为已经拿到内核速度（红线 R14 禁止假绿）。
"""
from ascend.kernels import ascend_env, gdn_asc

#: "partial" = 反向走 torch 自动微分，本波没有方言件；理由见 gdn_asc 模块 docstring
BWD_STATUS = gdn_asc.BWD_STATUS

forward = gdn_asc.forward
backward = gdn_asc.backward

__all__ = ["BWD_STATUS", "ascend_env", "backward", "forward"]
