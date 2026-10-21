"""p2-13 R2 方言件包：七类算子，每件两文件（`<op>_asc.py` 方言正文 + `<op>_kernel.py` 对外入口）。

【做什么】对外只暴露与 sys1/kernels 同名同签数（末尾多一个 `target` 开关）的入口模块；`_asc`
结尾的模块是方言侧的 plan/run 两级，不由业务代码直接调用。
【怎么做】`ascend_env` 是唯一的分发前置查询点（target 归一、可用性自证、编译缓存与阻塞登记）。
本 `__init__` 把八枚入口件显式导入，让下游 `import ascend.kernels` 之后可直接按属性取用
（`ascend.kernels.gemm_kernel.forward(...)`），不必逐个子模块 import。
【为什么】入口层与方言层分文件是被否方案的反面：混写会让 torch 回退口径被方言细节污染。
显式导入的代价是 `import ascend.kernels` 会连带拉起 tilelang（约秒级），换来的是"调用行与 P1
一模一样"——被否方案是保持惰性导入：那样每处下游都要多写一行子模块 import，反而更容易漂移。
"""
from ascend.kernels import (  # noqa: F401  八枚入口件，业务只应 import 这一层
    add_ln_kernel,
    ascend_env,
    attn_sw_kernel,
    gemm_bwd_dw_kernel,
    gemm_kernel,
    gdn_kernel,
    letter_readout_kernel,
    lora_kernel,
    rope_kernel,
)

#: 七类算子的对外入口名（linear 占两枚：前向件与 dW 反向件）
ENTRY_MODULES = (
    "gemm_kernel", "gemm_bwd_dw_kernel", "rope_kernel", "add_ln_kernel",
    "letter_readout_kernel", "attn_sw_kernel", "gdn_kernel", "lora_kernel",
)

__all__ = ("ENTRY_MODULES", "add_ln_kernel", "ascend_env", "attn_sw_kernel", "gemm_bwd_dw_kernel",
           "gemm_kernel", "gdn_kernel", "letter_readout_kernel", "lora_kernel", "rope_kernel")
