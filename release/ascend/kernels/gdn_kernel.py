"""GDN 对外入口件：把本包的名字对齐设计文档的"Gated DeltaRule 前向 + 反向（内核化）+ 短卷积"调用面。

【做什么】暴露 `forward(q, k, v, g, beta, out_dtype, target)`、`backward(...)`、
`conv_forward(x, w, bias, out_dtype, target)` 与状态标记 `BWD_STATUS`；实现全在
`gdn_asc.py`（递推前后向）与 `gdn_conv_asc.py`（短卷积），本文件不含任何算式。
【怎么做】直接 re-export（薄壳）。`BWD_STATUS` 一并导出，让上层与测试能读到"反向走到了哪一步"
这个边界，而不是靠猜——上游若要临时走混合栈（该层用 torch、其余算子用自研），按这个分支。
【为什么】分层理由同其余入口件；额外一条：GDN 是本域最重件，反向与 conv 的边界必须写在接口面
上，避免下游误以为已经拿到内核速度或误以为 conv 还在上游（红线 R14 禁止假绿）。
"""
from ascend.kernels import ascend_env, gdn_asc, gdn_conv_asc

#: 反向状态（P1-4 合流起）："kernelized" = ascend target 走 910B 标量面伴随件，六项梯度
#: dq/dk/dv/dg/dβ（+ dH0，`return_dh0=True` 时）全实现，形态见 gdn_asc.gdn_bwd_asc_impl；
#: cpu target 仍走 torch 自动微分（语义尺子路），并如实登记 blocker。
#: 两档：`ub`（伴随态常驻 UB，首推）/`pure`（零 UB、就地读写 GM 工作缓冲，跨 token 的
#: 读后写可见性待卡证 G-F3）。并行约束：反向按 **head** 切核 ⇒ 要求 heads 能被核块数均分
#: （核块数由 gdn_asc._pick_cores 挑，上限 env `SYS1_GDN_BLOCKS`，缺省 8）；前向按
#: (head, 值维列) 切，要求 heads*dv 能均分。理由与公式见 gdn_asc 模块与件内 docstring。
BWD_STATUS = gdn_asc.BWD_STATUS
#: `backward` 返回值的名字与顺序（`return_dh0=True` 时末尾多一个 dH0）
BWD_GRADES = gdn_asc.BWD_GRADES
#: 递推件/卷积件的昇腾档位名（细节见两件 docstring）
ASC_VARIANTS = gdn_asc.ASC_VARIANTS
CONV_ASC_VARIANTS = gdn_conv_asc.ASC_VARIANTS
#: 短卷积窗宽（GDN 约定 4）；非 4 抽头本件不开模，落回普通写法
CONV_KERNEL = gdn_conv_asc.CONV_KERNEL

forward = gdn_asc.forward
backward = gdn_asc.backward
plan = gdn_asc.plan
run = gdn_asc.run
plan_backward = gdn_asc.plan_backward
run_backward = gdn_asc.run_backward

#: conv 位点入口（P1-4 新增，D-int2(b)：原先"在上游之外"、接口 home 无入口）
conv_forward = gdn_conv_asc.conv_forward
conv_plan = gdn_conv_asc.plan
conv_run = gdn_conv_asc.run

__all__ = ["ASC_VARIANTS", "BWD_GRADES", "BWD_STATUS", "CONV_ASC_VARIANTS", "CONV_KERNEL",
           "ascend_env", "backward", "conv_forward", "conv_plan", "conv_run", "forward",
           "gdn_asc", "gdn_conv_asc", "plan", "plan_backward", "run", "run_backward"]
