# run: 1007-p2-13-c1-ascend-probe-de15

- 假设：待填写
- 观察：待填写

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。
结论：C1 底座决策：全 TileLang 自研栈在 910B4+CANN8.5.2 **不可行**（方言面向 950，三重证据）——按 D6 预授权回退 **torch_npu 底座跑三栈**（用户 2026-10-05 决策的既定回退，非临时改道）；G9 叙事调整为『910B 自研方言件待 CANN/950 版本线，本波 torch_npu 底座 + 回退决策全程留痕』。附加发现：① 0.8B bf16 OBS 载入仅 9s；② s3fs 挂载可死且不可自愈——数据面须冗余；③ 非持久 pip 依赖清单须留一行式恢复脚本。计时 20:34–21:00 ≈ 26min ≈ ¥8.7。

补录（重启后收口，21:20）：桥复通后重跑吞吐探针，**bf16 前向反向全通**（此前 SetPrecisionMode 崩溃系断桥伴生假象）：
- fwd 2×512：bf16 1047ms=978 tok/s / fp32 1072ms=955 tok/s（eval 口径，bs2 小批未喂满；GDN 层走参考实现——causal_conv1d/FLA 为 CUDA-only，NPU 亦无快路径，此数偏保守）
- backward 通（梯度产出 OK）；峰值 bf16 2.5GB / fp32 5.8GB
- gate 报价粗估（训练=前向×~2.5，bs8×1024 提效未计）：SFT 8M tok ≈ 2.5–6h ≈ **¥50–120**；三栈全链含采样 ≈ **¥150–350**，低于 ¥600 熔断线；精确报价待 C4 tiny 冒烟
- 依赖一行恢复（重启即失效）：pip3 install tilelang modelscope "transformers==5.18.0"（本次仅补 transformers，tilelang 判决已定非必需）
