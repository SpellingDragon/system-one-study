# run: 1010-p2-13-oncard-wave2-cube-and-gate-e079

- 假设：P1-1d 九参位序修复可消除 cube 面 507015；向量/标量面件真机数值可达 1e-7 级。
- 观察：addln 1.69e-05(有偏判别 7.65e-04)/gdn 6.35e-08/delta_fwd o=7.51e-08 s=6.03e-08 绿；gemm_l1 与 dw 仍 507015；readout rel=0.178 真偏差；train_step 因 ascend_env 探针用 SimtVF 判 eager 全旁路(grad_fn 缺失)。
- 结论：位序修复『必要不充分』，cube 崩溃收敛到 V1/V2 stride 单位域（本地考古，禁卡上试错）；新缺陷 G-gate=环境探针用 950 载体，须改 910B 合法最小件否则九件内核永被旁路；readout 属本地数值诊断。三件向量/递推数值真机证真达成，本窗使命完成即刻停卡。

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。
