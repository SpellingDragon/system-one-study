# run: 1010-p2-13-p11d-cube-fix-wave-0c72

- 假设：cube 面 gemm_l1/dW 真机 aicore 507015 源于 §12 非转置 a/b 装填参数位序错（7 参 vs 官方 9 参），修结构位序即可消除非法访问。
- 观察：官方 LoadData2DL12L0ACal(mm_impl.h:35) 9 参 (dst,src,startIndex,repeatTimes,srcStride,dstGap,sid,transpose,inc)；trunk 只发 7 参且 addrCalMode 顶入 sid 位、sid 丢弃、transpose 缺。补齐+回穿 sid 后本地 A2-COMPILE-PASS + D-DW-COMPILE-PASS + E2E-CUBE-ONLY PASS，主链无回归；transpose 路 arity 本就 8 参正确。F 代理 delta fwd/bwd 全绿(CPU 5.76e-07)，G 代理 rig 八 target compile-PASS。
- 结论：P1-1d 结构修复落地+集成回归全绿+delta/rig 就绪，制品化 bundle_ondemand.sh 单窗收数已 push(3a0ada3)；**位序修复为高置信、但数值真机复验(gemm_l1/dW rel 是否脱困 507015 及 V2 单位)为唯一欠账，待一次 ≤8min 卡窗**。单位错(元素/块)与位序错是两回事——本修只证后者。

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。
