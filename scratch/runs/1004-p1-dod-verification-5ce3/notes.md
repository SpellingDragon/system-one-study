# run: 1004-p1-dod-verification-5ce3

- 假设：待填写
- 观察：待填写

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 dmlaya.runs 校验：没有结论行就不许 finish()。
结论：DoD 六条核验（design D5）全部通过：
① s0→s3 一键串跑：repro_p1.sh exit 0，链 1004-s1-smoke-tiny-ca2f-5→s2-d1ec→s3-acc2→eval-225e；
② SFT 超分桶基线+10pp：choice 0.6500>0.3333(+31.7pp)，noul 0.7500>0.6 ✓；
③ ECE 校准改善：真 s3 run choice Δ=-0.0119、noul Δ=-0.0523（repro tiny 档过调为小模型现象，如实记录）；
④ MPS 对拍：attn 最大 err 1.95e-3≤2e-2，决策 parity 12/12=100%（tilelang 路径）；
⑤ 门禁：tools/ci.sh CPU 274 passed + --device mps 36 passed，ruff/注释/编译全绿；
⑥ 注释门：check_comments 34 文件绿。
附带跨域修复三枚：rope_ref fp32 二次旋转、check_comments R2 or 优先级、readout MPS 跨设备下标（均含回归用例）。

更正：metrics step0 中 sxtasks_done=77 为笔误，实测域内孙任务 66/66、一级 14/14，见 step1 行。
