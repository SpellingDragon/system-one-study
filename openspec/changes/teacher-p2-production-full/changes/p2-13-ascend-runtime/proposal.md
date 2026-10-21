# p2-13 · TileLang 昇腾算子优化（主线域，2026-10-07 目标重定向）

## Why
用户令：**目标是基于 TileLang 做训练/推理算子优化，探索脱离 CUDA 生态**。C1 实勘（run de15）定谳三事实：① tilelang ascend 后端 = 2026-09-30 一次性开源的 **950 专用**实现（主仓单 commit e5a02f9a，379 文件，无 910B 历史线）；② 卡点集中在 950 专属 SIMT 头（c_api/asc_simd.h、simt_api/*），而**生成的 AIC 通路代码（gm2l1/l0c/cube gemm 等 intrinsic）代系通用**；③ 910B4+CANN8.5.2 上 torch_npu 底座健康（978 tok/s 前向），三栈 vehicle 随时可跑。

## What Changes（两条研究路线并行）
- **路线 B（主线·910B 后端移植·原地制）**：不建分支，**主仓工作区原地修改**——首验"头桩实验"已落子：纯新增 `src/c_api/asc_simd.h`+`src/simt_api/{bf16,fp16,fp8,simt}.h` 五桩（零改上游文件），AIC-only 用例上卡即验；成立则七算子按 AIC 优先逐件移植，桩随验随补 910B 等价物。
- **路线 A（已废·2026-10-07 用户裁定 950 不可得）**：参照系降级为 TileKernels 官方卡面基准 + 本地 CPU 对拍（fp32 真值不变）；环境锚定 **ModelArts 镜像 `pytorch_2.7.1-cann_8.5.2-py_3.12-hce_2.0.2512-aarch64-snt9b`**（CANN 8.5.2 固定为移植目标线）。
- **三栈 vehicle**：训练先 torch_npu 底座（D6 预授权回退已启用），算子件成熟一个换一个（渐进替换，非全有全无）——G9 叙事即此过程本身。

## 边界与依赖
- 依赖：本仓 tilelang 主仓（**原地修改，不建分支**）、C1 实勘结论、ModelArts snt9b 镜像环境（950 不可得，路线A 已废）
- 被依赖方：p2-05/06/11（算子逐件替换的消费者）、p2-12（研究报告主章）
- 接口面：`ascend/kernels/` 七件接口不变（cpu 对拍资产为 day-1 活资产，保留）
- 禁止：不碰 `decision/`；主仓改动限于**原地新增桩/移植件**（零改上游既有文件），且须过 B1 上卡实证方可 commit 进主仓历史
