# p2-13 · design（910B 移植主线）

## 技术判断（C1 证据链）

**2026-10-07 P0-1 回写（原判"卡点=头而非通路"过乐观，据 349 符号实勘改判）**：① 移植主体是 **codegen 向量侧压在 950 SIMT/SIMD 方言上**（C 类 250/72%）；本仓七件 kernel 全部以 T.SimtVF 为唯一向量载体（8/8 文件实证），910B 无 SIMT 硬件模型 → 算子**改道**（SIMD 化/标量化）而非加头。② AIC(cube) 通路 API 族（asc_init/asc_mmad/l12l0 系 14 符号，**asc_init 由 codegen 无条件发射每 kernel 首行**）静态不可判 = P0-2 vecadd 生死线探针（probe910b.sh P1-P10 已备好）。③ B 类 20（类型代理）已离线闭环（host_selfcheck 三配置 PASS、20 万样本 0 mismatch）。
- 代系卡点=头而非通路：bisheng 对 910B（dav-2201）生成的 `asc_copy_gm2l1_nd2nz/ascend_gemm_l1<...>` 等调用在 C1 崩溃源码中完整出现且属 910B 兼容面；仅 `c_api/asc_simd.h`+`simt_api/*`（950 SIMD/SIMT 指令）缺失。故**头桩 MVP（原地制，不建分支）**：纯新增 `src/c_api/asc_simd.h`+`src/simt_api/*.h` 五桩（bisheng 的 -I 扫 src/ 即命中，零改上游文件），AIC-only 用例先通、缺失符号随验随补。
- 七算子分层（P0-1 修正）：**无一纯 AIC**（全含 SimtVF 向量面）→ 首批=P0-2 判定 AIC 通路可用后，以 readout/gemm/dW/add_ln（向量面最薄、可标量化）先行改道；rope/attn_sw/GDN 向量面厚，依赖 SIMD 改道立项。
- 路线A 参照：950 实例一旦可得，同批用例双跑，差异即"代系税"实证（研究报告素材）。

## 文件清单
`ascend/kernels/`（接口不动）+ 主仓工作区原地新增（五桩已落 `src/c_api/`+`src/simt_api/`）+ `ascend/port910b/`（移植件与远端 pip 包同步脚本）

## 风险与回退
- 头桩实验失败（AIC 通路也隐式依赖新头）→ 界定最小改造集，如实入档，路线A 权重上调
- 950 实例不可得 → 参照系改用 TileKernels 官方基准数字（卡面）+ 本地 CPU 对拍
- vehicle 永远可跑：torch_npu 底座已验，研究不阻塞训练实验

## 主战场本地化（2026-10-08，三轮上卡摩擦税所促）
- **关键事实**：bisheng 将 .asc 编译为 .o 属交叉编译，只需 CANN 工具链不需 NPU 在线；CANN 8.5.2 有 aarch64-Linux 公开版，Mac Docker 原生 aarch64 可跑 → **编译判决（方言可用性、头/类型/intrinsic 缺失、vecadd 生死）全部本地零边际成本迭代**；卡窗口仅保留运行时验证（数值/tflops，≤10min）。
- **制品化纪律**：上卡动作一律单文件自包含 bundle（幂等：装依赖→铺树→跑→输出 verdict JSON），禁止临场拼命令链。


## 真机数值首战判读（2026-10-10，run p2-13-oncard-wave1）
- **D-num1**：向量面（标量化+表加载+软件数学件）真机数值全绿（1e-7~1e-8 级）→ P1-1 向量侧改道方案**数值层闭合**。
- **D-num2**：cube 面 gemm_l1/dW 同型 aicore 507015（非法访问型崩溃）→ V1/V2（load_cbuf_to_ca 装填位段）从"数值待证"升格为"致命缺陷"；修复路径=官方 mm 链构造点逐参对照（P1-1d），本地修完单窗复验——禁止卡上试错（R20）。
- **D-num3**：gemm direct（UB 直读 mad）rel=nan 维持 backlog（l1 路修好后无生产必要性）。

## 接口 home 合流缺口（2026-10-10，P2 预置侦察所生）
- **D-int1**：P2 三栈联调的真实前置不是 kernel 数值，而是**接口 home `ascend/kernels/*_asc.py` 的 910B 编译合流**。证据：九件 `*_asc.py`（除 lora）的 ascend 路径普遍用 `SimtVF/T.Parallel`（950 载体），`gdn_asc.py:112` 自陈"target=ascend 本地不实编"——仅验过 target=cpu 语义；实测 `T.SimtVF` 件在 patched 910B tilelang 下 **COMPILE-FAIL**（"Ascend device compilation failed"）。与 P0-1（C 类 72% 压 950 SIMT 方言）/P1-1（SIMT 面 0 目录）既有实测吻合。**裁决**：把 attempts/ 已验证的 910B 可编形态（B/C/D/E/F：标量化+表加载+§11 数学件+AIC）逐个重写进接口 home 的 `*_ascend_impl`（接口 `forward/backward/plan/run` 保持稳定，因 attempts 是一次性移植件、`_asc.py` 才是产品），逐件 `target=ascend` 出 .o 判决（Docker 判决环境零卡成本，R20 兑现）。**被否方案**：直接上卡跑训练步——首个 SimtVF 件即编不出，卡窗纯烧钱；接口 home 路由到 attempts 脚本——attempts 是散件非产品层，层次倒挂。
- **D-int2**：合流时两处口径升级——(a) gdn 反向 `BWD_STATUS=partial` 被 F 代理超越（attempts/F 已出 910B 标量六梯反向 + CPU-golden 5.76e-07），撤 partial 接入接口；(b) conv 位点（R7 崩点）接口 home 现标"在上游之外"、无入口，需新增 conv 入口（c_gdn_conv 的 910B 形态）。

## cube 两轨制（2026-10-10 wave2 判读所生）
- **D-cube1**：wave2 实证 P1-1d 九参位序修复『必要不充分』（gemm_l1/dW 仍 507015）→ cube intrinsic 路的 V1/V2（stride 单位）与 GM→L1 分形布局（NZ padFuncMode）为深水区，**不再作为 P2 数值前置**。两轨制：**correctness-first**＝gemm/dW 的 ascend 实现改可移植标量体（wave2 已证该形态真机 1e-7 级可靠：delta/addln/gdn；双方言可编→cpu 跑数值 oracle、ascend 编译判决，本地闭环），P2 训练步数值里程碑与之解耦；**performance-track**＝intrinsic fractal 考古（§12 与官方 dav_c220 Matmul 构造链逐参对照）继续独立推进，成果以证据文档+可选 §12 修正交付，不阻塞 P2。被否：继续押注 intrinsic 单轨——三窗两崩已证其不可预期，P2 里程碑不该吊在最深研究问题上。

## D-cube2（2026-10-11 第二窗实测后新增）性能路唯一出路是 cube；标量路定位为正确性/研究车
**实测依据**（`runs/1011-p2-13-oncard-window2-throughput` §④）：标量路 `linear 512³ fp16 p50=294.3ms(0.912 TFLOPS)`、`attn_sw 1879.3ms`、`gdn_delta fwd/bwd 293/431ms`；外推（标注为**推断**）⇒ 24 层模型分钟级/步。
**决策**：① 正确性验收与回归继续走标量路（已真机 exit=0）；② §12/cube 路**不得降级**（原 review 预留的"慢 50× 则降级"分支被实测三个数量级推翻，投入价值抬高）；③ 过渡期吞吐靠第二栈 `torch_npu` 原生算子兜底，与"脱离 CUDA 探索"不冲突——底座保吞吐、逐件靠 cube 替换；④ **对照腿数据一律标 `track` 且物理合理性未核者不作结论**（`intrinsic 3366 TFLOPS` 即属此类，已作废）。
**替代方案与未采用理由**：放弃 cube 只用标量路——分钟级/步使训练不可经济；放弃 torch_npu 底座全押 cube——cube 路当前 507015 未闭，会同时失去正确性与吞吐，训练无法推进。
