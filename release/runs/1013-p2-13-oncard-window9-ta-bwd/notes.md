# 窗九台账：TA gdn_bwd 真机首验 + P1-1zi 一旗（P1-1zh ② 之窗九判决）

- 窗：`window9_card.sh`（port910b 根）+ `run_window9_local.sh`；TA 通道复用窗五 `ta_pkg`（sha256 d5607ec6…，34.8MB）
- 实例：ModelArts 910B（CANN 8.5.2，torch_npu 2.7.1.post2，py3.12）；实例 2026-10-13 上午启动
- 失败预案映射：[1]=F1 卡上编译差异  [2]=F2 L0 数值  [3]=P1-1zi 一旗

## a 轮（12:25-12:29）：F1 排除 + F2 坐实（芯片级归因）

- W9-COMPILE fp16/fp32 双 OK（src=24363/24365，比容器 23439/23441 略大＝bisheng 8.5.2 vs 8.5.0 微差）——**卡上编译面无虞，F1 出局**（窗五正面证据再添一条）
- L0 六 case 全数同因失败：`ACL stream synchronize failed, error code:507015` → 原始芯片异常（w9_l0.log 首条 EZ9999）：
  ```
  fftsplus aivector error, core id 24, vec error info: 0x400000056, mte error info: 0x6000077
  errorStr: The UB address accessed by the VEC instruction is not aligned. subErrType:4
  ```
- **归因：UB 运行期偏移标量访存**（`g_row[t]`/`beta_row[t]`/`scol_ub[0]` 家族——per-token 件新形态；窗五 fwd 六件全绿正因 chunk 形态无此用法）
- golden adjoint 自检卡上复跑 rel=5.999e-10 PASS（CPU 段不受影响）
- 一旗腿 SKIP（新实例无 /tmp/sys1，脚本缺陷）

## b 轮（12:38-12:45）：修复第一枪翻车（编译期）+ 一旗腿环境自污染定案

- 第一枪修法（alloc_var + GM 直读）上卡即编译失败，四枪诊断链（容器复现，全定案）：
  1. `TVMError: only accept int or float @ gdn_bwd.py:134`——TA `alloc_var(dtype, init)` 参数形态（accum_dtype="float" + positional PrimExpr init 不合）
  2. `Unresolved call Op(tir.exp)`——标量上下文 `T.exp(G[b,h,t])` 无 codegen 注册（exp 只在 tile API/向量上下文有实现；fork 的 `dec_ub[i]=T.exp(G[h,t,i])` 是 UB 元素级形态）
  3. `LowerParallelToVector` 拒广播赋值：`bts_ub[jj] = Beta[b,h,t]`（jj 无关 → copy 分解失败）
  4. `ascend_combinecv.cc:1217`：**GM 标量 store 必须显式 `T.Scope("C"/"V")`**（CombineCV 无法推断写者）
- 一旗腿 `ModuleNotFoundError: tilelang` 定案：主腿 [0] 段 `PYTHONPATH=$TA_ROOT` 泄漏进 [3] 段 card_run——tilelang 探测被骗（TLROOT 解析到 TA 包）、overlay 打进 TA 解包目录、pip 安装被跳过。修法=`env -u PYTHONPATH -u LD_LIBRARY_PATH -u TA_ROOT` 隔离引导
- 顺带定案：卡上 clone HEAD=048dfa1（origin/main 最新的 docs 系列提交，c59e47f 是其祖先，kernel 代码未动——虚惊）

## 终态修法（tile API 面，全签名实证合法）

每 trip：`T.copy(G[b,h,t:t+1], g_one[1])` region copy + `T.tile.exp(g_one, g_one)` + `T.copy(Beta…, b_one)` + 系数乘全走 `T.tile.mul(dst, src, g_one[0]/b_one[0])`（ascend_tile.py:1152 src1 签名接受 BufferLoad/PrimExpr）+ GM 标量 store 包 `T.Scope("V")`。**UB 运行期偏移标量访存与标量 ctx exp 全数消灭**。代价：每 trip 2 个块长 1 GM→UB copy + 2 GM 标量 store（v1 正确性优先；chunked v2 无 per-token 标量）
- 容器复验：**2/2 OK**（fp16 src=23763 / fp32 src=23915，探针 `.tmp_a1/w9b_compile_probe.py` + P5 shim 配方）

## c 轮（12:52-12:59）：tile API 形态上卡，507015 复发 ⇒ 进入探针二分

- 编译 fp16/fp32 双 OK（src=24419/24571）；L0 六 case 仍全 507015——嫌疑形态已灭而异常依旧，转入系统探针
- 一旗腿（env 修复后）：pip fork tilelang 已装、overlay OK，但 `torch_npu` autoload 失败（`Failed to load the backend extension`，run_delta_card import 链）——待下轮看完整前段

## d 轮（13:00-14:30）：19 数据点黑盒二分（探针 v1–v11，每轮多假设打包）

判据矩阵（全部真机判决，1,1,1,8,8 最小形；`PASS`=干净返回，`FAIL`=507015 aicore）：

| # | 形态 | 判决 |
|---|---|---|
| 1 | `[1]` region copy GM→UB | PASS |
| 2 | `tile.exp([1])` | PASS |
| 3 | `reduce_sum→[1]` | PASS |
| 4 | `tile.mul` 标量参数位（BufferLoad src1） | PASS |
| 5 | 2D reduce dim=1/0（平铺 kernel） | PASS |
| 6 | `Scope("V")` GM 标量 store | PASS |
| 7 | serial 64-trip copy+exp+mul 密度 | PASS |
| 8 | fp16 staging 1D cast copy | PASS |
| 9 | shape/dtype/workspace/多输出配置面（v4 三翻） | 全无关 |
| 10 | 2D 读 + 单跨维 add（`lam+do[j]`） | PASS |
| 11 | 纯外积（双 1D 源） | PASS |
| 12 | 2D 读×单跨维 mul（i 广播） | PASS |
| 13 | **i 广播配 dim=0** | FAIL |
| 14 | **j 广播配 dim=1** | FAIL |
| 15 | **1D[j]×2D 裸乘 + dim0**（dst/store/复合变体 5 种） | 全 FAIL |
| 16 | `(2D+2D)×1D[j]` 复合式 + dim0（L12，**唯一免疫样本**） | PASS |
| 17 | 同式括号/加 0.0/镜像变体（免疫不可复制） | 全 FAIL |
| 18 | `tile.broadcast` 官方通道展开 + 2D×2D mul + dim0 | FAIL |
| 19 | **serial-j 逐行累加（fork 真机全绿同构形态）** | FAIL |

- 结论：雷不在 kernel 写法层——**fork 全绿形态在 TA 上同样炸**，属 TA 编译器缺陷域
- 生成码 diff 取证（L12 PASS vs L18 FAIL，容器 dump，artifacts/w9_gen_L*.cpp）：TA 对 j 广播自动插入 `tl::ascend::Broadcast<float,2,1>`（额外 64 元素 UB 缓冲 → **全体后续 buffer 偏移漂移**）+ `PipeBarrier<PIPE_V>` 差异；L12/L18 的 UB 布局不同（dv_ub/scol_ub 偏移后移）
- 嫌疑面：`LowerParallelToVector` 的 Broadcast 展开 × UB memory planning 组合
- 附带定案：真件 507015 与 P5 原版 `g_row[t]` 假说无关（该形态已灭仍炸）；gdn_bwd 四轮修法（alloc_var/GM 直读/tile API/五处拆步）在卡上全部无效但**编译面教训全部有效**（见 §附）

## 裁决（用户 ok，2026-10-13 14:3x）

- **A：上游 issue 立即起草**（不发布，草稿存 `reconcile/UPSTREAM_ISSUE_910b_vec_unaligned.md`：最小复现 L18 腿 + 判据矩阵 + 三份生成码）
- **B：chunked 二版立项**（design §1.3.3；本波教训直入二版设计约束：禁 j 收缩、j 广播走 GEMM 域）
- C（混合管线）挂起，视训练紧迫度再启
- 实例可停；探针族+生成码已固化 artifacts/

## 附：窗九系列工程教训（待回写 checklist）

1. R20 再犯一次：b 轮修法未经容器编译门直接上卡（可用时窗 4 分钟代价换回纪律）——编译门必先行
2. TA 环境与 fork 环境在同一卡侧脚本共存的**env 隔离律**：跨世界调用必须 `env -u PYTHONPATH -u LD_LIBRARY_PATH`
3. TA 910B SIMD 面的标量纪律（供 tilelang-op-develop 技能回写）：per-token 标量系数合法面＝`[1]` region copy + tile exp/mul 标量参数位；禁 UB 运行期偏移读、禁标量 ctx `T.exp`、禁 Parallel 广播赋值、GM 标量 store 须 V Scope
