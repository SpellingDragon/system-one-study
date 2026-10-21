## ADDED Requirements

### Requirement: torch 参考实现先行
`dmlaya/testing/torch_ref/` SHALL 为 gemm、add_ln（残差 fp32）、rope、attn_sw（因果滑窗）各提供参考实现与属性测试；参考实现 MUST 先于 TileLang 内核合并（对拍生命线）。

#### Scenario: torch_ref 属性测试
- **WHEN** 运行 torch_ref 测试（CPU）
- **THEN** 全绿，且 attn_sw 掩码测试断言"窗口外未来 key 权重为 0"

### Requirement: TileLang MPS 方言算子
`dmlaya/kernels/` SHALL 以 `<op>_kernel.py`（入口/校验/分发）+ `<op>_mps.py`（`tilelang.metal.language` 方言）两件套实现四算子；教师版不实现 `_cuda`/`_asc`。`backends.py` SHALL 探测并仅分发 MPS 单后端，附 torch-MPS eager 回退路径。

#### Scenario: 逐算子对拍
- **WHEN** 在 MPS 设备上以随机输入对拍 tilelang_mps 与 torch_ref（fp16）
- **THEN** 逐元素 `max|err| ≤ 2e-2`

#### Scenario: 决策 argmax 一致
- **WHEN** 用对拍后算子组装的小模型对同一批决策输入前向
- **THEN** 选项 argmax 与 torch 路径 100% 一致

#### Scenario: 方言阻塞回退
- **WHEN** TileLang MPS 编译某算子失败（环境缺失/覆盖不全）
- **THEN** `backends.py` 回退 torch-MPS eager 并发出显式 warning；流程不中断，阻塞点可写入 run notes

### Requirement: M 动态与一次编译复用
GEMM 内核 SHALL 以 M 为动态符号，同一编译产物服务多 batch；序列维度 pad 到配置档位（ladder）。

#### Scenario: 多 batch 复用
- **WHEN** 以 M=1,7,16 三档复用同一 kernel 句柄
- **THEN** 无重复编译（编译计数不变），三档输出均通过误差门
