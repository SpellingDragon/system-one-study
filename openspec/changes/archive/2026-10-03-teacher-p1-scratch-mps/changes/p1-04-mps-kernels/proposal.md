# Proposal: p1-04-mps-kernels — 内核对拍（二级子变更）

> 父变更：teacher-p1-scratch-mps（W0，无前置依赖；M1 主战场，风险最前置）

## Why
M1 是一阶段的技术重心与最大不确定点（TileLang Metal 方言算子覆盖度未知）。对拍生命线 = torch 参考实现先行 + 方言冒烟最前置；教师版仅做 `_mps` 方言（学生版补 `_cuda`/`_asc` 时接口同源）。

## What Changes
- 新增 `dmlaya/testing/torch_ref/`：gemm / add_ln（残差 fp32）/ rope / attn_sw（因果滑窗）四参考实现 + 属性测试
- 新增 `dmlaya/kernels/`：`<op>_kernel.py`（入口/校验/分发）+ `<op>_mps.py`（`tilelang.metal.language`）四组两件套
- 新增 `dmlaya/kernels/backends.py`：MPS 探测 + tilelang/torch 双路径分发 + 编译失败回退 warning
- 新增 `tests/test_torch_ref.py`、`tests/test_tilelang_mps.py`（`@mps`）、`tests/test_backends.py`

## 边界与依赖（不耦合声明）
- 依赖：`tilelang`、torch；**零本仓依赖**（正交于模型层）。
- **被依赖方**：p1-10 eval（parity 轴）；p2-07 long-context（复用 attn_sw）。
- 接口面：`<op>_kernel.forward(...)` + `backends.resolve_device()`。
- **内核不阻主线**（父 design D3）：方言阻塞时 backends 回退 torch-MPS eager，主线照常。

## 验收
- spec scenarios 全过：`specs/mps-kernels/spec.md`（6 场景，含回退场景）
- 一级 DoD 关联项：④ fp16 err≤2e-2 且 argmax 一致
