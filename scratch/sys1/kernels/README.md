# sys1/kernels/ — TileLang 跨后端算子（一阶段 M1 主战场）

> 目的：决策模型的**计算主体**用 TileLang 手写，并以三件套范式跑通 **CUDA(参考) / MPS / 昇腾**。数值对拍是验收生命线。

## 三件套约定（源自 TileKernels，扩展 `_mps`）

```
<op>_kernel.py   # torch 面向外入口：校验/分配输出/按 backends.py 分发。不写 TileLang。
<op>_cuda.py     # 参考实现（CUDA 方言 tilelang.language）
<op>_mps.py      # Metal 方言（tilelang.metal.language：simdgroup/cooperative tensor）
<op>_asc.py      # 昇腾方言（tilelang.ascend.language：AIC/AIV、T.Stage、SIMD S.*）
```
参考：`refs/TileKernels/tile_kernels/moe/topk_gate_{kernel,cuda,asc}.py`（同算法双范式的样板）。

**教师版只交付 `_kernel.py` + `_mps.py`（两件套）**：回退用的 torch 写法一律放在 `_kernel.py` 的
`_eager()` 里，`_mps.py` 只留方言——学生版补 `_cuda.py` 时才能拿 `_mps.py` 逐段机械对照。

## 算子清单与要点

| 算子 | 说明 | 直接参考（refs/） |
|---|---|---|
| `gemm_*` | `C=act(A@Wᵀ+b)`，fp32 累加、`transpose_B`、`M` 动态 | **`laya/laya/tl_kernels.py::gemm_kernel`**（改 dtype/参数即可）；`tilelang/examples/gemm/` |
| `add_ln_*` | LN+残差融合，**残差流保 fp32**（laya 实测 bf16 残差会漂移） | `tl_kernels.py::add_ln_kernel`；注意 laya 用 LayerNorm 而非 RMSNorm |
| `rope_*` | packed-qkv 就地旋转（rotate-half），cos/sin 分表 | `tl_kernels.py::rope_kernel` |
| `attn_sw_*` | **因果滑窗** flash-attention（窗口 W：仅回看 W 个 key） | ⚠️ `tl_kernels.py::attn_kernel` 是**双向**滑窗——掩码改因果即为其一半工作量，方向别抄反 |
| `readout_gemm_*`（可选） | 末位 hidden × 26 字母行的小矩阵乘，可先行 torch 回退 | decision/README |

`backends.py`：探测顺序 `cuda → mps → npu`（昇腾判据示例：`os.path.exists('/dev/davinci_manager')`，参考 `refs/TileKernels/tile_kernels/config.py`），并收敛设备能力（核数/显存/dtype 白名单：MPS bf16 受限→fp16 回退）。
教师版实际裁成 `mps → cpu`，并额外提供 `get_compiled()/compile_count()/compiled_keys()/blockers()`
四个量，用来**自证**"一次编译多批复用"与"回退看得见"。

## 对拍纪律（进 `tests/`，按设备 skip）

- 先写 `testing/torch_ref/<op>_ref.py`，再写 kernel；逐算子对拍。
- bf16/fp16 下 `max|err| ≤ 2e-2`，且**决策 argmax 100% 一致**（M1 验收）。
- `@tilelang.jit` 一次编译多 batch 复用：`M` 为动态符号；序列长 pad 到固定 ladder（线性核按长度编译，参考 StartLux `PAD_LENGTHS` 思路）。
- 对拍用例必须能区分"走了内核"和"走了回退"：本域用 `blockers()` 为空 + `compiled_keys()` 含该算子
  键这两条一起卡，**不许把回退伪装成内核通过**。

## Metal 方言实测约束（tilelang 0.1.15 / macOS 14.7.2 / Apple M3 Pro，B1~B5 全部跑通）

这六条是踩出来的硬事实，写内核前先读，别浪费一轮编译：

1. **发射参数**：`target="metal"` 必须配 `execution_backend="tvm_ffi"`。默认的 torch adapter 在动态
   尺寸下把符号化网格当 python list 递给 C++，报
   `Unable to cast Python instance of type <class 'list'> to C++ type 'std::vector<unsigned long long>'`。
2. **累加器不能是 fragment**：对 `T.gemm` 的 fragment 累加器做逐元素加减，代码生成期直接失败——
   `Check failed: (dtype == Float(16) || Float(32) || BFloat(16)) ... but got float32x4`，MSL 侧则是
   `invalid operands to binary expression ('metal::simdgroup_float8x8' and 'float')`。做法：共享内存
   放 fp32 累加器，epilogue 在 shared 上做，出口 `T.copy` 自动转型。
3. **没有归约原语**：`T.reduce_sum`/`T.warp_reduce_*` 一律报
   `tl.reduce requires a target-specific implementation, but no reduce implementation is registered
   for {"kind":"metal"}`。行内求和/求最大值只能 `T.get_thread_binding()` + `T.serial` 手写线程级串行
   归约（`get_thread_binding()` 必须在 `with T.Kernel(...)` 帧内调用）。
4. **动态长度的边界只能靠 `T.copy` 的区域切片**：切片写法会被自动加谓词（生成的 MSL 里能看到
   `... < arg.m[0]`，哨兵实验证实不越界，实测 m=7/16/40 误差 0.0）；而在 `T.Parallel` 里写语句级
   `if by*bm+i < m:` 守卫**结果整体偏离**（实测误差 1.9e+01 ~ 2.6e+01，连不越界的 m=16 也错）。
   规避手段按优先级：切片 > 一元素一块（rope）> 入口把行数补齐到块倍数（add_ln）。
5. **threadgroup 内存约 32KB/块**：想在一块里驻留整行 fp32（dim≥512、多行）会超；`add_ln` 因此改成
   三趟全局重读而不是"搬进 shared 再合并"。
6. **别指望 cooperative tensor 快路径**：`tilelang.metal.language` 比 `tilelang.language` 多 8 个
   metal 独有符号（4 个 `simdgroup_*` + 4 个 `cooperative_tensor_*`），但本机
   `check_metal_availability()=True` 而 `check_metal4_availability()=False`——`cooperative_tensor_*`
   要求 metal4，这条直连全局内存的快路径在本机拿不到准入。本域四算子最终一个都没用到它们：
   最朴素的 shared + `T.gemm`（`transpose_B` / `clear_accum=False` 累加）就已经过了数值门。
