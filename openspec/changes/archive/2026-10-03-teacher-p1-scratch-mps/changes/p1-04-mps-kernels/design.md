# Design: p1-04-mps-kernels

## 技术要点
- **两件套范式**（TileKernels 范式裁剪为 MPS-only）：`_kernel.py` 不写 TileLang 只做校验/分配/分发；`_mps.py` 用 `tilelang.metal.language`（simdgroup/cooperative tensor）。
- 四算子要点：gemm（`C=act(A@Wᵀ+b)`，fp32 累加、`transpose_B`、**M 动态符号**一次编译多 batch 复用）；add_ln（**残差流保 fp32**——laya 实测 bf16 残差漂移）；rope（packed-qkv 就地 rotate-half，cos/sin 分表）；attn_sw（**因果**滑窗：仅回看 W key——⚠️ `refs/laya/tl_kernels.py::attn_kernel` 是双向，掩码改单向，方向勿抄反）。
- backends 探测顺序仅 `mps`（教师版裁剪）；`/dev/` 探测逻辑留给学生版扩展位（代码注释标注）。
- 序列 pad 到固定 ladder（线性核按长度编译）——与 p1-07 右 padding 不变性约定咬合。
- 蓝本：`refs/laya/tl_kernels.py`（GEMM/LN/RoPE 已验证，attn 改因果）；双范式参照 `refs/TileKernels/tile_kernels/moe/topk_gate_{cuda,asc}.py`；DSL 入门 `tilelang/examples/{quickstart,gemm,flash_attention}`。

## 文件清单
| 文件 | 职责 |
|---|---|
| `dmlaya/testing/torch_ref/{gemm,add_ln,rope,attn_sw}_ref.py` | 参考实现 |
| `dmlaya/kernels/{gemm,add_ln,rope,attn_sw}_{kernel,mps}.py` | 两件套 ×4 |
| `dmlaya/kernels/backends.py` | 探测/分发/回退 |
| `tests/` 三件 | 属性/对拍/回退 |

## 风险与回退
- [Metal 方言缺 attn_sw 所需原语] → W0 首孙任务即方言冒烟（elementwise 先行）；失败即触发父 design D3 回退：`backends.py` 走 torch-MPS eager + warning，阻塞点记 runs notes——**主线不停**。
- [fp16 精度漂移超 2e-2] → 累加器升 fp32；rope 的 cos/sin 预计算 fp32 存 fp16 用。
