# Tasks: p1-04-mps-kernels [W0 · 无依赖；风险最前置]

> 三级结构：`### 工作项` → 孙任务 checkbox。spec：`specs/mps-kernels/spec.md`。
> 铁律：torch_ref 先行（4A 先合并），TileLang 后对拍；方言阻塞走回退不阻主线。

### 工作项 A torch 参考实现（先行，CPU 可测）

- [x] A1 `dmlaya/testing/torch_ref/gemm_ref.py` + 属性测试（形状/累加精度） —— 验证：`python -m pytest tests/test_torch_ref.py -k gemm -q`
- [x] A2 `add_ln_ref.py`（残差流 fp32）+ 属性测试 —— 验证：`python -m pytest tests/test_torch_ref.py -k add_ln -q`
- [x] A3 `rope_ref.py`（rotate-half，cos/sin 分表）+ 属性测试 —— 验证：`python -m pytest tests/test_torch_ref.py -k rope -q`
- [x] A4 `attn_sw_ref.py`（因果滑窗：仅回看 W key）+ 测试断言"未来位置权重=0、窗口外=0" —— 验证：`python -m pytest tests/test_torch_ref.py -k attn_sw -q`

### 工作项 B TileLang MPS pilot（最险先行）

- [x] B1 方言冒烟：用 `tilelang.metal.language` 编译最小 elementwise kernel 并在 MPS 上运行；不可用则触发回退路径并记录 runs notes —— 验证：`python -m pytest tests/test_tilelang_mps.py -k smoke -q -m mps`
- [x] B2 `kernels/gemm_kernel.py + gemm_mps.py`（M 动态符号、一次编译多 batch 复用）+ 对拍 fp16 `max|err|≤2e-2` —— 验证：`python -m pytest tests/test_tilelang_mps.py -k gemm -q -m mps`
- [x] B3 `add_ln_kernel.py + add_ln_mps.py` + 对拍 —— 验证：`python -m pytest tests/test_tilelang_mps.py -k add_ln -q -m mps`
- [x] B4 `rope_kernel.py + rope_mps.py` + 对拍 —— 验证：`python -m pytest tests/test_tilelang_mps.py -k rope -q -m mps`
- [x] B5a `attn_sw_kernel.py + attn_sw_mps.py` 因果滑窗 kernel 实现（参考 `refs/laya/tl_kernels.py` 掩码改单向：仅回看 W key，方向勿抄反） —— 验证：`python -m pytest tests/test_tilelang_mps.py -k attn_compile -q -m mps`
- [x] B5b 对拍 torch_ref：fp16 `max|err|≤2e-2`，多档 W（64/128/256）× 多档 seq —— 验证：`python -m pytest tests/test_tilelang_mps.py -k attn_parity -q -m mps`
- [x] B5c 组装小前向后决策 argmax 与 torch 路径 100% 一致 —— 验证：`python -m pytest tests/test_tilelang_mps.py -k attn_argmax -q -m mps`

### 工作项 C backends 分发

- [x] C1 `dmlaya/kernels/backends.py`：MPS 探测（torch.backends.mps.is_available）+ tilelang/torch 双路径分发 + 编译失败回退 warning —— 验证：`python -m pytest tests/test_backends.py -q`（含阻塞回退场景 mock）
