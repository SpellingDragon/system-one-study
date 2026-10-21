# Tasks: p1-06-decoder-model [W1 · 依赖 p1-03 词表常数]

> 三级结构：`### 工作项` → 孙任务 checkbox。spec：`specs/decoder-model/spec.md`。

### 工作项 A 模型定义

- [x] A1 `dmlaya/model.py`：ModelConfig（d/L/heads/ctx/vocab/rope_theta/seed）+ pre-norm 因果注意力 + RoPE + MLP —— 验证：`python -m pytest tests/test_model.py -k config -q`
- [x] A2 save/load：safetensors + config.json + decision_config.json 同目录布局 —— 验证：`python -m pytest tests/test_model.py -k roundtrip -q`
- [x] A3 因果性属性单测（改位置 i 之后 token，位置 <i 的 hidden 不变） —— 验证：`python -m pytest tests/test_model.py -k causal -q`

### 工作项 B 接口与设备

- [x] B1 窄接口固化：`forward(input_ids, attn_mask) -> last_hidden` + `lm_head.weight` 引用；grep 断言 decision/ 不引用内部属性 —— 验证：`python -m pytest tests/test_model.py -k interface -q`
- [x] B2 MPS fp16 前向+反向冒烟（batch=4, seq=256） —— 验证：`python -m pytest tests/test_model.py -k mps -q -m mps`

> 执行记录（2026-10-04）：5/5 绿。`dmlaya/model.py` 385 行（check_comments 过、ruff 过），`tests/test_model.py` 21 用例（本文件全绿；全量 `pytest tests/ -m "not mps and not cuda and not npu"` 126 passed 无回归）。
> 因果性口径：改位置 i 之后 token 后，位置 ≤i 用 `torch.equal` 逐比特断言（非容差），另加两条阳性对照（改第 i 位必须动第 i 位、改首字必须动末位）+ 右 padding 键屏蔽（批内补洞 max|Δ|<1e-5）。
> 跨域发现：p1-04 `torch_ref/rope_ref.py` 在 **fp32 输入**下有就地别名缺陷（`out=qkv.clone()` 后 `part=out[:,slot].to(fp32)` 是视图，写回前半再用它算后半 → 后半被转两次）；fp16 档 `.to(fp32)` 产生真拷贝故未暴露，`kernels/rope_kernel.py` 先 clone snapshot 无此问题。本域按教科书口径实现，对拍只在 fp16 档做（见 `test_config_rope_theta_*`），未改动他域文件。
> 未做（越界）：父 design 提到的 `configs/` 冒烟档注释——白名单只允许 model.py/test_model.py，规格已写入 `ModelConfig` 文档字符串。
