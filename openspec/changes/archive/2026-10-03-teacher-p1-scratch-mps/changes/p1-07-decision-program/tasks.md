# Tasks: p1-07-decision-program [W1 · 依赖 p1-01 + p1-03；stub 并行]

> 三级结构：`### 工作项` → 孙任务 checkbox。spec：`specs/decision-program/spec.md`。

### 工作项 A 渲染

- [x] A1 `render.py::from_systemone()`：请求解析 + schema 校验接入 —— 验证：`python -m pytest tests/test_decision.py -k parse -q`
- [x] A2 渲染函数（system 行 + Evidence/Question/Options 字母序）+ `RENDER_VERSION` —— 验证：`python -m pytest tests/test_decision.py -k render -q`
- [x] A3 黄金快照测试（逐字节）+ 确定性测试（两次渲染相同） —— 验证：`python -m pytest tests/test_decision.py -k snapshot -q`

### 工作项 B 读出与温度

- [x] B1 `readout.py`：末位 hidden × lm_head 字母行 → 选项 logits（stub tokenizer 单测先行） —— 验证：`python -m pytest tests/test_decision.py -k readout -q`
- [x] B2 右 padding 因果不变性单测（单独 vs 批内 pad，概率偏差 ≤1e-5；以 stub/mock 前向测试为主，若改用真模型则前置 p1-06 的 A1 已合并） —— 验证：`python -m pytest tests/test_decision.py -k pad_invariance -q`
- [x] B3 无生成循环静态断言（decision/ 源码禁 `.generate(`） —— 验证：`python -m pytest tests/test_decision.py -k no_generate -q`
- [x] B4 `temperature.py`：qtype 温度表应用（clamp [0.2,5]）+ 缺表 T=1.0 标 `uncalibrated` —— 验证：`python -m pytest tests/test_decision.py -k temperature -q`

### 工作项 C wide 票选

- [x] C1 `wide.py`：k>26 分组（≤25+残差槽）多轮票选，输出和=1；k=50 单测 —— 验证：`python -m pytest tests/test_decision.py -k wide -q`
