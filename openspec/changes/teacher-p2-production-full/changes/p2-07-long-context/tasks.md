# Tasks: p2-07-long-context [W2 · 依赖 p2-01+03+05；与 06/08/09/10 并发]

> 三级结构：`### 工作项` → 孙任务 checkbox。spec：`specs/long-context/spec.md`。

### 工作项 A needle 评测

- [x] A1 needle 生成器：多针、档位 8K/32K/128K/256K、seed 固定入 registry —— 验证：`python -m pytest tests/test_longctx.py -k needle -q`
- [x] A2 实测召回曲线 run：8K/32K/128K 三档 + 峰值内存 + 设备标注（Mac 上限如实） —— 验证：runs/ 含 longctx run-id，报告三档召回

### 工作项 B 前缀复用

- [x] B1a `serving/prefix_cache.py` 缓存核心：state 前缀 hash → past_kv 存取（RENDER_VERSION 入 hash）+ LRU 淘汰 —— 验证：`python -m pytest tests/test_longctx.py -k prefix_core -q`
- [x] B1b 同 state 多问增量前向（token 计数断言：第 2 问起前向 token 数=问题段长度） —— 验证：`python -m pytest tests/test_longctx.py -k prefix_incr -q`
- [x] B1c 复用/不复用结果一致性（argmax 同；漂移超限弃缓存重算） —— 验证：`python -m pytest tests/test_longctx.py -k prefix_parity -q`
- [x] B2 跨请求命中单测 —— 验证：`python -m pytest tests/test_longctx.py -k cross_hit -q`

### 工作项 C 滑窗与 1M 口径

- [x] C1 因果滑窗长文路径（复用 P1 attn_sw；W 与层排布可配；显存对比记录） —— 验证：`python -m pytest tests/test_longctx.py -k sliding -q -m mps`
- [ ] C2 1M 云端配置与运行说明（双列口径：measured/config） —— 验证：报告 1M 行格式为 `measured: — / config: ready`
