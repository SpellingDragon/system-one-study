## ADDED Requirements

### Requirement: 前缀/状态复用
`serving/prefix_cache.py` SHALL 支持同 state 多问与同 state 跨请求的 KV/前缀复用；命中时前向成本与 state 渲染长度解耦（增量前向）。

#### Scenario: 同 state 多问复用
- **WHEN** 同一 state 连发 3 问
- **THEN** 第 2/3 问的前向 token 数仅为增量（计数器断言），答案与不复用路径一致（argmax 同）

#### Scenario: 跨请求命中
- **WHEN** 不同请求携带相同 state 前缀
- **THEN** 缓存命中报告 >0，结果一致

### Requirement: 因果滑窗长文路径
长文推理 SHALL 提供因果滑窗注意力路径（复用 P1 `attn_sw` 与 layers 排布配置）；窗口 W 与层排布可配。

#### Scenario: 滑窗正确性
- **WHEN** 以 W=128 对合成序列对比全注意力（短序列下两者应近似）
- **THEN** 选项 argmax 一致，长序列下显存峰值低于全注意力（实测记录）

### Requirement: needle 评测与档位
`sys1/eval/longctx.py` SHALL 生成合成 needle（多针、长度档 8K/32K/128K/256K，seed 固定）并报告召回曲线；Mac 实测上限档与 1M 云端配置双列呈现，不虚报。

#### Scenario: 召回随档位输出
- **WHEN** 跑 8K/32K/128K 三档
- **THEN** 报告三档召回率 + 各档峰值内存 + 实测设备标注

#### Scenario: 1M 双列口径
- **WHEN** 查看长文报告
- **THEN** 1M 行为 `measured: — / config: ready` 或实测值，无混淆
