## ADDED Requirements

### Requirement: 文本教师接口
`production/teachers/` SHALL 定义统一接口 `score_options(prompt, option_keys) -> {opt: p}`：文本教师取**末位字母 logprob** 归一化为选项分布；支持配置切换本地模型（transformers/MPS）。

#### Scenario: 末位字母分布
- **WHEN** 对 4 选项 prompt 调用文本教师
- **THEN** 返回 4 键分布，和=1，且与教师对各字母 token 的 softmax(logprob) 一致

### Requirement: 视觉教师接口
视觉教师 SHALL 接受 (image, prompt, option_keys) 产图文伪标分布；允许离线伪标包（parquet）替代在线 VLM。

#### Scenario: 离线伪标回放
- **WHEN** 以缓存包模式调用视觉教师
- **THEN** 同一请求返回与缓存写入时完全一致的分布（无在线推理）

### Requirement: 分布离线缓存
教师分布 SHALL 可按 (model_id, prompt_hash, option_keys_hash) 缓存为 parquet；三栈与多模态伪标 MUST 优先吃缓存；缓存 miss 才触发推理并回写。

#### Scenario: 二次调用零推理
- **WHEN** 同一批请求第二次调用（缓存命中）
- **THEN** 教师前向计数为 0，返回分布与首次一致

#### Scenario: 教师吞吐实测入档
- **WHEN** 冒烟 50 条未见请求
- **THEN** runs notes 记录教师 tok/s 与缓存前后耗时对比
