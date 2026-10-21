## ADDED Requirements

### Requirement: 配置化因果 decoder
`dmlaya/model.py` SHALL 定义 GPT 式因果 decoder（因果注意力 + RoPE + MLP pre-norm），全部超参（d/L/heads/ctx/vocab/rope_theta）来自 config；MUST 支持从 config.json 构造并导出（safetensors 权重 + config + 决策配置同目录布局），随机初始化种子入 config。

#### Scenario: 配置往返
- **WHEN** 以 `ModelConfig(d=128,L=2)` 构造模型并 save/load
- **THEN** 前向输出逐元素一致（同输入、eval 模式）

#### Scenario: 因果性属性
- **WHEN** 修改序列位置 i 之后的 token，检查位置 < i 的 hidden
- **THEN** hidden 严格不变（因果掩码正确）

### Requirement: 前向接口与"换脑"兼容
模型 SHALL 暴露 `forward(input_ids, attn_mask=None) -> last_hidden` 与 `lm_head` 权重引用；决策读出只允许使用这两者——二阶段替换为预训练 backbone 时接口不变。

#### Scenario: 决策程序仅依赖窄接口
- **WHEN** grep `dmlaya/decision/` 源码
- **THEN** 无任何对 model 内部属性（除 last_hidden 与 lm_head.weight）的引用

### Requirement: MPS 可运行
模型 SHALL 在 MPS 设备以 fp16 前向与反向跑通微批（吞吐档位写入配置注释）；CPU 路径为 CI 保底。

#### Scenario: MPS 前向反向
- **WHEN** 在 MPS 上对 batch=4, seq=256 跑一次 forward+backward
- **THEN** 无异常，loss 为有限值
