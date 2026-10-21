## ADDED Requirements

### Requirement: 预训练 backbone 装载
`production/assets.py` SHALL 载入 Qwen3-0.6B/0.8B 级 decoder backbone（ModelScope 优先，HF 兜底），以 fp16 适配 MPS；MUST 记录来源 repo/revision/commit 入 run config。

#### Scenario: 载入与换脑
- **WHEN** `load_backbone("qwen3-0.6b")` 后用一阶段 `decision/` 程序对 3 个 typed 问题决策
- **THEN** 输出合法选项分布（和=1），全程不触及一阶段 `model.py`

#### Scenario: 来源可溯
- **WHEN** 载入完成
- **THEN** run config 含 backbone repo id 与 revision 字段

### Requirement: 接缝三校验
装载时 SHALL 执行并通过：① `check_tokenizer`（26 字母单 token）；② `letter_rows` 随 backbone hidden_size 重建（读出行维度匹配）；③ chat template 思考关闭（渲染无思维链前缀，快照锁定）。任一失败 MUST 拒绝载入并报接缝名。

#### Scenario: 字母单 token 接缝
- **WHEN** backbone tokenizer 使 'A' 拆为多 token（构造 mock）
- **THEN** 载入抛 `SeamCheckError`，消息含 "字母单 token"

#### Scenario: 思考关闭快照
- **WHEN** 渲染同一 systemone 请求
- **THEN** prompt 与固化快照逐字节一致（无思考开启标记）
