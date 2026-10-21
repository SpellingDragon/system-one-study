## ADDED Requirements

### Requirement: 自训 BPE 训练入口
`learning/s0_tokenizer.py` + `dmlaya/lang/` SHALL 从中英混排语料训练 BPE（词表 8k–32k 可配），MUST 包含特殊 token（`<|pad|>`、`<|endoftext|>`），产物为 tokenizer 目录（vocab+merges+配置），全过程不加载任何第三方预训练词表。

#### Scenario: 中英编解码往返
- **WHEN** 用训练产物对 `"你好世界 hello world 123"` 编码再解码
- **THEN** 字符串完全还原（允许的清洗规则显式记录于配置）

#### Scenario: 语料流式训练
- **WHEN** 以 `--stream-limit N` 传入受限语料档
- **WHEN** 训练在 CPU 上 30 分钟内完成并产出合法 tokenizer 目录

### Requirement: 26 字母单 token 强校验
`check_tokenizer()` SHALL 断言 A–Z/a–z 各 26 字母在词表中均为**单 token**；任何字母被拆成多 token 即校验失败。

#### Scenario: 全字母单 token
- **WHEN** 对自训产物跑 `check_tokenizer`
- **THEN** 返回 52 个字母的 token id 映射，全部长度为 1

#### Scenario: 污染词表被拒
- **WHEN** 构造一个把 "AB" 合并成单 token 且导致 'A' 缺失的词表
- **THEN** `check_tokenizer` 抛出 `TokenizerCheckError` 并列出缺失字母
