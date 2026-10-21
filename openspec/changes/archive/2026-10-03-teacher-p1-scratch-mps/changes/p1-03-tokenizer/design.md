# Design: p1-03-tokenizer

## 技术要点
- 用 `tokenizers` 库的 BPE trainer（非手写 BPE——教师版把工时花在字母约束与契约，不在重造轮子；学生版若手写不阻拦）。
- **字母约束的实现**：initial alphabet 显式注入 26×2 字母 + 数字 + 常见中英标点，保证单 token；`check_tokenizer` 是唯一裁决（不信训练配置信校验结果）。
- 特殊 token：`<|pad|>`（id 固定 0 方便 pad 掩码）、`<|endoftext|>`（文档分隔）；二阶段将再加 image/video/mask token——本域预留 `special_tokens` 列表参数。
- 训练语料中英混排；清洗规则（去 URL/控制字符）显式入配置，保证编解码往返可推理。
- 参考落点：`llms-from-scratch-cn/Codes/ch02`（BPE 教学）、`StartLux jevfmt.py::check_tokenizer`（校验思路）。

## 文件清单
| 文件 | 职责 |
|---|---|
| `dmlaya/lang/__init__.py` + `bpe.py` | train/save/load/check_tokenizer |
| `learning/s0_tokenizer.py` | CLI + 头部注释四要素 |
| `tests/test_tokenizer.py` | 往返/校验/污染四场景 |

## 风险与回退
- [BPE 把高频双字母（如 "AB"）合并挤掉单字母] → initial alphabet 注入后单字母永不参与合并（tokenizers 库语义保证）+ check 兜底。
- [真实语料拉取网络失败] → 3C 孙任务可延后：单测用内置小语料；真实语料档在 W2 前补齐即可。
