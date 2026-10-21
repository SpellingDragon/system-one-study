# sys1/decision/ — 决策程序（两阶段同构桥梁）★

> 这是全项目最重要的解耦点：**读输出只依赖"最后一层隐状态 × 词表中 26 个字母的行"**，与 backbone 内部无关——一阶段小 GPT 与二阶段预训练 backbone 共用本目录全部代码。

## 计划文件与职责

| 文件 | 职责 | 参考落点（refs/） |
|---|---|---|
| `render.py` | `/v1/systemone` 请求 → chat prompt：`Evidence:\n<state>\n\nQuestion: <ins>\nOptions:\nA) ...\nB) ...`；固定 system 行；思考关闭前缀结尾 | `StartLux-Decision/startlux_decision/jevfmt.py`：`messages()/option_lines()/from_systemone()/check_tokenizer()` |
| `readout.py` | 末位 hidden × `lm_head.weight[letters]` → 选项 logits；`temperature.py` 之后 softmax；**右 padding 因果不变性**（读点位之前无 pad） | 同仓 `model.py`：`letter_rows/_slot_logits/_logits` |
| `temperature.py` | 每 qtype 一个温度；**温度不改 argmax**；clamp 与池化规则 | `StartLux/finetune/calibrate.py`；laya `common.py` `clamp_temperature` |
| `wide.py` | >26 选项：分组（≤25）多轮票选，top-keep 进决赛，未入选者留残差概率（和为 1） | 同仓 `model.py::_wide` |

## 硬约定（写测试守护）

- 键约定：`noul → {"false","true"}`；`score → "0".."n-1"`（低→高）；`choice → criteria 原键`。
- **26 字母必须各为单 token**且是答案位的合法延续——`check_tokenizer` 等价断言放进 `tests/`，S0 与二阶段换 tokenizer 时各跑一次（接缝三处之一，PRODUCTION §0）。
- `output_tokens = 0`：任何生成/解码循环都是路线错误。
- 渲染版本化（如 `RENDER_VERSION` 写入 run config），跨报告引用时随数字一起报。

## 验收钩子

- 一阶段 M0：同一输入在 eager 与 CUDA-graph 路径概率一致（StartLux `self_test` 思路，容差内）。
- 二阶段 G7：本目录代码零改动而三栈增益可测 = 同构成立的最直接证据。
