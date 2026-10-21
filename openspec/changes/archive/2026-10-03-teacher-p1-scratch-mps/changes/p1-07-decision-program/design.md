# Design: p1-07-decision-program

## 技术要点
- 渲染格式锁定（黄金快照逐字节）：system 固定行 + `Evidence:\n<state>` + `\n\nQuestion: <instructions>` + `\nOptions:\nA) ...`（字母序）；思考关闭前缀结尾（为二阶段 chat template 留位）。
- 读出：`末位 hidden @ lm_head.weight[letters].T` → 选项 logits → qtype 温度 → softmax。**右 padding 因果不变性**：读出位取每序列实际末位（非 padded 末列）——批内 pad 不改概率（≤1e-5）。
- `output_tokens = 0`：本域无任何生成/解码循环——静态 grep 测试断言（`.generate(` 禁现）。
- temperature.py：qtype→T 表（clamp [0.2,5]），缺表 T=1.0 + `uncalibrated` 标注；argmax 不变性由单调性保证并测试锁定。
- wide.py：k>26 分组（每组 ≤25 + 剩余槽 "其他"），组内票选 → top-keep 决赛 + 未入选者残差概率，输出和=1；决赛组内序与字母序一致。
- 参考落点：`StartLux-Decision/startlux_decision/jevfmt.py`（messages/option_lines/from_systemone/check_tokenizer）、同仓 `model.py`（letter_rows/_slot_logits/_wide/decide_batch）。

## 文件清单
| 文件 | 职责 |
|---|---|
| `dmlaya/decision/render.py` | 解析 + 渲染 + RENDER_VERSION |
| `dmlaya/decision/readout.py` | 末位读出（pad 不变） |
| `dmlaya/decision/temperature.py` | qtype 温度应用 |
| `dmlaya/decision/wide.py` | >26 分组票选 |
| `tests/test_decision.py` | 七场景 + 快照 |

## 风险与回退
- [渲染格式与二阶段 chat template 冲突] → RENDER_VERSION 版本化 + 快照测试：改格式 = 显式升版本 + 二阶段接缝校验（p2-01）同步更新。
- [wide 票选残差概率不正交] → 数值单测：50 选项输出和=1（1e-6）且各组概率非负。
