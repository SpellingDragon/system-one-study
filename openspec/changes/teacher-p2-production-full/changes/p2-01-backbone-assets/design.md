# Design: p2-01-backbone-assets

## 技术要点

- 载入用 transformers 5.18（`AutoModelForCausalLM` + fp16 + `.to("mps")`）；ModelScope `snapshot_download` 优先（实测 ~9.5MB/s），repo/revision 落 run config（溯源）。**梯度检查点在训练域（p2-05）开启**，本域只保载入与前向。
- **窄接口适配层**：薄 wrapper 暴露 `forward(input_ids, attn_mask) -> last_hidden` 与 `lm_head.weight`——P1 `decision/` 零改动的物理保证。
- 接缝三校验（`SeamCheckError` 拒载且消息含接缝名）：
  ① 复用 P1 `check_tokenizer`（26 字母单 token）——Qwen 系 BPE 字母前缀空格风险，chat template 去前导空格或 tokenizer 配置修正；仍不行按风险预案换 Qwen3-0.6B（已备）并记录决策；
  ② letter_rows 按 backbone hidden_size 重建（维度断言；Qwen3.5-0.8B hidden=1024 已从 config 实证）；
  ③ 思考关闭模板快照：Qwen3.5 thinking 开关的关闭前缀（`enable_thinking=False` 等价物）实测后逐字节固化入测试。
- **Qwen3.5-0.8B 特有接缝（第四校验，权重实证所生）**：config 顶层含 `image_token_id/vision_start/end_token_id`——载入时断言这些 id 存在且不与字母 id 冲突（多模态域 p2-08 依赖此接口）。
- 参考落点：`StartLux-Decision/startlux_decision/model.py::load_model`；0.6B 载入实操已在 probe run 验证（fp16/MPS/LoRA 全通）。

## 文件清单

| 文件 | 职责 |
|---|---|
| `release/production/assets.py` | 载入 + 四校验 + 接口适配 |
| `release/tests/test_assets.py` | 载入/接缝/换脑/设备/真实路径场景 |

## 风险与回退

- [Qwen3.5 字母非单 token] → 模板/分词修正；终则降 0.6B（已实测）记录决策。
- [Qwen3.5-0.8B 载 MPS 内存紧（视觉塔+262K config）] → fp16 权重 ~1.7GB + 塔 ~0.3GB，36GB 充裕（probe 已证同量级可行）；训练侧由 p2-05 的 GC 保。
- [思考关闭前缀形态不明] → 首孙任务实测固化（Open Question 回收）。
- **权重布局转换（v3 新增）**：HF safetensors → 自研栈张量布局的转换器（p2-13 算子布局对齐）；转换后前向对拍 HF 输出（容差内）才算接缝完成。
