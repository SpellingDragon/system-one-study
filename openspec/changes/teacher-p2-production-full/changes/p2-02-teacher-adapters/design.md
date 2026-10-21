# Design: p2-02-teacher-adapters

## 技术要点
- 文本教师取**末位字母 logprob**（与学生读出同构——教师与学生"看同样的东西"，伪标才可迁移）：对每选项字母 token 取 logprob，softmax 归一化为选项分布。
- **生成式兜底走 GLM-V verifier 协议（refs/GLM-V `vqa_verifier.py` 实证）**：教师 prompt 约束以 `<answer>X</answer>` 作答、max_tokens 压短（省 token 提一次通过率）；抽取按 **qtype 分域**（choice=字母精确匹配→规约；noul=true/false；score=数值容差），嵌套标签/空答案判 None 重试一轮——比自由生成再通用解析更准更省。
- 教师选型（D9 终态·三教师分工）：文本决策 = **StartLux-4B**（主力：决策专精、末位字母读出原生同构零 prompt 工程；载入走本地 `startlux_decision/model.py::load_model`）；视觉 = **GLM-5.3-Flash API 离线伪标**（320B 质量、§11-4 红线合规、几十元级、不占本地内存）；文本备选 = Qwen3.5-4B（权重不可得时降级并记录）；本地 Qwen3.5-9B 仅当 API 不可用且有内存余量时可选。**OPD 在线同驻内存账（仅文本教师本地驻留场景）**：学生 0.8B 训练 bs4×1024+GC 实测 16.2GB + StartLux-4B 推理 ~10GB ≈ 26GB < 29GB recommended，可行但紧，W0 冒烟实测定档；云端更大教师仅换配置。
- StartLux-4B 获取与接缝（已实证）：魔搭 **`StartLuxAI/StartLux-Decision-4B`**（32 文件/权重 2 分片 ~8GB/**自带 decision_config.json**——温度等决策配置随权重发布，伪标直读）；其字母读出即 P1 `decision/` 同构机制（`jevfmt check_tokenizer` 已有）——`score_options` 对接其 `letter_rows` 读出无需改其代码。**额外发现**：其仓含 vision/video preprocessor（原生多模态血统）——B3 顺带探测能否兼任视觉教师（能则视觉教师也免 API 依赖，不能则 GLM-5.3-Flash API 主路线不变）；GLM API key=环境变量 `ZAI_API_KEY`（已验证在位）。
- 缓存键 `sha256(model_id + prompt + sorted(keys))` 截断 16 hex；parquet 列：key/model_id/dist(json)/created_at。缓存目录入 `bench/teacher_cache/`（gitignore）。
- 视觉教师主路线（本地 9B 跑不动/不想下所改）：**GLM-5.3-Flash API 离线伪标**（320B MoE/18B active，质量远超本地 9B；红线合规——外部仅作教师离线产标，训练/评测/服务全链零在线 API）：主用逐选项置信打分（0–10→归一化 soft 分布），兜底 `<answer>` 协议 GLM-V 分域抽取；key 环境变量不入库；量级 3–6k 次调用（人民币几十元级）一次产完入 parquet（image_hash/prompt_hash/model_id 可复现）。本地 Qwen3.5-9B 降备选（单跑纯推理 ~18GB 可行时）。
- 前向计数器：`TeacherStats.forward_calls` 注入测试与训练日志（"零推理"断言的数据源）。

## 文件清单
| 文件 | 职责 |
|---|---|
| `production/teachers/text.py` | 文本教师 + stats |
| `production/teachers/vision.py` | 视觉教师 + 离线包 |
| `production/teachers/cache.py` | parquet 缓存 |
| `tests/test_teachers.py` | 五场景 |

## 风险与回退
- [教师字母 logprob 含 BPE 前缀干扰] → 与 p2-01 同源问题：模板保证字母为独立续写 token；数值单测对比 softmax(logprob) 锁定。
- [缓存与教师版本错配] → model_id 入键；换教师自动 miss，不污染。
