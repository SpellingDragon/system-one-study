# Tasks: p2-02-teacher-adapters [W0 · 无依赖]

> 三级结构：`### 工作项` → 孙任务 checkbox。spec：`specs/teacher-adapters/spec.md`。

### 工作项 A 接口

- [x] A1 `production/teachers/text.py`：`score_options(prompt, option_keys)` 末位字母 logprob → 分布 —— 验证：`python -m pytest tests/test_teachers.py -k text -q`
- [x] A2 数值一致性单测（返回分布与教师字母 logprob 的 softmax 一致） —— 验证：`python -m pytest tests/test_teachers.py -k numeric -q`

### 工作项 B 离线缓存

- [x] B1 parquet 缓存（(model_id, prompt_hash, keys_hash) 键）+ 二次调用零推理单测 —— 验证：`python -m pytest tests/test_teachers.py -k cache -q`
- [x] B2 教师吞吐实测：50 条未见请求冒烟，tok/s 与缓存前后耗时入 runs —— 验证：`ls runs/ | grep teacher` 且 notes 含 tok/s 行
- [x] B3 【真实路径验证】教师三路各验一次：① **StartLux-4B** 真拉真载——魔搭 `StartLuxAI/StartLux-Decision-4B` snapshot_download（~8GB）→ `load_model` + 一次 `score_options` 打分 + 顺带探测其 vision preprocessor 能否兼视觉教师；② **GLM-5.3-Flash API** 连通小批（10 条图文，key=环境变量 `ZAI_API_KEY`，记录延迟/成本）；③ 备选 Qwen3.5-4B 不验（触发降级时再验）—— 验证：run notes 含①②实测行 + 缓存目录含 4B 权重（【P1 教训】mock 不算完成）

### 工作项 C 视觉教师

- [x] C1 `production/teachers/vision.py`：(image, prompt, keys) → 分布 + 离线伪标包回放单测 —— 验证：`python -m pytest tests/test_teachers.py -k vision -q`
