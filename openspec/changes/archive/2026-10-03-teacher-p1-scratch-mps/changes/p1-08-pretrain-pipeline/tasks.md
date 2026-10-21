# Tasks: p1-08-pretrain-pipeline [W2 · 依赖 p1-03+p1-06+p1-02；与 p1-09 并发]

> 三级结构：`### 工作项` → 孙任务 checkbox。spec：`specs/pretrain-pipeline/spec.md`。

### 工作项 A 语料装配

- [x] A1 `dmlaya/data/pretrain_corpus.py`：流式拉取（fineweb-edu + 中文教育语料，配比 7:3 可配）、去重、分片 token 流 —— 验证：`python -m pytest tests/test_corpus.py -q`
- [x] A2 冒烟语料：`--stream-limit 200MB` 产出分片（token id 全部 < vocab，中英占比非零） —— 验证：`python -m dmlaya.data.pretrain_corpus --stream-limit 200MB --out runs/corpus_tiny --config tiny && python -m pytest tests/test_corpus.py -k tiny_output -q`

### 工作项 B 训练循环

- [x] B1a `learning/s1_pretrain_gpt.py` 训练循环主体：随机初始化 → 下一词 CE → metrics.jsonl 定时写（NaN 检测中断） —— 验证：`python -m pytest tests/test_s1_train.py -k loop -q`
- [x] B1b checkpoint 分片保存 + `--resume` 续训机制（config 一致性校验，不一致拒续） —— 验证：`python -m pytest tests/test_s1_train.py -k ckpt_resume_impl -q`
- [x] B2 两档配置 `learning/configs/{smoke_tiny,mps_main}.yaml`（超参不硬编码） —— 验证：`python -c "import yaml; [yaml.safe_load(open(f)) for f in ['learning/configs/smoke_tiny.yaml','learning/configs/mps_main.yaml']]"`
- [x] B3 断点续训测试（中断→resume→metrics 无重复 step） —— 验证：`python -m pytest tests/test_s1_train.py -k resume -q`

### 工作项 C 验收

- [x] C1 采样冒烟：温度采样 8 条 ≤40 token 短句入 run notes + 人工判定结论行 —— 验证：`python learning/s1_pretrain_gpt.py --config smoke_tiny --sample 8` 后 notes.md 含 8 条与结论
- [x] C2 smoke_tiny 端到端 CI 测试（`@slow`，CPU≤10min，首末 loss 下降断言） —— 验证：`python -m pytest tests/test_s1_train.py -k e2e -q -m slow`
