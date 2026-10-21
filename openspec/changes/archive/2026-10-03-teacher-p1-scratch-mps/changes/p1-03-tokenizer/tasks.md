# Tasks: p1-03-tokenizer [W0 · 无依赖；3C 可延后]

> 三级结构：`### 工作项` → 孙任务 checkbox。spec：`specs/tokenizer/spec.md`。

### 工作项 A 训练入口

- [x] A1 `dmlaya/lang/` 封装：train/save/load BPE（词表 8k–32k 可配）+ 特殊 token `<|pad|>`/`<|endoftext|>` —— 验证：`python -m pytest tests/test_tokenizer.py -k api -q`
- [x] A2 `learning/s0_tokenizer.py` CLI：`--corpus --vocab-size --stream-limit --out`，头部注释含目的/输入/产出 run-id/预计耗时 —— 验证：`python learning/s0_tokenizer.py --help` 且 `python tools/check_comments.py`
- [x] A3 内置小语料训练单测：编解码往返 `"你好世界 hello world 123"` 完全还原 —— 验证：`python -m pytest tests/test_tokenizer.py -k roundtrip -q`

### 工作项 B 字母强校验

- [x] B1 实现 `check_tokenizer()`：52 字母（A–Z/a–z）均单 token，返回 id 映射；`TokenizerCheckError` 列缺失字母 —— 验证：`python -m pytest tests/test_tokenizer.py -k check -q`
- [x] B2 污染词表被止单测（构造 'A' 缺失词表） —— 验证：`python -m pytest tests/test_tokenizer.py -k polluted -q`

### 工作项 C 真实语料冒烟（不阻塞 W1）

- [x] C1 以 200MB 级 fineweb-edu+中文语料流式训练真实 tokenizer，产物入 `runs/`（run-notebook 记录），中文/英文 token 覆盖率抽检入 notes —— 验证：`ls runs/ | grep s0` 且 run 目录含 tokenizer 产物与 notes.md 结论行
  > 执行记录（2026-10-03 UTC）：语料走 hf-mirror 拉 `opencsg/Fineweb-Edu-Chinese-V2.2` 2_3/001376.parquet + `HuggingFaceFW/fineweb-edu` sample/350BT/016_00000.parquet，转纯文本 249MB 存 `bench/corpus_s0/txt/`（gitignored）；run：`runs/1003-s0-bpe-16k-realedu-zh-en`（vocab 16000，读入 209,469,929 B / 1,291,642 行，训练 33.65 s CPU，52/52 字母单 token，中文 oov 0.0 / chars_per_token 1.642，英文 0.0 / 1.956，往返一致）。
