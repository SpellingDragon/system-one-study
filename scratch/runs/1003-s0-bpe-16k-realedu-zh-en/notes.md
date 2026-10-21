# run: 1003-s0-bpe-16k-realedu-zh-en — S0 自训中英 BPE

假设：initial alphabet 钉死 52 字母 + 读表时强校验，就能让字母各占一个编号，中英混排可无损往返。
观察：
- 语料：bench/corpus_s0/txt 等文档，读入 209469929 字节 / 1291642 行，训练耗时 33.65 秒（CPU）。
- 表规模 16000（配置上限 16000），pad 在 0 号，特殊 token ['<|pad|>', '<|endoftext|>']。
- check_tokenizer 通过：52/52 字母各占一个编号。
- 抽检 中文：{'chars': 68767, 'oov_char_rate': 0.0, 'tokens': 41877, 'chars_per_token': 1.642}；英文：{'chars': 65324, 'oov_char_rate': 0.0, 'tokens': 33390, 'chars_per_token': 1.956}。
- 往返：'你好世界 hello world 123' 编回后与原串一致。
结论：字母约束达成（52/52 单 token），中英覆盖与压缩比见上；若中文单字缺失率偏高，加大 --stream-limit 或 --vocab-size 复跑，不放行不合格产物。

复跑：python learning/s0_tokenizer.py --corpus bench/corpus_s0/txt --vocab-size 16000 --stream-limit 200MB --out $WORKSPACE_ROOT/deep-mutimodal-laya-by-tilelang/runs/1003-s0-bpe-16k-realedu-zh-en
