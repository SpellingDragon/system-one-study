"""p1-03 tokenizer 域验收测试（契约见 openspec/changes/teacher-p1-scratch-mps/changes/p1-03-tokenizer/specs/tokenizer/spec.md）。

四个场景对四个用例（-k 选择子即 tasks.md 里的验证命令）：
  api        —— A1：train/save/load 接口面 + 特殊 token（<|pad|> 固定 0 号）+ 词表 8k–32k 约束
  roundtrip  —— A3：内置小语料训练产物对 "你好世界 hello world 123" 编解码完全还原
  check      —— B1：check_tokenizer 返回 52 字母单 token 映射
  polluted   —— B2：'A' 被 "AB" 合并吃掉的污染词表被拒，且错误里列出缺失字母

全部离线（内置小语料，毫秒级训练），不联网、不加载任何第三方词表/权重。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

from sys1.lang import bpe

ROUNDTRIP_SAMPLE = "你好世界 hello world 123"
# 额外往返样本：换行/制表/多空格/中英标点混排——决策 prompt 的分段骨架必须原样回来
EXTRA_SAMPLES = (
    "AB CD 你好 世界 abc123 决策A",
    "Evidence: 今天多云，气温 18 度。\nQuestion: 下列哪个选项最稳妥？\nOptions:\nA. 稳妥 conservative\nB. 激进 aggressive",
    "  double  space ",
    "line1" + chr(10) + "line2" + chr(9) + "tail",
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
)


@pytest.fixture(scope="session")
def trained() -> Tokenizer:
    """会话级共享一次小语料自训（取词表下限 8k：语料太小，实际只会长出几百条）。"""
    return bpe.train(bpe.MINI_CORPUS, vocab_size=bpe.VOCAB_MIN, min_frequency=1)


def _roundtrip(tok: Tokenizer, text: str) -> str:
    return tok.decode(tok.encode(text, add_special_tokens=False).ids, skip_special_tokens=False)


# ------------------------------------------------------------------ A1 接口面
def test_api_train_save_load(trained: Tokenizer, tmp_path: Path):
    """train→save→load 全链路：产物目录齐三件套 + 配置，重载后编号与往返一致。"""
    assert trained.token_to_id(bpe.PAD_TOKEN) == 0, "<|pad|> 必须固定 0 号"
    assert trained.token_to_id(bpe.ENDOFTEXT_TOKEN) is not None, "缺 <|endoftext|>"
    assert trained.get_vocab_size() > 0

    out = bpe.save(trained, tmp_path / "tokenizer", meta={"corpus": "MINI_CORPUS"})
    for name in (bpe.TOKENIZER_FILE, bpe.VOCAB_FILE, bpe.MERGES_FILE, bpe.CONFIG_FILE):
        assert (out / name).is_file(), name
    config = json.loads((out / bpe.CONFIG_FILE).read_text(encoding="utf-8"))
    assert config["cleaning_rules"] == list(bpe.CLEANING_RULES), "清洗规则必须显式入配置"
    assert config["pad_token_id"] == 0
    assert set(config["special_tokens"]) == {bpe.PAD_TOKEN, bpe.ENDOFTEXT_TOKEN}

    reloaded = bpe.load(out)
    assert reloaded.get_vocab_size() == trained.get_vocab_size()
    assert _roundtrip(reloaded, ROUNDTRIP_SAMPLE) == ROUNDTRIP_SAMPLE


def test_api_vocab_size_must_be_within_8k_32k(trained: Tokenizer):
    """词表规格 8k–32k 可配，越界直接拒绝（p1-06 的 vocab 常数依赖这个口径）。"""
    assert (bpe.VOCAB_MIN, bpe.VOCAB_MAX) == (8000, 32000)
    for bad in (1000, 40000):
        with pytest.raises(ValueError, match="vocab_size"):
            bpe.train(bpe.MINI_CORPUS, vocab_size=bad)
    assert trained is not None


# ------------------------------------------------------------------ A3 编解码往返
def test_roundtrip_chinese_english_digits(trained: Tokenizer):
    """spec 场景一：中英数混排样本编回来一个字符都不许多也不许少。"""
    assert _roundtrip(trained, ROUNDTRIP_SAMPLE) == ROUNDTRIP_SAMPLE
    for sample in EXTRA_SAMPLES:
        assert _roundtrip(trained, sample) == sample, repr(sample)


# ------------------------------------------------------------------ B1 字母强校验
def test_check_tokenizer_returns_52_single_letter_ids(trained: Tokenizer):
    """spec 场景三：52 字母各占一个编号，且单独编它只出一个符号、符号原文就是它本身。"""
    mapping = bpe.check_tokenizer(trained)
    assert len(mapping) == 52
    assert set(mapping) == set(bpe.LETTERS_52)
    assert len(set(mapping.values())) == 52, "52 个字母编号必须两两不同"
    for letter, letter_id in mapping.items():
        assert trained.id_to_token(letter_id) == letter
        assert len(trained.encode(letter, add_special_tokens=False).ids) == 1
    assert bpe.letter_rows(mapping) == [mapping[c] for c in bpe.UPPERCASE]


def test_check_tokenizer_survives_frequent_double_letter(trained: Tokenizer):
    """高频双字母（"AB"）确实合成了整条，但单字母仍在册——字母约束的实现方式在此落证。"""
    assert "AB" in trained.get_vocab(), "小语料应把 AB 合成整条（用来考验字母约束）"
    assert len(bpe.check_tokenizer(trained)) == 52


# ------------------------------------------------------------------ B2 污染词表被拒
def _polluted_tokenizer() -> Tokenizer:
    """手工造一张"AB 是整条、单字母 A 不在册"的污染词表（spec 场景四的设定）。

    BPE 的构造器会拒绝引用不在词表内的符号的合并对，所以这里直接给出被污染后的终态：
    "AB" 作为一条成品在表里，而 "A" 缺席——单编 A 就什么都弹不出来。
    """
    vocab = {bpe.PAD_TOKEN: 0, bpe.ENDOFTEXT_TOKEN: 1, "AB": 2}
    for symbol in bpe.LETTERS_52[1:]:  # 故意不登记 "A"
        vocab[symbol] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.Split(pattern=Regex(bpe.SPLIT_PATTERN), behavior="isolated")
    tok.decoder = decoders.Sequence([])
    return tok


def test_polluted_vocab_is_rejected(tmp_path: Path):
    """spec 场景四：污染词表被 check_tokenizer 拒掉，并列出缺失字母；load 默认再兜一道。"""
    dirty = _polluted_tokenizer()
    assert "AB" in dirty.get_vocab() and "A" not in dirty.get_vocab()

    with pytest.raises(bpe.TokenizerCheckError) as exc:
        bpe.check_tokenizer(dirty)
    assert exc.value.missing == ["A"], exc.value.missing
    assert "A" in str(exc.value)

    out = bpe.save(dirty, tmp_path / "dirty")
    with pytest.raises(bpe.TokenizerCheckError):
        bpe.load(out)  # 落盘能过，读取默认强校验拦下：initial alphabet 注入之外的第二道保险
    assert bpe.load(out, check=False).get_vocab_size() == dirty.get_vocab_size()
