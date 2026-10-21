"""sys1.lang — 中英 BPE 自训封装（train / save / load / check_tokenizer）。

对外只暴露这四个动作加两个常量：字母表 LETTERS_52 与异常 TokenizerCheckError。
零本仓依赖（只用 tokenizers 库），被 p1-06（词表常数）、p1-07（encode + 字母 id 映射）、
p1-08（语料 token 化）依赖。
"""
from __future__ import annotations

from .bpe import (
    CLEANING_RULES,
    DEFAULT_SPECIAL_TOKENS,
    ENDOFTEXT_TOKEN,
    LETTERS_52,
    MINI_CORPUS,
    PAD_TOKEN,
    UPPERCASE,
    TokenizerCheckError,
    char_oov_rate,
    check_tokenizer,
    clean_text,
    initial_alphabet,
    load,
    save,
    train,
)

__all__ = [
    "CLEANING_RULES",
    "DEFAULT_SPECIAL_TOKENS",
    "ENDOFTEXT_TOKEN",
    "LETTERS_52",
    "MINI_CORPUS",
    "PAD_TOKEN",
    "UPPERCASE",
    "TokenizerCheckError",
    "char_oov_rate",
    "check_tokenizer",
    "clean_text",
    "initial_alphabet",
    "load",
    "save",
    "train",
]
