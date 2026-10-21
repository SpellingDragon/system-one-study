"""S0 自训 BPE 封装：train / save / load / check_tokenizer（字母约束的地基）。

【做什么】把一堆中英混排的文本文本，炼成一张"文本 ↔ 编号"对照表：训练时自己造表，用的时候靠它
把文字换成一串编号、再把编号换回文字，并保证 26 个英文字母大小写各自就是一个独立编号。
【怎么做】调 tokenizers 库的 BPE 训练器：先做文本清洗（去网址、去控制字符），再把 52 个字母、数字、
中英标点**预先钉进初始符号集**（initial_alphabet）——被钉住的符号从训练第一步就已是成品符号，
后续合并只会"往上叠长条"，绝不可能把它们拆没；训完落盘成 tokenizer.json + vocab.json + merges.txt
加一份自描述配置；读取时默认再跑一遍 check_tokenizer，把 52 个字母逐个单独送进去试编，只要有一个
不是"一个符号搞定"，就抛 TokenizerCheckError 并列出缺失字母。
【为什么】字母约束必须来自造表阶段而不是事后打补丁：决策读出要把"末位输出的 26 个字母行"当作答案
候选，字母编号一旦在训练中被高频双字母（如 "AB"）吃掉，读出层就会静默错位。被否方案一：直接加载
Qwen/GLM 现成词表——省事但违反从零红线，而且这类词表的字母多带前导空格变体，读不出干净的 26 行；
被否方案二：自己手写 BPE 算法——教学上更透明，但把工时花在重造轮子上，字母约束与契约才是本域重点
（学生版不阻拦手写）。
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

# ---------------------------------------------------------------- 常量：契约的"字面量"
# 特殊 token 用 <|...|> 包裹，避免与自然文本撞车；顺序即编号：<|pad|> 必须是 0 号，
# 下游按 pad 做掩码时靠的就是这个固定值（见 p1-06/p1-08）。
PAD_TOKEN = "<|pad|>"
ENDOFTEXT_TOKEN = "<|endoftext|>"
DEFAULT_SPECIAL_TOKENS: tuple[str, ...] = (PAD_TOKEN, ENDOFTEXT_TOKEN)

# 本域灵魂：决策读出需要的 26 字母 × 大小写 = 52 个"必须独占一个编号"的符号。
UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWERCASE = "abcdefghijklmnopqrstuvwxyz"
LETTERS_52 = UPPERCASE + LOWERCASE
# 数字与中英常见标点：一并钉住，保证中英混排文本不会因缺符号而丢字。
DIGITS = "0123456789"
PUNCTUATION = (
    "，。！？；：“”‘’（）《》【】〈〉「」『』—…·"  # 中文标点
    ",.;:!?'\"()[]{}<>/\\|@#$%^&*+=~`-"            # 英文标点
)
# 空白符也要有独立符号：换行与制表符是决策 prompt 的分段骨架，不能被吞。
WHITESPACE = " \t\n\r"

# 词表规格上下限（GUIDE §4 M0 与 p1-06 词表常数的共同口径）：低于 8k 中文覆盖不足，
# 高于 32k 端侧内存与训练时长不划算。
VOCAB_MIN = 8_000
VOCAB_MAX = 32_000

# 落盘文件名（三件套 + 一份自描述配置；名字即约定，p1-06/p1-08 按此拼目录）
TOKENIZER_FILE = "tokenizer.json"
VOCAB_FILE = "vocab.json"
MERGES_FILE = "merges.txt"
CONFIG_FILE = "tokenizer_config.json"

# 清洗规则显式入配置：任何"训练前对文本动过的手"都必须能被复查。
URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>，。！？；]+")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
CLEANING_RULES: tuple[str, ...] = (
    "http(s):// 与 www. 开头的网址整段删除",
    "控制字符删除（保留换行、回车、制表符）",
    "CRLF 归一为 LF；不做大小写折叠、不压缩空格——保留原样才能逐字还原",
)

# 切块与还原的组合：按空白把文本切成"词块 + 空白段"两类片段（空白段自己是片段），
# 还原时把片段直接首尾相接——这样英文单词被拆成小段后也不会被塞进多余空格。
# 被否方案：库自带的 Whitespace 预切会把空白丢掉，还原时只能靠"猜"补空格，
# 一个被拆成 8 小段的长词就还原成 8 段之间多 7 个空格的畸形文本。
SPLIT_PATTERN = r"\s+"

# 内置小语料（A3 单测用；离线、毫秒级）。刻意高频出现 "AB"，用来证明"高频双字母
# 合并"吃不掉单字母；同时覆盖中英、数字、标点与决策 prompt 的换行骨架。
MINI_CORPUS: tuple[str, ...] = (
    "你好世界 hello world 123",
    "世界你好，欢迎回来。welcome back, 2024!",
    "AB AB AB ABCD ABC AB CD 双字母合并测试",
    "决策 decision 选项 option 字母 letter",
    "Evidence: 今天多云，气温 18 度。",
    "Question: 下列哪个选项最稳妥？",
    "Options:",
    "A. 稳妥 conservative",
    "B. 激进 aggressive",
    "C. 未知 unknown",
    "深度学习需要算力，deep learning needs compute.",
    "中文和 English 混排 text 一起训练，标点：，。！？；：",
    "函数 def add(a, b): return a + # 代码片段 code snippet",
    "请输出字母 A 或 B，也可小写 a b c d e f g h i j k l m n o p q r s t u v w x y z",
    "数字 0123456789 与编号 ID-10086 都要能还原",
    "The quick brown fox jumps over the lazy dog 0123456789",
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "Safety first: 先测正确性，再谈性能 performance comes after correctness.",
)


class TokenizerCheckError(RuntimeError):
    """字母约束不达标时抛出；missing 列出所有没能独占一个编号的字母。"""

    def __init__(self, missing: Sequence[str], detail: dict[str, str] | None = None) -> None:
        self.missing = list(missing)
        self.detail = detail or {}
        preview = ", ".join(repr(c) for c in self.missing[:12])
        extra = f"（共 {len(self.missing)} 个）" if len(self.missing) > 12 else ""
        super().__init__(f"字母单 token 校验失败，缺失/被合并的字母: {preview}{extra}")


# ---------------------------------------------------------------- 文本清洗
def clean_text(text: str) -> str:
    """按 CLEANING_RULES 清洗单段文本；规则本身写进产物配置，便于复查。

    白话：炼表之前先给文本洗澡——网址整段剪掉、键盘敲不出来的控制乱码擦掉、Windows 的两字
    换行统一成一个，除此之外一律不动，好让"读回来的字"和"送进去的字"一个不多一个不少。
    """
    cleaned = URL_PATTERN.sub(" ", text)
    cleaned = CONTROL_CHARS.sub("", cleaned)
    return cleaned.replace("\r\n", "\n")


def initial_alphabet() -> list[str]:
    """返回必须钉死的初始符号集：52 字母 + 数字 + 中英标点 + 空白符（去重、保序）。

    白话：先把最要紧的那批单个符号（五十二个字母、十个数字、常见中英文标点）登记在册；
    登记过的就当成"已经是成品"，后面再怎么两两拼长都拼不到它们头上——这是字母约束的第一道
    保险，第二道是下面的校验。
    """
    ordered: list[str] = []
    for symbol in LETTERS_52 + DIGITS + PUNCTUATION + WHITESPACE:
        if symbol not in ordered:
            ordered.append(symbol)
    return ordered


# ---------------------------------------------------------------- 训练
def train(
    corpus: Iterable[str],
    vocab_size: int = 16_000,
    *,
    special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
    min_frequency: int = 2,
    alphabet: Sequence[str] | None = None,
    show_progress: bool = False,
) -> Tokenizer:
    """从中英混排语料自训一张 BPE 表；corpus 传文本行的迭代器（流式，不整档载入内存）。

    词表大小限定在 8k–32k；<|pad|> 固定 0 号；初始符号集默认取 initial_alphabet()。
    返回的是 tokenizers 库的 Tokenizer 对象，encode/decode 直接用。

    白话：把一堆文本交给现成的合并算法，让它从出现次数最多的相邻两截开始不断拼长条，拼到规定
    条数为止；开拼之前五十二个字母等要紧的单符号已经钉死在册，拼来拼去只会拼出更长的条目，
    动不到它们。
    """
    if not VOCAB_MIN <= vocab_size <= VOCAB_MAX:
        raise ValueError(f"vocab_size 必须在 [{VOCAB_MIN}, {VOCAB_MAX}] 区间，收到 {vocab_size}")
    if not special_tokens or special_tokens[0] != PAD_TOKEN:
        raise ValueError("<|pad|> 必须是第一个特殊 token（固定 0 号，下游 pad 掩码依赖它）")

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern=Regex(SPLIT_PATTERN), behavior="isolated")
    tokenizer.decoder = decoders.Sequence([])

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(special_tokens),
        initial_alphabet=list(alphabet or initial_alphabet()),
        # 允许从语料里新长出来的单符号数量上限=目标词表：语料里的汉字尽量都在册，
        # 否则遇到没登记的汉字会被静默丢弃（unk_token=None 的取舍，见模块头）。
        limit_alphabet=vocab_size,
        show_progress=show_progress,
    )
    tokenizer.train_from_iterator(_as_lines(corpus), trainer=trainer)

    if tokenizer.token_to_id(PAD_TOKEN) != 0:
        raise RuntimeError(f"<|pad|> 没有落在 0 号（实际 {tokenizer.token_to_id(PAD_TOKEN)}）")
    return tokenizer


def _as_lines(corpus: Iterable[str]) -> Iterator[str]:
    """把"整段文本"或"逐行文本"的迭代器统一成逐行（合并算法按行统计频次）。"""
    for chunk in corpus:
        yield from chunk.splitlines()


# ---------------------------------------------------------------- 落盘与读取
def save(tokenizer: Tokenizer, out_dir: str | Path, *, meta: dict | None = None) -> Path:
    """把产物写成一个目录：tokenizer.json + vocab.json + merges.txt + tokenizer_config.json。

    白话：把炼好的表和它的说明书一起放进一个文件夹——机器读的整包、人读的清单与合并记录、
    外加一份写清楚"洗过哪几遍、怎么切怎么还原"的配置单。换台机器打开也能对上号。
    """
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(target / TOKENIZER_FILE))
    # 库自带的可读导出：查词频、人工抽查都靠这两份纯文本
    tokenizer.model.save(str(target))
    config = default_meta(tokenizer)
    config.update(meta or {})
    (target / CONFIG_FILE).write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def default_meta(tokenizer: Tokenizer) -> dict:
    """生成交付配置的默认字段：表规模、特殊 token、清洗规则、切块与还原方式、生成时间。

    白话：随表附一张说明单，写清这张表有多大、里面钉了哪些不寻常的记号、造表之前文本洗过
    哪几遍、字符串是怎么切成小段又怎么接回去的。日后拿到产物却想不起这些细节，来回就对不上。
    """
    return {
        "vocab_size_actual": tokenizer.get_vocab_size(),
        "vocab_size_spec": f"{VOCAB_MIN}-{VOCAB_MAX}",
        "special_tokens": list(DEFAULT_SPECIAL_TOKENS),
        "pad_token_id": tokenizer.token_to_id(PAD_TOKEN),
        "initial_alphabet_size": len(initial_alphabet()),
        "cleaning_rules": list(CLEANING_RULES),
        "pre_tokenizer": f"Split(regex={SPLIT_PATTERN!r}, behavior=isolated)",
        "decoder": "Sequence([])：片段直接首尾相接，空白由独立片段承载",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "self-trained BPE (no third-party vocab/weights)",
    }


def load(in_dir: str | Path, *, check: bool = True) -> Tokenizer:
    """从目录读回 tokenizer；默认顺手跑一遍字母校验（双保险的第二道）。

    白话：开表即用之前，先抽查五十二个字母是不是每个都能一个符号单独表示；只要有一个被吞了，
    当场报错并列出是哪几个，绝不让这张有毛病的表混进训练。
    """
    path = Path(in_dir)
    tokenizer = Tokenizer.from_file(str(path / TOKENIZER_FILE))
    if check:
        check_tokenizer(tokenizer)
    return tokenizer


# ---------------------------------------------------------------- 字母强校验
def check_tokenizer(tokenizer: Tokenizer, letters: str = LETTERS_52) -> dict[str, int]:
    """逐个单独试编 52 个字母，返回"字母 → 编号"映射；不达标即抛 TokenizerCheckError。

    判定只看两条：单独编它只出一个符号，且那个符号原文就是它本身（不是 unk、不是带空白前缀
    的变体）。这是本域唯一裁决——训练配置写得再漂亮，这里过不了就算不合格。

    白话：五十二个字母挨个单独送进去试一遍，每个都只该弹出一个符号、而且弹出来的原文就是它
    自己；哪个被拼进了更长的条目里弹不出来，就把这些字母的名字记下来报错，绝不放行。
    """
    mapping: dict[str, int] = {}
    missing: list[str] = []
    detail: dict[str, str] = {}
    for letter in letters:
        ids = tokenizer.encode(letter, add_special_tokens=False).ids
        if len(ids) != 1:
            missing.append(letter)
            detail[letter] = f"编出 {len(ids)} 个符号: {[tokenizer.id_to_token(i) for i in ids]}"
            continue
        symbol = tokenizer.id_to_token(ids[0])
        if symbol != letter:
            missing.append(letter)
            detail[letter] = f"符号原文是 {symbol!r}，不是字母本身"
            continue
        mapping[letter] = ids[0]

    # 编号两两不同是"26 行读出"的前提（BPE 的词与编号本就一一对应，这里只作兜底）
    if len(set(mapping.values())) != len(mapping):
        seen: dict[int, str] = {}
        clash = [c for c in letters if c in mapping and seen.setdefault(mapping[c], c) != c]
        raise TokenizerCheckError(clash, {c: f"与 {seen[mapping[c]]} 撞同一编号" for c in clash})
    if missing:
        raise TokenizerCheckError(missing, detail)
    return mapping


def char_oov_rate(tokenizer: Tokenizer, text: str) -> float:
    """单字符覆盖率反面指标：文本里有多大比例的字符在表上找不到独立符号（会被丢掉）。

    真实语料冒烟用它抽查中/英两侧的覆盖情况，是"这张表能不能写中文"的量化证据。

    白话：拿一段文字逐个字查在册的单符号，数出有多少字根本查不到——查不到的比例越高，
    说明这张表对这类文字越生疏。
    """
    total = 0
    oov = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if tokenizer.token_to_id(ch) is None:
            oov += 1
    return (oov / total) if total else 0.0


def letter_rows(mapping: dict[str, int], letters: str = UPPERCASE) -> list[int]:
    """从字母映射里挑出 A–Z 的行号，按顺序排成决策读出用的 26 行索引。

    白话：五十二个字母的号都拿到了，这里只挑出大写那二十六号，按 A 到 Z 排成一列——
    读答案时就拿这一列去比对，不用每次再自己去表里翻。
    """
    return [mapping[c] for c in letters]
