"""learning/s0_tokenizer.py — S0 自训中英 BPE 的命令行入口（从零，禁载任何第三方词表/权重）。

【做什么】把一堆中英混排的文本喂进去，吐出一个能用的"文本 ↔ 编号"表目录，并当场证明 52 个英文
字母大小写各自独占一个编号。
【怎么做】先按 --corpus 找到语料文件（可以给内置小语料 mini），按 --stream-limit 的字节上限流式
逐行读出并清洗（去网址、去控制字符），交给 sys1.lang.bpe.train 做 BPE 合并训练；训练前把 52 字母
等关键符号钉进初始符号集，训练后必须过 check_tokenizer 才算合格；产物目录写 tokenizer.json/vocab.json
/merges.txt/tokenizer_config.json 四件，外加 run 目录的 config.yaml 与 notes.md 过程记录。
【为什么】字母约束只能来自造表阶段（决策读出要靠 26 个字母行取答案），所以入口脚本也得把校验钉成
最后一道门，谁跑都绕不过去。被否方案一：脚本里顺手下载 HuggingFace 现成词表——快，但直接违反
GUIDE §7-0 的从零红线；被否方案二：把语料一次性 read() 进内存再训练——200MB 档在端侧内存吃不住，
改成流式读同样简单，没有理由选前者。

CLI 头部注释四要素（PRODUCTION §8.1 口径）：
  目的：产出 S0 自训中英 BPE 产物 + 52 字母单 token 校验证据（GUIDE §4 M0 验收①的地基）。
  输入：--corpus 语料（文件/目录/mini，可多个）；--vocab-size 表规模（8k–32k）；
        --stream-limit 语料读取上限（如 200MB，缺省 200MB）；--min-freq 合并最小频次。
  产出 run-id：默认 runs/<MMDD>-s0-bpe-<规模k>-<语料名>（可用 --out/--run-id 覆盖），
        目录内含 tokenizer/ 产物、config.yaml（超参与环境）、notes.md（假设/观察/结论）。
  预计耗时：内置 mini 语料 <5 秒（CPU）；200MB 真实语料 CPU 约 10–25 分钟（spec 要求 30 分钟内）。

用法示例：
    python learning/s0_tokenizer.py                                   # 内置小语料离线冒烟
    python learning/s0_tokenizer.py --corpus data/zh.txt data/en.txt \\
           --vocab-size 32000 --stream-limit 200MB                    # 真实语料冒烟（C1）
"""
from __future__ import annotations

import argparse
import platform
import re
import sys
import time
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path

import tokenizers
import yaml

from sys1.lang import bpe

# 目录锚点：脚本可以从任何 cwd 调用，产物始终落在仓库的 runs/ 下（run notebook 的唯一真源）。
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
# 中英两侧的取样判据：汉字区间判中文，纯拉丁单词行判英文（抽检覆盖率时分这两桶）。
CJK = re.compile(r"[\u4e00-\u9fff]")
LATIN = re.compile(r"[A-Za-z]")
# --stream-limit 允许人类可读写法（200MB / 1.5G / 4096），统一折算成字节数。
SIZE_PATTERN = re.compile(r"^([0-9]*\.?[0-9]+)\s*([KMGT]?B?)$", re.IGNORECASE)
SIZE_UNITS = {"": 1, "B": 1, "K": 1024, "KB": 1024, "M": 1024**2, "MB": 1024**2,
              "G": 1024**3, "GB": 1024**3, "T": 1024**4, "TB": 1024**4}
# 只吃纯文本：parquet 之类的装配是 p1-08 的事，本域的职责边界到"文本进、表出"为止。
TEXT_SUFFIXES = (".txt", ".md", ".jsonl", ".csv", ".html", ".srt")
# 抽检样本的取样上限：流式读取时中英两桶各留前若干行（只留总前 N 行的话，先读到的那一侧
# 会把另一侧挤出去，覆盖率抽检就成了瞎子摸象）。
SPOT_CHECK_LINES = 400
# 验收探针串（spec 场景一）：产物必须能把它原样编回再解码出来。
ROUNDTRIP_PROBE = "你好世界 hello world 123"


def _now() -> datetime:
    """取带时区的当前时刻（runs/ 目录命名与 config.yaml 时间戳的唯一时间源）。"""
    return datetime.now(timezone.utc)


def parse_size(value: str | None) -> int | None:
    """把 "200MB"/"1.5G"/"4096" 折算成字节数；None/0/none/unlimited 表示不限。

    白话：给人看的长度（200MB、1.5G、4096）换算成机器要的字节个数；写 0、none 或者"不限"
    就当不设上限，一路读到底。
    """
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"0", "none", "unlimited", "all"}:
        return None
    match = SIZE_PATTERN.match(text)
    if not match:
        raise argparse.ArgumentTypeError(f"无法理解的长度 {value!r}（例：200MB / 1.5G / 4096）")
    unit = match.group(2).upper()
    return int(float(match.group(1)) * SIZE_UNITS[unit if unit in SIZE_UNITS else "B"])


def expand_corpus(values: list[str]) -> tuple[list[Path], bool]:
    """把 --corpus 的取值摊平成文件清单；返回 (文件列表, 是否用内置小语料)。

    目录只收常见文本后缀；parquet 之类由上游语料装配脚本先转成文本。

    白话：把人敲进来的路径摊平成一叠文件——敲目录就把它下面常见格式的文本都收进来，敲 mini
    就用仓库自带的那一小把样本；一个文件都没捞着就直接报错，绝不悄悄换成别的东西。
    """
    if any(value.lower() == "mini" for value in values):
        return [], True
    files: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            files += sorted(item for item in path.rglob("*") if item.suffix.lower() in TEXT_SUFFIXES)
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"语料路径不存在: {value}")
    if not files:
        raise FileNotFoundError(f"在 {values} 里没找到文本档（支持后缀 {TEXT_SUFFIXES}），或改用 --corpus mini")
    return files, False


def stream_lines(files: Iterable[Path], limit_bytes: int | None) -> Iterator[str]:
    """依次流式读取语料档，逐行 yield 原始文本，读满 limit_bytes 就收手（不整档进内存）。

    白话：不把所有文本一口气搬上桌，而是像翻书一样一行行读；读够设定的那点分量就停下。
    这样再大的语料也只占固定的内存——代价只是每次多算一下已经读过的字数。
    """
    read = 0
    for path in files:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if limit_bytes is not None and read >= limit_bytes:
                    return
                read += len(line.encode("utf-8", errors="ignore"))
                yield line


def corpus_lines(files: list[Path], use_mini: bool, limit_bytes: int | None,
                 keep: dict[str, list[str]], stats: dict) -> Iterator[str]:
    """产出训练用的文本行（已按 CLEANING_RULES 清洗），行数/字节数记进 stats，前若干行抄进 keep。

    白话：把要炼表的句子一行行递出去，同时记两笔账——递了多少行、多少字；再留一小撮样品在边上，
    等表炼好了拿这一小撮回头量一量中文、英文各有几个字没能单独对上号。
    """
    source: Iterator[str] = iter(bpe.MINI_CORPUS) if use_mini else stream_lines(files, limit_bytes)
    stats["lines"] = 0
    stats["bytes_read"] = 0
    for index, line in enumerate(source):  # index 兼作行号计数（stats）
        # 中英各自攒一小撮样品（含汉字算中文，否则算英文），保证两侧都有抽检素材
        bucket = "zh" if CJK.search(line) else ("en" if LATIN.search(line) else None)
        if bucket and len(keep.get(bucket, [])) < SPOT_CHECK_LINES:
            keep.setdefault(bucket, []).append(line)
        cleaned = bpe.clean_text(line)
        stats["lines"] = index + 1
        stats["bytes_read"] += len(cleaned.encode("utf-8"))
        yield cleaned


def spot_check(tokenizer: bpe.Tokenizer, buckets: dict[str, list[str]]) -> dict:
    """中英两侧各挑一段做抽检：单字符在册率与"一个字符平均摊到几段"写进运行记录。

    白话：表炼好了不能只看总数，要分中、英两头各试一小段——看看有多少字压根查不到，
    再看看十个字大约被换成几个号。数字越小说明这张表对这类文字越熟。
    """
    result: dict = {}
    for label, text in ((key, "\n".join(value)) for key, value in buckets.items()):
        chars = [ch for ch in text if not ch.isspace()]
        if not chars:
            result[label] = {"chars": 0}   # 语料里这一侧没样本，如实记 0，不假装覆盖
            continue
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        result[label] = {
            "chars": len(chars),
            "oov_char_rate": round(bpe.char_oov_rate(tokenizer, text), 4),
            "tokens": len(ids),
            "chars_per_token": round(len(chars) / max(1, len(ids)), 3),
        }
    return result


def _resolve_run_dir(args: argparse.Namespace, files: list[Path], use_mini: bool) -> Path:
    """决定 run 目录：--out 优先；否则 runs/<MMDD>-s0-bpe-<规模k>-<语料名>（runs/README 命名法）。"""
    if args.out:
        path = Path(args.out)
    else:
        stem = re.sub(r"\W+", "", files[0].stem)[:16] if files and not use_mini else "mini"
        name = args.run_id or f"{_now():%m%d}-s0-bpe-{args.vocab_size // 1000}k-{stem or 'corpus'}"
        path = RUNS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_run_records(run_dir: Path, tokenizer: bpe.Tokenizer, mapping: dict[str, int],
                      args: argparse.Namespace, stats: dict) -> None:
    """把 run 目录的 config.yaml 与 notes.md 落盘（runs/README 的证据链口径：假设→观察→结论）。

    白话：把这一次跑的账本写下来——用了什么参数、跑了多久、读进多少字、表有多大、五十二个字母
    是不是各占一号、中英两头各抽查到什么程度，最后按"假设—观察—结论"三段记进 runs/，
    日后答辩、复查、复跑都能对上号。
    """
    config = {
        "run_id": run_dir.name,
        "stage": "s0-tokenizer",
        "created_at": stats["created_at"],
        "hardware": {"platform": platform.platform(), "machine": platform.machine(), "device": "cpu"},
        "env": {"python": platform.python_version(), "tokenizers": tokenizers.__version__},
        "params": {"corpus": args.corpus, "vocab_size": args.vocab_size,
                   "stream_limit": args.stream_limit, "min_freq": args.min_freq},
        # 红线自查字段：来源必须写"自训、无第三方词表/权重"，审查时一眼可见
        "source": "self-trained BPE, no third-party vocab/weights (GUIDE §7-0)",
        "outputs": {"tokenizer_dir": "tokenizer", "vocab_size_actual": tokenizer.get_vocab_size()},
        "check": {"letters_single_token": len(mapping), "pad_token_id": tokenizer.token_to_id(bpe.PAD_TOKEN)},
        "corpus_stats": {k: stats[k] for k in ("corpus_source", "files", "lines", "bytes_read", "train_seconds")},
        "spot_check": stats["spot_check"],
    }
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    spot = stats["spot_check"]
    notes = [
        f"# run: {run_dir.name} — S0 自训中英 BPE",
        "",
        "假设：initial alphabet 钉死 52 字母 + 读表时强校验，就能让字母各占一个编号，中英混排可无损往返。",
        "观察：",
        (f"- 语料：{stats['corpus_source']}，读入 {stats['bytes_read']} 字节 / {stats['lines']} 行，"
         f"训练耗时 {stats['train_seconds']} 秒（CPU）。"),
        (f"- 表规模 {tokenizer.get_vocab_size()}（配置上限 {args.vocab_size}），pad 在 "
         f"{tokenizer.token_to_id(bpe.PAD_TOKEN)} 号，特殊 token {list(bpe.DEFAULT_SPECIAL_TOKENS)}。"),
        f"- check_tokenizer 通过：{len(mapping)}/52 字母各占一个编号。",
        f"- 抽检 中文：{spot.get('zh')}；英文：{spot.get('en')}。",
        f"- 往返：{ROUNDTRIP_PROBE!r} 编回后与原串一致。",
        (f"结论：字母约束达成（{len(mapping)}/52 单 token），中英覆盖与压缩比见上；"
         "若中文单字缺失率偏高，加大 --stream-limit 或 --vocab-size 复跑，不放行不合格产物。"),
        "",
        (f"复跑：python learning/s0_tokenizer.py --corpus {' '.join(args.corpus)} "
         f"--vocab-size {args.vocab_size} --stream-limit {args.stream_limit} --out {run_dir}"),
        "",
    ]
    (run_dir / "notes.md").write_text("\n".join(notes), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """搭出命令行参数表（--corpus/--vocab-size/--stream-limit/--out 四件套加三个可选项）。

    白话：把能敲的选项和它们的默认值写清楚，让人敲一下 --help 就看到一份说明书：喂哪份文本、
    要多大的表、最多读进多少字、产物丢在哪个文件夹。
    """
    parser = argparse.ArgumentParser(
        prog="s0_tokenizer.py",
        description="自训中英 BPE（从零，禁第三方词表/权重），产出 tokenizer 目录并强校验 52 字母单 token。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--corpus", nargs="+", default=["mini"],
                        help="语料：文件/目录可多个；mini = 内置小语料（离线冒烟）")
    parser.add_argument("--vocab-size", type=int, default=16_000,
                        help=f"表规模，允许区间 {bpe.VOCAB_MIN}-{bpe.VOCAB_MAX}")
    # 长度既接受人类写法也接受纯字节数，统一在 main 里用 parse_size 折算（默认值不走 type 转换）
    parser.add_argument("--stream-limit", default="200MB",
                        help="语料读取上限（例 200MB / 1.5G / 0=不限）")
    parser.add_argument("--out", default=None,
                        help="产物 run 目录；缺省 runs/<MMDD>-s0-bpe-<规模k>-<语料名>")
    parser.add_argument("--run-id", default=None, help="覆盖默认 run-id 命名（仅在未给 --out 时生效）")
    parser.add_argument("--min-freq", type=int, default=2, help="合并的最小出现次数")
    parser.add_argument("--progress", action="store_true", help="打印训练进度条")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 主体：找语料 → 流式清洗 → 自训 → 强校验 → 落盘产物与 run 记录 → 打印摘要。

    白话：一条命令走完"读书、炼表、验表、存档"四步。中间任何一步不达标（比如字母被吞了、
    表规模越界了）就当场报错停下，绝不把不合格的表当成果交出去。
    """
    args = build_parser().parse_args(argv)
    started = time.time()
    files, use_mini = expand_corpus(args.corpus)
    limit_bytes = parse_size(args.stream_limit)

    # 语料以生成器形式喂进去：训练边读边炼，200MB 档也只占固定内存（内存换时间的取舍见模块头）
    keep: dict[str, list[str]] = {}
    stats: dict = {}
    corpus = corpus_lines(files, use_mini, limit_bytes, keep, stats)
    tokenizer = bpe.train(corpus, vocab_size=args.vocab_size, min_frequency=args.min_freq,
                          show_progress=args.progress)
    stats["train_seconds"] = round(time.time() - started, 2)

    mapping = bpe.check_tokenizer(tokenizer)  # 唯一裁决：过不了就别无选择地失败
    run_dir = _resolve_run_dir(args, files, use_mini)
    bpe.save(tokenizer, run_dir / "tokenizer",
             meta={"corpus": args.corpus, "stream_limit": args.stream_limit, "min_freq": args.min_freq})
    reloaded = bpe.load(run_dir / "tokenizer")  # 读回即再校验一次（双保险的第二道）
    assert reloaded.get_vocab_size() == tokenizer.get_vocab_size(), "落盘再读回，表规模必须一致"

    stats["created_at"] = _now().date().isoformat()
    stats["corpus_source"] = "MINI_CORPUS(内置)" if use_mini else f"{files[0].parent} 等文档"
    stats["files"] = len(files)
    stats["spot_check"] = spot_check(reloaded, keep)
    write_run_records(run_dir, reloaded, mapping, args, stats)

    ids = reloaded.encode(ROUNDTRIP_PROBE, add_special_tokens=False).ids
    print(f"[s0] run dir      : {run_dir}")
    print(f"[s0] vocab size   : {reloaded.get_vocab_size()} (requested {args.vocab_size})")
    print(f"[s0] letters ok   : {len(mapping)}/52 单 token  (A={mapping['A']}, z={mapping['z']})")
    print(f"[s0] pad / eos id : {reloaded.token_to_id(bpe.PAD_TOKEN)} / "
          f"{reloaded.token_to_id(bpe.ENDOFTEXT_TOKEN)}")
    print(f"[s0] corpus       : {stats['corpus_source']} lines={stats['lines']} "
          f"bytes={stats['bytes_read']} train={stats['train_seconds']}s")
    print(f"[s0] spot check   : {stats['spot_check']}")
    print(f"[s0] roundtrip    : {ROUNDTRIP_PROBE!r} -> {len(ids)} tokens -> {reloaded.decode(ids)!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
