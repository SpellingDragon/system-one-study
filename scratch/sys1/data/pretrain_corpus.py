"""S1 下一词预训练语料装配器：流式取数（ModelScope 优先 / HF 兜底）→ 去重 → 分片编号流。

【做什么】把中英两边的教育类文本一行行读进来，去掉重复的稿子，换成一串可直接随机取用的
编号，写成一块块二进制片，让后面的预训练脚本不必再碰原文、也不必在循环里现场做字符串变换。
【怎么做】① 取数：每个来源先在配置给的本地路径找文件（存在即用，避免重复下载）；本地没有
才走远端——ModelScope 优先、HuggingFace 兜底；远端按"一份分片文件下到缓存目录 → 逐行读出
文本"推进，已下过的文件直接复用（断点续拉：中途断了重跑不会重复下载）。② 配比混合：中英
两条流按配置权重（默认 7:3）以"谁欠得多先取谁"的方式轮转出文档，读满 stream_limit 的字节
数、或攒够 max_tokens 的编号数即收手；配比按最终编号数计（按字数配会被中英压缩差带偏）。
③ 去重：先按规范化文本的 sha1 精确去重，再按 minhash-lite（num_perm 条带各取最小哈希，
全条带相同即判近重复）抓粘贴稿；两条规则都写死在配置里、可关。④ 换编号：用 sys1.lang.bpe
自训产物（GUIDE §7-0 红线：绝不引第三方对照表）把文档换成编号序列，文档之间插一个整篇结束
标记。⑤ 落盘：每满 shard_tokens 个编号写一个 uint32 二进制片，末尾附 index.json 记录片长、
中英编号占比、去重计数与表规模——训练侧只认这份清单，零拷贝随机取窗口。
【为什么】被否方案一：把整档文本 read() 进内存再一次性换编号——200MB 档在端侧内存吃不住，
而且装配是一次性的、训练是反复的，成本留在装配侧才对。被否方案二：直接载别人现成的对照表
换编号——省事但违反 §7-0 从零红线，且字母会被前导空格变体吃掉，下游 26 行读出的答案就
错位。被否方案三：完整 minhash（128 条带 + band/row 的 LSH 分桶）——召回更好，但一阶段只
处理 200MB 档，为它引入新依赖不划算，故用条带更少的 minhash-lite 并在配置里留 num_perm
旋钮。被否方案四：只落一个巨型连续文件——单文件上百 MB 时页缓存局部性变差，坏一次全废；
分片可逐片校验、也能分段续装（design 风险项）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sys1.lang import bpe

# ── 目录锚点：脚本可从任何 cwd 调用，默认配置与缓存都以仓库根为准 ─────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "learning" / "configs" / "corpus.yaml"  # 配比与去重规则的唯一真源
CACHE_DIR = REPO_ROOT / "bench" / "corpus_s1"                     # 远端分片文件的本地缓存目录

# ── 落盘格式（名字即约定：tests/test_corpus.py 与训练侧都按这套拼路径）────────────────
INDEX_FILE = "index.json"
SHARD_PREFIX = "shard-"
SHARD_SUFFIX = ".bin"
SHARD_PATTERN = re.compile(r"^shard-\d{5}\.bin$")
FORMAT_NAME = "uint32-memmap-v1"      # 元素类型 + 布局版本；改布局必须升这个串
DTYPE = np.uint32
DEFAULT_DOC_SEP_TOKEN = bpe.ENDOFTEXT_TOKEN  # 文档之间插这个整篇结束标记（自训表固定 1 号）
CJK = re.compile(r"[\u4e00-\u9fff]")          # 汉字区间：本地 txt 没有 lang 标签时用它分桶
LATIN = re.compile(r"[A-Za-z]")               # 纯拉丁行判英文（与 s0 的抽检口径一致）

# ── 远端拉取：ModelScope 优先、HuggingFace 兜底（design 的双源约束）────────────────────
MODELSCOPE_FILES_API = "https://www.modelscope.cn/api/v1/datasets/{id}/repo?Revision={rev}&FilePath="
MODELSCOPE_RAW_URL = "https://www.modelscope.cn/datasets/{id}/resolve/{rev}/{path}"
HUGGINGFACE_RAW_URL = "https://huggingface.co/datasets/{id}/resolve/{rev}/{path}"
HTTP_TIMEOUT_SEC = 60                 # 单次网络动作的上限；超时按源切换，不死等
PROGRESS_EVERY_TOKENS = 500_000       # 装配进度打印的节流粒度（200MB 档也能看清走到哪）

# 人类可读写法（200MB / 1.5G / 4096）统一折算成字节数，与 s0 入口同一套口径
SIZE_PATTERN = re.compile(r"^([0-9]*\.?[0-9]+)\s*([KMGT]?B?)$", re.IGNORECASE)
SIZE_UNITS = {"": 1, "B": 1, "K": 1024, "KB": 1024, "M": 1024**2, "MB": 1024**2,
              "G": 1024**3, "GB": 1024**3, "T": 1024**4, "TB": 1024**4}


class CorpusError(RuntimeError):
    """装配环节的可预期失败（配置缺失、下限不达标、表规模不匹配），统一一种异常好兜。"""
# ══ A1-1 配置：所有配比/去重/上限规则显式入配置，代码里不留第二个来源 ═══════════════════

def parse_size(value: Any) -> int | None:
    """把 "200MB"/"1.5G"/4096 这类写法折算成字节数；None/0/none/unlimited 表示不限。

    白话：给人看的长度写法换算成机器要的字节个数。写 0、none 或者"不限"，就当压根没设上限，
    一路读到底；写不明白的串直接报错，绝不悄悄当成"不限"蒙过去。
    """
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"0", "none", "unlimited", "all"}:
        return None
    match = SIZE_PATTERN.match(text)
    if not match:
        raise CorpusError(f"无法理解的长度 {value!r}（例：200MB / 1.5G / 4096）")
    unit = match.group(2).upper()
    return int(float(match.group(1)) * SIZE_UNITS[unit if unit in SIZE_UNITS else "B"])


def load_spec(config: str | Path, *, stream_limit: Any = None, max_tokens: int | None = None,
              tokenizer_dir: str | None = None, prefer: str = "auto") -> dict[str, Any]:
    """读装配配置并合并覆盖项，返回字段齐全、可直接执行的一份规格字典。

    config 可以是配置单里的档位名（如 tiny / main），也可以直接给一个 .yaml 路径。
    装配规则一律以 learning/configs/corpus.yaml 为准：代码只负责照单子执行，不改单子的内容。

    白话：先去配置单里挑出这一档要用的规矩——中英各占几成、最多读进来多少字、最多攒多少个
    编号、怎么去重、产物切成多大的片；再把外面临时交代的改动一样样盖上去。哪一项在单子里
    查不到就直接报错停下，绝不带着一份半懂的单子开工。
    """
    raw = _read_config_file(config)
    spec: dict[str, Any] = dict(raw.get("defaults") or {})
    spec.update(raw.get("profile") or {})
    spec.setdefault("profile", str(config))
    if stream_limit is not None:
        spec["stream_limit_bytes"] = parse_size(stream_limit)
    if max_tokens is not None:
        spec["max_tokens"] = int(max_tokens)
    if tokenizer_dir:
        spec["tokenizer_dir"] = tokenizer_dir
    for key in ("tokenizer_dir", "vocab_size", "max_tokens", "shard_tokens", "sources"):
        if key not in spec:
            raise CorpusError(f"配置缺字段 {key!r}（应在 {CONFIG_PATH} 的 defaults/profiles 里给出）")
    spec["prefer"] = prefer if prefer != "auto" else spec.get("prefer", "auto")
    spec["stream_limit_bytes"] = parse_size(spec.get("stream_limit_bytes"))
    spec["tokenizer_dir"] = str(_abs(spec["tokenizer_dir"]))
    spec["cache_dir"] = str(_abs(spec.get("cache_dir") or CACHE_DIR))
    if not spec["sources"]:
        raise CorpusError("sources 为空：至少要有中英两个来源才能配出 7:3")
    _check_sources(spec)
    spec.setdefault("min_tokens_expected", max(1, int(spec["max_tokens"] * 0.9)))
    spec.setdefault("dedup", {"exact": True, "minhash_lite": {"num_perm": 8, "shingle_chars": 8, "stride": 4}})
    spec.setdefault("doc_sep", DEFAULT_DOC_SEP_TOKEN)
    return spec


def _read_config_file(config: str | Path) -> dict[str, Any]:
    """把"档位名或 yaml 路径"解析成 {defaults, profile}；档位查不到就报错并列出可用档位。"""
    path = Path(str(config))
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return {"defaults": data, "profile": {}}
    if not CONFIG_PATH.exists():
        raise CorpusError(f"配置单不存在: {CONFIG_PATH}")
    doc = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    profiles = doc.get("profiles") or {}
    if str(config) not in profiles:
        raise CorpusError(f"配置单里没有档位 {config!r}，可用档位: {sorted(profiles)}")
    return {"defaults": doc.get("defaults") or {}, "profile": profiles[str(config)]}


def _check_sources(spec: dict[str, Any]) -> None:
    """来源自检：权重之和为 1（±1e-6）、每来源至少有一条能走通的路（本地文件或远端库名）。"""
    weights = [float(s.get("weight", 0)) for s in spec["sources"]]
    if abs(sum(weights) - 1.0) > 1e-6:
        raise CorpusError(f"中英权重之和必须是 1，实得 {sum(weights):.4f}（配比 7:3 写成 0.7/0.3）")
    for src in spec["sources"]:
        if not (src.get("local") or src.get("modelscope") or src.get("huggingface") or src.get("docs")):
            raise CorpusError(f"来源 {src.get('lang')} 既没给本地路径也没给远端库名，取不到文本")
        if src.get("lang") not in {"en", "zh"}:
            raise CorpusError(f"来源 lang 只支持 en/zh（配比统计按这两桶计），实得 {src.get('lang')!r}")


def _abs(path_like: Any) -> Path:
    """相对路径一律相对仓库根展开：脚本从任何 cwd 调用，产物落点都一样。"""
    p = Path(str(path_like)).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def _reanchor(path_str: Any) -> Any:
    """index.json 里历史绝对路径的迁移自愈：仓库改名/搬迁后按 '/runs/' 后缀重锚当前仓库根。

    背景：早期装配把 tokenizer_dir 以当时绝对路径烧进 index.json，仓库两次改名均致断链
    （e2e 测试实踩）；原样路径存在则直用，不存在则取 '/runs/' 之后段重拼当前仓库根
    （兼容旧版'仓根/runs/...'与新版'仓根/scratch/runs/...'两形态，两者 '/runs/' 后缀一致）。
    """
    if not isinstance(path_str, str) or not path_str:
        return path_str
    if Path(path_str).exists():
        return path_str
    marker = "/runs/"
    if marker in path_str:
        cand = (REPO_ROOT / path_str[path_str.index(marker) + 1:]).resolve()
        if cand.exists():
            return str(cand)
    return path_str



# ══ A1-2 取数：本地优先（避免重复下载），远端 ModelScope 优先 / HF 兜底 ══════════════════

def iter_text_docs(path: Path) -> Iterator[str]:
    """逐行流式读一个纯文本档并清洗，一行当一篇文档（本仓语料就是"一行一篇"的导出格式）。

    白话：像翻书一样一行行往下读，读一行洗一行（去网址、去乱码），不把所有内容一口气搬上桌；
    这样再大的文本也只占固定的一点内存，代价只是每行多算一次长度。
    """
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            doc = bpe.clean_text(line).strip()
            if doc:
                yield doc


def iter_parquet_docs(path: Path, text_key: str = "text") -> Iterator[str]:
    """按 row group 流式读一个 parquet 分片，取其中文本字段逐条吐出来（整块读进内存会爆）。

    白话：远端下来的那种压缩包只能一箱一箱开箱，每箱里再一行行取正文；一次只搬一箱上桌，
    读完就松手，所以一个几 GB 的文件也能在这台小机器上过完，不用先腾出同等大小的桌面。
    """
    import pyarrow.parquet as pq

    reader = pq.ParquetFile(str(path))
    for batch in reader.iter_batches(batch_size=1024, columns=[text_key]):
        for value in batch.column(text_key).to_pylist():
            if not value:
                continue
            doc = bpe.clean_text(str(value)).strip()
            if doc:
                yield doc


def docs_from_local(source: dict[str, Any]) -> Iterator[str]:
    """按来源配置里的 local 字段（文件或目录）找到文本并流式吐出；找不到返回空迭代。

    白话：照单子上写的本地路径去找现成的文本，目录就按常见后缀收齐，parquet 压缩包按箱打开；
    没找到就给个空的流水，让上层自己去决定要不要转而走网络。
    """
    target = _abs(source["local"])
    if not target.exists():
        return iter(())
    files: list[Path] = []
    if target.is_dir():
        files = sorted(p for p in target.rglob("*") if p.suffix.lower() in {".txt", ".md", ".parquet", ".jsonl"})
    else:
        files = [target]
    key = source.get("text_key", "text")
    for file in files:
        if file.suffix.lower() == ".parquet":
            yield from iter_parquet_docs(file, key)
        else:
            yield from iter_text_docs(file)


def list_remote_files(source: dict[str, Any]) -> list[str]:
    """列远端数据集里的分片文件名：ModelScope 优先、HuggingFace 兜底，两边都失败才报错。

    白话：先问第一个仓库"你那儿有哪几份文件"，问不着再问第二个；两边都不肯说（网断了、库名
    写错了、这库不给公开目录）就老实报错，并在单子里留一条后路——配置里可以直接写死文件名
    清单，跳过这一步。
    """
    if source.get("files"):
        return [str(f) for f in source["files"]]
    errors: list[str] = []
    for api_id, rev in _remote_candidates(source):
        url = MODELSCOPE_FILES_API.format(id=api_id, rev=rev) if "modelscope" in api_id else None
        if url:
            try:
                payload = json.loads(_http_get(url))
            except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
                errors.append(f"modelscope:{api_id} → {exc}")
                continue
            names = [str(item.get("Path")) for item in (payload.get("Data") or {}).get("Tree", [])
                     if str(item.get("Path", "")).endswith((".parquet", ".txt", ".jsonl"))]
            if names:
                return sorted(names)
            errors.append(f"modelscope:{api_id} → 目录里没有可读的分片")
    errors.append("（如需跳过列举，在配置的 files 里直接写分片文件名）")
    raise CorpusError("远端文件列举失败: " + "; ".join(errors))


def download_shard(source: dict[str, Any], filename: str, cache_dir: Path) -> Path:
    """把一个远端分片下到缓存目录；已下过且非空就直接复用（断点续拉），下载走双源回退。

    白话：要用的那份文件先在自家缓存文件夹里找一遍，找到就不再惊动网络——中途断了重跑时，
    前面已经搬完的那几箱不会重搬一遍。确实没有才去下：先试优先的那个仓库，试不通换兜底的，
    先写临时名、下完整了才改名，免得半截文件被下一次的"已存在"当成好的复用。
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / source["lang"] / filename
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    errors: list[str] = []
    for url in _raw_urls(source, filename):
        try:
            _http_to_file(url, tmp)
            _atomic_move(tmp, dest)
            return dest
        except (urllib.error.URLError, OSError) as exc:
            errors.append(f"{url.split('/datasets/')[0]} → {exc}")
            if tmp.exists():
                tmp.unlink()
    raise CorpusError(f"分片 {filename} 两个源都没下到: " + "; ".join(errors))


def _raw_urls(source: dict[str, Any], filename: str) -> list[str]:
    """按"ModelScope 优先、HuggingFace 兜底"的顺序拼出这份文件的下载地址（跳过没配的源）。"""
    urls: list[str] = []
    if source.get("modelscope"):
        urls.append(MODELSCOPE_RAW_URL.format(id=source["modelscope"], rev=source.get("revision", "master"),
                                              path=filename))
    if source.get("huggingface"):
        urls.append(HUGGINGFACE_RAW_URL.format(id=source["huggingface"], rev=source.get("revision", "main"),
                                               path=filename))
    return urls


def _remote_candidates(source: dict[str, Any]) -> list[tuple[str, str]]:
    """列举用的 (库名, 版本) 候选，顺序与 _raw_urls 一致；只有 HuggingFace 时交兜底分支处理。"""
    out: list[tuple[str, str]] = []
    if source.get("modelscope"):
        out.append((source["modelscope"], source.get("revision", "master")))
    return out


def _http_get(url: str) -> str:
    """取一个 URL 的正文（文本），带超时；网络问题一律抛给调用方去切源。"""
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_SEC) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _http_to_file(url: str, dest: Path) -> None:
    """边下边写把 URL 内容落到 dest（分块读写，不把整份文件压在内存里）。"""
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_SEC) as resp, dest.open("wb") as sink:
        while chunk := resp.read(1 << 20):
            sink.write(chunk)


def _atomic_move(src: Path, dest: Path) -> None:
    """原子-ish 改名（同目录内 rename）：临时名 → 正式名，供断点续拉判"这份已完整"。"""
    src.replace(dest)


# ══ A1-3 去重：精确全文哈希 + minhash-lite，规则全部来自配置 ════════════════════════════

class Deduper:
    """两层去重器：先按规范化全文哈希精确判重，再按 minhash-lite 抓"只差一点"的近重复。

    条带数（num_perm）、指纹长度（shingle_chars）与取样步长都从配置读；关掉任一层只影响
    召回，不影响其余流程。被拦下的篇数一律记账，写进 index.json 供事后复查。
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.exact = bool(cfg.get("exact", True))
        lite = cfg.get("minhash_lite") or {}
        self.num_perm = int(lite.get("num_perm", 0))
        self.shingle_chars = int(lite.get("shingle_chars", 8))
        self.stride = max(1, int(lite.get("stride", 4)))
        self._exact_seen: set[str] = set()
        self._lite_seen: set[tuple[int, ...]] = set()
        self.stats = {"seen": 0, "dup_exact": 0, "dup_near": 0}

    def duplicate(self, text: str) -> bool:
        """这篇是否已见过；见过就记一笔账，绝不静默吞掉（去重数量必须能复查）。

        白话：先把整篇压成一句"去掉空白、不分大小写"的紧凑话，给它算个指纹——指纹早出现过，
        就是同一篇稿子抄了两遍。再把这句紧凑话切成一段一段，每段各挑一个数出来凑成一串号；
        这串号整串撞上了，就当"只差一点的重复稿"也拦下，但只拦一次，别把只改了两个字的稿子
        当成新内容反复喂进去。拦了几篇、为什么拦，全记在账上。
        """
        self.stats["seen"] += 1
        if not self.enabled:
            return False
        norm = _normalize(text)
        if self.exact:
            digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()
            if digest in self._exact_seen:
                self.stats["dup_exact"] += 1
                return True
            self._exact_seen.add(digest)
        if self.num_perm > 0:
            signature = self._signature(norm)
            if signature in self._lite_seen:
                self.stats["dup_near"] += 1
                return True
            self._lite_seen.add(signature)
        return False

    def _signature(self, norm: str) -> tuple[int, ...]:
        """minhash-lite：每条带用不同盐对同一段落取样取最小哈希，拼成一条可比较的签名。"""
        shingles = self._shingles(norm)
        if not shingles:
            return ()
        out: list[int] = []
        for perm in range(self.num_perm):
            best = min(_hash32(perm, item) for item in shingles)
            out.append(best)
        return tuple(out)

    def _shingles(self, norm: str) -> list[str]:
        """按 shingle_chars 定长切、stride 定步取样；太短的文本整篇当一段。"""
        size, step = self.shingle_chars, self.stride
        if len(norm) <= size:
            return [norm]
        return [norm[i:i + size] for i in range(0, len(norm) - size + 1, step)]


def _normalize(text: str) -> str:
    """规范化：压掉所有空白并转小写——只做判重用的形态收敛，不改原文内容。"""
    return re.sub(r"\s+", "", text).lower()


def _hash32(salt: int, text: str) -> int:
    """把 (盐, 片段) 混成一个 32 位整数：同一片段在同一条带里永远得同一个数。"""
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=4,
                                          key=str(salt).encode()).digest(), "big")


# ══ A1-4 混合取数：按编号数配比（默认 7:3），"谁欠得多先取谁" ════════════════════════════

def _open_source(source: dict[str, Any], spec: dict[str, Any]) -> Iterator[str]:
    """给一个来源开流水：本地文件存在就用本地（避免重复下载），否则按需下分片再逐条读。"""
    if source.get("docs"):
        return _docs_inline(source["docs"])
    prefer = spec.get("prefer", "auto")
    local_ok = prefer in {"auto", "local"} and bool(source.get("local"))
    if local_ok:
        stream = docs_from_local(source)
        first = _peek(stream)
        if first is not None:
            return _chain_first(first, stream)
        if prefer == "local":
            raise CorpusError(f"来源 {source.get('lang')} 指定只用本地，但 {source['local']} 里没有文本")
    return _remote_stream(source, spec)


def _docs_inline(items: Iterable[Any]) -> Iterator[str]:
    """把配置里直接内联的文档清单变成流水（单测与离线小语料走这条路，完全不碰网络）。"""
    if isinstance(items, str):           # "mini" 这个特殊值 = 内置小语料（sys1.lang.bpe 自带）
        if items.lower() == "mini":
            return (bpe.clean_text(text) for text in bpe.MINI_CORPUS)
        items = [items]
    return (bpe.clean_text(str(text)) for text in items if str(text).strip())


def _remote_stream(source: dict[str, Any], spec: dict[str, Any]) -> Iterator[str]:
    """远端流式装配：逐份分片"下载到缓存 → 读正文 → 交出去"，读够上限就不再下新的。"""
    cache_dir = Path(spec["cache_dir"])
    key = source.get("text_key", "text")
    for filename in list_remote_files(source):
        path = download_shard(source, filename, cache_dir)
        if path.suffix.lower() == ".parquet":
            yield from iter_parquet_docs(path, key)
        else:
            yield from iter_text_docs(path)


def _peek(stream: Iterator[str]) -> str | None:
    """试探性取第一条：有内容才认定这条流水可用（本地路径存在但是空文件不算）。"""
    return next(stream, None)


def _chain_first(first: str, rest: Iterator[str]) -> Iterator[str]:
    """把试探取出的第一条还回流水头部，保证不丢第一篇。"""
    yield first
    yield from rest


def _pick_deficient(streams: list[dict[str, Any]]) -> dict[str, Any] | None:
    """挑"离自己该占的比例还差得最多"的那条流；全取完返回 None。

    比例按已产出的编号数算，不是按篇数也不是按字数：中英压出来的长度差得很远，按篇数轮转
    会把配比漂到六四甚至更偏，配置写的 7:3 就成了摆设。
    """
    live = [s for s in streams if not s["done"]]
    if not live:
        return None
    total = sum(s["tokens"] for s in streams) or 1
    return max(live, key=lambda s: s["weight"] * total - s["tokens"])


# ══ A1-5 装配主流程：混合 → 去重 → 换编号 → 分片落盘 + index.json ══════════════════════

class _ShardWriter:
    """按固定片长落 uint32 二进制片的写手：一片写满就换下一片，跨片的文档自然续写到下一片。

    预分配一块 shard_tokens 大小的缓冲反复复用，所以整个装配过程的常驻内存只跟片长有关、
    跟语料总量无关（这是"200MB 档在端侧跑得动"的前提）。
    """

    def __init__(self, out_dir: Path, shard_tokens: int, vocab: int) -> None:
        self.out_dir = out_dir
        self.shard_tokens = int(shard_tokens)
        self.vocab = int(vocab)
        self.buf = np.empty(self.shard_tokens, dtype=DTYPE)
        self.pos = 0
        self.index = 0
        self.shards: list[dict[str, Any]] = []
        self.current_langs: dict[str, int] = {"en": 0, "zh": 0, "other": 0}
        self.totals = {"tokens": 0, "id_min": None, "id_max": None}

    def push(self, ids: list[int], lang: str) -> int:
        """把一串编号接进当前片，片满就地落盘换新片；返回实际写进去的个数。

        白话：往当前这本账上继续记号；这本记满了就合上、另起一本接着记，一篇稿子跨在两本上
        也接得上，不会为了凑整本把中间一段丢掉。交回真正记进去的个数，好让上层知道攒到哪了。
        """
        if not ids:
            return 0
        self._guard(ids)
        written = 0
        cursor = 0
        while cursor < len(ids):
            room = self.shard_tokens - self.pos
            take = min(room, len(ids) - cursor)
            chunk = ids[cursor:cursor + take]
            self.buf[self.pos:self.pos + take] = np.asarray(chunk, dtype=DTYPE)
            self.pos += take
            cursor += take
            written += take
            self._tally(chunk, lang)
            if self.pos >= self.shard_tokens:
                self._flush()
        return written

    def close(self) -> list[dict[str, Any]]:
        """收尾：把最后一片的残段截齐落盘，返回全部片的清单（进 index.json）。

        白话：最后没写满的那本也要按实际行数裁齐后合上——不裁齐的话，读的人会按整本的长度
        去数，把末尾后面那点空位当成内容。裁完把每本多长、各侧占了几成都交出去当总清单。
        """
        if self.pos:
            self._flush()
        return self.shards

    def _flush(self) -> None:
        name = f"{SHARD_PREFIX}{self.index:05d}{SHARD_SUFFIX}"
        self.buf[:self.pos].tofile(self.out_dir / name)
        self.shards.append({"file": name, "tokens": int(self.pos), "lang_tokens": dict(self.current_langs)})
        self.index += 1
        self.current_langs = {"en": 0, "zh": 0, "other": 0}
        self.pos = 0

    def _tally(self, chunk: list[int], lang: str) -> None:
        bucket = lang if lang in self.current_langs else "other"
        self.current_langs[bucket] += len(chunk)
        self.totals["tokens"] += len(chunk)
        lo, hi = min(chunk), max(chunk)
        self.totals["id_min"] = lo if self.totals["id_min"] is None else min(self.totals["id_min"], lo)
        self.totals["id_max"] = hi if self.totals["id_max"] is None else max(self.totals["id_max"], hi)

    def _guard(self, ids: list[int]) -> None:
        """越界自检：任一编号 >= 表规模或为负即中止——宁可装配失败，不许把坏编号写进流里。"""
        lo, hi = min(ids), max(ids)
        if lo < 0 or hi >= self.vocab:
            raise CorpusError(f"编号越界（区间 [{lo}, {hi}]，表规模 {self.vocab}）；装配中止，不落坏片")


def build(spec: dict[str, Any], out: str | Path, *, tokenizer: Any = None,
          verbose: bool = True) -> dict[str, Any]:
    """装配主入口：混合取数 → 去重 → 换成编号 → 分片落盘 + index.json，返回那份索引字典。

    停止条件有三个，先到先收手：攒够 max_tokens 个编号、读满 stream_limit_bytes 的文本字节、
    或所有来源都取空。收尾时若产出低于配置写的 min_tokens_expected 下限，直接报错而不交付
    "看着齐活其实只有一半"的语料——下游按片取窗口，短了会把配比假象一路带进训练。

    白话：把要读书的几摊各开一条流水，哪一摊离自己该占的份数差得最多就先取它一篇；取来的
    稿子先跟已经读过的对一遍（一模一样的、只差一点的都算重复，丢掉），再换成一串号，号与号
    之间留一个"这一篇完了"的记号，攒满一格的长度就另起一本账本继续写。写满规定的本数或读到
    设定的分量就收工，最后另存一张总清单：每本多长、中英各占几成、丢了多少重复稿。
    """
    started = time.time()
    out_dir = _abs(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = tokenizer if tokenizer is not None else bpe.load(spec["tokenizer_dir"])
    vocab = int(spec["vocab_size"])
    if tok.get_vocab_size() != vocab:
        raise CorpusError(f"表规模不符：配置写 {vocab}，实际读到的表有 {tok.get_vocab_size()} 行")
    sep_id = int(tok.token_to_id(spec["doc_sep"]))
    writer = _ShardWriter(out_dir, int(spec["shard_tokens"]), vocab)
    deduper = Deduper(spec["dedup"])
    limit = spec.get("stream_limit_bytes")
    max_tokens = int(spec["max_tokens"])

    streams: list[dict[str, Any]] = []
    for source in spec["sources"]:
        streams.append({"lang": source["lang"], "weight": float(source["weight"]),
                        "it": _open_source(source, spec), "tokens": 0, "bytes": 0, "done": False})

    read_bytes = 0
    docs_seen = 0
    next_mark = PROGRESS_EVERY_TOKENS
    while writer.totals["tokens"] < max_tokens:
        if limit is not None and read_bytes >= limit:
            break
        stream = _pick_deficient(streams)
        if stream is None:
            break
        doc = next(stream["it"], None)
        if doc is None:
            stream["done"] = True
            continue
        docs_seen += 1
        nbytes = len(doc.encode("utf-8"))
        read_bytes += nbytes
        stream["bytes"] += nbytes
        if deduper.duplicate(doc):
            continue
        ids = tok.encode(doc, add_special_tokens=False).ids
        room = max_tokens - writer.totals["tokens"]
        if len(ids) + 1 > room:
            ids = ids[:max(room - 1, 0)]
            if not ids:
                break
        ids = ids + [sep_id]
        stream["tokens"] += writer.push(ids, stream["lang"])
        if verbose and writer.totals["tokens"] >= next_mark:
            print(f"[corpus] tokens={writer.totals['tokens']:,} read={read_bytes / 1e6:.1f}MB "
                  f"dup={deduper.stats['dup_exact'] + deduper.stats['dup_near']}")
            next_mark = writer.totals["tokens"] + PROGRESS_EVERY_TOKENS

    shards = writer.close()
    lang_tokens = {"en": sum(s["lang_tokens"]["en"] for s in shards),
                   "zh": sum(s["lang_tokens"]["zh"] for s in shards)}
    lang_tokens["other"] = sum(s["lang_tokens"]["other"] for s in shards)
    total_tokens = int(writer.totals["tokens"])
    index = {
        "format": FORMAT_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tokenizer_dir": str(_abs(spec["tokenizer_dir"])),
        "vocab_size": vocab,
        "doc_sep_id": sep_id,
        "shards": shards,
        "totals": {"tokens": total_tokens, "id_min": writer.totals["id_min"],
                   "id_max": writer.totals["id_max"], "docs_seen": docs_seen,
                   "bytes_read": read_bytes, "shards": len(shards)},
        "lang_tokens": lang_tokens,
        "lang_share": {k: round(v / max(1, total_tokens), 4) for k, v in lang_tokens.items()},
        "dedup": {**deduper.stats, "rules": spec["dedup"]},
        "spec": {k: v for k, v in spec.items() if k != "sources"} | {"sources": spec["sources"]},
        "source": "self-trained BPE tokenizer, corpus assembled from public educational text "
                  "(no third-party weights/vocab; GUIDE §7-0)",
        "elapsed_sec": round(time.time() - started, 2),
    }
    _final_check(index, spec)
    (out_dir / INDEX_FILE).write_text(json.dumps(index, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return index


def _final_check(index: dict[str, Any], spec: dict[str, Any]) -> None:
    """交付前的三道硬检查：产出够下限、编号不越界、中英两侧都不是零占比（spec 冒烟场景）。"""
    total = index["totals"]["tokens"]
    floor = int(spec["min_tokens_expected"])
    if total < floor:
        raise CorpusError(f"产出 {total} 个编号，低于配置下限 {floor}；加大 max_tokens 或换源重装配")
    hi = index["totals"]["id_max"]
    if hi is not None and hi >= index["vocab_size"]:
        raise CorpusError(f"最大编号 {hi} 没小于表规模 {index['vocab_size']}，装配流程有洞")
    for lang in ("en", "zh"):
        if index["lang_tokens"][lang] == 0:
            raise CorpusError(f"{lang} 侧编号占比为 0：这个来源压根没进流，配比 {index['lang_share']} 不成立")


# ══ A1-6 训练侧读取：只认 index.json，mmap 随机窗口，零拷贝 ═════════════════════════════

class CorpusReader:
    """一份装配好的编号流的句柄：按 index.json 打开各片，随机取一段窗口给训练用。

    片用内存映射（memmap）方式打开：读窗口时由操作系统按页调入，进程不搬整片进内存，
    所以 200MB 档的语料在 16GB 的机器上也是随取随用。
    """

    def __init__(self, index_path: str | Path) -> None:
        path = _abs(index_path)
        self.dir = path.parent if path.is_file() or path.name == INDEX_FILE else path
        file = path if path.name == INDEX_FILE else self.dir / INDEX_FILE
        if not file.exists():
            raise CorpusError(f"找不到索引 {file}；先用 python -m sys1.data.pretrain_corpus 装配")
        self.index = json.loads(file.read_text(encoding="utf-8"))
        if self.index.get("format") != FORMAT_NAME:
            raise CorpusError(f"索引格式 {self.index.get('format')!r} 不是本版本认的 {FORMAT_NAME}")
        if self.index.get("tokenizer_dir"):
            self.index["tokenizer_dir"] = _reanchor(self.index["tokenizer_dir"])  # 仓库改名/搬迁自愈
        self.vocab = int(self.index["vocab_size"])
        self.shards = [np.memmap(self.dir / s["file"], dtype=DTYPE, mode="r", shape=(s["tokens"],))
                       for s in self.index["shards"]]
        self.lengths = [int(s["tokens"]) for s in self.index["shards"]]
        self.total_tokens = int(self.index["totals"]["tokens"])

    def window(self, rng: np.random.Generator, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        """随机挑一片、随机挑起点，切出 (输入, 答案) 两段：答案就是输入整体左移一位。

        白话：从几本账本里随手抽一本，再在本子里挑一处开头，往下数 seq_len+1 行——前一行到
        倒数第二行当题目，从第二行到最后一行当答案。起点不会挑到本子最后 seq_len 行以内，
        免得切出半截窗口还要拿别的本子去接（跨片接续会让"这一段是连续的原文"这件事说不清）。
        """
        usable = [i for i, n in enumerate(self.lengths) if n > seq_len]
        if not usable:
            raise CorpusError(f"所有片都短于窗口长 {seq_len}；装配时把 shard_tokens 调大")
        pick = int(rng.choice(usable))
        shard = self.shards[pick]
        start = int(rng.integers(0, self.lengths[pick] - seq_len - 1))
        chunk = np.asarray(shard[start:start + seq_len + 1], dtype=np.int64)
        return chunk[:-1], chunk[1:]

    def stats(self) -> dict[str, Any]:
        """把索引里的关键统计原样交出（写 run 记录时要把配比与总量一起留下当证据）。

        白话：把这张总账上的几个要紧数字——一共记了多少号、分了几本、中英各占几成、最大的
        号是多少——照原样抄一份交出去。事后写报告时不用再去翻片本，一眼就能核对样本量。
        """
        return {"tokens": self.total_tokens, "shards": len(self.shards), "vocab_size": self.vocab,
                "lang_share": self.index["lang_share"], "id_max": self.index["totals"]["id_max"]}


def open_corpus(path: str | Path) -> CorpusReader:
    """按路径打开一份装配产物（目录或 index.json 都行），返回 CorpusReader。

    白话：给人家一个文件夹的名字或者索引文件的名字，把里面那几本账本按清单摊开在桌上备查；
    清单不在、格式不对都会当场报错，绝不拿着半份东西就开始练。
    """
    return CorpusReader(path)


# ══ A1-7 CLI：python -m sys1.data.pretrain_corpus --stream-limit 200MB --out ... --config tiny
#
# 目的：装配 S1 下一词预训练用的中英编号流（spec「冒烟语料产出」场景的执行入口）。
# 输入：--config 档位名或 yaml 路径（配比/去重/片长都在里面）；--stream-limit 文本读取上限；
#       --max-tokens 编号产出上限；--tokenizer 覆盖对照表目录；--prefer local|remote|auto。
# 产出：--out 目录下的 shard-*.bin + index.json（片长、中英占比、去重计数、编号上下界全在里面）。
# 预计耗时：内置小语料 <2 秒；200MB 真实语料档（8M 编号上限）CPU 约 1–3 分钟；
#           远端首拉受网络支配，本地已有语料时不联网（这也是 A2 冒烟档刻意复用本地 209MB 的原因）。

def build_parser() -> argparse.ArgumentParser:
    """搭出命令行参数表（--config/--stream-limit/--max-tokens/--out/--tokenizer/--prefer）。

    白话：把能敲的选项和默认值写清楚，让人敲一下 --help 就看见一份说明书——用哪档配比、
    最多读进来多少字、最多攒多少个号、产物丢进哪个文件夹、用哪张对照表。
    """
    parser = argparse.ArgumentParser(
        prog="pretrain_corpus.py",
        description="装配 S1 预训练语料：中英流式取数（本地优先/ModelScope 优先 HF 兜底）→ 去重 → 分片编号流。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="tiny", help=f"档位名（tiny/main）或 yaml 路径；默认查 {CONFIG_PATH}")
    parser.add_argument("--stream-limit", default=None, help="文本读取上限，例 200MB / 1.5G / 0=不限（默认用配置值）")
    parser.add_argument("--max-tokens", type=int, default=None, help="编号产出上限（默认用配置值）")
    parser.add_argument("--out", default=None, help="产物目录；默认 runs/corpus_<档位名>")
    parser.add_argument("--tokenizer", default=None, help="覆盖对照表目录（默认取配置里的 tokenizer_dir）")
    parser.add_argument("--prefer", choices=["auto", "local", "remote"], default="auto",
                        help="取数偏好：auto=本地有就用本地，local=只走本地，remote=强制走远端")
    parser.add_argument("--quiet", action="store_true", help="不打印装配进度")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 主体：读配置 → 装配 → 打印摘要；任何硬检查不过一律非零退出（不交半成品）。

    白话：一条命令走完"挑规矩、开流水、丢重复、换编号、分本子记账、写总清单"六步。中途
    哪一步没达标——号出格了、中英有一侧空了、产出没到下限——就当场停下报错，绝不把半成品
    当成果交出去，因为下游只会照着总清单取用，看不出里面缺了哪一角。
    """
    args = build_parser().parse_args(argv)
    spec = load_spec(args.config, stream_limit=args.stream_limit, max_tokens=args.max_tokens,
                     tokenizer_dir=args.tokenizer, prefer=args.prefer)
    out = args.out or f"runs/corpus_{Path(str(args.config)).stem}"
    index = build(spec, out, verbose=not args.quiet)
    totals = index["totals"]
    print(f"[corpus] out            : {_abs(out)}")
    print(f"[corpus] tokenizer      : {index['tokenizer_dir']} (vocab={index['vocab_size']}, "
          f"doc_sep_id={index['doc_sep_id']})")
    print(f"[corpus] shards         : {totals['shards']} 片 / {totals['tokens']:,} 个编号 "
          f"(id 区间 [{totals['id_min']}, {totals['id_max']}])")
    print(f"[corpus] lang share     : {index['lang_share']}  (en={index['lang_tokens']['en']:,} "
          f"zh={index['lang_tokens']['zh']:,})")
    print(f"[corpus] docs seen/dup  : {totals['docs_seen']} / {index['dedup']['dup_exact']} 全同 + "
          f"{index['dedup']['dup_near']} 近同")
    print(f"[corpus] bytes read     : {totals['bytes_read']:,} (limit={spec['stream_limit_bytes']})")
    print(f"[corpus] elapsed        : {index['elapsed_sec']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
