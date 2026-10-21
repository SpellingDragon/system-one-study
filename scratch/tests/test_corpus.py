"""p1-08 A1/A2 语料装配测试（`-k` 对应孙任务：A1 全量、A2 走 `tiny_output`）。

覆盖面：
  1) 配置解析与硬校验（权重和、缺字段、档位不存在、长度写法折算）；
  2) 装配主流程（内置小语料离线可跑）：分片落盘、index.json 字段、编号不越界、中英占比非零；
  3) 去重两层（精确全文哈希 + minhash-lite 近重复）与"关掉去重"的开关；
  4) 取数策略：本地优先不联网、强制远端、断点续拉（缓存命中不再下载）、ModelScope 优先 HF 兜底；
  5) 训练侧读取：memmap 随机窗口左移一位、片太短时报错；
  6) tiny_output：真实 200MB 冒烟档产物（A2 的证据，由 CLI 先装配好）。

全部用例默认离线；碰网络的分支一律 monkeypatch 成假下载器，CI 上不需要外网。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from sys1.data import pretrain_corpus as pc
from sys1.lang import bpe

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_OUTPUT = REPO_ROOT / "runs" / "corpus_tiny"   # A2 的 200MB 冒烟档产物目录


def _spec(profile: str = "unit", **overrides) -> dict:
    spec = pc.load_spec(profile)
    spec.update(overrides)
    return spec


def _build(spec: dict, out: Path) -> dict:
    return pc.build(spec, out, verbose=False)


# ── 1) 配置：显式入配置 + 硬校验 ────────────────────────────────────────────────────

def test_load_spec_reads_rules_from_yaml():
    """配比、去重、片长、上限全部来自 learning/configs/corpus.yaml，代码里不留第二处。"""
    spec = pc.load_spec("unit")
    assert [(s["lang"], s["weight"]) for s in spec["sources"]] == [("en", 0.7), ("zh", 0.3)]
    assert spec["dedup"]["exact"] is True and spec["dedup"]["minhash_lite"]["num_perm"] == 4
    assert spec["shard_tokens"] == 64 and spec["vocab_size"] == 16000
    assert Path(spec["tokenizer_dir"]).is_absolute(), "相对路径必须按仓库根展开，否则换 cwd 就找不到表"


def test_load_spec_unknown_profile_lists_alternatives(tmp_path):
    """档位名查不到时报错并列出可用档位，绝不"差不多用一个"。"""
    with pytest.raises(pc.CorpusError, match="可用档位"):
        pc.load_spec("nope")


def test_load_spec_rejects_bad_weight_sum():
    """权重之和不是 1、或关键字段缺失 → 直接拒，不带病开工。"""
    spec = _spec()
    spec["sources"][0]["weight"] = 0.9
    with pytest.raises(pc.CorpusError, match="权重之和"):
        pc._check_sources(spec)


def test_load_spec_missing_required_key_raises(tmp_path):
    """yaml 里少写 vocab_size → 报"配置缺字段"，不是后面某处 KeyError 崩在半路。"""
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"tokenizer_dir": "runs", "max_tokens": 10,
                                   "shard_tokens": 4, "sources": [{"lang": "en", "weight": 1.0}]}),
                   encoding="utf-8")
    with pytest.raises(pc.CorpusError, match="配置缺字段"):
        pc.load_spec(bad)


def test_parse_size_human_and_bytes():
    """200MB / 1.5G / 4096 / 0 / none 五种写法折算成字节数或"不限"。"""
    assert pc.parse_size("200MB") == 200 * 1024**2
    assert pc.parse_size("1.5G") == int(1.5 * 1024**3)
    assert pc.parse_size(4096) == 4096
    assert pc.parse_size("0") is None and pc.parse_size(None) is None
    with pytest.raises(pc.CorpusError):
        pc.parse_size("两百度")


# ── 2) 装配主流程（内置小语料，离线）────────────────────────────────────────────────

def test_build_writes_shards_and_index(tmp_path):
    """spec「冒烟语料产出」的离线版：分片 + index.json，编号全在表规模内、中英占比都非零。"""
    spec = _spec()
    index = _build(spec, tmp_path / "c")

    shards = sorted(p.name for p in (tmp_path / "c").glob(f"{pc.SHARD_PREFIX}*{pc.SHARD_SUFFIX}"))
    assert shards and all(re.fullmatch(pc.SHARD_PATTERN.pattern, name) for name in shards)
    assert (tmp_path / "c" / pc.INDEX_FILE).exists()
    assert index["format"] == pc.FORMAT_NAME
    assert index["totals"]["tokens"] == sum(s["tokens"] for s in index["shards"])
    assert index["totals"]["id_max"] < index["vocab_size"], "有编号越界就是装配流程有洞"
    assert index["lang_tokens"]["en"] > 0 and index["lang_tokens"]["zh"] > 0, "中英两侧都得进流"
    for name in shards:
        data = np.fromfile(tmp_path / "c" / name, dtype=pc.DTYPE)
        assert len(data) <= spec["shard_tokens"], "单片长度不得超配置"
        assert data.max() < index["vocab_size"]


def test_build_shard_lengths_match_declared(tmp_path):
    """每片声明的编号数必须和文件实际大小一致——训练侧按声明开 memmap，靠的就是这条。"""
    index = _build(_spec(), tmp_path / "c")
    for shard in index["shards"]:
        size = (tmp_path / "c" / shard["file"]).stat().st_size
        assert size == shard["tokens"] * np.dtype(pc.DTYPE).itemsize


def test_build_doc_sep_between_docs(tmp_path):
    """文档之间插整篇结束标记：串起来看，两个不同文档之间必有一个分隔号。"""
    spec = _spec()
    sep_id = bpe.load(spec["tokenizer_dir"]).token_to_id(spec["doc_sep"])
    index = _build(spec, tmp_path / "c")
    assert index["doc_sep_id"] == sep_id == 1, "分隔号取自自训表的整篇结束位（固定 1 号）"
    stream = np.memmap(tmp_path / "c" / index["shards"][0]["file"], dtype=pc.DTYPE, mode="r",
                       shape=(index["shards"][0]["tokens"],))
    assert int(stream[0]) != index["doc_sep_id"], "片首不该是分隔号（分隔号只出现在文档之间）"
    assert index["totals"]["id_min"] == index["doc_sep_id"], "整条流里最小号就是分隔号（pad 不进流）"


def test_build_fail_when_below_floor(tmp_path):
    """产出低于 min_tokens_expected → 报错中止，不交付半成品语料。"""
    spec = _spec(min_tokens_expected=10_000_000)
    with pytest.raises(pc.CorpusError, match="低于配置下限"):
        _build(spec, tmp_path / "c")


def test_build_rejects_oversized_token_id(tmp_path):
    """编号越界（>表规模）时装配当场中止，且不写 index.json。"""
    fake = SimpleNamespace(get_vocab_size=lambda: 16000, token_to_id=lambda _n: 1,
                           encode=lambda _t, add_special_tokens=False: SimpleNamespace(ids=[5, 999_999]))

    out = tmp_path / "c"
    with pytest.raises(pc.CorpusError, match="编号越界"):
        pc.build(_spec(), out, tokenizer=fake, verbose=False)
    assert not (out / pc.INDEX_FILE).exists(), "失败就不该留下可用的索引"


def test_build_checks_vocab_size_agreement(tmp_path):
    """配置写的表规模与实际读到的表行数不一致 → 拒装（下游窗口会整片错位）。"""
    small = SimpleNamespace(get_vocab_size=lambda: 123, token_to_id=lambda _n: 1,
                            encode=lambda _t, add_special_tokens=False: SimpleNamespace(ids=[1, 2]))

    with pytest.raises(pc.CorpusError, match="表规模不符"):
        pc.build(_spec(), tmp_path / "c", tokenizer=small, verbose=False)


# ── 3) 去重：精确 + minhash-lite，两层都可关 ────────────────────────────────────────

def test_dedup_drops_exact_and_near_duplicates():
    """同一篇抄两遍拦掉；只改了两个字的近重复也拦掉；全新的一篇放行。"""
    dedup = pc.Deduper({"enabled": True, "exact": True,
                        "minhash_lite": {"num_perm": 6, "shingle_chars": 6, "stride": 2}})
    doc = "先测正确性，再谈性能；correctness comes before performance on every device you own."
    assert dedup.duplicate(doc) is False
    assert dedup.duplicate(doc) is True, "一模一样的第二篇必须判重"
    near = doc.replace("performance", "perfomance")          # 只差一个字母
    assert dedup.duplicate(near) is False, "第一份近重复该收下（否则误杀）"
    assert dedup.duplicate(near) is True, "同一份近重复的第二遍必须拦"
    assert dedup.stats["dup_exact"] >= 1 and dedup.stats["seen"] == 4


def test_dedup_can_be_disabled_by_config():
    """配置关掉去重 → 重复稿照样进流（配比/规模实验要能单独看某一层的影响）。"""
    dedup = pc.Deduper({"enabled": False})
    doc = "同一段文字重复三遍也不许拦"
    assert [dedup.duplicate(doc) for _ in range(3)] == [False, False, False]


def test_build_counts_dedup_in_index(tmp_path):
    """装配账本里要能查到"看过几篇、丢了几篇全同、丢了几篇近同"，去重不是静默动作。"""
    docs = ["第一段 内容 A", "第一段 内容 A", "第二段 内容 B 与上面完全不同的一些汉字"]
    spec = _spec(sources=[{"lang": "en", "weight": 0.7, "docs": docs},
                          {"lang": "zh", "weight": 0.3, "docs": docs}],
                 min_tokens_expected=1)
    index = _build(spec, tmp_path / "c")
    # 流内：第二条与第一条同稿 → 1 次；第二条流把三条再送一遍 → 再 3 次，合计 4 次全同拦截
    assert index["dedup"]["seen"] == 6
    assert index["dedup"]["dup_exact"] == 4, "重复稿无论出现在同一条流还是另一条流，都只收第一次"
    assert index["dedup"]["rules"]["exact"] is True, "用的哪套规则也一并落账"


# ── 4) 取数策略：本地优先 / 远端双源 / 断点续拉（全部假下载器，不联网）───────────────

def test_local_source_preferred_and_no_network(tmp_path, monkeypatch):
    """本地语料存在时一步都不走网络（A2 复用 bench/ 下 209MB 的真实原因）。"""
    def boom(*_a, **_k):
        raise AssertionError("本地有语料时不该碰网络")

    monkeypatch.setattr(pc.urllib.request, "urlopen", boom)
    txt = tmp_path / "en.txt"
    txt.write_text("本地的一份英文文本，一行当一篇文档处理。\n", encoding="utf-8")
    spec = _spec(sources=[{"lang": "en", "weight": 0.7, "local": str(txt)},
                          {"lang": "zh", "weight": 0.3, "local": str(txt)}])
    stream = pc._open_source(spec["sources"][0], spec)
    assert next(stream).startswith("本地的一份英文文本")


def test_remote_stream_uses_modelscope_first_with_hf_fallback(tmp_path, monkeypatch):
    """远端下载按 ModelScope 优先、HF 兜底；第一个源失败要能自动换第二个。"""
    seen: list[str] = []

    def fake_to_file(url: str, dest: Path) -> None:
        seen.append(url)
        if "modelscope" in url:
            raise OSError("模拟优先源不可达")
        dest.write_text("兜底源下来的文本行。\n", encoding="utf-8")

    monkeypatch.setattr(pc, "_http_to_file", fake_to_file)
    source = {"lang": "en", "weight": 1.0, "modelscope": "AI-ModelScope/fineweb-edu",
              "huggingface": "HuggingFaceFW/fineweb-edu", "files": ["a.txt"], "revision": "master"}
    spec = _spec(sources=[source, {"lang": "zh", "weight": 0.0, "docs": ["中文兜底的一篇稿子"]}],
                 cache_dir=str(tmp_path / "cache"))
    out = tmp_path / "cache" / "en" / "a.txt"
    path = pc.download_shard(source, "a.txt", Path(spec["cache_dir"]))
    assert path == out and out.exists(), "优先源失败后必须由兜底源补齐"
    assert "modelscope" in seen[0] and "huggingface" in seen[1], "顺序必须是 ModelScope 优先 HF 兜底"


def test_download_shard_reuses_cache_no_second_download(tmp_path, monkeypatch):
    """断点续拉：缓存里已有完整分片就不再下载（重跑不重复搬同一箱）。"""
    calls: list[str] = []

    def fake_to_file(url: str, dest: Path) -> None:
        calls.append(url)
        dest.write_bytes(b"fake-shard")

    monkeypatch.setattr(pc, "_http_to_file", fake_to_file)
    source = {"lang": "en", "weight": 1.0, "modelscope": "ns/ds", "files": ["a.parquet"]}
    cache = tmp_path / "cache"
    first = pc.download_shard(source, "a.parquet", cache)
    assert first.exists() and len(calls) == 1
    second = pc.download_shard(source, "a.parquet", cache)
    assert second == first and len(calls) == 1, "第二次必须直接复用缓存，不再下载"


def test_partial_download_is_not_mistaken_for_complete(tmp_path, monkeypatch):
    """下载中途失败不留半截文件：临时名不会被下一次的"已存在"当成完整分片。"""
    def half(url, dest):
        dest.write_text("半截", encoding="utf-8")
        dest.unlink()                      # 模拟失败路径把临时文件清掉
        raise OSError("网络断了")

    monkeypatch.setattr(pc, "_http_to_file", half)
    monkeypatch.setattr(pc, "_raw_urls", lambda source, name: ["https://modelscope.cn/x"])
    with pytest.raises(pc.CorpusError, match="两个源都没下到"):
        pc.download_shard({"lang": "en", "modelscope": "ns/ds"}, "a.txt", tmp_path / "cache")
    assert not (tmp_path / "cache" / "en" / "a.txt").exists()


def test_list_remote_files_prefers_explicit_file_list(monkeypatch):
    """配置里写了 files 清单就跳过列举（网络列举不稳时的后路，design 的回退项）。"""
    monkeypatch.setattr(pc, "_http_get", lambda url: (_ for _ in ()).throw(AssertionError("不该发请求")))
    names = pc.list_remote_files({"lang": "en", "modelscope": "ns/ds", "files": ["b.parquet", "a.parquet"]})
    assert names == ["b.parquet", "a.parquet"]


def test_iter_parquet_docs_reads_text_column(tmp_path):
    """parquet 分片按文本字段逐条吐出，空串与清洗后为空的行不占位。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "part.parquet"
    table = pa.table({"text": ["第一箱的内容 http://a.example.com 带网址", "", "第二箱的内容"]})
    pq.write_table(table, str(path))
    docs = list(pc.iter_parquet_docs(path, "text"))
    assert len(docs) == 2 and "第一箱的内容" in docs[0], docs


# ── 5) 训练侧读取：memmap 窗口 ───────────────────────────────────────────────────────

def test_reader_window_is_shifted_by_one(tmp_path):
    """答案段就是输入段整体左移一位；起点不越到片尾，窗口永远是连续的原文。"""
    index = _build(_spec(), tmp_path / "c")
    reader = pc.open_corpus(tmp_path / "c")
    assert reader.total_tokens == index["totals"]["tokens"]
    x, y = reader.window(np.random.default_rng(7), 16)
    assert x.shape == y.shape == (16,)
    assert x.max() < reader.vocab and y.max() < reader.vocab


def test_reader_window_exact_next_token(tmp_path):
    """严格对拍：流里连续取一段，window 的 y 必须恰等于 x 的下一位。"""
    _build(_spec(), tmp_path / "c")
    reader = pc.open_corpus(tmp_path / "c")
    flat = np.concatenate([np.asarray(s) for s in reader.shards])
    rng = np.random.default_rng(0)
    for _ in range(20):
        x, y = reader.window(rng, 8)
        start = next(i for i in range(len(flat) - 8) if np.array_equal(flat[i:i + 8], x))
        assert np.array_equal(y, flat[start + 1:start + 9]), "下一位对不上就是窗口切错了"


def test_reader_rejects_shards_shorter_than_window(tmp_path):
    """片长不够窗口长就明确报错（而不是返回半截窗口或悄悄跨片拼接）。"""
    _build(_spec(), tmp_path / "c")
    reader = pc.open_corpus(tmp_path / "c")
    reader.lengths = [4, 4]
    with pytest.raises(pc.CorpusError, match="短于窗口长"):
        reader.window(np.random.default_rng(0), 8)


def test_reader_missing_index(tmp_path):
    """没有 index.json 的目录打不开：宁可报错，也不拿半份东西开始练。"""
    (tmp_path / "empty").mkdir()
    with pytest.raises(pc.CorpusError, match="找不到索引"):
        pc.open_corpus(tmp_path / "empty")


# ── 6) A2：真实 200MB 冒烟档产物（先跑 CLI 再跑本用例）───────────────────────────────

def test_tiny_output_meets_spec_requirements():
    """spec「冒烟语料产出」：`--stream-limit 200MB` 档的产物达标——够下限、号不越界、中英都非零。

    本用例读 runs/corpus_tiny（由 `python -m sys1.data.pretrain_corpus --stream-limit 200MB
    --out runs/corpus_tiny --config tiny` 装配）。目录不存在时跳过而非假装通过：跳过会显示在
    pytest 摘要里，不会被当成绿。
    """
    index_file = TINY_OUTPUT / pc.INDEX_FILE
    if not index_file.exists():
        pytest.skip(f"先执行 python -m sys1.data.pretrain_corpus --config tiny --out {TINY_OUTPUT}")
    index = json.loads(index_file.read_text(encoding="utf-8"))
    spec = pc.load_spec("tiny")

    assert index["totals"]["tokens"] >= spec["min_tokens_expected"], "产出必须达到配置下限"
    assert index["totals"]["id_max"] < index["vocab_size"], "编号全部小于表规模"
    assert index["lang_share"]["en"] > 0 and index["lang_share"]["zh"] > 0, "中英占比都非零"
    assert abs(index["lang_share"]["en"] - 0.7) < 0.05, f"7:3 配比漂了：{index['lang_share']}"
    assert index["dedup"]["dup_exact"] + index["dedup"]["dup_near"] >= 0
    assert len(index["shards"]) == index["totals"]["shards"] >= 2, "分片至少两片才算"
    # 分片实际字节数与声明一致，且真能按声明长度开 memmap
    reader = pc.open_corpus(index_file)
    assert reader.total_tokens == index["totals"]["tokens"]
    x, y = reader.window(np.random.default_rng(3), 32)
    assert len(x) == len(y) == 32 and x.max() < index["vocab_size"]


def test_reanchor_survives_repo_rename(tmp_path, monkeypatch):
    """迁移自愈回归：index.json 里的旧仓库绝对路径（改名后不存在）按 '/runs/' 后缀重锚当前根。"""
    from sys1.data import pretrain_corpus as pc
    real_tok = pc._abs("runs/1003-s0-bpe-16k-realedu-zh-en/tokenizer")
    if not real_tok.exists():
        pytest.skip("无真实 tokenizer 产物")
    stale = f"$WORKSPACE_ROOT/old-repo-name/scratch/{real_tok.relative_to(pc.REPO_ROOT)}"
    assert pc._reanchor(stale) == str(real_tok), "旧路径应重锚到当前仓库根下的同相对路径"
    assert pc._reanchor(str(real_tok)) == str(real_tok)          # 存在则原样
    assert pc._reanchor("/no/such/wherever") == "/no/such/wherever"  # 无可重锚则不动（报错留给上层）
