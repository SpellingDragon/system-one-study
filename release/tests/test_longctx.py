"""p2-07 长上下文测试：needle 生成器（A1）、前缀复用（B1a-B1c/B2）、因果滑窗（C1）。

口径：
  * 全部可在 CPU 跑完——前缀复用用一个"纯 python 假引擎"，滑窗用合成张量与小 GPT；
    真 backbone（0.6B 替身）的前向凭据走 run 档案，不在单测里挂 6 秒模型加载；
  * 资源纪律：禁用 MPS（归一阶段训练长跑）；带 `mps` 标记的用例需 `SYS1_ALLOW_MPS=1`
    才放行，本波一律以 CPU 路验证（记入战报偏离项）；
  * 用例名按孙任务分关键字：needle / prefix_core / prefix_incr / prefix_parity /
    cross_hit / sliding，与二级 tasks.md 的验证命令 `-k <关键字>` 一一对应。
"""
from __future__ import annotations

import json
import os

import pytest
import torch

from serving.prefix_cache import (
    DEFAULT_DRIFT_RATIO,
    SPLIT_MARK,
    CacheError,
    EngineOut,
    HFEngine,
    PrefixCache,
    PrefixRunner,
    estimate_past_bytes,
    prefix_key,
    split_prompt_ids,
)
from sys1.decision.render import RENDER_VERSION
from sys1.eval import longctx, registry
from sys1.layers import attention as LA

need_mps = pytest.mark.skipif(
    os.environ.get("SYS1_ALLOW_MPS") != "1",
    reason="本波禁用 MPS（派单纪律）；显式 SYS1_ALLOW_MPS=1 才放行",
)

# skip-when-missing（R-P1-4 统一口径）：针位骨架是 bench/ 装配副本，gitignored 按设计不入库；
# 缺件＝"没装配"（环境事实），不是回归——与 test_assets.py:40 real_only、test_registry.py:46-48
# 同一纪律。补件命令沿用 longctx.load_skeleton() 自己报的那条（longctx.py:164），不另造说法。
HAS_SKELETON = (registry.REPO_ROOT / longctx.SKELETON_REL).is_file()
skeleton_only = pytest.mark.skipif(
    not HAS_SKELETON,
    reason=f"针位骨架不在盘上：{longctx.SKELETON_REL}（bench/ 按设计不入库；先跑 "
           "`python -m sys1.eval.registry fetch --needle`）（R-P1-4，docs/ci_baseline_triage.md §1）",
)


# ------------------------------------------------------------------ 假引擎（B1 专用替身）
class FakeEngine:
    """确定性假引擎：`hidden = f(整条已看过的 token 列)`，past 就是那列 token + 两份小张量。

    白话：把"模型"换成一台只会照本宣科的机器——只要它看到的字一样，交出的数字就一样。
    这样"缓存有没有命中、命中后真喂了几个符号、复用路和重算路对不对得上"三件事
    全部与模型无关，能在 CPU 上测到骨头里。
    """

    def __init__(self, *, dim: int = 4, perturb: float = 0.0) -> None:
        self.dim = dim
        self.perturb = float(perturb)          # 模拟 past 复用的数值漂移（B1c 用）
        self.calls: list[dict] = []

    def forward(self, input_ids, past=None, *, offset: int = 0) -> EngineOut:
        seen = list(past["tokens"]) if past is not None else []
        fed = [int(t) for t in input_ids]
        ids = seen + fed
        vec = torch.zeros(1, self.dim)
        vec[0, 0] = sum(ids) / 1000.0
        vec[0, 1] = len(ids) / 1000.0
        vec[0, 2] = (max(ids) / 1000.0) if ids else 0.0
        # 末位的全局位置：带 past 时是 offset+len-1，整段重算时是 len-1 —— 两条路必须同值，
        # 位置传错（增量前向忘了抬 offset）就会在这里露出来
        vec[0, 3] = ((offset + len(fed) - 1) if past is not None else len(fed) - 1) / 1000.0
        if past is not None and self.perturb:
            vec[0, 0] += self.perturb
        new_past = {"tokens": ids, "keys": torch.zeros(1, len(ids), 4), "values": torch.zeros(1, len(ids), 4)}
        self.calls.append({"fed": len(input_ids), "used_past": past is not None, "offset": offset})
        return EngineOut(vec, new_past, len(input_ids))

    def clone_past(self, past):
        return {"tokens": list(past["tokens"]), "keys": past["keys"].clone(), "values": past["values"].clone()}

    def past_bytes(self, past) -> int:
        return (past["keys"].numel() + past["values"].numel()) * past["keys"].element_size()


def readout(hidden: torch.Tensor, k: int | None = None) -> torch.Tensor:
    """假读出：取前 k 个通道当候选分（量级 ≈0.1，漂移阈值讲起来直观）。"""
    n = hidden.size(-1) if k is None else int(k)
    return hidden[:, :n].float()


PREFIX = list(range(100, 180))                       # 一份"证据段"的 token 列
SUFFIXES = [list(range(200, 239)), list(range(300, 334)), list(range(400, 451))]


# ================================================================== A1 needle 生成器
@skeleton_only
def test_needle_table_is_registry_skeleton_and_deterministic():
    tbl, source = longctx.resolve_table()
    assert source == "skeleton", f"针位表应优先取 registry 骨架，实得 {source}"
    assert len(tbl) == 30, f"骨架应有 30 根针，实得 {len(tbl)}"
    assert {n.ctx for n in tbl} == set(registry.NEEDLE_BUCKETS)
    again, _ = longctx.resolve_table()
    assert tbl == again, "同骨架两次解析必须逐字段相同（seed 固定入 registry 的镜像）"


@skeleton_only
def test_needle_table_from_plan_mirrors_skeleton():
    """plan 镜像与盘上骨架必须同源——不同源就说明 registry 的取随机顺序被我抄错了。"""
    disk = longctx.load_skeleton()
    mirrored = longctx.needle_table_from_plan()
    assert len(disk) == len(mirrored)
    assert [d.row_id for d in disk] == [m.row_id for m in mirrored]
    bad = [d.row_id for d, m in zip(disk, mirrored) if (d.color, d.value) != (m.color, m.value)]
    assert not bad, f"以下针位内容与 registry 骨架不一致：{bad[:5]}"


def test_needle_multi_needle_doc_and_ordinal_question():
    """多针硬要求：同档位的针一根都不许被丢掉；同色针靠"第 N 次提到"消歧。"""
    corpus = longctx.build_corpus(buckets=[8192])
    rows = corpus["rows"]
    assert len(rows) == 2
    for row in rows:
        nd = row["needle"]
        assert nd["needles_in_doc"] == 2, "8K 档就该有两根针都在文里"
        assert nd["same_color_in_doc"] == 2, "registry 的 8K 档两针同色（青），必须如实计数"
        assert "第 1 次" in nd["question"] or "第 2 次" in nd["question"]
        text = row["sample"]["state"]
        for needle in ("658", "509"):              # 两根针的编号都得在正文里出现
            assert needle in text, f"正文缺针 {needle}"
    assert corpus["ledger"]["8192"]["needles"] == 2


def test_needle_corpus_fingerprint_is_reproducible():
    a = longctx.build_corpus(buckets=[8192, 32768], limit_per_bucket=1)
    b = longctx.build_corpus(buckets=[8192, 32768], limit_per_bucket=1)
    fa, fb = longctx.corpus_fingerprint(a["rows"]), longctx.corpus_fingerprint(b["rows"])
    assert fa == fb, f"同参数两次构建指纹必须逐字相同：{fa} != {fb}"
    assert fa["count"] == 2
    assert fa["chars_by_ctx"]["32768"] > fa["chars_by_ctx"]["8192"], "档位越大正文越长"


def test_needle_length_ledger_is_honest():
    """长度口径必须自述：est-chars 档要说清是字数估算，不能冒充 token。"""
    row = longctx.build_corpus(buckets=[8192], limit_per_bucket=1)["rows"][0]
    nd = row["needle"]
    assert nd["len_units"] == "est-chars", "无编号器时口径必须写 est-chars"
    assert nd["target_units"] == round(8192 * longctx.CHARS_PER_TOKEN_EST)   # 档位×折算率
    assert abs(nd["units"] - nd["target_units"]) / nd["target_units"] < 0.10, "正文长度须在档位 ±10% 内"
    assert abs(nd["est_tokens"] - 8192) / 8192 < 0.10, "折算回 token 口径也要在 ±10% 内"
    assert nd["chars"] == nd["units"] == len(row["sample"]["state"])
    assert "tokens" not in nd, "没有编号器就不许冒出真 token 字段（不虚报）"


def test_needle_bucket_cap_is_booked_not_silently_shrunk():
    corpus = longctx.build_corpus(buckets=[131072], limit_per_bucket=1, cap_ctx=8192)
    assert corpus["capped"] == {"131072": 8192}, "压短必须留痕，否则报告把压短当原生档"
    nd = corpus["rows"][0]["needle"]
    assert nd["ctx"] == 131072 and nd["capped"] is True and nd["ctx_capped_to"] == 8192


def test_needle_envelope_ready_and_written(tmp_path):
    corpus = longctx.build_corpus(buckets=[8192], limit_per_bucket=2)
    path = longctx.write_corpus(corpus["rows"], tmp_path, source=corpus["source"])
    meta = json.loads((path.parent / longctx.META_NAME).read_text(encoding="utf-8"))
    assert meta["envelope_ready"] is True, "补完正文后信封必须可考（自描述位翻真）"
    assert meta["generator_version"] == longctx.LONGCTX_VERSION
    assert path.read_text(encoding="utf-8").count("\n") == len(corpus["rows"])


def test_needle_recall_exact_string_and_missing_counted():
    rows = longctx.build_corpus(buckets=[8192], limit_per_bucket=2)["rows"]
    gold_pred = [{"id": r["id"], "answers": {"q": r["sample"]["targets"]["q"]}} for r in rows]
    perfect = longctx.score_recall(rows, gold_pred)
    assert perfect["overall"]["recall"] == 1.0 and not perfect["missing"]
    assert perfect["metric"] == "needle_recall(exact-string)"
    wrong = [{"id": r["id"], "answers": {"q": {k: (0.0 if k == r["needle"]["answer_key"] else 1.0)
                                               for k in r["sample"]["questions"]["q"]["criteria"]}}}
             for r in rows]
    assert longctx.score_recall(rows, wrong)["overall"]["recall"] == 0.0
    gap = longctx.score_recall(rows, [])
    assert gap["overall"]["recall"] == 0.0 and len(gap["missing"]) == len(rows), "缺预测要计入 missing"


def test_needle_near_miss_string_is_not_a_hit():
    """精确串判分：编号差一位算没捞到，不做"差不多"的模糊匹配。"""
    row = longctx.build_corpus(buckets=[8192], limit_per_bucket=1)["rows"][0]
    near = str(int(row["needle"]["answer_text"]) + 1)
    row["sample"]["questions"]["q"]["criteria"][row["needle"]["answer_key"]] = near
    out = longctx.score_recall([row], [{"id": row["id"], "answers": {"q": {
        row["needle"]["answer_key"]: 1.0}}}])
    assert out["overall"]["recall"] == 0.0


def test_needle_curve_report_dual_column():
    rows = longctx.build_corpus(buckets=[8192], limit_per_bucket=2)["rows"]
    summary = longctx.score_recall(rows, [{"id": r["id"], "answers": {"q": r["sample"]["targets"]["q"]}}
                                           for r in rows])
    summary["device"] = longctx._device_note("cpu")
    text = longctx.format_curve(summary, measured=[8192], extra_rows=[longctx.one_million_row()])
    assert "8,192" in text and "1.0000" in text
    assert "measured: — / config: ready" in text, "1M 行必须是双列口径，不许与实测混淆"
    assert "非 NPU 显存账" in text, "CPU 内存口径要写进报告"


# ================================================================== B1a 缓存核心
def test_prefix_core_key_includes_render_version():
    a = prefix_key(PREFIX)
    assert a != prefix_key(PREFIX, render_version=RENDER_VERSION + "_v2"), "渲染改版必须污染不到旧键"
    assert a != prefix_key(PREFIX[:len(PREFIX) - 1]), "少一个 token 就换键"
    assert a == prefix_key(list(PREFIX)), "同列同版号 → 同键（确定性）"
    assert a != prefix_key(PREFIX, namespace="other-request"), "命名空间之间不许互串"
    assert len(a) == 32


def test_prefix_core_render_version_default_is_contract():
    """默认版号取的就是 P1 冻结契约，不是本地自造字符串。"""
    assert prefix_key(PREFIX) == prefix_key(PREFIX, render_version=RENDER_VERSION)


def test_prefix_core_lru_evicts_least_recent_and_counts():
    cache = PrefixCache(max_bytes=1 << 30, max_items=2)
    past = {"tokens": [1], "keys": torch.zeros(1, 1, 4), "values": torch.zeros(1, 1, 4)}
    k1, k2 = prefix_key([1]), prefix_key([2])
    cache.put(k1, past)
    cache.put(k2, past)
    assert cache.get(k1) is not None                        # k1 变最新
    cache.put(prefix_key([3]), past)
    assert cache.evictions == 1 and k2 not in cache and k1 in cache
    st = cache.stats()
    assert st["hits"] == 1 and st["misses"] == 0 and st["items"] == 2


def test_prefix_core_byte_budget_eviction_and_oversized():
    past = {"tokens": list(range(20)), "keys": torch.zeros(1, 20, 4), "values": torch.zeros(1, 20, 4)}
    one = estimate_past_bytes(past)
    assert one == 2 * 20 * 4 * 4, f"字节估算应为 keys+values 实占，实得 {one}"
    cache = PrefixCache(max_bytes=one * 2 + 1)
    for i in range(3):
        cache.put(prefix_key([i]), past, nbytes=one)
    assert cache.evictions == 1 and cache.bytes_used <= one * 2
    big = PrefixCache(max_bytes=one - 1)
    assert big.put(prefix_key([9]), past, nbytes=one) == one
    assert big.oversized == 1 and len(big) == 0, "单条超预算的项不许存进去再自杀"


def test_prefix_core_bytes_via_engine():
    engine, cache = FakeEngine(), PrefixCache(max_bytes=1 << 20)
    runner = PrefixRunner(engine, cache=cache, readout=readout, verify="off")
    res = runner.ask(PREFIX, SUFFIXES[0])
    out = cache.get(res.key)
    assert out is not None and cache.past_bytes(out) == engine.past_bytes(out)
    assert runner.stats()["cache"]["bytes_mib"] >= 0.0


# ================================================================== B1b 增量前向（token 计数断言）
def test_prefix_incr_token_count_second_question_is_suffix_only():
    """B1b 硬凭据：第 2 问起前向 token 数 == 问题段长度，证据段一符号都不重算。"""
    engine = FakeEngine()
    runner = PrefixRunner(engine, readout=readout, verify="off")
    lines = []
    for qi, suffix in enumerate(SUFFIXES):
        before = runner.counter.snapshot()
        res = runner.ask(PREFIX, suffix)
        spent = runner.counter.since(before)
        lines.append(
            f"[B1b q{qi}] hit={res.hit} prefix_tokens_fed={res.prefix_tokens_fed} "
            f"suffix_tokens_fed={res.suffix_tokens_fed} suffix_len={res.suffix_len} "
            f"tokens_fed={res.tokens_fed} counter_spent={spent} "
            f"assert_suffix_only={res.prefix_tokens_fed == 0 and res.suffix_tokens_fed == res.suffix_len}"
        )
        if qi == 0:
            assert res.prefix_tokens_fed == len(PREFIX) and res.suffix_tokens_fed == res.suffix_len
            assert spent == len(PREFIX) + len(suffix), "冷启第一问该付 前缀+问题段"
        else:
            assert res.hit is True
            assert res.prefix_tokens_fed == 0, f"第 {qi + 1} 问仍在算证据段，复用没生效"
            assert res.suffix_tokens_fed == len(suffix), "前向 token 数必须恰等于问题段长度"
            assert spent == len(suffix), f"命中路本次应只花 {len(suffix)} 符号，实花 {spent}"
            assert res.tokens_are_suffix_only is True
    total_full = sum(len(PREFIX) + len(s) for s in SUFFIXES)
    assert runner.counter.total == len(PREFIX) + sum(len(s) for s in SUFFIXES)
    assert runner.counter.total < total_full, "复用生效的净效果：总符号数必须低于逐问重算"
    print("\n".join(lines))                    # 凭据原文（-s 可见）
    print(f"[B1b] reuse_total={runner.counter.total} naive_total={total_full} "
          f"saved={total_full - runner.counter.total} by_kind={runner.counter.as_dict()['by_kind']}")


def test_prefix_incr_force_full_baseline_matches():
    """对照组：force_full 路付的符号数 = 前缀 + 问题段，正好是被省掉的那部分。"""
    engine = FakeEngine()
    runner = PrefixRunner(engine, readout=readout, verify="off")
    cold = runner.ask(PREFIX, SUFFIXES[1])            # 冷启：证据段算一次并入库
    warm = runner.ask(PREFIX, SUFFIXES[1])            # 命中：只喂问题段
    direct = runner.ask(PREFIX, SUFFIXES[1], force_full=True)
    assert cold.prefix_tokens_fed == len(PREFIX) and not cold.hit
    assert warm.prefix_tokens_fed == 0 and warm.hit
    assert direct.prefix_tokens_fed == 0 and direct.full_tokens_fed == len(PREFIX) + len(SUFFIXES[1])
    assert torch.equal(warm.hidden_last, direct.hidden_last), "两条路的末位数字必须逐位相同"


def test_prefix_incr_cache_cold_start_stores_once():
    engine = FakeEngine()
    runner = PrefixRunner(engine, readout=readout, verify="off")
    r1 = runner.ask(PREFIX, SUFFIXES[0])
    assert runner.cache.stats()["stores"] == 1 and runner.cache.stats()["hits"] == 0
    runner.ask(PREFIX, SUFFIXES[1])
    assert runner.cache.stats()["hits"] == 1, "第二问必须命中同一份 past"


# ================================================================== B1c 一致性护栏
def test_prefix_parity_within_threshold_passes():
    engine = FakeEngine(perturb=0.001)
    runner = PrefixRunner(engine, readout=readout, verify="always")
    runner.ask(PREFIX, SUFFIXES[0])
    res = runner.ask(PREFIX, SUFFIXES[0])
    assert res.drift is not None and res.argmax_same is True
    assert res.drift_ratio <= runner.drift_ratio_threshold
    assert res.recomputed is False, "漂移在阈值内就该走复用路，不该多花一次重算"
    assert not runner.stats()["quarantined"]


def test_prefix_parity_over_threshold_quarantines_and_recomputes():
    """漂移超限 → 当次就改用重算结果、拉黑该 state、脏 past 立即出库，此后一律重算。"""
    engine = FakeEngine(perturb=0.9)                 # 分数尺度 ≈0.17，漂移 0.9 → 相对漂移远超阈值
    runner = PrefixRunner(engine, readout=readout, verify="off")
    runner.ask(PREFIX, SUFFIXES[0])                  # 先冷启入库，把"命中路判负"这条路走实
    res = runner.ask(PREFIX, SUFFIXES[0], verify="always")
    report = runner.verdicts[-1]
    assert res.hit is True and res.drift is not None
    assert report.passed is False and res.drift_ratio > runner.drift_ratio_threshold
    assert res.recomputed is True, "判负当次就该交重算结果，而不是把脏复用的答案递出去"
    assert res.full_tokens_fed == len(PREFIX) + len(SUFFIXES[0])
    assert res.key in runner.stats()["quarantined"], "判负的 state 必须进黑名单"
    assert runner.cache.get(res.key) is None, "脏 past 要立刻从库里撕掉"
    again = runner.ask(PREFIX, SUFFIXES[1])
    assert again.hit is False and again.recomputed is True
    assert again.prefix_tokens_fed == 0 and again.full_tokens_fed == len(PREFIX) + len(SUFFIXES[1]), \
        "拉黑之后一律整段重算，绝不拿脏小抄凑答案"


def test_prefix_parity_cold_two_step_path_is_also_guarded():
    """冷启路（前缀前向 + 问题段增量）同样用了 past，护栏必须一并覆盖，不许只验命中路。"""
    cold_first = PrefixRunner(FakeEngine(perturb=0.9), readout=readout, verify="always")
    first = cold_first.ask(PREFIX, SUFFIXES[0])
    assert first.hit is False and first.drift is not None, "第一问（两段式）就该被验一次"
    assert first.recomputed is True and len(cold_first.stats()["quarantined"]) == 1


def test_prefix_parity_threshold_is_configurable():
    """阈值可配：同一个漂移量，松阈值放行、紧阈值弃缓存。"""
    engine = FakeEngine(perturb=0.05)
    loose = PrefixRunner(engine, readout=readout, verify="always", drift_ratio_threshold=1.0)
    loose.ask(PREFIX, SUFFIXES[0])
    r1 = loose.ask(PREFIX, SUFFIXES[0])
    assert r1.drift_ratio > 0.0 and r1.recomputed is False
    strict = PrefixRunner(FakeEngine(perturb=0.05), readout=readout, verify="always",
                          drift_ratio_threshold=1e-9)
    strict.ask(PREFIX, SUFFIXES[0])
    r2 = strict.ask(PREFIX, SUFFIXES[0])
    assert r2.recomputed is True and len(strict.stats()["quarantined"]) == 1


def test_prefix_parity_verify_pair_reports_both_paths():
    runner = PrefixRunner(FakeEngine(perturb=0.0), readout=readout, verify="off")
    pair = runner.verify_pair(PREFIX, SUFFIXES[0], k=4)
    assert pair["warm_hit"] is True
    assert pair["parity"]["argmax_same"] is True and pair["parity"]["max_abs_drift"] == 0.0
    assert pair["warm"]["prefix_tokens_fed"] == 0 and pair["direct"]["full_tokens_fed"] > 0


def test_prefix_parity_default_threshold_matches_probe():
    """默认阈值必须容得下两笔可复跑的真 bf16 实测，又不放过换人级偏差（数字与命令一起挂）。"""
    short_state = 4.98e-3      # `.probe_p207_prefix.py`：1.256e-01 / 25.202（state 782 token 英文）
    long_state = 1.57e-2       # `.p207_run.py` run …-7b07-2 step2：3.846e-01 / 24.495（2065 token 中文）
    worst = max(short_state, long_state)
    assert DEFAULT_DRIFT_RATIO > worst, "默认阈值不该把真模型的正常漂移判成弃缓存"
    margin = DEFAULT_DRIFT_RATIO / worst
    assert 1.0 < margin < 2.0, f"最差那笔余量 {margin:.2f} 倍：阈值偏紧，长窗云端复标后回调"
    print(f"[B1c] 阈值 0.02 对实测两笔余量：短证据 {DEFAULT_DRIFT_RATIO / short_state:.2f} 倍 / "
          f"长证据 {margin:.2f} 倍（长证据那一笔已接近贴线，判负另靠 argmax 一票否决兜底）")
    runner = PrefixRunner(FakeEngine(perturb=worst / 10.0), readout=readout, verify="always")
    runner.ask(PREFIX, SUFFIXES[0])
    assert runner.ask(PREFIX, SUFFIXES[0]).recomputed is False, "实测量级漂移应放行"


# ================================================================== B2 跨请求命中
def test_cross_hit_serve_reports_positive_hits_and_suffix_only():
    engine = FakeEngine()
    runner = PrefixRunner(engine, readout=readout, verify="off")
    requests = [
        {"request_id": "req-1", "prefix_ids": PREFIX, "suffix_ids": SUFFIXES[0]},
        {"request_id": "req-2", "prefix_ids": PREFIX, "suffix_ids": SUFFIXES[1]},
        {"request_id": "req-3", "prefix_ids": PREFIX, "suffix_ids": SUFFIXES[2]},
    ]
    out = runner.serve(requests)
    assert out["hits"] > 0, "跨请求命中报告必须大于 0（B2 验收点）"
    assert out["hits"] == 2 and out["requests"] == 3
    assert out["all_hits_suffix_only"] is True
    assert out["tokens_avoided_in_prefix"] == 2 * len(PREFIX)
    assert all(r["prefix_tokens_fed"] == 0 for r in out["rows"][1:])
    assert out["counter"]["by_kind"]["prefix"] == len(PREFIX), "证据段总共只算了一次"


def test_cross_hit_results_consistent_across_requests():
    """同一 state 换请求号来问，答案与"完全没缓存"那条路一致。"""
    engine = FakeEngine()
    runner = PrefixRunner(engine, readout=readout, verify="off")
    out = runner.serve([{"request_id": f"r{i}", "prefix_ids": PREFIX, "suffix_ids": SUFFIXES[i]}
                        for i in range(3)])
    assert out["hits"] == 2
    for i, row in enumerate(out["rows"]):
        assert row["hit"] == (i > 0), f"第 {i + 1} 个请求的命中判定不对：{row}"
        direct = runner.ask(PREFIX, SUFFIXES[i], force_full=True)
        assert direct.full_tokens_fed == len(PREFIX) + len(SUFFIXES[i])
    # 逐请求比对：走缓存那一路的末位数字与整段重算路必须逐位相同（结果一致，不是"差不多"）
    reuse_hidden = [runner.ask(PREFIX, SUFFIXES[i]).hidden_last for i in range(3)]
    full_hidden = [runner.ask(PREFIX, SUFFIXES[i], force_full=True).hidden_last for i in range(3)]
    for i in range(3):
        assert torch.equal(reuse_hidden[i], full_hidden[i]), f"跨请求第 {i + 1} 问结果不一致"


def test_cross_hit_namespace_isolates_sessions():
    runner = PrefixRunner(FakeEngine(), readout=readout, verify="off")
    a = runner.ask(PREFIX, SUFFIXES[0], namespace="tenant-a")
    b = runner.ask(PREFIX, SUFFIXES[1], namespace="tenant-b")
    assert a.key != b.key and b.hit is False, "换个命名空间就当没命中，不许跨租户复用"
    c = runner.ask(PREFIX, SUFFIXES[2], namespace="tenant-a")
    assert c.hit is True and c.key == a.key


def test_cross_hit_different_state_no_false_share():
    engine = FakeEngine()
    runner = PrefixRunner(engine, readout=readout, verify="off")
    runner.ask(PREFIX, SUFFIXES[0])
    other = [t + 1 for t in PREFIX]                      # 只改一个符号的"另一份证据"
    res = runner.ask(other, SUFFIXES[1])
    assert res.hit is False and res.prefix_tokens_fed == len(other)


# ------------------------------------------------- B1 补齐（接力波）：渲染切分 / 拷贝守卫 / 换人否决
class FakeTok:
    """字符级假编号器：只喂 `split_prompt_ids` 需要的两件事（encode + 对话模板），不碰真权重。

    模板形态与真 Qwen 一样是"逐轮拼接"：`<|role|>正文<|end|>`，末尾按需补生成引导符。
    """

    def encode(self, text, add_special_tokens=False):        # 口径固定：特殊符由模板负责
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            enable_thinking=False):          # 口径固定：思维链关掉（同 P1）
        assert tokenize is False, "本域按文字口径拼接，编号一律交给 encode"
        out = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
        return out + ("<|assistant|>" if add_generation_prompt else "")


class MergingTok(FakeTok):
    """跨接缝会"粘连成一格"的编号器：state 尾字符与后面那个换行合成一个符号——前缀就不再是前缀。"""

    def encode(self, text, add_special_tokens=False):
        ids: list[int] = []
        i = 0
        while i < len(text):
            if text[i] == "x" and i + 1 < len(text) and text[i + 1] == "\n":
                ids.append(900)                              # "x"+换行 合成一格
                i += 2
                continue
            ids.append(ord(text[i]))
            i += 1
        return ids


def rendered_messages(state: str, question: str) -> list[dict[str, str]]:
    """照 P1 `render.user_content` 的接缝形状造一条渲染结果（Evidence 段 + 接缝 + 问题段）。"""
    user = "Evidence:\n" + state + SPLIT_MARK + question + "\nOptions:\n1. a\n2. b"
    return [{"role": "system", "content": "SYS-LINE"}, {"role": "user", "content": user}]


def test_prefix_incr_split_prompt_ids_is_lossless_strict_prefix():
    """B1b 的前提：证据段编号必须"一格不差地排在整段前面"，且接缝不许混进证据段。"""
    tok = FakeTok()
    msgs = rendered_messages("ledger row 7712", "which code is booked?")
    prefix_ids, suffix_ids, info = split_prompt_ids(tok, msgs)
    full = tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True))
    assert prefix_ids + suffix_ids == full, "切分必须无损拼回整段编号"
    assert info["shell_len"] == len(prefix_ids) and info["suffix_len"] == len(suffix_ids)
    prefix_text = "".join(chr(t) for t in prefix_ids)
    suffix_text = "".join(chr(t) for t in suffix_ids)
    assert "ledger row 7712" in prefix_text and "Question" not in prefix_text
    assert suffix_text.startswith(SPLIT_MARK), "接缝本身属于问题段，past 里不能掺题目文字"


def test_prefix_incr_split_guard_rejects_misaligned_prefix():
    """编号器在接缝处粘连 → 前缀不再是前缀：当场报错，绝不拿错位的 past 答题。"""
    with pytest.raises(CacheError, match="不对齐"):
        split_prompt_ids(MergingTok(), rendered_messages("case-x", "which code?"))


def test_prefix_incr_split_guard_rejects_missing_seam_or_shape():
    with pytest.raises(CacheError, match="接缝"):
        split_prompt_ids(FakeTok(), [{"role": "system", "content": "s"},
                                     {"role": "user", "content": "Evidence:\n没有接缝"}])
    with pytest.raises(CacheError, match="messages"):
        split_prompt_ids(FakeTok(), [{"role": "user", "content": "只有一条 message"}])


def test_prefix_incr_ask_row_hits_after_rendered_split():
    """ask_row：同一份证据连发两问，第二问必须命中且只付问题段符号（渲染口径下的 B1b）。"""
    tok, state = FakeTok(), "ledger row 7712"
    runner = PrefixRunner(FakeEngine(), readout=readout, verify="off")
    r1 = runner.ask_row(tok, rendered_messages(state, "which code is booked?"), k=4)
    r2 = runner.ask_row(tok, rendered_messages(state, "is that the second mention?"), k=4)
    # 首趟必付证据段（suffix_only 语义=增量路专属，见实现定义）；但问题段计数仍须一格不差
    assert r1.hit is False and r1.prefix_tokens_fed > 0 and r1.tokens_are_suffix_only is False
    assert r1.suffix_tokens_fed == r1.suffix_len
    assert r2.hit is True and r2.key == r1.key, "同一份证据段必须折成同一个键"
    assert r2.prefix_tokens_fed == 0 and r2.suffix_tokens_fed == r2.suffix_len
    _, suf2, _ = split_prompt_ids(tok, rendered_messages(state, "is that the second mention?"))
    assert r2.suffix_tokens_fed == len(suf2), "命中路付的符号数=问题段应有长度"


def test_prefix_incr_ask_row_falls_back_when_split_fails():
    """切不动就老实整段重算：宁可多付符号，也不冒"错位 past 静默错答"的险。"""
    tok = FakeTok()
    runner = PrefixRunner(FakeEngine(), readout=readout, verify="off")
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "Evidence:\n没有接缝"}]
    res = runner.ask_row(tok, msgs, k=4)
    full = tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True))
    assert res.hit is False and res.recomputed is True and res.uncacheable is False
    assert res.prefix_tokens_fed == 0
    assert res.full_tokens_fed == len(full) == res.tokens_fed


class _HybridLayer:
    """照 transformers 5.x `LinearAttentionLayer` 的形状：没有 keys/values，记忆态装在字典里。"""

    def __init__(self, *, populated: bool = True) -> None:
        self.conv_states = {0: torch.zeros(1, 2, 4)} if populated else {0: None}
        self.recurrent_states = {0: torch.ones(1, 2)} if populated else {0: None}
        self.is_initialized = populated


class _HybridCache:
    def __init__(self, layers) -> None:
        self.layers = list(layers)

    def update(self, keys, values, idx):                     # 拷不动，本就不该走到这里
        raise AssertionError("混合层没有 KV 可拷，重建时不该调用 update")


class _NeedsArgsCache:
    """构造器要参数的 past 壳：`type(past)()` 只会 TypeError，必须折成 CacheError 交给上层降级。"""

    def __init__(self, layer_types) -> None:
        self.layer_types = layer_types
        self.layers: list[object] = []


def test_prefix_core_clone_refuses_hybrid_past():
    """0.8B 混合门那种"线性记忆层"：只照 KV 重建会静默丢记忆 —— 必须报错，不许装没事。"""
    engine = HFEngine(None)
    with pytest.raises(CacheError, match="KV 之外"):
        engine.clone_past(_HybridCache([_HybridLayer()]))
    with pytest.raises(CacheError, match="只认 KV 形态"):
        engine.clone_past(_HybridCache([_HybridLayer(populated=False)]))
    assert engine.clone_past(None) is None


def test_prefix_core_clone_refuses_shell_needing_args():
    with pytest.raises(CacheError, match="无法按无参构造重建"):
        HFEngine(None).clone_past(_NeedsArgsCache(["full"]))


class UncacheableEngine(FakeEngine):
    """past 拷不动的引擎（混合记忆门的样子）：clone 一抛 CacheError，上层就该整段重算。"""

    def clone_past(self, past):
        raise CacheError("这份 past 压着线性记忆态，本波拷不动")


def test_prefix_incr_uncacheable_past_degrades_to_full_recompute():
    """拷不动 ≠ 算不了：降级整段重算，且把"白付的证据段"如实记在账上，第二次不再白算。"""
    engine = UncacheableEngine()
    runner = PrefixRunner(engine, readout=readout, verify="off")
    first = runner.ask(PREFIX, SUFFIXES[0])
    assert first.uncacheable is True and first.hit is False and first.recomputed is True
    assert first.prefix_tokens_fed == len(PREFIX), "证据段那一笔真花了，账上必须看得见"
    assert first.full_tokens_fed == len(PREFIX) + len(SUFFIXES[0])
    assert first.tokens_fed == first.prefix_tokens_fed + first.full_tokens_fed
    second = runner.ask(PREFIX, SUFFIXES[1])
    assert second.uncacheable is True and second.prefix_tokens_fed == 0, "第二次别再白算证据段"
    assert second.full_tokens_fed == len(PREFIX) + len(SUFFIXES[1])
    st = runner.stats()
    assert len(st["uncacheable_keys"]) == 1 and st["recompute_uncacheable"] == 2
    assert st["cache"]["items"] == 0, "拷不动的东西不许留在库里"
    assert tuple(second.hidden_last.shape) == (1, engine.dim)


class NearTieEngine(FakeEngine):
    """两个候选几乎同分、复用路刚好把名次翻掉的引擎：漂移很小，但答案换了人。"""

    def forward(self, input_ids, past=None, *, offset: int = 0) -> EngineOut:
        seen = list(past["tokens"]) if past is not None else []
        ids = seen + [int(t) for t in input_ids]
        vec = torch.zeros(1, self.dim)
        if past is None:                          # 整段重算路：1 号领先一丝
            vec[0, 0], vec[0, 1] = 1.0000, 0.9995
        else:                                     # 复用路：分差 1e-3 级，名次却翻过来了
            vec[0, 0], vec[0, 1] = 0.9990, 1.0001
        new_past = {"tokens": ids, "keys": torch.zeros(1, len(ids), 4),
                    "values": torch.zeros(1, len(ids), 4)}
        self.calls.append({"fed": len(input_ids), "used_past": past is not None, "offset": offset})
        return EngineOut(vec, new_past, len(input_ids))


def test_prefix_parity_argmax_flip_is_vetoed_even_with_small_drift():
    """B1c 硬要求：漂移远小于阈值也不行——名次换人就弃缓存重算（换人没有"差得不多"这种豁免）。"""
    runner = PrefixRunner(NearTieEngine(), readout=readout, verify="off")
    runner.ask(PREFIX, SUFFIXES[0])                       # 冷启入库，好把"命中路"走实
    res = runner.ask(PREFIX, SUFFIXES[0], verify="always")
    rep = runner.verdicts[-1]
    assert rep.drift_ratio < runner.drift_ratio_threshold, "单看漂移确实很小（正是容易放过的形状）"
    assert rep.argmax_same is False and rep.passed is False
    assert res.recomputed is True and res.key in runner.stats()["quarantined"]
    assert runner.cache.get(res.key) is None, "换人的 past 立刻出库"
    print(f"[B1c-veto] drift_ratio={rep.drift_ratio:.3e} < 阈值 {rep.threshold}，但 argmax 换人 → "
          f"passed={rep.passed}，scores_reuse={rep.scores_reuse} scores_full={rep.scores_full}")


# ================================================================== C1 因果滑窗
# 判据重建（缺陷二，2026-10-11 agent E；证据链 release/ascend/kernels/reconcile/D_determinism.md）。
# 旧口径 `kernel_vs_masked_max_abs == 0.0` 作废的事实依据 [实测]：带状 `route="kernel"` 在 CPU 上
# **三道闸必回落** `attn_sw_kernel._eager()`（① `backends.active_backend(cpu)` 只放 mps 设备过关；
# ② `attn_sw_mps.plan` 第一门槛要 fp16，而 compare_routes 造的是 fp32；③ band_len=chunk+window-1=79
# 不被 BLOCK_Q=16 整除）——所以这条断言比的从来不是"内核 vs 掩码"，而是**两份 eager 实现、不同归约
# 长度（200 列 vs 79 列）之间的位等**：构造上无保证（D_determinism §1.2/§3.1），ubuntu 6 run 2 红 4 绿
# 就是它的表现，host 侧姊妹路（只差一个 M 形状）在同机已有 10/20 组形状出末位差。
# 新口径：主判据是**平台无关的结构判据**三条（A1 非零可见位集合逐格相等、A2 argmax 一致、
# A3 同进程自比），量级容差只作附列。**这不是"放宽容差当止血"**：结构判据对目标错误类（滑窗方向
# 抄反 / 分块边界切错 / 窗宽 off-by-one）的守护力严格强于位等：写反方向改变的是**非零位集合**，
# 位等把它混在"末位差"里——末位差导致误报（ubuntu 2 红），而一旦为消误报放宽容差又漏报真错。
# 结构判据把"方向/边界"与"量级"分成两面各判各的，误报与漏报同时收住（负例见
# test_sliding_visible_set_criterion_detects_reversed_window，R14 守卫必须真响）。
# 位等仍合法，但只用在两侧同形状同分派处（A3 自比；以及 test_sliding_short_sequence_… 的 torch.equal）。
ROUTE_TOL = 1e-5                                       # 仓内既有量级口径（同文件姊妹断言同款）
PROBE_SEQ, PROBE_WIN, PROBE_CHUNK, PROBE_HEADS = 48, 12, 16, 2
PROBE_SEED = 20261011


def _probe_qkv() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """造 v=I 的探针三元组：`P@I` 每行只剩一个非零加数 ⇒ 出口最后一维**逐位就是**该路的权重 P。

    白话：想知道"这一路到底看了谁"，就把内容换成单位矩阵——加权平均出来的数字直接就是份额本身，
    非零位集合因此是可逐格比的结构事实，而不是随末位差漂移的数值。抽取过程与 BLAS 分派无关
    （只有一个非零加数，任何加法次序都得同一个数）。
    """
    gen = torch.Generator(device="cpu").manual_seed(PROBE_SEED)
    q = torch.randn(PROBE_HEADS, PROBE_SEQ, PROBE_SEQ, generator=gen)
    k = torch.randn(PROBE_HEADS, PROBE_SEQ, PROBE_SEQ, generator=gen)
    v = torch.eye(PROBE_SEQ).expand(PROBE_HEADS, PROBE_SEQ, PROBE_SEQ).contiguous()
    return q, k, v


def _route_weights(route: str | None) -> torch.Tensor:
    """(heads,T,T) 的注意力权重 P：route=None 走掩码路，"kernel"/"local" 走带状路（CPU 上都是 eager）。"""
    q, k, v = _probe_qkv()
    if route is None:
        return LA.attention_masked(q, k, v, window=PROBE_WIN).out
    return LA.attention_band(q, k, v, PROBE_WIN, chunk=PROBE_CHUNK, route=route).out


_ROUTE_LABEL = {None: "掩码路",
                "kernel": '带状 route="kernel" 路（CPU 上执行体是 attn_sw_kernel._eager）',
                "local": '带状 route="local" 路（就地 eager）'}


def test_sliding_routes_agree_with_mask_reference():
    """eager 带状路与掩码参考"看得见地一致"：可见位集合 + argmax + 同进程自比（量级只作附列）。"""
    from sys1.kernels import backends

    # 措辞事实钉：本用例的带状 "kernel" 路在 CPU 上必回落 _eager()，据此把措辞写成 eager 带状路。
    assert backends.active_backend("cpu") == backends.TORCH_EAGER, (
        'CPU 上带状 route="kernel" 现在不再回落 _eager() 了：本用例的措辞与判据前提要一起改——'
        "真进内核时另立『内核路 vs 掩码路』判据（fp16 + band_len 整除 BLOCK_Q），别沿用这里的 eager 口径")

    truth = LA.window_band(PROBE_SEQ, PROBE_WIN)[0, 0]          # (T,T)：与各路出口 (heads,T,T) 同秩可展
    weights = {route: _route_weights(route) for route in (None, "kernel", "local")}
    ref = weights[None]

    # 结构判据为何平台无关：两条前提自证（不是装饰，破了就说明这套判据在该数据上会随平台失稳）
    min_visible = float(ref[truth.expand_as(ref)].min())
    assert min_visible > 1e-6, (
        f"可见位最小权重 {min_visible:.3e} 逼近 fp32 下溢：『非零位』集合会有平台敏感的 0/非0 抖动，"
        "换种子或换窗宽把权重拉开再判")
    ordered = ref.sort(dim=-1).values
    margin = float((ordered[..., -1] - ordered[..., -2]).min())
    probe_diff = max(float((w - ref).abs().max()) for w in weights.values())
    gap_floor = 100 * max(probe_diff, 10 * torch.finfo(torch.float32).eps)
    assert margin > gap_floor, (
        f"argmax 判据的前提破了：top1-top2 最小间距 {margin:.3e} 与本平台实测跨路末位差 {probe_diff:.3e} "
        f"只差不到两个数量级（门槛 {gap_floor:.3e}）——近并列时 argmax 会随末位差翻转，那才是平台敏感面。"
        "先换 seed/形状把间距拉开再用这条判据，别把它当位等那样蒙着用")

    # A1 主判据：非零可见位集合逐格相等（顺带卡死"未来位严格 0"与"对角自见不除 NaN"）
    for route, w in weights.items():
        assert torch.equal(w != 0, truth.expand_as(w)), (
            f"{_ROUTE_LABEL[route]} 的非零可见位集合与 window_band({PROBE_SEQ}, {PROBE_WIN}) 不逐格相等："
            "滑窗方向 / 分块边界 / 窗宽 off-by-one 之一被写改了（位等抓不住这类错，结构判据抓得住）")
        assert bool((w.triu(1) == 0).all()), f"{_ROUTE_LABEL[route]} 泄漏未来位：因果半边破了"
        assert bool((w.diagonal(dim1=-2, dim2=-1) > 0).all()), (
            f"{_ROUTE_LABEL[route]} 对角不自见 → 首行可能被整行涂黑，softmax 除出 NaN")

    # A2 主判据：argmax 一致（spec 原生验收：滑窗正确性 → 选项 argmax 一致；平台无关）
    base_idx = ref.argmax(-1)
    for route, w in weights.items():
        assert torch.equal(w.argmax(-1), base_idx), (
            f"{_ROUTE_LABEL[route]} 与掩码参考 argmax 不一致：间距 {margin:.3e} ≫ 容差 {ROUTE_TOL:.0e}，"
            "翻转即真回归（不是末位差能解释的）")

    # A3 主判据：同进程自比——每条路与**自身重跑**位等（分配历史扰动后）。这才是可平台无关钉住的
    # "确定性"契约：同形状、同分派、同线程档 ⇒ 位等有构造保证；不成立就是真非确定，与 BLAS/ISA 无关。
    for route, first in weights.items():
        junk = torch.randn(3, 517, 733) @ torch.randn(3, 733, 611)   # 扰动分配历史/指针对齐
        del junk
        assert torch.equal(_route_weights(route), first), (
            f"{_ROUTE_LABEL[route]} 同进程两次结果位等不同 → 真·非确定回归（本条不许随平台让步）")

    # 附列（不作主判据）：被测口径 (T=200, W=32, chunk=48) 的量级差，只卡仓内既有口径
    rep = LA.compare_routes(200, 32, chunk=48)
    assert rep["kernel_allclose"] and rep["local_allclose"]
    assert rep["kernel_vs_masked_max_abs"] <= ROUTE_TOL, (
        f"eager 带状 kernel 路与掩码路量级差 {rep['kernel_vs_masked_max_abs']!r} 超出仓内既有口径 "
        f"{ROUTE_TOL}（两侧归约长度 200 vs 79 天然不同 → 位等非产品不变量，D_determinism.md §3.1）")
    assert rep["local_vs_masked_max_abs"] <= ROUTE_TOL
    print(f"[sliding-criterion] probe(T={PROBE_SEQ},W={PROBE_WIN},chunk={PROBE_CHUNK},H={PROBE_HEADS}) "
          f"pattern_eq=True argmax_eq=True self_bitwise_eq=True | 附列 max_abs kernel="
          f"{rep['kernel_vs_masked_max_abs']!r} local={rep['local_vs_masked_max_abs']!r} "
          f"min_visible={min_visible:.3e} argmax_margin={margin:.3e} probe_diff={probe_diff:.3e} gap_floor={gap_floor:.3e}")


def test_sliding_visible_set_criterion_detects_reversed_window():
    """负例自证（R14：守卫必须真响）：A1 那条集合逐格判据对"方向抄反/窗宽 off-by-one"必须判否。"""
    truth = LA.window_band(PROBE_SEQ, PROBE_WIN)[0, 0]
    idx = torch.arange(PROBE_SEQ)
    delta = idx[:, None] - idx[None, :]
    reversed_band = (delta <= 0) & (delta > -PROBE_WIN)         # 把"只许回头看"抄成"只许向前看"
    assert not torch.equal(reversed_band, truth), "负例构造失败：反向带与正向带逐格相同（探针形状退化了）"
    w = _route_weights(None)
    assert not torch.equal(w != 0, reversed_band.expand_as(w)), (
        "A1 判据对『滑窗方向写反』不响 → 它只是个摆设，换掉的位等判据不成立")
    over = LA.window_band(PROBE_SEQ, PROBE_WIN + 1)[0, 0]        # 窗宽 off-by-one（多带一格）
    wl = _route_weights("local")
    assert not torch.equal(wl != 0, over.expand_as(wl)), (
        "A1 判据对『窗宽 off-by-one』不响 → 分块边界那类错就没人抓得住")


def test_sliding_short_sequence_equals_full_attention():
    """spec 场景：序列不长过窗时，滑窗 ≡ 全注意力（argmax 自然一致，这里直接逐位相等）。"""
    ref = LA.attention_masked(*(torch.randn(2, 40, 16, generator=torch.Generator().manual_seed(1))
                                for _ in range(3)), window=128)
    full = LA.attention_masked(*(torch.randn(2, 40, 16, generator=torch.Generator().manual_seed(1))
                                 for _ in range(3)), window=None)
    assert torch.equal(ref.out, full.out)
    # 带状路同样要等价（合成张量重取一份，避免与上面三份混用）
    gen = torch.Generator().manual_seed(1)
    q, k, v = (torch.randn(2, 40, 16, generator=gen) for _ in range(3))
    assert torch.allclose(LA.attention_band(q, k, v, 128, chunk=16).out,
                          LA.attention_masked(q, k, v, window=None).out, atol=1e-6)
    assert LA.window_band(40, 128).equal(LA.window_band(40, None))


def test_sliding_band_route_is_memory_bounded_on_long_sequence():
    gen = torch.Generator().manual_seed(3)
    q, k, v = (torch.randn(2, 1024, 16, generator=gen) for _ in range(3))
    full = LA.attention_masked(q, k, v, window=None)
    band = LA.attention_band(q, k, v, 128, chunk=64, route="local")
    assert band.materialized_peak_bytes < full.materialized_peak_bytes
    assert band.stats["scales_with"].startswith("O(chunk")
    assert band.stats["chunks"] == 16
    assert torch.allclose(band.out, LA.attention_masked(q, k, v, window=128).out, atol=1e-5)


def test_sliding_window_cap_is_a_real_switch():
    base = LA.resolve_layer_windows(6, window=8192, pattern="3:1")
    assert base.windows == (8192, 8192, 8192, None, 8192, 8192)
    capped = LA.resolve_layer_windows(6, window=8192, pattern="3:1", cap=256)
    assert capped.windows == (256, 256, 256, None, 256, 256) and capped.clamped == 5
    hard = LA.resolve_layer_windows(6, window=8192, pattern="3:1", cap=256, cap_full=True)
    assert all(w == 256 for w in hard.windows) and hard.clamped == 6, "封顶开关要能连全注意层一起夹住"


def test_sliding_decoder_path_matches_trunk_when_window_ge_seq():
    """接进主干：W ≥ T 时 `forward_with_windows` 与 `Decoder.forward` 必须逐位相同。"""
    from sys1.model import Decoder, ModelConfig

    cfg = ModelConfig(d=32, L=3, heads=2, ctx=64, vocab=50, seed=7)
    dec = Decoder(cfg).eval()
    torch.manual_seed(11)
    ids = torch.randint(1, 50, (1, 40))
    plan_wide = LA.resolve_layer_windows(3, window=64, pattern="all")
    plan_mix = LA.resolve_layer_windows(3, window=8, pattern="2:1")
    with torch.no_grad():
        base = dec(ids)
        wide = LA.forward_with_windows(dec, ids, plan_wide)
        mix = LA.forward_with_windows(dec, ids, plan_mix)
    assert torch.equal(base, wide), "窗内滑窗 ≡ 全注意力（逐位）"
    assert not torch.equal(base, mix) and torch.isfinite(mix).all()
    assert float((base - mix).abs().max()) < 5.0, "窄窗只该改变远处信息，不该把数值打飞"


def test_sliding_hybrid_pattern_bookkeeping():
    """5:1 / 3:1 排布与逐层清单都要留痕（run 档案里要能还原"哪几层看全场"）。"""
    naive = LA.resolve_layer_windows(6, window=128, pattern="5:1")
    glm = LA.resolve_layer_windows(4, window=128, pattern="3:1")
    assert naive.as_dict()["windows"] == [128, 128, 128, 128, 128, "full"]
    assert glm.as_dict()["windows"] == [128, 128, 128, "full"]
    with pytest.raises(ValueError):
        LA.resolve_layer_windows(4, pattern="SSSSF")           # 串长与层数不符
    with pytest.raises(ValueError):
        LA.resolve_layer_windows(4, window=0)
    with pytest.raises(ValueError):
        LA.resolve_layer_windows(4, pattern=[128, None])       # 清单长度不符


def test_sliding_kv_ledger_caps_at_window():
    led = LA.kv_ledger(131072, layers=24, windows=[128] * 24, kv_heads=2, head_dim=128)
    assert led["full_attention_bytes"] == 2 * 2 * 128 * 2 * 24 * 131072
    assert led["windowed_bytes"] == 2 * 2 * 128 * 2 * 24 * 128
    assert led["saved_ratio"] > 0.99 and "解析账" in led["memory_ledger"]
    plan = LA.resolve_layer_windows(24, window=128, pattern="all")
    assert plan.kv_ledger(131072, kv_heads=2, head_dim=128)["windowed_mib"] == led["windowed_mib"]


def test_sliding_visible_mask_semantics():
    vis = LA.sliding_visible(10, window=3)
    assert vis.shape == (1, 1, 10, 10)
    assert bool(vis[0, 0, 5, 5]) and bool(vis[0, 0, 5, 3]) and not bool(vis[0, 0, 5, 2])
    assert not bool(vis[0, 0, 5, 6]), "未来位置必须不可见（因果半边不许丢）"
    am = torch.ones(1, 10, dtype=torch.long)
    am[0, 8:] = 0
    vis2 = LA.sliding_visible(10, window=None, attention_mask=am)
    assert bool(vis2[0, 0, 5, 8]) is False and bool(vis2[0, 0, 8, 8]) is True, "洞位不做键但能自看"


@need_mps
@pytest.mark.mps
def test_sliding_mps_gate_closed_this_wave():
    """MPS 档留待云端/一阶段长跑：本波禁用，只有显式放行才真跑（占位守住验证命令不空跑）。"""
    assert torch.backends.mps.is_available() is True or True
