"""p2-10 serving 本地半场单测：A1 HTTP 服务、A2 endpoint↔本地一致。

口径（与 test_longctx.py 同源）：
  * 全部 CPU 跑完——HTTP 与一致两路都用"纯 python 假引擎"，真 backbone（0.6B 替身）的
    前向凭据走 run 档案（`python -m serving.latency --new-run p210-*`），不在单测里挂
    六秒模型加载；
  * 资源纪律：禁用 MPS（归一阶段训练长跑）；本波一律 CPU 路验证；
  * 用例名按孙任务分关键字：http / parity / hit_count，与二级 tasks.md 的验证命令
    `-k <关键字>` 一一对应（hit_count = B2 前缀复用的进程内门）。
  * B2 门用的是**同一个假引擎**加一副最小 KV 面（`FakeKVEngine`）：末位数字一律回交
    `forward_hidden` 现算，复用路与重算路因此共用一条公式，护栏只会判等——命中计数涨没涨、
    命中题有没有重付前缀符号，测的都是服务路径，不是替身像不像真模型。

假引擎的设计意图：它只承诺"同输入同输出"的确定性，不承诺像真模型。于是
`DecisionService` 的进程内路（= 评测侧 `--model` 本地路的替身）与 HTTP `/v1/systemone`
路（= 评测侧 `--endpoint` 路的替身）走的是同一份权重、同一套 decision 原函数——
parity 用例断言的正是"网络这层皮不改变读数"（argmax 不换人、份额在容差内相同）。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest
import torch

from serving.prefix_cache import EngineOut
from serving.server import (
    BATCH_PATH,
    HEALTH_PATH,
    STATS_PATH,
    SYSTEMONE_PATH,
    DecisionService,
    spawn,
)
from sys1.decision import RENDER_VERSION, THINK_OFF_SUFFIX


# ------------------------------------------------------------------ 假引擎/假分词器（CPU 替身）
class FakeTokenizer:
    """把消息拼成整段文字并按字符编号；模板尾巴钉死"关闭思考"记号，与真模板同构。"""

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False) -> str:  # noqa: ANN001, ANN003 - 对齐 HF 签名
        text = "\n".join(str(m.get("content", "")) for m in messages)
        return text + THINK_OFF_SUFFIX if add_generation_prompt else text

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [(ord(ch) % 250) + 2 for ch in text]


class FakeEngine:
    """确定性假引擎：`hidden[b,t,j] = f(该行到 t 为止的编号累加, j, t)`，与模型无关。

    白话：只要送进去的编号一样，交出的逐位置数字就一样；26 个字母各占输出层一维
    （head 取单位阵），于是候选分 = 末位数字的前 k 个分量，读得出稳定名次。
    """

    def __init__(self, *, dim: int = 26) -> None:
        self.dim = dim
        self.tokenizer = FakeTokenizer()
        self.head_weight = torch.eye(dim)
        self.letter_ids = tuple(range(dim))
        self.pad_id = 0
        self.dtype = torch.float32
        self.device = "cpu"
        self.calls: list[tuple[int, int]] = []
        self.kv: "FakeKVEngine" = None              # kv_engine() 交出的替身面（B2：runner 用它）

    def forward_hidden(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        batch, width = int(input_ids.shape[0]), int(input_ids.shape[1])
        out = torch.zeros(batch, width, self.dim, dtype=torch.float32)
        for b in range(batch):
            running = 0
            for t in range(width):
                running += int(input_ids[b, t])
                for j in range(self.dim):
                    out[b, t, j] = float((running * (j + 3) + t) % 13) - 6.0 + 0.01 * j
        self.calls.append((batch, width))
        return out

    def kv_engine(self) -> "FakeKVEngine":
        """交出可续 KV 的替身面（B2）：与真引擎 `TorchEngine.kv_engine()` 同一张接口。"""
        self.kv = FakeKVEngine(self)
        return self.kv

    def describe(self) -> dict:
        return {
            "engine": "fake-deterministic", "loader": "minimal", "device": "cpu",
            "dtype": "float32", "backend": "torch_eager", "head_shape": [self.dim, self.dim],
            "pad_id": self.pad_id, "render_version": RENDER_VERSION, "provenance": {},
        }


class FakeKVEngine:
    """`KVEngine` 协议的假实现：past = 已看过的编号列，数字回交 `FakeEngine.forward_hidden`。

    白话：缓存层对"模型"的全部要求就三件事——能带着上次的小抄增量前向、能把小抄拷一份、
    能报小抄有多重。这里把"看过哪些字"直接当小抄，算数字时仍走服务侧那同一个
    `forward_hidden`，所以"复用路（只喂问题段）"与"重算路（整段重来）"交出的末位数字必然
    逐位相同。于是护栏判负只剩一种可能：服务路径真算错了——用例断言"命中数 ≥1"时测的是
    记账与付费面，而不是替身像不像真模型。增量前向还顺手核对 offset 必须等于已看过的长度：
    位置抬错（忘抬 offset）在真模型上是旋转角度全错，在这儿会当场炸给测试看，不静默到底。
    """

    def __init__(self, owner: FakeEngine) -> None:
        self.owner = owner
        self.forwards: list[dict] = []          # 逐次前向的开销账（命中/付费断言的独立凭据）

    def forward(self, input_ids, past=None, *, offset: int = 0) -> EngineOut:  # noqa: ANN001 - 协议签名
        seen = list(past["tokens"]) if past is not None else []
        fed = [int(t) for t in input_ids]
        if past is not None and offset != len(seen):
            raise AssertionError(
                f"增量前向的 offset={offset} 与已看过 {len(seen)} 个符号不符——位置抬错即全错")
        ids = seen + fed
        hidden = self.owner.forward_hidden(
            torch.tensor([ids], dtype=torch.long), torch.ones(1, len(ids), dtype=torch.long))
        last = hidden[0, len(ids) - 1].unsqueeze(0)          # (1, d)：读点 = 最后一个真位置
        new_past = {"tokens": ids,
                    "keys": torch.zeros(1, len(ids), 4), "values": torch.zeros(1, len(ids), 4)}
        self.forwards.append({"fed": len(fed), "used_past": past is not None})
        return EngineOut(last, new_past, len(fed))

    def clone_past(self, past):                              # noqa: ANN001 - 协议签名
        """拷小抄：编号列给新表、两张小张量 .clone()——原件被涂改会让后续命中全是脏的。"""
        return {"tokens": list(past["tokens"]),
                "keys": past["keys"].clone(), "values": past["values"].clone()}

    def past_bytes(self, past) -> int:                       # noqa: ANN001 - 协议签名
        return (past["keys"].numel() + past["values"].numel()) * past["keys"].element_size()


@pytest.fixture(scope="module")
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture(scope="module")
def service(engine: FakeEngine) -> DecisionService:
    return DecisionService(engine)


@pytest.fixture(scope="module")
def server(service: DecisionService):
    srv = spawn(service)
    try:
        yield srv
    finally:
        srv.stop()


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(url: str, payload: dict) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


REQUEST = {
    "id": "p210-req",
    "state": "weather: clear sky at 07:00, humidity 40%, rising pressure",
    "questions": {
        "q1": {"type": "choice", "instructions": "Which option fits the evidence?",
               "criteria": {"a": "sunny", "b": "rain", "c": "cloudy"}},
        "q2": {"type": "noul", "instructions": "Does the evidence support the claim?"},
        "q3": {"type": "score", "instructions": "Rate the confidence.",
               "criteria": ["low", "medium", "high"]},
    },
}


# ================================================================== A1 HTTP 服务（-k http）
def test_http_health_reports_engine_and_render_version(server):
    """/health 交回 status=ok 与引擎画像、渲染版号——起没起来一眼看清。"""
    body = _get(server.base_url + HEALTH_PATH)
    assert body["status"] == "ok"
    assert body["render_version"] == RENDER_VERSION
    assert body["engine"]["engine"] == "fake-deterministic"
    assert body["engine"]["device"] == "cpu"


def test_http_decide_single_answers_all_three_qtypes(server):
    """/v1/systemone 一次答 choice/noul/score 三型：份额和为 1、ms 字段齐全。"""
    body = _post(server.endpoint, REQUEST)
    assert body["id"] == "p210-req"
    assert set(body["answers"]) == {"q1", "q2", "q3"}
    # 每型候选集与 decision 契约对齐（choice→abc / noul→true,false / score→0,1,2）
    assert set(body["answers"]["q1"]) == {"a", "b", "c"}
    assert set(body["answers"]["q2"]) == {"true", "false"}
    assert set(body["answers"]["q3"]) == {"0", "1", "2"}
    for qid, probs in body["answers"].items():
        assert abs(sum(probs.values()) - 1.0) < 1e-5, qid
    assert isinstance(body["ms"], (int, float)) and body["ms"] >= 0.0
    assert body["ms_semantics"] == "forward_plus_readout_of_this_request"
    assert body["render_version"] == RENDER_VERSION
    assert "engine" in body


def test_http_batch_returns_five_with_ms_fields(server):
    """批量端点承 5 份请求：回 5 组 answers，ms/ms_batch 齐全（spec 批量场景）。"""
    batch = [{"id": f"b{i}", "state": REQUEST["state"],
              "questions": {"q1": {"type": "choice", "instructions": f"Which fits? #{i}",
                                   "criteria": {"a": "sunny", "b": "rain"}}}} for i in range(5)]
    body = _post(server.base_url + BATCH_PATH, {"requests": batch})
    assert len(body["results"]) == 5
    assert [r["id"] for r in body["results"]] == [f"b{i}" for i in range(5)]
    for r in body["results"]:
        assert set(r["answers"]["q1"]) == {"a", "b"}
        assert isinstance(r["ms"], (int, float))
        assert r["ms_batch"] == body["results"][0]["ms_batch"]
        assert r["ms_semantics"] == "amortized_within_batch"


def test_http_rejects_empty_questions_with_400(server):
    """不合协议的请求（空 questions）当场 400 并说清原因，绝不当 500 掩盖。"""
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(server.endpoint, {"id": "bad", "state": "s", "questions": {}})
    assert exc.value.code == 400
    detail = json.loads(exc.value.read().decode("utf-8"))
    assert detail["error"] == "bad_request"


def test_http_stats_endpoint_reports_counters(server):
    """/stats 交回记账字段（requests/rows/forwards/tokens_fed/prefix_*）——B2 验收面。"""
    body = _get(server.base_url + STATS_PATH)
    for key in ("requests", "rows", "forwards", "tokens_fed",
                "prefix_hits", "prefix_misses", "prefix_enabled"):
        assert key in body
    assert body["render_version"] == RENDER_VERSION


# ================================================================== A2 双路一致（-k parity）
def test_parity_endpoint_matches_inprocess(server, service):
    """同一副权重：HTTP `--endpoint` 路 vs 进程内 `--model` 路 —— argmax 不换人、份额同容差。

    白话：服务只是 decision 的网络皮，序列化/合批不该挪动任何一个名次。这里拿同一个
    请求分别问进程内和走 HTTP，逐题比 argmax 与整条份额分布。
    """
    local = service.decide(REQUEST)                 # 进程内路（--model 替身）
    served = _post(server.endpoint, REQUEST)        # HTTP 路（--endpoint 替身）
    assert set(local["answers"]) == set(served["answers"])
    for qid in local["answers"]:
        lo, so = local["answers"][qid], served["answers"][qid]
        assert set(lo) == set(so), qid
        assert max(lo, key=lo.get) == max(so, key=so.get), f"argmax 换人：{qid}"
        for code in lo:
            assert abs(lo[code] - so[code]) < 1e-6, f"{qid}/{code} 份额漂移超容差"


def test_parity_pad_invariance_batched_vs_single(server, service):
    """合批右补空洞不改变名次：同题单问 vs 混进长题批里 argmax 一致（spec"三问一次前向"的稳定性）。"""
    short = {"id": "s", "state": REQUEST["state"], "questions": {
        "q1": {"type": "choice", "instructions": "Short?", "criteria": {"a": "x", "b": "y"}}}}
    solo = service.decide(short)["answers"]["q1"]
    padded = {"id": "L", "state": REQUEST["state"], "questions": {
        "q_long": {"type": "choice", "instructions": "z" * 200,
                   "criteria": {"a": "x", "b": "y", "c": "zz", "d": "ww"}},
        **short["questions"]}}
    mixed = service.decide(padded)["answers"]["q1"]
    assert max(solo, key=solo.get) == max(mixed, key=mixed.get)
    for code in solo:
        assert abs(solo[code] - mixed[code]) < 1e-5


# ================================================================== B2 前缀复用（-k hit_count）
def test_hit_count_service_path_counts_hits_and_never_repays_prefix():
    """B2 进程内门：同 state 连问 → 服务面命中计数按 spec 达 ≥2，且命中题不再为前缀付一个符号。

    白话：`prefix=True` 起的服务把"证据段"（state）算一次就存进缓存，之后同一段证据的每一问
    只喂问题段。这条用例盯四件事，缺一样 B2 都不算过：① 命中计数真的涨了——记账在服务侧
    `stats()["prefix_hits"]`，不是只有 runner 内部自娱自乐；② 每个命中题的开销账写着
    `prefix_tokens_fed==0` 且为问题段付的恰是问题段长度（"没重付前缀"是硬账，不是"应该差不多"）；
    ③ 跨请求同 state 还接着命中（缓存不是只活一轮）；④ 护栏没被替身的数值噪声误触发（判负 0 次、
    黑名单空、服务侧重算 0 次）——一旦判负，后面全是整段重算，命中数看着对也没有复用效果。
    最后拿 answers 与 prefix=False 的服务逐题比 argmax：复用不改变读数，才是 B2 能进生产面的前提。
    """
    plain = DecisionService(FakeEngine())                  # 对照组：完全不走复用
    ref = plain.decide(REQUEST)["answers"]

    eng = FakeEngine()
    svc = DecisionService(eng, prefix=True, namespace="p210-hitcount")
    assert svc.runner is not None, "prefix=True 没拼出 runner（B2 面根本没挂上）"
    assert eng.kv is not None, "服务的 KV 面没被取用（runner 走的不是这副假引擎）"

    base = svc.stats()
    assert base["prefix_enabled"] is True and base["prefix_hits"] == 0

    body = svc.decide(REQUEST)                             # 一份请求三问：共用同一段证据
    rows = body["prefix"]["rows"]
    assert len(rows) == 3
    st = svc.stats()
    # spec「命中可观测」的口径是"同 state 连发 3 问 → 命中计数 ≥2"（第一问冷启、后两问复用）；
    # 派单里的 ≥1 只是下限，按 spec 收紧才不会让"只命中一次"的退化实现混过关
    assert st["prefix_hits"] - base["prefix_hits"] >= 2, \
        f"服务路径命中不足（三问同 state 应命中 ≥2）：{st['prefix_hits']}"
    hits = [r for r in rows if r["hit"]]
    assert hits, f"三问同 state 却一次没命中：{[(r['hit'], r['prefix_tokens_fed']) for r in rows]}"
    for r in hits:
        assert r["tokens_are_suffix_only"], f"命中题重付了前缀：{r}"
        assert r["prefix_tokens_fed"] == 0 and r["suffix_tokens_fed"] == r["suffix_len"]
    cold = [r for r in rows if not r["hit"]]
    assert sum(r["prefix_tokens_fed"] for r in rows) == sum(r["prefix_tokens_fed"] for r in cold), \
        "证据段被付了不止一次——前缀只该在冷启那回付费"

    again = svc.decide({**REQUEST, "id": "p210-req-2"})     # 跨请求同 state：该继续命中
    again_rows = again["prefix"]["rows"]
    st2 = svc.stats()
    assert st2["prefix_hits"] > st["prefix_hits"], \
        f"跨请求没接着命中：{st['prefix_hits']}->{st2['prefix_hits']}"
    assert all(r["tokens_are_suffix_only"] for r in again_rows), "第二次请求重付了前缀符号"

    run = svc.stats()["runner"]
    assert run["parity_failed"] == 0, f"护栏判负 {run['parity_failed']} 次（替身公式不一致？）"
    assert run["quarantined"] == [], f"键被拉黑，之后全是重算：{run['quarantined']}"
    assert svc.stats()["prefix_recomputes"] == 0, "服务侧记到重算——命中计数再高也没复用效果"
    for qid in ref:
        best_ref = max(ref[qid], key=ref[qid].get)
        best_hit = max(again["answers"][qid], key=again["answers"][qid].get)
        assert best_ref == best_hit, f"复用路改了名次：{qid} {best_ref} -> {best_hit}"
