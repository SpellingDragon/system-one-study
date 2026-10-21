"""serving/latency.py — p2-10 本地半场取证：延迟协议（B1）+ 双路一致（A2）+ 命中计数（B2）。

【做什么】
    拿一台真跑着 `/v1/systemone` 的服务出三类凭据，全部落进 run 档案：
    ① B1 延迟协议——固定请求模板（同一份 state + 1 choice + 1 noul + 1 score，一次前向答完），
       预热 20 条不记、正式 50 条串行逐条掐表，交 mean/P50/P95（外加 P99/最小/最大），
       端到端与"前向+读数"两列并排；设备/位宽/后端三项随报告带出。
    ② A2 双路一致——同一副权重、同一套渲染，走两条路读数：本地组批路（照 P1
       `sys1/eval/predict.py` 的分桶组批与读数次序）与服务 HTTP 路（`/v1/systemone`），
       逐题比 argmax 与份额。
    ③ B2 命中计数——开 p2-07 前缀复用后，同 state 连发多问，从 `/stats` 读命中数增量。

【怎么做】
    客户端不自己发明口径：`measure_latency()` 直接调 `sys1/eval/speed.py::measure_speed`
    （一阶段定的那条：串行 + 预热豁免 + torch.quantile 线性插值 + 设备三标注 +
    `output_tokens=0` 自证），本文件只提供一个满足它那套 duck-type 的 HTTP 适配器
    （`predict/encode/describe`）。`local_predict()` 复用 `predict.group_by_k` 与
    `predict._make_batch` 原件，唯一换掉的是"编号那一步"——backbone 必须套对话模板，
    而 P1 的小 GPT 直接编 `prompt_text`；这是换脑后不得不换的一环，其余次序照搬。
    数字落盘走 `sys1/runs`：`artifacts/*.json` + `metrics.jsonl` + notes 三行（假设/观察/结论）。

【为什么】
    被否方案一：在本文件里重写一套分位数与预热逻辑——与 P1 耗时轴两份口径，跨阶段
    的速度数字没法并排看（P50 到底含不含包装成本都会变成各说各话）。
    被否方案二：用进程内函数调用代替 HTTP 出延迟数——那测到的是函数耗时，不是服务
    耗时；协议转换与序列化正是线上手感的一部分，必须真发一次请求。
    被否方案三：把 warm-up 计入或把 N 提到几百——CPU 单次前向数百毫秒，70 条已够出
    分位；加条数只是加长噪声，且与一阶段"固定 warm-up + N≥30"的纪律同源更好比。
    被否方案四：只报 mean——一次换页就能把均值拖花，P50/P95 才代表手感与卡顿。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from serving.server import (
    BATCH_PATH,
    HEALTH_PATH,
    STATS_PATH,
    SYSTEMONE_PATH,
    DecisionService,
    PreparedRow,
    _encode_ids,
    _template_text,
    build_engine,
    prepare_request,
    spawn,
    verify_think_tail,
)
from sys1.decision import (
    LETTERS,
    RENDER_VERSION,
    THINK_OFF_SUFFIX,
    option_scores,
    temperatures_for,
    to_probs,
)
from sys1.eval import predict as P
from sys1.eval import speed as SP

#: B1 的协议常量（任务书钉死：warm-up 20 + N=50 串行）
WARMUP_PROTOCOL = 20
SAMPLES_PROTOCOL = 50
#: 双路一致的份额容差（float32 在不同批宽下末位加和顺序不同，允许极小漂移）
MAX_PROB_DRIFT = 5e-3
#: 延迟请求模板：一份证据 + 三种类型各一问，一次前向答完
FIXED_STATE = (
    "incident: payment retries doubled after 09:15; deploy 4b2c1 finished at 09:10; "
    "error rate 1.8%, rollback window 10 minutes"
)
FIXED_QUESTIONS: dict[str, dict[str, Any]] = {
    "action": {"type": "choice", "instructions": "Which single action best fits the evidence?",
               "criteria": {"a": "rollback 4b2c1", "b": "raise retry limit", "c": "wait one window"}},
    "causal": {"type": "noul", "instructions": "The deploy caused the retry increase.",
               "criteria": {"false": "no link shown", "true": "timing fits"}},
    "severity": {"type": "score", "instructions": "How severe is this incident right now?",
                 "criteria": {"1": "noise", "2": "notable", "3": "sev2"}},
}


# ---------------------------------------------------------------- 记录模板与取数
def template_records(count: int, *, id_prefix: str = "p210-lat") -> list[dict[str, Any]]:
    """造 `count` 条固定模板记录（一 request = 三问一次前向），供 B1 串行计时。

    白话：延迟要可比，题面就必须一动不动——同一段证据、同样三道题（选择/是否/打分），
    一次请求问完。变来变去的题面量出来的分布掺的是题目难度，不是服务的快慢。
    """
    return [{"id": f"{id_prefix}-{i:03d}", "sample": {"state": FIXED_STATE,
            "questions": {qid: dict(spec) for qid, spec in FIXED_QUESTIONS.items()}}}
            for i in range(int(count))]


def load_eval_records(limit: int = 10, *, axis: str = "quality") -> list[dict[str, Any]]:
    """从评测集取真题（A2 双路一致用），一行一题，取不到就交空表不造题。

    白话：一致的凭据要拿真题目来对，不能自己编几道顺手的——题目从 registry 装配好的
    副本里原样读出来，谁将来复跑都能拿到同一批。
    """
    from sys1.eval import run as E

    return E.load_axis_records(axis, limit=limit)


# ---------------------------------------------------------------- HTTP 客户端（measure_speed 的鸭子）
def http_json(url: str, payload: dict[str, Any] | None = None, *, timeout: float = 120.0) -> dict[str, Any]:
    """一条 json 进、一条 json 出的最薄客户端（None 载荷走 GET）。"""
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


@dataclass
class ServedRow:
    """一次服务问答的窄结果：耗时、逐题份额、胜出代号（延迟与一致两条轴都吃这几个字段）。"""

    id: str
    ms: float
    answers: dict[str, dict[str, float]]
    argmax: dict[str, str] = field(default_factory=dict)
    ms_semantics: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class ServicePredictor:
    """打真 HTTP 的预测器：满足 `measure_speed` 的鸭子口（predict/encode/describe）。

    白话：把"发一封问信、收一份份额"这件事做成评测侧认识的那副形状——它要单条串行
    就问单条，它要入参长度就报模板套壳后的符号数，它要设备画像就把服务那副身板的
    设备/位宽/后端原样抄过去，绝不替服务吹成 "remote" 了事。
    """

    def __init__(self, base_url: str, *, engine_desc: dict[str, Any] | None = None,
                 tokenizer: Any = None, timeout: float = 120.0) -> None:
        self.endpoint = base_url.rstrip("/") + SYSTEMONE_PATH
        self.batch_endpoint = base_url.rstrip("/") + BATCH_PATH
        self.stats_url = base_url.rstrip("/") + STATS_PATH
        self.health_url = base_url.rstrip("/") + HEALTH_PATH
        self.timeout = timeout
        self.engine_desc = dict(engine_desc or {})
        self.tokenizer = tokenizer
        self.calls = 0

    def predict(self, records: Sequence[dict[str, Any]], *, batch_size: int = 1) -> list[ServedRow]:
        """逐条发一次 `/v1/systemone`（一条请求 = 模板里的三问 = 服务侧一次前向）。"""
        out: list[ServedRow] = []
        for rec in records:
            sample = rec["sample"]
            payload = {"id": rec.get("id"), "state": sample["state"], "questions": sample["questions"]}
            reply = http_json(self.endpoint, payload, timeout=self.timeout)
            self.calls += 1
            answers = reply.get("answers") or {}
            argmax = {}
            for qid, probs in answers.items():
                if probs:
                    argmax[qid] = max(probs.items(), key=lambda kv: (kv[1], -_letter_rank(kv[0], list(probs))))[0]
            out.append(ServedRow(id=str(reply.get("id") or rec.get("id")), ms=float(reply.get("ms") or 0.0),
                                 answers=answers, argmax=argmax,
                                 ms_semantics=str(reply.get("ms_semantics", "")), raw=reply))
        return out

    def encode(self, records: Sequence[dict[str, Any]]) -> list[Any]:
        """报每条请求真喂进去的符号数（模板套壳后的三问编号拼接），供吞吐列用。"""
        import types

        out = []
        for rec in records:
            ids: list[int] = []
            if self.tokenizer is not None:
                rows = prepare_request(self.tokenizer, rec["sample"])
                ids = [i for r in rows for i in r.ids]
            out.append(types.SimpleNamespace(input_ids=ids, length=len(ids)))
        return out

    def describe(self) -> dict[str, Any]:
        """入口摘要：`mode=endpoint` 说明是打服务的，但设备/位宽/后端照服务侧真话写。"""
        desc = {"mode": "endpoint", "endpoint": self.endpoint, "timeout": self.timeout}
        desc.update({k: v for k, v in self.engine_desc.items()
                     if k in ("device", "dtype", "backend", "engine", "loader", "render_version")})
        desc.setdefault("device", self.engine_desc.get("device", "unknown"))
        desc.setdefault("dtype", self.engine_desc.get("dtype", "unknown"))
        desc.setdefault("backend", self.engine_desc.get("backend", "unknown"))
        return desc


# ---------------------------------------------------------------- 本地组批路（P1 口径的对照组）
def encode_with_template(record: dict[str, Any], engine: Any) -> list[P.EncodedSample]:
    """把一条记录编成 P1 的可组批样本，只把"编号"换成 backbone 必需的模板套壳。

    白话：抄信、定字母序、记人工份额，这些和 P1 的 `encode_record` 一模一样；唯一的
    分别是小 GPT 时代直接把整段文字编号，而 backbone 必须先套上对话壳（还要关闭思考）
    才有合法的读点。换的只有这一环，样本形状仍然与 P1 同构，所以组批那一步能复用原件。
    """
    sample = record["sample"]
    state = sample.get("state") or ""
    targets = sample.get("targets") or {}
    out: list[P.EncodedSample] = []
    for qid, spec in sample["questions"].items():
        rows = prepare_request(engine.tokenizer, {"state": state, "questions": {qid: spec}})
        row = rows[0]
        tmap = targets.get(qid) or {}
        mass = sum(float(v) for v in tmap.values()) or 1.0
        target = [float(tmap.get(code, 0.0)) / mass for code in row.order]
        out.append(P.EncodedSample(
            id=str(record.get("id") or qid), qid=row.qid, qtype=row.qtype, k=row.k,
            options=list(row.order), letter_ids=[int(v) for v in engine.letter_ids[:row.k]],
            input_ids=list(row.ids), length=len(row.ids), target=target,
        ))
    return out


def local_predict(engine: Any, records: Sequence[dict[str, Any]], *, batch_size: int = 8,
                  temperatures: dict[str, float] | None = None) -> list[ServedRow]:
    """进程内按 P1 的分桶组批读数（`group_by_k` + `_make_batch` 原件），交与服务同形状的行。

    白话：这是"不开服务、自己按评测那套逐桶问一遍"的对照路——桶的划法、补空洞的写法、
    分数折份额的次序都跟着 `predict.py` 走。它与服务唯一的差别是批宽与是否过网络，
    所以两边对得上，就说明服务这条新皮没把口径带偏。
    """
    encs: list[P.EncodedSample] = []
    for rec in records:
        encs.extend(encode_with_template(rec, engine))
    for i, ex in enumerate(encs):
        ex.seq = i
    by_record: dict[str, dict[str, dict[str, float]]] = {}
    ms_by_record: dict[str, float] = {}
    for rows in P.group_by_k(encs).values():
        for start in range(0, len(rows), max(1, batch_size)):
            chunk = rows[start:start + batch_size]
            batch = P._make_batch(chunk, int(engine.pad_id))
            t0 = time.perf_counter()
            with torch.no_grad():
                hidden = engine.forward_hidden(batch["input_ids"], batch["attn_mask"])
                scores = option_scores(hidden, engine.head_weight, batch["letter_ids"], batch["lengths"])
                values, _calibrated = temperatures_for(batch["qtypes"], temperatures, len(chunk))
                scaled = scores / torch.tensor(values, dtype=scores.dtype, device=scores.device).unsqueeze(1)
                probs = to_probs(scaled, 1.0)
            elapsed = (time.perf_counter() - t0) * 1000.0 / len(chunk)
            for i, ex in enumerate(chunk):
                answers = {ex.options[j]: float(probs[i][j]) for j in range(ex.k)}
                by_record.setdefault(ex.id, {})[ex.qid] = answers
                ms_by_record[ex.id] = ms_by_record.get(ex.id, 0.0) + elapsed
    out: list[ServedRow] = []
    for rec in records:
        rid = str(rec.get("id"))
        answers = by_record.get(rid, {})
        out.append(ServedRow(id=rid, ms=float(ms_by_record.get(rid, 0.0)), answers=answers,
                             argmax={q: max(p.items(), key=lambda kv: kv[1])[0] for q, p in answers.items()},
                             ms_semantics="local_batched_forward_plus_readout"))
    return out


def _letter_rank(code: str, order: list[str]) -> int:
    """候选代号在本题字母序里的位次（argmax 并列时取更靠左的，与读数层同一规矩）。"""
    try:
        return order.index(code)
    except ValueError:
        return len(order)


# ---------------------------------------------------------------- 三张凭据
def measure_latency(predictor: ServicePredictor, *, warmup: int = WARMUP_PROTOCOL,
                    samples: int = SAMPLES_PROTOCOL) -> dict[str, Any]:
    """B1：warm-up `warmup` + 正式 `samples` 条串行，走 P1 的 `measure_speed` 出分位。

    白话：前 20 次问答只当热身（首次碰权重、建内核这些一次性开销不进分布），后 50 次
    一次一问、前后不重叠，每次掐表；把所有读数排成一队取正中间与九成那两个位置，
    均值只当附列。报告里连着设备/位宽/后端一起交，并注明这次量的是 CPU 档。
    """
    records = template_records(warmup + samples)
    report = SP.measure_speed(predictor, records, warmup=warmup, min_samples=samples)
    report.update({
        "protocol": {"warmup": warmup, "samples": samples, "serial": True,
                     "request_template": "1 choice + 1 noul + 1 score（同一 state，一次前向答完）",
                     "fixed_state_sha16": _sha16(FIXED_STATE),
                     "questions": sorted(FIXED_QUESTIONS), "render_version": RENDER_VERSION},
        "device_note": "CPU / torch_eager 档（本波禁 MPS，NPU 后端画像待开卡注记）",
        "readout_column": "readout_ms = 服务端自报 ms（只包前向+读数）；e2e_ms = 客户端掐表（含序列化与本机回环）",
    })
    return report


def _sha16(text: str) -> str:
    """文字指纹前 16 位（固定模板进报告，复跑时能核对是不是同一份题面）。"""
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def parity_probe(engine: Any, url_base: str, records: Sequence[dict[str, Any]], *,
                 temperatures: dict[str, float] | None = None) -> dict[str, Any]:
    """A2：本地组批路 vs 服务 HTTP 路，逐题比 argmax 与份额，交一份可核对的对照表。

    白话：一副权重两条问法。看三件事——胜出候选换没换人（一票否决）、份额差多远
    （容差 `MAX_PROB_DRIFT`）、候选键集对不对得上。任一题换人即整场判负，绝不"平均下来还行"。
    """
    predictor = ServicePredictor(url_base, engine_desc=engine.describe(), tokenizer=engine.tokenizer)
    served = predictor.predict(records)
    local = local_predict(engine, records, temperatures=temperatures)
    by_id = {row.id: row for row in local}
    rows: list[dict[str, Any]] = []
    worst = 0.0
    argmax_ok = True
    for row in served:
        loc = by_id.get(row.id)
        for qid, probs in row.answers.items():
            lprobs = (loc.answers.get(qid) if loc else None) or {}
            keys = sorted(probs)
            if sorted(lprobs) != keys:
                argmax_ok = False
                rows.append({"id": row.id, "qid": qid, "issue": "key_mismatch",
                             "served_keys": keys, "local_keys": sorted(lprobs)})
                continue
            drift = max((abs(float(probs[c]) - float(lprobs[c])) for c in keys), default=0.0)
            worst = max(worst, drift)
            win_s = max(keys, key=lambda c: (float(probs[c]), -_letter_rank(c, keys)))
            win_l = max(keys, key=lambda c: (float(lprobs[c]), -_letter_rank(c, keys)))
            same = win_s == win_l
            argmax_ok = argmax_ok and same
            rows.append({"id": row.id, "qid": qid, "k": len(keys), "argmax_same": same,
                         "served": {c: round(float(probs[c]), 6) for c in keys},
                         "local": {c: round(float(lprobs[c]), 6) for c in keys},
                         "max_abs_drift": round(drift, 8)})
    return {
        "records": len(records), "compared": len(rows), "argmax_all_same": bool(argmax_ok),
        "worst_prob_drift": round(worst, 8), "tolerance": MAX_PROB_DRIFT,
        "passed": bool(argmax_ok and worst <= MAX_PROB_DRIFT),
        "local_semantics": "P1 组批口径（group_by_k + _make_batch 原件），只换编号那一步为对话模板",
        "rows": rows,
    }


def hits_probe(url_base: str, *, questions: Sequence[dict[str, Any]] | None = None,
               state: str = FIXED_STATE) -> dict[str, Any]:
    """B2：同 state 连发多问，从 `/stats` 读命中数增量（复用真发生了没有，看数不看感觉）。

    白话：先把证据段问一次（冷启，必然没命中），再换着题目问第二、第三问——只要证据
    一模一样，小抄就该被接上，命中数因此应当往上跳。这里量的是"命中数增量"而不是
    "感觉变快了"，快多少由 `tokens_fed` 的差额说话。
    """
    questions = list(questions or [dict(spec) for spec in FIXED_QUESTIONS.values()])
    before = http_json(url_base.rstrip("/") + STATS_PATH)
    fed_before = int(before.get("tokens_fed") or 0)
    seen: list[dict[str, Any]] = []
    for i, spec in enumerate(questions):
        payload = {"id": f"p210-hit-{i:02d}", "state": state, "questions": {f"q{i}": spec}}
        reply = http_json(url_base.rstrip("/") + SYSTEMONE_PATH, payload)
        seen.append({"id": payload["id"], "qid": f"q{i}", "answers": reply.get("answers"),
                     "ms": reply.get("ms"), "prefix": (reply.get("prefix") or {}).get("rows")})
    after = http_json(url_base.rstrip("/") + STATS_PATH)
    hits_delta = int(after.get("prefix_hits", 0)) - int(before.get("prefix_hits", 0))
    misses_delta = int(after.get("prefix_misses", 0)) - int(before.get("prefix_misses", 0))
    tokens_delta = int(after.get("tokens_fed", 0)) - fed_before
    # "命中且没为证据段付费"逐题核一遍：命中却仍付了前缀的钱，等于没省，不能算复用成立
    hit_rows = [r for row in seen for r in (row.get("prefix") or []) if r.get("hit")]
    suffix_only = all(r.get("prefix_tokens_fed") == 0 and
                      r.get("suffix_tokens_fed") == r.get("suffix_len") for r in hit_rows)
    return {
        "requests": len(questions), "prefix_hits_delta": hits_delta,
        "prefix_misses_delta": misses_delta, "tokens_fed_delta": tokens_delta,
        "hit_rows": len(hit_rows), "all_hits_suffix_only": bool(suffix_only),
        "runner_after": after.get("runner"), "rows": seen,
    }


# ---------------------------------------------------------------- 取证驱动（一条命令出全三份凭据）
def run_all(*, model: str = "qwen3-0.6b", loader: str = "minimal", device: str = "cpu",
            dtype: str = "float32", cache_dir: str | None = None,
            warmup: int = WARMUP_PROTOCOL, samples: int = SAMPLES_PROTOCOL,
            parity_records: int = 10, prefix: bool = True, threaded: bool = False,
            into_run: pathlib.Path | None = None, new_run_name: str | None = None,
            quiet: bool = False) -> dict[str, Any]:
    """起一次真服务（替身或正式 backbone），把 B1/A2/B2 三张凭据一口气出完并可落 run。

    白话：一条命令做完整个本地半场——载模型、验尾巴、开服务、热身 20 条再串 50 条掐表、
    拿 10 道真题把两条问法对一遍、开缓存连发三问数命中，最后把数字抄进 run 记录本并
    写结论行。中间任何一步判负都照样落盘（负结果也是结果），绝不静默吞掉。
    """
    engine = build_engine(model, loader=loader, device=device, dtype=dtype, cache_dir=cache_dir)
    tail = verify_think_tail(engine)
    run = None
    if new_run_name:
        from sys1.runs import new_run as _nr

        run = _nr(new_run_name, {
            "prefix_domain": "p2-10", "engine": engine.describe(), "think_off_suffix_ok": tail == THINK_OFF_SUFFIX,
            "warmup": warmup, "samples": samples, "parity_records": parity_records,
            "render_version": RENDER_VERSION, "letters": len(LETTERS),
        })
    out_dir = pathlib.Path(into_run) if into_run is not None else (run.path if run else None)

    # ① B1：默认无缓存基线（一次前向答三问的串行服务耗时）
    plain = DecisionService(engine)
    srv_plain = spawn(plain, threaded=threaded)
    try:
        predictor = ServicePredictor(srv_plain.base_url, engine_desc=engine.describe(),
                                     tokenizer=engine.tokenizer)
        latency = measure_latency(predictor, warmup=warmup, samples=samples)
        # ② A2：同一副权重，本地组批路 vs 服务 HTTP 路
        recs = load_eval_records(parity_records)
        parity = parity_probe(engine, srv_plain.base_url, recs) if recs else {
            "records": 0, "passed": False, "note": "评测集没取到真题，A2 不出凭据（不造题）"}
    finally:
        srv_plain.stop()

    # ③ B2：开 p2-07 前缀复用后连发三问数命中
    hits: dict[str, Any] = {"skipped": "prefix 未开"}
    if prefix:
        cached = DecisionService(engine, prefix=True, verify="once")
        srv_cached = spawn(cached, threaded=threaded)
        try:
            hits = hits_probe(srv_cached.base_url)
            hits["engine_uncacheable_note"] = (
                "混合记忆层（linear attention）的 past 拷不动时，runner 记 uncacheable 并整段重算，"
                "命中数会掉到 0——这是 p2-07 已登记的降级路径，不是静默错位")
        finally:
            srv_cached.stop()

    result = {"latency": latency, "parity": parity, "hits": hits,
              "engine": engine.describe(), "letters": len(LETTERS)}
    if out_dir is not None:
        _write_artifacts(out_dir, result)
        if run is not None:
            _record_into_run(run, result)
    if not quiet:
        print(json.dumps(_printable(result), ensure_ascii=False, indent=2))
    return result


def _printable(result: dict[str, Any]) -> dict[str, Any]:
    """打印用的窄表（逐题对照表太长，不进 stdout，只在 json 里）。"""
    slim = json.loads(json.dumps(result, default=str))
    rows = slim.get("parity", {}).get("rows")
    if isinstance(rows, list) and len(rows) > 6:
        slim["parity"]["rows"] = rows[:6] + [f"...({len(rows)} rows total)"]
    return slim


def _write_artifacts(out_dir: pathlib.Path, result: dict[str, Any]) -> None:
    """三张凭据分别落 json（延迟/一致/命中），路径固定在 `artifacts/`，复跑覆盖同名。"""
    art = out_dir / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    for name, key in (("latency.json", "latency"), ("parity.json", "parity"), ("hits.json", "hits")):
        (art / name).write_text(json.dumps(result.get(key), ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    (art / "engine.json").write_text(json.dumps(result.get("engine"), ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")


def _record_into_run(run: Any, result: dict[str, Any]) -> None:
    """把数字抄进 run 记录本：曲线一行、notes 三行（假设/观察/结论），最后 finish 验结论。"""
    from sys1.eval.run import fill_notes_skeleton

    lat, par, hit = result["latency"], result["parity"], result["hits"]
    e2e, read = lat.get("e2e_ms", {}), lat.get("readout_ms", {})
    run.log_metrics(0, e2e_p50_ms=e2e.get("p50"), e2e_p95_ms=e2e.get("p95"),
                    e2e_mean_ms=e2e.get("mean"), e2e_p99_ms=e2e.get("p99"),
                    readout_p50_ms=read.get("p50"), readout_p95_ms=read.get("p95"),
                    readout_mean_ms=read.get("mean"),
                    parity_passed=1 if par.get("passed") else 0,
                    parity_worst_drift=par.get("worst_prob_drift"),
                    prefix_hits_delta=hit.get("prefix_hits_delta"),
                    prefix_tokens_fed_delta=hit.get("tokens_fed_delta"))
    fill_notes_skeleton(run, hypothesis=(
        "p2-10 本地半场：/v1/systemone 用 decision 冻结契约驱动替身 backbone，"
        "固定模板（1 choice+1 noul+1 score）一次前向答完；CPU 档 warm-up20+N50 应能出 "
        "mean/P50/P95 三统计，endpoint 路与本地组批路 argmax 不换人，开 p2-07 前缀复用后"
        "同 state 连发三问命中增量 ≥2。"), observation=[
        f"- B1 延迟（device={lat.get('device')} dtype={lat.get('dtype')} backend={lat.get('backend')}，"
        f"serial n={lat.get('n')} warmup-excluded={lat.get('warmup_excluded')}）："
        f"e2e_ms mean={e2e.get('mean')} P50={e2e.get('p50')} P95={e2e.get('p95')} "
        f"P99={e2e.get('p99')} min={e2e.get('min')} max={e2e.get('max')}",
        f"- B1 读数段单列 readout_ms（服务端自报，只包前向+读数）："
        f"mean={read.get('mean')} P50={read.get('p50')} P95={read.get('p95')}",
        f"- B1 吞吐：tokens_in={lat.get('tokens_in')} tok/s(aggregate)={lat.get('throughput_tps_aggregate')}"
        f" output_tokens={lat.get('output_tokens')}",
        f"- A2 双路一致：records={par.get('records')} compared={par.get('compared')} "
        f"argmax_all_same={par.get('argmax_all_same')} worst_prob_drift={par.get('worst_prob_drift')} "
        f"tolerance={par.get('tolerance')} passed={par.get('passed')}",
        f"- B2 前缀命中：requests={hit.get('requests')} hits_delta={hit.get('prefix_hits_delta')} "
        f"misses_delta={hit.get('prefix_misses_delta')} tokens_fed_delta={hit.get('tokens_fed_delta')} "
        f"all_hits_suffix_only={hit.get('all_hits_suffix_only')}",
        f"- 引擎：{result.get('engine', {}).get('engine')} loader={result.get('engine', {}).get('loader')} "
        f"head={result.get('engine', {}).get('head_shape')} render_version={RENDER_VERSION}",
    ])
    verdicts = []
    verdicts.append(f"B1 三统计出数 P50={e2e.get('p50')}ms/P95={e2e.get('p95')}ms/mean={e2e.get('mean')}ms")
    verdicts.append("A2 " + ("双路 argmax 不换人且漂移在容差内" if par.get("passed")
                             else f"双路判负（worst drift={par.get('worst_prob_drift')}）"))
    if hit.get("requests"):
        verdicts.append(f"B2 连发 {hit.get('requests')} 问命中增量 {hit.get('prefix_hits_delta')}"
                        f"（喂入符号 {hit.get('tokens_fed_delta')}）")
    elif hit.get("skipped"):
        verdicts.append(f"B2 未开缓存：{hit.get('skipped')}")
    run.conclude("；".join(verdicts) + f"。CPU 档口径，NPU/MPS 画像延后（{lat.get('device_note')}）。")
    run.finish()


def main(argv: Sequence[str] | None = None) -> int:
    """命令行：`python -m serving.latency --model qwen3-0.6b --loader minimal --new-run p210-serving-local-half`。"""
    ap = argparse.ArgumentParser(description="p2-10 本地半场取证（B1 延迟协议 / A2 双路一致 / B2 命中计数）")
    ap.add_argument("--model", default="qwen3-0.6b")
    ap.add_argument("--loader", default="minimal", choices=["seam", "minimal"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--warmup", type=int, default=WARMUP_PROTOCOL)
    ap.add_argument("--samples", type=int, default=SAMPLES_PROTOCOL)
    ap.add_argument("--parity-records", type=int, default=10)
    ap.add_argument("--no-prefix", action="store_true", help="不做 B2（不挂前缀复用）")
    ap.add_argument("--threaded", action="store_true", help="并发服务（默认单线程串行）")
    ap.add_argument("--into-run", default=None, help="把 artifacts 写进已有 run 目录")
    ap.add_argument("--new-run", default=None, help="用 sys1.runs 开一个新 run 并落数字（推荐 p210-*）")
    args = ap.parse_args(argv)
    result = run_all(
        model=args.model, loader=args.loader, device=args.device, dtype=args.dtype,
        cache_dir=args.cache_dir, warmup=args.warmup, samples=args.samples,
        parity_records=args.parity_records, prefix=not args.no_prefix, threaded=args.threaded,
        into_run=pathlib.Path(args.into_run) if args.into_run else None,
        new_run_name=args.new_run)
    ok = bool(result["parity"].get("passed"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
