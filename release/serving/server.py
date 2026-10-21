"""serving/server.py — `/v1/systemone`：把 backbone 接成一台能被问的机器（p2-10 A1/A2/B2）。

【做什么】
    一个只认 P1 冻结决策契约的 HTTP 服务。收 `{id, state, questions:{qid:题目}}`，
    回 `{id, answers:{qid:{候选代号: 份额}}, ms, ...}`；外加 `/v1/systemone/batch`（跨请求
    合批）、`/health`（活着与引擎画像）、`/stats`（记账：请求数/前向数/喂进去的符号数/
    前缀缓存命中数）。协议字段与 `sys1/eval/predict.py::EndpointPredictor` 逐键对齐——
    评测侧 `--endpoint` 打过来就能直接出预测行，不需要本服务再配一个专用客户端。

【怎么做】
    三层各管一段，串起来是一条链：
    ① 引擎层 `TorchEngine`：把两种装载口归一成一个形状——`loader="seam"`（p2-01 正式门，
       会跑四道接缝体检）与 `loader="minimal"`（p2-05 为 0.6B 替身留的侧门）。服务只认
       四个成员：`forward_hidden(ids, mask)→(B,T,d)`、`head_weight (vocab,d)`、`letter_ids`
       (26 个)、`pad_id`，加上 `tokenizer` 与 `kv_engine()`。换 backbone 只换这里。
    ② 决策层 `DecisionService`：题目走 `from_systemone`→`render`→对话模板（关闭思考）→
       编号，一次前向把同一请求里的多道题合批（右补空洞，读点按各行真长度取），
       分数走 `option_scores`，倍数走 `temperatures_for/apply_temperature`，份额走 `to_probs`。
       全部是 `sys1/decision/` 的原函数，本文件不重写任何读数口径（decision 零改动的物理保证）。
       开 `--prefix` 时改走 p2-07 的 `PrefixRunner`：按 state 前缀指纹查缓存，命中就只喂
       问题段，护栏（相对漂移 + argmax 换人一票否决）判负即拉黑并整段重算。
    ③ HTTP 层：标准库 `http.server`（零新依赖），默认单线程串行——前向天然不并发，
       延迟口径因此稳定（并批摊薄是耗时轴的头号假账）。

【为什么】
    被否方案一：服务里自己写一遍"末位×字母行→softmax"——读数口径从此有两份，P1 与 P2
    的分数迟早漂移，而父 design D1 要的正是"换脑不换程序"。
    被否方案二：一次前向只服务一道题（按 qid 逐条问）——同一份证据被反复重算，白付三倍
    前向；本服务把"一个请求里的多道题"合进一次前向，读点各行自取，spec 的"三问一次前向"
    就是这么落的。
    被否方案三：上 fastapi/uvicorn——本波只要求协议成立，多两个依赖就多两份装不上
    （禁 pip 的现场纪律），标准库够且更薄。
    被否方案四：线程池并发服务——吞吐好看但 P50/P95 变成"排队+并发"的混合量，跨机
    对比失去意义；并发属服务化之后的事，本波先钉口径。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer, HTTPServer
from typing import Any, Iterable, Sequence

import torch

from sys1.decision import (
    LETTERS,
    RENDER_VERSION,
    THINK_OFF_SUFFIX,
    apply_temperature,
    from_systemone,
    option_scores,
    render,
    temperatures_for,
    to_probs,
)

#: 端点名字与 sys1/eval/predict.py 的单一事实来源保持一致（协议改动必须同步两处）
SYSTEMONE_PATH = "/v1/systemone"
BATCH_PATH = "/v1/systemone/batch"
HEALTH_PATH = "/health"
STATS_PATH = "/stats"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8931
#: 一次合批最多并几行（超出就分刀，避免单批把内存顶穿）
MAX_ROWS_PER_FORWARD = 64


class ServiceConfigError(RuntimeError):
    """服务装配期被拒：模板交不出"关闭思考"的尾巴，或引擎成员不齐。"""


class BadRequestError(ValueError):
    """请求不合协议（缺 questions、题目形态不认识）：HTTP 层折成 400，绝不当 500 掩盖。"""


# ---------------------------------------------------------------- 引擎层（两种装载口归一）
@dataclass
class TorchEngine:
    """服务对"模型"的全部所知：一组编号进、逐位置数字出，外加读点要的两件权重。

    白话：不管这副身板是从正式门（连带四道接缝体检）进来的，还是替身侧门进来的，
    到了服务这儿都只看四样——会算逐位置数字、有那张对照表、知道二十六个字母的号、
    知道补空洞用哪个号。门不一样只影响"进货检查"，不影响"怎么用"。
    """

    name: str
    body: Any
    tokenizer: Any
    head_weight: torch.Tensor
    letter_ids: tuple[int, ...]
    pad_id: int
    dtype: torch.dtype
    device: str
    loader: str
    model_ref: Any = None                              # 原始 Student（诊断用，不进决策链）
    provenance: dict[str, Any] = field(default_factory=dict)

    def forward_hidden(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """`(B,T)` 编号 + 可见位 → `(B,T,d)` 逐位置数字（不生成、不带 KV、不回传任何缓存）。

        白话：一排排字送进去，只收"每个位置读完留下的那串数字"；批次里被垫了空洞的
        行照样收，读点由调用方按每行真长度去取。
        """
        with torch.no_grad():
            out = self.body(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
        return out.last_hidden_state

    def kv_engine(self) -> Any:
        """交出可续 KV 的引擎（p2-07 `PrefixRunner` 要的那张接口），默认带 cache。"""
        from serving.prefix_cache import HFEngine

        return HFEngine(self.body)

    def describe(self) -> dict[str, Any]:
        """入口摘要：设备/位宽/后端/装载门/渲染版号——耗时报告必须连着这三项才有意义。"""
        from sys1.kernels import backends

        return {
            "engine": self.name,
            "loader": self.loader,
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "backend": backends.active_backend(self.device),
            "head_shape": [int(self.head_weight.shape[0]), int(self.head_weight.shape[1])],
            "pad_id": int(self.pad_id),
            "render_version": RENDER_VERSION,
            "provenance": dict(self.provenance),
        }


def build_engine(
    model: str = "qwen3-0.6b",
    *,
    loader: str = "minimal",
    device: str = "cpu",
    dtype: str = "float32",
    cache_dir: str | None = None,
) -> TorchEngine:
    """装载 backbone 成服务引擎：`seam` 走 p2-01 正式门，`minimal` 走 p2-05 替身侧门。

    白话：进货两条路——正式门会连着把"字母独占一格、模板尾巴是'不想了'那截、贴图记号
    不跟字母抢号"一起查一遍（0.8B 走这条）；侧门只逐枚验字母，给没有看图本钱的 0.6B
    替身留的权宜。两条路都从同一个快照仓取东西，出来交给服务的形状完全一样。
    """
    from production.sft import build_student

    want = getattr(torch, str(dtype)) if isinstance(dtype, str) else dtype
    student = build_student(
        model, loader=loader, device=device, dtype=want,
        cache_dir=None if cache_dir is None else pathlib.Path(cache_dir),
    )
    return TorchEngine(
        name=student.name, body=student.body, tokenizer=student.tokenizer,
        head_weight=student.head_weight, letter_ids=tuple(int(v) for v in student.letter_ids),
        pad_id=int(student.pad_id), dtype=student.dtype, device=str(device), loader=student.loader,
        model_ref=student, provenance=dict(getattr(student, "provenance", {}) or {}),
    )


# ---------------------------------------------------------------- 渲染 → 编号（决策链第一步）
def _template_text(tokenizer: Any, messages: Sequence[dict[str, str]]) -> str:
    """套对话模板取整段文字，口径 = `add_generation_prompt=True` + 关闭思考。

    白话：把排好的信交给模板套上壳，并明确要求"不想了"——读点必须落在候选字母该写的
    那一格上。老模板不认识这个开关就把开关摘掉再套一次，但尾巴对不对由服务装配那一步
    统一把关（见 `verify_think_tail`），不在这里偷偷放行。
    """
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _encode_ids(tokenizer: Any, text: str) -> list[int]:
    """按"不加任何额外特殊符"的口径编成编号列（特殊符归模板管，别编两遍）。"""
    out = tokenizer.encode(text, add_special_tokens=False)
    if isinstance(out, dict):
        return [int(t) for t in out["input_ids"]]
    if hasattr(out, "ids"):
        return [int(t) for t in out.ids]
    return [int(t) for t in out]


@dataclass(frozen=True)
class PreparedRow:
    """一题从"收到"到"可送前向"的中间态：编号、字母对应的候选代号、渲染结果。"""

    qid: str
    qtype: str
    order: tuple[str, ...]
    ids: tuple[int, ...]
    messages: tuple[dict[str, str], ...]

    @property
    def k(self) -> int:
        return len(self.order)


def prepare_request(tokenizer: Any, payload: dict[str, Any]) -> list[PreparedRow]:
    """把一个 `/v1/systemone` 请求体摊成若干可前向的行（一问一请求与一问多请求都走这里）。

    白话：前台收来的表格先逐题抄成同一张单子（`from_systemone`），再排成信（`render`）、
    套壳、编号；抄不进单子的（缺候选、类型不认识）当场退回并说清是哪道题，绝不"看着
    能算"就硬送——错位的读点比报错危险得多。
    """
    state = payload.get("state")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise BadRequestError("请求体必须有非空的 questions（{题目号: {type, instructions, criteria}}）")
    rows: list[PreparedRow] = []
    for qid, spec in questions.items():
        row = from_systemone(state if state is not None else "", dict(spec or {}), qid=str(qid))
        messages, order = render(row)
        ids = _encode_ids(tokenizer, _template_text(tokenizer, messages))
        if not ids:
            raise BadRequestError(f"题目 {qid} 编完是空的，没有可读的位置")
        rows.append(PreparedRow(
            qid=str(qid), qtype=str(row.get("type", spec.get("type", "choice"))),
            order=tuple(order), ids=tuple(ids), messages=tuple(dict(m) for m in messages),
        ))
    return rows


def verify_think_tail(engine: TorchEngine) -> str:
    """装配期体检：模板交出的尾巴必须是 decision 钉死的"思考关闭"记号，否则拒载。

    白话：整条读出全押在"末尾那一格正好是要写候选字母的位置"上。尾巴若不是那个记号，
    说明思考没真关掉，读点就落在思维链中途——这时候出数的份额全是废数，所以宁可不开门。
    """
    probe = {"type": "choice", "instructions": "tail probe", "criteria": {"a": "one", "b": "two"}}
    row = from_systemone("probe state", probe, qid="q")
    messages, _ = render(row)
    text = _template_text(engine.tokenizer, messages)
    if not text.endswith(THINK_OFF_SUFFIX):
        got = text[-len(THINK_OFF_SUFFIX) - 8:]
        raise ServiceConfigError(
            f"模板尾部是 {got!r}，不是 decision 钉死的 {THINK_OFF_SUFFIX!r}：读点会落在思维链中途，拒开服务"
        )
    return THINK_OFF_SUFFIX


# ---------------------------------------------------------------- 决策层（协议无关，可进程内直调）
class DecisionService:
    """把引擎 + decision 契约包成"能回答一个请求"的对象；HTTP 层只是它的一层皮。

    白话：所有答题动作都在这里，网络那层只负责收发字节。这样单测能绕过网络直接问，
    评测脚本也能在不开服务时拿同一个对象出数，两条路口径必然一致。
    """

    def __init__(
        self,
        engine: Any,
        *,
        temperatures: dict[str, float] | None = None,
        prefix: bool = False,
        verify: str = "once",
        namespace: str = "",
        drift_ratio: float | None = None,
        max_prefix_bytes: int | None = None,
    ) -> None:
        """`prefix=True` 挂上 p2-07 前缀复用；`temperatures` 是 `qtype→倍数` 表（None=全未标定）。"""
        self.engine = engine
        self.temperatures = dict(temperatures) if temperatures else None
        self.namespace = namespace
        self.prefix = bool(prefix)
        self.runner: Any = None
        self.started = time.time()
        self.counts: dict[str, int] = {
            "requests": 0, "rows": 0, "forwards": 0, "tokens_fed": 0,
            "prefix_hits": 0, "prefix_misses": 0, "prefix_recomputes": 0,
        }
        self._lock = threading.Lock()          # 前向串行用的那把锁（并发时也不让两批同时进模型）
        if self.prefix:
            self.runner = self._build_runner(verify=verify, drift_ratio=drift_ratio,
                                             max_prefix_bytes=max_prefix_bytes)

    # -- 缓存挂点 ---------------------------------------------------------------
    def _build_runner(self, *, verify: str, drift_ratio: float | None, max_prefix_bytes: int | None) -> Any:
        """按 p2-07 的件拼出 `PrefixRunner`（缓存/计数器/护栏都是原件，本域不重写）。"""
        from serving.prefix_cache import DEFAULT_DRIFT_RATIO, PrefixCache, PrefixRunner, TokenCounter

        cache = PrefixCache() if max_prefix_bytes is None else PrefixCache(max_bytes=int(max_prefix_bytes))
        runner_kwargs: dict[str, Any] = {
            "cache": cache,
            "counter": TokenCounter(),
            "verify": verify,
            "readout": self._score_last,
        }
        if drift_ratio is not None:
            runner_kwargs["drift_ratio_threshold"] = float(drift_ratio)
        _ = DEFAULT_DRIFT_RATIO  # 阈值缺省由 prefix_cache 自己定，这里只在显式给值时覆盖
        return PrefixRunner(self.engine.kv_engine(), **runner_kwargs)

    def _score_last(self, hidden_last: torch.Tensor, k: int | None) -> torch.Tensor:
        """`(1,d)` 末位数字 → `(1,k)` 候选分（`option_scores` 只吃三维，故补一维 T）。"""
        head = self._head_on(hidden_last.device)
        n = int(k) if k is not None else int(head.shape[0])
        n = max(1, min(n, len(self.engine.letter_ids)))
        ids = list(self.engine.letter_ids[:n])
        return option_scores(hidden_last.unsqueeze(1), head, ids, [1])

    def _head_on(self, device: Any) -> torch.Tensor:
        """把输出层权重挪到数字所在设备（同设备时零开销；MPS/NPU 档留给后端切换）。"""
        head = self.engine.head_weight
        return head if head.device == torch.device(device) else head.to(device)

    # -- 一次前向的两种走法 ------------------------------------------------------
    def _read_batched(self, rows: Sequence[PreparedRow]) -> tuple[list[torch.Tensor], int, list[dict[str, Any]]]:
        """同一请求的多行并排右补空洞，**一次前向**收全部候选分（spec 的"三问一次前向"）。

        白话：几道题的证据是同一段，只有末尾"问什么、有哪几个候选"不同，所以能并排坐
        进一批。空洞补在尾巴之后、每行按自己的真长度取读点，于是垫没垫都不该改变答案；
        这个"不该改变"在 p2-10 的探针里量过（右补空洞相对漂移 6.3e-07、argmax 不换人）。
        """
        scores: list[torch.Tensor] = []
        head = self._head_on(torch.device("cpu"))
        fed = 0
        for start in range(0, len(rows), MAX_ROWS_PER_FORWARD):
            chunk = list(rows[start:start + MAX_ROWS_PER_FORWARD])
            width = max(len(r.ids) for r in chunk)
            ids = torch.full((len(chunk), width), int(self.engine.pad_id), dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for i, r in enumerate(chunk):
                ids[i, : len(r.ids)] = torch.tensor(list(r.ids), dtype=torch.long)
                mask[i, : len(r.ids)] = 1
            hidden = self.engine.forward_hidden(ids, mask)
            head = self._head_on(hidden.device)
            with torch.no_grad():
                for i, r in enumerate(chunk):
                    scores.append(option_scores(
                        hidden[i:i + 1], head, list(self.engine.letter_ids[:r.k]), [len(r.ids)]
                    ))
            fed += sum(len(r.ids) for r in chunk)
            with self._lock:
                self.counts["forwards"] += 1
        per_row = [{"tokens": len(r.ids), "k": r.k} for r in rows]
        return scores, fed, per_row

    def _read_prefixed(self, rows: Sequence[PreparedRow]) -> tuple[list[torch.Tensor], int, list[dict[str, Any]]]:
        """逐行走 p2-07 复用路：state 前缀命中就只喂问题段（护栏在 runner 里）。"""
        scores: list[torch.Tensor] = []
        meta: list[dict[str, Any]] = []
        fed = 0
        for r in rows:
            res = self.runner.ask_row(self.engine.tokenizer, list(r.messages),
                                      namespace=self.namespace, k=r.k)
            fed += int(res.tokens_fed)
            with self._lock:
                self.counts["forwards"] += 1
                if res.hit:
                    self.counts["prefix_hits"] += 1
                else:
                    self.counts["prefix_misses"] += 1
                if res.recomputed:
                    self.counts["prefix_recomputes"] += 1
            scores.append(self._score_last(res.hidden_last, r.k))
            meta.append({"tokens": len(r.ids), "k": r.k, **res.as_dict()})
        return scores, fed, meta

    # -- 对外主入口 --------------------------------------------------------------
    def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        """回答一个请求：一次前向（或一趟复用）出全部候选份额，回 `/v1/systemone` 回包。

        白话：把这道请求里的每题都变成"各候选几成把握"，再告诉外面这趟花了多少毫秒。
        `ms` 只包前向+读数那一段，网络与序列化不算在里面——评测侧把 `ms` 当"读数耗时"，
        自己掐的表当端到端，两列并排摆，包装成本占多少一眼看得见。
        """
        rows = prepare_request(self.engine.tokenizer, payload)
        return self._answer(rows, req_id=payload.get("id"))

    def _answer(self, rows: Sequence[PreparedRow], *, req_id: Any, batch_ms: float | None = None,
                batch_rows: int | None = None) -> dict[str, Any]:
        """共用出口：走一次前向/复用，把分数折成份额并按契约拼回包。"""
        t0 = time.perf_counter()
        if self.runner is not None:
            scores, fed, per_row = self._read_prefixed(rows)
        else:
            scores, fed, per_row = self._read_batched(rows)
        with torch.no_grad():
            probs_list = [to_probs(apply_temperature(s, r.qtype, self.temperatures), 1.0)
                          for s, r in zip(scores, rows)]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        uncal = [not c for c in temperatures_for([r.qtype for r in rows], self.temperatures, len(rows))[1]]
        answers: dict[str, dict[str, float]] = {}
        for r, probs in zip(rows, probs_list):
            answers[r.qid] = {code: float(p) for code, p in zip(r.order, probs[0].tolist())}
        if batch_ms is not None:                       # 合批时 ms 是"本请求在该批里摊到的毫秒"
            share = (batch_rows or len(rows))
            reported = round(batch_ms / max(1, share), 3)
            ms_semantics = "amortized_within_batch"
        else:
            reported = round(elapsed_ms, 3)
            ms_semantics = "forward_plus_readout_of_this_request"
        with self._lock:
            self.counts["requests"] += 1
            self.counts["rows"] += len(rows)
            self.counts["tokens_fed"] += int(fed)
        reply: dict[str, Any] = {
            "id": req_id, "answers": answers, "ms": reported, "ms_semantics": ms_semantics,
            "render_version": RENDER_VERSION, "engine": self.engine.describe(),
        }
        if batch_ms is not None:
            reply["ms_batch"] = round(batch_ms, 3)
        if any(uncal):
            reply["uncalibrated_qids"] = [r.qid for r, u in zip(rows, uncal) if u]
        if self.runner is not None:
            reply["prefix"] = {"enabled": True, "rows": per_row}
        return reply

    def decide_batch(self, payloads: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """跨请求合批：把多份请求的题目并进同一次前向，再按请求把答案拆回去（P1 组批思路）。

        白话：五个人各问一道题，证据若是同一段，就让五道题并排坐一批走一遍模型。
        每份回包里如实写着"这批一共花了多少毫秒"与"摊到本请求头上是多少"，
        免得摊薄后的数字被当成单次请求耗时去和串行口径比。
        """
        payloads = list(payloads)
        if not payloads:
            return []
        prepared = [(p.get("id"), prepare_request(self.engine.tokenizer, p)) for p in payloads]
        flat = [r for _p, rs in prepared for r in rs]
        t0 = time.perf_counter()
        if self.runner is not None:
            scores, fed, per_row = self._read_prefixed(flat)
        else:
            scores, fed, per_row = self._read_batched(flat)
        batch_ms = (time.perf_counter() - t0) * 1000.0
        with torch.no_grad():
            probs = [to_probs(apply_temperature(s, r.qtype, self.temperatures), 1.0)
                     for s, r in zip(scores, flat)]
        by_qid = {r.qid: (r, p) for r, p in zip(flat, probs)}
        out: list[dict[str, Any]] = []
        cursor = 0
        for req_id, rs in prepared:
            answers = {}
            for r in rs:
                _rr, pv = by_qid[r.qid]
                answers[r.qid] = {code: float(v) for code, v in zip(r.order, pv[0].tolist())}
            out.append({
                "id": req_id, "answers": answers,
                "ms": round(batch_ms / max(1, len(flat)) * len(rs), 3),
                "ms_batch": round(batch_ms, 3),
                "ms_semantics": "amortized_within_batch",
                "render_version": RENDER_VERSION,
                "engine": self.engine.describe(),
            })
            cursor += len(rs)
        _ = cursor
        with self._lock:                               # 记账：一次合批算一次"服务了一个请求组"
            self.counts["requests"] += len(prepared)
            self.counts["rows"] += len(flat)
            self.counts["tokens_fed"] += int(fed)
        return out

    def health(self) -> dict[str, Any]:
        """存活探针：把引擎画像与渲染版号原样交出（起没起来、用的是哪副身板，一眼看清）。"""
        return {
            "status": "ok", "render_version": RENDER_VERSION,
            "prefix": self.prefix, "uptime_s": round(time.time() - self.started, 3),
            "engine": self.engine.describe(),
        }

    def stats(self) -> dict[str, Any]:
        """记账本：请求/行/前向/喂进去的符号数 + 前缀命中数（B2 的验收面）。"""
        with self._lock:
            snapshot = dict(self.counts)
        out: dict[str, Any] = {**snapshot, "prefix_enabled": self.prefix,
                              "seconds_since_start": round(time.time() - self.started, 3),
                              "render_version": RENDER_VERSION}
        if self.runner is not None:
            runner_stats = self.runner.stats()
            out["runner"] = runner_stats
            cache = runner_stats.get("cache") or {}
            out["cache_bytes"] = cache.get("bytes_used")
            out["cache_items"] = cache.get("items")
        return out


# ---------------------------------------------------------------- HTTP 层（标准库，单线程串行）
def make_handler(service: DecisionService) -> type[BaseHTTPRequestHandler]:
    """造一个绑定了服务的请求处理类（闭包传依赖，省得全局变量裸奔）。"""

    class SystemOneHandler(BaseHTTPRequestHandler):
        """三条路：POST 答问（单请求/批量）、GET 探针（health/stats）、其余 404。"""

        protocol_version = "HTTP/1.1"                 # 配 Content-Length，连接能复用
        server_version = "sys1-systemone/0.1"
        timeout = 120.0                               # 单次前向在 CPU 上可能上百毫秒，别掐太急

        def _send(self, code: int, obj: dict[str, Any]) -> None:
            """回一个 json 包（显式 Content-Length，HTTP/1.1 才知道这一条到哪儿结束）。"""
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            """读请求体 json；空体/坏 json 一律 `BadRequestError`（400），不猜也不 500。"""
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            if not raw:
                raise BadRequestError("请求体为空")
            try:
                data = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise BadRequestError(f"请求体不是合法 json：{exc}") from exc
            if not isinstance(data, dict):
                raise BadRequestError("请求体必须是一个 json 对象")
            return data

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 的命名约定
            if self.path == HEALTH_PATH:
                self._send(200, service.health())
            elif self.path == STATS_PATH:
                self._send(200, service.stats())
            else:
                self._send(404, {"error": "not_found", "paths": [
                    SYSTEMONE_PATH, BATCH_PATH, HEALTH_PATH, STATS_PATH]})

        def do_POST(self) -> None:  # noqa: N802 - 同上
            try:
                payload = self._read_json()
            except BadRequestError as exc:
                self._send(400, {"error": "bad_request", "detail": str(exc)})
                return
            try:
                if self.path == SYSTEMONE_PATH:
                    self._send(200, service.decide(payload))
                elif self.path == BATCH_PATH:
                    reqs = payload.get("requests") if isinstance(payload.get("requests"), list) else None
                    if reqs is None and isinstance(payload.get("questions"), dict):
                        reqs = [payload]            # 单请求误投批量端点：照单答，不装作没看见
                    if not reqs:
                        raise BadRequestError("批量端点需要 requests: [{id, state, questions}, ...]")
                    self._send(200, {"results": service.decide_batch(reqs)})
                else:
                    self._send(404, {"error": "not_found", "paths": [SYSTEMONE_PATH, BATCH_PATH]})
            except BadRequestError as exc:
                self._send(400, {"error": "bad_request", "detail": str(exc)})
            except Exception as exc:  # noqa: BLE001  服务端异常必须回结构体，别让半截 500 页骗到人
                self._send(500, {"error": "internal", "detail": f"{type(exc).__name__}: {exc}"})

        def log_message(self, fmt: str, *args: Any) -> None:
            """静音默认 stderr 访问日志：延迟测量里打印本身就是噪声源，计数已在服务里记着。"""

    return SystemOneHandler


def make_server(service: DecisionService, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                threaded: bool = False) -> HTTPServer:
    """绑端口造服务器：默认单线程串行（前向天然不并发，延迟口径稳），`threaded` 才并发。"""
    cls = ThreadingHTTPServer if threaded else HTTPServer
    return cls((host, port), make_handler(service))


@dataclass
class RunningServer:
    """后台线程里跑着的服务句柄：`base_url` 给客户端，`stop()` 收尾（单测与取证脚本用）。"""

    service: DecisionService
    httpd: HTTPServer
    thread: threading.Thread
    host: str
    port: int

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def endpoint(self) -> str:
        return self.base_url + SYSTEMONE_PATH

    def stats(self) -> dict[str, Any]:
        """直读服务对象的账（不经 HTTP，省得多算一次前向）。"""
        return self.service.stats()

    def stop(self, timeout: float = 10.0) -> None:
        """关监听再 join 线程：不让端口在下一用例开始前还被人连着。"""
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=timeout)


def spawn(service: DecisionService, *, host: str = DEFAULT_HOST, port: int = 0,
          threaded: bool = False) -> RunningServer:
    """在后台线程起服务；`port=0` 让内核挑端口（并行跑单测不打架）。"""
    httpd = make_server(service, host=host, port=port, threaded=threaded)
    bound_host, bound_port = httpd.server_address[0], int(httpd.server_address[1])
    thread = threading.Thread(target=httpd.serve_forever, name="sys1-systemone", daemon=True)
    thread.start()
    return RunningServer(service=service, httpd=httpd, thread=thread,
                         host=bound_host if bound_host != "0.0.0.0" else host, port=bound_port)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行起服务：`python -m serving.server --model qwen3-0.6b --loader minimal --port 8931`。"""
    ap = argparse.ArgumentParser(description="POST /v1/systemone —— P2 backbone 决策服务（本地半场）")
    ap.add_argument("--model", default="qwen3-0.6b", help="backbone 名（production.assets 注册表认得的那个）")
    ap.add_argument("--loader", default="minimal", choices=["seam", "minimal"],
                    help="seam=p2-01 正式门（0.8B 走这条）；minimal=0.6B 替身侧门")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--prefix", action="store_true", help="挂 p2-07 前缀复用（state 段只算一次）")
    ap.add_argument("--verify", default="once", choices=["off", "once", "always"],
                    help="复用路护栏何时比对整段重算")
    ap.add_argument("--namespace", default="", help="缓存命名空间（多模型/多租户隔离）")
    ap.add_argument("--temperatures", default=None, help="qtype→倍数 的 json 表路径（缺省全按未标定出数）")
    ap.add_argument("--threaded", action="store_true", help="并发服务（默认单线程串行，口径稳）")
    args = ap.parse_args(argv)

    table = None
    if args.temperatures:
        table = json.loads(pathlib.Path(args.temperatures).read_text(encoding="utf-8"))
    engine = build_engine(args.model, loader=args.loader, device=args.device, dtype=args.dtype,
                          cache_dir=args.cache_dir)
    tail = verify_think_tail(engine)
    service = DecisionService(engine, temperatures=table, prefix=args.prefix, verify=args.verify,
                              namespace=args.namespace)
    httpd = make_server(service, host=args.host, port=args.port, threaded=args.threaded)
    port = int(httpd.server_address[1])
    print(json.dumps({
        "listening": f"http://{args.host}:{port}", "endpoint": SYSTEMONE_PATH,
        "engine": engine.describe(), "think_off_suffix_ok": tail == THINK_OFF_SUFFIX,
        "prefix": service.prefix, "letters": len(LETTERS),
    }, ensure_ascii=False, indent=2), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
