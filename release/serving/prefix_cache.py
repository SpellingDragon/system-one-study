"""serving/prefix_cache.py — 同 state 多问 / 跨请求的 KV 前缀复用（p2-07 B1a+B1b+B1c+B2）。

【做什么】
    一句话："同一份证据（state）反复问"这件事，只该为证据算一次前向。三件事：
    ① B1a 缓存核心：把"state 渲染前缀的 token 列"散列成键，键 → 那份 past_key_values，
       带 LRU + 字节预算淘汰；② B1b 增量前向：命中后只喂问题段，并用**符号计数器**把
       "这次真喂进去几个 token"记成硬数字；③ B1c 一致性护栏：复用路与整段重算路比对，
       argmax 必须同人、相对漂移超阈值就把这个 state 拉黑名单、后续一律重算。

【怎么做】
    键 = sha256(`RENDER_VERSION` ‖ namespace ‖ 前缀长度 ‖ 前缀 token 列)。版号进键是刻意的：
    渲染层改版（多一个空格都算）会让旧 past 对不上新文字，若只按 token 列散列，这类污染
    表现为"命中了但答案错"——最难查的那种错。淘汰用 OrderedDict：`get` 命中即 move_to_end，
    超预算从最老的一端丢；单条就超预算的项直接拒存（记 `oversized`），避免"存进去立刻自杀"。
    前向由 `engine` 协议代劳（本模块不认识 transformers，`HFEngine` 只是协议的一个实现），
    所以缓存语义可以用一个纯 python 假引擎在 CPU 上测透，不需要真模型。
    **进出各克隆一份 past**：transformers 5.x 的 Cache 会被前向就地追加（探针实测
    seq_len 782→821），不克隆的话缓存里的 past 会被第二次问题污染，命中即错。

【为什么】
    长上下文的成本全在"证据段"：128K 的 state 配 40 token 的问题，整段重算等于把
    99.7% 的算力浪费在没变过的文字上。被否方案一：只按 state 原文字符串散列——同一
    state 在不同题下渲染出的前缀文字虽同，但编号器切出的 token 列可能因前置模板差异而
    不同，键必须落在"模型真正看到的 token"上。被否方案二：不做一致性护栏，默认"数值
    差一点点没关系"——bf16 下 past 复用的漂移是可复跑的两笔账：`.probe_p207_prefix.py`
    （782 token 英文证据）相对漂移 4.98e-3；`.p207_run.py` 落到 run
    1006-p207-longctx-local-half-7b07-2 step2（2065 token 中文证据）1.57e-2 —— 后者已贴着
    0.02 阈值。一旦某条 state 漂移到换人，静默复用就是静默错答（弃缓存只是慢，错答是事故）。
    TODO(p2-10)：服务层接上真实多请求调度后，`serve()` 的 `namespace` 用来隔离租户/会话，
    跨请求命中率的口径以 `stats()` 为准。
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence

import torch

from sys1.decision.render import RENDER_VERSION

__all__ = [
    "DEFAULT_DRIFT_RATIO",
    "SPLIT_MARK",
    "AskResult",
    "CacheError",
    "EngineOut",
    "HFEngine",
    "ParityReport",
    "PrefixCache",
    "PrefixRunner",
    "TokenCounter",
    "estimate_past_bytes",
    "prefix_key",
    "split_prompt_ids",
]

#: 渲染层"证据段/问题段"的固定接缝（`render.user_content` 里那两处换行 + Question:）。
#: 前缀切点必须落在这个接缝之前，否则 past 里会掺进题目文字，跨问就不等价了。
SPLIT_MARK = "\n\nQuestion: "

#: 复用路 vs 重算路的相对漂移阈值（`max_abs_drift / 分数尺度`）。两笔真 bf16 凭据（同机可复跑）：
#: `.probe_p207_prefix.py` → 0.6B/CPU，state 782 token（英文证据）：1.256e-01 / 25.202 ≈ 4.98e-3；
#: `.p207_run.py` → run 1006-p207-longctx-local-half-7b07-2 step2，state 2065 token（中文证据）：
#: 3.846e-01 / 24.495 ≈ 1.57e-2。所以 0.02 只包住最差那一笔（余量 1.27 倍，不是 4 倍）：证据越
#: 长漂移越大，这个阈值偏紧、可能误弃本可复用的 state —— 留待云端长窗实测再定标。为不把"偏紧"
#: 变成"漏判"，判负另加 argmax 换人一票否决（见 `compare_parity`）。
DEFAULT_DRIFT_RATIO = 0.02

#: LRU 默认字节预算：4 GiB。0.6B 上一条 800 token 的 past 约 86 MiB，够放 ~48 条 state；
#: 真正的显存/NPU 账等云端档实测（本波只给 CPU 口径）。
DEFAULT_MAX_BYTES = 4 * 1024 ** 3


class CacheError(ValueError):
    """缓存层的拒绝：对齐失败、引擎接口不合规、键校验不过等。"""


# ---------------------------------------------------------------- B1a：键与缓存核心
def prefix_key(
    token_ids: Sequence[int],
    *,
    render_version: str = RENDER_VERSION,
    namespace: str = "",
) -> str:
    """把"模型真正看到的 state 前缀 token 列"折成一个缓存键（含渲染版号与命名空间）。

    白话：先在这段文字的版本号、这一路请求的名字、还有这串 token 本身上面依次按个手印，
    合成一个指纹。版本号变了、或者 token 少了一个，指纹就完全不同——旧手印绝不会被
    当成新证据用上。

    :param token_ids: 前缀的 token 编号列（不是原文字符串，见模块头"为什么"）。
    :param render_version: 渲染版号，默认取 P1 冻结契约 `RENDER_VERSION`。
    :param namespace: 会话/租户/模型名之类的隔离前缀；不同命名空间互不命中。
    :returns: 64 位十六进制 sha256 摘要（截到 32 字符，碰撞概率对缓存场景足够低）。
    """
    ids = [int(t) for t in token_ids]
    digest = hashlib.sha256()
    digest.update(f"rv={render_version}\x00".encode())
    digest.update(f"ns={namespace}\x00".encode())
    digest.update(f"n={len(ids)}\x00".encode())
    # 逐编号定宽写入：避免 "1,23" 与 "12,3" 拼出同一条字符串这种拼接歧义
    payload = bytearray()
    for tid in ids:
        payload.extend(int(tid).to_bytes(4, "little", signed=True))
    digest.update(bytes(payload))
    return digest.hexdigest()[:32]


def estimate_past_bytes(past: Any) -> int:
    """数一遍 past_key_values 里各层 keys/values 的实占字节（壳、角度表、梯度都不算）。

    白话：缓存要按"真占了多少内存"来淘汰，不能按条数拍脑袋。这里只认成对的键值张量，
    一层层加起来；认不出来的形状回 0，让调用方自己决定要不要按条数兜底。
    """
    if past is None:
        return 0
    layers = getattr(past, "layers", None)
    if layers is None:
        layers = past if isinstance(past, (list, tuple)) else []
    if isinstance(past, dict):                       # 自建 past（测试替身/非 HF 引擎）常是 dict
        total = 0
        for item in past.values():
            if isinstance(item, torch.Tensor):
                total += item.numel() * item.element_size()
            elif isinstance(item, (list, tuple)):
                total += sum(x.numel() * x.element_size() for x in item if isinstance(x, torch.Tensor))
        return total
    total = 0
    for lay in layers:
        for item in _past_pair(lay):
            if isinstance(item, torch.Tensor):
                total += item.numel() * item.element_size()
    return total


def _past_pair(lay: Any) -> tuple[Any, Any]:
    """从"一层"里取出 `(keys, values)`：兼容 `(k, v)` 元组老口径与 `DynamicLayer` 新口径。"""
    if isinstance(lay, (list, tuple)):
        return (lay[0], lay[1]) if len(lay) >= 2 else (None, None)
    return getattr(lay, "keys", None), getattr(lay, "values", None)


class PrefixCache:
    """L1：state 前缀键 → past_key_values 的 LRU 缓存（字节预算 + 淘汰 + 命中记账）。

    白话：一本按"最近用过"排序的账簿。取一份就把它挪到最新的一端；本子写满了就从最久
    没动的那一页开始撕。每次撕都记账，事后能问出"是命中率低还是本子太小"。
    """

    def __init__(self, *, max_bytes: int = DEFAULT_MAX_BYTES, max_items: int = 0) -> None:
        """建一个空缓存；`max_items=0` 表示不限量、只按字节预算淘汰。"""
        if max_bytes <= 0:
            raise CacheError(f"max_bytes 需为正整数（字节），实得 {max_bytes}")
        self.max_bytes = int(max_bytes)
        self.max_items = int(max_items)
        self._items: OrderedDict[str, tuple[Any, int]] = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.oversized = 0
        self.stores = 0

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self._items

    @property
    def bytes_used(self) -> int:
        return self._bytes

    def get(self, key: str) -> Any | None:
        """按键取 past；命中则刷新新鲜度，未命中记一笔 miss 并回 None。

        白话：报指纹取货。取到了就把这盒挪到"刚摸过"那一头，好让淘汰先挑没人碰的；
        取不到就照实记"没命中"，别把 miss 说成 0 命中率的运气问题。
        """
        item = self._items.get(key)
        if item is None:
            self.misses += 1
            return None
        self._items.move_to_end(key)
        self.hits += 1
        return item[0]

    def put(self, key: str, past: Any, *, nbytes: int | None = None) -> int:
        """存入一份 past，回本次占用字节；单条就超预算的存不下（记 `oversized`）。

        白话：先把最久没碰的一盒盒撕掉，撕到本子能装下这一盒为止。要是光这一盒就比整个
        本子大，就不硬塞了——塞进去立刻把自己挤掉的写入等于没写，还白搭一次拷贝。
        """
        size = self.past_bytes(past) if nbytes is None else int(nbytes)
        if size > self.max_bytes:
            self.oversized += 1
            return size
        if key in self._items:                        # 覆盖旧条目：先把旧账扣干净再记新账
            self._bytes -= self._items[key][1]
            del self._items[key]
        self._items[key] = (past, size)
        self._bytes += size
        self.stores += 1
        self._evict()
        return size

    def past_bytes(self, past: Any) -> int:
        """白话：报这份 past 的字节体量——优先问引擎本身，问不到就按通用口径估。"""
        engine = getattr(self, "_engine", None)
        if engine is not None and hasattr(engine, "past_bytes"):
            return int(engine.past_bytes(past))
        return estimate_past_bytes(past)

    def bind_engine(self, engine: Any) -> "PrefixCache":
        """把引擎挂上，让 `put` 不传 nbytes 时用引擎自己的字节口径（HF 与假引擎算法不同）。"""
        self._engine = engine
        return self

    def drop(self, key: str) -> bool:
        """白话：从账本里划掉一条前缀缓存（一致性护栏判负时用）；键不存在回 False。"""
        item = self._items.pop(key, None)
        if item is None:
            return False
        self._bytes -= item[1]
        return True

    def clear(self) -> None:
        """清空内容与条目数，但保留 hits/misses/evictions 记账（便于同一次 run 里对比）。"""
        self._items.clear()
        self._bytes = 0

    def keys(self) -> list[str]:
        """当前在库的键，按"最久未用 → 最新用"排序（淘汰顺序即此序，可拿来核 LRU 语义）。"""
        return list(self._items.keys())

    def _evict(self) -> None:
        """按条数上限与字节预算两头淘汰，永远从最老的一端下手。"""
        while self.max_items and len(self._items) > self.max_items:
            _, item = self._items.popitem(last=False)
            self._bytes -= item[1]
            self.evictions += 1
        while self._bytes > self.max_bytes:
            _, item = self._items.popitem(last=False)
            self._bytes -= item[1]
            self.evictions += 1

    def stats(self) -> dict[str, Any]:
        """回一份可直接进 run 档案的记账快照。

        白话：命中率、在库条数、占了多少内存、撕过几页——这四样够回答"缓存到底有没有用、
        是不是太小了"。字节数一律标 CPU 口径，NPU 显存账要等云端实测才能填。
        """
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": (self.hits / total) if total else 0.0,
            "items": len(self._items),
            "bytes_used": self._bytes,
            "bytes_mib": round(self._bytes / 2 ** 20, 1),
            "stores": self.stores,
            "evictions": self.evictions,
            "oversized": self.oversized,
            "max_bytes": self.max_bytes,
            "max_items": self.max_items,
            "memory_ledger": "cpu-rss/past-bytes，非 NPU 显存账",
        }


class TokenCounter:
    """符号计数器：把"每次前向真喂进去几个 token"按类别记下来（B1b 硬断言的数据源）。

    白话：复用生没生效，不看时间看算力账。时间会被后台进程、热降频骗过去；喂进去的
    token 数骗不了——问题段 39 个符号，命中那次就只该记 39。
    """

    def __init__(self) -> None:
        self.total = 0
        self.calls = 0
        self.by_kind: dict[str, int] = {}

    def add(self, kind: str, count: int) -> int:
        """白话：往开销账上记一笔——按类别（前缀/后缀/整段）累加喂入的格数，返回该类型累计。

        `kind` 用 "prefix"/"suffix"/"full" 三类。"""
        n = int(count)
        self.total += n
        self.calls += 1
        self.by_kind[kind] = self.by_kind.get(kind, 0) + n
        return n

    def snapshot(self) -> int:
        return self.total

    def since(self, mark: int) -> int:
        """回"自 mark 以来新增了多少 token"，用来在调用方就地断言单次开销。"""
        return self.total - mark

    def as_dict(self) -> dict[str, Any]:
        return {"total": self.total, "calls": self.calls, "by_kind": dict(self.by_kind)}


def _maybe_clone(value: Any) -> Any:
    """张量就 `.clone()`，其他（None、标量、缓存壳）原样交回。"""
    return value.clone() if isinstance(value, torch.Tensor) else value


def _has_extra_state(lay: Any) -> bool:
    """这一层除了 keys/values 还压着别的张量状态吗（线性记忆/卷积态就是这种形状）。

    白话：小抄盒子里除了"键值两叠纸"，可能还夹着别的东西——线性记忆那种"读到哪儿就更新到
    哪儿"的进度条。进度条没抄走就交下一问，等于让模型拿半截记忆答题：答得漂亮却全错。
    """
    holder = lay if isinstance(lay, dict) else getattr(lay, "__dict__", {})
    for name, val in dict(holder).items():
        if name in ("keys", "values"):
            continue
        if isinstance(val, torch.Tensor):
            if val.numel() > 0:
                return True
        elif isinstance(val, dict):        # conv_states / recurrent_states：{槽号: 张量|None}
            if any(isinstance(x, torch.Tensor) and x.numel() > 0 for x in val.values()):
                return True
        elif isinstance(val, (list, tuple)):
            if any(isinstance(x, torch.Tensor) and x.numel() > 0 for x in val):
                return True
    return False


# ---------------------------------------------------------------- 引擎协议
@dataclass(frozen=True)
class EngineOut:
    """一次前向的产出：末位数字串 + 可续用的 past + 真喂进去的 token 数。"""

    hidden_last: torch.Tensor      # (1, d)：最后一个真位置的逐位置数字（读点由上层定）
    past: Any                      # 本次之后可继续复用的 KV（重算路可为 None）
    tokens: int                    # 本次前向喂进去的 token 数（计数断言的唯一凭据）


class KVEngine(Protocol):
    """缓存层对"模型"的全部要求：能前向、能克隆 past、能报 past 字节。"""

    def forward(self, input_ids: Sequence[int], past: Any = None, *, offset: int = 0) -> EngineOut:
        """白话：喂入编号做一次前向，交出末位隐层与本次开销账。"""
        ...

    def clone_past(self, past: Any) -> Any:
        ...

    def past_bytes(self, past: Any) -> int:
        ...


class HFEngine:
    """`KVEngine` 的 transformers 实现：包住 HF 因果 LM 的 `body`，只暴露三件能力。

    白话：把"送进去一段字、拿回末位数字和一份可续的 KV 小抄"这件事封成一个动作；
    小抄要不要留着、留多少由上层缓存说，这里只管算和给。

    本类**懒导** transformers：`serving` 在没装 production extras 的环境里也能被导入
    （假引擎测试照样跑），只有真去前向时才要求装得到。
    """

    def __init__(self, body: Any, *, use_cache: bool = True) -> None:
        self.body = body
        self.use_cache = bool(use_cache)

    def forward(self, input_ids: Sequence[int], past: Any = None, *, offset: int = 0) -> EngineOut:
        """白话：真引擎的一次前向——喂编号（可带 past 走增量路），交出末位隐层、新 past 与本次实际吃进的 token 数。

        喂 `input_ids`（可带 past 做增量），回末位数字 + 新 past + 本次 token 数。

        带 past 时必须给 `position_ids`：不给了就是让模型以为这一段还从第 0 位开始，
        RoPE 的角度全错——这是前缀复用最常见的翻车点，所以在本层一次性兜住。
        """
        ids = torch.tensor([[int(t) for t in input_ids]], dtype=torch.long, device=self._device)
        kwargs: dict[str, Any] = {"input_ids": ids, "return_dict": True}
        if past is not None:
            kwargs["past_key_values"] = past
            kwargs["position_ids"] = torch.arange(
                offset, offset + ids.size(1), dtype=torch.long, device=self._device
            ).unsqueeze(0)
        with torch.no_grad():
            out = self.body(**kwargs)
        hidden = out.last_hidden_state[:, -1:, :]        # (1,1,d) → 交上层读末位
        return EngineOut(hidden.reshape(1, -1), out.past_key_values, ids.size(1))

    def clone_past(self, past: Any) -> Any:
        """深拷一份 past：前向会就地追加，缓存里的原件一旦被污染，后续命中全是脏的。

        白话：小抄要留原件、也要留复件——复件交出去被人涂改，原件才还能给下一问接着用。

        transformers 5.x 的 `Cache` 没有 `copy()`（实测），所以按层重建：新建一个空
        Cache，把每层的 keys/values `.clone()` 后按原层号 `update()` 回去。重建不了的两类
        一律抛 `CacheError`（由 `PrefixRunner.ask` 降级成整段重算），绝不静默少拷：
        ① past 壳的构造器要参数（`EncoderDecoderCache` 一类），空壳重建只会 TypeError；
        ② 层里带着 KV 之外的状态（`LinearAttentionLayer` 的 conv_states/recurrent_states，
        0.8B 混合门正是这种形状）——只照 KV 重建就把递归记忆整层丢了，比没缓存更坏。
        """
        if past is None:
            return None
        layers = getattr(past, "layers", None)
        if layers is None:                              # 退回 (keys, values) 元组列表的老口径
            if isinstance(past, list):
                return [(_maybe_clone(k), _maybe_clone(v)) for k, v in past]
            return past
        try:
            fresh = type(past)()
        except TypeError as exc:                        # 空壳建不出来：这型 past 本波不认
            raise CacheError(
                f"past 壳 {type(past).__name__} 无法按无参构造重建（原始错误：{exc}）："
                "DynamicCache 平铺 KV 之外的 past 形态一律拒绝复用（本层只认 KV 形态；防静默错位），改走整段重算"
            ) from exc
        for idx, lay in enumerate(layers):
            keys, values = getattr(lay, "keys", None), getattr(lay, "values", None)
            if isinstance(keys, torch.Tensor) and isinstance(values, torch.Tensor):
                fresh.update(keys.clone(), values.clone(), idx)
            elif isinstance(lay, (list, tuple)) and len(lay) >= 2:      # (k, v) 元组老口径
                fresh.update(_maybe_clone(lay[0]), _maybe_clone(lay[1]), idx)
            elif not hasattr(lay, "keys") and not hasattr(lay, "values"):
                raise CacheError(
                    f"第 {idx} 层（{type(lay).__name__}）没有 keys/values 接口（线性记忆层的形状），"
                    f"本波 clone_past 只认 KV 形态的 past：静默跳过等于把这一层从缓存里整个抹掉，"
                    f"改走整段重算"
                )
            elif _has_extra_state(lay):
                raise CacheError(
                    f"第 {idx} 层（{type(lay).__name__}）压着 KV 之外的状态（线性记忆的 "
                    f"conv_states/recurrent_states 一类）：只照 KV 重建会静默丢记忆，本波改走"
                    f"整段重算；0.8B 混合门要复用，得等 D7 三件套第二件（GDN 旁路）落成"
                )
        return fresh

    def past_bytes(self, past: Any) -> int:
        return estimate_past_bytes(past)

    @property
    def _device(self) -> Any:
        return next(self.body.parameters()).device


# ---------------------------------------------------------------- 前缀切分（编号器口径）
def _encode(tokenizer: Any, text: str) -> list[int]:
    """把文字编成编号列，且强制 `add_special_tokens=False`：特殊符由模板负责，别编两遍。"""
    out = tokenizer.encode(text, add_special_tokens=False)
    if isinstance(out, dict):
        return [int(t) for t in out["input_ids"]]
    if hasattr(out, "ids"):
        return [int(t) for t in out.ids]
    return [int(t) for t in out]


def _template(tokenizer: Any, messages: Sequence[dict[str, str]], **kw: Any) -> str:
    """统一走对话模板；`enable_thinking=False` 与 P1/p2-01 的渲染口径一致（思维链关掉）。"""
    return tokenizer.apply_chat_template(list(messages), tokenize=False, enable_thinking=False, **kw)


def split_prompt_ids(
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    *,
    split_mark: str = SPLIT_MARK,
) -> tuple[list[int], list[int], dict[str, Any]]:
    """把一条渲染结果切成 `(state 前缀编号, 问题段编号)`，并证明前缀是整段的**严格前缀**。

    白话：想让"证据段只算一次"，前提是"证据段的编号"恰好排在"整封短信编号"的最前面、
    一个不差。这里先把 system 壳与证据文字拼出来、按接缝截断，再逐格对照整段编号；
    对不上就当场报错——静默错位比报错危险得多（错位的 past 会让每道题都答得漂漂亮亮地错）。

    :returns: 三元组；第三项是切分凭据（shell_len/suffix_len/full_len/mark_found）。
    :raises CacheError: 没找到接缝、或前缀编号不是整段编号的前缀时抛出。
    """
    if len(messages) < 2:
        raise CacheError(f"渲染结果需含 system + user 两条 messages，实得 {len(messages)}")
    user = messages[1]["content"]
    if split_mark not in user:
        raise CacheError(f"user 正文里找不到接缝 {split_mark!r}，无法界定 state 前缀")
    prefix_user = user.split(split_mark, 1)[0]
    sys_part = _template(tokenizer, [messages[0]], add_generation_prompt=False)
    shell_text = sys_part + prefix_user
    # 只带证据段的 user 轮，模板会补上"用户轮闭合符"；那个尾巴不是 state 的一部分，按长度切掉
    pre_text = _template(
        tokenizer,
        [messages[0], {"role": "user", "content": prefix_user}],
        add_generation_prompt=False,
    )
    if not pre_text.startswith(sys_part):
        raise CacheError("对话模板不是「逐轮拼接」的形态，无法安全截出 state 壳")
    pos = pre_text.find(prefix_user, len(sys_part))
    if pos < 0:
        raise CacheError("user 正文在模板拼接结果中不连续可见（模板含改写/合并），无法安全截出 state 壳")
    state_shell = pre_text[: pos + len(prefix_user)]
    ids_full = _encode(tokenizer, _template(tokenizer, messages, add_generation_prompt=True))
    ids_shell = _encode(tokenizer, state_shell)
    if ids_shell != ids_full[: len(ids_shell)]:
        raise CacheError(
            f"state 前缀与整段编号不对齐（前缀 {len(ids_shell)} 格，公共前缀 "
            f"{_common_prefix_len(ids_shell, ids_full)} 格）：past 复用会静默错位，改走整段重算"
        )
    info = {
        "full_len": len(ids_full),
        "shell_len": len(ids_shell),
        "suffix_len": len(ids_full) - len(ids_shell),
        "mark_found": True,
        "shell_chars": len(state_shell),
    }
    return ids_shell, ids_full[len(ids_shell):], info


def _common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    """两列编号从头数到第一个不同处的格数（对齐失败时的凭据）。"""
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


# ---------------------------------------------------------------- B1b/B1c：复用编排
@dataclass(frozen=True)
class AskResult:
    """一次"问一题"的全部产出与开销账（B1b 断言就看 `prefix_tokens_fed/suffix_tokens_fed`）。"""

    key: str
    hit: bool                          # 前缀是否命中缓存
    recomputed: bool                   # 是否走了整段重算（冷启或护栏判负）
    hidden_last: torch.Tensor          # (1, d) 末位数字，交给 option_scores
    prefix_tokens_fed: int             # 为证据段付的 token 数（命中时必为 0）
    suffix_tokens_fed: int             # 为问题段付的 token 数（恒等于 len(suffix)）
    full_tokens_fed: int               # 护栏重算额外付的 token 数（未触发时 0）
    tokens_fed: int                    # 三者之和 = 本题真喂进去的总 token
    suffix_len: int                    # 问题段应有长度（断言的右半边）
    drift: float | None = None         # 护栏实测 max_abs_drift（未开护栏 None）
    drift_ratio: float | None = None   # 相对漂移 = drift / 分数尺度
    argmax_same: bool | None = None
    elapsed_s: float = 0.0
    uncacheable: bool = False    # 引擎的 past 拷不动（混合记忆态）→ 本题只能整段重算

    @property
    def tokens_are_suffix_only(self) -> bool:
        """B1b 的硬断言位：本次没为证据段付一个符号，且为问题段付的恰是问题段长度。"""
        return self.prefix_tokens_fed == 0 and self.suffix_tokens_fed == self.suffix_len

    def as_dict(self) -> dict[str, Any]:
        """白话：把一次问答的全部开销与一致性凭据摊成窄表——张量不进产物文件，只留可核账的数字。"""
        return {
            "key": self.key, "hit": self.hit, "recomputed": self.recomputed,
            "prefix_tokens_fed": self.prefix_tokens_fed,
            "suffix_tokens_fed": self.suffix_tokens_fed,
            "full_tokens_fed": self.full_tokens_fed,
            "tokens_fed": self.tokens_fed, "suffix_len": self.suffix_len,
            "tokens_are_suffix_only": self.tokens_are_suffix_only,
            "drift": self.drift, "drift_ratio": self.drift_ratio,
            "argmax_same": self.argmax_same, "elapsed_s": round(self.elapsed_s, 4),
            "uncacheable": self.uncacheable,
        }


@dataclass(frozen=True)
class ParityReport:
    """复用路 vs 重算路的一致性结论（B1c 的凭据）。"""

    key: str
    max_abs_drift: float
    score_scale: float
    drift_ratio: float
    argmax_same: bool
    passed: bool
    threshold: float
    scores_reuse: list[float]
    scores_full: list[float]

    def as_dict(self) -> dict[str, Any]:
        """白话：把一致性护栏的漂移账（最大偏移/分数尺度/比值/两侧名次）摊成可入档的窄表。"""
        return {
            "key": self.key, "max_abs_drift": self.max_abs_drift,
            "score_scale": self.score_scale, "drift_ratio": self.drift_ratio,
            "argmax_same": self.argmax_same, "passed": self.passed,
            "threshold": self.threshold,
            "scores_reuse": self.scores_reuse, "scores_full": self.scores_full,
        }


class PrefixRunner:
    """把"缓存 + 引擎 + 计数器 + 一致性护栏"串成一个可问的对象（L2：同 state 多问）。

    白话：先报证据的指纹。库里有对应的小抄，就只把新问的那一句送进模型；没有就老实算
    一遍证据、把小抄存起来。每次问完都告诉你"这次真喂了几个符号"，让"省了多少"这件事
    有数可对，而不是凭感觉。
    """

    def __init__(
        self,
        engine: KVEngine,
        *,
        cache: PrefixCache | None = None,
        counter: TokenCounter | None = None,
        drift_ratio_threshold: float = DEFAULT_DRIFT_RATIO,
        verify: str = "once",
        readout: Any | None = None,
    ) -> None:
        """`verify` ∈ {"off","once","always"}：护栏何时比对；`readout` 是 (hidden,k)→分数 的可调用。"""
        if verify not in ("off", "once", "always"):
            raise CacheError(f"verify 只支持 off/once/always，实得 {verify!r}")
        self.engine = engine
        self.cache = cache if cache is not None else PrefixCache()
        self.cache.bind_engine(engine)
        self.counter = counter if counter is not None else TokenCounter()
        self.drift_ratio_threshold = float(drift_ratio_threshold)
        self.verify = verify
        self.readout = readout
        self._quarantine: set[str] = set()
        self._uncacheable: set[str] = set()
        self._checked: set[str] = set()
        self.verdicts: list[ParityReport] = []
        self.recompute_forced = 0
        self.recompute_uncacheable = 0

    # -- 前向三件套 --------------------------------------------------------------
    def _forward_full(self, ids: Sequence[int]) -> EngineOut:
        out = self.engine.forward(ids, None)
        self.counter.add("full", out.tokens)
        return out

    def _forward_prefix(self, ids: Sequence[int]) -> EngineOut:
        out = self.engine.forward(ids, None)
        self.counter.add("prefix", out.tokens)
        return out

    def _forward_suffix(self, ids: Sequence[int], past: Any, offset: int) -> EngineOut:
        out = self.engine.forward(ids, past, offset=offset)
        self.counter.add("suffix", out.tokens)
        return out

    # -- 对外主入口 --------------------------------------------------------------
    def ask(
        self,
        prefix_ids: Sequence[int],
        suffix_ids: Sequence[int],
        *,
        namespace: str = "",
        force_full: bool = False,
        verify: str | None = None,
        k: int | None = None,
    ) -> AskResult:
        """问一题：命中则只喂问题段，未命中则算证据段并存 past；返回开销账与末位数字。

        白话：`force_full=True` 是"我不信缓存，全部重算"的对照组——两条路都走得通、
        末位数字对得上，才谈得上"复用没改变答案"。
        """
        mode = self.verify if verify is None else verify
        if mode not in ("off", "once", "always"):
            raise CacheError(f"verify 只支持 off/once/always，实得 {mode!r}")
        key = prefix_key(prefix_ids, namespace=namespace)
        t0 = time.perf_counter()
        prefix_fed = suffix_fed = full_fed = 0
        drift: float | None = None
        ratio: float | None = None
        same: bool | None = None
        hit = False
        uncacheable = False

        if force_full or key in self._quarantine or key in self._uncacheable:
            if key in self._uncacheable:
                uncacheable = True                       # 拷不动的 past：每题都从整段重算走
                self.recompute_uncacheable += 1
            out = self._forward_full(list(prefix_ids) + list(suffix_ids))
            full_fed = out.tokens
            self.recompute_forced += 1
            hidden = out.hidden_last
        else:
            try:
                past = self.cache.get(key)
                hit = past is not None
                if past is None:                        # 冷启：证据段算一次并入库
                    pout = self._forward_prefix(list(prefix_ids))
                    prefix_fed = pout.tokens
                    self.cache.put(key, self.engine.clone_past(pout.past))
                    past = pout.past
                out = self._forward_suffix(list(suffix_ids), self.engine.clone_past(past),
                                           len(prefix_ids))
                suffix_fed = out.tokens
                hidden = out.hidden_last
            except CacheError:
                # 引擎说"这份 past 拷不动"：证据段那一笔已经花了，照实记账，再补一次整段重算，
                # 并把键记进 uncacheable——下一次别再白算一遍证据段
                uncacheable = True
                self._uncacheable.add(key)
                self.recompute_uncacheable += 1
                self.cache.drop(key)
                suffix_fed = 0
                out = self._forward_full(list(prefix_ids) + list(suffix_ids))
                full_fed = out.tokens
                hit = False
                hidden = out.hidden_last

        # B1c 护栏：与整段重算比对；判负即拉黑名单 + 用重算结果覆盖本次答案
        used_past = (not force_full) and (not uncacheable) and (key not in self._quarantine)
        need_check = (
            used_past
            and self.readout is not None
            and (mode == "always" or (mode == "once" and key not in self._checked))
        )
        if need_check:
            report = self._parity(key, prefix_ids, suffix_ids, hidden, namespace=namespace, k=k)
            self._checked.add(key)
            self.verdicts.append(report)
            drift, ratio, same = report.max_abs_drift, report.drift_ratio, report.argmax_same
            if not report.passed:
                self._quarantine.add(key)
                self.cache.drop(key)                     # 脏小抄立刻从库里撕掉，不再祸害后来者
                fout = self._forward_full(list(prefix_ids) + list(suffix_ids))
                full_fed += fout.tokens
                hidden = fout.hidden_last
                recomputed = True
            else:
                recomputed = False
        else:
            recomputed = bool(force_full) or uncacheable or key in self._quarantine

        return AskResult(
            key=key, hit=hit, recomputed=recomputed, hidden_last=hidden,
            prefix_tokens_fed=prefix_fed, suffix_tokens_fed=suffix_fed,
            full_tokens_fed=full_fed, tokens_fed=prefix_fed + suffix_fed + full_fed,
            suffix_len=len(suffix_ids), drift=drift, drift_ratio=ratio, argmax_same=same,
            elapsed_s=time.perf_counter() - t0, uncacheable=uncacheable,
        )

    def ask_row(
        self,
        tokenizer: Any,
        messages: Sequence[dict[str, str]],
        *,
        namespace: str = "",
        **kw: Any,
    ) -> AskResult:
        """按渲染结果自动切前缀/问题段后问一题（切分不过就退回整段重算，绝不静默错位）。

        白话：把"从哪一刀切开才算证据段"这件麻烦事收在这儿——调用方只管把渲染好的
        messages 递进来。切不动（模板形状变了）就当没缓存，老老实实整段算一遍。
        """
        try:
            prefix_ids, suffix_ids, _ = split_prompt_ids(tokenizer, messages)
        except CacheError:
            ids = _encode(tokenizer, _template(tokenizer, messages, add_generation_prompt=True))
            return self.ask([], ids, namespace=namespace, force_full=True, **kw)
        return self.ask(prefix_ids, suffix_ids, namespace=namespace, **kw)

    def serve(
        self,
        requests: Iterable[dict[str, Any]],
        *,
        namespace: str = "",
    ) -> dict[str, Any]:
        """L2 批面：一串请求按各自的 state 前缀走同一本缓存，回命中报告与逐题开销账。

        白话：这就是"跨请求"那副样子——不同请求号、不同题目，只要证据一样，第二份起
        就只喂问题段。命中数从 0 变成正数，是这一层的验收点（B2）。
        """
        rows: list[dict[str, Any]] = []
        for req in requests:
            prefix_ids, suffix_ids = list(req["prefix_ids"]), list(req["suffix_ids"])
            res = self.ask(prefix_ids, suffix_ids, namespace=namespace)
            # prefix_len 由调用侧的入参给出（不塞进 AskResult，免得缓存层凭空"知道"长度）
            rows.append({"request_id": req.get("request_id"), "prefix_len": len(prefix_ids),
                         **res.as_dict()})
        hits = sum(1 for r in rows if r["hit"])
        # 省下的证据段 token = 命中且本次没为前缀付费的那些题，各自的前缀长度之和
        avoided = sum(r["prefix_len"] for r in rows if r["hit"] and r["prefix_tokens_fed"] == 0)
        return {
            "rows": rows,
            "requests": len(rows),
            "hits": hits,
            "hit_ratio": round((hits / len(rows)), 4) if rows else 0.0,
            "tokens_fed_total": sum(r["tokens_fed"] for r in rows),
            "tokens_avoided_in_prefix": avoided,
            "all_hits_suffix_only": all(
                r["prefix_tokens_fed"] == 0 and r["suffix_tokens_fed"] == r["suffix_len"]
                for r in rows if r["hit"]
            ),
            "cache": self.cache.stats(),
            "counter": self.counter.as_dict(),
        }

    # -- 一致性护栏 --------------------------------------------------------------
    def _parity(
        self,
        key: str,
        prefix_ids: Sequence[int],
        suffix_ids: Sequence[int],
        hidden_reuse: torch.Tensor,
        *,
        namespace: str = "",
        k: int | None = None,
    ) -> ParityReport:
        """复用路 vs 重算路的分数比对（重算路自己前向一遍，不碰缓存）。"""
        fout = self._forward_full(list(prefix_ids) + list(suffix_ids))
        return compare_parity(
            key, hidden_reuse, fout.hidden_last, self.readout,
            threshold=self.drift_ratio_threshold, k=k,
        )

    def verify_pair(
        self,
        prefix_ids: Sequence[int],
        suffix_ids: Sequence[int],
        *,
        namespace: str = "",
        k: int | None = None,
    ) -> dict[str, Any]:
        """显式做一次"复用/重算"双臂对照（B1c 的测试与 run 凭据入口）。

        白话：先按没缓存的样子把证据算出来存好，再问一次题走命中路，同时整段重算一遍
        作对照；两条路的分数、漂移、名次一并交出去。
        """
        cold = self.ask(prefix_ids, suffix_ids, namespace=namespace, verify="off")
        warm = self.ask(prefix_ids, suffix_ids, namespace=namespace, verify="off")
        direct = self.ask(prefix_ids, suffix_ids, namespace=namespace, force_full=True)
        report = compare_parity(
            warm.key, warm.hidden_last, direct.hidden_last, self.readout,
            threshold=self.drift_ratio_threshold, k=k,
        )
        self.verdicts.append(report)
        if not report.passed:
            self._quarantine.add(warm.key)
            self.cache.drop(warm.key)
        return {
            "cold": cold.as_dict(), "warm": warm.as_dict(),
            "direct": direct.as_dict(), "parity": report.as_dict(),
            "warm_hit": warm.hit,
        }

    def stats(self) -> dict[str, Any]:
        """汇总：缓存记账 + 计数器记账 + 护栏结论 + 黑名单规模。

        白话：一屏回答四问——命中多少、真喂了多少符号、护栏判过几次负、有几条 state 被
        拉黑。护栏判负的键会自动进黑名单，之后的命中请求一律重算。
        """
        return {
            "cache": self.cache.stats(),
            "counter": self.counter.as_dict(),
            "quarantined": sorted(self._quarantine),
            "uncacheable_keys": sorted(self._uncacheable),
            "verified_keys": len(self._checked),
            "parity_failed": sum(1 for r in self.verdicts if not r.passed),
            "recompute_forced": self.recompute_forced,
            "recompute_uncacheable": self.recompute_uncacheable,
        }


def compare_parity(
    key: str,
    hidden_reuse: torch.Tensor,
    hidden_full: torch.Tensor,
    readout: Any,
    *,
    threshold: float = DEFAULT_DRIFT_RATIO,
    k: int | None = None,
) -> ParityReport:
    """把两路的末位数字各自折成候选分，报 max_abs_drift / 相对漂移 / argmax 是否同人。

    白话：分差要除以分数本身的尺度才可比——25 分量级上差 0.12 是小事，1 分量级上差
    0.12 就够换人了。判负两条并排走：①相对漂移超阈值（阈值可配）；②名次换人——一票否决。
    第二条不看幅度，因为"答成了另一个选项"没有"差得不多"这种豁免。

    :param readout: `readout(hidden_last, k) -> (1, k)` 分数张量；通常就是 `option_scores` 的偏应用。
    """
    s_reuse = readout(hidden_reuse, k).float()
    s_full = readout(hidden_full, k).float()
    drift = (s_reuse - s_full).abs().max().item()
    scale = max(s_full.abs().mean().item(), 1e-6)
    ratio = drift / scale
    same = bool(int(s_reuse.argmax()) == int(s_full.argmax()))
    return ParityReport(
        key=key, max_abs_drift=float(drift), score_scale=float(scale),
        drift_ratio=float(ratio),
        argmax_same=same,
        passed=bool(same and ratio <= threshold), threshold=float(threshold),
        scores_reuse=[round(v, 6) for v in s_reuse[0].tolist()],
        scores_full=[round(v, 6) for v in s_full[0].tolist()],
    )
