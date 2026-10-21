"""sys1/layers/attention.py — 因果滑窗长文路径（p2-07 C1；D7 三件套"滑窗封顶"的第一件基座）。

【做什么】
    给长上下文一条**成本与序列长度解耦**的注意力路：每个位置只回看左边的 W 个（因果半窗，
    与 P1 `sys1/kernels/attn_sw` 同一口径），并且能按层混排（一部分层滑窗、一部分层全注意）。
    对外四件事：① 层排布配置（`resolve_layer_windows`：全滑窗 / 5:1 / 3:1 / 逐层清单）；
    ② 窗上限开关（`cap`：把任何超过上限的窗夹住，全注意力层也可选择一并夹住——这就是
    D7 说的"滑窗封顶"，KV 因此有硬上界）；③ 两条实现路（掩码语义路 `attention_masked`
    与内存封顶的分块带状路 `attention_band`）并互相印证；④ KV 内存账（解析式 + CPU 实测）。

【怎么做】
    语义只有一条式子：`可见 ⟺ 0 <= i - j < window`（`window=None` 退化成全因果）。
    - 掩码路直接物化 (heads,T,T) 的分数——序列一长就是 T² 的内存，只当"口径尺"用；
    - 带状路把查询切成 chunk，每块只带 `chunk + window - 1` 长的键值窗口过去：内存峰值
      由 `chunk+window` 决定而与 T 无关。其中 `route="kernel"` 走 P1 的 `attn_sw_kernel.forward`
      （它要求三路同形，所以整带一起喂、取带尾 chunk 行——多算的是带首那段的重复查询，
      换来的是"内核口径"与"掩码口径"必须逐位相等）；`route="local"` 就地算带内 fp32 分数，
      少算重复行但走的是本地实现。
    接进模型不动主干：`DecoderBlock.forward(x, cos, sin, visible)` 本来就收可见位，
    `forward_with_windows` 按层把带状掩码喂进去即可（`sys1/model.py` 零改动）。

【为什么】
    长文的痛不在"算得慢"而在"T² 的分数矩阵装不下"。被否方案一：只加掩码不做分块——
    128K 序列的分数矩阵是 128K² × 头数，任何本地设备都装不下，等于没解决。
    被否方案二：分块路自己另写一份 softmax 而不与内核互证——两份实现各说各话，滑窗方向
    抄反（`j - i < window`）这种错就没人抓得住；所以带状路必须与掩码路做 allclose 断言。
    被否方案三：把窗口配置塞进 `ModelConfig`——那要改 P1 冻结的模型契约文件，而窗与排布
    是"策略"不是"权重形状"，本层用 `LongContextPlan` 承载，序列化进 run 档案即可复现。
    TODO(p2-07 C2/云端)：`kv_ledger` 是解析账，NPU 上的真实显存峰值要等开卡实测；本波
    一律标 CPU 口径（`memory_ledger` 字段就是这么写的）。
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "DEFAULT_WINDOW",
    "AttnOut",
    "LongContextPlan",
    "attention_band",
    "attention_masked",
    "compare_routes",
    "forward_with_windows",
    "kv_ledger",
    "layer_visibles",
    "resolve_layer_windows",
    "rss_bytes",
    "sliding_visible",
    "window_band",
]

#: 默认因果半窗宽度（参考落点：Naive-N0.5-Flash 的 SWA=128）
DEFAULT_WINDOW = 128
NEG_INF = float("-inf")


# ---------------------------------------------------------------- 层排布与窗上限
@dataclass(frozen=True)
class LongContextPlan:
    """一次长文前向的"看谁"配置：逐层窗口 + 窗上限 + 口径说明。

    白话：一张排班表——哪几层只许看身边 W 个、哪几层许看全场，外加一道封顶闸门。
    表本身可序列化，所以 run 档案里写的"W=128、5:1 混排、封顶 256K"能被原样复现。
    """

    layers: int
    windows: tuple[int | None, ...]      # None = 这一层走全注意力
    cap: int | None = None               # 窗上限开关（D7 滑窗封顶）
    cap_full: bool = False               # 封顶是否连全注意力层一起夹住
    pattern: str = "all"                 # 排布口径（"all"/"5:1"/"3:1"/逐层串）
    window: int = DEFAULT_WINDOW         # 排布里"滑窗层"的默认窗宽
    clamped: int = 0                     # 被 cap 夹住的层数（记账用）

    def as_dict(self) -> dict[str, Any]:
        """摊进 run 档案：窗上限与夹了几层必须留痕，否则"封顶"只是嘴上封顶。

        白话：把这张排班表抄成一份能读的账——每层看多宽、封顶闸门设在哪、夹住了几层，一行
        都不落。日后追问"你那次到底封没封顶"，账上就有答案，不用回头翻代码。
        """
        return {
            "layers": self.layers, "pattern": self.pattern, "window": self.window,
            "windows": [w if w is not None else "full" for w in self.windows],
            "cap": self.cap, "cap_full": self.cap_full, "clamped_layers": self.clamped,
        }

    def kv_ledger(self, seq: int, *, kv_heads: int, head_dim: int, dtype_bytes: int = 2) -> dict[str, Any]:
        """按本排布给 KV 的解析内存账（全注意力同长序列做对照）。"""
        return kv_ledger(seq, layers=self.layers, windows=self.windows, kv_heads=kv_heads,
                         head_dim=head_dim, dtype_bytes=dtype_bytes)


def resolve_layer_windows(
    layers: int,
    *,
    window: int = DEFAULT_WINDOW,
    pattern: str | Sequence[Any] = "all",
    cap: int | None = None,
    cap_full: bool = False,
) -> LongContextPlan:
    """把"排布写法"折成逐层窗口清单，并施加窗上限。

    白话：说清三件事就够了——一共几层、每层看多远、最多允许看多远。写法随你：
    "all" 全滑窗；"5:1" 是五层滑窗配一层全注意（Naive 的形态），"3:1" 同理（GLM 形态）；
    也可直接给一串逐层的数（`None`/"full" 表示这层看全场）。封顶闸门把超宽的窗夹住，
    开了 `cap_full` 连"看全场"的那几层也一并夹住——那样 KV 才有硬上界。

    :raises ValueError: 层数非正、窗宽非正、排布写法不认识时抛出。
    """
    if int(layers) <= 0:
        raise ValueError(f"layers 需为正整数，实得 {layers}")
    if int(window) < 1:
        raise ValueError(f"window 需 >= 1，实得 {window}")
    n = int(layers)
    raw = _pattern_to_windows(n, window, pattern)
    clamped = 0
    windows: list[int | None] = []
    for w in raw:
        if cap is None:
            windows.append(w)
            continue
        if w is None:
            if cap_full:
                windows.append(int(cap))
                clamped += 1
            else:
                windows.append(None)
        elif w > cap:
            windows.append(int(cap))
            clamped += 1
        else:
            windows.append(w)
    return LongContextPlan(
        layers=n, windows=tuple(windows), cap=None if cap is None else int(cap),
        cap_full=bool(cap_full),
        pattern=pattern if isinstance(pattern, str) else "explicit",
        window=int(window), clamped=clamped,
    )


def _pattern_to_windows(n: int, window: int, pattern: str | Sequence[Any]) -> list[int | None]:
    """排布写法 → 长度恰为 n 的逐层窗宽清单（`None` = 全注意力）。"""
    if isinstance(pattern, str):
        text = pattern.strip().lower()
        if text in ("all", "sliding", "swa"):
            return [window] * n
        if text in ("full", "none"):
            return [None] * n
        if ":" in text:                                   # "5:1" = 5 滑窗 + 1 全注意，循环铺满
            parts = text.split(":")
            if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
                raise ValueError(f"比例排布需写成 'N:M'（N 滑窗层数 : M 全注意层数），实得 {pattern!r}")
            sliding, full = int(parts[0]), int(parts[1])
            if sliding + full <= 0:
                raise ValueError(f"比例排布 {pattern!r} 的两侧不能同时为 0")
            cycle = [window] * sliding + [None] * full
            return [cycle[i % len(cycle)] for i in range(n)]
        if set(text) <= set("sf"):                       # "SSSSF" 逐层标记：s 滑窗 / f 全注意
            if len(text) != n:
                raise ValueError(f"逐层标记串 {pattern!r} 长度 {len(text)} 与层数 {n} 不符")
            return [window if c == "s" else None for c in text]
        raise ValueError(f"认不出的层排布写法 {pattern!r}（支持 all/full/'N:M'/'SSSSF'/逐层清单）")
    seq = list(pattern)
    if len(seq) != n:
        raise ValueError(f"逐层窗口清单长度 {len(seq)} 与层数 {n} 不符")
    out: list[int | None] = []
    for item in seq:
        if item is None or (isinstance(item, str) and item.strip().lower() in ("full", "none", "f")):
            out.append(None)
            continue
        if isinstance(item, bool):
            raise ValueError(f"逐层窗口不接受 bool，实得 {item!r}")
        w = int(item)
        if w < 1:
            raise ValueError(f"逐层窗口需 >= 1 或 None（全注意力），实得 {item!r}")
        out.append(w)
    return out


# ---------------------------------------------------------------- 掩码语义（与内核同口径）
def window_band(seq: int, window: int | None, *, device: Any = None) -> torch.Tensor:
    """(1,1,T,T) 布尔带：`True ⟺ 0 <= i - j < window`（`window=None` 只卡因果半边）。

    白话：画一张"谁可以望见谁"的方格纸——每格代表"第 i 位看不看得到第 j 位"，
    只留自己这一格和左边紧挨着的 W-1 格，其余一律涂黑。式子与 P1 内核的 `_mask` 逐字同式，
    所以"掩码口径"与"内核口径"有对得上的前提。
    """
    idx = torch.arange(seq, device=device)
    delta = idx[:, None] - idx[None, :]
    allowed = delta >= 0 if window is None else (delta >= 0) & (delta < int(window))
    return allowed[None, None]


def sliding_visible(
    seq: int,
    *,
    window: int | None = None,
    attention_mask: torch.Tensor | None = None,
    batch: int = 1,
    device: Any = None,
) -> torch.Tensor:
    """给 `DecoderBlock` 的可见位：因果 ∧ 窗带 ∧（批内补洞位不做键）。

    白话：三条规矩叠在一起才准看——不许看未来、不许看窗外的过去、不许把垫出来的空洞
    当内容；同时保证空洞位自己看自己，免得整行被涂黑后 softmax 除出 NaN。
    """
    band = window_band(seq, window, device=device)
    if attention_mask is None:
        return band
    if attention_mask.shape != (batch, seq):
        raise ValueError(f"attention_mask 需为 (B,T)=({batch},{seq})，实得 {tuple(attention_mask.shape)}")
    key_ok = attention_mask.to(torch.bool)[:, None, None, :]
    eye = torch.eye(seq, dtype=torch.bool, device=device)[None, None]
    return band & (key_ok | eye)


def layer_visibles(
    seq: int,
    plan: LongContextPlan | Sequence[int | None],
    *,
    attention_mask: torch.Tensor | None = None,
    batch: int = 1,
    device: Any = None,
) -> list[torch.Tensor | None]:
    """逐层的可见位清单：滑窗层给窄带，全注意力层给"纯因果带"。

    白话：这里刻意**不留空不发**。按主干的规矩，"这张纸不发"等于"谁都能看见谁"，因果当场
    漏掉；所以连"看全场"那一层也要发一张满宽的"只许回头看"，才敢直接逐层喂下去。

    技术注：`visible=None` 在 `CausalAttention` 里是"任何掩码都不加"的双向分支，并不是
    "全因果"——只有 `Decoder.forward` 才替它补上因果带；逐层清单要能直接喂进 `DecoderBlock`，
    所以"看全场"这一层也得把因果带画出来。
    """
    windows = plan.windows if isinstance(plan, LongContextPlan) else tuple(plan)
    return [
        sliding_visible(seq, window=w, attention_mask=attention_mask, batch=batch, device=device)
        for w in windows
    ]


def forward_with_windows(
    decoder: Any,
    input_ids: torch.Tensor,
    plan: LongContextPlan | Sequence[int | None],
    *,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """按逐层窗口跑一遍 `sys1.model.Decoder`：(B,T) 进、(B,T,d) 出，模型本体零改动。

    白话：模型本来就肯收"谁可以望见谁"这张纸，这里不过是给每层各发一张——滑窗层发窄的，
    全注意层干脆不发（让它按原来的全因果走）。

    窗口 `None`（全注意力）层的可见位取"纯因果带"：与主干 `Decoder.forward` 自己在
    `attn_mask=None` 时算出的那张纸逐格同值（`visible=None` 在主干里是"不加任何掩码"的
    双向分支，绝不能拿来当全注意力，否则因果性当场破了）。副作用是滑窗层的
    `_kernel_ok` 判据（"带掩码就不走内核"）会落回 eager 写法——CPU 档本来就没内核，
    真上 NPU 时要走内核路得让 attn_sw 侧接受带状键值，见本域 C2/云端后续。
    """
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids 需为 (B,T) 两维，实得 {tuple(input_ids.shape)}")
    batch, seq = input_ids.shape
    windows = plan.windows if isinstance(plan, LongContextPlan) else tuple(plan)
    if len(windows) != decoder.config.L:
        raise ValueError(f"窗口清单有 {len(windows)} 层，模型有 {decoder.config.L} 层，对不上")
    cos, sin = decoder._tables(seq, input_ids.device)          # 与主干同一张角度表（同真源）
    if attn_mask is None:
        # 全注意力层用的"纯因果带"，与主干 _visible_mask(seq, None, ...) 逐格同值
        visible_all = window_band(seq, None, device=input_ids.device)
    else:
        visible_all = _full_visible(seq, attn_mask, batch, input_ids.device)
    x = decoder.tok_emb(input_ids)
    for block, w in zip(decoder.blocks, windows):
        vis = visible_all if w is None else sliding_visible(
            seq, window=w, attention_mask=attn_mask, batch=batch, device=input_ids.device)
        x = block(x, cos, sin, vis)
    return decoder.final_norm(x)


def _full_visible(seq: int, attn_mask: torch.Tensor, batch: int, device: Any) -> torch.Tensor:
    """全注意力 + 批内补洞的可见位：借用主干那份实现，避免两处各写一遍三条规矩。"""
    from sys1.model import _visible_mask                        # 同包内私有件，方向单一不成环

    return _visible_mask(seq, attn_mask, batch, device)


# ---------------------------------------------------------------- 两条实现路
@dataclass(frozen=True)
class AttnOut:
    """一次注意力前向的产出与内存账（`stats` 是"实测/解析"凭据，不是装饰）。"""

    out: torch.Tensor                                  # (heads, seq, dim)
    stats: dict[str, Any]

    @property
    def materialized_peak_bytes(self) -> int:
        return int(self.stats.get("materialized_peak_bytes", 0))


def attention_masked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    window: int | None = None,
    scale: float | None = None,
) -> AttnOut:
    """掩码语义路：物化整张 (heads,T,T) 分数再做带状 softmax——口径尺，长文别用它。

    白话：把每个人对所有人的相关度一次摊满一张大方桌再算份额。清楚是清楚，桌子面积随
    人数平方长，人一多就摆不下；它的职责是给带状路当"标准答案"。
    """
    q3, k3, v3 = _flatten_heads(q, k, v)
    heads, seq, dim = q3.shape
    scale = dim ** -0.5 if scale is None else float(scale)
    score = q3.to(torch.float32) @ k3.to(torch.float32).transpose(-1, -2) * scale
    band = window_band(seq, window, device=q.device)[0]        # (1,T,T)：与 score 同秩，masked_fill 不升维
    score = score.masked_fill(~band, NEG_INF)
    out = torch.softmax(score, dim=-1) @ v3.to(torch.float32)
    return AttnOut(out.reshape(lead_shape(q) + (seq, dim)), {
        "route": "masked", "window": window, "seq": seq, "heads": heads,
        "materialized_peak_bytes": score.numel() * score.element_size(),
        "memory_ledger": "cpu-alloc 口径，非 NPU 显存账",
    })


def attention_band(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
    *,
    chunk: int = 512,
    route: str = "kernel",
    scale: float | None = None,
) -> AttnOut:
    """分块带状路：每次只带 `chunk + window - 1` 长的键值，内存与总长 T 解耦。

    白话：把长队切成一小段一小心地看。看第 500 到 512 个人时，只把前面窗那么远的人请
    进来作陪，后面的人压根不必到场——桌子的大小由"窗 + 一小段"决定，与人总数无关。

    :param route: `"kernel"` 复用 P1 `sys1/kernels/attn_sw_kernel.forward`（整带同形喂入，
        取带尾 chunk 行出口）；`"local"` 就地算带内 fp32 分数（不做带首那段的重复查询）。
    :raises ValueError: 窗口非正、路名不认识时抛出。
    """
    if int(window) < 1:
        raise ValueError(f"window 需 >= 1，实得 {window}")
    if route not in ("kernel", "local"):
        raise ValueError(f"route 只支持 kernel/local，实得 {route!r}")
    q3, k3, v3 = _flatten_heads(q, k, v)
    heads, seq, dim = q3.shape
    scale = dim ** -0.5 if scale is None else float(scale)
    w = int(window)
    step = max(1, int(chunk))
    band_len = step + w - 1
    kernel_forward = None
    if route == "kernel":
        from sys1.kernels import attn_sw_kernel                # 懒导：P1 内核入口
        kernel_forward = attn_sw_kernel.forward
    out = torch.empty_like(q3, dtype=torch.float32)
    peak = 0
    for start in range(0, seq, step):
        stop = min(seq, start + step)
        band_start = max(0, start - (w - 1))
        band_stop = stop
        qb = q3[:, band_start:band_stop] if route == "kernel" else q3[:, start:band_stop]
        kb, vb = k3[:, band_start:band_stop], v3[:, band_start:band_stop]
        if route == "kernel":
            # 内核要求三路同形：整带一起算，取带尾 (stop-start) 行——带首那些行是重复查询
            got = kernel_forward(qb, kb, vb, window=w, scale=scale, out_dtype=torch.float32)
            piece = got[:, qb.size(1) - (stop - start):]
            peak = max(peak, qb.size(1) * qb.size(1) * heads * 4)
        else:
            score = qb.to(torch.float32) @ kb.to(torch.float32).transpose(-1, -2) * scale
            delta = torch.arange(start, stop, device=q.device)[:, None] - \
                torch.arange(band_start, band_stop, device=q.device)[None, :]
            score = score.masked_fill(~((delta >= 0) & (delta < w)), NEG_INF)
            piece = torch.softmax(score, dim=-1) @ vb.to(torch.float32)
            peak = max(peak, score.numel() * score.element_size())
        out[:, start:stop] = piece
    stats = {
        "route": route, "window": w, "seq": seq, "heads": heads, "chunk": step,
        "band_len": band_len, "chunks": math.ceil(seq / step),
        "materialized_peak_bytes": int(peak),
        "scales_with": "O(chunk x (chunk+window))，与总长无关" if route == "local"
        else "O((chunk+window)^2) x 块数，单块峰值与总长无关",
        "memory_ledger": "cpu-alloc 口径，非 NPU 显存账",
    }
    return AttnOut(out.reshape(lead_shape(q) + (seq, dim)), stats)


def compare_routes(
    seq: int,
    window: int,
    *,
    heads: int = 2,
    dim: int = 16,
    chunk: int = 64,
    seed: int = 20261005,
    device: Any = None,
) -> dict[str, Any]:
    """造合成 q/k/v，把"掩码路 vs 带状 kernel 路 vs 带状 local 路"三方对齐并回凭据。

    白话：同一段人造内容，三种算法各跑一遍，把两两差值的最大值报出来。相等才说明
    "分块"没把窗口的方向或边界切错——这是滑窗路径最容易被悄悄写反的地方。
    """
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    q = torch.randn(heads, seq, dim, generator=gen, device=device)
    k = torch.randn(heads, seq, dim, generator=gen, device=device)
    v = torch.randn(heads, seq, dim, generator=gen, device=device)
    ref = attention_masked(q, k, v, window=window)
    full_ref = attention_masked(q, k, v, window=None)
    res: dict[str, Any] = {"seq": seq, "window": window, "heads": heads, "dim": dim, "chunk": chunk}
    for route in ("kernel", "local"):
        got = attention_band(q, k, v, window, chunk=chunk, route=route)
        diff = (got.out - ref.out).abs().max().item()
        res[f"{route}_vs_masked_max_abs"] = float(diff)
        res[f"{route}_peak_bytes"] = got.materialized_peak_bytes
        res[f"{route}_allclose"] = bool(torch.allclose(got.out, ref.out, atol=2e-4, rtol=1e-4))
    res["masked_peak_bytes"] = ref.materialized_peak_bytes
    res["full_attention_peak_bytes"] = full_ref.materialized_peak_bytes
    res["peak_ratio_vs_full"] = (
        res["local_peak_bytes"] / res["full_attention_peak_bytes"]
        if res["full_attention_peak_bytes"] else 0.0
    )
    res["window_equals_full_when_seq_le_window"] = bool(
        window >= seq and torch.allclose(ref.out, full_ref.out, atol=1e-6, rtol=0)
    )
    res["stats_band"] = attention_band(q, k, v, window, chunk=chunk, route="local").stats
    return res


# ---------------------------------------------------------------- 内存账与 CPU 实测
def kv_ledger(
    seq: int,
    *,
    layers: int,
    windows: Sequence[int | None] | None = None,
    kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> dict[str, Any]:
    """KV 常驻字节的解析账：全注意力 vs 逐层窗口，另给"封顶后还剩多少"的比例。

    白话：KV 这笔账很好算——每层每头每个位置都留键、值两份数字。滑窗的省法在于：
    窗外的过去既然谁也看不见，那份数字就大可丢掉。于是同样 128K 长度，"看全场"与
    "只看 128 个"的字节数差出几个量级，这就是 D7 里"KV 封顶"那条的算术底。
    """
    per_token = 2 * int(kv_heads) * int(head_dim) * int(dtype_bytes) * int(layers)
    full_bytes = per_token * int(seq)
    wins = list(windows) if windows is not None else [DEFAULT_WINDOW] * int(layers)
    win_bytes = 0
    for w in wins:
        resident = int(seq) if w is None else min(int(seq), int(w))
        win_bytes += 2 * int(kv_heads) * int(head_dim) * int(dtype_bytes) * resident
    return {
        "seq": int(seq), "layers": int(layers), "kv_heads": int(kv_heads),
        "head_dim": int(head_dim), "dtype_bytes": int(dtype_bytes),
        "kv_bytes_per_token_per_layer": 2 * int(kv_heads) * int(head_dim) * int(dtype_bytes),
        "full_attention_bytes": full_bytes,
        "windowed_bytes": win_bytes,
        "full_attention_mib": round(full_bytes / 2 ** 20, 1),
        "windowed_mib": round(win_bytes / 2 ** 20, 1),
        "saved_ratio": round(1 - win_bytes / full_bytes, 4) if full_bytes else 0.0,
        "windows": [w if w is not None else "full" for w in wins],
        "memory_ledger": "解析账（CPU 口径），NPU 显存实测待开卡",
    }


def rss_bytes() -> int:
    """本进程峰值驻留内存（macOS/Linux 的 `ru_maxrss` 单位是字节，Linux 内核上是 KB）。

    白话：向操作系统问一句"这个进程最多同时占过多少内存"。这是**进程级**的粗账，
    看不到"哪一次前向占的"，所以只在报告里当辅助数，主数用解析账与物化字节。
    """
    import platform
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if platform.system() != "Linux" else int(peak) * 1024


def _flatten_heads(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 (seq,dim) / (heads,seq,dim) / (B,heads,seq,dim) 统一摊成 (头总数, seq, dim)。

    与 P1 `attn_sw_kernel` 入口同一摊法：前导维全当成"头"这一根轴，带状切片只碰第 1 轴。
    """
    if q.dim() not in (2, 3, 4):
        raise ValueError(f"q 需为 (seq,dim)/(heads,seq,dim)/(B,heads,seq,dim)，实得 {q.dim()} 维")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v 需同形，实得 {tuple(q.shape)}/{tuple(k.shape)}/{tuple(v.shape)}")
    if q.size(-1) != k.size(-1):
        raise ValueError("q 与 k 的头宽必须一致")
    flat = (-1, q.size(-2), q.size(-1))
    return q.reshape(flat).contiguous(), k.reshape(flat).contiguous(), v.reshape(flat).contiguous()


def lead_shape(t: torch.Tensor) -> torch.Size:
    """还原前导维（batch×heads 摊平前那一截），出口形状与入口严格一致。"""
    return t.shape[:-2]
