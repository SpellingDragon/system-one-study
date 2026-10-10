"""tests/test_attention_shapes.py — `sys1/layers/attention.py` 两条路的**秩契约**回归钉（缺陷一，agent E）。

钉的是产品形状违约，不是数值口径。修复前实测（凭据：`D_determinism.md §3.2` +
`E_probe_shapes.py matrix`，本机 2026-10-11 复跑）：

| 入口开本 | `attention_masked` | `attention_band`（kernel/local 同） |
|---|---|---|
| `(seq,dim)` 2 维 | 出口 (1,1,seq,dim) **升两维** | 出口 (1,seq,dim) **升一维** |
| `(heads,seq,dim)` 3 维 | 出口 (1,heads,seq,dim) **升一维** | 出口同秩（唯二走对的一档） |
| `(B,heads,seq,dim)` 4 维 | **RuntimeError**：`shape '[1, 2, 16]' is invalid for input of size 6400` | **同一个 RuntimeError** |

根因两处（同一族）：① `masked_fill` 拿 (1,1,T,T) 掩码去套 (H,T,T) 分数，按广播出形 ⇒ 升维；
② 出口 reshape 写成 `lead_shape(q) + (dim,)`，把 seq 那一维丢了 ⇒ 4 维入必然元素数对不上。

秩契约的出处 **[文档]**：`AttnOut.out` 字段注释 `(heads, seq, dim)`；`lead_shape` docstring
"出口形状与入口严格一致"；`_flatten_heads` 明写接受 (seq,dim)/(heads,seq,dim)/(B,heads,seq,dim)。
消费方预期秩 **[实测]**（`grep -rn "attention_masked" release/ --include=*.py`：仓内调用方只有
`compare_routes` 与 `tests/test_longctx.py`，两处都喂 3 维；`attention_band` 的 `out[:, start:stop]`
就地写入也只认 3 维）：4 维开本今天**没有**仓内消费者，但契约件明写支持、一调即崩，属产品缺陷
（长文批推理接上就炸）；2 维开本同理（`_flatten_heads` 声明支持，`attn_sw_kernel` 同口径）。
"""
from __future__ import annotations

import pytest
import torch

from sys1.layers import attention as LA

SEQ, HEADS, DIM, BATCH, WINDOW, CHUNK = 24, 2, 8, 3, 16, 6
SHAPES = {2: (SEQ, DIM), 3: (HEADS, SEQ, DIM), 4: (BATCH, HEADS, SEQ, DIM)}


def _qkv(rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """同一颗 seed 给 q/k/v 三份同形数据，按 `_flatten_heads` 声明的三种开本各取一份。"""
    gen = torch.Generator(device="cpu").manual_seed(20261011)
    return (torch.randn(SHAPES[rank], generator=gen) for _ in range(3))


def _routes(window: int) -> dict[str, object]:
    """三条路统一成 `f(q,k,v) -> out`：掩码路 + 带状 kernel 路 + 带状 local 路。"""
    return {
        "masked": lambda q, k, v: LA.attention_masked(q, k, v, window=window).out,
        "band_kernel": lambda q, k, v: LA.attention_band(q, k, v, window, chunk=CHUNK, route="kernel").out,
        "band_local": lambda q, k, v: LA.attention_band(q, k, v, window, chunk=CHUNK, route="local").out,
    }


# ---------------------------------------------------------------- 钉①：入出同秩（2/3/4 维全档）
@pytest.mark.parametrize("rank", [2, 3, 4])
def test_attention_routes_preserve_rank(rank: int):
    """`_flatten_heads` 声明的三种开本，两条路出口秩都必须与入口一致（缺陷一的直接回归钉）。"""
    q, k, v = _qkv(rank)
    assert tuple(q.shape) == SHAPES[rank]
    for name, fn in _routes(WINDOW).items():
        out = fn(q, k, v)
        assert out.dim() == rank and tuple(out.shape) == SHAPES[rank], (
            f"{name}: {rank} 维入 {SHAPES[rank]} → 出 {tuple(out.shape)}，"
            "违反 AttnOut 声明与 lead_shape『出口形状与入口严格一致』"
            "（回归形态：掩码广播升维 / 出口 reshape 丢 seq 维）")


# ---------------------------------------------------------------- 钉②：4 维批入 ≡ 逐样本循环
@pytest.mark.parametrize("route", ["masked", "band_kernel", "band_local"])
def test_batched_4d_matches_per_sample_loop(route: str):
    """`(B,heads,seq,dim)` 批量入：出口必须 4 维且与逐样本 3 维循环同值（修复前这一档**直接崩**）。"""
    q, k, v = _qkv(4)
    fn = _routes(WINDOW)[route]
    batched = fn(q, k, v)
    per = torch.stack([fn(q[b], k[b], v[b]) for b in range(BATCH)])
    assert tuple(batched.shape) == SHAPES[4] == tuple(per.shape), (
        f"{route}: 批入出口 {tuple(batched.shape)} 与入口 {SHAPES[4]} 不同秩")
    max_abs = float((batched - per).abs().max())
    # 位等只用在"同形状同分派"处（D_determinism §1.3 的辨析）：批 6 头与单 2 头是两种 BLAS 分派面，
    # 这里主判据取"秩 + 仓内量级口径"，位等改卡在同形同分派的自比上（下一条断言）。
    assert torch.allclose(batched, per, atol=1e-6, rtol=1e-5), (
        f"{route}: 批与逐样本量级差 {max_abs!r} 超出 1e-6/1e-5 口径 → 批内头/样本被串味了")
    assert torch.equal(fn(q, k, v), batched), f"{route}: 同进程、同形、同分派两次结果位等不同 → 真·非确定回归"
    print(f"[shape-pin] {route}: batched={tuple(batched.shape)} loop={tuple(per.shape)} "
          f"max_abs={max_abs!r} bitwise_eq={bool(torch.equal(batched, per))}")


# ---------------------------------------------------------------- 钉③：升维的现场证据（广播形状）
def test_masked_fill_band_is_same_rank_as_scores():
    """根因①钉：喂给 `masked_fill` 的带状掩码必须与 (heads,T,T) 分数同秩——(1,1,T,T) 会按广播升维。"""
    band = LA.window_band(SEQ, WINDOW)
    assert tuple(band.shape) == (1, 1, SEQ, SEQ), "window_band 的对外形状是 (1,1,T,T)（sliding_visible/主干按此消费），不许改"
    scores = torch.zeros(HEADS, SEQ, SEQ)
    promoted = scores.masked_fill(~band, float("-inf"))
    assert promoted.dim() == scores.dim() + 1, (
        f"广播升维演示失效（torch 行为变了？）：实得 {tuple(promoted.shape)}——本用例的说明要随之改写")
    same_rank = scores.masked_fill(~band[0], float("-inf"))
    assert tuple(same_rank.shape) == (HEADS, SEQ, SEQ), "band[0]=(1,T,T) 才是与分数同秩的正确喂法（attention.py 修复口径）"
