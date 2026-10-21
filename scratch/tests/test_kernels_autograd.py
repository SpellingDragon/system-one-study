"""kernels.autograd 的梯度正确性验收：每一路可微算子都与纯 torch 自动微分对拍。

【做什么】证明 TileLang 内核包进求导图后，前向数值与反向梯度和 eager 路径一致（fp16 容差内），
并确认真走了内核（编译键在场），回退路径同样成立。
【怎么做】① 标量线性泛函 loss=Σ(out·固定权) 下比较各输入的 .grad；② rope 的参照用可导的
   rope_ref；attn 的参照用 attn_sw_ref；add_ln 参照 h=x+res 后接 F.layer_norm 的两步写法；
③ 设备双档：CPU 走回退闭式（逻辑正确性），MPS 走内核（@mps，并断言 gemm_dw 编译键在场，
   防"回退冒充内核"假绿）。
【为什么】反向若有一路算错，前向 loss 看不出来、训练悄悄发散——梯度对拍是唯一能提前
抓住它的门。被否方案：只测前向数值——反向闭式是本文件新写的主要数学，恰是最危险处。
"""
from __future__ import annotations

import pytest
import torch

from sys1.kernels import autograd, backends
from sys1.testing.torch_ref.attn_sw_ref import attn_sw_ref
from sys1.testing.torch_ref.rope_ref import rope_angle_tables, rope_ref

TOL = 2e-2  # fp16 预算内的对拍门（与 p1-04 前向对拍同口径）


DEVS = ["cpu"] + ([pytest.param("mps", marks=pytest.mark.mps)] if torch.backends.mps.is_available() else [])


def _dev(name):
    return torch.device(name)


def _linfun(out, seed):
    # 随机数在 CPU 生好再搬到目标设备：torch 的 mps Generator 支持度存疑，不拿它赌崩溃
    g = torch.randn(out.shape, generator=torch.Generator().manual_seed(seed)).to(out)
    return (out * g).sum()


@pytest.mark.parametrize("device", DEVS)
def test_linear_grad_matches_torch(device):
    """linear 的 dA/dW/db 三路梯度与 eager 全自动微分对拍。"""
    dev = _dev(device)
    torch.manual_seed(0)
    A0 = torch.randn(64, 32, dtype=torch.float16, device=dev)
    W0 = torch.randn(48, 32, dtype=torch.float16, device=dev)
    b0 = torch.randn(48, dtype=torch.float32, device=dev)

    def run(fn, half: bool):
        cast = (lambda t: t.half()) if half else (lambda t: t.float())
        A, W, b = cast(A0.clone()).requires_grad_(), cast(W0.clone()).requires_grad_(), b0.clone().requires_grad_()
        out = fn(A, W, b)
        _linfun(out, 7).backward()
        return A.grad, W.grad, b.grad

    # 参照取 fp32 数学真值：torch 的 linear 不接受 fp16 输入配 fp32 偏置（dtype 契约），
    # 而本件的设计就是 fp32 偏置 + fp32 累加——故拿高精度 eager 当尺，在 2e-2 预算内比。
    a1, w1, g1 = run(lambda A, W, b: torch.nn.functional.linear(A, W, b), half=False)
    a2, w2, g2 = run(autograd.linear, half=True)
    for ref, got, name in ((a1, a2, "dA"), (w1, w2, "dW"), (g1, g2, "dbias")):
        err = (got.float() - ref.float()).abs().max().item()
        rel = err / max(ref.float().abs().max().item(), 1e-6)
        assert rel < TOL, f"{name} 梯度 rel={rel:.3e}"
    if device == "mps" and backends.active_backend(dev) == backends.TILELANG:
        assert any("gemm_dw|" in k for k in backends.compiled_keys()), "dW 未真走方言内核（假绿）"


@pytest.mark.parametrize("device", DEVS)
def test_rope_grad_orthogonal_roundtrip(device):
    """rope 反向=负角回转：对拍参照取可导的 rope_ref（fp32 拷贝语义）。"""
    dev = _dev(device)
    torch.manual_seed(1)
    tokens, heads, dim = 32, 2, 16
    qkv0 = torch.randn(tokens, 3, heads, dim, dtype=torch.float16, device=dev)
    cos, sin = rope_angle_tables(tokens, dim // 2, device=dev)

    def run(fn):
        x = qkv0.clone().requires_grad_()
        out = fn(x, cos, sin)
        _linfun(out, 3).backward()
        return x.grad

    ref = run(lambda x, c, s: rope_ref(x, c, s))
    got = run(autograd.rope)
    rel = ((got.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
    assert rel < TOL, f"rope 梯度 rel={rel:.3e}"


@pytest.mark.parametrize("device", DEVS)
def test_add_ln_grad_matches_torch(device):
    """add_ln 融合件的双输出梯度（dx/dres/dg/db）对拍两步 eager 写法。"""
    dev = _dev(device)
    torch.manual_seed(2)
    x0 = torch.randn(24, 32, dtype=torch.float16, device=dev)
    r0 = torch.randn(24, 32, dtype=torch.float32, device=dev)
    g0 = torch.randn(32, dtype=torch.float32, device=dev)
    b0 = torch.randn(32, dtype=torch.float32, device=dev)

    def refs():
        x, r, g, b = x0.clone().requires_grad_(), r0.clone().requires_grad_(), g0.clone().requires_grad_(), b0.clone().requires_grad_()
        h = x.float() + r
        y = torch.nn.functional.layer_norm(h, (32,), g, b, 1e-5).to(torch.float16)
        l1 = _linfun(y, 5)
        l2 = _linfun(h.to(torch.float16), 6)          # h 也另有来账（主干直传）
        (l1 + l2).backward()
        return x.grad, r.grad, g.grad, b.grad

    def mine():
        x, r, g, b = x0.clone().requires_grad_(), r0.clone().requires_grad_(), g0.clone().requires_grad_(), b0.clone().requires_grad_()
        y, h = autograd.add_ln(x, r, g, b, eps=1e-5)
        (_linfun(y, 5) + _linfun(h.to(torch.float16), 6)).backward()   # 双输出合一次回传，免二次过图
        return x.grad, r.grad, g.grad, b.grad

    for ref, got, name in zip(refs(), mine(), ("dx", "dres", "dg", "db")):
        rel = ((got.float() - ref.float()).abs().max()
               / max(ref.float().abs().max().item(), 1e-6)).item()
        assert rel < TOL, f"add_ln {name} 梯度 rel={rel:.3e}"


@pytest.mark.parametrize("device", DEVS)
def test_attn_sw_grad_matches_ref(device):
    """attn_sw 的 dq/dk/dv 对拍可导参考实现（整窗=全因果档）。"""
    dev = _dev(device)
    torch.manual_seed(3)
    B, H, T, D = 2, 2, 32, 16
    q0, k0, v0 = (torch.randn(B, H, T, D, dtype=torch.float16, device=dev) for _ in range(3))

    def run(fn):
        q, k, v = q0.clone().requires_grad_(), k0.clone().requires_grad_(), v0.clone().requires_grad_()
        out = fn(q, k, v)
        _linfun(out, 9).backward()
        return q.grad, k.grad, v.grad

    refs = run(lambda q, k, v: attn_sw_ref(q, k, v, window=T))
    gots = run(lambda q, k, v: autograd.attn_sw(q, k, v, window=T))
    for ref, got, name in zip(refs, gots, ("dq", "dk", "dv")):
        rel = ((got.float() - ref.float()).abs().max()
               / max(ref.float().abs().max().item(), 1e-6)).item()
        assert rel < TOL, f"attn_sw {name} 梯度 rel={rel:.3e}"


def test_dw_entry_rejects_shape_violation():
    """形状违约必须当场报错，不静默算错形状（入口校验在回退之前）。"""
    A = torch.randn(8, 16, dtype=torch.float16)
    dY_bad = torch.randn(9, 24, dtype=torch.float16)   # 行数与 A 不同步
    with pytest.raises(ValueError):
        autograd.gemm_bwd_dw_kernel.backward(dY_bad, A)


@pytest.mark.mps
def test_tiny_decoder_dual_path_forward_and_grad():
    """模型级双路对拍：同一份权重下 off 与 tilelang 的 forward/loss/全参数梯度必一致。

    只测单算子不够——接线处（形状变换/精度交换/回退条件）才是错漏高发地；
    同 seed 构造两实例权重完全相同，梯度拉平成向量一次比完。

    白话：同一套旋钮、同一道题，让“手工做法”和“上模具做法”各算一遍再对答案——
    不仅比结果，还比“每个旋钮该往哪拧”的那张清单，两张都对上才算接线没接错。
    """
    from sys1.model import Decoder, ModelConfig

    cfg = ModelConfig(d=64, L=2, heads=4, ctx=64, vocab=128, seed=3)
    ids = torch.randint(1, 128, (2, 32), device="mps")
    got = {}
    for backend in ("off", "tilelang"):
        m = Decoder(cfg, kernel_backend=backend).to("mps")
        out = m(ids)
        loss = (m.lm_head(out.float()).float() ** 2).mean()
        m.zero_grad()
        loss.backward()
        gflat = torch.cat([p.grad.reshape(-1).float() for p in m.parameters() if p.grad is not None])
        got[backend] = (out.float(), loss.item(), gflat.cpu())
    for idx, name in ((0, "forward"), (2, "grad")):
        a, b = got["off"][idx], got["tilelang"][idx]
        rel = ((b - a).abs().max() / a.abs().max().clamp_min(1e-6)).item()
        assert rel < TOL, f"{name} 双路偏差 rel={rel:.3e}"
    rel_loss = abs(got["tilelang"][1] - got["off"][1]) / max(abs(got["off"][1]), 1e-6)
    assert rel_loss < TOL, f"loss 双路偏差 rel={rel_loss:.3e}"
    assert any("gemm_dw|" in kk for kk in backends.compiled_keys()), "训练链未真走 dW 内核（假绿）"
