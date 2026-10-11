#!/usr/bin/env python3
"""B3_probe_explore — 探索件（非交付）：为 gradcheck_card 选形状/ε、摸清键与 blocker 边界。

跑法：cd release && .venv/bin/python ascend/kernels/reconcile/B3_probe_explore.py [--eps a,b,c] [--op x]
仅 host(cpu target)。只读产品件，不改任何东西。
"""
from __future__ import annotations

import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if RELEASE not in sys.path:
    sys.path.insert(0, RELEASE)

import torch  # noqa: E402

from ascend.autograd_asc import count_fallback_events  # noqa: E402
from ascend.kernels import ascend_env  # noqa: E402
from ascend import autograd_asc as ops  # noqa: E402

T = "cpu"


def _snap():
    return set(ascend_env.compiled_keys()), dict(ascend_env.blockers())


def _delta(before):
    k0, b0 = before
    nk = [k for k in ascend_env.compiled_keys() if k not in k0 and k.startswith(f"{T}|")]
    nb = {k: v for k, v in ascend_env.blockers().items() if k not in b0}
    return nk, nb


def _build(op):
    torch.manual_seed(0)
    if op == "linear":
        A, W, bias = torch.randn(6, 16), torch.randn(16, 16), torch.randn(16)
        f = lambda A, W, b: ops.linear(A, W, b, out_dtype=torch.float32, target=T)
        ins, dy = [A, W, bias], torch.randn(6, 16)
    elif op == "add_ln":
        x, r, gg, bb = torch.randn(6, 16), torch.randn(6, 16), torch.randn(16), torch.randn(16)
        f = lambda x, r, gg, bb: ops.add_ln(x, r, gg, bb, eps=1e-5, out_dtype=torch.float32, target=T)[0]
        ins, dy = [x, r, gg, bb], torch.randn(6, 16)
    elif op == "rope":
        qkv, cos, sin = torch.randn(4, 3, 2, 8), torch.randn(4, 4), torch.randn(4, 4)
        f = lambda q: ops.rope(q, cos, sin, rotate_slots=(0, 1), target=T)
        ins, dy = [qkv], torch.randn(4, 3, 2, 8)
    elif op == "attn_sw":
        q, k, v = torch.randn(2, 5, 16), torch.randn(2, 5, 16), torch.randn(2, 5, 16)
        f = lambda q, k, v: ops.attn_sw(q, k, v, 4, scale=16 ** -0.5, out_dtype=torch.float32, target=T)
        ins, dy = [q, k, v], torch.randn(2, 5, 16)
    elif op == "gdn_delta":
        q, k, v = torch.randn(2, 4, 8), torch.randn(2, 4, 8), torch.randn(2, 4, 8)
        gg, be = -torch.rand(2, 4, 8) * 0.2, torch.rand(2, 4) * 0.9 + 0.05
        f = lambda q, k, v, gg, be: ops.gdn_delta(q, k, v, gg, be, out_dtype=torch.float32, target=T)
        ins, dy = [q, k, v, gg, be], torch.randn(2, 4, 8)
    elif op == "gdn_conv":
        x, w, b = torch.randn(2, 8, 6), torch.randn(8, 4), torch.randn(8)
        f = lambda x, w, b: ops.gdn_conv(x, w, b, out_dtype=torch.float32, target=T)
        ins, dy = [x, w, b], torch.randn(2, 8, 6)
    elif op == "readout":
        rows = torch.randn(8, 16)
        ids = torch.tensor([3, 0, 3, 7, 5], dtype=torch.int64)
        f = lambda rows: ops.readout(rows, ids, out_dtype=torch.float32, target=T)
        ins, dy = [rows], torch.randn(5, 16)
    elif op == "lora_apply":
        x, a, b = torch.randn(6, 16), torch.randn(16, 16) * 0.2, torch.randn(16, 16) * 0.2
        f = lambda x, a, b: ops.lora_apply(x, a, b, scaling=0.5, out_dtype=torch.float32, target=T)
        ins, dy = [x, a, b], torch.randn(6, 16)
    else:
        raise KeyError(op)
    return f, ins, dy


OPS = ["linear", "add_ln", "rope", "attn_sw", "gdn_delta", "gdn_conv", "readout", "lora_apply"]


def main():
    eps_list = [0.05, 0.02, 0.01, 5e-3, 2e-3, 1e-3]
    if "--eps" in sys.argv:
        eps_list = [float(x) for x in sys.argv[sys.argv.index("--eps") + 1].split(",")]
    only = sys.argv[sys.argv.index("--op") + 1] if "--op" in sys.argv else None
    K = 6
    print(f"== explore target={T} eps_list={eps_list} K={K} fp64loss=on ==")
    for op in OPS:
        if only and op != only:
            continue
        ascend_env.reset()
        f, ins_raw, dy = _build(op)
        dy = dy.double()
        before = _snap()
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            leaves = [t.clone().detach().requires_grad_(True) for t in ins_raw]
            t0 = time.time()
            out = f(*leaves)
            grads = torch.autograd.grad((out.double() * dy).sum(), leaves)
            f(*[t.detach() for t in leaves])            # warmup 命中缓存
        ev = count_fallback_events([str(x.message) for x in wlist])
        nk, nb = _delta(before)
        keys = ", ".join(sorted(k.split("|", 1)[1] if "|" in k else k for k in nk))
        print(f"[{op}] keys=[{keys}] blockers={list(nb)} fallback={ev} wall={time.time() - t0:.2f}s")
        with torch.no_grad():
            base = [t.detach() for t in leaves]
            for eps in eps_list:
                worst = 0.0
                for kk in range(K):
                    torch.manual_seed(100 + kk)
                    for li in range(len(base)):
                        v = torch.randn_like(base[li])
                        lp = f(*[t if i != li else (t + eps * v) for i, t in enumerate(base)])
                        lm = f(*[t if i != li else (t - eps * v) for i, t in enumerate(base)])
                        gn = ((lp.double() * dy).sum() - (lm.double() * dy).sum()) / (2 * eps)
                        ga = (grads[li].double() * v.double()).sum()
                        rel = float((gn - ga).abs() / max(ga.abs().item(), 1e-30))
                        worst = max(worst, rel)
                print(f"    eps={eps:<8g} worst_rel_proj={worst:.3e}")


if __name__ == "__main__":
    main()
