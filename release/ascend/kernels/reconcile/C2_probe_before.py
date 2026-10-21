#!/usr/bin/env python3
"""reconcile/C2_probe_before.py — C2 改动前的 fp16 现状取证（cpu-alias 轨真编真跑）。

只回答一件事：fp16 生产形状今天走到 *_asc 的 ascend 正文时，是崩、是静默错数据、还是根本没开模。
"""
import os
import sys
import types

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
RELEASE = "$REPO_ROOT/release"
sys.path.insert(0, RELEASE)

import torch
import tilelang
import tilelang.cpu.language as tcpu
from ascend.kernels import (add_ln_asc, ascend_env, attn_sw_asc, gemm_asc,
                            gemm_bwd_dw_asc, gemm_kernel, letter_readout_asc,
                            lora_asc, rope_asc, gdn_asc)


def alias():
    """run_numerics._prod_setup 的 cpu-alias 分支（两处一起指 + compile_kwargs 翻 c 后端）。"""
    pkg = sys.modules.get("tilelang.ascend")
    if pkg is None:
        pkg = types.ModuleType("tilelang.ascend")
        pkg.__path__ = []
        sys.modules["tilelang.ascend"] = pkg
        setattr(tilelang, "ascend", pkg)
    sys.modules["tilelang.ascend.language"] = tcpu
    pkg.language = tcpu
    ascend_env.compile_kwargs = lambda t: {"target": "c", "target_host": "c",
                                           "execution_backend": "cython"}
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda *a, **k: ascend_env.TILELANG
    ascend_env.reset()


def rel(got, ref):
    num = (got.float() - ref.float()).abs().max().item()
    den = max(ref.float().abs().max().item(), 1e-12)
    return num / den


def keys():
    return sorted(ascend_env.compiled_keys())


def blockers():
    return dict(ascend_env.blockers())


print("=" * 78)
print("[1] gemm linear：fp16 A/W 走生产入口（gemm_kernel.forward, target=ascend）")
alias()
torch.manual_seed(0)
m, n, k = 32, 64, 64
A = torch.randn(m, k, dtype=torch.float16)
W = torch.randn(n, k, dtype=torch.float16)
b = torch.randn(n, dtype=torch.float32)
try:
    got = gemm_kernel.forward(A, W, b, "none", torch.float16, "ascend")
    ref = (A.float() @ W.float().T + b).to(torch.float16)
    print(f"    rel={rel(got, ref):.3e}  keys={keys()}  blockers={list(blockers())}")
    print(f"    got[:2,:3]={got[0,:3].tolist()}  ref[:2,:3]={ref[0,:3].tolist()}")
except Exception as exc:
    print(f"    EXC {type(exc).__name__}: {str(exc)[:300]}")

print("-" * 78)
print("[1'] 直接追正文看声明位宽（绕过入口）：gemm_scalar_body 的 tensor 声明吃什么 dtype")
alias()
kw = {"n": n, "k": k, "bm": gemm_asc.BLOCK_M, "bn": gemm_asc._largest_block(n),
      "bk": gemm_asc._largest_block(k), "act_mode": 0}
T = ascend_env.dialect("ascend")
prog = gemm_asc.gemm_scalar_body(T, A, W, b, torch.zeros(m, n, dtype=torch.float16), **kw)
src = str(prog)
import re
print("    声明行：", [ln.strip() for ln in src.splitlines() if "A" in ln and "m, k" in ln][:3])

print("=" * 78)
print("[2] dW：fp16 dY/A + fp32 GW 走生产 plan/run")
alias()
dY = torch.randn(m, n, dtype=torch.float16)
Ak = torch.randn(m, k, dtype=torch.float16)
from ascend.kernels import gemm_bwd_dw_kernel
try:
    gw = gemm_bwd_dw_kernel.backward(dY, Ak, out_dtype=torch.float32, target="ascend")
    ref = dY.float().T @ Ak.float()
    print(f"    rel={rel(gw, ref):.3e}  keys={keys()}  blockers={list(blockers())}")
except Exception as exc:
    print(f"    EXC {type(exc).__name__}: {str(exc)[:300]}")

print("=" * 78)
print("[3] rope：fp16 packed qkv 走生产 forward（target=ascend）")
alias()
tokens, heads, dim = 8, 2, 16
qkv = torch.randn(tokens, 3, heads, dim, dtype=torch.float16)
cos = torch.randn(tokens, dim // 2, dtype=torch.float16)
sin = torch.randn(tokens, dim // 2, dtype=torch.float16)
try:
    out = rope_asc.forward(qkv, cos, sin, rope_asc.ROTATE_SLOTS, "ascend")
    print(f"    keys={keys()}  blockers={list(blockers())}  → {'内核' if keys() else 'eager（未内核化）'}")
    print(f"    _plan(fp16) 返回={'None(拒)' if rope_asc._plan(qkv, cos, sin, 2, 1, 'ascend') is None else 'spec'}")
except Exception as exc:
    print(f"    EXC {type(exc).__name__}: {str(exc)[:300]}")

print("=" * 78)
print("[4] add_ln / readout / attn_sw / gdn / lora：fp16 入口是否把张量物化成 fp32 再进内核")
alias()
x = torch.randn(33, 64, dtype=torch.float16)
res = torch.randn(33, 64, dtype=torch.float16)
g = torch.randn(64, dtype=torch.float32)
bt = torch.randn(64, dtype=torch.float32)
y, h = add_ln_asc.forward(x, res, g, bt, 1e-5, torch.float16, "ascend")
print(f"    add_ln  keys={keys()}  y.dtype={y.dtype} h.dtype={h.dtype}")

alias()
rows = torch.randn(12, 16, dtype=torch.float16)
ids = torch.tensor([3, 0, 7, 3, 11, 5], dtype=torch.int32)
o = letter_readout_asc.forward(rows, ids, torch.float16, "ascend")
print(f"    readout keys={keys()}  out.dtype={o.dtype} bitexact={torch.equal(o, rows[ids.long()])}")

alias()
q = torch.randn(2, 16, 32, dtype=torch.float16)
o2 = attn_sw_asc.forward(q, q.clone(), q.clone(), 8, None, torch.float16, "ascend")
print(f"    attn_sw keys={keys()}  out.dtype={o2.dtype}")

alias()
heads_, seq, dk_, dv_ = 2, 8, 16, 16
qq = torch.randn(heads_, seq, dk_, dtype=torch.float16)
oo = gdn_asc.forward(qq, qq.clone(), torch.randn(heads_, seq, dv_, dtype=torch.float16),
                    -torch.rand(heads_, seq, dk_, dtype=torch.float16),
                    torch.rand(heads_, seq, dtype=torch.float16) * 0.5,
                    torch.float16, "ascend")
print(f"    gdn     keys={keys()}  out.dtype={oo.dtype}")

alias()
xa = torch.randn(8, 32, dtype=torch.float16)
la = torch.randn(16, 32, dtype=torch.float16)
lb = torch.randn(64, 16, dtype=torch.float16)
lo = lora_asc.apply(xa, la, lb, 2.0, None, torch.float16, "ascend")
print(f"    lora    keys={keys()}  out.dtype={lo.dtype}")
print("=" * 78)
