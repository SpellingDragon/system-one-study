#!/usr/bin/env python3
"""reconcile/C2_probe_alias_shim.py — cpu-alias 轨的两处工装垫片能不能把 add_ln/attn_sw/gdn 接上。

事实（C2_probe_before.py 取证）：
  · add_ln  ascend 正文 → InternalError: Only one dynamic shared memory allocation is allowed
    （c 后端不允许多个 alloc_shared；910B 那份有 4 个 UB 缓冲）
  · attn_sw / gdn / rope 正文里的 `with T.Vector()` → AttributeError: cpu.language has no attribute 'Vector'
垫片口径：
  · `T.Vector()` → nullcontext（910B 的 AIV 向量作用域标记，c 后端无此层；正文算式与之无关）
  · `T.alloc_shared(shape, dt)` → `T.alloc_local(...)`（与 gemm_asc._alloc_tile 的既有分叉同口径：
    分的是"存储层归属"不是算式）
本探针只判"垫了之后能不能真编真跑"，不判数值。
"""
import contextlib
import os
import sys
import types

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

import torch
import tilelang
import tilelang.cpu.language as tcpu
from ascend.kernels import add_ln_asc, ascend_env, attn_sw_asc, gdn_asc, rope_asc

SHIMMED = []


def alias(shim=True):
    pkg = sys.modules.get("tilelang.ascend")
    if pkg is None:
        pkg = types.ModuleType("tilelang.ascend")
        pkg.__path__ = []
        sys.modules["tilelang.ascend"] = pkg
        setattr(tilelang, "ascend", pkg)
    sys.modules["tilelang.ascend.language"] = tcpu
    pkg.language = tcpu
    if shim:
        if not hasattr(tcpu, "Vector"):
            tcpu.Vector = lambda *a, **k: contextlib.nullcontext()
            SHIMMED.append("Vector→nullcontext")
        if not hasattr(tcpu, "_c2_orig_alloc_shared"):
            tcpu._c2_orig_alloc_shared = tcpu.alloc_shared
            tcpu.alloc_shared = lambda shape, dt, *a, **k: tcpu.alloc_local(shape, dt)
            SHIMMED.append("alloc_shared→alloc_local")
    ascend_env.compile_kwargs = lambda t: {"target": "c", "target_host": "c",
                                          "execution_backend": "cython"}
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda *a, **k: ascend_env.TILELANG
    ascend_env.reset()


def rel(got, ref):
    return ((got.float() - ref.float()).abs().max().item()
            / max(ref.float().abs().max().item(), 1e-12))


print(f"[shim] {SHIMMED}")

print("-" * 70)
print("[add_ln] 前向正文（4 个 UB 缓冲）")
alias()
x = torch.randn(33, 64, dtype=torch.float32)
res = torch.randn(33, 64, dtype=torch.float32) + 4.0
g = torch.randn(64, dtype=torch.float32)
bt = torch.randn(64, dtype=torch.float32)
y = torch.empty(33, 64, dtype=torch.float32)
h = torch.empty(33, 64, dtype=torch.float32)
try:
    spec = add_ln_asc._plan_fwd(x, res, g, bt, y, h, 1e-5, "ascend")
    ok = spec is not None and add_ln_asc._run((x, res, g, bt, y, h), spec)
    print(f"  plan={'ok' if spec else 'None'} run={ok} keys={sorted(ascend_env.compiled_keys())}")
    if ok:
        print(f"  blockers={list(ascend_env.blockers())}")
except Exception as exc:  # noqa: BLE001
    print(f"  EXC {type(exc).__name__}: {str(exc)[:240]}")

print("-" * 70)
print("[ln_bwd] 反向正文")
alias()
try:
    hh = torch.randn(33, 64, dtype=torch.float32)
    dy = torch.randn(33, 64, dtype=torch.float32)
    dx = torch.zeros_like(hh)
    dg = torch.zeros(64, dtype=torch.float32)
    db = torch.zeros(64, dtype=torch.float32)
    spec = add_ln_asc._plan_bwd(hh, g, dy, dx, dg, db, 1e-5, "ascend")
    ok = spec is not None and add_ln_asc._run((hh, g, dy, dx, dg, db), spec)
    print(f"  plan={'ok' if spec else 'None'} run={ok} keys={sorted(ascend_env.compiled_keys())}")
except Exception as exc:  # noqa: BLE001
    print(f"  EXC {type(exc).__name__}: {str(exc)[:240]}")

print("-" * 70)
print("[attn_sw] 正文（T.Vector + 2 UB）")
alias()
try:
    q = torch.randn(2, 16, 32, dtype=torch.float32)
    o = torch.empty(2, 16, 32, dtype=torch.float32)
    spec = attn_sw_asc.plan(q, q, q, o, 8, 32 ** -0.5, "ascend")
    ok = spec is not None and attn_sw_asc.run(q, q, q, o, spec)
    print(f"  plan={'ok' if spec else 'None'} run={ok} keys={sorted(ascend_env.compiled_keys())}")
    if ok:
        print(f"  rel vs _eager={rel(o, attn_sw_asc._eager(q, q, q, 8, 32 ** -0.5)):.3e}")
except Exception as exc:  # noqa: BLE001
    print(f"  EXC {type(exc).__name__}: {str(exc)[:240]}")

print("-" * 70)
print("[rope] 正文（T.Vector）")
alias()
try:
    qkv = torch.randn(8, 3, 2, 16, dtype=torch.float32)
    cs = torch.randn(8, 8, dtype=torch.float32)
    sn = torch.randn(8, 8, dtype=torch.float32)
    spec = rope_asc._plan(qkv, cs, sn, 2, 1, "ascend")
    ok = spec is not None and rope_asc._run(qkv, cs, sn, spec)
    print(f"  plan={'ok' if spec else 'None'} run={ok} keys={sorted(ascend_env.compiled_keys())}")
except Exception as exc:  # noqa: BLE001
    print(f"  EXC {type(exc).__name__}: {str(exc)[:240]}")

print("-" * 70)
print("[gdn] 正文（T.Vector + UB）")
alias()
try:
    h_, s_, dk_, dv_ = 2, 8, 16, 16
    qq = torch.randn(h_, s_, dk_, dtype=torch.float32) * .3
    kk = torch.randn(h_, s_, dk_, dtype=torch.float32) * .3
    vv = torch.randn(h_, s_, dv_, dtype=torch.float32)
    gg = -torch.rand(h_, s_, dk_, dtype=torch.float32)
    bb = torch.rand(h_, s_, dtype=torch.float32) * .5
    out = torch.empty(h_, s_, dv_, dtype=torch.float32)
    spec = gdn_asc.plan(qq, kk, vv, gg, bb, out, "ascend")
    ok = spec is not None and gdn_asc.run(qq, kk, vv, gg, bb, out, spec)
    print(f"  plan={'ok' if spec else 'None'} run={ok} keys={sorted(ascend_env.compiled_keys())}")
    if ok:
        print(f"  rel vs _eager={rel(out, gdn_asc._eager(qq, kk, vv, gg, bb)):.3e}")
except Exception as exc:  # noqa: BLE001
    print(f"  EXC {type(exc).__name__}: {str(exc)[:400]}")
print("=" * 70)
