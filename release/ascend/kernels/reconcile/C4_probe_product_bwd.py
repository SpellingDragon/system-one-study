#!/usr/bin/env python3
"""C4_probe_product_bwd.py — 探针：**产品 home**（`ascend/kernels/gdn_asc.py`）的反向能否走
cpu-alias 在 host 真编真跑，并给出六项梯度 vs fp64 golden 的 rel；同时打印 `compiled_keys()`。

背景：run_delta_card.py 现只判 **F 波执行件** fwd。待判的是训练步里那个 `gdnb` 模具
（`gdn_asc.plan_backward` 产键 `gdnb[ascend]|...`，只在 train_step 内"执行过"，从未逐元素判过）。
本探针验证「--target cpu --bwd 走产品 home」这条路可用（ub 档在 cpu 后端不可编 ⇒ 用 pure）。

本探针只读产品件、不写任何产品件；产物落 reconcile/。
"""
import contextlib
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))            # release/ascend/kernels/reconcile
ASCEND = os.path.dirname(os.path.dirname(HERE))              # release/ascend
RELEASE = os.path.dirname(ASCEND)                            # release
sys.path.insert(0, os.path.join(ASCEND, "port910b", "attempts", "F"))
if RELEASE not in sys.path:
    sys.path.insert(0, RELEASE)                              # `import ascend.kernels`

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
os.environ["SYS1_GDN_VARIANT"] = "pure"                      # cpu 后端单动态 shared ⇒ pure

import numpy as np  # noqa: E402
import torch  # noqa: E402
import tilelang  # noqa: E402
import tilelang.cpu.language as tcpu  # noqa: E402


def install_cpu_alias():
    proxy = types.ModuleType("tilelang.ascend.language")
    for n in dir(tcpu):
        if not n.startswith("__"):
            setattr(proxy, n, getattr(tcpu, n))
    missing = [n for n in ("Vector", "SimtVF", "Cube", "SimdVF") if not hasattr(proxy, n)]

    @contextlib.contextmanager
    def _noop(*a, **k):
        yield None

    for n in missing:
        setattr(proxy, n, _noop)
    sys.modules["tilelang.ascend.language"] = proxy
    setattr(tilelang.ascend, "language", proxy)
    return missing


def main():
    missing = install_cpu_alias()
    print(f"PROBE-PROD-ALIAS installed; shimmed={missing}")
    from ascend.kernels import ascend_env, gdn_asc
    import f_gdn_golden as gd

    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda *a, **k: ascend_env.TILELANG
    ascend_env.compile_kwargs = lambda t: {"target": "c", "target_host": "c",
                                           "execution_backend": "cython"}
    ascend_env.reset()

    H, TT, DK, DV = 2, 8, 8, 8
    rng = np.random.default_rng(11)
    Q = rng.standard_normal((H, TT, DK)).astype(np.float32)
    Kt = (rng.standard_normal((H, TT, DK)) * 0.3).astype(np.float32)
    V = rng.standard_normal((H, TT, DV)).astype(np.float32)
    BETA = (rng.random((H, TT)) * 0.5 + 0.25).astype(np.float32)
    Gt = (-rng.random((H, TT))).astype(np.float32)
    DO = rng.standard_normal((H, TT, DV)).astype(np.float32)

    tq, tk, tv, tb, tg, tdo = (torch.from_numpy(np.ascontiguousarray(x))
                               for x in (Q, Kt, V, BETA, Gt, DO))
    o = gdn_asc.forward(tq, tk, tv, tg, tb, out_dtype=torch.float32, target="ascend")
    grads = gdn_asc.backward(tq, tk, tv, tg, tb, tdo, target="ascend", return_dh0=True)
    dq, dk, dv, dg, db, dh0 = (g.detach().numpy().astype(np.float64) for g in grads)
    print("PROBE-PROD-RUN forward shape", tuple(o.shape), "grads", len(grads))
    print("PROBE-PROD-KEYS", list(ascend_env.compiled_keys()))
    print("PROBE-PROD-BLOCKERS", dict(ascend_env.blockers()))

    # fp64 golden（同一输入；g 为每 token 标量）
    O64, HIST64, _ = gd.fwd_exact(H, TT, DK, DV, Q, Kt, V, BETA, Gt, np.zeros((H, DV, DK)))
    dqa, dka, dva, dba, dga, dh0a = gd.bwd_adjoint(H, TT, DK, DV, Q, Kt, V, BETA, Gt,
                                                   HIST64, DO, order="pure")
    print(f"PROBE-PROD rel_o={gd.rel(o.detach().numpy(), O64):.3e}")
    for n, a, b in (("dq", dq, dqa), ("dk", dk, dka), ("dv", dv, dva),
                    ("dbeta", db, dba), ("dg(sum-ch)", dg.sum(-1), dga), ("dh0", dh0, dh0a)):
        print(f"PROBE-PROD {n:11s} rel={gd.rel(a, b):.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
