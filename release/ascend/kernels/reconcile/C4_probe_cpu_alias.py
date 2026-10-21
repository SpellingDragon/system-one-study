#!/usr/bin/env python3
"""C4_probe_cpu_alias.py — 探针：F 波 GDN delta bwd DSL 件在 host 上的 `cpu-alias` 可编/可跑性。

目的（为 run_delta_card.py --target cpu --bwd 选路）：
  判定 `attempts/F/f_gdn_delta_bwd_910b.py` 的 kernel 能否用 run_numerics 的 cpu-alias 手法
  （把 `tilelang.ascend.language` 指向 cpu 方言 + target="c"）在 host 真编真跑。

  ub 档实测：cpu 后端 `Only one dynamic shared memory allocation is allowed`（7 个 UB shared
  不受支持）⇒ ub 档在 cpu 后端不可编。本探针续测 pure 档（零 alloc_shared）是否可编可跑，
  并给出与 fp64 golden 的六项 rel。

本探针只读 F 件、不写任何产品件；产物落 reconcile/。
"""
import contextlib
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))          # release/ascend/kernels/reconcile
ASCEND = os.path.dirname(os.path.dirname(HERE))            # release/ascend
sys.path.insert(0, os.path.join(ASCEND, "port910b", "attempts", "F"))
os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")

import numpy as np  # noqa: E402
import tilelang  # noqa: E402
import tilelang.cpu.language as tcpu  # noqa: E402


def install_alias():
    """run_numerics::_prod_setup 的 cpu-alias 手法 + `Vector` no-op shim（cpu 方言无此 frame）。"""
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
    try:
        setattr(tilelang.ascend, "language", proxy)
    except Exception as e:  # noqa: BLE001
        print("setattr tilelang.ascend.language failed:", e)
    return missing


def main():
    missing = install_alias()
    print(f"PROBE-ALIAS installed; shimmed frames (absent in cpu方言) = {missing}")
    import f_gdn_delta_bwd_910b as mod
    import f_gdn_golden as gd
    print("PROBE-IMPORT F 件 OK")

    H, TT, DK, DV, CORES = 2, 8, 8, 8, 1
    DT = "float32"
    rng = np.random.default_rng(11)
    Q = rng.standard_normal((H, TT, DK)).astype(DT)
    Kt = (rng.standard_normal((H, TT, DK)) * 0.3).astype(DT)
    V = rng.standard_normal((H, TT, DV)).astype(DT)
    BETA = (rng.random((H, TT)) * 0.5 + 0.25).astype(DT)
    G = (-rng.random((H, TT))).astype(DT)
    H0 = (rng.standard_normal((H, DV, DK)) * 0.3).astype(DT)
    DO = rng.standard_normal((H, TT, DV)).astype(DT)
    HIST = gd.fwd_exact(H, TT, DK, DV, Q, Kt, V, BETA, G, H0)[1].astype(DT)
    ref = gd.bwd_adjoint(H, TT, DK, DV, Q, Kt, V, BETA, G, HIST, DO, order="pure")
    names = ["dQ", "dK", "dV", "dBETA", "dG", "dH0"]

    import torch
    for variant in ("pure", "ub"):
        try:
            fn = mod.gdn_delta_rule_bwd(H=H, T_LEN=TT, DK=DK, DV=DV, cores=CORES,
                                        dtype=DT, variant=variant, dstate="zero")
            k = tilelang.compile(fn, target="c")
        except Exception as e:  # noqa: BLE001
            print(f"PROBE-{variant}-COMPILE-C-FAIL {type(e).__name__}: {str(e)[:200]}")
            continue
        print(f"PROBE-{variant}-COMPILE-C-OK src_len={len(k.get_kernel_source())}")
        try:
            dev = lambda x: torch.from_numpy(np.ascontiguousarray(x))  # noqa: E731
            inps = [dev(x) for x in (Q, Kt, V, DO, BETA, G, HIST, np.zeros((H, DV, DK), DT))]
            outs = [torch.zeros_like(dev(x)) for x in
                    (Q, Kt, V, BETA, G, np.zeros((H, DV, DK), DT))]
            k(*inps, *outs)
            got = [o.numpy().astype(np.float64) for o in outs]
            for n, a, b in zip(names, got, ref):
                print(f"PROBE-{variant}-RUN {n} rel={gd.rel(a, b):.3e}")
            print(f"PROBE-{variant}-RUN-OK")
        except Exception as e:  # noqa: BLE001
            print(f"PROBE-{variant}-RUN-FAIL {type(e).__name__}: {str(e)[:300]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
