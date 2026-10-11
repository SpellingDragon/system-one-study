#!/usr/bin/env python3
"""A4_probe_addln_compile.py — 探针：真身 `add_ln_asc_impl` 在 c 后端（cpu-alias 形态）能否编过。

三个变体：
  A) 原装：`tilelang.ascend.language` -> `tilelang.cpu.language`（run_numerics `_prod_setup` 手法）
  B) shim：在 A 之上把 `alloc_shared` 换名到 `alloc_local`（c 后端无 UB 语义）
  C) 对照组：readout 的 `gather_asc_impl`（已知可过）同法编译
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

import torch  # noqa: E402
import tilelang  # noqa: E402
import tilelang.cpu.language as tcpu  # noqa: E402

KW = {"target": "c", "target_host": "c", "execution_backend": "cython"}


def _install_alias(shim_local=False):
    if "tilelang.ascend" not in sys.modules:
        pkg = types.ModuleType("tilelang.ascend")
        pkg.__path__ = []
        sys.modules["tilelang.ascend"] = pkg
    if shim_local:
        shim = types.ModuleType("tilelang.ascend.language")
        for nm in dir(tcpu):
            setattr(shim, nm, getattr(tcpu, nm))
        shim.alloc_shared = tcpu.alloc_local          # c 后端没有 UB：shared.dyn→local
        shim.__tilelang_dialect__ = "cpu"
        lang = shim
    else:
        lang = tcpu
    sys.modules["tilelang.ascend.language"] = lang
    sys.modules["tilelang.ascend"].language = lang


def _try(label, fn):
    try:
        k = fn()
        src = len(str(k.get_kernel_source())) if hasattr(k, "get_kernel_source") else 0
        print(f"[{label}] OK src~{src}B")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[{label}] FAIL {type(e).__name__}: {str(e)[:180]}")
        return False


def main():
    from ascend.kernels import add_ln_asc, letter_readout_asc

    rows, dim, bm = 64, 512, 16

    def call_addln():
        X = torch.randn(rows, dim)
        Res = torch.randn(rows, dim)
        G = torch.randn(dim)
        Bt = torch.randn(dim)
        Y = torch.empty(rows, dim)
        H = torch.empty(rows, dim)
        prog = add_ln_asc.add_ln_asc_impl(X, Res, G, Bt, Y, H, dim, 1e-5, bm)
        return tilelang.compile(prog, out_idx=[], **KW)

    def call_readout():
        total, dim, ids = 16, 64, torch.arange(4, dtype=torch.long)
        r = torch.randn(total, dim)
        out = torch.empty(4, dim)
        prog = letter_readout_asc.gather_asc_impl(r, ids, out, dim)
        return tilelang.compile(prog, out_idx=[], **KW)

    for label, local in (("A原装cpu-alias", False), ("B shim shared→local", True)):
        _install_alias(local)
        _try(f"addln {label}", call_addln)
    _install_alias(False)
    _try("C readout cpu-alias", call_readout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
