#!/usr/bin/env python3
"""A4_probe_addln.py — 原型探针：在**不改** run_numerics.py 的前提下，先把 addln_prod case 的
正文逻辑跑通（复用 run_numerics 的 `_prod_setup`/`_call`/`_route_key`/`relerr`/`_np` 与三档 track）。

用法（host）：
  cd release && .venv/bin/python ascend/kernels/reconcile/A4_probe_addln.py --track cpu-alias
判据与 run_numerics.num_addln_intrinsic 逐字同（r_h<1e-5 / r_y<3e-4 / r_y<r_unb-2e-4），
只把正文换成产品件 `add_ln_kernel.forward → add_ln_asc.add_ln_asc_impl`。
本文件是**用完即弃的探针**，不参与判决；判决面在 run_numerics.py 的 addln_prod case。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # .../release
PORT = os.path.join(RELEASE_ROOT, "ascend", "port910b")
for p in (RELEASE_ROOT, PORT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_numerics as R  # noqa: E402

EPS = 1e-5
REL_TOL_H = 1e-5
REL_TOL_Y = 3e-4
DISC_MIN_UNB = 2e-4

ADDLN_SHAPES = (
    (64, 512, "现 addln 档：rows=64=4×bm16、dim=512（与 num_addln_intrinsic 同形）"),
    (33, 512, "尾块：rows=33 不整除 bm=16 ⇒ 行号守卫真生效"),
)


def _relerr(a, b):
    return R.relerr(a, b)


def _sample(rows, dim, device):
    torch.manual_seed(0)
    X = torch.randn(rows, dim, device=device, dtype=torch.float32)
    Res = torch.randn(rows, dim, device=device, dtype=torch.float32) + 4.0
    G = torch.randn(dim, device=device, dtype=torch.float32)
    Bt = torch.randn(dim, device=device, dtype=torch.float32)
    return X, Res, G, Bt


def _cpu_alias_storage_shim():
    """cpu-alias 轨专用：把 UB 落点 `alloc_shared`→`alloc_local`（c 后端无 UB、只许一个 shared.dyn）。"""
    import types
    import tilelang.cpu.language as tcpu
    shim = types.ModuleType("tilelang.ascend.language")
    for nm in dir(tcpu):
        setattr(shim, nm, getattr(tcpu, nm))
    shim.alloc_shared = tcpu.alloc_local
    shim.__tilelang_dialect__ = "cpu"
    sys.modules["tilelang.ascend.language"] = shim
    sys.modules["tilelang.ascend"].language = shim
    return shim


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="auto", choices=("auto", "card", "cpu-alias"))
    ap.add_argument("--no-run", action="store_true")
    a = ap.parse_args()
    R.ARGS.track, R.ARGS.no_run, R.ARGS.only = a.track, a.no_run, "addln_prod"

    from ascend.kernels import ascend_env, add_ln_kernel
    dev, restore, skip, box = R._prod_setup()
    tk = R._track()
    if tk == "cpu-alias":
        _cpu_alias_storage_shim()
    seen, worst_h, worst_y, worst_unb = set(), 0.0, 0.0, None
    try:
        for rows, dim, note in ADDLN_SHAPES:
            X, Res, G, Bt = _sample(rows, dim, dev)
            res, only = R._call(skip, lambda: add_ln_kernel.forward(X, Res, G, Bt, target="ascend"))
            if only:
                seen.add(R._route_key("add_ln"))
                continue
            y, h = res
            xv, rv = R._np(X).astype(np.float32), R._np(Res).astype(np.float32)
            gv, bv = R._np(G).astype(np.float32), R._np(Bt).astype(np.float32)
            v = (xv + rv).astype(np.float32)
            mu = v.mean(axis=1, keepdims=True)
            dvb = (v - mu).astype(np.float32)
            var_b = (dvb * dvb).mean(axis=1, keepdims=True).astype(np.float32)
            var_u = (dvb * dvb).sum(axis=1, keepdims=True) / (dim - 1)
            ref_h = v
            ref_y = ((dvb * (1.0 / np.sqrt(var_b + EPS)).astype(np.float32)) * gv + bv).astype(np.float16)
            alt_y = ((dvb * (1.0 / np.sqrt(var_u + EPS)).astype(np.float32)) * gv + bv).astype(np.float16)
            got_y = R._np(y).astype(np.float32)
            got_h = R._np(h).astype(np.float32)
            r_h = _relerr(got_h, ref_h.astype(np.float32))
            r_y = _relerr(got_y, ref_y.astype(np.float32))
            r_unb = _relerr(got_y, alt_y.astype(np.float32))
            print(f"  [shape rows={rows} dim={dim}] r_h={r_h:.3e} r_y={r_y:.3e} r_unb={r_unb:.3e} "
                  f"y.dtype={y.dtype} h.dtype={h.dtype}")
            assert r_h < REL_TOL_H, f"Hout rel={r_h:.3e}（fp32 残差路应逐位同）"
            assert r_y < REL_TOL_Y, f"Y rel={r_y:.3e}"
            assert r_y < r_unb - DISC_MIN_UNB, f"有偏/无偏判别失败 r_y={r_y:.3e} r_unb={r_unb:.3e}"
            seen.add(R._route_key("add_ln"))
            worst_h = max(worst_h, r_h)
            worst_y = max(worst_y, r_y)
            worst_unb = r_unb if worst_unb is None else min(worst_unb, r_unb)
        if skip:
            assert len(seen) >= 1, "一件都没编成"
            print(f"Y-addln_prod-ASC-COMPILE-PASS keys={len(seen)} key={sorted(seen)[0]} "
                  f"src~{box['src']}B track=card(no-run) note=无设备⇒只判编译")
            return 0
        want = len({dim for _r, dim, _n in ADDLN_SHAPES})
        got_n = ascend_env.compile_count()
        assert got_n == want, f"换批复用失败：compile_count={got_n} ≠ distinct(dim)={want}"
        print(f"Y-addln_prod-PROD-PASS rel_y={worst_y:.2e} rel_h={worst_h:.2e} "
              f"unb_disc={worst_unb:.2e} reuse={got_n}/{want} shapes={len(ADDLN_SHAPES)} track={tk}")
        return 0
    finally:
        restore()


if __name__ == "__main__":
    sys.exit(main())
