#!/usr/bin/env python3
"""D3_probe_dbeta_order.py — P1-1z / D3 **下窗一枪**：`dbeta` 顺序/prefill 判决驱动。

**占卡件（本窗未执行、未在真机验证）** —— 本窗只把它写好备枪，真机判决留下窗。

判什么（同卡、同进程、单次发射序，一次跑完三件事）：
  ① **逐项判决**：clean 跑的六项 `dq/dk/dv/dbeta/dg/dh0` 各自 rel vs fp64 golden
     （与 `run_delta_card.py --target ascend --bwd` 同输入、同 golden），看是"只坏 dbeta"
     还是"多项坏"。
  ② **顺序/prefill 判决（同源假设 H-b）**：目标档前先插一趟"脏填趟"（同形状、不同内容），
     再跑同一 clean 目标；若 clean 的 `dbeta` **随前一趟内容变化** ⇒ 存在 stale 成分
     （同源假设**证实**）；若逐位不变 ⇒ 顺序无关（同源假设**证伪**）。
     - `none` ：不插（对照）
     - `dirty`：插一趟 seed=999 的反向（内容不同）
     - `zero` ：插一趟 V=K=Q=DO 全 0 的反向（≈ S3 真值表第二行"预清零"档）
  ③ **多档对照（共指同一 UB 区）**：`--variant ub` 与 `--variant pure` 各跑一遍 clean；
     若 `pure`（零 alloc_shared、Λ 在 GM）**过**而 `ub`（Λ/暂存全在 UB）**挂** ⇒ 缺陷
     **UB 专属**（共指 UB 面）；若两档同坏 ⇒ 非 UB 专属。

跑法（**下窗**）：
    # 卡上（单进程单卡一次）
    python3 ascend/kernels/reconcile/D3_probe_dbeta_order.py --dev npu --variant ub
    python3 ascend/kernels/reconcile/D3_probe_dbeta_order.py --dev npu --variant pure
    # host 自测（不占卡；cpu 轨只跑 pure）
    .venv/bin/python ascend/kernels/reconcile/D3_probe_dbeta_order.py --dev cpu --variant pure

真值表（见同目录 D3_NEXTSHOT_dbeta.md §2）：
    · clean 三跑 dbeta 逐位同 且 rel 仍 ~6e-1 ⇒ H-b 不适用 gdn（同源 **证伪**）
    · clean dbeta 随 prefill 内容变 ⇒ H-b 适用（同源 **证实**），且可借差异指认落点
    · ub 挂 / pure 过 ⇒ 缺陷 UB 专属（共指 UB 面）
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
HERE = os.path.dirname(os.path.abspath(__file__))                     # .../release/ascend/kernels/reconcile
RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))     # .../release
sys.path.insert(0, os.path.join(RELEASE, "ascend", "port910b", "attempts", "F"))
if RELEASE not in sys.path:
    sys.path.insert(0, RELEASE)

import numpy as np  # noqa: E402

H, TT, DK, DV, CORES = 2, 8, 8, 8, 1
DT = "float32"
BWD_ITEMS = ("dq", "dk", "dv", "dbeta", "dg", "dh0")
TOL_F32 = 2e-4

AP = argparse.ArgumentParser()
AP.add_argument("--dev", choices=("npu", "cpu"), default="npu")
AP.add_argument("--variant", choices=("ub", "pure"), default="ub")
AP.add_argument("--target", choices=("ascend", "cpu"), default="ascend",
                help="产品 home 的 target 口径（真机恒 ascend）")
A = AP.parse_args()


def relerr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)


def inputs(seed):
    """与 run_delta_card._bwd_inputs 同抽序（seed=11）；dirty/zero 另给。"""
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((H, TT, DK)).astype(DT)
    Kt = (rng.standard_normal((H, TT, DK)) * 0.3).astype(DT)
    V = rng.standard_normal((H, TT, DV)).astype(DT)
    BETA = (rng.random((H, TT)) * 0.5 + 0.25).astype(DT)
    G = (-rng.random((H, TT))).astype(DT)
    DO = rng.standard_normal((H, TT, DV)).astype(DT)
    return Q, Kt, V, BETA, G, DO


def zero_inputs():
    Q, Kt, V, BETA, G, DO = inputs(11)
    z = np.zeros_like
    return z(Q), z(Kt), z(V), BETA.copy(), G.copy(), z(DO)


def force_env(target):
    """镜像 run_delta_card._force_kernel_env：强制走内核路，不当 torch 回退。"""
    from ascend.kernels import ascend_env
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda *a, **k: ascend_env.TILELANG
    if target == "cpu":
        ascend_env.compile_kwargs = lambda t: {"target": "c", "target_host": "c",
                                               "execution_backend": "cython"}
    ascend_env.reset()


def to_dev(dev):
    if dev == "npu":
        import torch_npu  # noqa: F401
        import torch
        return lambda x: torch.from_numpy(np.ascontiguousarray(x)).npu()
    import torch
    return lambda x: torch.from_numpy(np.ascontiguousarray(x))


def one_bwd(ins, variant, dev):
    """发一次反向；交回 dict(item -> np.ndarray)。返回 (meas, keys)。"""
    from ascend.kernels import ascend_env, gdn_asc
    t = to_dev(dev)
    Q, Kt, V, BETA, G, DO = ins
    tq, tk, tv, tg, tb, tdo = (t(x) for x in (Q, Kt, V, G, BETA, DO))
    grads = gdn_asc.backward(tq, tk, tv, tg, tb, tdo, target=A.target,
                             return_dh0=True, variant=variant)
    if dev == "npu":
        import torch
        torch.npu.synchronize()
    # 交回序 = (dq, dk, dv, dg, dbeta, dh0)
    dq, dk, dv, dg, db, dh0 = (g.detach().cpu().numpy().astype(np.float64) for g in grads)
    meas = {"dq": dq, "dk": dk, "dv": dv, "dbeta": db, "dg": dg.sum(-1), "dh0": dh0}
    keys = [k for k in ascend_env.compiled_keys() if "gdnb[" in k]
    return meas, keys


def fp(arr):
    return hashlib.md5(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()[:8]


def golden(ins):
    import f_gdn_golden as gd
    Q, Kt, V, BETA, G0, DO = ins
    _O, HIST, _S = gd.fwd_exact(H, TT, DK, DV, Q, Kt, V, BETA, G0, np.zeros((H, DV, DK)))
    dqa, dka, dva, dba, dga, dh0a = gd.bwd_adjoint(H, TT, DK, DV, Q, Kt, V, BETA, G0,
                                                   HIST, DO, order=A.variant)
    return {"dq": dqa, "dk": dka, "dv": dva, "dbeta": dba, "dg": dga, "dh0": dh0a}


def main() -> int:
    force_env(A.target)
    clean = inputs(11)
    dirty = inputs(999)
    zero = zero_inputs()

    print("=" * 78)
    print(f"D3-ORDER dev={A.dev} variant={A.variant} target={A.target}")
    print("=" * 78)

    ref = golden(clean) if A.dev != "cpu" else golden(clean)

    runs = {}
    # ① 对照跑（不插）
    runs["none"] = one_bwd(clean, A.variant, A.dev)
    # ② dirty 预填（同形状不同内容）
    one_bwd(dirty, A.variant, A.dev)
    runs["dirty"] = one_bwd(clean, A.variant, A.dev)
    # ② zero 预填（≈ 预清零档）
    one_bwd(zero, A.variant, A.dev)
    runs["zero"] = one_bwd(clean, A.variant, A.dev)

    # ① 逐项判决（对最后一次 clean 跑）
    meas, keys = runs["zero"]
    print(f"gdnb keys = {keys or 'MISSING'}")
    for it in BWD_ITEMS:
        v = relerr(meas[it], ref[it])
        verdict = "PASS" if v <= TOL_F32 else "FAIL"
        print(f"ITEM {it} rel= {v:.3e} verdict= {verdict}")

    # ② order/prefill 判决（看 clean 三次的指纹是否一致）
    print("-" * 78)
    print("DBETA-FP  clean 三跑指纹（none / after-dirty / after-zero）:")
    fps = {pf: fp(runs[pf][0]["dbeta"]) for pf in ("none", "dirty", "zero")}
    for pf in ("none", "dirty", "zero"):
        print(f"  {pf:6s} dbeta-fp={fps[pf]}")
    same = len(set(fps.values())) == 1
    print(f"ORDER-VERDICT {'逐位同 ⇒ 顺序无关(H-b 证伪)' if same else '随 prefill 变 ⇒ 存在 stale(H-b 证实)'}")
    for it in BWD_ITEMS:
        f = {pf: fp(runs[pf][0][it]) for pf in ("none", "dirty", "zero")}
        if len(set(f.values())) != 1:
            print(f"  ORDER-DIFF item={it} fps={f}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
