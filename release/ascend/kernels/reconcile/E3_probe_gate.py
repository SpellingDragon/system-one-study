#!/usr/bin/env python3
"""E3_probe_gate — 判「B 段 gdn_delta 的 `nan` 是不是探针输入分布不良态」的**量具良态核查**（host，不占卡）。

【要回答的一句话】`gradcheck_card.py::_build("gdn_delta")` 喂给递推门 `g` 的分布，会不会让
**参考/数值自身**在本次形状（h2/seq4/dk8）内爆出 inf/nan？——若会 ⇒ 探针缺陷；若不会 ⇒ 嫌疑转产品侧。

比对三处口径（task (2) 要求）：
  A. gradcheck_card `g = -rand*0.2 ∈ (-0.2,0]`，`β = rand*0.9+0.05 ∈ [0.05,0.95]`，seq=4
  B. bench_card V2 `_gate3`：`g = -rand ∈ (-1,0]`，`β = rand/dk`（递推收缩口径）
  C. run_delta_card C4 seed=11：`G = -rand ∈ (-1,0]`，`BETA = rand*0.5+0.25 ∈ [0.25,0.75]`
并给 V2 实证的**坏口径对照**（`g=randn` + seq=512）：证明"爆 nan"需要哪两件事同时成立。

判据（可复跑、只读产品件）：
  ① 三处口径的 `g` 都非正 ⇒ `exp(g) ≤ 1` ⇒ 递推**不自增长**；
  ② 用 gradcheck 本件实测输入跑 fp64 精确递推（`autograd_asc._gdn_torch`）⇒ 输出有限；
  ③ 累积衰减连乘 `∏_t exp(g_t)` 有界；④ host fp32 前向 vs fp64 参考 rel 有限。

跑法：cd release && .venv/bin/python ascend/kernels/reconcile/E3_probe_gate.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
ASCEND = os.path.join(RELEASE, "ascend")
for p in (RELEASE, ASCEND):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from ascend import autograd_asc as aa  # noqa: E402
from ascend.kernels import ascend_env  # noqa: E402
import gradcheck_card as gc  # noqa: E402

CPU = torch.device("cpu")
_ok = {"n": 0, "fail": 0}


def _check(label: str, cond: bool, detail: str) -> None:
    _ok["n" if cond else "fail"] += 1
    print(f"E3-GATE {label} {'PASS' if cond else 'FAIL'} {detail}", flush=True)


def _stats(name, t):
    t = t.detach().cpu()
    print(f"    {name}: shape={tuple(t.shape)} min={t.min():.4g} max={t.max():.4g} "
          f"mean={t.mean():.4g}", flush=True)
    return float(t.min()), float(t.max())


def part_a_gradcheck_dist():
    """A：gradcheck_card 本件实测输入（与 _build 同源，同 seed 同抽序）。"""
    print("—— A. gradcheck_card::_build('gdn_delta') 本件口径（cpu device）——", flush=True)
    torch.manual_seed(20241011)
    # 复刻 _build 的抽序（q,k,v randn 后 g,beta rand）
    q, k, v = gc._rn(CPU, 2, 4, 8), gc._rn(CPU, 2, 4, 8), gc._rn(CPU, 2, 4, 8)
    g = -torch.rand(2, 4, 8, dtype=torch.float32, device=CPU) * 0.2
    beta = torch.rand(2, 4, dtype=torch.float32, device=CPU) * 0.9 + 0.05
    gmin, gmax = _stats("g(gate)", g)
    bmin, bmax = _stats("beta", beta)
    _check("A-gate<=0", gmax <= 0.0, f"g.max={gmax:.4g} ≤ 0 ⇒ exp(g)≤1（非正衰减门）")
    return q, k, v, g, beta


def part_bc_other_dists(dk=8):
    """B/C：另两处口径的关键不变量（只列统计，不跑内核）。"""
    print("—— B. bench_card V2 `_gate3` 口径 ——", flush=True)
    gB = -torch.rand(4, 512, dk)
    gminB, gmaxB = _stats("g=_gate3", gB)
    print(f"    beta=rand/dk: ∈[0,{1.0/dk:.4g}]（递推收缩）", flush=True)
    _check("B-gate<=0", gmaxB <= 0.0, f"g.max={gmaxB:.4g} ≤ 0")
    print("—— C. run_delta_card C4 seed=11 口径 ——", flush=True)
    rng = np.random.default_rng(11)
    _ = rng.standard_normal((2, 8, 8)); _ = (rng.standard_normal((2, 8, 8)) * 0.3)
    _ = rng.standard_normal((2, 8, 8))
    BETA = rng.random((2, 8)) * 0.5 + 0.25
    G = -rng.random((2, 8))
    gminC, gmaxC = _stats("G", torch.as_tensor(G))
    _stats("BETA", torch.as_tensor(BETA))
    _check("C-gate<=0", gmaxC <= 0.0, f"G.max={gmaxC:.4g} ≤ 0")
    _check("ABC-same-sign", gmax <= 0 and gmaxB <= 0 and gmaxC <= 0,
           "三处 g/G 都非正 ⇒ 口径同族（gradcheck 并未用 randn 门）")


def part_d_v2_pathology():
    """V2 实证的**坏口径**：randn 门 + 长序列 ⇒ 递推在 **fp32**（探针实际位宽）里溢出。

    判据：`g=randn` 有一半为正 ⇒ `exp(g)>1` 越乘越大；seq=512 后连乘 ~1e20、状态 ~1e165
    （fp64 尚有限，但 **fp32 上限 3.4e38** ⇒ 溢出成 inf/nan）⇒ 参考自身坏掉 = `rel=nan`。
    """
    print("—— D. V2 坏口径对照：g=randn + seq=512（参考自身在 fp32 里溢出）——", flush=True)
    torch.manual_seed(0)
    h, s, dk, dv = 1, 512, 8, 8
    g = torch.randn(h, s, dk)
    q = torch.randn(h, s, dk)
    k = torch.randn(h, s, dk)
    v = torch.randn(h, s, dv)
    beta = torch.rand(h, s)
    prod = torch.cumprod(torch.exp(g), dim=1)
    out32 = aa._gdn_torch(q, k, v, g, beta)
    fin32 = bool(torch.isfinite(out32).all())
    print(f"    ∏_t exp(g_t) max={float(prod.abs().max()):.4g}; "
          f"fp32 递推输出: finite={fin32} inf={int(torch.isinf(out32).sum())} "
          f"nan={int(torch.isnan(out32).sum())}", flush=True)
    _check("D-badblows", not fin32,
           "randn门∩seq=512 ⇒ fp32 参考自身即 inf/nan（**这才是**会让 rel=nan 的探针缺陷形态）")
    print("    （对照 A 口径 g=-rand*0.2 且 seq=4：连乘≤1、输出有限 —— 见 E）", flush=True)


def part_e_reference_finite(q, k, v, g, beta):
    """E：用 A 的**实测输入**跑 fp64 精确递推（产品尺子）⇒ 输出有限 + 连乘有界。"""
    print("—— E. A 口径实测：fp64 参考输出 + 连乘放大（判『参考自身是否爆』）——", flush=True)
    prod = torch.cumprod(torch.exp(g.to(torch.float64)), dim=1)
    print(f"    ∏_t exp(g_t): max_abs={float(prod.abs().max()):.6g}（≤1 即递推不放大）",
          flush=True)
    out64 = aa._gdn_torch(q.to(torch.float64), k.to(torch.float64), v.to(torch.float64),
                          g.to(torch.float64), beta.to(torch.float64))
    finite = bool(torch.isfinite(out64).all())
    print(f"    fp64 ref out: shape={tuple(out64.shape)} max_abs={float(out64.abs().max()):.6g} "
          f"finite={finite}", flush=True)
    _check("E-ref-finite", finite and float(prod.abs().max()) <= 1.0 + 1e-9,
           f"参考输出有限={finite}，连乘≤1 ⇒ gradcheck 门分布**良态**（参考不会自身爆 nan）")
    # fp32 host 前向 vs fp64 参考
    out32 = aa.gdn_delta(q, k, v, g, beta, out_dtype=torch.float32, target="cpu")
    rel = float((out32.double() - out64).norm() / (out64.norm() + 1e-30))
    _check("E-fwd-rel", np.isfinite(rel) and rel < 1e-3,
           f"host fp32 前向 vs fp64 参考 rel={rel:.3e}（有限、小）")


def main() -> int:
    print("== E3 量具良态核查：gradcheck 的 gdn 门/β 分布 ==", flush=True)
    ascend_env.reset()
    q, k, v, g, beta = part_a_gradcheck_dist()
    global gmax
    gmax = float(g.max())
    part_bc_other_dists()
    part_d_v2_pathology()
    part_e_reference_finite(q, k, v, g, beta)
    print(f"E3-GATE-DONE pass={_ok['n']} fail={_ok['fail']}", flush=True)
    return 1 if _ok["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
