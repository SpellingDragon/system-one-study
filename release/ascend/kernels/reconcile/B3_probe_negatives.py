#!/usr/bin/env python3
"""B3_probe_negatives — 交付件 `ascend/gradcheck_card.py` 的三个负例自证（host，不占卡）。

三个负例（对应 P1-1y 的「防假绿」要求）：
  N-a 桩 `backend_available=False` ⇒ 八件**全 INVALID**（eager 伪装被 KERNEL-USED 三证拦住）；
  N-b 桩 `gemm_asc.plan=None`（plan 拒形状）⇒ linear 该件 INVALID（`reason=no-kernel`），**不冒充 FAIL**；
  N-c 人为错梯度（解析参考 ×1.001）⇒ linear **FAIL**（数值 vs 解析真能咬住 ×0.1% 偏差）。

跑法：cd release && .venv/bin/python ascend/kernels/reconcile/B3_probe_negatives.py
只读产品件 + 交付件，不改任何东西（monkeypatch 现场生效、当场还原）。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
ASCEND = os.path.join(RELEASE, "ascend")
for p in (RELEASE, ASCEND):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from ascend.kernels import ascend_env, gemm_asc  # noqa: E402
import gradcheck_card as gc  # noqa: E402

CPU = torch.device("cpu")
_ok = {"n": 0, "fail": 0}


def _check(label: str, cond: bool, detail: str) -> None:
    _ok["n" if cond else "fail"] += 1
    print(f"B3-NEG {label} {'PASS' if cond else 'FAIL'} {detail}", flush=True)


def _run_all(target="cpu", n_dirs=1, inject=None):
    return [gc.run_one(op, target, no_run=False, device=CPU, n_dirs=n_dirs, inject=inject)
            for op in gc.OPS]


def neg_a_backend_unavailable():
    """桩 backend_available=False：内核一律不可用 ⇒ 八件应全 INVALID reason=no-kernel。"""
    print("—— N-a 桩 backend_available=False ——", flush=True)
    orig = ascend_env.backend_available
    ascend_env.backend_available = lambda *a, **k: False
    try:
        res = _run_all()
    finally:
        ascend_env.backend_available = orig
    for r in res:
        print("    " + gc._line(r, False), flush=True)
    n_inv = sum(1 for r in res if r["verdict"] == "INVALID")
    n_no_kernel = sum(1 for r in res if r["reason"] == "no-kernel")
    n_keys = sum(1 for r in res if r["keys"])
    _check("N-a", n_inv == len(gc.OPS) and n_no_kernel == len(gc.OPS) and n_keys == 0,
           f"invalid={n_inv}/{len(gc.OPS)} reason=no-kernel 全中 keys={n_keys}（应 0）")


def neg_b_plan_rejects_shape():
    """桩 gemm_asc.plan=None（plan 拒形状）：linear 前向静默落 eager ⇒ INVALID，不是 FAIL。"""
    print("—— N-b 桩 gemm_asc.plan=None（plan 拒形状） ——", flush=True)
    orig = gemm_asc.plan
    gemm_asc.plan = lambda *a, **k: None
    try:
        r = gc.run_one("linear", "cpu", no_run=False, device=CPU, n_dirs=1)
    finally:
        gemm_asc.plan = orig
    print("    " + gc._line(r, False), flush=True)
    _check("N-b", r["verdict"] == "INVALID" and r["reason"] == "no-kernel" and not r["keys"],
           f"linear verdict={r['verdict']} reason={r['reason']}（应 INVALID/no-kernel，非 FAIL）")


def neg_c_wrong_gradient():
    """解析参考 ×1.001：数值 vs 解析应咬住 ⇒ linear FAIL（证明对拍真在量数，不是空转）。"""
    print("—— N-c 人为错梯度（解析 ×1.001） ——", flush=True)
    r = gc.run_one("linear", "cpu", no_run=False, device=CPU, n_dirs=2,
                   inject=lambda g: g * 1.001)
    print("    " + gc._line(r, False), flush=True)
    _check("N-c", r["verdict"] == "FAIL",
           f"linear verdict={r['verdict']} max_rel={r['rel']:.3e}（应 FAIL，~1e-3 > rtol=1e-4）")


def main() -> int:
    print("== B3 负例自证：gradcheck_card ==", flush=True)
    neg_a_backend_unavailable()
    neg_b_plan_rejects_shape()
    neg_c_wrong_gradient()
    print(f"B3-NEG-DONE pass={_ok['n']} fail={_ok['fail']}", flush=True)
    return 1 if _ok["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
