#!/usr/bin/env python3
"""L4_probe_nanfix — 交付件 `ascend/gradcheck_card.py` 「非有限值一律 FAIL」补闸的负例自证（host，不占卡）。

【被修的是什么】`_gradcheck_op` 原判据 `ok = abs_err <= ATOL + RTOL*ana_norm`，两侧同为 `+inf`
时 `inf <= inf` 为 **True**，而 `rel = inf/inf = nan` ⇒ 打出「GRADCHECK-<op> PASS max_rel=nan」
的**假绿**（窗四 V1 gdn_delta 原文）。解析梯度里只要有**一个** ±inf，`ana_norm` 与 `abs_err`
（`d = n − a` 含 −inf）就双双溢出 `+inf`，正中此坑。

【本件证什么（全部 host/cpu，注入式，不动产品件）】
  F-frozen : **冻结旧式**对照——用改前的 `ok` 表达式重算「注入 inf」样本 ⇒ 必须 = PASS（复现病灶），
             同一 (abs_err,ana_norm) 用**现行**判据 ⇒ 必须 = FAIL（证补闸生效）。
  N-green  : 八件 `inject=identity`（正常梯度）⇒ **全 PASS**（补闸不误伤好样本）。
  N-inf    : 八件注入 inf ⇒ **全 FAIL**（窗四病灶形态，补闸后必红）。
  N-oneinf : 单元素注入 +inf ⇒ **FAIL**（一个 inf 就够触发旧假绿 ⇒ 一个 inf 就该被补闸拦住）。
  N-nan    : 注入 nan ⇒ **FAIL**（nan 本就 FAIL，回归不破）。

退出码：全符合预期 0；任一不符 1。
跑法：cd release && .venv/bin/python ascend/kernels/reconcile/L4_probe_nanfix.py
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

from ascend.kernels import ascend_env  # noqa: E402
import gradcheck_card as gc  # noqa: E402

CPU = torch.device("cpu")
ATOL, RTOL = gc.ATOL, gc.RTOL     # 逐字沿用交付件阈值，不另造
_bad = {"n": 0}


def _check(label: str, cond: bool, detail: str) -> None:
    if not cond:
        _bad["n"] += 1
    print(f"L4-NANFIX {label} {'PASS' if cond else 'FAIL'} {detail}", flush=True)


def _inj_identity(g):
    return g


def _inj_inf(g):
    return torch.full_like(g, float("inf"))


def _inj_one_inf(g):
    g = g.clone()
    g.reshape(-1)[0] = float("inf")
    return g


def _inj_nan(g):
    return g * float("nan")


def _frozen_old_ok(abs_err: float, ana_norm: float) -> bool:
    """改前的 `ok` 表达式（冻结对照，仅本探针用；不参与产品件）。"""
    return bool(abs_err <= ATOL + RTOL * ana_norm)


def frozen_repro():
    """F-frozen：证明旧式在「两侧 +inf」时判 PASS，而现行判据判 FAIL。"""
    print("—— F-frozen 冻结旧式 vs 现行判据（abs_err=ana_norm=+inf）——", flush=True)
    inf = float("inf")
    old = _frozen_old_ok(inf, inf)
    new = bool(__import__("math").isfinite(inf) and (inf <= ATOL + RTOL * inf))
    rel = inf / max(inf, 1e-30)
    print(f"    (abs_err,ana_norm)=({inf},{inf}) rel={rel!r}", flush=True)
    _check("F-frozen-old", old is True, f"冻结旧式 ok={old}（应 True＝复现假绿病灶）")
    _check("F-frozen-new", new is False, f"现行判据 ok={new}（应 False＝补闸生效）")


def _run(op: str, inject):
    return gc.run_one(op, "cpu", no_run=False, device=CPU, n_dirs=2, inject=inject)


def n_green():
    print("—— N-green 八件正常梯度（identity）——", flush=True)
    for op in gc.OPS:
        r = _run(op, _inj_identity)
        _check(f"N-green[{op}]", r["verdict"] == "PASS" and r.get("finite") is True,
               gc._line(r, False) + f" finite={r.get('finite')}")


def n_inf():
    print("—— N-inf 八件注入 inf（窗四病灶形态）——", flush=True)
    for op in gc.OPS:
        r = _run(op, _inj_inf)
        _check(f"N-inf[{op}]", r["verdict"] == "FAIL" and r.get("finite") is False,
               gc._line(r, False) + f" finite={r.get('finite')} ana_norm={r.get('ana_norm')}")


def n_oneinf():
    print("—— N-oneinf 单元素 +inf ——", flush=True)
    r = _run("gdn_delta", _inj_one_inf)
    _check("N-oneinf[gdn_delta]",
           r["verdict"] == "FAIL" and r.get("finite") is False,
           gc._line(r, False) + f" finite={r.get('finite')} ana_norm={r.get('ana_norm')}")


def n_nan():
    print("—— N-nan 注入 nan ——", flush=True)
    r = _run("gdn_delta", _inj_nan)
    _check("N-nan[gdn_delta]",
           r["verdict"] == "FAIL" and r.get("finite") is False,
           gc._line(r, False) + f" finite={r.get('finite')}")


def main() -> int:
    ascend_env.reset()
    print("== L4 非有限补闸负例自证：gradcheck_card ==", flush=True)
    frozen_repro()
    n_green()
    n_inf()
    n_oneinf()
    n_nan()
    print(f"L4-NANFIX-DONE fail={_bad['n']}", flush=True)
    return 1 if _bad["n"] else 0


if __name__ == "__main__":
    sys.exit(main())
