#!/usr/bin/env python3
"""E3_probe_nanpath — 反推 `gradcheck_card.py` 判决行里 `max_rel=nan` 与 `max_rel=n/a` 的**代码路径**。

【为什么写这个】窗三 B 段两件 FAIL 形态不同（`gdn_delta max_rel=nan` / `gdn_conv max_rel=n/a`），
"n/a" 有歧义：可能是 FD 不可得、解析梯度为 None、或异常被吞。本探针在 **host（cpu）**上人工制造
这两种形态，逐字复现卡上判决行，从而把"哪种坏等于哪条代码路径"钉死（只读产品件 + 交付件，
monkeypatch 现场生效、当场还原，不改任何文件）。

三条自证：
  P-nan  解析梯度注入 nan（`inject=lambda g: g*nan`）⇒ 无异常、`max_rel=nan`
         ⇒ 证明 `max_rel=nan` = 「算完了，但梯度里有 nan/inf」，**不是**异常分支。
  P-na_ok  `_conv_silu_torch` 桩成抛 RuntimeError（模拟卡上 SetPrecisionMode 500001）
         ⇒ `max_rel=n/a` + `err=RuntimeError:…`
         ⇒ 证明 `max_rel=n/a` = 「run_one 走了 `exc is not None` 分支」（rel 恒为 None）。
  P-na_early 前向就抛（三证拿不到前向键）⇒ 判 **INVALID** 而非 `n/a`
         ⇒ 证明 `n/a` 只在「前向键已到手（内核已用）**之后**才抛」时出现。

跑法：cd release && .venv/bin/python ascend/kernels/reconcile/E3_probe_nanpath.py
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

# 必须用 `ascend.autograd_asc`（gradcheck_card 走的就是包名前缀那条 import），
# 否则顶层 `autograd_asc` 是**另一个模块对象**，patch 打不到产品实际调用的那个。
from ascend import autograd_asc as aa  # noqa: E402
from ascend.kernels import ascend_env, gdn_kernel  # noqa: E402
import gradcheck_card as gc  # noqa: E402

CPU = torch.device("cpu")
_ok = {"n": 0, "fail": 0}


def _check(label: str, cond: bool, detail: str) -> None:
    _ok["n" if cond else "fail"] += 1
    print(f"E3-NANPATH {label} {'PASS' if cond else 'FAIL'} {detail}", flush=True)


def p_nan_analytic():
    """解析侧注入 nan：应得 `FAIL max_rel=nan`，且 `err` 为空（无异常）。"""
    print("—— P-nan 解析梯度注入 nan（linear, host）——", flush=True)
    nan = float("nan")
    r = gc.run_one("linear", "cpu", no_run=False, device=CPU, n_dirs=1,
                   inject=lambda g: g * nan)
    line = gc._line(r, False)
    print("    " + line, flush=True)
    print(f"    · verdict={r['verdict']} rel={r['rel']!r} err={r['err']!r}", flush=True)
    _check("P-nan", r["verdict"] == "FAIL" and r["rel"] is not None
           and float(r["rel"]) != float(r["rel"]) and r["err"] is None,
           "max_rel 应为 nan（rel is not None ∧ isnan）且 err 为 None ⇒ nan 走『算完但含 nan』路")


def _raise_precision():
    raise RuntimeError(
        "SetPrecisionMode:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:175 "
        "NPU function error: at_npu::native::AclSetCompileopt(aclCompileOpt::ACL_PRECISION_MODE, "
        "precision_mode), error code is 500001")


def p_na_after_kernel():
    """前向用内核成功、反向抛异常：应得 `FAIL max_rel=n/a` + `err=…`（复刻卡上 gdn_conv）。"""
    print("—— P-na_ok 前向后、反向抛（gdn_conv, host）——", flush=True)
    orig = aa._conv_silu_torch
    aa._conv_silu_torch = lambda *a, **k: _raise_precision()
    try:
        r = gc.run_one("gdn_conv", "cpu", no_run=False, device=CPU, n_dirs=1)
    finally:
        aa._conv_silu_torch = orig
    line = gc._line(r, False)
    print("    " + line, flush=True)
    print(f"    · verdict={r['verdict']} rel={r['rel']!r} err={r['err']!r} keys={r['keys']}",
          flush=True)
    _check("P-na_ok",
           r["verdict"] == "FAIL" and r["rel"] is None and r["keys"]
           and r["err"] and r["err"].startswith("RuntimeError: SetPrecisionMode"),
           "max_rel=n/a ⇔ run_one 走 exc 分支（rel 恒 None）；前向键仍在 ⇒ 非 INVALID")


def p_na_early_forward():
    """前向编译前就抛（拿不到前向键）：应判 INVALID（不是 n/a）—— 反证 n/a 的边界。"""
    print("—— P-na_early 前向入口就抛（gdn_conv, host）——", flush=True)
    orig = gdn_kernel.conv_forward
    gdn_kernel.conv_forward = lambda *a, **k: _raise_precision()
    try:
        r = gc.run_one("gdn_conv", "cpu", no_run=False, device=CPU, n_dirs=1)
    finally:
        gdn_kernel.conv_forward = orig
    line = gc._line(r, False)
    print("    " + line, flush=True)
    _check("P-na_early", r["verdict"] == "INVALID" and not r["keys"],
           "前向未吃到内核却抛 ⇒ INVALID（n/a 不会在此出现）")


def main() -> int:
    print("== E3 判决行形态反推：nan vs n/a 的两条代码路径 ==", flush=True)
    ascend_env.reset()
    p_nan_analytic()
    p_na_after_kernel()
    p_na_early_forward()
    print(f"E3-NANPATH-DONE pass={_ok['n']} fail={_ok['fail']}", flush=True)
    return 1 if _ok["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
