#!/usr/bin/env python3
"""E3_probe_cardtriage — 下窗「一枪」：把窗三 B 段两件 FAIL 的**抛点 / 坏点**钉到「哪一路 · 哪个输入 · 哪一帧」。

【为什么需要它】窗三 B 段判词只给到 `max_rel=nan` / `max_rel=n/a` 两行（`w3_gradcheck.log:39-40`），
形态不同但都没说「坏在哪」。本探针在**卡上一条命令**把两者拆开：

  · `gdn_delta`（判词 `nan`，无异常）：逐路（**数值差分路** vs **解析反传路**）× 逐输入（q/k/v/g/β）
    报「是否含 nan/inf + 最大绝对值」。
        - 若 **数值路**先含 nan ⇒ 病灶在**前向内核** `gdn_impl`（数值路只读前向）；
        - 若**解析路**含 nan 而数值路干净 ⇒ 病灶在**反向内核** `gdn_bwd_impl`（解析路是唯一吃反向的）；
        - 若两路都干净 ⇒ 不是 nan（回到 PASS 形态），说明是别的路坏（本件照实报）。
  · `gdn_conv`（判词 `n/a` = 异常被吞）：打印**完整 traceback 每帧 `file:line`**，把抛点钉死——
    是产品反向闭式 `autograd_asc._conv_silu_torch`（→ `F.conv1d`，torch_npu 侧）还是 tilelang 发射。

【判据一字不动】阈值 / rtol / atol / INVALID 语义全部沿用 `gradcheck_card`；本件**只加可观测性**，
不改产品件、不改判据。输出机器可 grep：`E3-TRIAGE-<op> VERDICT=…`。

跑法（卡上，一枪判两件）：
    cd /work && python3 ascend/kernels/reconcile/E3_probe_cardtriage.py
host 自证（两件本应 PASS，用 `--selftest` 人造两种坏形态，证诊断分支真能报）：
    cd release && .venv/bin/python ascend/kernels/reconcile/E3_probe_cardtriage.py --selftest
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
ASCEND = os.path.join(RELEASE, "ascend")
for p in (RELEASE, ASCEND):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from ascend import autograd_asc as aa  # noqa: E402
from ascend.kernels import ascend_env, gdn_kernel  # noqa: E402
import gradcheck_card as gc  # noqa: E402

#: 逐输入的语义名（locate 到「哪个输入」）
NAMES = {
    "gdn_delta": ("q", "k", "v", "g", "beta"),
    "gdn_conv": ("x", "w", "bias"),
}
#: traceback 只留仓内帧（剥掉 torch/framework 噪声，抛点更醒目）
_MARK = ("/ascend/", "/release/", "gradcheck_card", "autograd_asc", "gdn_", "_asc")


def _repo_frames(exc: BaseException) -> list[str]:
    out = []
    for fr in traceback.extract_tb(exc.__traceback__):
        if any(m in fr.filename for m in _MARK):
            out.append(f"{os.path.relpath(fr.filename, RELEASE)}:{fr.lineno} in {fr.name}")
    return out


def _fin(t: torch.Tensor) -> tuple[bool, int, int, float]:
    return (bool(torch.isfinite(t).all()), int(torch.isnan(t).sum()),
            int(torch.isinf(t).sum()), float(t.abs().max()))


def diag_op(op: str, target: str, device: torch.device, n_dirs: int) -> dict:
    """带诊断的 gradcheck：逐路逐输入报 nan/inf 与量级；异常则打印仓内 traceback 帧。"""
    names = NAMES.get(op, tuple(f"in{i}" for i in range(9)))
    eps = gc.EPS.get(op, 0.01)
    f, raw = gc._build(op, target, device)
    print(f"E3-TRIAGE-{op} BEGIN target={target} device={device.type} eps={eps:g} "
          f"shapes={[tuple(t.shape) for t in raw]}", flush=True)
    try:
        with torch.no_grad():
            out0 = f(*raw)
        print(f"E3-TRIAGE-{op} FWD finite={_fin(out0)[0]} nan={_fin(out0)[1]} "
              f"inf={_fin(out0)[2]} maxabs={_fin(out0)[3]:.4g}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"E3-TRIAGE-{op} VERDICT=exception-at-forward", flush=True)
        for fr in _repo_frames(exc):
            print(f"    raise-frame {fr}", flush=True)
        print(f"    exc={type(exc).__name__}: {(str(exc).splitlines() or [''])[0]}", flush=True)
        return {"verdict": "exception-at-forward"}

    torch.manual_seed(101)
    dys = [torch.randn(out0.shape, dtype=torch.float32, device=device) for _ in range(n_dirs)]
    base = [t.detach().clone().contiguous() for t in raw]
    leaves = [t.clone().detach().requires_grad_(True) for t in raw]
    sq_err = sq_ana = 0.0
    verdict = "clean"
    for di, dy in enumerate(dys):
        # ---- 解析路（唯一吃反向内核的那一路）----
        try:
            out = f(*leaves)
            g_ana = list(torch.autograd.grad((out * dy).sum(), leaves))
        except Exception as exc:  # noqa: BLE001
            print(f"E3-TRIAGE-{op} VERDICT=exception-at-analytic dir={di}", flush=True)
            for fr in _repo_frames(exc):
                print(f"    raise-frame {fr}", flush=True)
            print(f"    exc={type(exc).__name__}: {(str(exc).splitlines() or [''])[0]}", flush=True)
            return {"verdict": "exception-at-analytic"}
        # ---- 数值差分路（只读前向内核）----
        try:
            g_num = gc._numeric_grad(f, base, eps, dy)
        except Exception as exc:  # noqa: BLE001
            print(f"E3-TRIAGE-{op} VERDICT=exception-at-numeric dir={di}", flush=True)
            for fr in _repo_frames(exc):
                print(f"    raise-frame {fr}", flush=True)
            return {"verdict": "exception-at-numeric"}
        for name, a, n in zip(names, g_ana, g_num):
            fa, na, ia, ma = _fin(a)
            fn, nn, inn, mn = _fin(n)
            print(f"    dir{di} {name:5s} ana finite={fa} nan={na} inf={ia} maxabs={ma:.4g} | "
                  f"num finite={fn} nan={nn} inf={inn} maxabs={mn:.4g}", flush=True)
            if not fa:
                verdict = "nan-in-analytic(反向内核)"
            if not fn and verdict == "clean":
                verdict = "nan-in-numeric(前向内核)"
            sq_err += float(((n.double() - a.double()) ** 2).sum())
            sq_ana += float((a.double() ** 2).sum())
    rel = (sq_err ** 0.5) / max(sq_ana ** 0.5, 1e-30)
    print(f"E3-TRIAGE-{op} VERDICT={verdict} rel={rel:.4e} "
          f"pass={sq_err ** 0.5 <= gc.ATOL + gc.RTOL * sq_ana ** 0.5}", flush=True)
    return {"verdict": verdict, "rel": rel}


def _selftest():
    """host 自证：人造两种坏形态，证诊断分支真能定位（判据不动）。"""
    print("== selftest：gdn_delta 解析路注入 nan ==", flush=True)
    # 解析路唯一吃 `gdn_kernel.backward`（cpu target 上它就是那把 torch 尺子）⇒ 桩它最贴切
    from ascend.kernels import gdn_kernel as gk
    real_bwd = gk.backward

    def _nan_bwd(*a, **k):
        grads = list(real_bwd(*a, **k))
        grads[0] = grads[0] * float("nan")   # dq 注 nan
        return tuple(grads)

    gk.backward = _nan_bwd
    try:
        diag_op("gdn_delta", "cpu", torch.device("cpu"), 1)
    finally:
        gk.backward = real_bwd
    print("== selftest：gdn_conv 反向闭式抛（模拟 SetPrecisionMode）==", flush=True)
    real_conv = aa._conv_silu_torch
    aa._conv_silu_torch = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("SetPrecisionMode:synthetic 500001"))
    try:
        diag_op("gdn_conv", "cpu", torch.device("cpu"), 1)
    finally:
        aa._conv_silu_torch = real_conv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E3 卡上分诊一枪（nan 归路 / n/a 归帧）")
    ap.add_argument("--target", default=None, choices=("cpu", "ascend"))
    ap.add_argument("--dirs", type=int, default=3)
    ap.add_argument("--selftest", action="store_true", help="host 人造坏形态，证诊断分支")
    ap.add_argument("--op", default=None, choices=("gdn_delta", "gdn_conv"))
    args = ap.parse_args(argv)

    ascend_env.reset()
    if args.selftest:
        _selftest()
        return 0

    target = ascend_env.normalize_target(args.target)
    if target == ascend_env.TARGET_ASCEND:
        os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
    npu_ok = hasattr(torch, "npu") and bool(getattr(torch.npu, "is_available", lambda: False)())
    device = torch.device("npu") if (target == ascend_env.TARGET_ASCEND and npu_ok) else torch.device("cpu")
    ops = [args.op] if args.op else ["gdn_delta", "gdn_conv"]
    for op in ops:
        diag_op(op, target, device, args.dirs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
