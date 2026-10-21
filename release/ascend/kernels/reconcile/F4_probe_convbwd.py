#!/usr/bin/env python3
"""F4_probe_convbwd.py — GDN 短卷积反向（P1-1zb F4）的 host 三证探针（**不占卡**）。

【做什么】把"新反向不再依赖 F.conv1d"这件事在 host 上验成可复跑的四条判据：
  T1 产品反向 `gdn_conv_asc.conv_backward(target="cpu")` vs **独立参考**（对定义式
     `F.conv1d+silu` 现场自动微分）——fp32 相对误差。
  T2 同上 fp16 档：产品反向出口仍 fp16（不静默升宽），数值与 fp32 参考同量级。
  T3 **昇腾内核算法镜像**：逐字复刻 `conv_bwd_asc_impl` 的两趟（dz 归约 + 翻转抽头 gather，
     含上界"先夹后掩"）用 torch 标量面走一遍，与产品 `_bwd_eager` 对拍——证明"内核设计"与
     "已在 host 验证的尺子"是同一套账（方言翻译之外无额外逻辑）。
  T4 **负例自证**：把反向注入两处错（N1 dx 抽头方向写反；N2 dw 漏掉一个抽头），经**判据件
     `gradcheck_card.run_one`** 复跑 ⇒ 必红（before/after 对照），证"改完看起来对"骗不过判据。

【为什么】昇腾反向件本窗不能上卡，host 只能证"编得出"；数值正确性若只靠肉眼读算式就是自证。
本件用"独立参考 + 算法镜像 + 判据件负例"三路把数学钉死，卡窗只补"同一判决在真机再出一次"。

跑法：`cd release && .venv/bin/python ascend/kernels/reconcile/F4_probe_convbwd.py`
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _RELEASE not in sys.path:
    sys.path.insert(0, _RELEASE)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ascend import gradcheck_card as gc  # noqa: E402
from ascend.kernels import gdn_conv_asc  # noqa: E402

_RESULTS: list[tuple[bool, str]] = []


def _ok(cond: bool, msg: str) -> None:
    _RESULTS.append((bool(cond), msg))
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")


def _rel(a: torch.Tensor, ref: torch.Tensor) -> float:
    d = (a.double() - ref.double())
    return float(d.norm() / max(float(ref.double().norm()), 1e-30))


def _ref_fwd(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """独立参考：定义式 depthwise 因果 conv1d + SiLU（只在探针里用 conv1d，非产品路径）。"""
    chans, k = int(w.size(0)), int(w.size(1))
    pad = k - 1
    xs = F.pad(x, (pad, 0))
    return F.silu(F.conv1d(xs, w.reshape(chans, 1, k), bias, groups=chans))


def _ref_bwd(x, w, b, dy):
    """参考梯度：对定义式现场自动微分（与产品反向逐位不可比，故是"独立参考"）。"""
    lx = x.detach().clone().requires_grad_(True)
    lw = w.detach().clone().requires_grad_(True)
    lb = b.detach().clone().requires_grad_(True)
    y = _ref_fwd(lx, lw, lb)
    return torch.autograd.grad((y * dy).sum(), [lx, lw, lb])


# ────────────────────────── T3：昇腾内核算法镜像（逐字复刻 conv_bwd_asc_impl）──────────────
def _asc_mirror(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, dy: torch.Tensor):
    """把 `conv_bwd_asc_impl` 的两趟用 torch 标量面逐字走一遍（含上界先夹后掩），当数学锚。"""
    B, C, L = x.shape
    K = int(w.size(1))
    pad = K - 1
    ds = torch.zeros_like(x)
    db = torch.zeros(C, dtype=x.dtype)
    dw = torch.zeros(C, K, dtype=x.dtype)
    # 趟 1：逐格算 dz，同时归约偏置账与抽头账
    for bb in range(B):
        for t in range(L):
            z = b.clone()
            for j in range(K):
                idx = t - pad + j
                xv = x[bb, :, max(idx, 0)] if idx >= 0 else torch.zeros(C, dtype=x.dtype)
                z = z + w[:, j] * xv
            sig = 1.0 / (1.0 + torch.exp(-z))
            d = dy[bb, :, t] * sig * (1.0 + z * (1.0 - sig))
            ds[bb, :, t] = d
            db = db + d
            for j in range(K):
                idx = t - pad + j
                xv = x[bb, :, max(idx, 0)] if idx >= 0 else torch.zeros(C, dtype=x.dtype)
                dw[:, j] = dw[:, j] + d * xv
    # 趟 2：dx = 翻转抽头的加权 gather（读"未来"格，上界先夹到 L-1 再掩）
    dx = torch.zeros_like(x)
    for bb in range(B):
        for s in range(L):
            acc = torch.zeros(C, dtype=x.dtype)
            for j in range(K):
                t = s + pad - j
                tcl = min(t, L - 1)
                z = b.clone()
                for jj in range(K):
                    idx = tcl - pad + jj
                    xv = x[bb, :, max(idx, 0)] if idx >= 0 else torch.zeros(C, dtype=x.dtype)
                    z = z + w[:, jj] * xv
                sig = 1.0 / (1.0 + torch.exp(-z))
                d = dy[bb, :, tcl] * sig * (1.0 + z * (1.0 - sig))
                acc = acc + (w[:, j] * d if t < L else torch.zeros(C, dtype=x.dtype))
            dx[bb, :, s] = acc
    return dx, dw, db


def test_t1_fp32_matches_independent_reference() -> None:
    print("T1 产品反向(cpu) vs 独立参考（定义式自动微分）— fp32")
    torch.manual_seed(7)
    B, C, L, K = 2, 8, 6, 4
    x = torch.randn(B, C, L)
    w = torch.randn(C, K) * 0.5
    b = torch.randn(C) * 0.2
    dy = torch.randn(B, C, L)
    gx, gw, gb = _ref_bwd(x, w, b, dy)
    dx, dw, db = gdn_conv_asc.conv_backward(x, w, b, dy, target="cpu")
    _ok(dx.dtype == torch.float32 and dw.dtype == torch.float32 and db.dtype == torch.float32,
        f"dtype 全 fp32（dx={dx.dtype} dw={dw.dtype} db={db.dtype}）")
    for name, a, r in (("dx", dx, gx), ("dw", dw, gw), ("db", db, gb)):
        rel = _rel(a, r)
        _ok(rel < 1e-5, f"{name} rel={rel:.3e} < 1e-5")


def test_t2_fp16_delivers_fp16() -> None:
    print("T2 产品反向 fp16 档：出口按输入位宽交货（不静默升宽）")
    torch.manual_seed(7)
    B, C, L, K = 2, 8, 6, 4
    x = torch.randn(B, C, L)
    w = torch.randn(C, K) * 0.5
    b = torch.randn(C) * 0.2
    dy = torch.randn(B, C, L)
    gx, gw, gb = _ref_bwd(x, w, b, dy)                  # fp32 参考
    dx16, dw16, db16 = gdn_conv_asc.conv_backward(x.half(), w.half(), b.half(), dy.half(),
                                                  target="cpu")
    _ok(dx16.dtype == torch.float16 and dw16.dtype == torch.float16 and db16.dtype == torch.float16,
        f"dtype 全 fp16（dx={dx16.dtype} dw={dw16.dtype} db={db16.dtype}）")
    for name, a, r in (("dx", dx16, gx), ("dw", dw16, gw), ("db", db16, gb)):
        rel = _rel(a, r)
        _ok(rel < 2e-3, f"{name} fp16 vs fp32参考 rel={rel:.3e} < 2e-3")


def test_t3_ascend_kernel_algorithm_mirror() -> None:
    print("T3 昇腾内核算法镜像 vs 产品 _bwd_eager（同一套账）")
    torch.manual_seed(11)
    B, C, L, K = 3, 6, 5, 4
    x = torch.randn(B, C, L)
    w = torch.randn(C, K) * 0.5
    b = torch.randn(C) * 0.2
    dy = torch.randn(B, C, L)
    mdx, mdw, mdb = _asc_mirror(x, w, b, dy)
    edx, edw, edb = gdn_conv_asc._bwd_eager(x, w, b, dy)
    for name, a, r in (("dx", mdx, edx), ("dw", mdw, edw), ("db", mdb, edb)):
        rel = _rel(a, r)
        _ok(rel < 1e-5, f"{name} 镜像 rel={rel:.3e} < 1e-5")


def test_t4_negative_gradcheck_red() -> None:
    print("T4 负例：向新反向注入错 ⇒ 判据件 gradcheck_card 必红（before/after）")
    dev = torch.device("cpu")

    def _run() -> dict:
        return gc.run_one("gdn_conv", "cpu", no_run=False, device=dev, n_dirs=2)

    good = _run()
    _ok(good["verdict"] == "PASS", f"before(正身) {gc._line(good, False)}")

    orig = gdn_conv_asc._bwd_eager

    def _buggy_dir(x, w, b, dy):                 # N1：dx 抽头方向写反（不翻转）
        dx, dw, db = orig(x, w, b, dy)
        lead = tuple(x.shape[:-2])
        C, L, K = int(x.size(-2)), int(x.size(-1)), int(w.reshape(int(w.size(0)), -1).size(1))
        pad = K - 1
        xs = gdn_conv_asc._flatten(x).float()
        dys = gdn_conv_asc._flatten(dy).float()
        ws = w.reshape(C, K).float()
        xsp = F.pad(xs, (pad, 0))
        z = torch.zeros_like(xs)
        for j in range(K):
            z = z + ws[:, j].reshape(1, C, 1) * xsp[:, :, j:j + L]
        z = z + b.float().reshape(1, C, 1)
        sig = torch.sigmoid(z)
        ds = dys * sig * (1.0 + z * (1.0 - sig))
        xsp_ds = F.pad(ds, (pad, 0))
        bad = torch.zeros_like(xs)
        for j in range(K):                       # 方向错：用 pad-j 变成 j
            bad = bad + ws[:, j].reshape(1, C, 1) * xsp_ds[:, :, j:j + L]
        return bad.reshape(*lead, C, L).to(x.dtype), dw, db

    def _buggy_drop(x, w, b, dy):                # N2：dw 漏掉最后一个抽头
        dx, dw, db = orig(x, w, b, dy)
        dw = dw.clone()
        dw[..., -1] = 0.0
        return dx, dw, db

    try:
        gdn_conv_asc._bwd_eager = _buggy_dir
        res_dir = _run()
        _ok(res_dir["verdict"] == "FAIL" and res_dir["rel"] is not None,
            f"N1(方向错) after {gc._line(res_dir, False)}")
        gdn_conv_asc._bwd_eager = _buggy_drop
        res_drop = _run()
        _ok(res_drop["verdict"] == "FAIL" and res_drop["rel"] is not None,
            f"N2(漏抽头) after {gc._line(res_drop, False)}")
    finally:
        gdn_conv_asc._bwd_eager = orig

    again = _run()
    _ok(again["verdict"] == "PASS", f"after(复原) {gc._line(again, False)}")


def main() -> int:
    print("=== F4_probe_convbwd（host，不占卡）===")
    for fn in (test_t1_fp32_matches_independent_reference, test_t2_fp16_delivers_fp16,
               test_t3_ascend_kernel_algorithm_mirror, test_t4_negative_gradcheck_red):
        fn()
    n_pass = sum(1 for ok, _ in _RESULTS if ok)
    n_fail = len(_RESULTS) - n_pass
    print(f"F4-CONVBWD-DONE pass={n_pass} fail={n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
