#!/usr/bin/env python3
"""N5_probe_keyshape.py — 容器 `--no-run` 键形的 host 本地替身（无设备）。

本沙箱到昇腾容器的 ssh **不可达**（TCP 连上但 SSH banner 超时，见 N5 战报②），故不能跑窗七那
条容器 `--no-run`。改用 host 直接构造 `plan_backward` 的 spec 键：证明 A2/N5 的改动**没有**
触碰键的生成（`plan`/`plan_backward` 逐字未改），键形仍是窗七的 `c2ubz`。

判据行：`N5-KEY …verdict=`。
用法：cd release && .venv/bin/python ascend/kernels/reconcile/N5_probe_keyshape.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # .../release
if RELEASE not in sys.path:
    sys.path.insert(0, RELEASE)

import torch  # noqa: E402

from ascend.kernels import gdn_asc  # noqa: E402

# run_delta_card 的 GDN 输入口径（H,TT,DK,DV = 2,8,8,8）
H, TT, DK, DV = 2, 8, 8, 8


def _z(*shape):
    return torch.zeros(shape, dtype=torch.float32)


def main() -> int:
    q, k, v = _z(H, TT, DK), _z(H, TT, DK), _z(H, TT, DV)
    g, beta, o = _z(H, TT, DK), _z(H, TT), _z(H, TT, DV)
    # 容器 `--target ascend` 口径：target 名 ascend、档位 ub、末状态梯度 zero
    spec = gdn_asc.plan_backward(q, k, v, g, beta, o,
                                 target="ascend", variant="ub", dstate="zero")
    assert spec is not None, "plan_backward 未开模（契约变了？）"
    fwd_key, bwd_key = spec["fwd"]["key"], spec["key"]
    print(f"N5-KEY fwd={fwd_key}")
    print(f"N5-KEY bwd={bwd_key}")
    want_bwd = "gdnb[ascend]|h2k8v8|c2ubz"
    print(f"N5-KEY bwd-unchanged verdict= {'PASS' if bwd_key == want_bwd else 'FAIL'}"
          f" want={want_bwd}")
    print(f"N5-KEY cores={spec['kwargs']['cores']} variant={spec['kwargs']['variant']}"
          f" dstate={spec['kwargs']['dstate']}")
    ok = bwd_key == want_bwd
    print(f"N5-KEY-DONE pass={1 if ok else 0} fail={0 if ok else 1}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
