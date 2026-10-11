#!/usr/bin/env python3
"""A4_probe_neg.py — 负例自证：把 add_ln 产品件的 `_plan_fwd` 桩坏 ⇒ `addln_prod` 必须 FAIL。

证明两件事：
  ① 新腿真接的是**产品接口件**（`add_ln_kernel.forward → add_ln_asc.add_ln_asc_impl`）——
     plan 一坏，入口件落回 torch `_ln` 快路、`compiled_keys` 里没有 `ascend|add_ln[...]`，
     `_route_key` 假绿守卫当场抛 AssertionError ⇒ 判决变 `Y-addln_prod-PROD-FAIL`（rc=1）；
  ② 它**不会**偷偷去用 §7 的旧 attempts 内联件兜底（旧件是另一份正文，能编出键的是它自己）。
对照：不桩时同参数 → PASS（见 run_numerics.py --only addln_prod --track cpu-alias）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
PORT = os.path.join(RELEASE_ROOT, "ascend", "port910b")
for p in (RELEASE_ROOT, PORT):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_numerics as R  # noqa: E402
from ascend.kernels import add_ln_asc  # noqa: E402


def main():
    R.ARGS.only, R.ARGS.track, R.ARGS.no_run = "addln_prod", "cpu-alias", False
    orig = add_ln_asc._plan_fwd
    add_ln_asc._plan_fwd = lambda *a, **k: None          # 桩坏产品 plan（entry 必落 torch 快路）
    try:
        line = R.prod_addln()
        print(f"UNEXPECTED-PASS（桩坏后仍绿 ⇒ 判据不可信）: {line}")
        rc = 0
    except AssertionError as e:
        print(f"Y-addln_prod-PROD-FAIL AssertionError: {str(e)[:220]}")
        rc = 1
    except Exception as e:  # noqa: BLE001
        print(f"Y-addln_prod-PROD-FAIL {type(e).__name__}: {str(e)[:220]}")
        rc = 1
    finally:
        add_ln_asc._plan_fwd = orig
    print(f"NEG-EXIT-CODE={rc}（期望 1）")
    return rc


if __name__ == "__main__":
    sys.exit(main())
