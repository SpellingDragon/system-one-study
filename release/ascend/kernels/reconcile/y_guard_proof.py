"""R14 非空转自证（Y 代理，p2-13 P1-1i）：产品件 case 的**假绿守卫**必须真的会响。

做法：把 `ascend_env.compile_kwargs` 打成 host 真值（target="ascend" ⇒ host 无 ascend 后端 ⇒ 真编译
必失败）⇒ 入口件 `gemm_kernel.forward` 会静默落 torch eager 并交出**正确**的数。若守卫有效，
`prod_linear()` 必须抛"假绿守卫"，绝不能打印 PROD-PASS。
跑法：cd release && .venv/bin/python ../scratch/y_guard_proof.py
"""
import os
import sys

REL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "release")
sys.path.insert(0, REL)
sys.path.insert(0, os.path.join(REL, "ascend", "port910b"))
sys.argv = ["run_numerics.py", "--only", "linear_prod", "--no-run", "--track", "cpu-alias"]

import run_numerics as R  # noqa: E402  import 期即解析 argv


def main():
    orig = R._prod_setup

    def sabotaged():
        dev, restore, skip, box = orig()
        from ascend.kernels import ascend_env
        ascend_env.compile_kwargs = lambda t: {"target": "ascend"}   # 还原成真值：host 编不出
        return dev, restore, skip, box

    R._prod_setup = sabotaged
    try:
        line = R.prod_linear()
    except AssertionError as exc:
        print(f"GUARD-FIRED OK: Y-linear_prod-PROD-FAIL AssertionError: {str(exc)[:160]}")
        return 0
    print(f"GUARD-BROKEN（竟然出了 PASS 行 ⇒ 守卫没拦住 eager 假绿）: {line}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
