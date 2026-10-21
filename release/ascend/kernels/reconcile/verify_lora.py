#!/usr/bin/env python3
"""reconcile/verify_lora.py — K 代理：lora_asc.py 的 910B target=ascend 编译判决（verify-only）。

lora_asc **无自身方言正文**：三个入口全是组合已验证的 linear 两件
（apply/merge → gemm_kernel.forward；backward → gemm 前向腿 + gemm_bwd_dw 权重梯度腿）。
故本件的可编性 = 其组成 kernel 的可编性。配方同 verify_addln：绕设备门 + monkeypatch compile；
**关键**：真编后返回"发射即 no-op"的替身（不抛），好让多腿链跑到底、backward 的 dW 腿也被
真正触发（否则第一条 gemm 腿一 launch 就中止，dW 腿根本执行不到 → 无法取证）。无设备，
no-op 发射不影响编译判决（本波只判编不判数值）。

判决分层（如实报，不假绿）：
  · 前向腿（gemm，P1-4 已合流）：apply/merge/backward 的 _lin → 应出 ascend|gemm[...] key；
  · 反向 dW 腿（l0tr，受 §12 asc_fill_l1 阻）：backward 的 gemm_bwd_dw → 应出 ascend|gemm_dw
    blocker（编译失败）；
  → 只要 dW 腿仍阻，lora 整件不得判全绿；打 VERIFY-ONLY/PARTIAL，注明待 §12 补齐。

形状：x(m=32,k=64) A(r=32,k=64) B(n=64,r=32)，全部可被块阶梯整除。

用法（容器内）：
  source /usr/local/Ascend/cann-8.5.0/set_env.sh
  export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201
  cd /work && TILELANG_CACHE_DIR=$(mktemp -d) timeout 300 python3 ascend/kernels/reconcile/verify_lora.py
"""
import os
import sys

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

import torch           # noqa: E402
import tilelang        # noqa: E402
from ascend.kernels import ascend_env, lora_asc  # noqa: E402


COMPILED_SRC = {"acc": ""}


def _stub_compile():
    """真编译后返回 no-op 发射替身：能编的腿静默"跑完"，多腿链得以跑到 dW 腿。"""
    real = tilelang.compile

    def _fake(*a, **k):
        kern = real(*a, **k)   # 编不出会在此抛 → get_compiled 记 blocker → 上层回退
        src = str(kern.get_kernel_source()) if hasattr(kern, "get_kernel_source") else ""
        COMPILED_SRC["acc"] += src

        class _Dummy:
            def __call__(self, *aa, **kk):
                return None   # no-op 发射（无设备，不判数值）

        return _Dummy()

    tilelang.compile = _fake


def _gate():
    ascend_env.reset()
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda target=None, device=None: ascend_env.TILELANG


def _drive() -> int:
    _stub_compile()
    _gate()
    torch.manual_seed(0)
    m, k, r, n = 32, 64, 32, 64
    x = torch.randn(m, k, dtype=torch.float16)
    a = torch.randn(r, k, dtype=torch.float16)
    b = torch.randn(n, r, dtype=torch.float16)
    dy = torch.randn(m, n, dtype=torch.float16)

    def _safe(fn, *args, **kw):
        try:
            fn(*args, **kw)
            return "done"
        except Exception as e:
            return f"exc:{type(e).__name__}:{str(e)[:60]}"

    res = {
        "apply": _safe(lora_asc.apply, x, a, b, 1.0, target="ascend"),
        "merge": _safe(lora_asc.merge, a, b, 1.0, target="ascend"),
        "backward": _safe(lora_asc.backward, dy, x, a, b, 1.0, target="ascend"),
    }
    keys = sorted(x2 for x2 in ascend_env.compiled_keys() if x2.startswith("ascend|"))
    bl = {x2: v for x2, v in ascend_env.blockers().items() if x2.startswith("ascend|")}
    src = COMPILED_SRC["acc"]
    print(f"calls={res}")
    print(f"keys={keys}")
    print(f"blockers={ {kk: vv[:120] for kk, vv in bl.items()} }")
    print(f"gemm_keys={sum(1 for x2 in keys if x2.startswith('ascend|gemm['))} "
          f"simt_in_src={('SimtVF' in src) or ('T.Parallel' in src)} .o~{len(src)}B")

    fwd_ok = any(x2.startswith("ascend|gemm[") for x2 in keys)
    dw_ok = any(x2.startswith("ascend|gemm_dw[") for x2 in keys)
    dw_bl = [vv for kk, vv in bl.items() if kk.startswith("ascend|gemm_dw")]
    print(f"forward-leg(gemm) compiled={fwd_ok}  dW-leg compiled={dw_ok} "
          f"dW-leg-blocked={bool(dw_bl)}")
    if fwd_ok and dw_ok and not bl:
        print("K-LORA ASC-COMPILE-PASS")
        return 0
    # lora 本体无可改写方言：可编性完全承接 linear 两件；如实分层报，不假绿
    if fwd_ok and (dw_bl or not dw_ok):
        print("K-LORA ASC-COMPILE-PARTIAL(前向腿可编；反向 dW 腿承 §12 asc_fill_l1 阻)")
        for vv in dw_bl:
            print(f"  ! dW: {vv[:200]}")
    print("K-LORA ASC-COMPILE-VERIFY-ONLY(无自有方言正文；整件全绿待 dW §12 补齐)")
    return 1


if __name__ == "__main__":
    sys.exit(_drive())
