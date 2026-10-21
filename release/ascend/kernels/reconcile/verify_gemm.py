#!/usr/bin/env python3
"""reconcile/verify_gemm.py — K 代理：gemm_asc.py 的 910B target=ascend 编译判决（无卡、发射隔离）。

配方（P1-4 任务书 / D-int1）：绕设备门（patch backend_available/_npu_present/**active_backend**
——最后这个必须：cpu 张量 + target=ascend 会被 active_backend 判成 torch_eager，根本走不到编译），
真 tilelang.compile 跑完整链后返回"调用即抛 LaunchSkip"的替身；用真实形状（gemm A(m,k) W(n,k)
bias(n)f32 C(m,n)）驱动 gemm_kernel.forward(target="ascend")，判据 = compiled_keys() 含
ascend|gemm[...] 且无 ascend| blocker，且生成码不含 SimtVF/T.Parallel（910B 无此方言面）。

形状取 n=k=64（bn=bk=64 整除）、m=32（动态符号，一次编译服务任意批）；act 覆盖 none/relu 两条
收尾分支，位宽覆盖 f16/f32（ascend 白名单，plan 门 C.dtype==A.dtype）。

--cpu：target=cpu 语义回归（真编真跑，与 gemm_kernel._eager 尺子对拍，前向件不得回归）。

用法（容器内，release 挂 /work）：
  source /usr/local/Ascend/cann-8.5.0/set_env.sh
  export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201
  cd /work && TILELANG_CACHE_DIR=$(mktemp -d) timeout 300 python3 ascend/kernels/reconcile/verify_gemm.py
"""
import os
import sys

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
HERE = os.path.dirname(os.path.abspath(__file__))   # .../release/ascend/kernels/reconcile
RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

import torch           # noqa: E402
import tilelang        # noqa: E402
from ascend.kernels import ascend_env, gemm_kernel  # noqa: E402


class LaunchSkip(Exception):
    """kernel 被"发射"了：本判决只到编译为止，不判发射（无设备）。"""


SIZE_SLOT = {"n": 0}
SRC_SLOT = {"acc": ""}


def _stub_compile():
    """真编译之后换成一次性替身：拿得到生成码与 .o 尺寸，调用即抛（隔离发射）。"""
    real = tilelang.compile

    def _fake(*a, **k):
        kern = real(*a, **k)
        src = str(kern.get_kernel_source()) if hasattr(kern, "get_kernel_source") else ""
        SIZE_SLOT["n"] = max(SIZE_SLOT["n"], len(src))
        SRC_SLOT["acc"] += src

        class _Dummy:
            def __call__(self, *aa, **kk):
                raise LaunchSkip()

        return _Dummy()

    tilelang.compile = _fake


def _gate():
    ascend_env.reset()
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda target=None, device=None: ascend_env.TILELANG


def _cases():
    torch.manual_seed(0)
    m, n, k = 32, 64, 64
    for dt in (torch.float16, torch.float32):
        for act in ("relu", "none"):
            A = torch.randn(m, k, dtype=dt)
            W = torch.randn(n, k, dtype=dt)
            b = torch.randn(n, dtype=torch.float32)
            yield f"{str(dt).split('.')[-1]}-{act}", A, W, b, act


def _drive_ascend() -> int:
    _stub_compile()
    _gate()
    hit = []
    for tag, A, W, b, act in _cases():
        try:
            gemm_kernel.forward(A, W, b, act=act, out_dtype=torch.float16, target="ascend")
        except LaunchSkip:
            hit.append(tag)
    keys = [x for x in ascend_env.compiled_keys() if x.startswith("ascend|")]
    bl = {x: v for x, v in ascend_env.blockers().items() if x.startswith("ascend|")}
    src = SRC_SLOT["acc"]
    simt = ("SimtVF" in src) or ("T.Parallel" in src)
    print(f"driven={hit}")
    print(f"keys={keys}")
    print(f"blockers={bl}")
    print(f".o~{SIZE_SLOT['n']}B simt_in_src={simt} cube_in_src={'Cube' in src or 'mad' in src} "
          f"transposeB_in_src={'transpose' in src or 'l12l0' in src}")
    want_act = any("a relu" in x or "arelu" in x or "relu" in x for x in keys) and \
               any("none" in x for x in keys)
    if keys and not bl and not simt:
        print("K-GEMM ASC-COMPILE-PASS")
        return 0
    print("K-GEMM ASC-COMPILE-FAIL")
    for v in bl.values():
        print(f"  ! {v[:300]}")
    return 1


def _drive_cpu() -> int:
    """target=cpu 语义不回归：真编真跑 cpu 正文，与 gemm_kernel._eager 尺子对拍。"""
    _gate()
    torch.manual_seed(0)
    m, n, k = 32, 64, 64
    A = torch.randn(m, k, dtype=torch.float32)
    W = torch.randn(n, k, dtype=torch.float32)
    b = torch.randn(n, dtype=torch.float32)
    ok = True
    for act in ("relu", "none"):
        c = gemm_kernel.forward(A, W, b, act=act, out_dtype=torch.float32, target="cpu")
        ref = gemm_kernel._eager(A, W, b, act, torch.float32)
        good = torch.allclose(c, ref, atol=1e-2, rtol=1e-2)
        print(f"cpu act={act} allclose={good} max_abs_diff={(c - ref).abs().max().item():.3e}")
        ok = ok and good
    keys = [x for x in ascend_env.compiled_keys() if x.startswith("cpu|")]
    print(f"cpu_keys={keys}")
    if ok and keys:
        print("K-GEMM CPU-SEMANTIC-PASS")
        return 0
    print("K-GEMM CPU-SEMANTIC-FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(_drive_cpu() if "--cpu" in sys.argv else _drive_ascend())
