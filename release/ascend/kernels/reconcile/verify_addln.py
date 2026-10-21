#!/usr/bin/env python3
"""reconcile/verify_addln.py — H 代理：add_ln_asc.py 的 910B target=ascend 编译判决（无卡、发射隔离）。

配方（P1-4 任务书）：绕设备门（patch backend_available/_npu_present/**active_backend**——
最后这个是必须的：cpu 张量 + target=ascend 会被 active_backend 判成 torch_eager，根本走不到
编译；compile_all_asc.py 现版没打这层补丁，折回时注意），真 tilelang.compile 跑完整链后
返回"调用即抛 LaunchSkip"的替身；用真实形状（与 compile_all_asc._drive_add_ln 同形：
x=(16,64)f16, residual=(16,64)f16, weight/bias=(64,)f32）驱动 add_ln_asc.forward/backward
(target="ascend")，判据 = compiled_keys() 同时含 ascend|add_ln[...] 与 ascend|ln_bwd[...]
且无 ascend| blocker，且生成码不含 SIMT 面载体。

--cpu：target="cpu" 语义回归判决（真编译真发射，与件内 torch 回退尺子 allclose）。

用法（容器内，release 挂 /work）：
  source /usr/local/Ascend/cann-8.5.0/set_env.sh
  export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201
  cd /work && TILELANG_CACHE_DIR=$(mktemp -d) timeout 300 python3 ascend/kernels/reconcile/verify_addln.py
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
from ascend.kernels import add_ln_asc, ascend_env  # noqa: E402


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


def _drive_ascend() -> int:
    _stub_compile()
    ascend_env.reset()
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda target=None, device=None: ascend_env.TILELANG
    x = torch.randn(16, 64, dtype=torch.float16)
    res = torch.randn(16, 64, dtype=torch.float16)
    w = torch.randn(64, dtype=torch.float32)
    b = torch.randn(64, dtype=torch.float32)
    h = torch.randn(16, 64, dtype=torch.float32)
    dy = torch.randn(16, 64, dtype=torch.float32)
    for fn, args in (("forward", (x, res, w, b)), ("backward", (h, w, dy))):
        try:
            getattr(add_ln_asc, fn)(*args, target="ascend")
        except LaunchSkip:
            pass
    keys = [k for k in ascend_env.compiled_keys() if k.startswith("ascend|")]
    bl = {k: v for k, v in ascend_env.blockers().items() if k.startswith("ascend|")}
    want = ("ascend|add_ln", "ascend|ln_bwd")
    got = [p for p in want if any(k.startswith(p) for k in keys)]
    src = SRC_SLOT["acc"]
    simt = ("SimtVF" in src) or ("T.Parallel" in src) or ("asc_simd" in src)
    print(f"keys={keys}")
    print(f".o~{SIZE_SLOT['n']}B simt_in_src={simt} rsqrtf_in_src={'rsqrtf' in src} "
          f"gm2ub_in_src={'asc_copy_gm2ub' in src}")
    if bl:
        print(f"blockers={bl}")
    if len(got) == len(want) and not bl and not simt:
        print("H-ADDLN-ASC-COMPILE-PASS")
        return 0
    print("H-ADDLN-ASC-COMPILE-FAIL")
    for v in bl.values():
        print(f"  ! {v[:260]}")
    return 1


def _drive_cpu() -> int:
    """target=cpu 语义不回归：直编直跑 cpu 正文（builder+compile 参数与 _run 完全同式），
    与件内 torch 回退尺子对拍；随后再走一遍入口 forward/backward(target="cpu") 记其分派去向。

    注：容器内 ascend_env 的 cpu 自探测探针(_compile_probe)在 py3.10 cython 适配下有既有
    TypeError（与本波改动无关，ascend_env 属禁改件），入口会被门控落到 torch_eager——
    因此判决主体走直编直跑（这正是 day-1 cpu 对拍资产同一条编译链），门控状态如实打印。
    """
    torch.manual_seed(0)
    kwargs = {"target": "c", "target_host": "c", "execution_backend": "cython"}
    x = torch.randn(16, 64, dtype=torch.float32)
    res = torch.randn(16, 64, dtype=torch.float32)
    w = torch.randn(64)
    b = torch.randn(64)
    y = torch.empty_like(x)
    h = torch.empty_like(x)
    k = tilelang.compile(add_ln_asc.add_ln_cpu_impl(x, res, w, b, y, h, dim=64, eps=1e-5, bm=16),
                         out_idx=[], **kwargs)
    k(x, res, w, b, y, h)
    ok_f = (torch.allclose(y, add_ln_asc._ln(x + res, w, b, 1e-5), atol=2e-3, rtol=2e-3)
            and torch.equal(h, x + res))
    dy = torch.randn(16, 64)
    dx = torch.zeros_like(x)
    dg = torch.zeros(64)
    db = torch.zeros(64)
    h32 = x + res
    kb = tilelang.compile(add_ln_asc.ln_bwd_cpu_impl(h32, w, dy, dx, dg, db, dim=64, eps=1e-5, bm=16),
                          out_idx=[], **kwargs)
    kb(h32, w, dy, dx, dg, db)
    dref, dgref, dbref = add_ln_asc._ln_bwd_torch(h32, w, dy, 1e-5)
    ok_b = (torch.allclose(dx, dref, atol=2e-3, rtol=2e-3)
            and torch.allclose(dg, dgref, atol=1e-2, rtol=1e-2)
            and torch.allclose(db, dbref, atol=1e-2, rtol=1e-3))
    # 入口分派兜底：torch_eager 或 cpu 内核，两条都必须是同一语义答案
    y2, h2 = add_ln_asc.forward(x, res, w, b, out_dtype=torch.float32, target="cpu")
    e2 = torch.allclose(y2, add_ln_asc._ln(x + res, w, b, 1e-5), atol=2e-3, rtol=2e-3)
    keys = [k2 for k2 in ascend_env.compiled_keys() if k2.startswith("cpu|")]
    gate = ascend_env.blockers().get("cpu:probe", "")
    print(f"cpu-kernel fwd_ok={ok_f} bwd_ok={ok_b} entry_ok={e2} cpu_keys={keys} gate={gate[:120]}")
    if ok_f and ok_b and e2:
        print("H-ADDLN-CPU-SEMANTIC-PASS")
        return 0
    print("H-ADDLN-CPU-SEMANTIC-FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(_drive_cpu() if "--cpu" in sys.argv else _drive_ascend())
