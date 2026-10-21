#!/usr/bin/env python3
"""reconcile/verify_letter_readout.py — H 代理：letter_readout_asc.py 的 910B target=ascend 编译判决。

配方与 verify_addln.py 同源：绕设备门（patch backend_available/_npu_present/**active_backend**，
最后这个必须——cpu 张量 + target=ascend 否则会被判 torch_eager 走不到编译），真编译后
"调用即抛 LaunchSkip"的替身隔离发射。真实形状与 compile_all_asc._drive_letter_readout 同形：
rows=(16,64)f32、ids=int64（入口层转 int32）、picked=4；判决覆盖 forward(gather) 与
backward(scatter_add) 两个键。

--cpu：target="cpu" 语义回归判决（真编译真发射，与 rows[ids] / index_add 尺子对拍）。

用法（容器内，release 挂 /work）：
  source /usr/local/Ascend/cann-8.5.0/set_env.sh
  export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201
  cd /work && TILELANG_CACHE_DIR=$(mktemp -d) timeout 300 python3 ascend/kernels/reconcile/verify_letter_readout.py
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
from ascend.kernels import ascend_env, letter_readout_asc  # noqa: E402


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
    rows = torch.randn(16, 64, dtype=torch.float32)
    ids = torch.tensor([0, 3, 3, 15], dtype=torch.int64)   # 重复行号，入口层转 int32
    dy = torch.randn(4, 64, dtype=torch.float32)
    for fn, args in (("forward", (rows, ids)), ("backward", (16, ids, dy))):
        try:
            getattr(letter_readout_asc, fn)(*args, target="ascend")
        except LaunchSkip:
            pass
    keys = [k for k in ascend_env.compiled_keys() if k.startswith("ascend|")]
    bl = {k: v for k, v in ascend_env.blockers().items() if k.startswith("ascend|")}
    want = ("ascend|gather", "ascend|scatter_add")
    got = [p for p in want if any(k.startswith(p) for k in keys)]
    src = SRC_SLOT["acc"]
    simt = ("SimtVF" in src) or ("T.Parallel" in src) or ("asc_atomic" in src)
    print(f"keys={keys}")
    print(f".o~{SIZE_SLOT['n']}B simt_or_atomic_in_src={simt} "
          f"bounds_guard_in_src={'0 <=' in src or '< 0' in src}")
    if bl:
        print(f"blockers={bl}")
    if len(got) == len(want) and not bl and not simt:
        print("H-READOUT-ASC-COMPILE-PASS")
        return 0
    print("H-READOUT-ASC-COMPILE-FAIL")
    for v in bl.values():
        print(f"  ! {v[:260]}")
    return 1


def _drive_cpu() -> int:
    """target=cpu 语义不回归：直编直跑 cpu 正文（与 _run 同式），对拍 rows[ids]/index_add；
    入口 forward/backward(target="cpu") 的分派去向如实打印（容器内 ascend_env 探针门控属既有
    环境事实，与本波改动无关，判决主体走直编直跑）。"""
    torch.manual_seed(0)
    kwargs = {"target": "c", "target_host": "c", "execution_backend": "cython"}
    rows = torch.randn(16, 64, dtype=torch.float32)
    ids = torch.tensor([0, 3, 3, 15], dtype=torch.int32)   # 重复行号
    out = torch.zeros(4, 64, dtype=torch.float32)
    k = tilelang.compile(letter_readout_asc.gather_cpu_impl(rows, ids, out, dim=64),
                         out_idx=[], **kwargs)
    k(rows, ids, out)
    ok_f = torch.equal(out, rows[ids])
    dy = torch.randn(4, 64, dtype=torch.float32)
    acc = torch.zeros(16, 64, dtype=torch.float32)
    kb = tilelang.compile(letter_readout_asc.scatter_add_cpu_impl(dy, ids, acc, dim=64),
                          out_idx=[], **kwargs)
    kb(dy, ids, acc)
    ref = torch.zeros(16, 64, dtype=torch.float32)
    ref.index_add_(0, ids.long(), dy)
    ok_b = torch.equal(acc, ref)
    # 入口分派兜底：torch_eager 或 cpu 内核，两条都必须是同一语义答案
    e_f = torch.equal(letter_readout_asc.forward(rows, ids.long(), target="cpu"), rows[ids])
    e_b = torch.allclose(letter_readout_asc.backward(16, ids.long(), dy, target="cpu"), ref,
                         atol=1e-6, rtol=1e-6)
    keys = [k2 for k2 in ascend_env.compiled_keys() if k2.startswith("cpu|")]
    gate = ascend_env.blockers().get("cpu:probe", "")
    print(f"cpu-kernel gather_ok={ok_f} scatter_ok={ok_b} entry_f={e_f} entry_b={e_b} "
          f"cpu_keys={keys} gate={gate[:120]}")
    if ok_f and ok_b and e_f and e_b:
        print("H-READOUT-CPU-SEMANTIC-PASS")
        return 0
    print("H-READOUT-CPU-SEMANTIC-FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(_drive_cpu() if "--cpu" in sys.argv else _drive_ascend())
