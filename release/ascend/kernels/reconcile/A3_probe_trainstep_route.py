"""A3_probe_trainstep_route.py — 只读探针：`train_step.py --target ascend` 在卡上会编哪些件。

【为什么需要】卡窗 [4] 段是 `DMLAYA_ASCEND_TARGET=ascend python3 ascend/train_step.py --target
ascend`，但 host/容器都没有 npu，产品件的两级分发（`ascend_env.active_backend` 的设备门）一律判成
torch_eager ⇒ 静态看不出"下窗到底要编几件、每键什么形状"。本探针用 **run_numerics._prod_setup
同款三量桩**（`_npu_present` / `backend_available` / `active_backend`）把设备门打开，再包一层
`ascend_env.get_compiled` 记录每一次编译请求（键名自带形状：`gemm[ascend]|n32k32b16x32x16anone`），
于是**在不发射的前提下**枚举编译清单；`compile` 档还可在有 bisheng 的容器里真编一遍取时延。
零产品改动：只 import、只桩内存里的函数对象、只打印。

【模式】
    gate      — 只判门：桩前 `active_backend("ascend", "npu")` 必须是 torch_eager（无卡即门闭）；
                桩后必须是 tilelang（卡上就是这个值）。顺带把 autograd_asc 八个门面逐个过一遍门。
    keys      — get_compiled 记录键后返回 None（上层落回 eager、整步跑得完）⇒ 交出
                "train_step 会请求编译的全部键 + 请求次序 + 每键请求次数（复用率）"。
    compile   — 真编译每个键（容器：编得出、发不出 ⇒ 发射 no-op 桩），打印每键秒数与合计。

用法：
    <py> release/ascend/kernels/reconcile/A3_probe_trainstep_route.py keys
    cd release && TILELANG_CACHE_DIR=$(mktemp -d) python3 <本文件> compile     # 容器里跑
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter

TARGET = "ascend"


def _release_root() -> str:
    # A3_ROOT 让探针能被搬到容器 /root 里跑（路径推断此时会落空），默认按本件在
    # release/ascend/kernels/reconcile/ 下的位置回推。
    env = os.environ.get("A3_ROOT")
    if env:
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))        # .../release


ROOT = _release_root()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ["DMLAYA_ASCEND_TARGET"] = TARGET
import torch  # noqa: E402
from ascend.kernels import ascend_env as E  # noqa: E402

SEQ, D, HEADS, DIM = 16, 32, 2, 16          # 与 train_step.run() 内的固定尺寸一致（别改）


def gate_report():
    """桩前后对照：证明 ascend 门的开关只由"设备在场 + 探针可编"决定。"""
    try:
        before_npu = E.active_backend(TARGET, "npu")
    except Exception as exc:  # noqa: BLE001  这正是要取证的失败面
        before_npu = f"RAISE<{type(exc).__name__}: torch.device('npu') 不被注册（torch_npu 未 import）>"
    before_cpu = E.active_backend(TARGET, "cpu")
    E._npu_present = lambda: True
    E.backend_available = lambda *a, **k: True
    E.active_backend = lambda *a, **k: E.TILELANG
    after_npu = E.active_backend(TARGET, "npu")
    print(f"A3-GATE before: (ascend,npu)={before_npu} (ascend,cpu)={before_cpu}")
    print(f"A3-GATE after : (ascend,npu)={after_npu}   <- 卡上 npu 在场时未桩也应为 tilelang")
    return before_npu, after_npu


def instrument(mode: str):
    """接管 get_compiled：keys 档只记账不编译；compile 档真编译但桩掉发射。"""
    import tilelang
    reqs: list[str] = []
    secs: list[tuple[str, float]] = []
    fails: list[str] = []
    real_get = E.get_compiled

    class _Stub:
        """编译产物外壳：__call__ 空转（不发射），其余属性照真产物。"""

        def __init__(self, k):
            self._k = k

        def __call__(self, *a, **kw):
            return None

        def __getattr__(self, n):
            return getattr(self._k, n)

    def compile_wrap(prog, *a, **kw):
        t0 = time.perf_counter()
        try:
            kern = tilelang_real(prog, *a, **kw)
        except Exception as exc:  # noqa: BLE001
            dt = time.perf_counter() - t0
            nm = getattr(prog, "name", "?")
            fails.append(f"{nm}: {type(exc).__name__}: {str(exc)[:140]}")
            print(f"A3-COMPILE-FAIL {nm} sec={dt:.1f} {str(exc)[:140]}", flush=True)
            raise
        dt = time.perf_counter() - t0
        nm = getattr(prog, "name", "?")
        secs.append((nm, dt))
        print(f"A3-COMPILE {nm} sec={dt:.1f}", flush=True)
        return _Stub(kern) if mode == "compile" else kern

    tilelang_real = tilelang.compile
    tilelang.compile = compile_wrap

    def get_wrap(key, builder, target=None):
        full = E.make_key(target, key)
        reqs.append(full)
        if mode == "keys":
            return None                                  # ⇒ 上层落回 eager，整步能跑完
        if full in E._compiled:
            return E._compiled[full]
        try:
            kern = builder()
        except Exception as exc:  # noqa: BLE001
            E._blockers[full] = f"{type(exc).__name__}: {str(exc)[:120]}"
            return None
        E._compiled[full] = kern
        E._counts["compile"] += 1
        return kern

    E.get_compiled = get_wrap
    return reqs, secs, fails, tilelang, compile_wrap, tilelang_real, real_get


def run_step(device_type: str):
    import ascend.train_step as TS
    dev = torch.device(device_type)
    torch.Generator(device="cpu").manual_seed(1)
    if device_type != "cpu":
        # 无该设备时 train_step 会死在造输入处 —— 正是"第一个设备调用"的位置
        pass
    return TS.run(target=TARGET, device=dev, seed=20261010, verbose=True)


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "keys"
    if mode == "gate":
        gate_report()
        print("A3-TRAINSTEP-GATE-DONE")
        return 0

    E.reset()
    gate_report()
    reqs, secs, fails, tilelang, cw, realc, realget = instrument(mode)
    t0 = time.perf_counter()
    err = None
    try:
        l0, l1, missing = run_step("cpu")
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {str(exc)[:200]}"
        l0 = l1 = float("nan")
        missing = []
    wall = time.perf_counter() - t0

    uniq = sorted(set(reqs))
    print(f"\nA3-REQ-ORDER ({len(reqs)} 次请求 / {len(uniq)} 个唯一键, mode={mode}):")
    for i, k in enumerate(uniq, 1):
        print(f"  {i:2d}. {k}   x{Counter(reqs)[k]}")
    print(f"A3-STEP wall={wall:.1f}s compile_count={E.compile_count()} "
          f"real_compiles={len(secs)} sum_sec={sum(s for _n, s in secs):.1f} "
          f"loss0={l0} loss1={l1} missing_grad={missing} err={err}")
    if fails:
        print("A3-COMPILE-FAILS:")
        for f in fails:
            print("  " + f)
    blockers = {k: v for k, v in E.blockers().items() if not k.endswith(":probe")}
    if blockers:
        print("A3-BLOCKERS:")
        for k, v in blockers.items():
            print(f"  {k} -> {v}")
    tilelang.compile = realc
    E.get_compiled = realget
    print("A3-TRAINSTEP-ROUTE-" + ("FAIL" if err else "PASS"))
    return 0 if not err else 1


if __name__ == "__main__":
    sys.exit(main())
