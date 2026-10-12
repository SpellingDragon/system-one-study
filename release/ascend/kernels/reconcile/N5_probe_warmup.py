#!/usr/bin/env python3
"""N5_probe_warmup.py — A2「首发射预热」单元级自证（host，无设备、无卡）。

`gdn_asc._launch_dev` 的三条行为（判决行 `N5-WARMUP …verdict=`）：

  ① 默认：**真机 NPU 张量** + key 首次 ⇒ 预热发射一次（丢弃）+ 真发射 ⇒ 共 **2 次 kernel
     调用、2 次同步、一行 `WARMUP <key>`**；其后同 key ⇒ **单次**发射（不再预热）；
  ② 负例：`SYS1_DISABLE_WARMUP=1` ⇒ 单次发射、零 WARMUP（行为回到无预热 —— 证明开关为真，
     同时留给窗八做 A0/A2 对照臂）；
  ③ host 门：cpu-alias（张量在 CPU、target 名仍 `"ascend"`，run_delta_card `--target cpu`
     真实形态）与 cpu target ⇒ **不走**预热，且 `_WARMED` 不被污染。

不 import 设备栈：假张量只带 `.device.type`，同步用计数桩替换。真机行为由窗八判决。
用法：cd release && .venv/bin/python ascend/kernels/reconcile/N5_probe_warmup.py
"""
import contextlib
import io
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # .../release
if RELEASE not in sys.path:
    sys.path.insert(0, RELEASE)

from ascend.kernels import gdn_asc  # noqa: E402

_PASS = 0
_FAIL = 0


def judge(ok: bool, tag: str, detail: str) -> None:
    global _PASS, _FAIL
    if ok:
        _PASS += 1
    else:
        _FAIL += 1
    print(f"N5-WARMUP {tag} verdict= {'PASS' if ok else 'FAIL'} {detail}")


class FakeT:
    """只有 `.device.type` 的假张量：不必真有设备也能走 `_on_npu` 判据。"""

    def __init__(self, type_: str):
        self.device = types.SimpleNamespace(type=type_)


def _session():
    """开一段独立的发射会话：返回 (launch, state, drain, stop)。

    launch(key, tensor, target) 记一次 `_launch_dev`；state 累计 kernel 调用/同步/预热行；
    drain() 取走截至目前累计的 `WARMUP` 行；stop() 还原同步桩。
    """
    gdn_asc._WARMED.clear()
    state = {"k": 0, "s": 0}
    real_sync = gdn_asc._sync_device
    gdn_asc._sync_device = lambda: state.__setitem__("s", state["s"] + 1)
    buf = io.StringIO()

    def launch(key, tensor, target):
        def kernel(*_a):
            state["k"] += 1
        with contextlib.redirect_stdout(buf):
            gdn_asc._launch_dev(kernel, (tensor,), key, target)

    def drain():
        lines = buf.getvalue().splitlines()
        buf.truncate(0)
        buf.seek(0)
        return [ln for ln in lines if ln.startswith("WARMUP ")]

    def stop():
        gdn_asc._sync_device = real_sync

    return launch, state, drain, stop


def _scenario_enabled():
    """默认（未禁）下：首键预热、同键不重复、异键各自预热。"""
    os.environ.pop(gdn_asc.ENV_DISABLE_WARMUP, None)
    launch, st, drain, stop = _session()
    try:
        launch("K1", FakeT("npu"), "ascend")           # 首键首次
        k1, s1, w1 = st["k"], st["s"], drain()
        launch("K1", FakeT("npu"), "ascend")           # 同键再来
        k2, s2, w2 = st["k"], st["s"], drain()
        launch("K2", FakeT("npu"), "ascend")           # 异键首次
        k3, _s3, w3 = st["k"], st["s"], drain()
    finally:
        stop()
    judge(k1 == 2 and s1 == 2 and w1 == ["WARMUP K1"],
          "01 default-first-key",
          f"launches={k1} syncs={s1} warmupline={w1}")
    judge(k2 == 3 and w2 == [],
          "02 default-same-key-again",
          f"launches={k2 - k1} cumulative={k2} warmupline={w2}")
    judge(k3 - k2 == 2 and w3 == ["WARMUP K2"],
          "03 default-distinct-key",
          f"launches={k3 - k2} warmupline={w3}")


def _scenario_disabled():
    """负例：SYS1_DISABLE_WARMUP=1 ⇒ 任何键都单次发射、零 WARMUP（回到无预热）。"""
    launch, st, drain, stop = _session()
    os.environ[gdn_asc.ENV_DISABLE_WARMUP] = "1"
    try:
        launch("KD", FakeT("npu"), "ascend")
        k1, s1, w1 = st["k"], st["s"], drain()
        launch("KD", FakeT("npu"), "ascend")
        k2, _s2, w2 = st["k"], st["s"], drain()
    finally:
        stop()
        os.environ.pop(gdn_asc.ENV_DISABLE_WARMUP, None)
    judge(k1 == 1 and s1 == 0 and w1 == [],
          "04 disabled-escape-hatch",
          f"launches={k1} syncs={s1} warmupline={w1}")
    judge(k2 == 2 and w2 == [],
          "05 disabled-second-launch",
          f"launches={k2 - k1} cumulative={k2} warmupline={w2}")


def _scenario_host_gate():
    """host 门：cpu-alias（CPU 张量 + target 名 ascend）与 cpu target 都不走预热。"""
    os.environ.pop(gdn_asc.ENV_DISABLE_WARMUP, None)
    launch, st, drain, stop = _session()
    try:
        launch("KC", FakeT("cpu"), "ascend")           # cpu-alias：设备是 CPU
        k1, _s1, w1 = st["k"], st["s"], drain()
        warmed_after_cpu = "KC" in gdn_asc._WARMED
        launch("KP", FakeT("npu"), "cpu")              # cpu target：即便张量在 npu 也不预热
        k2, _s2, w2 = st["k"], st["s"], drain()
    finally:
        stop()
    judge(k1 == 1 and w1 == [],
          "06 host-cpu-alias-no-warmup",
          f"launches={k1} warmupline={w1}")
    judge(warmed_after_cpu is False,
          "07 cpu-alias-not-marked-warmed",
          f"_WARMED 含 KC={warmed_after_cpu}")
    judge(k2 == 2 and w2 == [],
          "08 cpu-target-no-warmup",
          f"launches={k2 - k1} warmupline={w2}")


def main() -> int:
    _scenario_enabled()
    _scenario_disabled()
    _scenario_host_gate()
    print(f"N5-WARMUP-DONE pass={_PASS} fail={_FAIL}")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
