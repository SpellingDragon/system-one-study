#!/usr/bin/env python3
"""F4_probe_convbwd_compile.py — GDN 短卷积**反向**新件（`conv_bwd_asc_impl`）在真 ascend 方言下的编译判决。

【判什么的】host 腿只能证"数学对、出口位宽对"（F4_probe_convbwd.py 15/15）；本腿在**有 bisheng 的
无设备容器**里判：`target=ascend` 路由下，新的反向模具 `conv_bwd_asc_impl` 到底出不出得了产物
（`.o`），货架上的键是不是 `ascend|gdnconvb[ascend]|…`，以及有没有被静默回退（blocker / 回退痕迹）。

手法与 `compile_all_asc.py` 的接口腿同构：绕设备门三处 + `tilelang.compile` 换"真编译后发射抛异常"
的替身（无设备 ⇒ 本腿不判数值，只判编译与键账；发射被替身拦下是**期望路径**）。不复制任何判据，
只读 `ascend_env.compiled_keys()/blockers()`（真编过才留痕，回退不留痕 = R14 防假绿）。

对照：同批编**前向**件 `conv_asc_impl`（已知可编），同一套账下比出反向没编成的原因归属。

判决行：`F4-CARDBWD-<tag> PASS|FAIL keys=<n> first=<key> .o~<bytes>B blockers=<n>`；末行
`F4-CARDBWD-SUMMARY pass=<n> fail=<m>`。

用法（容器 cann910b-h 内，release 挂 /work；跑前确认 install compat md5=ca023c2e）：
  docker exec cann910b-h bash -lc 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \
    export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201; \
    cd /work && TILELANG_CACHE_DIR=/dev/shm/f4cache \
    python3 ascend/kernels/reconcile/F4_probe_convbwd_compile.py'
注意：**不能**加 `from __future__ import annotations`。
"""
import os
import sys
import traceback

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if RELEASE not in sys.path:
    sys.path.insert(0, RELEASE)

import torch           # noqa: E402
import tilelang        # noqa: E402

from ascend.kernels import ascend_env, gdn_conv_asc  # noqa: E402

_TARGET = ascend_env.TARGET_ASCEND
_RESULTS = []


class _LaunchSkip(Exception):
    """kernel 被"发射"时抛它：本 harness 只判编译，不判发射（无设备）。"""


_SAVED = {}
_SLOT = {"n": 0}


def _install_stubs():
    """绕设备门三处 + compile_kwargs 快照；`tilelang.compile` 换真编译后返回"发射即抛"的替身。"""
    for n in ("backend_available", "_npu_present", "active_backend", "compile_kwargs"):
        _SAVED[n] = getattr(ascend_env, n)
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda *a, **k: ascend_env.TILELANG
    real_compile = tilelang.compile
    _SAVED["tilelang.compile"] = real_compile

    def _compile(*a, **k):
        kern = real_compile(*a, **k)
        _SLOT["n"] = len(str(kern.get_kernel_source())) if hasattr(kern, "get_kernel_source") else 0

        class _Dummy:
            def __call__(self, *aa, **kk):
                raise _LaunchSkip()

            def __getattr__(self, n):
                return lambda *aa, **kk: 0

        return _Dummy()

    tilelang.compile = _compile


def _line(tag, verdict, detail):
    print(f"F4-CARDBWD-{tag} {verdict} {detail}", flush=True)


def _cache_files():
    """TILELANG_CACHE_DIR 下与本件相关的产物文件 → 字节数（`.aibin` 设备件 / `.so` 可执行 / `.asc` 源）。"""
    root = os.environ.get("TILELANG_CACHE_DIR", "")
    out = {}
    if root and os.path.isdir(root):
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.endswith((".aibin", ".so", ".asc")):
                    p = os.path.join(dp, fn)
                    try:
                        out[p] = os.path.getsize(p)
                    except OSError:
                        pass
    return out


def _report(tag):
    keys = [k for k in ascend_env.compiled_keys() if k.startswith(f"{_TARGET}|")]
    bl = {k: v for k, v in ascend_env.blockers().items() if k.startswith(f"{_TARGET}|")}
    if keys:
        detail = f"keys={len(keys)} first={keys[0]} .o~{_SLOT['n']}B blockers={len(bl)}"
        return "PASS", detail
    if bl:
        return "FAIL", f"keys=0 blockers={len(bl)} first={next(iter(bl.values()))[:200]}"
    return "FAIL", "keys=0 blockers=0（未达 tilelang.compile：形状/plan 拒或入口静默回退）"


def _drive(tag, fn):
    """跑一条腿：只清 ascend_env 账本（**不重复装桩**——重装会把上一轮的桩当"真编译"捕获，
    导致 `get_kernel_source()` 落到替身兜底、字节数量成假值）。"""
    _SLOT["n"] = 0
    ascend_env.reset()
    before = _cache_files()
    err = ""
    try:
        fn()
    except _LaunchSkip:
        pass                                   # 发射被替身拦下 = 本 harness 期望路径
    except Exception as exc:                   # noqa: BLE001
        err = (traceback.format_exception_only(type(exc), exc) or [""])[0].strip()
    new = {p: n for p, n in _cache_files().items() if p not in before}
    verdict, detail = _report(tag)
    if err and verdict != "PASS":
        detail = f"{detail} err={err[:160]}"
    _line(tag, verdict, detail)
    for p, n in sorted(new.items()):
        print(f"  · artifact {os.path.basename(p)} {n}B", flush=True)
    _RESULTS.append(verdict)


def _fwd():
    x = torch.randn(2, 16, 32)
    w = torch.randn(16, 4)
    b = torch.randn(16)
    gdn_conv_asc.conv_forward(x, w, b, torch.float32, _TARGET)


def _bwd():
    x = torch.randn(2, 16, 32)
    w = torch.randn(16, 4)
    b = torch.randn(16)
    dy = torch.randn(2, 16, 32)
    gdn_conv_asc.conv_backward(x, w, b, dy, _TARGET)


def main():
    print(f"=== F4_probe_convbwd_compile target={_TARGET} tilelang={tilelang.__version__} ===",
          flush=True)
    try:
        from tilelang.env import TILELANG_TEMPLATE_PATH  # noqa: F401
    except Exception as exc:                            # noqa: BLE001
        print(f"  · (模板路径探测失败，不影响编译：{exc})", flush=True)
    import hashlib
    _hdr = "/usr/local/lib/python3.10/dist-packages/tilelang/src/tl_templates/ascend/port910b_compat.h"
    if os.path.isfile(_hdr):
        print(f"  · compat(installed) md5={hashlib.md5(open(_hdr, 'rb').read()).hexdigest()}",
              flush=True)
    _install_stubs()                                    # 只装一次：真编译替身内部持有真 compile
    _drive("fwd", _fwd)                                 # 对照：已知可编的前向件
    _drive("bwd", _bwd)                                 # 判：新反向件
    n_pass = sum(1 for v in _RESULTS if v == "PASS")
    n_fail = len(_RESULTS) - n_pass
    print(f"F4-CARDBWD-SUMMARY pass={n_pass} fail={n_fail}", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
