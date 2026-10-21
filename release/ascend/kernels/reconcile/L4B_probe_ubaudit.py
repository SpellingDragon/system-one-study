#!/usr/bin/env python3
"""L4B_probe_ubaudit — 审计 GDN 短卷积**反向**件 `conv_bwd_asc_impl` 的 UB 槽「首写/首读」次序，
并对照前向件（`conv_asc_impl` scalar / `gdn_asc_impl` ub），判"是否存在 read-first（首访问是读）"。

【为何】K4 定案统一假说：GDN 族 device-only 缺陷群（dbeta 0.80 / conv 5.5e4 / gdn_delta 2.2e5）
疑同属**首次发射读取未初始化 UB**。本件只做**只读审计**（不改任何产品件）：
  - 逐 UB 槽在**源码**里标首写/首读行号；
  - 在**无设备容器真编**后，于**生成码（CCE）**里再标首写/首读行号；
  - 回答：是否存在"首访问=读"的槽。

手法：真编三份模具（fwd-conv / bwd-conv / fwd-gdn），把 `get_kernel_source()` 逐行编号打印，
凡含本件 UB 缓冲名的行全部列出（带行号），供人工对齐首写/首读。

跑法（容器 cann910b-h，release 挂 /work）：
  docker exec cann910b-h bash -lc 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \
    export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201; \
    cd /work && TILELANG_CACHE_DIR=/dev/shm/l4b python3 ascend/kernels/reconcile/L4B_probe_ubaudit.py'
注意：**不能**加 `from __future__ import annotations`（方言追踪依赖运行期注解）。
"""
import os
import sys

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if RELEASE not in sys.path:
    sys.path.insert(0, RELEASE)

import torch          # noqa: E402
import tilelang       # noqa: E402

from ascend.kernels import ascend_env, gdn_asc, gdn_conv_asc  # noqa: E402

_SAVED = {}


def _bypass_gate():
    """绕设备门三处（无设备容器里让 active_backend 认 tilelang），不动任何产品逻辑。"""
    for n in ("backend_available", "_npu_present", "active_backend"):
        _SAVED[n] = getattr(ascend_env, n)
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda *a, **k: ascend_env.TILELANG


def _compile(fn, name):
    """真编 + 取源；返回 (verdict, source or err)。"""
    try:
        kern = tilelang.compile(fn(), out_idx=[], target="ascend")
        src = kern.get_kernel_source() if hasattr(kern, "get_kernel_source") else ""
        return True, str(src)
    except Exception as exc:      # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _dump(name, src, needles):
    """逐行编号打印；另列含 needles 的行（生成码里 UB 缓冲的读写行）。"""
    lines = src.splitlines()
    print(f"\n===== {name}: source lines={len(lines)} =====", flush=True)
    hits = [(i + 1, ln) for i, ln in enumerate(lines)
            if any(nd in ln for nd in needles)]
    print(f"----- {name}: UB-touch lines (needles={needles}) -----", flush=True)
    for i, ln in hits:
        print(f"{i:4d}| {ln.rstrip()}", flush=True)
    # 另把整源存进容器缓存目录（仓库外落盘，供人工复核），并给行号索引
    root = os.environ.get("TILELANG_CACHE_DIR", "")
    if root and os.path.isdir(root):
        outp = os.path.join(root, f"{name}.cce")
        with open(outp, "w") as f:
            for i, ln in enumerate(lines, 1):
                f.write(f"{i:4d}| {ln}\n")
        print(f"  · dumped => {outp}", flush=True)
    return lines


def _bwd_conv():
    # 卡窗键口径：gdnconvb[ascend]|c8k4b2|c8scalar  ⇒ chans=8,k=4,batch=2,cores=8
    x = torch.randn(2, 8, 32)
    w = torch.randn(8, 4)
    b = torch.randn(8)
    dy = torch.randn(2, 8, 32)
    dx = torch.empty_like(x)
    dw = torch.empty_like(w)
    db = torch.empty_like(b)
    return gdn_conv_asc.conv_bwd_asc_impl(x, w, b, dy, dx, dw, db, 8, 4, 2, 8)


def _fwd_conv():
    x = torch.randn(2, 8, 32)
    w = torch.randn(8, 4)
    b = torch.randn(8)
    y = torch.empty_like(x)
    return gdn_conv_asc.conv_asc_impl(x, w, b, y, 8, 4, 2, 8, variant="scalar")


def _fwd_gdn():
    # 卡窗键口径：gdn[ascend]|h2k8v8|c8ubh ⇒ heads=2,dk=8,dv=8,cores=8,upc=2
    Q = torch.randn(2, 8, 8)
    K = torch.randn(2, 8, 8)
    V = torch.randn(2, 8, 8)
    G = torch.randn(2, 8, 8)
    Beta = torch.randn(2, 8)
    H0 = torch.randn(2, 8, 8)
    Hist = torch.empty(2, 8, 8, 8)
    Sout = torch.empty(2, 8, 8)
    O = torch.empty(2, 8, 8)
    return gdn_asc.gdn_asc_impl(Q, K, V, G, Beta, H0, Hist, Sout, O,
                                heads=2, dk=8, dv=8, cores=8, variant="ub")


def _bwd_gdn():
    # 已知 stale 敏感/dbeta 坏的那件（K4）——纳入审计以检验"静态 read-first 是否可判"
    Q = torch.randn(2, 8, 8)
    K = torch.randn(2, 8, 8)
    V = torch.randn(2, 8, 8)
    G = torch.randn(2, 8, 8)
    Beta = torch.randn(2, 8)
    DO = torch.randn(2, 8, 8)
    Hist = torch.randn(2, 8, 8, 8)
    Lam = torch.zeros(2, 8, 8)
    DQ = torch.empty(2, 8, 8)
    DKG = torch.empty(2, 8, 8)
    DV_ = torch.empty(2, 8, 8)
    DBeta = torch.empty(2, 8)
    DG = torch.empty(2, 8, 8)
    DH0 = torch.empty(2, 8, 8)
    return gdn_asc.gdn_bwd_asc_impl(Q, K, V, G, Beta, DO, Hist, Lam, DQ, DKG, DV_,
                                    DBeta, DG, DH0, heads=2, dk=8, dv=8, cores=2,
                                    variant="ub", dstate="zero")


def main():
    print(f"=== L4B_probe_ubaudit tilelang={tilelang.__version__} "
          f"py={sys.version.split()[0]} ===", flush=True)
    _bypass_gate()
    cases = [
        ("bwd_conv", _bwd_conv, ["dw_ub"]),
        ("fwd_conv", _fwd_conv, ["w_ub", "row_ub", "out_ub"]),
        ("fwd_gdn", _fwd_gdn, ["st_ub", "k_ub", "q_ub", "dec_ub"]),
        ("bwd_gdn", _bwd_gdn, ["lam_ub", "dec_ub", "u_scr", "dp_scr", "k_ub", "q_ub",
                               "v_scr", "do_scr"]),
    ]
    for name, fn, needles in cases:
        ok, src = _compile(fn, name)
        if not ok:
            print(f"L4B-UBAUDIT {name} COMPILE-FAIL {src[:300]}", flush=True)
            continue
        lines = _dump(name, src, needles)
        # 只按缓冲名列出生成码首写/首读（写入侧含 '=', 读取侧无）
        for nd in needles:
            idx = [(i + 1, ln) for i, ln in enumerate(lines) if nd in ln]
            if not idx:
                print(f"  · {name}.{nd}: 生成码无命中（可能被常量折叠/改名）", flush=True)
    print("\nL4B-UBAUDIT-DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
