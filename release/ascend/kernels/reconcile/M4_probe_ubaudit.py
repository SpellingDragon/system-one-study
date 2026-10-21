#!/usr/bin/env python3
"""M4_probe_ubaudit.py — P1-1zg / M4：GDN 族「UB 槽首写/首读次序」**机械**审计（偏移法）。

【为何】K4 定案把 `gdnb` fresh（首次发射）时 `dbeta` 错归因于"生成码里存在首迭代
        读-before-写路径"。本探针**机械地**（非人工目测）逐 UB 槽在**生成码**里标出
        firstW / firstR，回答"是否存在某槽的**首访问=读**（read-first）"。
        覆盖五只模具：gdnb-ub（设备真跑档 c2ubz）/ gdnb-pure / gdn-fwd-ub /
        gdnconvb-bwd / gdnconv-fwd。

【手法】真编后取 `get_kernel_source()`。生成码把每个 `alloc_shared` 展平成**同一块**
        `buf_dyn_shmem` + **常量偏移**（缓冲名不再出现）⇒ 不能按名找，只能按**偏移**归类：
          · 每个槽的 [base, base+size) 由源码 `alloc_shared` 次序/形状定（见 SLOTS）；
          · 对每处 `buf_dyn_shmem)[EXPR]` 访问，用"顶和拆分"求其**常量项**判槽：
            把 EXPR 按 '+' 拆成项，凡项内**不含字母**者累加其整数（纯常量项）——这样
            `(j*8)+i` 的 `8`（变量系数）不会被误当基址，`(i+64)` 的 `64`（真基址）会被认出。
          · 写/读：紧跟 `=`（且非 `==`）者是写，否则是读；**行内 RMW 按"读先于写"处理**
            （保守，宁可多报 read-first）。
        行号即**生成码行号**（`get_kernel_source()` 的 1-based 行）。

跑法（容器 cann910b-h，release 挂 /work）：
  docker exec cann910b-h bash -lc 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \
    export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201; \
    cd /work && TILELANG_CACHE_DIR=/dev/shm/m4 python3 ascend/kernels/reconcile/M4_probe_ubaudit.py'
注意：**不能**加 `from __future__ import annotations`（方言追踪依赖运行期注解）。
"""
import os
import re
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
    try:
        kern = tilelang.compile(fn(), out_idx=[], target="ascend")
        return True, str(kern.get_kernel_source())
    except Exception as exc:      # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


_ACC = re.compile(r"buf_dyn_shmem\)\[([^\]]*)\]")
_CONST_PART = re.compile(r"^\D*(\d+)\D*$")
_NAMED_DECL = re.compile(r"__ubuf__\s+\w+\s*\*\s*(\w+)\s*=\s*\(__ubuf__")


def _const_off(expr: str) -> int:
    """顶和拆分：只累加"不含字母"的项里的整数（纯常量项 = 基址偏移）。"""
    total = 0
    for part in expr.split("+"):
        if re.search(r"[A-Za-z_]", part):
            continue
        m = _CONST_PART.match(part.strip())
        if m:
            total += int(m.group(1))
    return total


def _slot_of(off: int, slots):
    for sname, base, size in slots:
        if base <= off < base + size:
            return sname
    return None


def _is_write(ln: str, end: int) -> bool:
    rest = ln[end:].lstrip()
    return rest.startswith("=") and not rest.startswith("==")


def _collect(src: str, slots):
    """收集 UB 访问。返回 ([(ln_no, slot, is_write)], named_slots)。

    生成码有两形态，都要覆盖：
      · **命名单缓冲**：只有一处 alloc_shared 时 codegen 留名（如 convb 的 `dw_ub`）
        ⇒ `__ubuf__ float *dw_ub = (__ubuf__ float *)0;` + `dw_ub[EXPR]`；
      · **展平**：多处 alloc_shared 时合并成 `buf_dyn_shmem` + 常量偏移 ⇒ 需按偏移归类。
    """
    lines = src.splitlines()
    named = sorted({m.group(1) for ln in lines for m in _NAMED_DECL.finditer(ln)}
                   - {"buf_dyn_shmem"})
    pats = [(n, re.compile(r"\b" + re.escape(n) + r"\[([^\]]*)\]")) for n in named]
    out = []
    for ln_no, ln in enumerate(lines, 1):
        occ = []
        if "buf_dyn_shmem" in ln:
            for m in _ACC.finditer(ln):
                slot = _slot_of(_const_off(m.group(1)), slots)
                if slot is not None:
                    occ.append((slot, _is_write(ln, m.end())))
        for n, pat in pats:
            for m in pat.finditer(ln):
                occ.append((n, _is_write(ln, m.end())))
        # 行内：先记读、后记写（RMW 保守判 read-first）
        for slot, is_w in sorted(occ, key=lambda t: t[1]):
            out.append((ln_no, slot, is_w))
    return out, named


def audit(src: str, slots, name: str):
    """返回 (verdict_rows, readfirst_rows)。verdict_rows: (slot, firstW, firstR, verdict, nW, nR)。"""
    lines = src.splitlines()
    accesses, named = _collect(src, slots)
    firstW, firstR, cntW, cntR = {}, {}, {}, {}
    for ln_no, slot, is_w in accesses:
        if is_w:
            firstW.setdefault(slot, ln_no)
            cntW[slot] = cntW.get(slot, 0) + 1
        else:
            firstR.setdefault(slot, ln_no)
            cntR[slot] = cntR.get(slot, 0) + 1
    all_slots = list(slots) + [("(named)" + n, None, None) for n in named]
    rows = []
    for sname, _b, _s in all_slots:
        w, r = firstW.get(sname), firstR.get(sname)
        nW, nR = cntW.get(sname, 0), cntR.get(sname, 0)
        if w is None and r is None:
            verdict = "未触及"
        elif w is None:
            verdict = "READ-FIRST(无写!)"
        elif r is None:
            verdict = "只写不读(死缓冲?)"
        elif w <= r:
            verdict = "先写后读 OK"
        else:
            verdict = "READ-FIRST ⚠"
        rows.append((sname, w, r, verdict, nW, nR))
    print(f"\n===== AUDIT {name}: src_lines={len(lines)} =====", flush=True)
    print(f"  {'slot':10s} {'firstW':>7s} {'firstR':>7s} {'#W':>4s} {'#R':>4s}  verdict", flush=True)
    rf = []
    for sname, w, r, verdict, nW, nR in rows:
        print(f"  {sname:10s} {str(w):>7s} {str(r):>7s} {nW:>4d} {nR:>4d}  {verdict}", flush=True)
        if verdict.startswith("READ-FIRST"):
            rf.append((sname, w, r))
    print(f"  => read-first slots: {rf or 'NONE'}", flush=True)
    return rows, rf


def _dump(name, src):
    root = os.environ.get("TILELANG_CACHE_DIR", "")
    if root and os.path.isdir(root):
        outp = os.path.join(root, f"{name}.cce")
        with open(outp, "w") as f:
            for i, ln in enumerate(src.splitlines(), 1):
                f.write(f"{i:4d}| {ln}\n")
        print(f"  · dumped => {outp}", flush=True)


# ── 槽表（float 单位；由源码 alloc_shared 次序 × shape 定）─────────────────────────
# gdnb-ub (dk=dv=8): lam(dv,dk) dec(dk) u(dv) dp(dv) k(dk) q(dk) v(dv) do(dv)
SLOTS_GDNB = [("lam_ub", 0, 64), ("dec_ub", 64, 8), ("u_scr", 72, 8), ("dp_scr", 80, 8),
              ("k_ub", 88, 8), ("q_ub", 96, 8), ("v_scr", 104, 8), ("do_scr", 112, 8)]
# gdn-fwd-ub (dk=dv=8, upc=2): st(dv) k(dk) q(dk) dec(dk)
SLOTS_GDNF = [("st_ub", 0, 8), ("k_ub", 8, 8), ("q_ub", 16, 8), ("dec_ub", 24, 8)]
# gdnconvb-bwd (k=4): dw(k)
SLOTS_CONVB = [("dw_ub", 0, 4)]
# gdnconv-fwd ubstage (k=4): w(k) row(length+pad) out(length) —— 动态长度 ⇒ 只审 w_ub 基址
SLOTS_CONVF = [("w_ub", 0, 4)]


def _bwd_gdn():
    Q = torch.randn(2, 8, 8); K = torch.randn(2, 8, 8); V = torch.randn(2, 8, 8)
    G = torch.randn(2, 8, 8); Beta = torch.randn(2, 8); DO = torch.randn(2, 8, 8)
    Hist = torch.randn(2, 8, 8, 8); Lam = torch.zeros(2, 8, 8)
    DQ = torch.empty(2, 8, 8); DKG = torch.empty(2, 8, 8); DV_ = torch.empty(2, 8, 8)
    DBeta = torch.empty(2, 8); DG = torch.empty(2, 8, 8); DH0 = torch.empty(2, 8, 8)
    return gdn_asc.gdn_bwd_asc_impl(Q, K, V, G, Beta, DO, Hist, Lam, DQ, DKG, DV_,
                                    DBeta, DG, DH0, heads=2, dk=8, dv=8, cores=2,
                                    variant="ub", dstate="zero")


def _bwd_gdn_pure():
    Q = torch.randn(2, 8, 8); K = torch.randn(2, 8, 8); V = torch.randn(2, 8, 8)
    G = torch.randn(2, 8, 8); Beta = torch.randn(2, 8); DO = torch.randn(2, 8, 8)
    Hist = torch.randn(2, 8, 8, 8); Lam = torch.zeros(2, 8, 8)
    DQ = torch.empty(2, 8, 8); DKG = torch.empty(2, 8, 8); DV_ = torch.empty(2, 8, 8)
    DBeta = torch.empty(2, 8); DG = torch.empty(2, 8, 8); DH0 = torch.empty(2, 8, 8)
    return gdn_asc.gdn_bwd_asc_impl(Q, K, V, G, Beta, DO, Hist, Lam, DQ, DKG, DV_,
                                    DBeta, DG, DH0, heads=2, dk=8, dv=8, cores=2,
                                    variant="pure", dstate="zero")


def _fwd_gdn():
    Q = torch.randn(2, 8, 8); K = torch.randn(2, 8, 8); V = torch.randn(2, 8, 8)
    G = torch.randn(2, 8, 8); Beta = torch.randn(2, 8); H0 = torch.randn(2, 8, 8)
    Hist = torch.empty(2, 8, 8, 8); Sout = torch.empty(2, 8, 8); O = torch.empty(2, 8, 8)
    return gdn_asc.gdn_asc_impl(Q, K, V, G, Beta, H0, Hist, Sout, O,
                                heads=2, dk=8, dv=8, cores=8, variant="ub")


def _bwd_conv():
    x = torch.randn(2, 8, 32); w = torch.randn(8, 4); b = torch.randn(8)
    dy = torch.randn(2, 8, 32); dx = torch.empty_like(x)
    dw = torch.empty_like(w); db = torch.empty_like(b)
    return gdn_conv_asc.conv_bwd_asc_impl(x, w, b, dy, dx, dw, db, 8, 4, 2, 8)


def _fwd_conv():
    x = torch.randn(2, 8, 32); w = torch.randn(8, 4); b = torch.randn(8)
    y = torch.empty_like(x)
    return gdn_conv_asc.conv_asc_impl(x, w, b, y, 8, 4, 2, 8, variant="ubstage")


def main():
    print(f"=== M4_probe_ubaudit tilelang={tilelang.__version__} "
          f"py={sys.version.split()[0]} ===", flush=True)
    _bypass_gate()
    cases = [
        ("gdnb_ub", _bwd_gdn, SLOTS_GDNB),
        ("gdnb_pure", _bwd_gdn_pure, []),      # pure 无 alloc_shared ⇒ 期望 0 槽
        ("gdn_fwd_ub", _fwd_gdn, SLOTS_GDNF),
        ("convb_bwd", _bwd_conv, SLOTS_CONVB),
        ("convf_fwd_ubstage", _fwd_conv, SLOTS_CONVF),
    ]
    summary = {}
    for name, fn, slots in cases:
        ok, src = _compile(fn, name)
        if not ok:
            print(f"\nM4-UBAUDIT {name} COMPILE-FAIL {src[:300]}", flush=True)
            continue
        _dump(name, src)
        rows, rf = audit(src, slots, name)
        summary[name] = rf
    print("\n===== M4-UBAUDIT SUMMARY =====", flush=True)
    for name, rf in summary.items():
        print(f"  {name}: read-first={rf or 'NONE'}", flush=True)
    print("M4-UBAUDIT-DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
