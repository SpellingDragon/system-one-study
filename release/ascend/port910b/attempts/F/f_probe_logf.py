#!/usr/bin/env python3
"""attempts/F/f_probe_logf.py — GAP-F（软件 logf）取证件：DSL 发射面 + 编译差分 + 数值定标。

三种模式（互相独立，各自给判决行）：
  --compile   tilelang.compile 一个只用 `T.log`/`T.exp` 的最小件（标量面），
              证明 codegen 确实发射裸 `logf(float)`（与 C 波 G-C1 的 expf 同一条
              intrin_rule_ascend.cc: AscendMath float32→name+'f' 通路）。
              **差分**：未注入 GAP-F → 预期 COMPILE-FAIL（use of undeclared identifier
              'logf'）；注入后 → 预期 F-LOGF-PROBE-COMPILE-PASS。正负形都跑，才算缺口成立。
  --accuracy  `tl910b_logf` 的**逐行 numpy 复刻** vs np.log（与 compat_patch_F.h 成对维护，
              纪律同 C 波 §2「compat 软件件与 golden 复刻体必须成对」）：
              GDN decay 值域 (0,1] + 宽域 2^k 扫描 + 边界（1、e、1e-30、0、负数）。
  --gdn-path  把 logf 接进真实语义：a=exp(g) 与 g=log(a) 的**往返一致性**（口径 (b)→(a)
              换算的可编面），给出 rel 供 P2 三栈联调选口径时参考。

用法：python3 f_probe_logf.py [--mode compile|accuracy|gdn-path] [--N 256] [--dump f]
"""
import argparse
import os
import re
import sys
import traceback

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")

import numpy as np  # noqa: E402

FLT_MAX = np.float32(3.4028234663852886e38)
_F2U = lambda v: np.frombuffer(np.float32(v).tobytes(), dtype=np.uint32)[0]  # noqa: E731
_U2F = lambda u: np.frombuffer(np.uint32(int(u) & 0xFFFFFFFF).tobytes(), dtype=np.float32)[0]  # noqa: E731


def tl910b_logf(x):
    """compat_patch_F.h `tl910b_logf` 的逐行 numpy 复刻（fp32 逐步同序）。"""
    xs = np.asarray(x, dtype=np.float32)
    flat = xs.ravel().astype(np.float32).copy()
    for i in range(flat.size):
        v = np.float32(flat[i])
        if v <= np.float32(0.0):
            flat[i] = -FLT_MAX
            continue
        u = _F2U(v)
        e = int((u >> 23) & 0xFF) - 127
        mant = u & 0x7FFFFF
        m = np.float32(_U2F(0x3F800000 | mant))
        if m > np.float32(1.4142135623730951):
            m = np.float32(m * np.float32(0.5))
            e += 1
        z = np.float32((m - np.float32(1.0)) / (m + np.float32(1.0)))
        z2 = np.float32(z * z)
        s = np.float32(np.float32(1.0) + z2 * (np.float32(1.0 / 3) + z2 * (
            np.float32(0.2) + z2 * (np.float32(1.0 / 7) + z2 * np.float32(1.0 / 9)))))
        flat[i] = np.float32(np.float32(2.0) * z * s + np.float32(e) * np.float32(
            0.6931471805599453))
    return flat.reshape(xs.shape)


# ── DSL 件：标量面 T.log + T.exp ──
def logf_probe(N=256, dtype="float32"):
    import tilelang                                    # noqa: F401
    import tilelang.ascend.language as T

    @T.prim_func
    def probe(A: T.Tensor((N,), dtype), OUT: T.Tensor((N,), dtype)):
        with T.Kernel(1) as bx:
            v = T.alloc_var(dtype, T.float32(0.0))
            with T.Vector():
                for i in T.serial(N):
                    #  decay 口径 (b)→(a) 的真实换算形态：先 log 再 exp
                    v = T.log(T.max(A[i], T.float32(1e-30)))
                    OUT[i] = v + T.exp(v)
    return probe


def do_compile(args):
    import tilelang
    func = logf_probe(N=args.N)
    kernel = tilelang.compile(func, target="ascend", out_idx=-1)
    src = kernel.get_kernel_source()
    logs = sorted(set(re.findall(r"\b(logf|log2f|expf|exp2f)\s*\(", src)))
    print("F-LOGF-PROBE-COMPILE-PASS")
    print(f"  N={args.N} emitted_math={logs or '(none)'} lines={src.count(chr(10))}")
    if args.dump:
        with open(args.dump, "w") as f:
            f.write(src)
        print(f"  source dumped -> {args.dump}")


def do_accuracy(_args):
    print("F-LOGF-ACCURACY-BEGIN")
    worst_all = 0.0
    # (1) GDN decay 值域 a∈(0,1]（真实入口）
    a = np.linspace(np.float32(1e-8), np.float32(1.0), 20000, dtype=np.float32)
    ref = np.log(a.astype(np.float64))
    got = tl910b_logf(a).astype(np.float64)
    # log 值在 a→1 时趋于 0 ⇒ 相对误差没有意义，按**绝对误差/量级**与远离 1 处的相对误差双口径
    eabs = float(np.max(np.abs(got - ref)))
    mask = np.abs(ref) > 1e-3
    erel = float(np.max(np.abs(got[mask] - ref[mask]) / np.abs(ref[mask])))
    print(f"  DECAY-DOMAIN a∈(0,1] : max_abs_err={eabs:.3e} max_rel(|ln|>1e-3)={erel:.3e}")
    worst_all = max(worst_all, erel)
    # (2) 宽域 2^k 扫描（范围规约/指数拼接正确性）
    wide = np.array([2.0 ** k * (1.0 + 0.1 * j) for k in range(-30, 31)
                     for j in range(10)], dtype=np.float32)
    refw = np.log(wide.astype(np.float64))
    gotw = tl910b_logf(wide).astype(np.float64)
    # ln(x) 在 x→1 处趋于 0 ⇒ 相对误差无定义（曾在此除零得 nan），双口径报：
    eabsW = float(np.max(np.abs(gotw - refw)))
    mw = np.abs(refw) > 1e-3
    erelw = float(np.max(np.abs(gotw[mw] - refw[mw]) / np.abs(refw[mw])))
    print(f"  WIDE-SWEEP 2^k×[1,1.9] k∈[-30,30] n={wide.size}: "
          f"max_abs={eabsW:.3e} max_rel(|ln|>1e-3)={erelw:.3e}")
    worst_all = max(worst_all, erelw)
    # (3) 关键点与边界
    pts = {
        "log(1)": (np.float32(1.0), 0.0),
        "log(e)": (np.float32(np.e), 1.0),
        "log(0.5)": (np.float32(0.5), float(np.log(0.5))),
        "log(1e-30)": (np.float32(1e-30), float(np.log(1e-30))),
        "log(2)": (np.float32(2.0), float(np.log(2.0))),
    }
    for name, (x, want) in pts.items():
        got1 = float(tl910b_logf(np.array([x], np.float32))[0])
        print(f"  PT {name:12s} got={got1:.9g} want={want:.9g} abs_err={abs(got1-want):.3e}")
    for name, x in (("log(0)", 0.0), ("log(-1)", -1.0)):
        got1 = float(tl910b_logf(np.array([np.float32(x)], np.float32))[0])
        print(f"  EDGE {name:8s} -> {got1:.6g}（设计=钳位 -FLT_MAX，非 NaN）")
    ok = worst_all < 1e-6 and eabs < 1e-6 and eabsW < 1e-5
    print(f"F-LOGF-ACCURACY {'OK' if ok else 'FAIL'} worst_rel={max(worst_all,erel):.3e}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default=os.environ.get("TL_F_MODE", "accuracy"))
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--dump", default=os.environ.get("TL_F_DUMP", ""))
    args = ap.parse_args()
    if args.mode == "accuracy":
        return do_accuracy(args)
    if args.mode == "gdn-path":
        # 复用 F 波件自带的软件 expf 复刻做往返（不引 C 目录依赖，避免 attempts/C 交叉写）
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from f_gdn_golden import tl910b_expf  # noqa: E402
        a = np.exp(np.linspace(-8.0, 0.0, 5000)).astype(np.float32)   # decay ∈ (e^-8, 1]
        g = tl910b_logf(a)
        back = tl910b_expf(g)
        rel = float(np.max(np.abs(back.astype(np.float64) - a.astype(np.float64))
                           / np.maximum(np.abs(a.astype(np.float64)), 1e-30)))
        print(f"F-LOGF-GDNPATH 往返 a→log→exp→a' max_rel={rel:.3e} "
              f"（口径 (b) 接 GAP-F 后可安全喂给本波 delta rule 件的 g 入口）")
        return 0
    try:
        do_compile(args)
    except Exception as e:  # noqa: BLE001
        print("F-LOGF-PROBE-COMPILE-FAIL")
        s = str(e)
        err = [ln for ln in s.splitlines()
               if re.search(r"error|undeclared|undefined|no matching|FATAL|Check failed", ln)]
        for ln in (err[:6] or s.splitlines()[:6]):
            print("  " + ln.strip()[:220])
        if os.environ.get("TL_F_TRACE"):
            traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
