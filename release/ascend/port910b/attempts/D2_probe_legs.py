#!/usr/bin/env python3
"""attempts/D2_probe_legs.py — p2-13/P1-1n（D2）：**窗内四趟**判别器 runner（一个命令打完）。

它做三件事，且**一个禁改件都不碰**：
  1. 注入 -DTL910B_VARIANT_x：唯一入口 = monkeypatch `tilelang.contrib.bisheng.
     get_bisheng_compile_options`（`ascend/backend.py:23` 运行时取模块属性、
     `bisheng.py:187` 同模块全局名、`libgen.py:145` 函数内 import——三处都吃这个 patch）
     ⇒ 不需要改 `patch_bisheng.py` / `card_run.sh` / 任何 `*_asc.py` / `run_numerics.py`。
  2. 逐组（default / A / B / C …）各起**一个干净子进程**跑腿，每组**独立 TILELANG_CACHE_DIR**：
        run_numerics.py --only gemm_l1_intrinsic --track card [--no-run]  ┐ 真 §12 腿
        run_numerics.py --only dw_intrinsic      --track card [--no-run]  ┘
        compile_all_asc.py --target ascend                                —— 九件回归门
  3. 每组编完 digest 缓存目录，逐**核**输出三件指纹并可与基线比 SAME/DIFF：
        device_kernel.asc  —— 源码面（变体改头文件不改发射 ⇒ 期望恒 SAME）
        executable.so      —— 机器码面（**每个核一个**，归因用它）
        kernel.aibin       —— 设备镜像面（全局计数 + 指纹集合）

⚠ 判读口径（重要，别读反）：
   · `compile_all_asc.py` 打印的 `.o~NNNNB` 是 `len(get_kernel_source())`＝**源码文本长度**，
     §12 头文件变体对它**零影响**（九个数在 default/A/B/C 下都相同）。所以"九个数一致"只
     能证"没跑挂"，**不能**证等价，也**不能**证变体生效——机器码面看 executable.so/.aibin。
   · 真正的"默认路径逐字节等价"三件套 = 九腿判决行 9/9 PASS + 两腿 .asc sha 同基线 +
     两腿 .executable.so sha 同基线。

用法（容器/卡内，先 source CANN set_env）：
  python3 D2_probe_legs.py --combos default --baseline /tmp/d2_cache_base
  python3 D2_probe_legs.py --combos default,A,B,C --baseline /tmp/d2_cache_base
  python3 D2_probe_legs.py --combos A --card --with-compile-all   # 真机数值 + 九件门
  python3 D2_probe_legs.py --digest-only /tmp/d2_cache_base       # 只出某目录的指纹表
"""
import argparse
import glob
import hashlib
import os
import re
import subprocess
import sys

REPO = os.environ.get("P910B_REPO", "/work/ascend")
RUNNERS = {
    "compile_all": (os.path.join(REPO, "compile_all_asc.py"),
                    ["--target", "ascend"]),
    "gemm_l1_intrinsic": (os.path.join(REPO, "port910b", "run_numerics.py"),
                          ["--only", "gemm_l1_intrinsic", "--track", "card"]),
    "dw_intrinsic": (os.path.join(REPO, "port910b", "run_numerics.py"),
                     ["--only", "dw_intrinsic", "--track", "card"]),
}
SEC12_LEGS = ["gemm_l1_intrinsic", "dw_intrinsic"]

COMBO_FLAGS = {
    "default": [],
    "A": ["-DTL910B_VARIANT_A"],
    "B": ["-DTL910B_VARIANT_B"],
    "C": ["-DTL910B_VARIANT_C"],
    "A+C": ["-DTL910B_VARIANT_A", "-DTL910B_VARIANT_C"],
    "A+B+C": ["-DTL910B_VARIANT_A", "-DTL910B_VARIANT_B", "-DTL910B_VARIANT_C"],
}


def sha16(b):
    return hashlib.sha256(b).hexdigest()[:16]


# ─────────────────────────── digest ───────────────────────────
def scan(cache):
    """返回 (per-kernel 指纹 dict, aibin 指纹 list)。同名核按出现次序加 #idx。"""
    kernels = {}
    for p in sorted(glob.glob(os.path.join(cache, "*", "linux-aarch64", "kernels", "*"))) + \
              sorted(glob.glob(os.path.join(cache, "kernels", "*"))):
        asc_p = os.path.join(p, "device_kernel.asc")
        if not os.path.exists(asc_p):
            continue
        raw = open(asc_p, "rb").read()
        m = re.findall(rb"__global__\s+__[a-z]+__\s+void\s+([A-Za-z_0-9]+)", raw)
        name = (m[0].decode() if m else os.path.basename(p)[:10])
        row = {"asc": (len(raw), sha16(raw))}
        so_p = os.path.join(p, "executable.so")
        if os.path.exists(so_p):
            sb = open(so_p, "rb").read()
            row["so"] = (len(sb), sha16(sb))
        else:
            row["so"] = (0, "-")
        idx = sum(1 for k in kernels if k.startswith(name + "#"))
        kernels["%s#%d" % (name, idx)] = row
    aib = []
    for p in sorted(glob.glob(os.path.join(cache, "*", "ascend-binaries", "*", "kernel.aibin"))) + \
            sorted(glob.glob(os.path.join(cache, "ascend-binaries", "*", "kernel.aibin"))):
        b = open(p, "rb").read()
        aib.append((len(b), sha16(b)))
    return kernels, aib


def fp(rows):
    h = hashlib.sha256()
    for r in sorted(rows):
        h.update(("%s\n" % r).encode())
    return h.hexdigest()[:16]


def combo_fp(kernels, key):
    return fp(["%s %d %s" % (k, v[key][0], v[key][1]) for k, v in kernels.items()])


def print_digest(tag, kernels, aib):
    print("   [%s] 每核指纹（asc=源码面 so=机器码面）：" % tag)
    for k in sorted(kernels):
        print("        %-22s asc=%6dB %s | so=%7dB %s"
              % (k, kernels[k]["asc"][0], kernels[k]["asc"][1],
                 kernels[k]["so"][0], kernels[k]["so"][1]))
    print("        aibin %d 件 总=%dB fp=%s"
          % (len(aib), sum(x[0] for x in aib), fp(["%d %s" % (x[0], x[1]) for x in aib])))


def compare(tag, cur, base):
    """base 为 None 时只打 fp。逐核给 SAME/DIFF。"""
    if not base:
        return None
    bk, ba = base
    ck, ca = cur
    verdicts = {}
    for k in sorted(set(ck) | set(bk)):
        c, b = ck.get(k), bk.get(k)
        if c is None or b is None:
            verdicts[k] = "MISSING(%s)" % ("cur" if b else "base")
            continue
        va = "SAME" if c["asc"][1] == b["asc"][1] else "DIFF"
        vs = "SAME" if c["so"][1] == b["so"][1] else "DIFF"
        verdicts[k] = "asc=%s so=%s" % (va, vs)
        print("        [%s] %-22s %s%s" % (tag, k, verdicts[k],
                                           "" if vs == "SAME" else "  ← 机器码变了"))
    print("        [%s] aibin 集合交集 %d/%d（cur/base）"
          % (tag, len(set(x[1] for x in ca) & set(x[1] for x in ba)), max(len(ca), len(ba))))
    return verdicts


# ─────────────────────────── 子进程（装 -D 后跑腿） ───────────────────────────
def inprocess_main(a):
    flags = list(COMBO_FLAGS.get(a.combo, []))
    if a.extra_d:
        flags += [x for x in a.extra_d.split() if x and x.startswith("-D")]
    os.environ["TILELANG_CACHE_DIR"] = a.cache
    import tilelang  # noqa: F401
    import tilelang.contrib.bisheng as bs
    if flags:
        orig = bs.get_bisheng_compile_options

        def patched(npu_arch=None, options=None, *rest, **kw):
            base = list(orig(npu_arch, options, *rest, **kw))
            return base + [f for f in flags if f not in base]

        bs.get_bisheng_compile_options = patched
    print("## inprocess combo=%s flags=%s cache=%s tilelang=%s"
          % (a.combo, flags or "（默认 OFF）", a.cache, tilelang.__version__))
    rc_all = 0
    for leg in a.legs.split(","):
        script, argv = RUNNERS[leg]
        full = list(argv)
        if leg in SEC12_LEGS and a.compile_only:
            full += ["--no-run"]
        print("   $ %s %s" % (os.path.relpath(script, "/"), " ".join(full)))
        sys.stdout.flush()
        d = os.path.dirname(script)              # 脚本当 dir 原身靠 sys.path[0] 同级 import
        if d not in sys.path:
            sys.path.insert(0, d)
        sys.argv = [script] + full
        try:
            with open(script) as f:
                exec(compile(f.read(), script, "exec"),
                    {"__file__": script, "__name__": "__main__"})
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
            print("   -> exit rc=%s" % rc)
            rc_all |= (1 if rc else 0)
        except Exception as exc:  # noqa: BLE001
            print("   -> EXCEPTION %s: %s" % (type(exc).__name__, str(exc)[:200]))
            rc_all = 1
        finally:
            sys.argv = list(a._orig_argv)
    return rc_all


# ─────────────────────────── 编排 ───────────────────────────
def spawn(combo, legs, cache, card, extra_d):
    cmd = [sys.executable, os.path.abspath(__file__), "--inprocess", "--combo", combo,
           "--legs", ",".join(legs), "--cache", cache]
    if not card:
        cmd.append("--compile-only")
    if extra_d:
        cmd += ["--extra-d", extra_d]
    p = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    bad = [l for l in (p.stderr or "").splitlines() if l.strip()]
    if p.returncode != 0 or bad:
        for l in bad[-6:]:
            sys.stdout.write("   [stderr] %s\n" % l)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inprocess", action="store_true")
    ap.add_argument("--combo", default="default")
    ap.add_argument("--legs", default=",".join(SEC12_LEGS))
    ap.add_argument("--cache", default="/tmp/d2_legs")
    ap.add_argument("--compile-only", action="store_true", dest="compile_only")
    ap.add_argument("--card", action="store_false", dest="compile_only")
    ap.add_argument("--extra-d", default="")
    ap.add_argument("--combos", default="default,A,B,C")
    ap.add_argument("--baseline", default="", help="基线缓存目录（逐核 SAME/DIFF 对照）")
    ap.add_argument("--digest-only", default="")
    ap.add_argument("--with-compile-all", action="store_true",
                    help="每组额外跑 compile_all 九件回归门")
    ap.set_defaults(compile_only=True)
    a = ap.parse_args()

    if a.inprocess:
        a._orig_argv = list(sys.argv)
        return inprocess_main(a)
    if a.digest_only:
        k, x = scan(a.digest_only)
        print_digest(os.path.basename(a.digest_only.rstrip("/")), k, x)
        return 0

    base = scan(a.baseline) if a.baseline else None
    if base:
        print("== 基线 %s：%d 核 / %d aibin  fp_asc=%s fp_so=%s"
              % (a.baseline, len(base[0]), len(base[1]),
                 combo_fp(base[0], "asc"), combo_fp(base[0], "so")))
        print_digest("baseline", base[0], base[1])

    digs = {}
    for combo in [x for x in a.combos.split(",") if x]:
        legs = list(SEC12_LEGS)
        if a.with_compile_all or combo == "default":
            legs = ["compile_all"] + legs
        cache = "/tmp/d2_legs_%s" % combo.replace("+", "").replace(" ", "")
        print("\n════════ combo=%s  legs=%s  cache=%s ════════"
              % (combo, ",".join(legs), cache))
        rc = spawn(combo, legs, cache, not a.compile_only, a.extra_d)
        digs[combo] = (scan(cache), rc)
        print_digest(combo, digs[combo][0][0], digs[combo][0][1])
        print("   [%s] spawn rc=%d" % (combo, rc))

    print("\n════════ 结论表 ════════")
    if base and "default" in digs:
        print(" [default vs 基线]（期望：全 SAME = 默认路径逐字节等价）")
        compare("default", digs["default"][0], base)
    d = digs.get("default", (None, None))[0]
    for combo in digs:
        if combo == "default" or not d:
            continue
        print(" [+%s vs default]" % combo)
        vs = compare(combo, digs[combo][0], d)
        if vs:
            so_live = any("so=DIFF" in v for v in vs.values())
            print("   ⇒ 变体 %s：机器码面%s" % (combo, "**有反应**（活开关）" if so_live
                                                else "**无反应**（本窗载荷上的空开关）"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
