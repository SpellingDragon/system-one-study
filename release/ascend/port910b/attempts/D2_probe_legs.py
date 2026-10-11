#!/usr/bin/env python3
"""attempts/D2_probe_legs.py — p2-13/P1-1n（D2）：**窗内四趟**判别器 runner（一个命令打完）。

⚠ P1-1r 修复（2026-10-11，归因工具自身缺陷）——本文件是**唯一可改的产品面**，改动只加
   "可归因 + 判据刚性"，不改任何禁改件、不改任何判定语义：
   · 病灶 1（真病因，已在容器 1:1 复现）：`REPO` 默认写死容器路径 `/work/ascend`，而**卡上
     repo 根在 `/tmp/sys1`** ⇒ `run_numerics.py` 路径不存在 ⇒ 每腿 FileNotFoundError ⇒ 子进程
     rc=1、缓存 0 件、无 `NUM-*` 行 = 第二窗"两趟作废"的**逐字节同型现场**
     （`aibin 0 件 总=0B fp=e3b0c44298fc1c14` + `spawn rc=1`）。修法：跑前 precheck 逐腿断言
     runner 存在，缺则 `FATAL D2-PRECHECK` 并退 2（不再静默 rc=1）；并显式提示 `P910B_REPO` 写法。
   · 病灶 2：`--card` 形同虚设——父进程只"少传 `--compile-only`"，子进程 default 仍是
     compile_only=True ⇒ 真机数值腿从未执行（第二窗即此形态）。修法：父显式传 `--run`/`--no-run`。
   · 病灶 3：子进程 stdout 走 pipe + 行/块缓冲 ⇒ 硬杀时尾部丢失 = "静默 rc=1"。修法：子进程
     `-u` + `PYTHONUNBUFFERED=1`，stdout/stderr 落**文件**（非 pipe），父无条件回显 + 打 `child-rc`。
   · 病灶 4：`so=DIFF/SAME` 只在给 `--baseline` 时才打 ⇒ 第二窗命令 `--combos A,C --card`
     **结构上不可能**出该行。修法：每腿无条件打 `SO-VERDICT so=<SAME|DIFF|BASE>`（无 baseline 时
     以首趟 combo 为参照；参照腿记 `BASE`）。
   · 新增刚性判据：每腿必须齐 ①`## inprocess combo=<c>` ②`SO-VERDICT so=` ③`NUM-*`；缺任一
     该腿自判 `D2-LEG-INVALID`（不许算数据），并汇总 `D2-LEGS-SUMMARY valid=k/n`。

它做三件事，且**一个禁改件都不碰**：
  1. 注入 -DTL910B_VARIANT_x：唯一入口 = monkeypatch `tilelang.contrib.bisheng.
     get_bisheng_compile_options`（`ascend/backend.py:23` 运行时取模块属性、
     `bisheng.py:187` 同模块全局名、`libgen.py:145` 函数内 import——三处都吃这个 patch）
     ⇒ 不需要改 `patch_bisheng.py` / `card_run.sh` / 任何 `*_asc.py` / `run_numerics.py`。
  2. 逐组（default / A / B / C …）各起**一个干净子进程**跑腿，每组**独立 TILELANG_CACHE_DIR**：
        run_numerics.py --only gemm_l1_intrinsic --track card [--no-run]  ┐ 真 §12 腿
        run_numerics.py --only dw_intrinsic      --track card [--no-run]  ┘
        compile_all_asc.py --target ascend                               —— 九件回归门
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
  # 容器（有 bisheng 无设备）：编译面自证，不需要卡
  python3 D2_probe_legs.py --combos default,A,B,C --baseline /dev/shm/d2base/d2_legs_default
  # 真机（卡上）：必须先 export P910B_REPO=<repo 根>（卡上通常是 /tmp/sys1），否则 precheck FATAL
  P910B_REPO=/tmp/sys1 python3 D2_probe_legs.py --combos default,A,B,C --card \
        --baseline /tmp/sys1/d2_cache_base --cache-root /dev/shm/d2_legs
  python3 D2_probe_legs.py --digest-only /tmp/d2_cache_base       # 只出某目录的指纹表
"""
import argparse
import glob
import hashlib
import os
import re
import subprocess
import sys

# 卡上 repo 根**不一定是** /work/ascend（容器里是；卡上是 /tmp/sys1）。
# 第二窗作废现场就是这里写死容器路径、卡上又没 export P910B_REPO ⇒ 逐腿 FileNotFoundError。
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

# 每腿三条例合同判据（P1-1r）：缺任一 ⇒ 该腿 D2-LEG-INVALID，不许算数据。
RE_INJECT = re.compile(r"^## inprocess combo=(?P<c>[A-Za-z0-9+]+)\b", re.M)
RE_NUM = re.compile(r"^NUM-", re.M)
RE_SOV = re.compile(r"^\[(?P<c>[A-Za-z0-9+]+)\] SO-VERDICT so=(?P<v>SAME|DIFF|BASE)\b", re.M)


def sha16(b):
    return hashlib.sha256(b).hexdigest()[:16]


def combo_slug(combo):
    return combo.replace("+", "").replace(" ", "")


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


def so_verdict(cur, base):
    """机器码面单值判决：全核 so=SAME ⇒ SAME；任一 DIFF ⇒ DIFF；无核/无基线 ⇒ 特殊值。"""
    ck = cur[0]
    if not ck:
        return "NOKERNELS", "本腿 0 核（编译没出件）"
    if base is None:
        return "BASE", "无基线可比（该腿即参照）"
    bk = base[0]
    diffs = []
    for k in sorted(ck):
        if k not in bk or ck[k]["so"][1] != bk[k]["so"][1]:
            diffs.append(k)
    if diffs:
        return "DIFF", "机器码变了：%s" % ",".join(diffs[:6])
    return "SAME", "逐核 so 同基线（%d 核）" % len(ck)


def contract_missing(text, combo, so_val):
    """三条例合同判据。text=子进程原文（含 NUM 行）。返回缺失项列表。"""
    miss = []
    m = RE_INJECT.search(text)
    if not (m and m.group("c") == combo):
        miss.append("inject(## inprocess combo=%s)" % combo)
    if not RE_NUM.search(text):
        miss.append("num(NUM-*)")
    if so_val not in ("SAME", "DIFF", "BASE"):
        miss.append("so(so=%s)" % so_val)
    return miss


# ─────────────────────────── 子进程（装 -D 后跑腿） ───────────────────────────
def inprocess_main(a):
    flags = list(COMBO_FLAGS.get(a.combo, []))
    if a.extra_d:
        flags += [x for x in a.extra_d.split() if x and x.startswith("-D")]
    os.environ["TILELANG_CACHE_DIR"] = a.cache
    os.makedirs(a.cache, exist_ok=True)
    # 第一行先落盘（哪怕下面 import 就死，也留得下可归因的现场；父进程无条件回显）
    print("D2-CHILD-START combo=%s cache=%s run=%s legs=%s"
          % (a.combo, a.cache, a.run, a.legs), flush=True)
    try:
        import tilelang  # noqa: F401
        import tilelang.contrib.bisheng as bs
    except Exception as exc:  # noqa: BLE001
        print("D2-INJECT-FAIL combo=%s reason=%s: %s"
              % (a.combo, type(exc).__name__, str(exc)[:300]), flush=True)
        return 1
    if flags:
        orig = bs.get_bisheng_compile_options

        def patched(npu_arch=None, options=None, *rest, **kw):
            base = list(orig(npu_arch, options, *rest, **kw))
            return base + [f for f in flags if f not in base]

        bs.get_bisheng_compile_options = patched
    # 注入确认行（判据 ①）——**必须**打
    print("## inprocess combo=%s flags=%s cache=%s tilelang=%s"
          % (a.combo, flags or "（默认 OFF）", a.cache, tilelang.__version__), flush=True)
    rc_all = 0
    for leg in a.legs.split(","):
        script, argv = RUNNERS[leg]
        full = list(argv)
        if leg in SEC12_LEGS and not a.run:
            full += ["--no-run"]
        exists = os.path.exists(script)
        print("   $ %s %s%s" % (os.path.relpath(script, "/"), " ".join(full),
                                "" if exists else "   ← 脚本不存在（P910B_REPO=%s）" % REPO),
              flush=True)
        if not exists:
            print("D2-LEG-EXEC-FAIL leg=%s script=%s reason=FileNotFoundError" % (leg, script),
                  flush=True)
            rc_all = 1
            continue
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
            print("   -> exit rc=%s" % rc, flush=True)
            rc_all |= (1 if rc else 0)
        except Exception as exc:  # noqa: BLE001
            print("   -> EXCEPTION %s: %s" % (type(exc).__name__, str(exc)[:300]), flush=True)
            rc_all = 1
        finally:
            sys.argv = list(a._orig_argv)
    return rc_all


# ─────────────────────────── 编排 ───────────────────────────
def spawn(combo, legs, cache, do_run, extra_d, log_dir):
    """起干净子进程；子进程 stdout/stderr 落**文件**（非 pipe，硬杀也留尾）+ 无条件回显。"""
    logp = os.path.join(log_dir, "leg_%s.log" % combo_slug(combo))
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--inprocess", "--combo", combo,
           "--legs", ",".join(legs), "--cache", cache,
           "--run" if do_run else "--no-run"]
    if extra_d:
        cmd += ["--extra-d", extra_d]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["P910B_REPO"] = REPO
    try:
        with open(logp, "wb") as lf:
            p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        rc = p.returncode
    except Exception as exc:  # noqa: BLE001
        open(logp, "ab").write(("D2-SPAWN-FAIL %s: %s\n"
                                % (type(exc).__name__, str(exc)[:300])).encode())
        rc = 99
    raw = open(logp, "rb").read() if os.path.exists(logp) else b""
    text = raw.decode("utf-8", "replace")
    sys.stdout.write(text)
    if text and not text.endswith("\n"):
        sys.stdout.write("\n")
    # 无条件打：子进程退码 + 日志落点 + 字节数（0 字节且非 0 退码 = 硬杀，不再静默）
    print("   [%s] child-rc=%d log=%s bytes=%d%s"
          % (combo, rc, logp, len(raw),
             "   ← 子进程零输出即退出（硬杀/未启动），此腿设备现场不可信" if raw == b"" else ""),
          flush=True)
    return rc, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inprocess", action="store_true")
    ap.add_argument("--combo", default="default")
    ap.add_argument("--legs", default=",".join(SEC12_LEGS))
    ap.add_argument("--cache", default="/tmp/d2_legs")
    ap.add_argument("--compile-only", action="store_true", dest="compile_only",
                    help=argparse.SUPPRESS)
    ap.add_argument("--card", action="store_false", dest="compile_only")
    ap.add_argument("--run", action="store_true", help="子进程：真机执行（不桩掉发射）")
    ap.add_argument("--no-run", action="store_false", dest="run", help="子进程：只编译（桩掉发射）")
    ap.add_argument("--extra-d", default="")
    ap.add_argument("--combos", default="default,A,B,C")
    ap.add_argument("--baseline", default="", help="基线缓存目录（逐核 SAME/DIFF 对照）")
    ap.add_argument("--cache-root", default=os.environ.get("D2_CACHE_ROOT", "/tmp"),
                    help="每组缓存目录根：<root>/d2_legs_<combo>（盘紧时可指 /dev/shm）")
    ap.add_argument("--log-dir", default="", help="子进程日志落点（默认 <cache-root>/d2_logs）")
    ap.add_argument("--digest-only", default="")
    ap.add_argument("--with-compile-all", action="store_true",
                    help="每组额外跑 compile_all 九件回归门")
    ap.set_defaults(compile_only=True, run=False)
    a = ap.parse_args()

    if a.inprocess:
        a._orig_argv = list(sys.argv)
        return inprocess_main(a)
    if a.digest_only:
        k, x = scan(a.digest_only)
        print_digest(os.path.basename(a.digest_only.rstrip("/")), k, x)
        return 0

    do_run = a.run or (not a.compile_only)

    # ── precheck：任何 runner 缺失 = 立刻 FATAL（第二窗"静默 rc=1"的解药）──
    print("== D2-PRECHECK P910B_REPO=%s run=%s cache_root=%s" % (REPO, do_run, a.cache_root))
    bad = [(leg, s) for leg, (s, _) in RUNNERS.items() if not os.path.exists(s)]
    if bad:
        for leg, s in bad:
            print("FATAL D2-PRECHECK missing-runner leg=%s path=%s" % (leg, s))
        print("FATAL D2-PRECHECK 提示：卡上 repo 根通常不是 /work/ascend；"
              "请 export P910B_REPO=$(cd <repo> && git rev-parse --show-toplevel)"
              "（卡上实测多为 /tmp/sys1）后重跑。缺件腿一律不得算数据。")
        return 2
    print("D2-PRECHECK-OK runner=compile_all+gemm_l1_intrinsic+dw_intrinsic 全部在场")

    log_dir = a.log_dir or os.path.join(a.cache_root, "d2_logs")
    os.makedirs(log_dir, exist_ok=True)

    base = scan(a.baseline) if a.baseline else None
    if base:
        print("== 基线 %s：%d 核 / %d aibin  fp_asc=%s fp_so=%s"
              % (a.baseline, len(base[0]), len(base[1]),
                 combo_fp(base[0], "asc"), combo_fp(base[0], "so")))
        print_digest("baseline", base[0], base[1])

    combos = [x for x in a.combos.split(",") if x]
    digs = {}
    for combo in combos:
        legs = list(SEC12_LEGS)
        if a.with_compile_all or combo == "default":
            legs = ["compile_all"] + legs
        cache = os.path.join(a.cache_root, "d2_legs_%s" % combo_slug(combo))
        print("\n════════ combo=%s  legs=%s  cache=%s ════════"
              % (combo, ",".join(legs), cache))
        rc, text = spawn(combo, legs, cache, do_run, a.extra_d, log_dir)
        kernels, aib = scan(cache)
        digs[combo] = {"cache": cache, "rc": rc, "text": text,
                       "kernels": kernels, "aib": aib, "scan": (kernels, aib)}
        print_digest(combo, kernels, aib)
        print("   [%s] spawn rc=%d" % (combo, rc))

    # ── 每腿 so 判决 + 合同判据（②）──
    ref = "default" if "default" in digs else (combos[0] if combos else None)
    for combo in combos:
        r = digs[combo]
        if not r["kernels"]:
            so_val, so_note = "NOKERNELS", "本腿 0 核（编译没出件 ⇒ 无数据可算）"
        elif base:
            so_val, so_note = so_verdict(r["scan"], base)
        elif combo == ref:
            so_val, so_note = "BASE", "无 --baseline ⇒ 本腿为窗内参照（其余腿与它比）"
        else:
            so_val, so_note = so_verdict(r["scan"], digs[ref]["scan"])
        print("[%s] SO-VERDICT so=%s  （%s；asc_fp=%s so_fp=%s）"
              % (combo, so_val, so_note,
                 combo_fp(r["kernels"], "asc"), combo_fp(r["kernels"], "so")))
        miss = contract_missing(r["text"], combo, so_val)
        r["so"] = so_val
        r["invalid"] = miss
        print("[%s] LEG-VERDICT %s"
              % (combo, ("D2-LEG-INVALID missing=" + ",".join(miss))
                 if miss else "D2-LEG-OK（三判决行齐：inject+so+num）"))

    print("\nD2-LEGS-SUMMARY valid=%d/%d combos=%s invalid=%s"
          % (sum(1 for c in combos if not digs[c]["invalid"]), len(combos),
             ",".join(combos),
             ",".join(c for c in combos if digs[c]["invalid"]) or "（无）"))

    print("\n════════ 结论表 ════════")
    if base and "default" in digs:
        print(" [default vs 基线]（期望：全 SAME = 默认路径逐字节等价）")
        compare("default", digs["default"]["scan"], base)
    d = digs.get("default", {}).get("scan")
    for combo in combos:
        if combo == "default" or not d:
            continue
        print(" [+%s vs default]" % combo)
        vs = compare(combo, digs[combo]["scan"], d)
        if vs:
            so_live = any("so=DIFF" in v for v in vs.values())
            print("   ⇒ 变体 %s：机器码面%s" % (combo, "**有反应**（活开关）" if so_live
                                                else "**无反应**（本窗载荷上的空开关）"))
    return 0 if not any(digs[c]["invalid"] for c in combos) else 1


if __name__ == "__main__":
    sys.exit(main())
