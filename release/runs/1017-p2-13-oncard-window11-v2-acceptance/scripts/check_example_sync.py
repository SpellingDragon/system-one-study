#!/usr/bin/env python3
"""example_gdn_bwd.py 的 kernel 段必须与 gdn_bwd_v3.py 逐行一致。

背景：example 是 kernel 文件的"上库单文件"副本，历史上已漂移三次
（compensated GEMM / DQKTL / q3 fp32 / dk fp32 四处、以及 H>1 连续化），
每次都是上卡跑到一半才由数值或异常暴露。这里做成可复跑的对账门。

比对范围：从 `gdn_v3_ab` 的 `@tilelang.jit` 装饰行到 kernel 文件末尾；
example 侧到 `# ========== Golden` 之前。

唯一允许的差异：example 内联了 v2 的三个前置，因此删掉了
`try: from gdn_bwd_v2 import ...` / `except ImportError: ...` 两段。
其余任何差异都判 FAIL（exit 1）。

第二段（SYNC2）：example 内联的 `gdn_bwd_v2_cumsum / gdn_bwd_v2_solve_a /
gdn_bwd_v2_scan1` 三个函数体必须与 `examples/gdn_bwd_v2/gdn_bwd_v2.py` 的同名
函数逐行一致。函数体的归一化只做 rstrip + 去空行；第一段的
`_drop_v2_import_guards` 是整文件级例外，**不得**用于函数体（会掩盖真漂移）。
背景：A-F4 内联前置件在 `solve_a` 上漂移过一次（UB 中转 vs GM→L1 直拷），
原同步门只看 kernel 主体，漂移被漏掉。

用法：
    python3 scripts/check_example_sync.py            # 对账
    python3 scripts/check_example_sync.py --root DIR # 指定 tilelang-ascend 根
"""
import argparse
import difflib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# 默认 TA 根：与本仓同级（<workspace>/tilelang-ascend，由脚本位置上推 5 级到 workspace）。
# 不写死个人绝对路径；需要时用 TA_ROOT 覆盖这条推导。
DEFAULT_ROOT = os.environ.get(
    "TA_ROOT",
    os.path.join(HERE, *[os.pardir] * 5, "tilelang-ascend"))

KERNEL_REL = "examples/gdn_bwd_v3/gdn_bwd_v3.py"
EXAMPLE_REL = "examples/gdn_bwd_v3/example_gdn_bwd.py"
V2_REL = "examples/gdn_bwd_v2/gdn_bwd_v2.py"

# SYNC2 的对账对象：example 内联的三个前置件（与 v2 同名函数逐行比）
SYNC2_FUNCS = ("gdn_bwd_v2_cumsum", "gdn_bwd_v2_solve_a", "gdn_bwd_v2_scan1")

ANCHOR = "def gdn_v3_ab("
EXAMPLE_END = "# ========== Golden"


def _lines(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read().splitlines()


def _slice_from_anchor(lines, end_pred, path):
    """从 `gdn_v3_ab` 的装饰行（含）切到 end_pred 首次为真的行（不含）。"""
    try:
        i = next(n for n, s in enumerate(lines) if s.startswith(ANCHOR))
    except StopIteration:
        raise SystemExit(f"FAIL: {path} 里找不到 {ANCHOR!r}")
    if i == 0 or not lines[i - 1].lstrip().startswith("@tilelang.jit"):
        raise SystemExit(f"FAIL: {path}:{i + 1} 的 {ANCHOR} 上方不是 @tilelang.jit")
    start = i - 1
    for n in range(start, len(lines)):
        if end_pred(lines[n]):
            return lines[start:n]
    return lines[start:]


def _drop_v2_import_guards(lines):
    """去掉 `try:` + import gdn_bwd_v2 的整段（example 已内联前置，属合法差异）。"""
    out, i = [], 0
    while i < len(lines):
        if lines[i].strip() == "try:" and any(
                "gdn_bwd_v2" in s for s in lines[i:i + 9]):
            j = i
            while j < len(lines) and not lines[j].startswith("    except ImportError:"):
                j += 1
            j += 1
            while j < len(lines) and (lines[j].startswith("        ") or not lines[j].strip()):
                j += 1
            i = j
            continue
        out.append(lines[i])
        i += 1
    return out


def _norm(lines):
    lines = _drop_v2_import_guards(lines)
    return _norm_lines(lines)


def _norm_lines(lines):
    """函数体归一化：只做 rstrip + 去空行（不带第一段的 import-guard 例外）。"""
    return [s.rstrip() for s in lines if s.strip()]


def _func_body(lines, name, path):
    """提取顶层函数体：`def <name>(` 行（含）到下一顶层 def / 装饰器 / 章节注释前。

    函数体内不会出现列 0 的 `def ` / `@` / `# =` / `# -`，三个终止条件不会误伤。
    """
    start = next((n for n, s in enumerate(lines) if s.startswith(f"def {name}(")), None)
    if start is None:
        raise SystemExit(f"FAIL: {path} 里找不到 def {name}(")
    end = len(lines)
    for n in range(start + 1, len(lines)):
        s = lines[n]
        if s.startswith("def ") or s.startswith("@") \
                or s.startswith("# =") or s.startswith("# -"):
            end = n
            break
    return lines[start:end]


def _check_sync2(root):
    """第二段：example 内联前置件（cumsum/solve_a/scan1）vs v2 现行源码。"""
    v2path = os.path.join(root, V2_REL)
    epath = os.path.join(root, EXAMPLE_REL)
    if not os.path.exists(v2path):
        raise SystemExit(f"FAIL: 缺文件 {v2path}")
    v2lines, elines = _lines(v2path), _lines(epath)
    bad = 0
    for name in SYNC2_FUNCS:
        a = _norm_lines(_func_body(v2lines, name, v2path))
        b = _norm_lines(_func_body(elines, name, epath))
        if a == b:
            print(f"SYNC2 {name} OK（{len(a)} 行，已归一化）")
            continue
        bad += 1
        d = list(difflib.unified_diff(
            a, b, fromfile=V2_REL, tofile=EXAMPLE_REL, lineterm=""))
        print(f"SYNC2-FAIL {name} v2={len(a)} example={len(b)} 行，差异 {len(d)} 行：")
        print("\n".join(d[:40]))
    if bad:
        print(f"SYNC2-FAIL {bad}/{len(SYNC2_FUNCS)} 个内联前置件与 gdn_bwd_v2.py 不一致")
        return 1
    print(f"SYNC2-OK {len(SYNC2_FUNCS)} 个内联前置件与 gdn_bwd_v2.py 逐行一致"
          f"（{', '.join(SYNC2_FUNCS)}）")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args()

    kpath = os.path.join(args.root, KERNEL_REL)
    epath = os.path.join(args.root, EXAMPLE_REL)
    for p in (kpath, epath):
        if not os.path.exists(p):
            raise SystemExit(f"FAIL: 缺文件 {p}")

    kern = _slice_from_anchor(_lines(kpath), lambda s: s.startswith("if __name__"), kpath)
    exam = _slice_from_anchor(_lines(epath), lambda s: s.startswith(EXAMPLE_END), epath)
    a, b = _norm(kern), _norm(exam)

    rc = 0
    if a == b:
        print(f"SYNC-OK kernel={len(a)} example={len(b)} 行（已归一化）")
    else:
        d = list(difflib.unified_diff(
            a, b, fromfile=KERNEL_REL, tofile=EXAMPLE_REL, lineterm=""))
        print(f"SYNC-FAIL kernel={len(a)} example={len(b)} 行，差异 {len(d)} 行：")
        print("\n".join(d[:80]))
        rc = 1

    # ---- 第二段：内联前置件（example 自包含的三件 vs v2 现行源码）----
    rc |= _check_sync2(args.root)
    return rc


if __name__ == "__main__":
    sys.exit(main())
