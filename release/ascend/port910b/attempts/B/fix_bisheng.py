#!/usr/bin/env python3
"""attempts/B/fix_bisheng.py — 910B 编译选项注入器（cann910b-b 专用，幂等 + 自愈）

背景：bundle_ondemand.sh [3] 的锚注入假设 `'"-O2", "-fPIC", "-std=c++20"'` 就是列表结尾，
但 tilelang 0.1.15 的 bisheng.py 该行实际是
    result = ["-O2", "-fPIC", "-std=c++20", "-mllvm", "-cce-aicore-dcpreload-args=false"]
注入串 `", "-DTL..", "-I..impl", "-I..include"]` 把列表截断成非法语法（SyntaxError），
而 `import tilelang` 会加载 contrib/__init__.py -> bisheng，因此修复器**不能用 import
tilelang 定位自身**，改为走 sys.path 探测。

流程：① 撤销任何已存在的坏注入；② 以"追加列表元素"的正确形态注入
-DTL_PORT910B_NATIVE_TYPES 与 -I<asc/impl> -I<asc/include>；③ 首次改动前备份 .bak910bB；
④ ast.parse 自证语法可用。
"""
import ast
import glob
import pathlib
import re
import shutil
import sys

def find_bisheng() -> pathlib.Path:
    cands = []
    for root in sys.path:
        if not root:
            continue
        p = pathlib.Path(root) / "tilelang" / "contrib" / "bisheng.py"
        if p.is_file():
            cands.append(p)
    if not cands:
        raise SystemExit("FATAL: bisheng.py not found on sys.path")
    print("candidates:", [str(c) for c in cands])
    return cands[0]

p = find_bisheng()
s = p.read_text()

# ── [1] 撤销坏注入（bundle_ondemand.sh 形态：c++20 项后被塞了 `", "-DTL...", ...]`）
BAD = re.compile(
    r'-std=c\+\+20"", "-DTL_PORT910B_NATIVE_TYPES", "-I[^"]+/asc/impl", "-I[^"]+/asc/include"\]'
)
n_bad = len(BAD.findall(s))
if n_bad:
    s = BAD.sub('-std=c++20"', s)
    print(f"REVERTED-BAD-INJECTION x{n_bad}")

# ── [2] 正确注入：只在 -std=c++20 之后追加列表元素，不碰后续 -mllvm 项
ANCHOR = '"-O2", "-fPIC", "-std=c++20"'
if "TL_PORT910B_NATIVE_TYPES" not in s:
    assert ANCHOR in s, "anchor drift"
    inc = glob.glob("/usr/local/Ascend/*/aarch64-linux") + glob.glob(
        "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux"
    )
    with_impl = [d for d in inc if pathlib.Path(d, "asc", "impl").is_dir()]
    inc = with_impl or inc
    assert inc, "no aarch64-linux asc include root found"
    d = inc[0]
    s = s.replace(
        ANCHOR,
        ANCHOR + f', "-DTL_PORT910B_NATIVE_TYPES", "-I{d}/asc/impl", "-I{d}/asc/include"',
        1,
    )
    if not pathlib.Path(str(p) + ".bak910bB").exists():
        shutil.copy2(p, str(p) + ".bak910bB")
    p.write_text(s)
    print(f"PATCHED-OK inc_root={d}")
else:
    print("already-present (verify syntax only)")

ast.parse(p.read_text())
print("AST-PARSE-OK")
for i, line in enumerate(p.read_text().splitlines(), 1):
    if "TL_PORT910B_NATIVE_TYPES" in line:
        print(f"L{i}: {line.strip()[:220]}")
