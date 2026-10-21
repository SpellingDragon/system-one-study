#!/usr/bin/env python3
"""attempts/C/patch_bisheng.py — 幂等、可自愈的 bisheng.py 910B 选项注入。

为什么不直接用 bundle_ondemand.sh [3] 的原片段（两个坑）：
  1) anchor '"-O2", "-fPIC", "-std=c++20"' 后追加 '", "-DTL_..."' 会拼出
     `"-std=c++20""`（双引号）并把 list 变成 tuple → SyntaxError。pip 0.1.15 的
     该行形如 `result = ["-O2", "-fPIC", "-std=c++20", "-mllvm", "-cce-aicore-..."]`，
     锚点后仍有内容，所以必须整行重写。
  2) 原片段用 `__import__("tilelang")` 定位文件，而 tilelang 自身 import 就要
     加载被改坏的 bisheng.py → 死锁（坏了就再也修不回来）。本件走文件系统定位。

行为：正则匹配 `    result = ["-O2"...]` 整行，重写为
    result = ["-O2", "-fPIC", "-std=c++20", "-DTL_PORT910B_NATIVE_TYPES",
              "-I<asc>/asc/impl", "-I<asc>/asc/include", "-mllvm",
              "-cce-aicore-dcpreload-args=false"]
已正确注入则 no-op；被上一版打坏则自愈。落盘前 compile() 语法自检。
"""
import glob
import os
import pathlib
import re
import sys

TAIL = '["-O2", "-fPIC", "-std=c++20", '
PAT = re.compile(r'^    result = \["-O2".*$', re.M)


def find_bisheng():
    pats = [
        '/usr/local/lib/python3*/dist-packages/tilelang/contrib/bisheng.py',
        '/usr/local/lib/python3*/site-packages/tilelang/contrib/bisheng.py',
        os.path.expanduser('~/.local/lib/python3*/site-packages/tilelang/contrib/bisheng.py'),
        '/usr/lib/python3*/site-packages/tilelang/contrib/bisheng.py',
    ]
    for pt in pats:
        for hit in sorted(glob.glob(pt)):
            if os.path.isfile(hit):
                return pathlib.Path(hit)
    try:  # 兜底：tilelang 若还能 import
        return pathlib.Path(__import__('tilelang').__file__).parent / 'contrib' / 'bisheng.py'
    except Exception:  # noqa: BLE001
        raise SystemExit('C-BISHENG-PATCH-FAIL: bisheng.py not found')


def asc_root():
    cands = sorted(
        glob.glob('/usr/local/Ascend/*/aarch64-linux')
        + glob.glob('/usr/local/Ascend/ascend-toolkit/*/aarch64-linux')
        + [os.path.join(os.environ.get('ASCEND_HOME_PATH', ''), 'aarch64-linux')]
    )
    for d in cands:
        if d and os.path.isdir(os.path.join(d, 'asc', 'impl')):
            return d
    raise SystemExit('C-BISHENG-PATCH-FAIL: no <root>/asc/impl under %r' % (cands,))


def main():
    p = find_bisheng()
    s = p.read_text()
    if not PAT.search(s):
        raise SystemExit('C-BISHENG-PATCH-FAIL: anchor line not found (drift?) in %s' % p)
    if '-DTL_PORT910B_NATIVE_TYPES' in PAT.search(s).group(0) and '""' not in PAT.search(s).group(0):
        print('bisheng.py already patched (%s)' % p)
        return
    root = asc_root()
    flags = '"-DTL_PORT910B_NATIVE_TYPES", "%s", "%s", ' % (
        '-I%s/asc/impl' % root, '-I%s/asc/include' % root)
    new_line = '    result = %s%s"-mllvm", "-cce-aicore-dcpreload-args=false"]' % (TAIL, flags)
    s2 = PAT.sub(lambda m: new_line, s, count=1)
    compile(s2, str(p), 'exec')  # 语法自检通过才落盘
    p.write_text(s2)
    print('C-BISHENG-PATCHED %s (root=%s)' % (p, root))
    print('  -> %s' % new_line.strip())


if __name__ == '__main__':
    sys.exit(main())
