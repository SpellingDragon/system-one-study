#!/usr/bin/env python3
"""reconcile/F2_apply3.py — guard 覆盖面补全：`@` 运算符面（__matmul__/__rmatmul__）。

实测（F2 host 探测）：方法面 `a.matmul(b)` 与 `a.sum()` 能被现有 guard 抓到，但组装面
最常见的内积写法是运算符 `@`——它查的是类型槽 `__matmul__`，不经过 `Tensor.matmul`
方法包装；`torch.mm(a, b)` 一类模块级 C builtin 则包了模块属性也管不住方法面，二者互
为补充不可混用。本补丁把 `__matmul__`/`__rmatmul__` 也按 dot 面罩上（同判据：任一操作
数为 fp16 ⇒ 违规）。代价如实：guard 仅在 fp16 档的 run() 窗口内安装，所有 matmul 多走
一层 Python（fp32 档与步外不受影响）。锚点数≠期望则整批不动。
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]  # release/
TS = ROOT / "ascend" / "train_step.py"

OLD = """        for _n in _GUARDED_DOTS:
            _f2_wrap(_n, dot=True)
"""
NEW = """        for _n in _GUARDED_DOTS:
            _f2_wrap(_n, dot=True)
        # `@` 运算符走类型槽 __matmul__，不经过 Tensor.matmul 方法包装（F2 实测），
        # 故把运算符面一并罩上：任一操作数为 fp16 即记违规（判据与 dot 面一致）。
        _f2_wrap("__matmul__", dot=True)
        _f2_wrap("__rmatmul__", dot=True)
"""


def main() -> int:
    text = TS.read_text(encoding="utf-8")
    n = text.count(OLD)
    if n != 1:
        print(f"锚点异常：命中 {n}（期望 1），未做任何改动")
        return 2
    TS.write_text(text.replace(OLD, NEW), encoding="utf-8")
    print(f"patched {TS.relative_to(ROOT.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
