"""负控件①：方言件却带 future-import（坑①红线，应被拦）。"""
from __future__ import annotations
import ast  # noqa: F401  故意留一个死引用，验 F401 等价检查


def factory(n):
    import tilelang.cpu.language as T

    @T.prim_func
    def k_impl(A: T.Tensor((n, k), "float32")):
        """负控用的假件。

        白话：这行只是让注释门不拦我，本件不真编译，只验证体检脚本会不会响。
        """
        return None
