"""负控件②：宿主件带 future-import，且被追踪函数注解里有"只在注解里出现"的名字 k。"""
from __future__ import annotations


def build(k):
    import tilelang.cpu.language as T

    @T.prim_func
    def probe(A: T.Tensor((k,), "float32")):
        """负控用的假件二。

        白话：这里 k 只在注解里出现，PEP 563 之后闭包单元丢失，正是坑①的真实形态。
        """
        return T.copy(A, A)
