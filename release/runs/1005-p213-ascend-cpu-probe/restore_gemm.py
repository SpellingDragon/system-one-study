"""重建 gemm_asc.py 被误删的两段"正文前奏"（外层 docstring 收尾 + 方言 import + 动态维 + prim_func 头）。

误删来自 repair_docs.py 把 BAD_HEADS 里的"白话"当垃圾头，从合法 docstring 的白话行一路吃到下一个
with ...Kernel( 边界，连带吞掉了外层 docstring 的收尾三引号、方言 import、m/nb 两行与 @T.prim_func
的 def 头。此处按同包其余件（attn_sw_asc/rope_asc 等）的同一前奏形态逐行补回，签名以受损前一次
grep 的真实输出为准（def gemm_impl(A: (m,k), W: (n,k), bias: (n,), C: (m,n))），补完必须重跑对拍。
"""
from pathlib import Path

p = Path("ascend/kernels/gemm_asc.py")
src = p.read_text()

cpu_old = '''    """CPU(c) 方言正文：全程 fp32，出口 fp32；网格只铺编译期常量列块，行块用动态上界串行推。

        with T.Kernel(nb) as mx:'''
cpu_new = '''    """CPU(c) 方言正文：全程 fp32，出口 fp32；网格只铺编译期常量列块，行块用动态上界串行推。

    追踪期约定：n/k/bm/bn/bk/act_mode 必须是 python int，只有行数 m 是动态符号；act_mode
    0=不截断、1=负值压零，口径选择在生成期用普通 if 完成。列块数 nb 也在生成期算好。

    白话：把一大片乘加切成小方块，每个小方块自己算自己那格，行数是多少都不影响这套切法；
    算完先在公共小格子里加上加成、把负数压成零，再整块写回大表格。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile）。
    """
    import tilelang.cpu.language as T

    m = T.dynamic("m")
    nb = n // bn  # 列块数：n 已被 plan 判过整除，是编译期常量

    @T.prim_func
    def gemm_impl(A: T.Tensor((m, k), "float32"), W: T.Tensor((n, k), "float32"),
                  bias: T.Tensor((n,), "float32"), C: T.Tensor((m, n), "float32")):
        with T.Kernel(nb) as mx:'''

asc_old = '''    """昇腾方言正文：L1 放两个操作数、L0C 放累加器、UB 上做收尾；块划分与 CPU 那份同构。

        with T.Kernel(NUM_BLOCKS) as bx:'''
asc_new = '''    """昇腾方言正文：L1 放两个操作数、L0C 放累加器、UB 上做收尾；块划分与 CPU 那份同构。

    追踪期约定：与 CPU 那份共用同一组整数常量（n/k/bm/bn/bk/act_mode），动态的只有行数 m；
    网格固定为 NUM_BLOCKS 个向量核块（该后端不吃动态网格上界），列块数 nb 在生成期算好。

    白话：同一本账换一台机器摆——两个操作数先搬到近处的大仓，乘加在专门做累加的那块仓里完成，
    加偏置与压负数这类逐格活挪到手边的小格子交给一堆小工一起做，最后整块写回。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile(target="ascend")，本地不实编）。
    """
    import tilelang.ascend.language as T

    m = T.dynamic("m")
    nb = n // bn

    @T.prim_func
    def gemm_impl(A: T.Tensor((m, k), "float32"), W: T.Tensor((n, k), "float32"),
                  bias: T.Tensor((n,), "float32"), C: T.Tensor((m, n), "float32")):
        with T.Kernel(NUM_BLOCKS) as bx:'''

for old, new in ((cpu_old, cpu_new), (asc_old, asc_new)):
    assert src.count(old) == 1, src.count(old)
    src = src.replace(old, new, 1)
p.write_text(src)
import ast
ast.parse(src)
print("gemm_asc restored, SYNTAX OK")
