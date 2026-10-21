"""把 gemm_asc.py 的受损部分**照受损前 .pyc 的原文**补齐（不再凭记忆改写）。

凭据来源：ascend/kernels/__pycache__/gemm_asc.cpython-312.pyc（时间戳 14:30，早于受损的 15:51），
其中函数 docstring 以常量原样保存、代码体由 dis 指令流反推。做三件事：
① 用 pyc 原文替换我上一版凭记忆重写的两个外层 docstring；
② 补回 plan() 的"白话"段与收尾三引号（被误删）；
③ 追加被整段吞掉的 plan() 函数体与 run()（模块指令流确认 run 之后没有别的东西，故这是文件结尾）。
"""
from pathlib import Path

p = Path("ascend/kernels/gemm_asc.py")
src = p.read_text()

cpu_new = '''    """CPU(c) 方言正文：全程 fp32，出口 fp32；网格只铺编译期常量列块，行块用动态上界串行推。

    白话：竖着切成固定的几列，每一列自己从上往下把行一段段搬进小格子算；有多少行都不影响切法，
    算完先在公共小格子里加加成、把负数压成零，再整块写回大表格。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile）。
    """'''
asc_new = '''    """昇腾方言正文：L1 放两个操作数、L0C 放累加器、UB 上做收尾；块划分与 CPU 那份同构。

    白话：把要乘的两小片先搬到近便的中间仓库，让专用的乘加单元连着算几轮并把账续在同一个
    格子里，算完再挪到另一块仓库做加成和压负，最后整块送回大表格。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile(target="ascend")，本地不实编）。
    """'''

cpu_cur = '''    """CPU(c) 方言正文：全程 fp32，出口 fp32；网格只铺编译期常量列块，行块用动态上界串行推。

    追踪期约定：n/k/bm/bn/bk/act_mode 必须是 python int，只有行数 m 是动态符号；act_mode
    0=不截断、1=负值压零，口径选择在生成期用普通 if 完成。列块数 nb 也在生成期算好。

    白话：把一大片乘加切成小方块，每个小方块自己算自己那格，行数是多少都不影响这套切法；
    算完先在公共小格子里加上加成、把负数压成零，再整块写回大表格。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile）。
    """'''
asc_cur = '''    """昇腾方言正文：L1 放两个操作数、L0C 放累加器、UB 上做收尾；块划分与 CPU 那份同构。

    追踪期约定：与 CPU 那份共用同一组整数常量（n/k/bm/bn/bk/act_mode），动态的只有行数 m；
    网格固定为 NUM_BLOCKS 个向量核块（该后端不吃动态网格上界），列块数 nb 在生成期算好。

    白话：同一本账换一台机器摆——两个操作数先搬到近处的大仓，乘加在专门做累加的那块仓里完成，
    加偏置与压负数这类逐格活挪到手边的小格子交给一堆小工一起做，最后整块写回。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile(target="ascend")，本地不实编）。
    """'''
for cur, new in ((cpu_cur, cpu_new), (asc_cur, asc_new)):
    assert src.count(cur) == 1, cur[:40]
    src = src.replace(cur, new, 1)

# ②③ 文件当前在 plan 的 docstring 中途被截断，把 tail 整段接回去
marker = "    None，由入口层换普通写法。\n"
assert src.count(marker) == 1
head = src.split(marker)[0] + marker
tail = '''
    白话：先看这活儿的尺寸和记法合不合现成模具；不合就不勉强开模，直接换手工做法。
    """
    name = ascend_env.normalize_target(target)
    if act not in SUPPORTED_ACTS:
        return None
    n, k = int(W.size(0)), int(W.size(1))
    bn, bk = _largest_block(n), _largest_block(k)
    if bn is None or bk is None:
        return None
    if C.dtype != A.dtype or bias is not None and bias.dtype != torch.float32:
        return None
    if name == ascend_env.TARGET_CPU and A.dtype != torch.float32:
        return None
    if name == ascend_env.TARGET_ASCEND and A.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return None
    if A.dtype != W.dtype or not (A.is_contiguous() and W.is_contiguous() and C.is_contiguous()):
        return None
    kwargs = {"n": n, "k": k, "bm": BLOCK_M, "bn": bn, "bk": bk, "act_mode": ACT_MODE[act]}
    impl = gemm_asc_impl if name == ascend_env.TARGET_ASCEND else gemm_cpu_impl
    key = f"gemm[{name}]|n{n}k{k}b{BLOCK_M}x{bn}x{bk}a{act}"
    return {"target": name, "key": key, "kwargs": kwargs, "impl": impl, "shape": (n, k)}


def run(A: torch.Tensor, W: torch.Tensor, bias: torch.Tensor | None, C: torch.Tensor,
        spec: dict[str, Any]) -> bool:
    """真跑一次内核并把结果写进 C；编译产物取不到就返回 False（由入口层落回退）。

    缓存键不含行数：同一份产物服务 8、33、8192 等不同批大小（m 是 PrimFunc 里的动态符号），
    换批重编的秒级开销会吞掉收益。首编用当次张量做形状追踪，之后只喂出口缓冲。

    白话：模具没开成就说没开成，返回一个假字让上层换手工做法，绝不"假装算过"把空表交出去。

    :returns: True 表示 C 已被内核写入。
    """
    zero = bias if bias is not None else torch.zeros(spec["shape"][0], dtype=torch.float32,
                                                     device=A.device)
    kernel = ascend_env.get_compiled(
        spec["key"],
        lambda: tilelang.compile(spec["impl"](A, W, zero, C, **spec["kwargs"]),
                                 out_idx=[], **ascend_env.compile_kwargs(spec["target"])),
        spec["target"])
    if kernel is None:
        return False
    kernel(A, W, zero, C)
    return True
'''
p.write_text(head + tail)
import ast
ast.parse(p.read_text())
print("gemm_asc rebuilt from pyc, SYNTAX OK, lines =", len(p.read_text().splitlines()))
