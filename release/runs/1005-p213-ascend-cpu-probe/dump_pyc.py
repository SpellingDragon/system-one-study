"""从受损前的 .pyc 里取回 gemm_asc 的真实 docstring 与指令流（重建被误删的行只用这一个凭据）。

pyc 里存的是编译产物：函数 docstring 作为常量原样保留，代码体只能靠指令流反推，所以本脚本
把两者都摊开打印，人工按同包件（rope_asc/gdn_asc）的同构写法核对补齐——不看记忆、不看猜测。
"""
import dis
import importlib.util
import marshal
import sys
from pathlib import Path

PYC = Path("ascend/kernels/__pycache__/gemm_asc.cpython-312.pyc")
data = PYC.read_bytes()
code = marshal.loads(data[16:])          # 跳过头部 16 字节的 pyc 元信息


def walk(co, depth=0):
    print("=" * 20, co.co_name, f"(args={co.co_varnames[:co.co_argcount]}, consts_str={len([c for c in co.co_consts if isinstance(c, str)])})")
    for c in co.co_consts:
        if isinstance(c, str) and len(c) > 40:
            print("--- DOC/STR ---")
            print(c)
    if depth < 2:
        for c in co.co_consts:
            if hasattr(c, "co_name"):
                walk(c, depth + 1)


walk(code)
print("#" * 30, "DIS: plan / run")
for c in code.co_consts:
    if hasattr(c, "co_name") and c.co_name in ("plan", "run"):
        print(">>>>", c.co_name)
        dis.dis(c)
