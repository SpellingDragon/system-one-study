"""只摊 plan() 的指令流：确定 docstring 之后到 if act not in SUPPORTED_ACTS 之间的真实语句与行号。"""
import dis
import marshal
from pathlib import Path

code = marshal.loads(Path("ascend/kernels/__pycache__/gemm_asc.cpython-312.pyc").read_bytes()[16:])
for c in code.co_consts:
    if hasattr(c, "co_name") and c.co_name == "plan":
        print("CONSTS:", [x for x in c.co_consts if not hasattr(x, "co_name")])
        dis.dis(c)
