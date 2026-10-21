"""只摊 module 层指令流：确认受损前的 gemm_asc 在 run 之后还有什么（__all__？别的函数？）。"""
import dis
import marshal
from pathlib import Path

code = marshal.loads(Path("ascend/kernels/__pycache__/gemm_asc.cpython-312.pyc").read_bytes()[16:])
print("NAMES:", code.co_names)
ins = list(dis.get_instructions(code))
tail = [i for i in ins if i.starts_line or i.opname in ("STORE_NAME", "LOAD_CONST", "BUILD_LIST", "STORE_SUBSCR")]
for i in ins[-40:]:
    print(i.lineno, i.opname, i.argrepr)
