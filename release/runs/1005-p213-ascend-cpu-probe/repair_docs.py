"""修上一次插 docstring 时漏了三引号造成的语法破坏（只有 gemm_asc.py 落盘受损）。

上一版把文案当裸代码行插入后，脚本在 ast 校验处抛异常，其余文件未被触碰。
本脚本按"从裸文案行开始、直到下一个 with *.Kernel( 边界为止"删掉这些插入块，
把文件还原成插入前的形态（随后再跑修好的插入脚本）。
"""
import ast
from pathlib import Path

BAD_HEADS = ("被追踪的那一层", "白话", "行数是多少都不影响这套切法", "算完先在公共小格子里")
REL = "ascend/kernels/gemm_asc.py"

p = Path(REL)
lines = p.read_text().splitlines(keepends=True)
out, i, removed = [], 0, 0
while i < len(lines):
    ln = lines[i]
    if ln.strip().startswith(BAD_HEADS):
        while i < len(lines) and not lines[i].strip().startswith("with ") and ".Kernel(" not in lines[i]:
            i += 1
            removed += 1
        continue
    out.append(ln)
    i += 1
p.write_text("".join(out))
ast.parse(p.read_text())
print(REL, "removed", removed, "SYNTAX OK")
