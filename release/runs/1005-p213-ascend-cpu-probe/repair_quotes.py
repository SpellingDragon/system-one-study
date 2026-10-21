"""修上一次 patch_docstrings.py 造成的语法破坏（只有 gemm_asc.py 被写坏了）。

上一版把文案当字面量插入时**漏了三引号**，于是 "被追踪的那一层……" 变成了裸代码行；脚本在
ast 校验处抛异常，因此只有第一个文件 gemm_asc.py 落盘受损，其余文件未被触碰。
本脚本按"从裸文案行开始、到下一个 with *.Kernel( 边界为止"删掉这些插入块，把文件还原成
插入前的形态（随后再跑修好的插入脚本）。
"""
from pathlib import Path

BAD_HEADS = ("被追踪的那一层", "白话", "行数是多少都不影响这套切法", "算完先在公共小格子里")

for rel in ["ascend/kernels/gemm_asc.py"]:
    p = Path(rel)
    lines = p.read_text().splitlines(keepends=True)
    out, i, removed = [], 0, 0
    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith(BAD_HEADS):
            # 一直丢到下一个 `with ...Kernel(` 边界（边界本身要保留）
            while i < len(lines) and not lines[i].strip().startswith("with ") \
                    and ".Kernel(" not in lines[i]:
                i += 1
                removed += 1
            continue
        out.append(ln)
        i += 1
    p.write_text("".join(out))
    print(f"{rel}: removed {removed} 行裸文案")

import ast  # noqa: E402
for rel in ["ascend/kernels/gemm_asc.py"]:
    ast.parse(Path(rel).read_text())
    print(f"{rel}: SYNTAX OK")
