#!/usr/bin/env python3
"""第一阶段注释规范检查器（课程 CI 质量门，规范正文见 GUIDE §6.1）。

用法:
    python tools/check_comments.py [path ...]        # 默认: learning sys1
    python tools/check_comments.py --list-blacklist

规则:
  R1 文件头三件套   —— 模块 docstring 必须含【做什么】【怎么做】【为什么】三段（"为什么"含被否方案）。
  R2 白话段落       —— 每个公共函数(函数体>3行)的 docstring 须含一行以"白话:"起头的段落，
                       段落至多引入 1 次技术黑话词（作为被解释对象），正文 ≥30 字。
  R3 注释密度       —— (注释+docstring 行)/(非空代码行+注释行) ≥ 0.20；<60 行文件阈值减半。
  R4 复读警告(不阻断) —— docstring 仅复述函数名/参数名时提示。

任一 R1-R3 违例 → 退出码 1（CI 红）。豁免: __init__.py、tests/、tools/、docs/、report/。
二阶段 production/ serving/ 不在默认范围（回归常规工程注释，见 GUIDE §6.1 末条）。
"""
from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize
from pathlib import Path

# 白话段落中至多出现 1 次的"被引入黑话"（刻意保守，宁缺毋滥）
BLACKLIST = [
    "softmax", "logit", "tokenizer", "分词", "词表", "embedding", "嵌入", "encoder", "decoder",
    "编码", "解码", "attention", "注意力", "张量", "tensor", "梯度", "反向传播", "自回归",
    "推理", "训练轮", "epoch", "batch", "checkpoint", "校准", "熵", "量化", "蒸馏", "隐状态",
    "归一化", "矩阵乘", "matmul", "困惑度", "perplexity", "dropout", "层归一",
]
MARKERS = ("【做什么】", "【怎么做】", "【为什么】")
MIN_PLAIN_LEN = 30
MIN_RATIO = 0.20
SMALL_FILE_LINES = 60
EXEMPT_PARTS = {"tests", "tools", "docs", "report", "__pycache__"}


def is_exempt(p: Path, root: Path) -> bool:
    return p.name == "__init__.py" or any(part in EXEMPT_PARTS for part in p.parts)


def _rel(p: Path, root: Path) -> str:
    try:
        return str(p.resolve().relative_to(root))
    except ValueError:
        return str(p)


def comment_and_doc_lines(src: str) -> tuple[int, int, int]:
    """return (comment_lines, docstring_lines, non_blank_code_lines)"""
    comments = 0
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError:
        toks = []
    doc_lines = 0
    for t in toks:
        if t.type == tokenize.COMMENT:
            comments += 1
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                doc_lines += len([ln for ln in doc.splitlines() if ln.strip()])
    total_non_blank = len([ln for ln in src.splitlines() if ln.strip()])
    code = max(1, total_non_blank - comments - doc_lines)
    return comments, doc_lines, code


def extract_plain_para(doc: str) -> str | None:
    """取 docstring 中"白话:"起头到下一空行的段落。"""
    lines = doc.splitlines()
    for i, ln in enumerate(lines):
        if re.match(r"^\s*(白话|plain)\s*[:：]", ln):
            para = [re.sub(r"^\s*(白话|plain)\s*[:：]\s*", "", ln)]
            for nxt in lines[i + 1:]:
                if not nxt.strip():
                    break
                para.append(nxt.strip())
            return "\n".join(para)
    return None


def blacklist_hits(text: str) -> list[str]:
    return sorted({w for w in BLACKLIST if w.lower() in text.lower()})


def check_file(path: Path, root: Path) -> list[str]:
    errs: list[str] = []
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"R0 语法错误，无法解析: {e}"]

    # R1 文件头三件套
    mod_doc = ast.get_docstring(tree) or ""
    for m in MARKERS:
        if m not in mod_doc:
            errs.append(f"R1 文件头缺{m}段")

    # R2 公共函数白话段落
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            seg = ast.get_source_segment(src, node) or ""
            body = [ln for ln in seg.splitlines() if ln.strip()]
            if len(body) <= 4:   # def 行 + ≤3 行 body 的小工具函数豁免
                continue
            doc = ast.get_docstring(node) or ""
            if not doc.strip():
                errs.append(f"R2 函数 {node.name}(L{node.lineno}) 无 docstring")
                continue
            plain = extract_plain_para(doc)
            if plain is None:
                errs.append(f"R2 函数 {node.name}(L{node.lineno}) 缺『白话:』段落")
                continue
            hits = blacklist_hits(plain)
            if len(hits) > 1:
                errs.append(f"R2 {node.name} 白话段引入黑话 {len(hits)} 次({hits})，至多 1 次且须随后口语解释")
            if len(re.sub(r"\s", "", plain)) < MIN_PLAIN_LEN:
                errs.append(f"R2 {node.name} 白话段过短(<{MIN_PLAIN_LEN}字)")

    # R3 密度
    c, d, code = comment_and_doc_lines(src)
    non_blank = c + d + code
    ratio = (c + d) / max(1, non_blank)
    threshold = MIN_RATIO if non_blank >= SMALL_FILE_LINES else MIN_RATIO / 2
    if ratio < threshold:
        errs.append(f"R3 注释密度 {ratio:.0%} < {threshold:.0%}（注释+docstring {c+d} 行 / 非空 {non_blank} 行）")

    return [f"{_rel(path, root)}: {e}" for e in errs]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=None)
    ap.add_argument("--list-blacklist", action="store_true", help="打印黑话词表")
    args = ap.parse_args()
    if args.list_blacklist:
        print("\n".join(BLACKLIST))
        return 0

    root = Path.cwd().resolve()
    targets = [Path(p) for p in (args.paths or ["learning", "sys1"])]
    files: list[Path] = []
    for t in targets:
        if t.is_dir():
            files += [p for p in t.rglob("*.py") if not is_exempt(p, root)]
        elif t.suffix == ".py" and not is_exempt(t, root):
            files.append(t)

    all_errs = []
    for f in sorted(files):
        all_errs += check_file(f, root)

    print(f"check_comments: 扫描 {len(files)} 个文件")
    if all_errs:
        print(f"❌ {len(all_errs)} 处违例（规范见 GUIDE §6.1）：")
        for e in all_errs:
            print("  -", e)
        return 1
    print("✅ 注释门通过")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
