"""ruff 在本 venv 里装不上（任务书禁 pip install），用这个 AST 体检当**替代凭据**。

【做什么】对 ascend/ 下全部 .py 查四类本域真会踩的雷：语法、死引用 import（F401 等价）、
同作用域重复定义（F811 等价）、坑①守卫（方言件的注解被 PEP 563 字符串化后闭包常量丢失）。
【怎么做】① 死引用＝ast 取 import 的名字后，在"剔掉 import 行本身"的源码里搜不到；
② 坑①守卫的判据是**确切的失效条件**而不是文件名：文件带 future-import 时，逐个查被
`@prim_func` 追踪的函数，注解里出现的自由名若在「函数体加载过的名字 ∪ 形参名 ∪ 模块顶层名」
之外，就说明这个名字只存在于注解里——PEP 563 后它没有闭包单元，tilelang 的
get_func_nonlocals 拿不到，编译期必炸 NameError（gemm 件的 m/n/k 正是这一类）。
③ 两个实测教训已内建：future 位值随 Python 版本变（3.12 是 0x1000000，写死 0x10000 会让守卫
静默空转），以及 `compile()` 默认继承调用方的 future 设置（本脚本自己就带 future-import，
不加 dont_inherit=True 会误报"所有文件都有 future-import"）。
【为什么这样替代】ruff 对本域的真实价值就是 F401/F811/语法错误加一条本域专属红线，四条全覆盖；
行宽、import 排序一类风格规则不属于本域验收项，缺 ruff 不阻塞交付但必须如实记录。

白话：这台机器装不了官方体检工具，就自己写一个只查"本域真会踩的四类雷"的小 checker，
每次交活前跑一遍，宁可自己多查一遍也不把带病的件交出去。
"""
from __future__ import annotations

import __future__
import ast
import py_compile
import sys
from pathlib import Path

CO_FUTURE_ANNOTATIONS = __future__.annotations.compiler_flag
ROOT = Path("ascend")
problems: list[str] = []
notes: list[str] = []
files = sorted(ROOT.rglob("*.py"))


def traced_funcs(tree: ast.AST):
    """挑出被 tilelang 追踪的函数：装饰器里出现 prim_func 的那些。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            if any("prim_func" in ast.dump(d) for d in node.decorator_list):
                yield node


def annotation_names(fn: ast.FunctionDef) -> set[str]:
    """形参注解里出现的所有名字（这些会被 PEP 563 变成字符串去求值）。"""
    found: set[str] = set()
    for arg in list(fn.args.posonlyargs) + list(fn.args.args) + list(fn.args.kwonlyargs):
        if arg.annotation is None:
            continue
        found |= {n.id for n in ast.walk(arg.annotation) if isinstance(n, ast.Name)}
    return found


def module_level_names(tree: ast.Module) -> set[str]:
    """模块顶层能查到的名字：顶层赋值/导入/函数/类，以及 import 的模块别名。"""
    out: set[str] = set()
    for node in tree.body:
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for t in targets:
            out |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                out.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
    return out


def annotation_risk(fn: ast.FunctionDef, globs: set[str]) -> set[str]:
    """返回"只在注解里出现"的危险名字集合（坑①的确切触发条件）。"""
    loaded: set[str] = set()
    # 只数函数体语句：注解本身也是 fn 子树的一部分，把 ast.walk(fn) 整体算进来会让
    # "只在注解里出现"的名字自己给自己作证，负控实测就是这样漏掉的
    for stmt in fn.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name):
                loaded.add(node.id)     # Load/Store 都算：名字在体里落地就有闭包单元
    params = {a.arg for a in
              list(fn.args.posonlyargs) + list(fn.args.args) + list(fn.args.kwonlyargs)}
    return annotation_names(fn) - loaded - params - globs


for path in files:
    src = path.read_text()
    # ① 语法必须能编译（产物写临时目录，不污染仓库）
    try:
        py_compile.compile(str(path), cfile="/tmp/_ascend_lint.pyc", doraise=True)
    except py_compile.PyCompileError as exc:
        problems.append(f"{path}: 语法错误 {exc.msg.splitlines()[0]}")
        continue
    tree = ast.parse(src)
    future_ann = bool(compile(src, str(path), "exec", dont_inherit=True)
                      .co_flags & CO_FUTURE_ANNOTATIONS)

    # ② 坑①守卫
    if future_ann:
        if path.name.endswith("_asc.py"):
            problems.append(f"{path}: 方言件带 future-import（坑①红线，件形一律禁）")
            continue
        globs = module_level_names(tree)
        risky = {}
        for fn in traced_funcs(tree):
            bad = annotation_risk(fn, globs)
            if bad:
                risky[fn.name] = sorted(bad)
        n_traced = sum(1 for _ in traced_funcs(tree))
        if risky:
            problems.append(f"{path}: 被追踪函数注解含「只在注解里出现」的名字 {risky}（坑①）")
        elif n_traced:
            notes.append(f"{path}: 带 future-import，{n_traced} 个被追踪函数的注解名字在体内也有"
                         f"出现（实测安全）")
        else:
            notes.append(f"{path}: 带 future-import，无被追踪函数（宿主件，不涉坑①）")

    imported: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported[(alias.asname or alias.name).split(".")[0]] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":      # __future__ 不是普通名字，ruff 也不报 F401
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                imported[alias.asname or alias.name] = node.lineno
    # ③ 死引用：剔掉 import 行本身再搜，避免"自己引用自己"的假阳性
    body = "\n".join(ln for i, ln in enumerate(src.splitlines(), 1)
                     if i not in set(imported.values()))
    for name, lineno in imported.items():
        if name not in body:
            problems.append(f"{path}:{lineno} 未使用的 import：{name}（F401 等价）")
    # ④ 同作用域重复定义（后者静默盖掉前者，是真实缺陷而非风格）
    for scope in ast.walk(tree):
        if not isinstance(scope, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            continue
        seen: dict[str, int] = {}
        for child in scope.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if child.name in seen:
                    problems.append(f"{path}:{child.lineno} 重复定义 {child.name}"
                                    f"（先前在 {seen[child.name]}）")
                seen[child.name] = child.lineno

print(f"扫描 {len(files)} 个文件（ascend/ 全量）")
for n in notes:
    print("  信息: " + n)
if problems:
    print("发现问题：")
    for p in problems:
        print("  " + p)
    sys.exit(1)
print("✅ AST 体检通过（语法 / F401 / F811 / 坑① future-import 守卫 四项皆清）")
