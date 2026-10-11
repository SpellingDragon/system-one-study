"""A3_probe_clone_deps.py — 只读探针：卡窗所需文件是否**全部已入库 origin/main**。

卡上 `git clone --depth 1 https://github.com/SpellingDragon/system-one-study.git` 拿到的就是
origin/main 那棵树（A3 已核 `git ls-remote --symref origin HEAD` → refs/heads/main）。
本探针做三件事，零写入：
  ① 从 card_run.sh 里抠出所有字面路径依赖（patches/*、run_numerics.py、run_delta_card.py、
     release/ascend/train_step.py …）；
  ② 从三个入口（run_numerics.py / run_delta_card.py / train_step.py）做 import 闭包
     （含 importlib.import_module 的字面量、open/np.load 的字面量相对路径），递归到仓内文件；
  ③ 把②得到的每一个候选路径拿去和 `git ls-tree -r origin/main --name-only` 比对，
     列出任何"本地有但未入库"或"引用了但树里没有"的路径 —— 漏一个 = 废一次窗。

用法（system-one 仓根）：
    git show origin/main:release/ascend/... 之类不需要；本脚本自己调 git ls-tree。
    python3 release/ascend/kernels/reconcile/A3_probe_clone_deps.py
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]          # .../system-one
P910B = REPO / "release" / "ascend" / "port910b"
REL = REPO / "release"

ENTRIES = [
    P910B / "run_numerics.py",
    P910B / "run_delta_card.py",
    REL / "ascend" / "train_step.py",
]

# run_numerics.py / run_delta_card.py 的 sys.path 规则（照抄源码，别臆测）
EXTRA_SYS_PATHS = [REL, P910B / "attempts" / "C", P910B / "attempts" / "D",
                   P910B / "attempts" / "E", P910B / "attempts" / "F"]


def tracked() -> set[str]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "origin/main",
                          "--name-only"], capture_output=True, text=True, check=True)
    return {l for l in out.stdout.splitlines() if l}


def is_third_party(mod: str) -> bool:
    top = mod.split(".")[0]
    return top not in ("ascend", "sys1", "common", "numpy", "torch") or top in (
        "numpy", "torch", "tilelang", "os", "sys", "argparse", "importlib", "time",
        "warnings", "pathlib", "collections", "typing", "re", "math", "functools",
        "itertools", "dataclasses", "json", "subprocess", "tempfile", "hashlib",
    )


def resolve(mod: str) -> Path | None:
    """把模块名按 run_numerics 的 sys.path 规则解析成仓内文件。"""
    rel = mod.replace(".", "/")
    for root in EXTRA_SYS_PATHS + [REL / "ascend" / "kernels", P910B]:
        for cand in (root / f"{rel}.py", root / rel / "__init__.py"):
            if cand.exists():
                return cand
    # ascend.kernels.xxx 形态
    cand = REL / "ascend" / (rel + ".py")
    if cand.exists():
        return cand
    cand = REL / (rel + ".py")
    return cand if cand.exists() else None


def data_refs(src: str, base: Path) -> list[str]:
    """抓 open(...)/np.load(...) 里的字面相对路径（含 os.path.join(ATT, "D", "x.npz")）。"""
    hits = []
    for m in re.finditer(r'(?:open|np\.load|np\.savez)\(\s*"([^"]+)"', src):
        hits.append(m.group(1))
    for m in re.finditer(r'os\.path\.join\(\s*(HERE|ATT|RELEASE_ROOT)\s*((?:,\s*"[^"]+"\s*)*)\)', src):
        parts = re.findall(r'"([^"]+)"', m.group(2))
        root = {"HERE": P910B, "ATT": P910B / "attempts", "RELEASE_ROOT": REL}[m.group(1)]
        hits.append(str(root.joinpath(*parts).relative_to(REPO)))
    return [h for h in hits if not h.startswith(("/tmp", "http"))]


def closure(entry: Path) -> tuple[set[str], set[str]]:
    seen: set[Path] = set()
    pend = [entry]
    data = set()
    while pend:
        f = pend.pop()
        if f in seen or not f.exists():
            continue
        seen.add(f)
        src = f.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            print(f"[warn] AST fail {f}: {e}")
            continue
        mods = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                if n.module and n.level == 0:
                    mods.append(n.module)
                    mods += [n.module + "." + a.name for a in n.names]
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                    and n.func.attr == "import_module" and n.args \
                    and isinstance(n.args[0], ast.Constant):
                mods.append(str(n.args[0].value))
        for m in mods:
            if m.split(".")[0] in ("tilelang", "torch", "numpy", "triton", "tvm"):
                continue
            r = resolve(m)
            if r is not None:
                pend.append(r)
        for d in data_refs(src, f):
            data.add(d if d.startswith("release/") else os.path.relpath(
                (f.parent if not d.startswith("attempts") else P910B) / d, REPO))
    return {str(p.relative_to(REPO)) for p in seen}, data


def main() -> int:
    TR = tracked()
    all_files: set[str] = set()
    all_data: set[str] = set()
    for e in ENTRIES:
        fs, ds = closure(e)
        all_files |= fs
        all_data |= ds
        print(f"[entry] {e.relative_to(REPO)}: {len(fs)} 个仓内 py 在闭包里, {len(ds)} 个数据引用")

    # card_run.sh 的字面依赖
    sh = (P910B / "card_run.sh").read_text()
    lit = {"release/ascend/port910b/" + x for x in re.findall(r'patches/[A-Za-z0-9_./]+', sh)}
    lit |= {f"release/ascend/port910b/{p}" for p in
            re.findall(r'run_[a-z_]+\.py', sh)}
    lit |= {"release/ascend/train_step.py", "release/ascend/port910b/card_run.sh"}
    # patches 里通配的实目录内容也要列全（cp patches/c_api/*.h 之类）
    for sub in ("c_api", "simt_api"):
        d = P910B / "patches" / sub
        if d.is_dir():
            lit |= {f"release/ascend/port910b/patches/{sub}/{x.name}" for x in sorted(d.iterdir())}

    files_only = {d for d in all_data if (REPO / d).is_file()}
    dirs = sorted(all_data - files_only)
    for d in dirs:
        print(f"[note] 目录级引用（非单文件，不参与入库判定）：{d}")
    need = sorted(all_files | files_only | lit)
    need = [n for n in need if not n.endswith('/')]
    missing = [n for n in need if n not in TR]
    print(f"\nNEED_TOTAL={len(need)}  IN_ORIGIN_MAIN={len(need) - len(missing)}  MISSING={len(missing)}")
    for m in missing:
        print(f"  MISSING-FROM-ORIGIN-MAIN: {m}")
    print("\n--- 依赖闭包（产品腿，train_run 会用到的仓内 py）---")
    for f in sorted(all_files):
        print(f"  {f}" + ("" if f in TR else "   <== NOT TRACKED"))
    print("\n--- card_run.sh 字面依赖 ---")
    for l in sorted(lit):
        print(f"  {l}" + ("" if l in TR else "   <== NOT TRACKED"))
    print("\n--- 数据引用 ---")
    for d in sorted(all_data):
        print(f"  {d}" + ("" if d in TR else "   <== NOT TRACKED"))
    print("\nA3-CLONE-DEPS-" + ("PASS" if not missing else "FAIL"))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
