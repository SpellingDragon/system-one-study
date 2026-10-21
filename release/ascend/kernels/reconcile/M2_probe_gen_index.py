#!/usr/bin/env python3
"""reconcile/M2_probe_gen_index.py — 给归档真机生成码（sys1_kernels.tgz）建**位宽/归约面/UB 布局**索引。

【为什么需要它】`DTYPE-TRACE`/`KERNEL-USED` 两张证照只看得见 **Python 层**：内核内部的累加位宽
不经 Python，证照量不到。唯一能直达那一层的证据是卡上落盘的 `device_kernel.asc` 正文本身。
本探针把 46 份生成码逐条解析成一张可 grep 的表，回答四个问题：

  ① GM 指针位宽（`__gm__ half*` = fp16 原生内核；`mixed` = fp16 主干 + fp32 表/出口）；
  ② **归约面的声明位宽**（主判据）。只看两处"归约落在哪儿"：自增式标量（`acc = (acc + …)`，
     类型由同体声明表回查）与 UB 上的就地累加写（`ub[..] = (ub[..] + x)`，看**视图类型**）。
     出现 `half`/`bfloat16_t` ⇒ `accum=FP16`（降位累加，违规）；只有 `float` ⇒ `accum=fp32`。
     `half condval` 这类**出口截断临时量**不算归约面（交货位宽本就要求出口降位），单列
     `half_temps` 供人工判读；
  ③ UB 各缓冲的**字节区间**（同一 UB 基址按"常数偏移"分块，extent 由该块各访问的最外层
     常量循环上界推）＋与 host_kernel.c 申报的动态 UB 字节数对账（实写最高字节 > 申报 = 踩
     UB 越界）⇒ 块间重叠 = 互相踩账。这两项 cpu 方言与 Python 证照都看不见；
  ④ `rsqrtf/sqrtf/expf/logf` 与非字面量**真除法**的落点行号（0/0、rsqrt(≤0) 类灾难候选面）。

【口径与局限（如实）】纯静态判读，不发射、不判数值；循环上界只认字面量（`i < 32`），认不了的
（`((m + 15) >> 4)`、`min(16, max(rows-…))`）在该块打 `DYN` 标记、不参与重叠判定，不猜数。
踩过的三个解析坑（都已在码里修掉，留此备案）：codegen 的 `/*0.000000e+00*/` 字面量注释要先整段
剥掉（否则注释里的 `/` 被当除法，v1 报出一整片假风险）；UB 下标外裹成对括号要先剥（否则 extent
全退化成 `[0,4)` 垃圾）；自增行的类型在**声明处**而非赋值处（`acc = (acc + …)` 无前缀），必须
用同体声明表回查，否则 46 枚全被判成"无归约面"。每条判决附行号，可在解压目录逐字复核。

用法：
  tar xzf release/runs/1011-p2-13-oncard-accept-2002/artifacts/sys1_kernels.tgz -C /tmp/m2tgz
  release/.venv/bin/python ascend/kernels/reconcile/M2_probe_gen_index.py [--root /tmp/m2tgz] [--dt half] [--tiles]
判决行：`M2-GEN <hash12> <kernel> gm=<..> accum=<fp32|FP16|n/a> fp16_reduce=<NONE|..> ub=<used/declared=..>`
末行：`M2-GEN-INDEX-PASS violations=0` / `M2-GEN-INDEX-FAIL violations=<n>`
"""
import argparse
import json
import os
import re
import sys

VIEW_BYTES = {"half": 2, "bfloat16_t": 2, "float": 4, "int32_t": 4, "int64_t": 8, "uint8_t": 1,
              "uint16_t": 2}
SIG_RE = re.compile(r'extern "C" __global__[^;]*?void\s+(\w+)\s*\(([^)]*)\)')
UB_ACCESS_RE = re.compile(r'\(\(__ubuf__\s+(\w+)\s*\*\)\s*buf_dyn_shmem\)\s*\[([^\]]*)\]')
LOOP_RE = re.compile(r'for\s*\(int32_t\s+(\w+)\s*=\s*0;\s*\1\s*<\s*([^;]+?);')
MATH_RE = re.compile(r'\b(rsqrtf|sqrtf|expf|logf|powf)\s*\(')
#: 浮点标量声明（codegen 形态：`float acc = float(0x0p+0f);` / `float condval;`）
DECL_RE = re.compile(r'^\s*(float|half|bfloat16_t|int32_t|int64_t|uint8_t)\s+(\w+)\s*(?:=.*)?;',
                     re.M)
#: 自增式归约行（类型在声明处，这里只认变量名；`[^;=]*` 保证 rhs 的 lhs 不是别的赋值）
SELF_ACC_RE = re.compile(r'(?<![\w*])(\w+)\s*=\s*[^;=]*\b\1\b[^;]*[+*]')
#: UB 上就地累加写：`ub[..] = (ub[..] + x)`（rhs 前允许多层括号）
UB_ACC_RE = re.compile(r'\(\(__ubuf__\s+(\w+)\s*\*\)\s*buf_dyn_shmem\)\s*\[[^\]]*\]\s*='
                       r'\s*[\(\s]*\(\(__ubuf__\s+(\w+)\s*\*\)')
COMMENT_RE = re.compile(r'/\*.*?\*/')
FP16_T = ("half", "bfloat16_t")


def strip_comments(text):
    return COMMENT_RE.sub('', text)


def _peel(expr):
    """剥掉成对的最外层括号：`(((a * 32) + b) + 1024)` → `(a * 32) + b + 1024` 形态。"""
    e = expr.strip()
    while e.startswith("(") and e.endswith(")"):
        depth, balanced = 0, True
        for i, ch in enumerate(e):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(e) - 1:
                    balanced = False
                    break
        if not balanced:
            break
        e = e[1:-1].strip()
    return e


def _split_top(expr):
    """按顶层 +- 切成带符号的项。"""
    parts, depth, cur = [], 0, ""
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch in "+-" and depth == 0 and cur.strip():
            parts.append(cur)
            cur = ch
            continue
        cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def _parse_index(expr):
    """返回 (常数, [(变量, 系数)], 未解析项[])。支持 `A`、`A * 数字`、`数字 * A`、`(T)A * 数字`。"""
    const, vars_, rest = 0, [], []
    for p in _split_top(_peel(expr)):
        neg = p[0] == "-"
        body = _peel(p.lstrip("+-").strip())
        if re.fullmatch(r'\d+', body):
            v = int(body)
            const += -v if neg else v
            continue
        m = re.fullmatch(r'(?:\((?:int32_t|int64_t|uint32_t|long)\)\s*)?(\w+)\s*\*\s*(\d+)', body)
        if m:
            name, coef = m.group(1), int(m.group(2))
        else:
            m = re.fullmatch(r'(\d+)\s*\*\s*(?:\((?:int32_t|int64_t)\)\s*)?(\w+)', body)
            if m:
                name, coef = m.group(2), int(m.group(1))
            elif re.fullmatch(r'\w+', body):
                name, coef = body, 1
            else:
                rest.append(body)
                continue
        if name.isdigit():
            const += int(name) * (1 if not neg else -1)
            continue
        vars_.append((name, coef if not neg else -coef))
    return const, vars_, rest


def _enclosing_bounds(lines):
    """每行起手的 {循环变量: 常量上界}（上界非字面量 ⇒ None=动态）。"""
    stack, out, depth = [], [], 0
    for ln in lines:
        out.append({v: ub for v, ub, _d in stack})
        m = LOOP_RE.search(ln)
        if m:
            lit = re.fullmatch(r'\d+', m.group(2).strip())
            stack.append((m.group(1), int(lit.group(0)) if lit else None, depth + 1))
        depth += ln.count("{") - ln.count("}")
        while stack and stack[-1][2] > depth:
            stack.pop()
    return out


def _tile_ranges(lines):
    """UB 块表：{base_bytes: {"view","lo","hi","lines","dyn"}}，extent 由常量循环上界推。"""
    scopes = _enclosing_bounds(lines)
    groups = {}
    for i, ln in enumerate(lines):
        for mm in UB_ACCESS_RE.finditer(ln):
            view, idx = mm.group(1), mm.group(2)
            vb = VIEW_BYTES.get(view, 4)
            const, vs, rest = _parse_index(idx)
            lo, hi = const * vb, (const + 1) * vb
            dyn = bool(rest)
            for v, coef in vs:
                b = scopes[i].get(v)
                if b is None:
                    dyn = True
                    continue
                lo += min(0, (b - 1) * coef) * vb
                hi += max(0, (b - 1) * coef) * vb
            base = const * vb
            g = groups.setdefault(base, {"view": view, "base": base, "lo": lo, "hi": hi,
                                         "lines": set(), "dyn": dyn})
            g["lo"], g["hi"] = min(g["lo"], lo), max(g["hi"], hi)
            g["lines"].add(i + 1)
            g["dyn"] = g["dyn"] or dyn
    return groups


def _declared_ub(root_dir):
    """host_kernel.c 里申报的动态 UB 字节数（codegen 把 shmem 大小作为 FFI 实参塞进去）。"""
    p = os.path.join(root_dir, "host_kernel.c")
    if not os.path.isfile(p):
        return None
    hits = [int(x) for x in re.findall(r'v_int64\) = \(int64_t\)(\d+);', open(p).read())
            if int(x) >= 32]
    return max(hits) if hits else None


def audit(path):
    raw = open(path).read()
    src = strip_comments(raw)
    lines = src.splitlines()
    body_at = next((i for i, l in enumerate(lines) if l.startswith("extern \"C\"")), 0)
    sig = SIG_RE.search(src)
    name, argtxt = (sig.group(1), sig.group(2)) if sig else ("?", "?")
    half = len(re.findall(r'__gm__\s+half\s*\*', argtxt))
    bf16 = len(re.findall(r'__gm__\s+bfloat16_t\s*\*', argtxt))
    flt = len(re.findall(r'__gm__\s+float\s*\*', argtxt))
    gm = ("bf16" if bf16 else "fp16" if (half and not flt) else
          "mixed(half+float)" if half else ("fp32" if flt else "int-only"))
    decl_map = {}
    decls, half_temps = [], []
    for i, ln in enumerate(lines[body_at:], start=body_at + 1):
        m = DECL_RE.match(ln)
        if m:
            decl_map.setdefault(m.group(2), m.group(1))
        if m and m.group(1) in ("float", "half", "bfloat16_t"):
            decls.append(f"{m.group(1)} {m.group(2)}@{i}")
            if m.group(1) in FP16_T:
                half_temps.append(f"{m.group(1)} {m.group(2)}@{i}")
    body = "\n".join(lines[body_at:])
    self_acc = sorted(set(f"{decl_map.get(v, '?')} {v}" for v in SELF_ACC_RE.findall(body)))
    ub_acc = sorted(set(f"(__ubuf__ {a}*) += (__ubuf__ {b}*)" for a, b in UB_ACC_RE.findall(src)))
    fp16_reduce = [s for s in self_acc if s.split()[0] in FP16_T] + [
        s for s in ub_acc if s.split()[1].strip("()*") in FP16_T]
    tiles = _tile_ranges(lines)
    spans = [(g["base"], g["lo"], g["hi"], g["view"], sorted(g["lines"]), g["dyn"])
             for g in tiles.values()]
    overlap = []
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            a, b = spans[i], spans[j]
            if not a[5] and not b[5] and a[1] < b[2] and b[1] < a[2]:
                overlap.append(f"{a[3]}@{a[0]}[{a[1]},{a[2]})~{b[3]}@{b[0]}[{b[1]},{b[2]})")
    hi_used = max([s[2] for s in spans if not s[5]] or [0])
    declared = _declared_ub(os.path.dirname(path))
    ub_verdict = "n/a" if declared is None or not hi_used else (
        "OK" if hi_used <= declared else f"OVERRUN({hi_used}>{declared})")
    risk = []
    for i, ln in enumerate(lines):
        if i < body_at:
            continue
        for m in MATH_RE.finditer(ln):
            risk.append(f"{m.group(1)}@{i+1}")
        for m in re.finditer(r'([^\s(]+)\s*/\s*([^;\s)]+)', ln):
            den = m.group(2)
            if re.fullmatch(r'float\(0x[0-9a-fp]+\)?', den) or den.isdigit():
                continue
            risk.append(f"div@{i+1}:{m.group(0).strip()[:44]}")
    accum = "FP16" if fp16_reduce else ("fp32" if self_acc or ub_acc else "NO-REDUCE")
    return {"hash": os.path.basename(os.path.dirname(path)), "name": name, "gm": gm,
            "gm_args": " ".join(argtxt.split()), "accum": accum,
            "decls": decls, "half_temps": half_temps, "self_acc": self_acc, "ub_acc": ub_acc,
            "fp16_reduce": fp16_reduce, "tiles": sorted(spans), "overlap": overlap,
            "ub_declared": declared, "ub_used": hi_used, "ub_verdict": ub_verdict,
            "risk": sorted(set(risk)), "size": len(raw)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/m2tgz")
    ap.add_argument("--dt", default="", help="过滤 gm 档子串：fp16/mixed/half/fp32/bf16（空=全部）")
    ap.add_argument("--name", default="", help="过滤内核名子串")
    ap.add_argument("--tiles", action="store_true", help="打印 UB 块表明细")
    args = ap.parse_args()
    base = os.path.join(args.root, "0.1.15", "linux-aarch64", "kernels")
    if not os.path.isdir(base):
        print(f"M2-GEN-INDEX-FAIL: 找不到 {base}（先 tar xzf sys1_kernels.tgz -C {args.root}）")
        return 2
    rows = []
    for d in sorted(os.listdir(base)):
        p = os.path.join(base, d, "device_kernel.asc")
        if not os.path.isfile(p):
            continue
        r = audit(p)
        try:
            man = json.load(open(os.path.join(base, d, "manifest.json")))
            r["asc_sha256"] = man["files"]["device_kernel.asc"]["sha256"]
        except Exception:
            r["asc_sha256"] = "?"
        rows.append(r)
    if args.dt:
        rows = [r for r in rows if args.dt in r["gm"]]
    if args.name:
        rows = [r for r in rows if args.name in r["name"]]
    print(f"M2-GEN-INDEX kernels={len(rows)} root={base}")
    bad = 0
    for r in sorted(rows, key=lambda x: (x["gm"], x["name"], x["hash"])):
        hits = (1 if r["accum"] == "FP16" else 0) + (
            1 if str(r["ub_verdict"]).startswith("OVERRUN") else 0) + len(r["overlap"])
        bad += hits
        print(f"M2-GEN {r['hash'][:12]} {r['name']:<22} gm={r['gm']:<18} bytes={r['size']:<6} "
              f"accum={r['accum']:<10} self_acc={','.join(r['self_acc']) or '-'} "
              f"ub_acc={len(r['ub_acc'])} fp16_reduce={r['fp16_reduce'] or 'NONE'} "
              f"half_temps={r['half_temps'] or 'NONE'} ub={r['ub_used']}/{r['ub_declared']}"
              f"={r['ub_verdict']} tiles={len(r['tiles'])} overlap={r['overlap'] or 'NONE'} "
              f"violations={hits}")
        print(f"     decls[{r['hash'][:12]}] {', '.join(r['decls'][:12]) or '-'}")
        if r["risk"]:
            print(f"     risk[{r['hash'][:12]}] " + " ; ".join(r["risk"]))
        if args.tiles:
            for (base_, lo, hi, view, lns, dyn) in r["tiles"]:
                print(f"     UB[{r['hash'][:12]}] view={view:<10} base={base_:<6} "
                      f"bytes=[{lo},{hi}) lines={lns[:8]}{'...' if len(lns) > 8 else ''}"
                      f"{' DYN' if dyn else ''}")
        print(f"     sig[{r['hash'][:12]}] {r['gm_args']}")
    print("M2-GEN-SHA " + " ".join(f"{r['hash'][:12]}={r['asc_sha256'][:12]}" for r in rows))
    print(f"M2-GEN-INDEX-{'FAIL' if bad else 'PASS'} violations={bad} "
          f"(判据：half/bf16 归约面、实写字节越 UB 申报、UB 块重叠)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
