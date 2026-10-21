#!/usr/bin/env python3
"""reconcile/U2_probe_k48_gen_digest.py — 归档真机生成码的 **DMA 形态普查**（取证脚本，只读）。

【为什么需要它】"本地不复现 ⇒ 嫌疑在装填面"这个结论要靠一条统计支撑：**全盘 46 枚真机产物里，
究竟有几枚真正发过多突发跨步的 GM→UB DMA**。若只有 k=48/k=96 那几枚，则"唯一变量是 K=48"
从"观察"升级为"生成码层面的完备枚举"。

【做法】对解包后的每个内核目录，从 `device_kernel.asc` 抠四个 token（都是 codegen 落码的
字面常数，不猜语义）：
  1. 内核名与签名位宽（`extern "C" __global__ __vector__ void <name>(__gm__ <dtype>* ...)`）
  2. 每条 `asc_copy_gm2ub_align(...)` 的实参 (blockCount, blockLen, …, srcStride, dstStride)
  3. A 侧标量读的行步长常数 `* (int64_t)K`（= 该内核的 k）
  4. 是否存在 K 轮外层循环（`for (int32_t ko = 0; ko < NK`）
判据（与 `src/ascend/op/copy.cc` 的 plan_single_row 等价、纯文本可判）：
  **SINGLE（步长不参与）= blockCount==1 或 srcStride==blockLen；否则 STRIDED（多突发跨步）**。

【跑法】
  cd release
  .venv/bin/python ascend/kernels/reconcile/U2_probe_k48_gen_digest.py \
      --dir scratch/U2_k48/0.1.15/linux-aarch64/kernels
  # 解包命令（一手件）：tar -xzf runs/1011-p2-13-oncard-accept-2002/artifacts/sys1_kernels.tgz -C scratch/U2_k48
"""
from __future__ import annotations

import argparse
import os
import re
import sys

RE_SIG = re.compile(r"__global__\s+__\w+__\s+void\s+(\w+)\s*\(([^)]*)\)")
RE_DMA = re.compile(r"asc_copy_gm2ub_align\((.*?)\);", re.S)
RE_ARGS = re.compile(r"(?:^|,\s*)(?:static_cast<[^>]*>)?\(?[^,()]*\)?,?\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(?:static_cast<asc_load_l2_cache_mode>\((\d+)\)|\w+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
RE_K = re.compile(r"\*\s*\(int64_t\)(\d+)\)")
RE_KO = re.compile(r"for \(int32_t ko = 0; ko < (\d+)")

CACHE_DIR_FALLBACK = "0.1.15/linux-aarch64/kernels"


def _split_top_level(argstr: str) -> list[str]:
    """按**顶层逗号**切实参（圆括号/尖括号内不切）——实参里有 `((int32_t)block_idx)*1536`
    这类带常数的索引表达式，用"末 8 个整数"这种粗取会串槽，故必须结构化切。"""
    out, depth, cur = [], 0, ""
    for ch in argstr:
        if ch in "(<":
            depth += 1
        elif ch in ")>":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _dma_calls(src: str) -> list[tuple[int, int, int, int]]:
    """抠出每条 asc_copy_gm2ub_align 的 (blockCount, blockLen, srcStride, dstStride)（字节）。"""
    res = []
    key = "asc_copy_gm2ub_align("
    i = 0
    while True:
        i = src.find(key, i)
        if i < 0:
            break
        depth, j = 0, i + len(key) - 1
        while j < len(src):
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        args = _split_top_level(src[i + len(key):j])
        tail = args[2:]                      # 丢掉 dst / src 两个指针表达式
        nums = [int(x) for x in (
            re.fullmatch(r"\s*(-?\d+)\s*", a).group(1) if re.fullmatch(r"\s*(-?\d+)\s*", a)
            else (re.search(r"\((\s*-?\d+\s*)\)\s*$", a).group(1) if a.startswith("static_cast")
                  else None) for a in tail) if x is not None]
        # 槽 0（nBurst）在全部产物里都是字面常数；槽 1/6/7 可能是 `min(...)*128` 这类
        # **运行期表达式**（add_ln / ln_bwd 的尾块长度就是），此时不必猜——因为
        # plan_single_row 的第一条件 `n_rows == 1` 已经独立决定了形态：blockCount==1 ⇒ SINGLE。
        bc = nums[0] if nums else None
        bl = nums[1] if len(nums) > 1 else None
        ss = nums[6] if len(nums) == 8 else None
        ds = nums[7] if len(nums) == 8 else None
        expr = any(not re.fullmatch(r"\s*-?\d+\s*", a) for a in tail[1:7])
        res.append((bc, bl, ss, ds, "expr-len" if expr and ss is None else ""))
        i = j
    return res


def digest(path: str) -> dict:
    src = open(path, encoding="utf-8", errors="replace").read()
    sig = RE_SIG.search(src)
    name = sig.group(1) if sig else "?"
    dt = "fp16" if re.search(r"__gm__\s+half|__gm__\s+float16", src) else (
         "bfloat16" if "bfloat16" in src else "fp32")
    ks = sorted({int(x) for x in RE_K.findall(src)})
    ko = RE_KO.search(src)
    dma = _dma_calls(src)
    forms = {"SINGLE": 0, "STRIDED": 0}
    for bc, bl, ss, ds, note in dma:
        if bc is None:
            forms["UNPARSED"] = forms.get("UNPARSED", 0) + 1
            continue
        if bc == 1:
            forms["SINGLE"] += 1                     # 单发连续：步长槽根本不参与
        elif ss is None:
            forms["STRIDED_EXPR"] = forms.get("STRIDED_EXPR", 0) + 1
        else:
            forms["SINGLE" if ss == bl else "STRIDED"] += 1
    return {"name": name, "dtype": dt, "k_consts": ks, "nk": int(ko.group(1)) if ko else 1,
            "n_dma": len(dma), "forms": forms, "dma": [d[:4] for d in dma]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="解包后的 kernels 目录")
    a = ap.parse_args()
    root = a.dir
    if root is None:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.join(here, "..", "..", "..", "scratch", "U2_k48", CACHE_DIR_FALLBACK)
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        print("U2-DIGEST-ABSENT %s（先解包 sys1_kernels.tgz）" % root)
        return 2
    rows = []
    for d in sorted(os.listdir(root)):
        asc = os.path.join(root, d, "device_kernel.asc")
        if not os.path.isfile(asc):
            continue
        rows.append((d[:12], digest(asc)))
    gemm = [r for r in rows if r[1]["name"].startswith("gemm_impl")]
    strided = [(h, r) for h, r in rows
               if r["forms"]["STRIDED"] + r["forms"].get("STRIDED_EXPR", 0) > 0]
    unparsed = [(h, r) for h, r in rows if r["forms"].get("UNPARSED")]
    print("U2-DIGEST root=%s 条目=%d（gemm_impl=%d，其余=%d）" % (
        root, len(rows), len(gemm), len(rows) - len(gemm)), flush=True)
    print("U2-DIGEST 走 STRIDED（多突发跨步）DMA 的内核：%d 枚" % len(strided), flush=True)
    for h, r in strided:
        print("  %s %-16s %-8s k_consts=%-10s nk=%-2d dma=%s" % (
            h, r["name"], r["dtype"], r["k_consts"], r["nk"], r["dma"]), flush=True)
    print("\nU2-DIGEST 全表（内核 / 位宽 / A 行步长常数 / K 轮数 / DMA 条数 / 形态 / 实参）", flush=True)
    for h, r in rows:
        print("  %s %-22s %-8s k=%-10s nk=%-2d n_dma=%-2d %-8s %s" % (
            h, r["name"], r["dtype"], r["k_consts"], r["nk"], r["n_dma"],
            ("STRIDED" if r["forms"]["STRIDED"] else
             "STRIDED(len=运行期表达式)" if r["forms"].get("STRIDED_EXPR") else
             ("SINGLE" if r["n_dma"] else "no-DMA"))
            + (" UNPARSED=%d" % r["forms"]["UNPARSED"] if r["forms"].get("UNPARSED") else ""),
            r["dma"] if r["n_dma"] else "—"), flush=True)
    if unparsed:
        print("\nU2-DIGEST ⚠ UNPARSED 槽数异常（不猜，如实登记）：%s" % [h for h, _ in unparsed], flush=True)
    print("\nU2-DIGEST-VIEW 仅 gemm_impl（前向 linear 产物）：", flush=True)
    for h, r in gemm:
        print("  %s %-8s k=%-10s nk=%-2d %-8s %s" % (
            h, r["dtype"], r["k_consts"], r["nk"],
            "STRIDED" if r["forms"]["STRIDED"] else "SINGLE", r["dma"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
