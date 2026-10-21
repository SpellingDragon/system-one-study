#!/usr/bin/env python3
"""reconcile/M2_probe_dma_cfg.py — fp16 反向 NaN（P1-1p）与 K 非对齐真机错值（P1-1q）的**同根判据**。

【要回答的问题】P1-1p 的 NaN 生在 step0 反向。归档生成码已证"内核内累加面恒 fp32"
（`M2_probe_gen_index.py`：46 枚 `violations=0`，没有任何 half/bf16 归约面），所以嫌疑只剩
**MTE2 装填形态**这一层。本探针把这一层变成可机器判的两件事：

  A) **归档侧**：逐枚解析 `asc_copy_gm2ub_align / asc_copy_ub2gm_align` 的实参
     `(n_bursts, burst_bytes, src_stride_bytes, dst_stride_bytes)`（按顶层逗号切，两种 arity
     各自定位：gm2ub=10 参、ub2gm=7 参），按形态分档：
       SINGLE          n_bursts==1 或 src_stride==burst ⇒ 单发连续突发
       STRIDED-BIG     多行跨步，突发 >= 128B
       STRIDED-SMALL   多行跨步，突发 <  128B   ← 坏档候选（阈值来源见下）
     并校验各字节量是否 32 的倍数（`port910b_compat.h:599 TL910B_BYTES_TO_BLK(x)=x/32` 是
     **整除截断**，非 32 倍数会静默少搬）。
  B) **产品侧**：不读归档、只用产品件自己的 `gemm_asc.plan()` 现场算 (n,k,位宽) 会选出的
     `bn/bk`，再按 codegen 的 `plan_single_row` 条件（`tilelang/src/ascend/op/copy.cc:286`：
     单行 或 src_stride==row_bytes）复算它**会编成哪一档** ⇒ "形状 + 位宽"直接映射到 DMA 档。

【阈值 128B 从何而来（不是规范，是两点真机实测夹出来的）】
  · 坏：`5d732a8f…`（fp32, m16 n96 k48，bk=16 ⇒ 突发 64B / 跨距 192B）＝ P1-1q 已判真机数值错
    （notes §1：`rel=1.150e+00`，relu 档 96 个 NaN）。
  · 好：`a421afa3…`（fp32, n32 k96，bk=32 ⇒ 突发 128B / 跨距 384B）与全部 fp32 训练步件，
    同窗 fp32 训练步 PASS（loss 1.082224→0.945310）。
  ⇒ 本案只敢说"64B 突发档在卡上已被实测证坏、128B 档已被实测证好"，**不**断言 128 是硬件门槛。
【修复取向】把嫌疑形状推回 **SINGLE**（`bk==k` ⇒ src_stride==row_bytes ⇒ 单发连续），因为
SINGLE 档在**同一 fp16 位宽**上已被同窗真机证清白（`8f30a21f`/`e07da648` 前向 fp16 各 1×2048B
单发；H_case2 前向 15 次调用零 NaN 零 inf）。

用法：
  cd release && .venv/bin/python ascend/kernels/reconcile/M2_probe_dma_cfg.py            # A+B
  ... --root /tmp/m2tgz --only-half        # 只看含 half 指针的归档件
  ... --shapes "32x32,96x32,32x96" --dtypes fp32,fp16
判决行：
  `M2-DMA <hash12> <kernel> <dir> <档> <形态说明> line=<L> mult32=<ok|BAD|n/a>`
  `M2-PLAN <n>x<k> <dtype> bn= bk= => <档> nk=<..> burst=<..>B stride=<..>B key=<..>`
  末行 `M2-DMA-SUMMARY small=.. single=.. big=.. dyn=.. parse_miss=.. mult32_bad=.. plan_small=..`
退出码：归档侧出现 STRIDED-SMALL 或 mult32_bad ⇒ 1（这是"闸"的语义，不是脚本崩溃）。
"""
import argparse
import os
import re
import sys

RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if RELEASE not in sys.path:
    sys.path.insert(0, RELEASE)

CALL_RE = re.compile(r'asc_copy_(gm2ub|ub2gm)_align\(')
SIG_RE = re.compile(r'extern "C" __global__[^;]*?void\s+(\w+)\s*\(([^)]*)\)')
COMMENT_RE = re.compile(r'/\*.*?\*/')


def _split_args(s):
    """按顶层逗号切开（括号内不切）。"""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
            continue
        cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _num(x):
    x = x.strip()
    return int(x) if re.fullmatch(r'-?\d+', x) else x


def parse_copy(line):
    """返回 (direction, n_bursts, burst, src_stride, dst_stride) 或 None。"""
    m = CALL_RE.search(line)
    if not m:
        return None
    i = m.end() - 1
    depth, j = 0, i
    while j < len(line):
        if line[j] == "(":
            depth += 1
        elif line[j] == ")":
            depth -= 1
            if depth == 0:
                break
        j += 1
    args = _split_args(line[i + 1:j])
    direction = m.group(1)
    if direction == "gm2ub" and len(args) >= 10:
        # (dst, src, n_bursts, burst_len, rsvd0, right_pad, rsvd2, mode, srcStrideB, dstStrideB)
        return (direction, _num(args[2]), _num(args[3]), _num(args[8]), _num(args[9]))
    if direction == "ub2gm" and len(args) >= 7:
        # (dst, src, burst_num, burst_len, store_mode, burst_dst_stride, burst_src_stride)
        return (direction, _num(args[2]), _num(args[3]), _num(args[6]), _num(args[5]))
    return (direction, "?", "?", "?", "?")


def classify(n_bursts, burst, src_stride, dst_stride):
    """按 codegen 的 plan_single_row 条件 + 突发宽度分档。"""
    vals = (n_bursts, burst, src_stride, dst_stride)
    if any(isinstance(v, str) for v in vals):
        return ("DYN", f"nb={n_bursts} burst={burst} src_stride={src_stride}")
    if n_bursts == 1 or (src_stride == burst and dst_stride == burst):
        return ("SINGLE", f"{n_bursts}×{burst}B 连续(dst_stride={dst_stride})")
    return (("STRIDED-BIG" if burst >= 128 else "STRIDED-SMALL"),
            f"{n_bursts}×{burst}B 跨距{src_stride}B dst{dst_stride}B")


def scan_archived(root, only_half):
    base = os.path.join(root, "0.1.15", "linux-aarch64", "kernels")
    rows = []
    for d in sorted(os.listdir(base)):
        p = os.path.join(base, d, "device_kernel.asc")
        if not os.path.isfile(p):
            continue
        src = COMMENT_RE.sub('', open(p).read())
        lines = src.splitlines()
        sig = SIG_RE.search(src)
        name, args = (sig.group(1), sig.group(2)) if sig else ("?", "?")
        if only_half and "__gm__ half" not in args:
            continue
        for i, ln in enumerate(lines, 1):
            if "asc_copy_gm2ub_align" not in ln and "asc_copy_ub2gm_align" not in ln:
                continue
            parsed = parse_copy(ln)
            if parsed is None:
                continue
            direction, nb, burst, ss, ds = parsed
            cls, why = classify(nb, burst, ss, ds)
            nums = [v for v in (burst, ss, ds) if isinstance(v, int) and v]
            mult32 = "BAD" if any(v % 32 for v in nums) else ("ok" if nums else "n/a")
            rows.append((d[:12], name, direction, cls, why, i, mult32))
    return rows


def plan_shapes(shapes, dtypes):
    """B) 用产品件自己的 plan 复算 (n,k,位宽) 选出的 bn/bk ⇒ 会编成的 DMA 档。"""
    import torch
    from ascend.kernels import gemm_asc
    out = []
    for spec in shapes:
        n, k = (int(x) for x in spec.lower().split("x"))
        for dtn in dtypes:
            dt = torch.float32 if dtn == "fp32" else torch.float16
            A = torch.zeros(16, k, dtype=dt)
            W = torch.zeros(n, k, dtype=dt)
            C = torch.zeros(16, n, dtype=dt)
            sp = gemm_asc.plan(A, W, None, C, "none", "ascend")
            if sp is None:
                out.append((n, k, dtn, None, None, None,
                            "PLAN-REJECT ⇒ 入口层落 eager（内核一步没走，证照不计 blocker）"))
                continue
            bn, bk = sp["kwargs"]["bn"], sp["kwargs"]["bk"]
            eb = 2 if dt is torch.float16 else 4
            nk = k // bk
            burst, stride = bk * eb, k * eb
            cls, why = classify(nk, burst, stride, burst)
            out.append((n, k, dtn, bn, bk, sp["key"],
                        f"{cls} {why} nk={nk} burst={burst}B stride={stride}B"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/m2tgz")
    ap.add_argument("--only-half", action="store_true", help="只扫含 __gm__ half 指针的归档件")
    ap.add_argument("--shapes", default="32x32,96x32,32x96,32x48,96x48,2x32,8x32,32x8,16x16",
                    help="B 部分：nxk 形状表（逗号分隔）")
    ap.add_argument("--dtypes", default="fp32,fp16")
    ap.add_argument("--no-plan", action="store_true")
    args = ap.parse_args()
    base = os.path.join(args.root, "0.1.15", "linux-aarch64", "kernels")
    tally, bad_hashes, mult32_bad = {}, [], []
    if os.path.isdir(base):
        print(f"M2-DMA-A 归档侧 root={base} only_half={args.only_half}")
        for h, name, direction, cls, why, line, mult32 in scan_archived(args.root, args.only_half):
            tally[cls] = tally.get(cls, 0) + 1
            if cls == "STRIDED-SMALL":
                bad_hashes.append(f"{h}/{name}")
            if mult32 == "BAD":
                mult32_bad.append(h)
            print(f"M2-DMA {h} {name:<22} {direction:<5} {cls:<14} {why} line={line} mult32={mult32}")
    else:
        print(f"M2-DMA-A SKIP: 找不到 {base}（先 tar xzf sys1_kernels.tgz -C {args.root}）")
    plan_small = []
    if not args.no_plan:
        print("M2-DMA-B 产品侧：plan 现场选的 (bn,bk) ⇒ 会编成的 DMA 档")
        for n, k, dtn, bn, bk, key, verdict in plan_shapes(
                [s for s in args.shapes.split(",") if s],
                [d for d in args.dtypes.split(",") if d]):
            flag = ""
            if "STRIDED-SMALL" in verdict:
                flag = "   <<< 坏档候选（与 P1-1q 归档件 5d732a8f 同档）"
                plan_small.append(f"{n}x{k}/{dtn}")
            print(f"M2-PLAN {n}x{k:<4} {dtn} bn={bn} bk={bk} => {verdict} key={key}{flag}")
    print(f"M2-DMA-SUMMARY small={len(bad_hashes)} single={tally.get('SINGLE', 0)} "
          f"big={tally.get('STRIDED-BIG', 0)} dyn={tally.get('DYN', 0)} "
          f"parse_miss={tally.get('PARSE-MISS', 0)} mult32_bad={mult32_bad or 'NONE'} "
          f"archived_small={bad_hashes or 'NONE'} plan_small={plan_small or 'NONE'}")
    return 1 if (mult32_bad or plan_small) else 0   # 闸只看产品侧（归档是历史事实，不参与门禁）


if __name__ == "__main__":
    sys.exit(main())
