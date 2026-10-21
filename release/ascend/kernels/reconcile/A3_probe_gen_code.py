"""A3_probe_gen_code.py — 只读探针：把产品件**真编译出来的设备侧生成码**落盘并数 §12 位段落点。

【为什么卡窗需要它】容器/卡上 `NUM-*` / `Y-*-PROD-*` 只给数值 rel。数值不对时，第一个要回答的问题
是"**§12 那套 9 参 DMA 位序到底进没进产物**"——这只有 bisheng 实际吃进去的那份 `.asc`/生成码能回答。
本探针按产品入口件的同一条路（`*_asc.plan → impl → tilelang.compile(target="ascend")`）编一件、
取 `get_kernel_source()`、落盘，并数四个锚点：
    nd2nz   = `copy_gm_to_cbuf_multi_nd2nz` 出现次数   （§12 成套 ND2NZ 搬运是否真被 lower 出来）
    fill    = `asc_fill_l1` 出现次数                   （P1-4b set_l1_2d 填充路）
    sidcast = `(uint8_t)sid` 出现次数                  （Z 落地后**应为 0**：旧的回穿写法已清）
    nine    = `LoadData2D` / `SetSysWorkspace` 之类 9 参宏的原样出现（位序直证）
零产品改动：只 compile、只读源码、只往 /tmp（或 --out 指定处）写文件。

用法（容器/卡，overlay 已就位）：
    cd release && DMLAYA_ASCEND_TARGET=ascend python3 <本文件> gemm --out /root/a3_gem.cc
    <本文件> list                                   # 看可选件名
件名：gemm gemm_dw readout addln attnsw gdn gdnb lora_gemm rope
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.environ.get("A3_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
from ascend.kernels import ascend_env as E  # noqa: E402

OPS = ("gemm", "gemm_dw", "readout", "addln", "attnsw", "gdn", "gdnb", "lora_gemm", "rope")


def open_gate():
    """run_numerics._prod_setup 同款三量桩（无设备也能达到 tilelang.compile）。"""
    E.reset()
    E._npu_present = lambda: True
    E.backend_available = lambda *a, **k: True
    E.active_backend = lambda *a, **k: E.TILELANG


def build(op: str):
    """返回 (prim_func, 人读的键名)。全部走 *_asc 的 plan/spec，与产品入口件同口径。"""
    import tilelang  # noqa: F401
    from ascend.kernels import (add_ln_asc, attn_sw_asc, gemm_asc, gemm_bwd_dw_asc,
                                gdn_asc, letter_readout_asc, rope_asc)
    d = "float32"
    if op == "gemm":
        m, n, k = 33, 64, 32
        A = torch.randn(m, k, dtype=torch.float16)
        W = torch.randn(n, k, dtype=torch.float16)
        b = torch.randn(n, dtype=torch.float32)
        C = torch.empty(m, n, dtype=torch.float16)
        spec = gemm_asc.plan(A, W, b, C, "none", "ascend")
        return lambda: spec["impl"](A, W, b, C, **spec["kwargs"]), spec["key"]
    if op == "gemm_dw":
        m, n, k = 33, 64, 32
        dY = torch.randn(m, n, dtype=torch.float16)
        X = torch.randn(m, k, dtype=torch.float16)
        GW = torch.zeros(n, k, dtype=torch.float16)
        spec = gemm_bwd_dw_asc.plan(dY, X, GW, "ascend")
        return lambda: spec["impl"](dY, X, GW, **spec["kwargs"]), spec["key"]
    if op == "addln":
        rows, dim = 33, 32
        X = torch.randn(rows, dim, dtype=torch.float16)
        R = torch.randn(rows, dim, dtype=torch.float16)
        g = torch.ones(dim, dtype=torch.float16)
        bt = torch.zeros(dim, dtype=torch.float16)
        Y = torch.empty(rows, dim, dtype=torch.float16)
        H = torch.empty(rows, dim, dtype=torch.float16)
        spec = add_ln_asc.plan(X, R, g, bt, Y, H, 1e-5, "ascend")
        return lambda: spec["impl"](X, R, g, bt, Y, H, **spec["kwargs"]), spec["key"]
    if op == "attnsw":
        h, seq, dim, win = 2, 16, 16, 16
        Q = torch.randn(h, seq, dim, dtype=torch.float16)
        K = torch.randn(h, seq, dim, dtype=torch.float16)
        V = torch.randn(h, seq, dim, dtype=torch.float16)
        O = torch.empty(h, seq, dim, dtype=torch.float16)
        spec = attn_sw_asc.plan(Q, K, V, O, win, "ascend")
        return lambda: spec["impl"](Q, K, V, O, **spec["kwargs"]), spec["key"]
    if op in ("gdn", "gdnb"):
        h, seq, dk, dv = 2, 8, 16, 16
        q = torch.randn(h, seq, dk, dtype=d)
        k = torch.randn(h, seq, dk, dtype=d)
        v = torch.randn(h, seq, dv, dtype=d)
        g = torch.zeros(h, seq, dv, dtype=d)
        beta = torch.full((h, seq), 0.5, dtype=d)
        s = torch.zeros(h, dv, dk, dtype=d)
        o = torch.empty(h, seq, dv, dtype=d)
        if op == "gdn":
            spec = gdn_asc.plan(q, k, v, g, beta, o, "ascend")
            return lambda: spec["impl"](q, k, v, g, beta, o, s, **spec["kwargs"]), spec["key"]
        fwd = gdn_asc.plan(q, k, v, g, beta, o, "ascend")
        spec = gdn_asc.plan_bwd(fwd, q, k, v, g, beta, o, "ascend") \
            if hasattr(gdn_asc, "plan_bwd") else None
        if spec is None:
            raise SystemExit(f"A3-GENCODE-SKIP {op}: gdn_asc 无 plan_bwd 入口（件名或已改）")
        return lambda: spec["impl"](q, k, v, g, beta, o, **spec["kwargs"]), spec["key"]
    if op == "lora_gemm":
        m, r, k = 16, 8, 32
        x = torch.randn(m, k, dtype=torch.float16)
        a = torch.randn(r, k, dtype=torch.float16)
        c = torch.empty(m, r, dtype=torch.float16)
        spec = __import__("ascend.kernels.gemm_asc", fromlist=["gemm_asc"]).plan(
            x, a, None, c, "none", "ascend")
        if spec is None:
            return None, "plan 拒形（r=8 不被块宽整除）⇒ 产品件在 LoRA 段本就落 eager"
        return lambda: spec["impl"](x, a, torch.zeros(r, dtype=torch.float32), c,
                                    **spec["kwargs"]), spec["key"]
    if op == "readout":
        dim = 32
        X = torch.randn(8, dim, dtype=torch.float16)
        Y = torch.empty(8, dim, dtype=torch.float16)
        spec = letter_readout_asc.plan(X, Y, "ascend")
        return lambda: spec["impl"](X, Y, **spec["kwargs"]), spec["key"]
    if op == "rope":
        h, seq, dim = 2, 16, 16
        qkv = torch.randn(seq, 3, h, dim, dtype=d)
        cos = torch.ones(seq, dim // 2, dtype=d)
        sin = torch.zeros(seq, dim // 2, dtype=d)
        freq = torch.ones(seq, dim // 2, dtype=d)
        spec = rope_asc.plan(qkv, cos, sin, freq, (0, 1), "ascend")
        return lambda: spec["impl"](qkv, cos, sin, freq, **spec["kwargs"]), spec["key"]
    raise SystemExit(f"A3-GENCODE-BADOP {op}，可选：{','.join(OPS)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("op")
    ap.add_argument("--out", default="/tmp/a3_gencode.cc")
    a = ap.parse_args()
    if a.op == "list":
        print(",".join(OPS))
        return 0
    open_gate()
    import tilelang
    builder, key = build(a.op)
    if builder is None:
        print(f"A3-GENCODE-SKIP {a.op} {key}")
        return 0
    prog = builder()
    kern = tilelang.compile(prog, out_idx=[], **E.compile_kwargs("ascend"))
    src = str(kern.get_kernel_source())
    with open(a.out, "w") as fh:
        fh.write(src)
    anchors = {
        "nd2nz": src.count("copy_gm_to_cbuf_multi_nd2nz"),
        "fill": src.count("asc_fill_l1"),
        "sidcast": src.count("(uint8_t)sid"),
        "LoadData2D": src.count("LoadData2D"),
        "DataCopy": src.count("DataCopy"),
        "Cube": src.count("Cube"),
        "matmul": src.count("Matmul") + src.count("matmul"),
    }
    print(f"A3-GENCODE {a.op} key={key} bytes={len(src)} out={a.out} "
          + " ".join(f"{k}={v}" for k, v in anchors.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
