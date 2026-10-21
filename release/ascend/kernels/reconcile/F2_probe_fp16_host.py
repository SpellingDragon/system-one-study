#!/usr/bin/env python3
"""reconcile/F2_probe_fp16_host.py — p2-13 F2：**生产位宽的组装面**在 cpu-alias 轨真编真跑。

【判什么的】C2_probe_dtype_matrix 证的是"每件 fp16 单独真编真跑对数"，train_step --dtype fp16
host 轨证的是"接线与判据"；本探针夹在中间补最后一环——**target=ascend 路由下的全图训练步**
（14 处门面调用、8 个 Function、两 fwd 一 bwd 一 opt.step）在 fp16 模具组合下编得出、跑得起、
降得动、位宽账干净。alias 工装配方与 ascend/kernels/reconcile/C2_probe_alias_shim.py 的
`alias()` 完全同口径（tilelang.ascend.language→cpu 方言 + Vector/alloc_shared 垫片 +
ascend_env 三门 monkeypatch），此处按 import-safe 方式内联，不引 C2 文件（禁改清单）。

【假绿守卫】同 C2：每站必须真出键（`ascend|<件>` 进 compiled_keys），落 eager 不算；
原生三件的键必须带 fp16 位宽段（gemm/dw ⇒ `dfloat16`；rope ⇒ `qfloat16afloat32` 生产混宽），
物化六件的键段如实分型打印（入口 fp16/内核 fp32 是 C2 §5 的既定形态，不是缺陷）。

用法（缓存纪律同 C2）：
  TILELANG_CACHE_DIR=$(mktemp -d) .venv/bin/python ascend/kernels/reconcile/F2_probe_fp16_host.py
判决行：F2-HOST-<用例>-PASS/FAIL；末行 F2-HOST-SUMMARY n/m。
注意：**不能**加 `from __future__ import annotations`。
"""
import contextlib
import os
import sys
import types

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if RELEASE not in sys.path:
    sys.path.insert(0, RELEASE)

import torch           # noqa: E402
import tilelang        # noqa: E402
import tilelang.cpu.language as tcpu   # noqa: E402
from ascend.kernels import ascend_env  # noqa: E402

SHIMMED = []


def alias():
    """cpu-alias 轨工装（逐字同 C2_probe_alias_shim.alias()）。"""
    pkg = sys.modules.get("tilelang.ascend")
    if pkg is None:
        pkg = types.ModuleType("tilelang.ascend")
        pkg.__path__ = []
        sys.modules["tilelang.ascend"] = pkg
        setattr(tilelang, "ascend", pkg)
    sys.modules["tilelang.ascend.language"] = tcpu
    pkg.language = tcpu
    if not hasattr(tcpu, "Vector"):
        tcpu.Vector = lambda *a, **k: contextlib.nullcontext()
        SHIMMED.append("Vector→nullcontext")
    if not hasattr(tcpu, "_c2_orig_alloc_shared"):
        tcpu._c2_orig_alloc_shared = tcpu.alloc_shared
        tcpu.alloc_shared = lambda shape, dt, *a, **k: tcpu.alloc_local(shape, dt)
        SHIMMED.append("alloc_shared→alloc_local")
    ascend_env.compile_kwargs = lambda t: {"target": "c", "target_host": "c",
                                          "execution_backend": "cython"}
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda *a, **k: ascend_env.TILELANG
    ascend_env.reset()


RESULTS = []


def verdict(name, ok, detail=""):
    print(f"F2-HOST-{name}-{'PASS' if ok else 'FAIL'} {detail}", flush=True)
    RESULTS.append(bool(ok))


def main():
    alias()
    from ascend import train_step as ts
    print(f"F2-HOST-SETUP alias 垫片={SHIMMED or '无需'} track=target=ascend on cpu-alias(c 后端)",
          flush=True)

    l0, l1, missing = ts.run(target="ascend", device=torch.device("cpu"),
                             dtype=torch.float16, verbose=True)
    audit = dict(ts.LAST_AUDIT)
    keys = sorted(ascend_env.compiled_keys())
    bl = dict(ascend_env.blockers())

    # 1) 步本身：降 loss + 全参 grad
    verdict("STEP", (l1 < l0) and not missing,
            f"loss0={l0:.6f} loss1={l1:.6f} reduced={l1 < l0} grad_missing={missing or '无'}")

    # 2) 位宽账：16 参全 fp16、guard 零违规、DTYPE-TRACE 行逐字段可核
    line = ts.dtype_trace_line(audit, fallback_events=0)
    verdict("AUDIT", audit.get("fp16") == 16 and audit.get("params") == 16
            and not audit.get("violations"),
            f"{line} fp32_acc计数={audit.get('fp32_acc')}")

    # 3) 假绿守卫：全图各站必须真出键（落 eager 不算 PASS）
    #: 件名按 compiled_keys 真桶名（readout 入口的键面是 gather/scatter_add，
    #: add_ln 反向是 ln_bwd——C2 矩阵同款命名，非本探针臆造）。
    want_pieces = ["gemm[ascend]", "gemm_dw[ascend]", "rope[ascend]", "add_ln[ascend]",
                   "ln_bwd[ascend]", "attn_sw[ascend]", "gdn[ascend]", "gdnb[ascend]",
                   "gather[ascend]", "scatter_add[ascend]"]
    hit = {p: [k for k in keys if p in k] for p in want_pieces}
    absent = [p for p, v in hit.items() if not v]
    verdict("KEYS-ROUTE", not absent,
            f"出件={len(want_pieces) - len(absent)}/{len(want_pieces)} 缺={absent or '无'} "
            f"总键数={len(keys)}")

    # 4) 原生三件 fp16 段（生产混宽：rope=表 fp32 ⇒ qfloat16afloat32）
    g16 = [k for k in keys if "gemm[ascend]" in k and "dfloat16" in k]
    w16 = [k for k in keys if "gemm_dw[ascend]" in k and "dfloat16" in k]
    r16 = [k for k in keys if "rope[ascend]" in k and "qfloat16afloat32" in k]
    verdict("KEYS-DTYPE", bool(g16) and bool(w16) and bool(r16),
            f"gemm×{len(g16)} dw×{len(w16)} rope×{len(r16)} 例={g16[:1] + w16[:1] + r16[:1]}")

    # 5) 物化件键段分型如实打印（入口 fp16/内核 fp32 是 C2 §5 既定形态，不作 FAIL 判）
    for p in ["add_ln[ascend]", "ln_bwd[ascend]", "attn_sw[ascend]", "gdn[ascend]",
              "gdnb[ascend]", "gather[ascend]", "scatter_add[ascend]"]:
        print(f"F2-HOST-KEYTYPE {p} keys={hit.get(p, [])}", flush=True)

    # 6) blocker 账：alias 轨全图 fp16 不应有任何回落公告
    verdict("NO-BLOCKER", not bl, f"blockers={list(bl) or '无'}")

    n_ok = sum(RESULTS)
    print(f"F2-HOST-SUMMARY {n_ok}/{len(RESULTS)}", flush=True)
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
