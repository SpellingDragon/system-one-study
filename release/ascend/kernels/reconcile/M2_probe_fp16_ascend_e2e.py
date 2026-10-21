#!/usr/bin/env python3
"""reconcile/M2_probe_fp16_ascend_e2e.py — P1-1p：fp16 训练步**反向面**的 host 逐 op 取证轨。

【为什么还要这条轨】卡上的 H 案（`artifacts/H_case2.log`）只包了**前向** 15 次调用，"第一个 BAD
＝ step1 第一个 linear 入参带 NaN（in nan=1056）"是**反向产物落地之后**才看见的；反向中间量在卡上
没有任何逐 op 证照。本探针把 `gemm_kernel.forward` / `gemm_bwd_dw_kernel.backward` 两个入口整个包
起来（探针侧 monkeypatch，产品件一行未改），前向与**反向**每一次调用都落 `in/out 的 nan/inf/absmax`
＋该次调用按产品 plan 会落在哪一档/哪个键 ⇒ 字段与卡上 H-OP 同款、但覆盖反向的 trace。
另抄一份 `_build_params` 的返回字典，步末逐参数审计 `.grad` 的 nan/inf/absmax（H-OP 016 的
`in nan=1056` 就是这一层的产物：W1(32×32=1024)+b1(32) 正好 1056）。

【三档对照（同一份正文，三条轨）】
  T1 `--target cpu`  ：入口件对 fp16 输入**先 `.to(float32)` 再走内核**（`gemm_kernel.forward` 的
      `to32 = name == TARGET_CPU`）⇒ 这一档从接口上就把 fp16 面抬掉了，天然不会复现。
  T2 `--target ascend` + cpu-alias（本探针缺省）：位宽、路由、键名、正文**全是产品口径**（fp16
      原生进内核），但 `tilelang.ascend.language` 被别名到 cpu 方言 ⇒ `T.copy` 走 **c 后端
      lowering**，编译产物里根本没有 `asc_copy_gm2ub_align` 这条 MTE2 跨步突发 DMA。
  T3 真机：跨步 DMA 在场（归档 `ea1059e6…` 行 31 = `…, 32, 64, 0,0,0, mode, 192, 64`）。
  ⇒ **期望结论是 NOT-REPRODUCED（T2 无 nan）**，"复现不了的原因"钉成可复核两条：①T1 是入口抬宽；
    ②T2 是 c 后端没有 MTE2 跨步装填面（与 P1-1q 的 U2 同口径）。**NOT-REPRODUCED ≠ 该档已判好。**

【A/B】`--tag` 只影响打印。修复前后各跑一遍：T2 里 n32k96 的档从 `STRIDED burst=64B` 变
  `SINGLE`，trace 数值应仍在 fp16 噪声量级 ⇒ 佐证"改 bk 不改交货数值"。

用法：
  cd release && .venv/bin/python ascend/kernels/reconcile/M2_probe_fp16_ascend_e2e.py
  ... --dtype fp32 / --target cpu / --tag pre-fix
判决行：`M2-OP <idx> fwd|dw ... in[nan,inf,max]=(..) out=(..) <档> key=<..>`
        `M2-GRAD <param> shape=.. nan=.. inf=.. absmax=..`
        `M2-E2E nan_total=.. inf_total=.. loss0=.. loss1=.. ⇒ REPRODUCED|NOT-REPRODUCED`
退出码：0=NOT-REPRODUCED（host 轨本来的期望），2=REPRODUCED（真在 host 上抓到 nan）。
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import types

import torch

RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if RELEASE not in sys.path:
    sys.path.insert(0, RELEASE)
os.environ.setdefault("DMLAYA_ASCEND_TARGET", "ascend")

import tilelang                              # noqa: E402
import tilelang.cpu.language as tcpu         # noqa: E402
from ascend.kernels import ascend_env, gemm_asc  # noqa: E402

SHIMMED: list[str] = []
REC = {"ops": [], "nan": 0, "inf": 0, "idx": 0, "keys": []}
CAPTURED: dict = {}


def alias():
    """cpu-alias 工装（逐字同 C2_probe_alias_shim.alias() / F2_probe_fp16_host.alias()）。"""
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
        SHIMMED.append("T.Vector")
    if not hasattr(tcpu, "_c2_alloc_shared"):
        tcpu._c2_alloc_shared = tcpu.alloc_shared
        tcpu.alloc_shared = lambda shape, dt, *a, **k: tcpu.alloc_local(shape, dt)
        SHIMMED.append("T.alloc_shared->alloc_local")
    ascend_env.compile_kwargs = lambda t: {"target": "c", "target_host": "c",
                                          "execution_backend": "cython"}
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda *a, **k: ascend_env.TILELANG
    ascend_env.reset()


def stat(t):
    """(nan, inf, absmax)：一律升到 fp32 面统计（fp16 的 nan/inf 升宽后仍是 nan/inf）。"""
    if t is None:
        return (0, 0, 0.0)
    tf = t.detach().float().reshape(-1)
    if tf.numel() == 0:
        return (0, 0, 0.0)
    return (int(torch.isnan(tf).sum()), int(torch.isinf(tf).sum()), float(tf.abs().max()))


def _form(A, W, act, dt, label):
    """问产品自己的 plan（只问不发射）：这组形状/位宽会落在哪个键、哪一档 DMA。"""
    m, n, k = int(A.size(0)), int(W.size(0)), int(W.size(1))
    a = A.detach().to(dt).contiguous()
    w = W.detach().to(dt).contiguous()
    c = torch.empty(m, n, dtype=dt)
    sp = gemm_asc.plan(a, w, None, c, act, label)
    if sp is None:
        return "PLAN-REJECT(=>eager) key=-"
    bn, bk = sp["kwargs"]["bn"], sp["kwargs"]["bk"]
    eb = 2 if dt is torch.float16 else 4
    if bk == k:
        form = f"SINGLE burst={bn * bk * eb}B(nk=1)"
    else:
        form = f"STRIDED burst={bk * eb}B stride={k * eb}B nk={k // bk}"
    return f"{form} key={sp['key']}"


def trace_gemm():
    """包住前向/dW 两个入口：反向的 dA 在 autograd 里同样走 gemm_kernel.forward（:169）。"""
    from ascend.kernels import gemm_bwd_dw_kernel, gemm_kernel

    fwd, dwbwd = gemm_kernel.forward, gemm_bwd_dw_kernel.backward

    def fw(A, W, bias=None, act="relu", out_dtype=torch.float16, target=None):
        C = fwd(A, W, bias, act, out_dtype, target)
        form = _form(A, W, act, A.dtype, target or "ascend")
        i, j, o = stat(A), stat(W), stat(C)
        REC["idx"] += 1
        REC["nan"] += i[0] + j[0] + o[0] + stat(bias)[0]
        REC["inf"] += i[1] + j[1] + o[1] + stat(bias)[1]
        REC["ops"].append((REC["idx"], "fwd", o[0]))
        REC["keys"].append(form.split("key=")[-1])
        b = f" bias={stat(bias)}" if bias is not None else ""
        print(f"M2-OP {REC['idx']:03d} fwd  A={tuple(A.shape)} W={tuple(W.shape)} dt={A.dtype} "
              f"in=({i[0]},{i[1]},{i[2]:.4g})+({j[0]},{j[1]},{j[2]:.4g}){b} "
              f"out=({o[0]},{o[1]},{o[2]:.4g}) | {form}", flush=True)
        return C

    def dw(dC, A, out_dtype=None, target=None):
        GW = dwbwd(dC, A, out_dtype=out_dtype, target=target)
        i, o = stat(dC), stat(GW)
        REC["idx"] += 1
        REC["nan"] += i[0] + o[0]
        REC["inf"] += i[1] + o[1]
        REC["ops"].append((REC["idx"], "dw", o[0]))
        print(f"M2-OP {REC['idx']:03d} dw   dC={tuple(dC.shape)} A={tuple(A.shape)} dt={dC.dtype} "
              f"in=({i[0]},{i[1]},{i[2]:.4g}) out=({o[0]},{o[1]},{o[2]:.4g}) outdt={GW.dtype}",
              flush=True)
        return GW

    gemm_kernel.forward, gemm_bwd_dw_kernel.backward = fw, dw


def capture_params(ts):
    """抄 `_build_params` 的返回字典：步末逐参数审计 .grad（不改 ts.run 的一行调用）。"""
    orig = ts._build_params

    def patched(*a, **k):
        p, cos, sin = orig(*a, **k)
        CAPTURED["p"] = p
        return p, cos, sin

    ts._build_params = patched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="ascend", choices=("ascend", "cpu"))
    ap.add_argument("--dtype", default="fp16", choices=("fp16", "fp32"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.target == "ascend":
        alias()
        print(f"M2-E2E-SETUP track=cpu-alias target=ascend dtype={a.dtype} "
              f"垫片={SHIMMED or '无需'}（产品路由/键名/正文，位宽原生）", flush=True)
    else:
        print(f"M2-E2E-SETUP track=native-cpu dtype={a.dtype} "
              "（入口对 fp16 先 .to(float32) ⇒ 内核只见 fp32）", flush=True)
    from ascend import train_step as ts

    trace_gemm()
    capture_params(ts)
    dt = torch.float16 if a.dtype == "fp16" else torch.float32
    l0, l1, missing = ts.run(target=a.target, device=torch.device("cpu"), dtype=dt, verbose=True)
    keys = sorted(ascend_env.compiled_keys())
    bl = dict(ascend_env.blockers())

    p = CAPTURED.get("p", {})
    gn = 0
    for name, t in sorted(p.items()):
        g = None if t is None else t.grad
        s = stat(g)
        gn += s[0]
        REC["nan"] += s[0]
        REC["inf"] += s[1]
        print(f"M2-GRAD {name:<10} shape={tuple(g.shape) if g is not None else 'None'} "
              f"dt={g.dtype if g is not None else '-'} nan={s[0]} inf={s[1]} absmax={s[2]:.4g}",
              flush=True)

    op_bad = sum(1 for _, kind, nancnt in REC["ops"] if nancnt)
    fwd_ops = [x for x in REC["ops"] if x[1] == "fwd"]
    dw_ops = [x for x in REC["ops"] if x[1] == "dw"]
    print(f"M2-E2E-OPS calls={len(REC['ops'])}（fwd {len(fwd_ops)} / dw {len(dw_ops)}）"
          f" 带nan的调用={op_bad} keys={len(keys)} blockers={list(bl) or '无'}", flush=True)
    print(f"M2-E2E-KEYS n32k96/n96k32 档 = {[k for k in keys if 'n32k96' in k or 'n96k32' in k]}",
          flush=True)
    verdict = "REPRODUCED" if (REC["nan"] or REC["inf"]) else "NOT-REPRODUCED"
    print(f"M2-E2E{(' ' + a.tag) if a.tag else ''} dtype={a.dtype} target={a.target} "
          f"nan_total={REC['nan']} inf_total={REC['inf']} grad_nan={gn} "
          f"loss0={l0:.6f} loss1={l1:.6f} reduced={l1 < l0} ⇒ {verdict}", flush=True)
    print("M2-E2E-REASON host 轨没有 MTE2 跨步突发装填面（cpu 方言的 T.copy 走 c 后端 lowering，"
          "产物里不出现 asc_copy_gm2ub_align）；target=cpu 再叠一层入口 .to(float32) ⇒ 量不到 "
          "P1-1p 的嫌疑面；NOT-REPRODUCED **不等于**该档已判好（判好与否只有真机说）", flush=True)
    return 0 if verdict == "NOT-REPRODUCED" else 2


if __name__ == "__main__":
    sys.exit(main())
