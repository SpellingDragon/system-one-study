#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V2_probe_intrinsic_truth.py — P1-1t 收尾：intrinsic 对照腿「真伪核查」探针（V2 / 不占卡）。

【一句话结论（本件 --mode static 复跑得出，全文见 V2_THROUGHPUT_TRUTH.md）】
第二窗那条 `BENCH-linear_intrinsic dtype=fp32 shape=512x512x512 p50_ms=0.080 tflops=3365.919`
的 intrinsic 件（`S_perf_track_cube.gemm_cube_impl`）**根本没做 512³ 的活**——它落成
`extern "C" __global__ __mix__(1, 2) void gemm_impl_kernel(...)`，正文**整段**嵌在
`if ASC_IS_AIV { if ASC_IS_AIC { <搬运 asc_copy_gm2ub_align / 乘加 asc_mmad / 写回 asc_copy_ub2gm_align> } }`
里；而 910B compat 层把这两个宏钉成 `ASC_IS_AIV (0)` / `ASC_IS_AIC (true)`
（`port910b_compat.h:568 / :530`，在 `TL_PORT910B_NATIVE_TYPES` 段；该宏由容器内 pip
`bisheng.py:111` 的 `-DTL_PORT910B_NATIVE_TYPES` 注入）⇒ 展开即 `if (0) { if (1) { ... } }`
= **死代码**。故 0.080ms 是"空转 + 发射/同步开销"，不是 512³ 的实算。
归类：**未真执行**（非形状不符、非计时未夹同步、非参数顺序错）。

【对照（同窗另一条 intrinsic 腿，反证分类）】`dw_l0tr_cube_impl`（`dW_intrinsic`）落成
`__global__ __cube__ void dw_impl_kernel(...)`（**无** `if ASC_IS_AIV` 包裹）⇒ 真执行 ⇒
窗口2 里它如实 aicore exception **507015**（DMA 非法访问）。即：`__mix__` 腿是被 compat
宏杀死的死代码（假绿），`__cube__` 腿是真的跑并真的崩——两条腿的表现与生成码形态一一对应。

【三闸（G1 是硬闸，G2/G3 是加固）】
  G1 数值前置：torch 参考对拍（A@Wᵀ+bias / dYᵀ@A）；rel 不过阈值 ⇒ **拒绝计时**、打 FAIL。
  G2 峰值闸：tflops ≤ 910B 理论峰值（出处见 PEAK_SRC，**来自 CANN platform_config，非记忆**）。
  G3 计时夹同步：每个样本 `sync(); t0; call(); sync()` 前后夹计。

【dev 模式】
  --mode static ：**codegen-only**（`tilelang.lower(..., enable_device_compile=False)`，不起
                  bisheng）+ 打形状常数/循环边界 + **死代码判定**（容器可跑，无设备/无编译器）
  --mode truth  ：G1→G3→G2 全跑（**需真卡**，无设备时如实 SKIP）
  --mode neg    ：负例自证——喂错的输出 / 超峰值 tflops ⇒ 断言 G1/G2 必挂（**无设备可跑**）

用法：
  # 容器（cann910b-h，有 bisheng 无设备）——出"死代码"原文与判定：
  docker exec cann910b-h bash -lc 'cd /work/ascend/kernels/reconcile && \
      TILELANG_CACHE_DIR=/dev/shm/v2probe python3 V2_probe_intrinsic_truth.py --mode static'
  # host——出闸逻辑的负例自证：
  release/.venv/bin/python ascend/kernels/reconcile/V2_probe_intrinsic_truth.py --mode neg
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")

_REC = str(pathlib.Path(__file__).resolve().parent)
_ROOT = str(pathlib.Path(_REC).resolve().parents[2])          # .../release
for _p in (_REC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: G1 对拍阈值（相对误差）：fp32 参考对 fp32 内核，容忍归约顺序差。与
#: `port910b/run_numerics.py::num_gemm_direct`（`r < 0.02`）同尺。
REL_TOL = 1e-2

# ── 峰值出处（**不得凭记忆**）────────────────────────────────────────────────
# 权威来源：容器内 CANN 8.5.x 的 `platform_config/*.ini`（本机可复跑）：
#   docker exec cann910b-h grep -E "^(ai_core_cnt|cube_freq|cube_[mnk]_size|NpuArch)" \
#     /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/data/platform_config/Ascend910B*.ini
# 实测（cann-8.5.0，2026-10-11 复跑）：
#   Ascend910B1.ini  NpuArch=2201 ai_core_cnt=24 cube_freq=1850 cube_m/n/k_size=16/16/16
#   Ascend910B2.ini  NpuArch=2201 ai_core_cnt=24 cube_freq=1800 ...
#   Ascend910B3.ini  NpuArch=2201 ai_core_cnt=20 cube_freq=1800 ...
#   Ascend910B4.ini  NpuArch=2201 ai_core_cnt=20 cube_freq=1500 ...
#   Ascend910B.ini   NpuArch=1001 ai_core_cnt=30 cube_freq= 900 ...   （旧代，非 2201）
# 推导：peak_fp16 = ai_core_cnt × cube_freq(Hz) × (cube_m × cube_n × cube_k) × 2(每 MAC 2 flop)
#   Ascend910B1 → 24 × 1.85e9 × 4096 × 2 = 3.637e14 = **363.7 TFLOPS**（族内最大）
#   910B2=353.9 / 910B3=294.9 / 910B4=245.8
# 注：卡的 SoC 变体未落进窗口日志（`NpuArch=2201` 被 910B1/2/3/4 共享）
#     ⇒ 取**族内最大**（910B1）当硬上界（宁松勿误杀合法数；本件 G1 才是主闸）。
#     tilelang 自身的 `tools/Analyzer.py::ARCH_CONFIGS` 只有 NVIDIA sm80/86/89，无昇腾，
#     故"理论峰值查无第二出处"这一条已复核（tilelang/tools/Analyzer.py:179-195）。
PEAK_TFLOPS_FP16 = 363.7
PEAK_SRC = ("CANN platform_config/Ascend910B1.ini (ai_core_cnt=24 × cube_freq=1850MHz × "
            "cube 16×16×16 × 2) = 363.7 TFLOPS；族内最大，取作硬上界")

#: compat 层的死代码根源（实证：port910b_compat.h:568 在 TL_PORT910B_NATIVE_TYPES 段）。
_COMPAT_HINT = ("port910b_compat.h: `#define ASC_IS_AIV (0)` / `#define ASC_IS_AIC (true)` "
                "(under -DTL_PORT910B_NATIVE_TYPES, injected by pip bisheng.py:111)")


# ─────────────────────────────────────────────────── 闸（纯函数，不依赖设备/torch）
def relerr(got, ref):
    """相对 L2 误差；numpy 参考恒 fp64 以消除参考自身噪声。"""
    import numpy as np
    a = np.asarray(got, dtype=np.float64).ravel()
    b = np.asarray(ref, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return float("inf")
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def numeric_gate(got, ref, rel_tol=REL_TOL):
    """G1：数值前置（硬闸）。返回 `(ok, detail)`；任何异常都判不过。"""
    try:
        r = relerr(got, ref)
    except Exception as exc:                                  # noqa: BLE001
        return False, f"relerr-raised:{type(exc).__name__}"
    return (r <= rel_tol), f"rel={r:.3e}"


def peak_gate(tflops, peak=PEAK_TFLOPS_FP16):
    """G2：峰值闸。返回 `(ok, detail)`。"""
    if not (tflops == tflops) or not (tflops > 0):
        return False, f"tflops={tflops}"
    return (tflops <= peak), f"tflops={tflops:.3f} peak={peak:.1f}"


# ─────────────────────────────────────────────────── device 源体检
def classify_devsrc(src):
    """从生成的 device 源判定核函数"是否真会执行"。

    `__mix__(...)` 形态：codegen 把 CUBE/VECTOR 两块分别裹进 `if ASC_IS_AIC/ASC_IS_AIV`；compat 层
    把 `ASC_IS_AIV` 钉成 0 ⇒ 只要正文落在 `if ASC_IS_AIV` 内，就是死代码。`__cube__`/`__vector__`
    形态没有这层包裹 ⇒ 会执行。
    """
    glob = re.findall(r"__global__\s+(__mix__\([0-9, ]*\)|__cube__|__vector__)", src)
    mode = glob[0] if glob else "?"
    has_aiv_guard = bool(re.search(r"\bif\s+ASC_IS_AIV\b", src))
    # 计算核在源里的位置（搬运/乘加/写回任一）
    comp = re.search(r"\b(asc_mmad|ascend_gemm_l1|asc_copy_gm2ub_align|asc_copy_gm2l1_nd2nz|"
                     r"tl::write_gm_bypass_dcache)\b", src)
    dead = (mode.startswith("__mix__") and has_aiv_guard and comp is not None)
    verdict = "NOT-EXECUTED" if dead else ("EXECUTES" if mode != "?" else "UNKNOWN")
    return {"mode": mode, "has_if_ASC_IS_AIV": has_aiv_guard,
            "compute_present": comp is not None, "verdict": verdict}


def shape_loop_evidence(src):
    """抓形状常数与循环边界（静态面证据）。"""
    return {
        "global": re.findall(r"__global__[^{]*", src)[:1],
        "loops": re.findall(r"for\s*\(int32_t\s+(\w+)\s*=\s*0;\s*\1\s*<\s*([^;]+);", src),
        "gm_strides": sorted({int(x) for x in re.findall(r"int64_t\)(\d+)", src)}),
        "mmad": re.findall(r"asc_mmad\([^;]{0,160}", src)[:1],
        "gemm_l1": re.findall(r"ascend_gemm_l1<[^;]*", src)[:1],
        "copy_gm2ub": re.findall(r"asc_copy_gm2ub_align\([^;]{0,120}", src)[:1],
        "copy_l0c2gm": re.findall(r"asc_copy_l0c2gm\([^;]{0,120}", src)[:1],
    }


# ─────────────────────────────────────────────────── intrinsic 件装配（与 bench_card 同源）
def build_prog(kind, m, n, k, device="cpu"):
    """按 `ascend/bench_card.py::_cube_setup` 的同一条路把存档 Cube 形态整成 PrimFunc。

    返回 `(prog, args, out, ref)`：`args` 是调用参数、`out` 是它会就地写的张量，
    `ref` 是 torch 参考（fp64 算完再降回 fp32）。**逐字对齐 `_cube_setup`**（同 `arc.BLOCK_M`
    / `arc.BLOCK_TC` / bn=64 / bk=64）——本探针测的就是那条腿。
    """
    import torch
    import S_perf_track_cube as arc

    if kind == "gemm":
        A = torch.randn(m, k, device=device, dtype=torch.float32)
        W = torch.randn(n, k, device=device, dtype=torch.float32)
        b = torch.randn(n, device=device, dtype=torch.float32)
        C = torch.empty(m, n, device=device, dtype=torch.float32)
        prog = arc.gemm_cube_impl(A, W, b, C, n=n, k=k, bm=arc.BLOCK_M, bn=64, bk=64,
                                  act_mode=0)
        args = (A, W, b, C)
        ref = lambda: (A.double() @ W.double().t() + b.double()).float()   # noqa: E731
        out = C
    else:
        dY = torch.randn(m, n, device=device, dtype=torch.float32)
        A = torch.randn(m, k, device=device, dtype=torch.float32)
        GW = torch.empty(n, k, device=device, dtype=torch.float32)
        prog = arc.dw_l0tr_cube_impl(dY, A, GW, n=n, k=k, tc=arc.BLOCK_TC, bn=64, bk=64)
        args = (dY, A, GW)
        ref = lambda: (dY.double().t() @ A.double()).float()               # noqa: E731
        out = GW
    return prog, args, out, ref


def devsrc_of(prog, target="ascend"):
    """**codegen-only** 取 device 源（不起 bisheng）：`tilelang.lower(..., enable_device_compile=False)`。

    这是本件能在 cann910b-h（cann-8.5.0，无卡）复跑的根因——cann-8.5.0 的 bisheng 对
    `__mix__`+死代码源会 **SIGSEGV**（clang frontend exit 139），走真编译反而拿不到证据；
    而窗口2 卡环境（cann-8.5.2）能编过，故那边出的是"编过了、但跑的是死代码"的假绿数。
    """
    import tilelang
    art = tilelang.lower(prog, target=target, enable_device_compile=False)
    return str(art.kernel_source)


# ─────────────────────────────────────────────────── 模式
def mode_static(a):
    """容器可跑：codegen-only 取源 + 打形状/循环边界 + 死代码判定（无设备、无编译器）。"""
    print(f"INTRINSIC-STATIC kind={a.kind} shape={a.m}x{a.n}x{a.k}")
    print(f"  compat_hint: {_COMPAT_HINT}")
    prog, _args, _out, _ref = build_prog(a.kind, a.m, a.n, a.k, device="cpu")
    src = devsrc_of(prog, target=a.lower_target)
    print(f"  [lower] target={a.lower_target} enable_device_compile=False → src_len={len(src)}")
    ev = shape_loop_evidence(src)
    for kk, v in ev.items():
        print(f"  {kk} = {v}")
    cls = classify_devsrc(src)
    for kk, v in cls.items():
        print(f"  {kk} = {v}")
    print(f"INTRINSIC-STATIC-VERDICT kind={a.kind} shape={a.m}x{a.n}x{a.k} "
          f"mode={cls['mode']} verdict={cls['verdict']}")
    if cls["verdict"] == "NOT-EXECUTED":
        print("  ⇒ 归类 = **未真执行**（死代码）；该件任何 p50/tflops 都是空转数，作废。")
    elif cls["verdict"] == "EXECUTES":
        print("  ⇒ 真执行：该件若崩则如实证 507015（DMA 非法访问），不产假绿数。")
    print("--- device source (head 1400) ---")
    print(src[:1400])
    return 0


def mode_truth(a):
    """真卡：G1 数值前置 → 过则 G3 夹计 → G2 峰值闸。"""
    import time

    import torch
    try:
        import torch_npu  # noqa: F401
        assert torch.npu.is_available()
    except Exception:                                         # noqa: BLE001
        print(f"INTRINSIC-TRUTH-SKIP kind={a.kind} reason=no-npu-device "
              f"(本机/容器无设备；真机请带 --mode truth 在卡上跑)")
        return 0
    import tilelang
    prog, args, out, ref = build_prog(a.kind, a.m, a.n, a.k, device="npu")
    kern = tilelang.compile(prog, out_idx=[], target="ascend")
    # G1：数值前置——先跑一次，对拍；不过 ⇒ 拒绝计时，直接 FAIL
    try:
        kern(*args)
        torch.npu.synchronize()
    except Exception as exc:                                  # noqa: BLE001
        print(f"INTRINSIC-FAIL kind={a.kind} shape={a.m}x{a.n}x{a.k} gate=G1 "
              f"reason=launch-raised:{type(exc).__name__}（拒绝计时）")
        return 1
    ok, detail = numeric_gate(out.detach().float().cpu().numpy(), ref().cpu().numpy())
    if not ok:
        print(f"INTRINSIC-FAIL kind={a.kind} shape={a.m}x{a.n}x{a.k} gate=G1 "
              f"reason=numeric-mismatch({detail})（**拒绝计时**，不产 p50/tflops）")
        return 1
    # G3：夹 sync 计时
    for _ in range(a.warmup):
        kern(*args)
    torch.npu.synchronize()
    ts = []
    for _ in range(a.iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        kern(*args)
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    p50 = ts[len(ts) // 2]
    flops = 2.0 * a.m * a.n * a.k
    tflops = flops / (p50 / 1e3) / 1e12
    # G2：峰值闸
    ok2, d2 = peak_gate(tflops)
    print(f"INTRINSIC-TRUTH kind={a.kind} shape={a.m}x{a.n}x{a.k} gate=G1:{detail} p50_ms={p50:.4f} "
          f"tflops={tflops:.4f} peak={PEAK_TFLOPS_FP16} gate=G2:{d2}")
    if not ok2:
        print(f"INTRINSIC-FAIL kind={a.kind} gate=G2 reason=over-peak({tflops:.3f}>{PEAK_TFLOPS_FP16})")
        return 1
    return 0


def mode_neg(a):
    """负例自证（无设备）：喂错的输出 ⇒ G1 必挂；喂超峰值 tflops ⇒ G2 必挂。"""
    import numpy as np
    ref = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    ok_good, d_good = numeric_gate(ref.copy(), ref)               # 真值 ⇒ 必过
    ok_bad, d_bad = numeric_gate(ref * 0.0, ref)                  # 全零（= 空转/未写回）⇒ 必挂
    ok_swap, d_swap = numeric_gate(ref.T.copy(), ref)             # 转置（= 朝向错）⇒ 必挂
    ok_huge, d_huge = peak_gate(3365.919, peak=PEAK_TFLOPS_FP16)  # void 那条的印出值 ⇒ 必挂
    ok_okp, d_okp = peak_gate(3.355, peak=PEAK_TFLOPS_FP16)       # 真 TFLOP/s 量级 ⇒ 过
    print(f"NEG-SELFPROOF numeric_good ok={ok_good} ({d_good})")
    print(f"NEG-SELFPROOF numeric_zeros ok={ok_bad} ({d_bad})")
    print(f"NEG-SELFPROOF numeric_transposed ok={ok_swap} ({d_swap})")
    print(f"NEG-SELFPROOF peak_3365 ok={ok_huge} ({d_huge})")
    print(f"NEG-SELFPROOF peak_3p355 ok={ok_okp} ({d_okp})")
    bad = (not ok_good) or ok_bad or ok_swap or ok_huge or (not ok_okp)
    print(f"NEG-SELFPROOF-DONE all_gates_behave={not bad} "
          f"(期望 numeric_good=True / numeric_zeros=False / numeric_transposed=False / "
          f"peak_3365=False / peak_3p355=True)")
    print(f"NEG-SELFPROOF peak_src={PEAK_SRC}")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="intrinsic 对照腿真伪核查探针（P1-1t 收尾 V2）")
    ap.add_argument("--mode", choices=("static", "truth", "neg"), default="static")
    ap.add_argument("--kind", choices=("gemm", "dw"), default="gemm")
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--lower-target", default="ascend",
                    help="codegen 的目标方言（static 档用；不起编译器）")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    a = ap.parse_args(argv)
    return {"static": mode_static, "truth": mode_truth, "neg": mode_neg}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
