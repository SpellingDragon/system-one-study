#!/usr/bin/env python3
"""ascend/bench_card.py — P1-1o 吞吐/报价工装（R6/R7 与 C5 gate 报价的前置）。

【这是什么】在一次卡上窗口里顺手采完的**报价表**：对 7 类接口件（linear / dW / add_ln / rope /
attn_sw / gdn conv / gdn delta fwd+bwd）做"预热 + 多次统计"的吞吐测量，每个 (op, 位宽, 形状)
配置打印**一行** `BENCH-*`，末尾一行 `BENCH-SUMMARY` + 若干 `BENCH-QUOTE`，判决行可直接抄进 run
notes。本件**不做数值判决**（数值归 `port910b/run_numerics.py`），只补 D12 分段记账缺的那两个
物理量：**每次调用多少毫秒、每小时能完成多少工作量**。

【环境前提（先读，别把两轨混起来）】
 - **本机（macOS / `release/.venv`）无 torch_npu、无昇腾设备** ⇒ 在本机只能做静态自证：
   ① `py_compile` + `--help`；② `--device cpu` 的 **CPU 轨冒烟**——同一套计时骨架、同一套统计
   聚合、同一套异常隔离与假绿守卫**真跑一遍**，但产出的 ms/tflops 是**骨架自证物，不是卡上
   性能**，禁止抄进任何 run 结论（脚本自身在 cpu 轨上**不产** `BENCH-QUOTE`/`BENCH-ABLATE` 行，
   且每行尾挂 `track=cpu/...` + `note=cpu-track-NOT-card-perf`）。
 - 真报价须在 910B 窗内：`python3 ascend/bench_card.py`（默认 `--device npu`，target 自动 ascend）。
 - 容器（有 bisheng 无设备）用 `--compile-check`：只证"每条 BENCH 背后的模具真编得出来"，不计时。
 - 消费方式：`card_run.sh` 以 `SYS1_BENCH=1` 可选调用（B 代理接线）。**契约**只有两条：文件名
   `bench_card.py` + 打印 `BENCH-*` 行。本件不改 card_run.sh、不改任何 `*_asc.py`/`*_kernel.py`/
   `autograd_asc.py`/`train_step.py`/`run_numerics.py`。

【纪律（为什么这些数可信）】
 ① 预热 + 多次统计：warmup 次不计入；正式 iters 次**每次**用 `torch.npu.synchronize()` 前后夹计
    （不是单次、不是整段平均），报 p50（中位数）/ mean / p95；
 ② 单次异常不中断整表：每个配置独立 try/except，失败打 `BENCH-<op> FAIL <原因首行>`，其余照跑
    （`--selftest-fail` 可注入一条必然异常来自证这条纪律真生效）；
 ③ 预算可控：默认每 op iters=20，`--quick`（只各 op 最小档形状 + iters 5 + 预算 240s）、
    `--iters N`、`--budget-sec S`（超预算的后续配置打 `SKIP budget-exhausted`；绝不拖成小时级）；
 ④ **假绿守卫**（判据与措辞沿用 `run_numerics.py::_route_key`，不另造判据）：入口件在 plan 拒形状
    /后端不可用/设备-目标不匹配时会**静默落 torch eager**，照样交出正确的数——量到的却是库速度。
    所以每条 BENCH 行必须自证 `ascend_env.compiled_keys()` 里出现了 `<target>|<op>[...]` 键，
    并把该键打进 `kernel=` 字段；守卫不响 ⇒ 该配置进 FAIL 而不是进表；
 ⑤ 计时包裹的是**接口 home 入口件**（`ascend/kernels/*_kernel.py`），即 `train_step` 经
    `autograd_asc` 消费的同一份内核与同一条路由；autograd 记账开销不计入（报价按 op 计）。

【标量 vs intrinsic 并排（G9 三栈消融）】`--intrinsic auto` 追加 `*_intrinsic` 对照腿：linear/dW
用 K 波 Cube/l0tr **存档形态**（`kernels/reconcile/S_perf_track_cube.py`，P1-1f 改道前的 intrinsic
路，可编但数值属 V 系待证、真机已知 507015）。对照腿**排在整表最后**（cube 腿可能把 aicore 打到
异常态，先行的报价行已逐行 flush，不会被连坐）。两腿同名 op + `_intrinsic` 后缀并排，另补
`BENCH-ABLATE`（同形状 标量 p50 / intrinsic p50 比值）——那正是研究报告要的消融数据。
若卡窗紧张或担心设备上下文被污染：`--intrinsic off`。对照腿的失败单列 `fail_ref`，不混进
`fail_core`（"存档件在卡上崩"是既有事实，不是本工装的缺陷，也不该被算成报价表的失败）。

【定价口径（本脚本不定价）】D12 的 ¥20/h 与 ¥600 熔断线由编排者持有，本件只交物理量
（`calls_per_h` / `unit_per_h` / `gflop_per_h`），换算式写在 `BENCH-NOTE` 行里：
    ¥/1k_token = ¥per_h × 1000 ÷ unit_per_h ； 卡时数 = Σ(每步该 op 调用次数 ÷ calls_per_h) ；
    熔断判定 = 该卡时数 × ¥per_h 与 ¥600 比大小（D12 记 ¥600 ÷ ¥20/h = 30 卡时，引用非本件定价）。

用法：
    python3 ascend/bench_card.py                                # 卡上全表（npu/ascend）
    python3 ascend/bench_card.py --quick --only linear,dW        # 卡上赶时间
    python3 ascend/bench_card.py --device cpu --iters 5          # 本机 CPU 轨冒烟（非性能结论）
    python3 ascend/bench_card.py --device cpu --target ascend --compile-check   # 容器证真件
    python3 ascend/bench_card.py --device cpu --selftest-fail    # 自证异常隔离
    python3 ascend/bench_card.py --list-ops                      # 看配置矩阵
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys
import time

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")

_ROOT = str(pathlib.Path(__file__).resolve().parents[1])        # .../release
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from ascend.kernels import (  # noqa: E402
    add_ln_kernel,
    ascend_env,
    attn_sw_kernel,
    gemm_bwd_dw_kernel,
    gemm_kernel,
    gdn_kernel,
    rope_kernel,
)

DT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

#: 模具共享登记：入口件的缓存键（见 `gemm_asc.plan` 返回的 key）只到 n/k/块宽/act，**不含位宽**
#: ⇒ 同形状换位宽会复用同一份编译产物。后命中的位宽行尾打 `same-mold-as=<首个位宽>`，免得
#:    报价行声称"量到了本行位宽的模具"。要每档位宽各取一份模具：按位宽分两趟进程跑。
_MOLD = {}

# ─────────────────────────────────────────────────────────────────── 形状矩阵（表行）
# 位宽口径：linear/dW 的昇腾白名单是 fp32/bf16/fp16 **同宽操作数** + fp32 偏置/累加（见
# gemm_asc.plan / gemm_bwd_dw_asc.plan）；而**向量面与递推面的内核本体目前只吃 fp32**
# （rope 的 ascend 路由限 fp32 是 P1-4 合流时防静默错的显式门，add_ln/attn_sw/gdn 的 plan 同样
# 硬卡 fp32）⇒ 那几件 dtype 如实写 fp32，fp16 档属 P1-1m（dtype 矩阵未收，见 BENCH-NOTE）。
#: linear：三档方阵 × 两档位宽（任务书要求 512/1024/2048³，bf16→fp32 与 fp16 档）
LIN_MATRIX = ((512, "bf16>fp32"), (1024, "bf16>fp32"), (2048, "bf16>fp32"),
              (512, "fp16>fp16"), (1024, "fp16>fp16"), (2048, "fp16>fp16"))
#: dW：(m=token 轴, n, k=权重形状) 阶梯 × 两档位宽；m 是产品件的动态维
DW_MATRIX = ((1024, 512, 512, "bf16>fp32"), (2048, 1024, 1024, "bf16>fp32"),
             (4096, 2048, 2048, "bf16>fp32"), (1024, 512, 512, "fp16>fp16"),
             (2048, 1024, 1024, "fp16>fp16"), (4096, 2048, 2048, "fp16>fp16"))
#: 向量/递推面：(op, dtype, shape 标签, tier, 守卫子串, 装配器名, 装配参数)；tier0=最小档
VEC_MATRIX = (
    ("add_ln", "fp32", "rows8192xdim1024", 0, "add_ln[", "add_ln", (8192, 1024)),
    ("add_ln", "fp32", "rows16384xdim2048", 1, "add_ln[", "add_ln", (16384, 2048)),
    ("rope", "fp32", "tok4096xh8xd128", 0, "rope[", "rope", (4096, 8, 128)),
    ("rope", "fp32", "tok16384xh8xd128", 1, "rope[", "rope", (16384, 8, 128)),
    ("attn_sw", "fp32", "h8xseq1024xd128w512", 0, "attn_sw[", "attn_sw", (8, 1024, 128, 512)),
    ("attn_sw", "fp32", "h8xseq4096xd128w512", 1, "attn_sw[", "attn_sw", (8, 4096, 128, 512)),
    ("gdn_conv", "fp32", "C1024xL2048xk4", 0, "gdnconv[", "conv", (1024, 2048)),
    ("gdn_conv", "fp32", "C2048xL8192xk4", 1, "gdnconv[", "conv", (2048, 8192)),
    ("gdn_delta_fwd", "fp32", "h8xseq512xk64xv64", 0, "gdn[", "delta_fwd", (8, 512, 64, 64)),
    ("gdn_delta_fwd", "fp32", "h8xseq2048xk128xv128", 1, "gdn[", "delta_fwd", (8, 2048, 128, 128)),
    ("gdn_delta_bwd", "fp32", "h8xseq512xk64xv64", 0, "gdnb[", "delta_bwd", (8, 512, 64, 64)),
    ("gdn_delta_bwd", "fp32", "h8xseq2048xk128xv128", 1, "gdnb[", "delta_bwd", (8, 2048, 128, 128)),
)
#: intrinsic 对照腿形状（与 scalar 腿的 shape 标签**逐字同形**，否则 BENCH-ABLATE 配不上对）
CUBE_SHAPES = ((512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048))
DW_CUBE_SHAPES = tuple((m, n, k) for m, n, k, _dt in DW_MATRIX[:3])

#: 形状族出处：512/1024/2048³ 承 `run_numerics.py::num_gemm_direct`（bf16 2048³ 的 e2e 形态）与
#: P0-2 卡上唯余项 `RUNTIME-PASS{json}`（2048³ bf16 数值+tflops）；块口径参照 `attempts/A/a2_probe.py`
#: （64 基块）与 `reconcile/S_RESULT.md` 的核块预算（NUM_BLOCKS=8、BLOCK_LADDER=64/32/16、
#: UB 每核块 ~24KB）；token/头/维档参照 `train_step.py` 的生产拓扑（h8 / head_dim128 / window512）。


# ────────────────────────────────────────────────────────── 装配器（计时区之外建张量）
# 约定：`setup(device, target) -> (call, flops_or_None, (unit_name, work_per_call))`
#   call  = 跑一次该 op 的零参闭包（计时就包它）
#   flops = 该 op 每次调用的乘加次数；None 表示没有可辩护的口径（纯 elementwise/归约）
#           ⇒ 打印 tflops=na，绝不拿假 TFLOPS 充数
#   unit  = "每小时可完成工作量"的计价单位（本表统一用 token）
def _build_linear(m, n, k, in_dt, out_dt):
    def setup(device, target):
        A = torch.randn(m, k, device=device, dtype=in_dt)
        W = torch.randn(n, k, device=device, dtype=in_dt)
        b = torch.randn(n, device=device, dtype=torch.float32)
        call = lambda: gemm_kernel.forward(A, W, b, "none", out_dt, target)  # noqa: E731
        return call, 2.0 * m * n * k, ("token", m)
    return setup


def _build_dw(m, n, k, in_dt, out_dt):
    """dW = dYᵀ @ A（反向权重梯度，产品件 `gemm_bwd_dw_kernel.backward`）。"""
    def setup(device, target):
        dY = torch.randn(m, n, device=device, dtype=in_dt)
        X = torch.randn(m, k, device=device, dtype=in_dt)
        call = lambda: gemm_bwd_dw_kernel.backward(dY, X, out_dtype=out_dt,  # noqa: E731
                                                   target=target)
        return call, 2.0 * m * n * k, ("token", m)
    return setup


def _build_add_ln(rows, dim):
    def setup(device, target):
        x = torch.randn(rows, dim, device=device, dtype=torch.float32)
        res = torch.randn(rows, dim, device=device, dtype=torch.float32)
        g = torch.ones(dim, device=device, dtype=torch.float32)
        b = torch.zeros(dim, device=device, dtype=torch.float32)
        call = lambda: add_ln_kernel.forward(x, res, g, b, 1e-5, torch.float32,  # noqa: E731
                                             target)
        return call, None, ("token", rows)          # LN=归约+逐元素，无乘加口径 ⇒ tflops=na
    return setup


def _build_rope(tokens, heads, dim):
    def setup(device, target):
        qkv = torch.randn(tokens, 3, heads, dim, device=device, dtype=torch.float32)
        cos = torch.randn(tokens, dim // 2, device=device, dtype=torch.float32)
        sin = torch.randn(tokens, dim // 2, device=device, dtype=torch.float32)
        call = lambda: rope_kernel.forward(qkv, cos, sin, rope_kernel.DEFAULT_SLOTS,  # noqa: E731
                                           target)
        return call, None, ("token", tokens)
    return setup


def _build_attn_sw(heads, seq, dim, window):
    def setup(device, target):
        q = torch.randn(heads, seq, dim, device=device, dtype=torch.float32)
        k = torch.randn(heads, seq, dim, device=device, dtype=torch.float32)
        v = torch.randn(heads, seq, dim, device=device, dtype=torch.float32)
        call = lambda: attn_sw_kernel.forward(q, k, v, window, None,          # noqa: E731
                                              torch.float32, target)
        pairs = _window_pairs(seq, window)
        return call, 4.0 * heads * pairs * dim, ("token", seq)   # QK^T 与 PV 两段乘加
    return setup


def _build_conv(chans, length):
    def setup(device, target):
        x = torch.randn(chans, length, device=device, dtype=torch.float32)
        w = torch.randn(chans, gdn_kernel.CONV_KERNEL, device=device, dtype=torch.float32)
        b = torch.randn(chans, device=device, dtype=torch.float32)
        call = lambda: gdn_kernel.conv_forward(x, w, b, torch.float32, target)  # noqa: E731
        return call, 2.0 * chans * length * gdn_kernel.CONV_KERNEL, ("token", length)
    return setup


def _build_delta_fwd(heads, seq, dk, dv):
    def setup(device, target):
        q, k, gate = (_rand3(heads, seq, dk, device), _rand3(heads, seq, dk, device),
                      _rand3(heads, seq, dk, device))
        v = _rand3(heads, seq, dv, device)
        beta = torch.rand(heads, seq, device=device, dtype=torch.float32)
        call = lambda: gdn_kernel.forward(q, k, v, gate, beta, torch.float32,  # noqa: E731
                                          target)
        return call, 4.0 * heads * seq * dv * dk, ("token", seq)
    return setup


def _build_delta_bwd(heads, seq, dk, dv):
    def setup(device, target):
        q, k, gate = (_rand3(heads, seq, dk, device), _rand3(heads, seq, dk, device),
                      _rand3(heads, seq, dk, device))
        v = _rand3(heads, seq, dv, device)
        beta = torch.rand(heads, seq, device=device, dtype=torch.float32)
        dout = _rand3(heads, seq, dv, device)
        call = lambda: gdn_kernel.backward(q, k, v, gate, beta, dout, target)   # noqa: E731
        return call, 4.0 * heads * seq * dv * dk, ("token", seq)
    return setup


#: 装配器名 → 工厂（VEC_MATRIX 只存名字，避免模块级前向引用）
FACTORIES = {"add_ln": _build_add_ln, "rope": _build_rope, "attn_sw": _build_attn_sw,
             "conv": _build_conv, "delta_fwd": _build_delta_fwd,
             "delta_bwd": _build_delta_bwd}


def _rand3(h, s, d, device):
    """(heads, seq, width) 的 fp32 输入。递推门用 randn 即可——本件只量时间不判数值。"""
    return torch.randn(h, s, d, device=device, dtype=torch.float32)


def _window_pairs(seq, window):
    """因果滑窗里真正参与加权的 (query, key) 对数：Σ_i min(i+1, window)。"""
    if seq <= window:
        return seq * (seq + 1) // 2
    tri = window * (window + 1) // 2                  # 前 window 行是三角段
    rest = seq - window
    return tri + rest * window


def _split_dt(label):
    """`bf16>fp32` → (torch.bfloat16, torch.float32)。"""
    a, b = label.split(">")
    return DT[a], DT[b]


# ────────────────────────────────────────────────────────── 配置清单展开
def build_configs(quick=False):
    """把形状矩阵展平成"每行一条 BENCH"的配置清单（顺序即打印顺序，报价表按此抄）。"""
    cfgs = []
    lin_min = min(s for s, _d in LIN_MATRIX)
    for side, dt_label in LIN_MATRIX:
        if quick and side != lin_min:
            continue
        in_dt, out_dt = _split_dt(dt_label)
        cfgs.append(("linear", dt_label, f"{side}x{side}x{side}", 0 if side == lin_min else 1,
                     "gemm[", _build_linear(side, side, side, in_dt, out_dt)))
    dw_min = min(m for m, _n, _k, _d in DW_MATRIX)
    for m, n, k, dt_label in DW_MATRIX:
        if quick and m != dw_min:
            continue
        in_dt, out_dt = _split_dt(dt_label)
        cfgs.append(("dW", dt_label, f"m{m}xn{n}xk{k}", 0 if m == dw_min else 1,
                     "gemm_dw[", _build_dw(m, n, k, in_dt, out_dt)))
    for op, dt_label, shape, tier, sub, fac, args in VEC_MATRIX:
        if quick and tier:
            continue
        cfgs.append((op, dt_label, shape, tier, sub, FACTORIES[fac](*args)))
    return cfgs


def build_intrinsic_configs(target, quick=False):
    """`*_intrinsic` 对照腿（K 波 Cube/l0tr 存档形态，**非产品件**）。

    只在 ascend 路由下挂：存档件正文 import `tilelang.ascend.language` 并用 `T.Cube`/alloc_l1，
    cpu 方言编不出（真挂上去只会得到如实的 FAIL）。shape 标签与 scalar 腿逐字同形 ⇒ ABLATE 配对。
    """
    if target != ascend_env.TARGET_ASCEND:
        return []
    lin = CUBE_SHAPES[:1] if quick else CUBE_SHAPES
    dw = DW_CUBE_SHAPES[:1] if quick else DW_CUBE_SHAPES
    out = []
    for m, n, k in lin:
        out.append(("linear_intrinsic", "fp32", f"{m}x{n}x{k}", 1, "n/a",
                    _cube_setup("gemm", m, n, k)))
    for m, n, k in dw:
        out.append(("dW_intrinsic", "fp32", f"m{m}xn{n}xk{k}", 1, "n/a",
                    _cube_setup("dw", m, n, k)))
    return out


def _cube_setup(kind, m, n, k):
    """把 K 波 Cube/l0tr 存档形态整成一次调用（性能对照专用，不经 ascend_env 路由）。

    出处：`ascend/kernels/reconcile/S_perf_track_cube.py`（P1-1f 改道前的 intrinsic 路；签名与
    当时的 `gemm_asc_impl`/`dw_asc_impl` 逐字一致，两件都 fp32 声明、m 是动态轴）。因为它自己
    `tilelang.compile` 出句柄、没有"落 eager"这条路，守卫子串记 `n/a`（见 `_route_key`）。
    """
    def setup(device, target):
        import tilelang
        rec = str(pathlib.Path(_ROOT) / "ascend" / "kernels" / "reconcile")
        if rec not in sys.path:
            sys.path.insert(0, rec)
        import S_perf_track_cube as arc

        A = torch.randn(m, k, device=device, dtype=torch.float32)
        W = torch.randn(n, k, device=device, dtype=torch.float32)
        b = torch.randn(n, device=device, dtype=torch.float32)
        C = torch.empty(m, n, device=device, dtype=torch.float32)
        dY = torch.randn(m, n, device=device, dtype=torch.float32)
        GW = torch.empty(n, k, device=device, dtype=torch.float32)
        if kind == "gemm":
            prog = arc.gemm_cube_impl(A, W, b, C, n=n, k=k, bm=arc.BLOCK_M, bn=64, bk=64,
                                      act_mode=0)
            args = (A, W, b, C)
        else:
            prog = arc.dw_l0tr_cube_impl(dY, A, GW, n=n, k=k, tc=arc.BLOCK_TC, bn=64, bk=64)
            args = (dY, A, GW)
        kern = tilelang.compile(prog, out_idx=[], **ascend_env.compile_kwargs(target))
        call = lambda: kern(*args)  # noqa: E731
        return call, 2.0 * m * n * k, ("token", m)
    return setup


# ────────────────────────────────────────────────────────── 计时核心
def _sync_fn(device):
    """npu 轨返回 `torch.npu.synchronize`；cpu 轨返回空操作（无设备可同步）。"""
    if device == "npu" and hasattr(torch, "npu"):
        return torch.npu.synchronize
    return lambda: None


def measure(call, iters, warmup, sync):
    """预热 warmup 次（不计入），再逐次"夹计"iters 个样本；返回毫秒样本列表。

    白话：先让机器把活儿干顺（首编、缓存、频率爬升都算在热身里），然后每次开工前收一次尾、
    收工后再同步一次掐表——单次的抖动不污染中位数，p95 又能把尾巴露出来。
    """
    for _ in range(max(0, warmup)):
        call()
    sync()
    samples = []
    for _ in range(max(1, iters)):
        sync()
        t0 = time.perf_counter()
        call()
        sync()
        samples.append((time.perf_counter() - t0) * 1e3)
    return samples


def _pct(xs, q):
    """线性插值分位数（n=1 时等于自身；p50 即中位数）。"""
    s = sorted(xs)
    if not s:
        return float("nan")
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _stats(samples):
    return {"n": len(samples), "p50": _pct(samples, 0.50),
            "mean": sum(samples) / max(1, len(samples)), "p95": _pct(samples, 0.95)}


def _mem_mb(device):
    """(当前存活分配, 峰值分配) MB；cpu 轨按任务书口径降级为 `na`（分配器语义不同源）。"""
    if device != "npu" or not hasattr(torch, "npu"):
        return "na", "na"
    try:
        return (f"{torch.npu.memory_allocated() / 2 ** 20:.1f}",
                f"{torch.npu.max_memory_allocated() / 2 ** 20:.1f}")
    except Exception:  # noqa: BLE001  取不到内存计数不该废掉吞吐判决
        return "na", "na"


def _reset_peak(device):
    """每个配置开始前把峰值计数清零并回收缓存块 ⇒ `mem_peak_mb` 是**这一条 op 的**峰值。"""
    if device == "npu" and hasattr(torch, "npu"):
        try:
            torch.npu.reset_peak_memory_stats()
            torch.npu.empty_cache()
        except Exception:  # noqa: BLE001
            pass


# ────────────────────────────────────────────────────────── 假绿守卫（照 run_numerics._route_key）
def _route_key(substr, target, before, cnt0):
    """这一趟必须真开了模具：`compiled_keys()` 里要有 `<target>|...<substr>...` 键。

    判据与措辞沿用 `port910b/run_numerics.py::_route_key`（**不另造判据**）：入口件在 plan 拒形状
    /后端不可用/方言别名未生效时静默落 torch eager 并照样交出正确的数——报价表把那种数进表就是
    骗人。返回 (键, 复用状态)：`new`=本次新编（键是这一趟出现的），`cache`=此前已编成并复用。
    """
    if substr == "n/a":            # 对照腿不经 ascend_env 缓存（自己 compile），句柄即证据
        return f"{target}|out-of-band-cube-archive", "new"
    keys = [k for k in ascend_env.compiled_keys()
            if substr in k and k.startswith(f"{target}|")]
    if not keys:
        raise AssertionError(
            f"假绿守卫：compiled_keys 无 {target}|*{substr}* ⇒ 本趟走的是 torch eager，不是内核"
            f"（plan 拒形状 / 后端不可用 / 设备-目标不匹配）；"
            f"blockers={list(ascend_env.blockers().items())[:2]}")
    fresh = any(k not in before for k in keys) or ascend_env.compile_count() > cnt0
    return keys[-1], ("new" if fresh else "cache")


# ────────────────────────────────────────────────────────── 行格式
def _fmt(v):
    return "na" if (v != v) else f"{v:.3f}"          # NaN ⇒ na（不给假数）


def bench_line(op, dtype, shape, st, flops, iters, warmup, device, target, key, reuse, note):
    """契约行：`BENCH-<op> dtype= shape= p50_ms= mean_ms= p95_ms= iters= warmup= tflops=
    mem_alloc_mb= mem_peak_mb=`，后随本件自证的附加字段 track/kernel/reuse/note。"""
    tf = "na"
    if flops is not None and st["p50"] == st["p50"] and st["p50"] > 0:
        tf = f"{flops / (st['p50'] / 1e3) / 1e9:.3f}"          # TFLOPS 口径 = p50（中位数）
    alloc, peak = _mem_mb(device)
    parts = [f"BENCH-{op}", f"dtype={dtype}", f"shape={shape}",
             f"p50_ms={_fmt(st['p50'])}", f"mean_ms={_fmt(st['mean'])}",
             f"p95_ms={_fmt(st['p95'])}", f"iters={iters}", f"warmup={warmup}",
             f"tflops={tf}", f"mem_alloc_mb={alloc}", f"mem_peak_mb={peak}",
             f"track={device}/{target}", f"kernel={key}", f"reuse={reuse}"]
    if note:
        parts.append(f"note={note}")
    return " ".join(parts)


def quote_line(op, dtype, shape, p50, unit_name, work_per_call, flops):
    """把 p50 换成"每小时可完成工作量"——D12 分段记账要的物理量（本件不乘价格）。"""
    if not (p50 > 0) or p50 != p50:
        return None
    calls_h = 3600.0 / (p50 / 1e3)
    g = "" if flops is None else f" gflop_per_h={flops * calls_h / 1e9:.1f}"
    return (f"BENCH-QUOTE op={op} dtype={dtype} shape={shape} unit={unit_name} "
            f"work_per_call={work_per_call} calls_per_h={calls_h:.1f} "
            f"unit_per_h={calls_h * work_per_call:.1f}{g} basis=p50 track=npu")


def ablate_line(base, dtype, shape, ms_scalar, ms_intr, intr_dtype):
    """标量路 vs intrinsic 路并排（G9 三栈消融行）。任一腿没成 ⇒ ratio=na，绝不硬凑。"""
    r = "na" if not (ms_scalar and ms_intr and ms_intr > 0) else f"{ms_scalar / ms_intr:.1f}x"
    return (f"BENCH-ABLATE op={base} dtype={dtype}/{intr_dtype} shape={shape} "
            f"scalar_p50_ms={_fmt(ms_scalar)} intrinsic_p50_ms={_fmt(ms_intr)} ratio={r} "
            f"track=npu")


# ────────────────────────────────────────────────────────── 容器档：只证"计时包裹的是真件"
class _LaunchSkip(RuntimeError):
    """编译已成、发射被桩掉时抛它（口径照 `run_numerics.py::_LaunchSkip`）。"""


def _install_launch_skip():
    """把 `tilelang.compile` 换成"真编译 + 调用即跳发射"的替身（无设备只判编译时用）。

    手法照 `port910b/run_numerics.py::_install_launch_skip`（**复用既有口径，不另造判据**）：
    编译真发生、发射不尝试，`box["src"]` 收最后一次真编译产物的源码字节数。
    """
    import tilelang as _tl

    real = _tl.compile
    box = {"src": 0}

    def _c(*a, **k):
        kern = real(*a, **k)                 # 真编译：失败原样抛，由 get_compiled 记 blocker
        box["src"] = len(str(kern.get_kernel_source())) if hasattr(kern, "get_kernel_source") else 0

        class _NoLaunch:
            def __call__(self, *aa, **kk):
                raise _LaunchSkip("compile-check 只判编译，不发射")

            def __getattr__(self, n):
                return getattr(kern, n)

        return _NoLaunch()

    _tl.compile = _c

    def restore():
        _tl.compile = real

    return restore, box


def _stub_device_gate():
    """无设备时把"设备-目标匹配门"桩开（照 `run_numerics._prod_setup` 的 card+无 npu 档）。

    只有 `--compile-check` 用得上：这一档**不产任何计时行**，产出是"这条 BENCH 配置背后的模具
    确实编得出来"，所以不能让它因为"张量在 cpu、产物属 ascend"被判 eager。
    """
    saved = {n: getattr(ascend_env, n) for n in
             ("backend_available", "_npu_present", "active_backend")}
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda *a, **k: ascend_env.TILELANG

    def restore():
        for n, v in saved.items():
            setattr(ascend_env, n, v)

    return restore


def compile_check(cfg, device, target):
    """容器档：装配 + 真编译（不计时、不发射），打一行 `BENCH-<op>-COMPILE-CHECK`。"""
    op, dtype, shape, _tier, substr, setup = cfg
    ascend_env.reset()                       # 每件从零：compiled_keys/blockers 不许串件
    restore_gate = _stub_device_gate()
    restore_compile, box = _install_launch_skip()
    try:
        call, _flops, _unit = setup(device, target)
        try:
            call()                            # 真编译在装配/首调里；发射必抛 _LaunchSkip
        except _LaunchSkip:
            pass
        else:
            raise AssertionError("发射没被桩掉（_LaunchSkip 未响）⇒ compile-check 前提不成立")
        try:
            key = _route_key(substr, target, before=set(), cnt0=0)[0]
        except AssertionError as exc:
            if box["src"]:
                raise AssertionError(
                    f"{_first_line(exc)} | 注: src_bytes={box['src']} 说明本趟真开过模，只是该"
                    f" op 自己的键还没进货架——复合腿（如 gdn 反向要先恢复前向再开伴随模具）在"
                    f"无设备档会被发射桩截断；卡上有设备时此腿应自行走通") from exc
            raise
        return (f"BENCH-{op}-COMPILE-CHECK dtype={dtype} shape={shape} target={target} "
                f"compiled_key={key} src_bytes={box['src']} "
                f"track={device}/{target} note=无设备⇒只证编译不计时")
    finally:
        restore_compile()
        restore_gate()


# ────────────────────────────────────────────────────────── 单配置执行
def run_one(cfg, device, target, iters, warmup, sync):
    """跑一个配置：装配 → 夹计 → 守卫 → 返回 (BENCH 行, p50, flops, unit, work)。

    异常一律上抛给调用方隔离（`BENCH-<op> FAIL`），本函数自己不吞异常。
    """
    op, dtype, shape, _tier, substr, setup = cfg
    before = set(ascend_env.compiled_keys())
    cnt0 = ascend_env.compile_count()
    _reset_peak(device)
    call, flops, (unit_name, work_per_call) = setup(device, target)
    samples = measure(call, iters, warmup, sync)
    key, reuse = _route_key(substr, target, before, cnt0)
    st = _stats(samples)
    note = "cpu-track-NOT-card-perf" if device == "cpu" else ""
    first = _MOLD.setdefault(key, dtype)
    if first != dtype:        # 同模具换了个位宽名字 ⇒ 如实标注（见文件头 _MOLD 注释）
        note = f"{note};same-mold-as={first}" if note else f"same-mold-as={first}"
    line = bench_line(op, dtype, shape, st, flops, iters, warmup, device, target,
                      key, reuse, note)
    return line, st["p50"], flops, unit_name, work_per_call


def _boom(_device, _target):
    """`--selftest-fail` 的合成配置：**装配阶段就抛**，证明连"还没开始计时"的失败也不中断表。"""
    raise RuntimeError("注入的失败：验证 BENCH 单件异常隔离（其余 op 必须照跑）")


def _first_line(exc):
    text = str(exc) or repr(exc)
    return f"{type(exc).__name__}: {text.splitlines()[0][:180]}"


# ────────────────────────────────────────────────────────── 主流程
def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description="P1-1o 吞吐/报价工装：每个 (op,位宽,形状) 配置打一行 BENCH-*（R6/R7 与 C5 "
                    "gate 报价的前置）。本机 --device cpu 只自证测量骨架，不是卡上性能。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--device", choices=("npu", "cpu"), default="npu",
                    help="张量所在/计时对象设备。npu=真报价；cpu=骨架自证（不产报价行）")
    ap.add_argument("--target", choices=("ascend", "cpu", "auto"), default="auto",
                    help="内核方言路由（传给入口件的 target=）。auto：npu⇒ascend，cpu⇒cpu")
    ap.add_argument("--iters", type=int, default=20, help="正式计时次数（每次前后夹 sync）")
    ap.add_argument("--warmup", type=int, default=3, help="预热次数（首编/热身，不计入统计）")
    ap.add_argument("--quick", action="store_true",
                    help="只跑各 op 最小档形状（tier0）+ iters 5 + warmup 2 + 预算 240s")
    ap.add_argument("--only", default="",
                    help="逗号分隔 op 名过滤（linear/dW/add_ln/rope/attn_sw/gdn_conv/"
                          "gdn_delta_fwd/gdn_delta_bwd，含 *_intrinsic 对照腿同基名）")
    ap.add_argument("--intrinsic", choices=("auto", "off"), default="auto",
                    help="追加 *_intrinsic 对照腿（K 波 Cube/l0tr 存档件；仅 ascend 路由挂载，"
                          "且排在整表最后）")
    ap.add_argument("--budget-sec", type=float, default=0.0,
                    help="整表墙钟预算（0=自动：--quick⇒240s，否则⇒600s）；超预算的后续配置打 SKIP，绝不拖成小时级（卡很贵）")
    ap.add_argument("--compile-check", action="store_true",
                    help="容器档：只装配+真编译证『计时包裹的是真件』，不计时（无设备可用）")
    ap.add_argument("--selftest-fail", action="store_true",
                    help="在第 4 条配置处注入一条必然异常，自证异常隔离不中断整表")
    ap.add_argument("--list-ops", action="store_true", help="列出配置矩阵后退出")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if args.list_ops:
        for cfg in build_configs():
            print(f"OP {cfg[0]:16s} dtype={cfg[1]:10s} shape={cfg[2]:22s} tier={cfg[3]}")
        for cfg in build_intrinsic_configs(ascend_env.TARGET_ASCEND):
            print(f"OP {cfg[0]:16s} dtype={cfg[1]:10s} shape={cfg[2]:22s} tier={cfg[3]} "
                  f"(intrinsic 对照腿)")
        return 0

    device = args.device
    if device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        print("BENCH-ENV device=npu npu_available=False ⇒ 本机无昇腾设备")
        print("BENCH-ABORT no-npu：真报价请在 910B 窗内跑；本机自证用 `--device cpu`（骨架冒烟）"
              "或容器 `--device cpu --target ascend --compile-check`（证真件）")
        return 2

    target = args.target
    if target == "auto":
        target = ascend_env.TARGET_ASCEND if device == "npu" else ascend_env.TARGET_CPU
    if device == "cpu" and target == ascend_env.TARGET_ASCEND and not args.compile_check:
        print("BENCH-NOTE target-downgraded=ascend->cpu（cpu 设备 + ascend 路由必判 eager，"
              "报价行拿不到内核数；容器证真件请加 --compile-check）")
        target = ascend_env.TARGET_CPU

    cfgs = build_configs(quick=args.quick)
    if args.intrinsic == "auto":
        cfgs += build_intrinsic_configs(target, quick=args.quick)
    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        cfgs = [c for c in cfgs if c[0] in want or c[0].replace("_intrinsic", "") in want]
    if args.selftest_fail:
        cfgs.insert(min(3, len(cfgs)), ("selftest", "n/a", "n/a", 0, "", _boom))

    iters, warmup = args.iters, args.warmup
    if args.quick and args.iters == 20:                  # 用户没显式点名 iters 时才降档
        iters, warmup = 5, 2
    budget = args.budget_sec if args.budget_sec > 0 else (240.0 if args.quick else 600.0)
    sync = _sync_fn(device)

    print(f"BENCH-ENV device={device} target={target} torch={torch.__version__} "
          f"npu_available={hasattr(torch, 'npu') and bool(torch.npu.is_available())} "
          f"iters={iters} warmup={warmup} budget_s={budget if budget != float('inf') else 'inf'} "
          f"configs={len(cfgs)} mode={'compile-check' if args.compile_check else 'time'}")

    t0 = time.perf_counter()
    ok = fail = skipped = fail_ref = fail_inj = 0
    scalar_ms, intr_ms, quotes = {}, {}, []
    for cfg in cfgs:
        op, dtype, shape = cfg[0], cfg[1], cfg[2]
        if time.perf_counter() - t0 > budget:
            print(f"BENCH-{op} SKIP budget-exhausted dtype={dtype} shape={shape} "
                  f"wall_s={time.perf_counter() - t0:.1f}", flush=True)
            skipped += 1
            continue
        try:
            if args.compile_check:
                print(compile_check(cfg, device, target), flush=True)
            else:
                line, p50, flops, unit_name, work = run_one(cfg, device, target, iters,
                                                            warmup, sync)
                print(line, flush=True)
                is_ref = op.endswith("_intrinsic")
                base = op[:-len("_intrinsic")] if is_ref else op
                if is_ref:
                    intr_ms[(base, shape)] = (p50, dtype)
                else:
                    scalar_ms[(base, shape, dtype)] = p50
                if device == "npu":
                    q = quote_line(op, dtype, shape, p50, unit_name, work, flops)
                    if q:
                        quotes.append(q)
            ok += 1
        except Exception as exc:  # noqa: BLE001  单次异常不中断整表（任务书纪律②）
            print(f"BENCH-{op} FAIL [{dtype} {shape}] {_first_line(exc)}", flush=True)
            fail += 1
            if op == "selftest":
                fail_inj += 1
            elif op.endswith("_intrinsic"):
                fail_ref += 1

    wall = time.perf_counter() - t0
    for q in quotes:
        print(q, flush=True)
    if device == "npu" and intr_ms:
        for (base, shape, dtype), ms in scalar_ms.items():
            got = intr_ms.get((base, shape))
            if base in ("linear", "dW") and got:
                print(ablate_line(base, dtype, shape, ms, got[0], got[1]), flush=True)

    print(f"BENCH-SUMMARY ops={len(cfgs)} ok={ok} fail={fail} wall_s={wall:.1f} "
          f"skip={skipped} fail_core={fail - fail_ref - fail_inj} fail_ref={fail_ref} fail_inj={fail_inj} "
          f"track={device}/{target} iters={iters} warmup={warmup}", flush=True)
    _emit_notes(device, target, args.compile_check)
    if args.compile_check:
        return 0
    return 0 if ok and (fail - fail_ref - fail_inj) == 0 else 1


def _emit_notes(device, target, compile_check):
    """把 D12 记账的换算口径写成可直接抄的说明行（本件不定价，价格归编排者）。"""
    print("BENCH-NOTE price-authority=orchestrator(本脚本不定价) "
          "formula='¥/1k_token = ¥per_h × 1000 ÷ unit_per_h' "
          "formula='卡时 = Σ(每步该op调用次数 ÷ calls_per_h)' "
          "formula='熔断 = 卡时 × ¥per_h 对比 ¥600（D12：¥600 ÷ ¥20/h = 30 卡时，引用非定价）'")
    print("BENCH-NOTE mold-dtype=报价表的 dtype 列务必与 kernel= 字段互证：同形状换位宽后若键名不变、或行尾出现 same-mold-as= ⇒ 该行量到的是别档位宽的模具，须按位宽分趟进程重采；自证工具 ascend/kernels/reconcile/E2_probe_mold.py，实测记录见 reconcile/E2_RESULT.md")
    print("BENCH-NOTE basis='tflops@p50' stats='p50(mean,p95)' "
          "sync='torch.npu.synchronize 每次前后夹计' warmup='预热不计入' "
          "unit_per_h='每小时可完成工作量（报价直接抄这一列）'")
    print("BENCH-NOTE ablation='标量路(op) vs intrinsic 路(op_intrinsic，K 波 Cube/l0tr 存档件，"
          "kernels/reconcile/S_perf_track_cube.py，非产品件)' ⇒ G9 三栈消融数据；"
          "对照腿已知 507015/V 系未证真 ⇒ 其 FAIL 计入 fail_ref 不计入 fail_core")
    if compile_check:
        print("BENCH-NOTE mode=compile-check 无设备⇒只证『计时包裹的是真编译产物』（发射已桩掉），"
              "本档不产任何耗时/吞吐结论")
    elif device == "cpu":
        print("BENCH-NOTE track=cpu 本表只自证测量骨架（装配/夹计/聚合/守卫/异常隔离真跑一遍），"
              "**禁止把 cpu 轨耗时当卡上性能写进任何结论**；QUOTE/ABLATE 行已按纪律抑制；"
              "cpu 轨上 gdn_delta_bwd 必落 torch（cpu target 无伴随模具）⇒ 那条 FAIL 属预期，"
              "不是骨架缺陷（fail_core 仍如实计数，退出码照它走）")
    else:
        print("BENCH-NOTE dtype-gap=P1-1m 未收：bf16 标量 cast 缺件、向量面/递推面内核只吃 fp32 "
              "⇒ bf16/fp16 行只证可发射与吞吐，不证数值（数值判据归 run_numerics.py）")


if __name__ == "__main__":
    sys.exit(main())
