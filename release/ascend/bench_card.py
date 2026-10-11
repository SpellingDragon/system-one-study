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
 ⑥ **数值自证硬前置（P1-1t 收尾新增，本波正身）**：任何腿（含 `*_intrinsic` 对照腿）在计时**之前**
    必须过数值对拍——核函数输出对 torch 参考（入口件 `target="cpu"` 的尺子路，或存档件的 torch
    矩阵式）算相对 L2 误差，不过阈值就**拒绝计时**。未带对拍的腿打
    `BENCH-<op> NOTIMED reason=no-numeric-selfproof`，对拍挂的腿打
    `... NOTIMED reason=numeric-mismatch(rel=...)`；**两条都不产 `p50/tflops/BENCH-QUOTE`**。
    动机（第二窗实证）：`*_intrinsic` 对照腿的存档 Cube 形态落成 `__mix__(1,2)` 死代码，卡上
    "跑"出 0.080ms / 3365.919 "tflops" 的物理不可能数——那正是**没有数值前置**时假绿数进表的
    写照。自证工具 `kernels/reconcile/V2_probe_intrinsic_truth.py`，定档
    `kernels/reconcile/V2_THROUGHPUT_TRUTH.md`。
    `--selftest-notimed`（无对拍腿 ⇒ 必须 NOTIMED）与 `--selftest-mismatch`（对拍必挂腿 ⇒ 必须
    NOTIMED）负例自证这条硬闸真生效。
 ⑦ **tflops 单位修正（P1-1t 收尾新增）**：`bench_line` 的 `tflops=` 原按 `/1e9` 折算（实为
    GFLOP/s）却标 TFLOPS ⇒ 1000× 虚高；本波改回 `/1e12`（真 TFLOP/s），并把单位写进
    `BENCH-NOTE`。第二窗"3365.919 tflops"一半由此错标造成（真值 3.37 TFLOP/s），另一半是死
    代码腿的空转时间（见 ⑥ 与 `reconcile/V2_THROUGHPUT_TRUTH.md`）。

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

        def verify():
            # 参考 = 入口件自身的 torch 尺子路（target="cpu" ⇒ active_backend 判 eager，fp32 累加后降位宽）
            ref = gemm_kernel.forward(A, W, b, "none", out_dt, "cpu")
            return _verify_vs(call(), ref, _NUM_TOL[out_dt])
        return call, 2.0 * m * n * k, ("token", m), verify
    return setup


def _build_dw(m, n, k, in_dt, out_dt):
    """dW = dYᵀ @ A（反向权重梯度，产品件 `gemm_bwd_dw_kernel.backward`）。"""
    def setup(device, target):
        dY = torch.randn(m, n, device=device, dtype=in_dt)
        X = torch.randn(m, k, device=device, dtype=in_dt)
        call = lambda: gemm_bwd_dw_kernel.backward(dY, X, out_dtype=out_dt,  # noqa: E731
                                                   target=target)

        def verify():
            ref = gemm_bwd_dw_kernel.backward(dY, X, out_dtype=out_dt, target="cpu")
            return _verify_vs(call(), ref, _NUM_TOL[out_dt])
        return call, 2.0 * m * n * k, ("token", m), verify
    return setup


def _build_add_ln(rows, dim):
    def setup(device, target):
        x = torch.randn(rows, dim, device=device, dtype=torch.float32)
        res = torch.randn(rows, dim, device=device, dtype=torch.float32)
        g = torch.ones(dim, device=device, dtype=torch.float32)
        b = torch.zeros(dim, device=device, dtype=torch.float32)
        call = lambda: add_ln_kernel.forward(x, res, g, b, 1e-5, torch.float32,  # noqa: E731
                                             target)

        def verify():
            ref = add_ln_kernel.forward(x, res, g, b, 1e-5, torch.float32, "cpu")
            return _verify_vs_pair(call(), ref, _NUM_TOL[torch.float32])
        return call, None, ("token", rows), verify      # LN=归约+逐元素，无乘加口径 ⇒ tflops=na
    return setup


def _build_rope(tokens, heads, dim):
    def setup(device, target):
        qkv = torch.randn(tokens, 3, heads, dim, device=device, dtype=torch.float32)
        cos = torch.randn(tokens, dim // 2, device=device, dtype=torch.float32)
        sin = torch.randn(tokens, dim // 2, device=device, dtype=torch.float32)
        call = lambda: rope_kernel.forward(qkv, cos, sin, rope_kernel.DEFAULT_SLOTS,  # noqa: E731
                                           target)

        def verify():
            # rope 是**就地**语义（改 qkv 并交回同一个对象）⇒ 参考要喂在调用前的副本上
            base = qkv.clone()
            got = call()
            ref = rope_kernel.forward(base, cos, sin, rope_kernel.DEFAULT_SLOTS, "cpu")
            return _verify_vs(got, ref, _NUM_TOL[torch.float32])
        return call, None, ("token", tokens), verify
    return setup


def _build_attn_sw(heads, seq, dim, window):
    def setup(device, target):
        q = torch.randn(heads, seq, dim, device=device, dtype=torch.float32)
        k = torch.randn(heads, seq, dim, device=device, dtype=torch.float32)
        v = torch.randn(heads, seq, dim, device=device, dtype=torch.float32)
        call = lambda: attn_sw_kernel.forward(q, k, v, window, None,          # noqa: E731
                                              torch.float32, target)

        def verify():
            ref = attn_sw_kernel.forward(q, k, v, window, None, torch.float32, "cpu")
            return _verify_vs(call(), ref, _NUM_TOL[torch.float32])
        pairs = _window_pairs(seq, window)
        return call, 4.0 * heads * pairs * dim, ("token", seq), verify   # QK^T 与 PV 两段乘加
    return setup


def _build_conv(chans, length):
    def setup(device, target):
        x = torch.randn(chans, length, device=device, dtype=torch.float32)
        w = torch.randn(chans, gdn_kernel.CONV_KERNEL, device=device, dtype=torch.float32)
        b = torch.randn(chans, device=device, dtype=torch.float32)
        call = lambda: gdn_kernel.conv_forward(x, w, b, torch.float32, target)  # noqa: E731

        def verify():
            ref = gdn_kernel.conv_forward(x, w, b, torch.float32, "cpu")
            return _verify_vs(call(), ref, _NUM_TOL[torch.float32])
        return call, 2.0 * chans * length * gdn_kernel.CONV_KERNEL, ("token", length), verify
    return setup


def _build_delta_fwd(heads, seq, dk, dv):
    def setup(device, target):
        q, k, gate = (_rand3(heads, seq, dk, device), _rand3(heads, seq, dk, device),
                      _gate3(heads, seq, dk, device))   # 门取非正分布 ⇒ 递推稳定（见 _gate3）
        v = _rand3(heads, seq, dv, device)
        beta = torch.rand(heads, seq, device=device, dtype=torch.float32) / dk  # β~1/dk ⇒ 递推收缩
        call = lambda: gdn_kernel.forward(q, k, v, gate, beta, torch.float32,  # noqa: E731
                                          target)

        def verify():
            ref = gdn_kernel.forward(q, k, v, gate, beta, torch.float32, "cpu")
            return _verify_vs(call(), ref, _NUM_TOL[torch.float32])
        return call, 4.0 * heads * seq * dv * dk, ("token", seq), verify
    return setup


def _build_delta_bwd(heads, seq, dk, dv):
    def setup(device, target):
        q, k, gate = (_rand3(heads, seq, dk, device), _rand3(heads, seq, dk, device),
                      _gate3(heads, seq, dk, device))   # 门取非正分布 ⇒ 递推稳定（见 _gate3）
        v = _rand3(heads, seq, dv, device)
        beta = torch.rand(heads, seq, device=device, dtype=torch.float32) / dk  # β~1/dk ⇒ 递推收缩
        dout = _rand3(heads, seq, dv, device)
        call = lambda: gdn_kernel.backward(q, k, v, gate, beta, dout, target)   # noqa: E731

        def verify():
            # 反向交回梯度元组（dq,dk,dv,dg,dbeta）⇒ 逐项对拍，全过才算过
            ref = gdn_kernel.backward(q, k, v, gate, beta, dout, "cpu")
            got = call()
            if len(got) != len(ref):
                return False, f"arity-got={len(got)} ref={len(ref)}"
            worst_ok, worst = True, "rel=-"
            for i, (gi, ri) in enumerate(zip(got, ref)):
                ok, d = _verify_vs(gi, ri, _NUM_TOL[torch.float32])
                worst = f"[{i}]{d}"
                if not ok:
                    return False, worst
            return worst_ok, worst
        return call, 4.0 * heads * seq * dv * dk, ("token", seq), verify
    return setup


#: 装配器名 → 工厂（VEC_MATRIX 只存名字，避免模块级前向引用）
FACTORIES = {"add_ln": _build_add_ln, "rope": _build_rope, "attn_sw": _build_attn_sw,
             "conv": _build_conv, "delta_fwd": _build_delta_fwd,
             "delta_bwd": _build_delta_bwd}


def _rand3(h, s, d, device):
    """(heads, seq, width) 的 fp32 输入。递推门用 randn 即可——本件只量时间不判数值。"""
    return torch.randn(h, s, d, device=device, dtype=torch.float32)


def _gate3(h, s, d, device):
    """递推门（gdn 的 `g`）的**数值安全**输入：g ≤ 0 ⇒ `exp(g) ∈ (0,1]`，状态不溢出。

    动机（P1-1t 收尾）：gdn 递推是 `S *= exp(g[t])`；若 g 取 `randn`（`_rand3`），`exp(randn)`
    在 seq=512 步后必爆成 `inf`→`nan`，此时**连 torch 参考自身都是 nan**，数值自证退化成
    `rel=nan`（正确内核也被误判 NOTIMED）。故递推门单独给非正分布，让"数值自证"对 gdn 腿可用。
    只改门的分布、不改形状/工作量 ⇒ 计时口径不变（这些内核控制流与数据无关）。
    """
    return -torch.rand(h, s, d, device=device, dtype=torch.float32)


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

        def verify():
            # 存档形态 fp32 声明、出口写回 args 里的 C/GW ⇒ 直接用 torch 矩阵式对拍（独立尺子）。
            # 语义（读 S_perf_track_cube 正文）：gemm C = A @ Wᵀ + b；dW GW = dYᵀ @ A。
            call()                       # 就地写回 C / GW
            if kind == "gemm":
                ref = A @ W.t() + b
                return _verify_vs(C, ref, _NUM_TOL[torch.float32])
            ref = dY.t() @ A
            return _verify_vs(GW, ref, _NUM_TOL[torch.float32])
        return call, 2.0 * m * n * k, ("token", m), verify
    return setup


# ────────────────────────────────────────────────────────── 数值自证（计时硬前置）
# 判据（P1-1t 收尾）：计时前必须先用 torch 参考对本腿输出对拍，不过阈值 ⇒ 拒绝计时、打 NOTIMED。
# 参考来源两条，都是**独立于被测内核**的尺子：
#   * 入口件腿：同一入口件以 `target="cpu"` 再算一遍（`*_asc` 的 `_eager` 尺子路，fp32 累加后降位宽）；
#   * intrinsic 对照腿：存档形态本就 fp32 声明，直接用 torch 矩阵式对拍。
# 相对 L2 误差按出口位宽给容差（fp32 紧、低精度松）；死代码/空转件输出是未初始化内存，
# rel ~ O(1)，任何容差都拦得住。
_NUM_TOL = {torch.float32: 1e-2, torch.float16: 3e-2, torch.bfloat16: 3e-2}


def _rel_l2(got, ref):
    """相对 L2 误差（fp64 归约）；形状不符 ⇒ inf。"""
    import numpy as np
    a = np.asarray(got, dtype=np.float64).ravel()
    b = np.asarray(ref, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return float("inf")
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


@torch.no_grad()
def _verify_vs(got, ref, tol):
    """对拍核函数输出 `got` 与 torch 参考 `ref`。返回 `(ok, detail)`；异常一律判不过。"""
    try:
        if isinstance(got, (tuple, list)):        # add_ln 交回 (y, h)
            got = got[0]
        if isinstance(ref, (tuple, list)):
            ref = ref[0]
        rel = _rel_l2(got.detach().float().cpu().numpy(), ref.detach().float().cpu().numpy())
    except Exception as exc:                                  # noqa: BLE001
        return False, f"verify-raised:{type(exc).__name__}"
    return (rel <= tol), f"rel={rel:.3e} tol={tol:g}"


def _verify_vs_pair(got, ref, tol):
    """`add_ln` 的交货是 (y, h) 两个张量 ⇒ 逐张对拍，两张都过才算过。"""
    try:
        gy, gh = got
        ry, rh = ref
    except Exception as exc:                                  # noqa: BLE001
        return False, f"verify-raised:{type(exc).__name__}"
    ok_y, d_y = _verify_vs(gy, ry, tol)
    ok_h, d_h = _verify_vs(gh, rh, tol)
    return (ok_y and ok_h), f"y:{d_y} h:{d_h}"


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
    if v is None:
        return "na"                                   # 缺腿（None）⇒ na，别拿 None 去格式化
    return "na" if (v != v) else f"{v:.3f}"          # NaN ⇒ na（不给假数）


def bench_line(op, dtype, shape, st, flops, iters, warmup, device, target, key, reuse, note):
    """契约行：`BENCH-<op> dtype= shape= p50_ms= mean_ms= p95_ms= iters= warmup= tflops=
    mem_alloc_mb= mem_peak_mb= verified=`，后随本件自证的附加字段 track/kernel/reuse/note。

    `verified=true` 恒成立：本行只在数值对拍（`run_one` 硬前置）通过后才可能被打出。
    """
    tf = "na"
    if flops is not None and st["p50"] == st["p50"] and st["p50"] > 0:
        # TFLOPS 口径 = p50（中位数）。**单位是 TFLOP/s = /1e12**（P1-1t 收尾修正：原写 /1e9
        # 是 GFLOP/s 却标 tflops= ⇒ 1000× 虚高；第二窗"3365.919 tflops"一半是这处错标造成）。
        tf = f"{flops / (st['p50'] / 1e3) / 1e12:.6f}"
    alloc, peak = _mem_mb(device)
    parts = [f"BENCH-{op}", f"dtype={dtype}", f"shape={shape}",
             f"p50_ms={_fmt(st['p50'])}", f"mean_ms={_fmt(st['mean'])}",
             f"p95_ms={_fmt(st['p95'])}", f"iters={iters}", f"warmup={warmup}",
             f"tflops={tf}", f"mem_alloc_mb={alloc}", f"mem_peak_mb={peak}",
             f"verified=true",
             f"track={device}/{target}", f"kernel={key}", f"reuse={reuse}"]
    if note:
        parts.append(f"note={note}")
    return " ".join(parts)


def notimed_line(op, dtype, shape, reason, device, target):
    """未过数值自证 ⇒ **拒绝出数**：只报 NOTIMED + 原因，绝不产 p50/tflops/BENCH-QUOTE。

    这是 P1-1t 收尾的硬闸落点（动机：第二窗死代码腿"跑"出 3365.919 假 tflops 进表）。
    """
    return (f"BENCH-{op} NOTIMED dtype={dtype} shape={shape} reason={reason} "
            f"track={device}/{target}")


def quote_line(op, dtype, shape, p50, unit_name, work_per_call, flops):
    """把 p50 换成"每小时可完成工作量"——D12 分段记账要的物理量（本件不乘价格）。"""
    if p50 is None or not (p50 > 0) or p50 != p50:
        return None
    calls_h = 3600.0 / (p50 / 1e3)
    g = "" if flops is None else f" gflop_per_h={flops * calls_h / 1e9:.1f}"
    return (f"BENCH-QUOTE op={op} dtype={dtype} shape={shape} unit={unit_name} "
            f"work_per_call={work_per_call} calls_per_h={calls_h:.1f} "
            f"unit_per_h={calls_h * work_per_call:.1f}{g} basis=p50 track=npu")


def ablate_line(base, dtype, shape, ms_scalar, ms_intr, intr_dtype, verified):
    """标量路 vs intrinsic 路并排（G9 三栈消融行）。任一腿没成 ⇒ ratio=na，绝不硬凑。

    `verified` 为真**仅当两腿都过了数值自证**（⇒ 都真计了时）；否则 ratio=na 且 verified=false
    ——把"数字不可用"写在行里，杜绝死代码腿的假比值（第二窗 3690.3x）再被引用。
    """
    both = bool(ms_scalar and ms_intr and ms_intr > 0)
    r = f"{ms_scalar / ms_intr:.1f}x" if both else "na"
    dt = "n/a" if intr_dtype is None else f"{dtype}/{intr_dtype}"
    return (f"BENCH-ABLATE op={base} dtype={dt} shape={shape} "
            f"scalar_p50_ms={_fmt(ms_scalar)} intrinsic_p50_ms={_fmt(ms_intr)} ratio={r} "
            f"verified={'true' if verified else 'false'} track=npu")


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
        call, _flops, _unit, _verify = setup(device, target)
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
class _NotTimed(RuntimeError):
    """未过数值自证 ⇒ **拒绝计时**（不是失败、也不是成功）：`reason` 原样进 NOTIMED 行。"""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def run_one(cfg, device, target, iters, warmup, sync):
    """跑一个配置：装配 → **数值自证** → 夹计 → 守卫 → 返回 (BENCH 行, p50, flops, unit, work)。

    **数值自证是计时硬前置**（P1-1t 收尾，本波正身）：`verify()` 缺失或不过 ⇒ 抛 `_NotTimed`，
    由调用方打成 `BENCH-<op> NOTIMED`，**绝不产 p50/tflops/BENCH-QUOTE**。
    动机：第二窗死代码腿（`__mix__` 落死）"跑"出 0.080ms/3365.919 假 tflops 进表——
    有数值前置就拦得住（死代码输出是未初始化内存，rel~O(1)）。
    其余异常一律上抛给调用方隔离（`BENCH-<op> FAIL`），本函数自己不吞异常。
    """
    op, dtype, shape, _tier, substr, setup = cfg
    before = set(ascend_env.compiled_keys())      # 必须在 setup/verify 之前（保 reuse=new 语义）
    cnt0 = ascend_env.compile_count()
    _reset_peak(device)
    call, flops, (unit_name, work_per_call), verify = setup(device, target)
    if verify is None:                            # 没带对拍的腿 = 没有自证 ⇒ 拒绝计时
        raise _NotTimed("no-numeric-selfproof")
    ok_v, detail = verify()
    if not ok_v:                                  # 对拍挂了的腿 ⇒ 拒绝计时（数字不可信）
        raise _NotTimed(f"numeric-mismatch({detail})")
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


def _notimed_selftest(device, _target):
    """`--selftest-notimed`：合成一条**无对拍**腿（`verify=None`）⇒ 硬闸必须打 NOTIMED。

    负例自证：没有数值自证的腿**不得**产 p50/tflops/QUOTE（这正是第二窗假数的入口）。
    """
    x = torch.ones(8, 8, device=device, dtype=torch.float32)
    return (lambda: x * 2.0), None, ("token", 8), None


def _mismatch_selftest(device, _target):
    """`--selftest-mismatch`：合成一条**对拍必挂**腿（交了 x*2 却拿 x*3 当参考）⇒ 必须 NOTIMED。"""
    x = torch.ones(8, 8, device=device, dtype=torch.float32)
    call = lambda: x * 2.0  # noqa: E731

    def verify():
        return _verify_vs(call(), x * 3.0, _NUM_TOL[torch.float32])

    return call, None, ("token", 8), verify


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
    ap.add_argument("--selftest-notimed", action="store_true",
                    help="注入一条**无对拍**腿（verify=None）⇒ 硬闸必须打 NOTIMED（负例自证）")
    ap.add_argument("--selftest-mismatch", action="store_true",
                    help="注入一条**对拍必挂**腿 ⇒ 硬闸必须打 NOTIMED（负例自证）")
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
    if args.selftest_notimed:
        cfgs.insert(min(3, len(cfgs)),
                    ("selftest_notimed", "n/a", "n/a", 0, "", _notimed_selftest))
    if args.selftest_mismatch:
        cfgs.insert(min(3, len(cfgs)),
                    ("selftest_mismatch", "n/a", "n/a", 0, "", _mismatch_selftest))

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
    ok = fail = skipped = notimed = fail_ref = fail_inj = 0
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
        except _NotTimed as exc:   # 未过数值自证 ⇒ 拒绝出数（既非成功亦非失败，绝不产 p50/QUOTE）
            print(notimed_line(op, dtype, shape, exc.reason, device, target), flush=True)
            notimed += 1
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
    if device == "npu" and (intr_ms or scalar_ms):
        for (base, shape, dtype), ms in scalar_ms.items():
            if base not in ("linear", "dW"):
                continue
            got = intr_ms.get((base, shape))          # 只在 intrinsic 腿**过了数值自证**时才有
            i_ms, i_dt = got if got else (None, None)
            print(ablate_line(base, dtype, shape, ms, i_ms, i_dt,
                              verified=got is not None), flush=True)

    print(f"BENCH-SUMMARY ops={len(cfgs)} ok={ok} fail={fail} notimed={notimed} wall_s={wall:.1f} "
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
          "tflops_unit='TFLOP/s（=flops/(p50/1e3)/1e12；P1-1t 收尾从 /1e9 修正，原为 1000× 虚高）' "
          "sync='torch.npu.synchronize 每次前后夹计' warmup='预热不计入' "
          "unit_per_h='每小时可完成工作量（报价直接抄这一列）'")
    print("BENCH-NOTE ablation='标量路(op) vs intrinsic 路(op_intrinsic，K 波 Cube/l0tr 存档件，"
          "kernels/reconcile/S_perf_track_cube.py，非产品件)' ⇒ G9 三栈消融数据；"
          "对照腿已知 507015/V 系未证真 ⇒ 其 FAIL 计入 fail_ref 不计入 fail_core")
    print("BENCH-NOTE numeric-gate=计时硬前置：每腿先过数值对拍（P1-1t 收尾）——未带对拍 ⇒ "
          "`BENCH-<op> NOTIMED reason=no-numeric-selfproof`，对拍挂 ⇒ "
          "`reason=numeric-mismatch(rel=...)`，**两条都不产 p50/tflops/BENCH-QUOTE**；"
          "带数字的 BENCH 行尾 `verified=true`，ABLATE 行 `verified=<true|false>`（两腿都过自证才 true）")
    print("BENCH-NOTE numeric-gate-selfproof='--selftest-notimed'（无对拍腿）与 "
          "'--selftest-mismatch'（对拍必挂腿）应各打一条 NOTIMED——负例自证硬闸真生效")
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
