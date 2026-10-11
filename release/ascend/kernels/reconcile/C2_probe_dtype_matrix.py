#!/usr/bin/env python3
"""reconcile/C2_probe_dtype_matrix.py — P1-1m 产品位宽矩阵：8 行 × {fp32, fp16} 真编真跑＋bf16 实测。

【为什么 host 能吃 fp16】cpu-alias 轨（配方 `run_numerics._prod_setup`）：`tilelang.ascend.language`
这个名字**两处一起**（sys.modules + 父包属性）指向 cpu 方言，并把 `ascend_env.compile_kwargs` 翻成
c 后端三件套 ⇒ 递进去的 target 仍是 "ascend"，路由/缓存/键名/正文全是产品件的，host 无卡也能真编
真跑。本工装另加两处**垫片**（只补 cpu 方言缺失的“载体层”，不碰算式、不碰被测件）：
  · `T.Vector()` → nullcontext（910B 的 AIV 作用域标记，c 后端没有这个名字）
  · `T.alloc_shared` → `T.alloc_local`（c 后端不允多个动态 shared 分配；口径同 `gemm_asc._alloc_tile`
    既有分叉：“分存储层归属、不分算式”）
可行性凭据见同目录 C2_probe_alias_shim.py（add_ln/attn_sw/rope/gdn 全部 run=True 出键）。

【假绿守卫（load-bearing）】入口件在 plan 拒形状/编译失败时会**静默落 torch eager 并照样交出看起来
对的数**，所以每格在判数之前先要 `compiled_keys()` 里真有 `ascend|<件>[ascend]`；没有键＝内核没跑＝
不算 PASS（详情带 blocker 原文首行，或“plan 返回 None”）。接口张量是**真位宽**（`q()` 交 dt  dtype，
不是“fp32 装 fp16 的值”——那正是本探针首版踩过的假绿坑，改掉了）。

【对拍口径】参考＝“入参按实际位宽量化 → 全程 fp32 算”（生产口径：主干 fp16 / 归约 fp32 / 出口不降位）。
比较时把参考**降到被测输出的实际 dtype** 再算 rel：出口 fp32 的件（dW、gdn 五梯、lora 梯度）不引入
量化噪声，出口 dt 的件（linear/rope/readout/attn_sw 交货）允许 1 ulp。fp32 档阈值沿用各件既有判据
（verify_*.py / run_numerics.py 原文），fp16 档取 2 ulp(fp16)≈1e-3。为防“放宽阈值把静默错数据放过来”，
另加两条判别：错朝向/漏块（WRONG_DIR_MIN=1e-2）、降位累加（linear，把参考换成“每步降回 fp16 的累加”）。

bf16 不作支持宣称：预期 UNSUPPORTED；报错原文与行号进 C2_dtype_matrix.md 的缺件清单，卡面 bisheng
证据由 C2_probe_bf16_scalar.py 另取。

一行判决：`C2-<row>-<dtype>-PASS rel=<v> tol=<t> key=<k>`
用法（缓存纪律：每次判决换新 TILELANG_CACHE_DIR）：
  TILELANG_CACHE_DIR=$(mktemp -d) .venv/bin/python ascend/kernels/reconcile/C2_probe_dtype_matrix.py
  可选 --dtypes fp32,fp16,bf16  --rows linear,dw,readout,add_ln,gdn_fwd,gdn_bwd,rope,attn_sw,lora  --neg
注意：**不能**加 `from __future__ import annotations`（PEP 563 会让闭包形状常量失去闭包单元）。
"""
import argparse
import contextlib
import os
import sys
import types

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

import numpy as np
import torch
import tilelang
import tilelang.cpu.language as tcpu

from ascend.kernels import (add_ln_asc, ascend_env, attn_sw_asc, gemm_bwd_dw_asc, gemm_asc,
                            gemm_kernel, gemm_bwd_dw_kernel, gdn_asc, letter_readout_asc,
                            lora_asc, rope_asc, rope_kernel)

DT = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
#: fp32 档＝各件既有判据原文；fp16 档＝2 ulp(fp16)
TOL = {
    "linear": {"fp32": 2e-5, "fp16": 1e-3},       # S_cpu_oracle REL_TOL=2e-5
    "dw": {"fp32": 2e-5, "fp16": 1e-3},
    "readout": {"fp32": 0.0, "fp16": 0.0},         # 纯搬运 ⇒ 逐位同
    "add_ln": {"fp32": 3e-4, "fp16": 3e-4},        # r_h<1e-5 / r_y<3e-4（run_numerics:282-283）
    "gdn_fwd": {"fp32": 1e-4, "fp16": 1e-4},       # verify_gdn rtol=1e-4
    "gdn_bwd": {"fp32": 1e-4, "fp16": 1e-4},
    "rope": {"fp32": 1e-5, "fp16": 1e-3},          # verify_rope:214 max_abs<1e-5
    "attn_sw": {"fp32": 1e-4, "fp16": 1e-4},       # verify_attn_sw:223 max_abs<1e-4
    "lora": {"fp32": 2e-5, "fp16": 1e-3},
}
#: bf16 档只为让每格跑到守卫/编译面，取 fp16 同阈值（预期判决都是 UNSUPPORTED，不作支持宣称）
for _v in TOL.values():
    _v["bf16"] = _v["fp16"]
WRONG_DIR_MIN = 1e-2       # 错朝向/漏块须差出这个量级，否则“对上数”可能只是巧合
# 三例：尾块 m=33、尾块 relu m=8、方阵 n==k=32（只有方阵谈得上“翻/不翻”朝向判别）
LIN = ((33, 64, 32, "none"), (8, 64, 32, "relu"), (16, 32, 32, "none"))

_ROWS, _LOG = [], []


def log(msg):
    _LOG.append(msg)
    print(msg, flush=True)


# ── 工装：cpu-alias 轨＋两处垫片，每格从零 ────────────────────────────────────
def alias():
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
    if not hasattr(tcpu, "_c2_alloc_shared"):
        tcpu._c2_alloc_shared = tcpu.alloc_shared
        tcpu.alloc_shared = lambda shape, dt, *a, **k: tcpu.alloc_local(shape, dt)
    ascend_env.compile_kwargs = lambda t: {"target": "c", "target_host": "c",
                                          "execution_backend": "cython"}
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda *a, **k: ascend_env.TILELANG
    ascend_env.reset()


def route(sub, want_n=1):
    """假绿守卫：compiled_keys 里必须真有 `ascend|...sub...`；把键交出去当凭据。"""
    keys = sorted(k for k in ascend_env.compiled_keys() if k.startswith("ascend|") and sub in k)
    if len(keys) < want_n:
        blk = list(ascend_env.blockers().values())
        raise AssertionError(f"NOKEY:{sub} 需≥{want_n} 个 ascend 键，实得 {keys}；"
                             f"blocker={str(blk[0])[:170] if blk else 'plan 返回 None（无键也无 blocker）'}")
    return ";".join(keys[:want_n]) if want_n > 1 else keys[-1]


# ── 数值工具 ─────────────────────────────────────────────────────────────────
def q(t, dt):
    """交**真位宽**的接口张量（值也按该位宽量化）。"""
    return t.to(dt)


def _cmp(got, ref):
    """把参考降到被测输出的实际位宽（出口不降位的件＝fp32，不引入量化噪声）。"""
    return (got.detach().float().reshape(-1),
            ref.detach().to(got.dtype).float().reshape(-1))


def rel(got, ref):
    g, r = _cmp(got, ref)
    return ((g - r).abs().max().item() / max(r.abs().max().item(), 1e-12))


def mabs(got, ref):
    g, r = _cmp(got, ref)
    return (g - r).abs().max().item()


def acc16_ref(x, w, relu=False):
    """“每步降回 fp16 累加”的错误口径参考（linear 的归约位宽判别用）。"""
    xh, wh = x.numpy().astype(np.float16), w.numpy().astype(np.float16)
    out = np.zeros((xh.shape[0], wh.shape[0]), dtype=np.float16)
    for kk in range(xh.shape[1]):
        out = (out + (xh[:, kk:kk + 1] * wh[:, kk][None, :])).astype(np.float16)
    if relu:
        out = np.maximum(out, np.float16(0.0))
    return torch.from_numpy(out.astype(np.float32))


# ── 9 格本体：每格返回 (worst_rel, tol, 判别明细, 键) ─────────────────────────
def row_linear(dtn):
    dt, tol = DT[dtn], TOL["linear"][dtn]
    worst, notes, key = 0.0, [], None
    for m, n, k, act in LIN:
        torch.manual_seed(0)
        A, W = q(torch.randn(m, k), dt), q(torch.randn(n, k), dt)
        b = torch.randn(n)                                   # 偏置契约恒 fp32（入口 .to(float32)）
        got = gemm_kernel.forward(A, W, b, act, dt, "ascend")
        key = key or route("gemm[ascend]")                   # 先守卫再判数（防 eager 假绿）
        pre = A.float() @ W.float().T + b
        ref = pre.clamp_min(0.0) if act == "relu" else pre
        r = rel(got, ref)
        assert r < tol, f"m{m}n{n}k{k}/{act} rel={r:.3e} ≥ {tol:g}"
        line = f"m{m}n{n}k{k}/{act} rel={r:.2e} max_abs={mabs(got, ref):.2e}"
        if n == k:                                           # 方阵才谈得上“翻/不翻”
            unb = A.float() @ W.float() + b
            wrong = rel(got, unb.clamp_min(0.0) if act == "relu" else unb)
            assert wrong > WRONG_DIR_MIN, f"m{m}n{n}k{k} 朝向判别失效 rel={wrong:.3e}"
            line += f" 不翻朝向={wrong:.2e}"
        if dtn == "fp16":
            d16 = rel(got, acc16_ref(A, W, act == "relu") + b)
            assert r <= d16, f"降位累加判别失效 r={r:.2e} r16acc={d16:.2e}"
            line += f" fp16累加口径={d16:.2e}"
        worst = max(worst, r)
        notes.append(line)
    return worst, tol, " | ".join(notes), key


def row_dw(dtn):
    dt, tol = DT[dtn], TOL["dw"][dtn]
    torch.manual_seed(1)
    m, n, k = 33, 64, 32                                     # 尾块 m=33 激励 tc=16 的三块累加
    dY, A = q(torch.randn(m, n), dt), q(torch.randn(m, k), dt)
    got = gemm_bwd_dw_kernel.backward(dY, A, out_dtype=dt, target="ascend")
    key = route("gemm_dw[ascend]")                           # 先守卫：fp16 编不成＝零内核覆盖
    ref = dY.float().T @ A.float()
    r = rel(got, ref)
    assert r < tol, f"rel={r:.3e} ≥ {tol:g}"
    wrong = rel(got, dY[:16].float().T @ A[:16].float())     # “漏掉后两块 token”的错误口径
    assert wrong > WRONG_DIR_MIN, f"token 漏块判别失效 rel={wrong:.3e}"
    return r, tol, f"m{m}n{n}k{k} rel={r:.2e} max_abs={mabs(got, ref):.2e} 漏块判别={wrong:.2e} " \
        f"内核出口 GW=fp32(不降位)→交货 {dtn}", key


def row_readout(dtn):
    dt = DT[dtn]
    torch.manual_seed(2)
    rows = q(torch.randn(16, 32), dt)
    ids = torch.randint(0, 16, (8,), dtype=torch.int32)
    got = letter_readout_asc.forward(rows, ids, dt, "ascend")
    key = route("gather[ascend]")
    ref = rows[ids.long()]                                   # 纯搬运：期望逐位同
    r, exact = rel(got, ref), torch.equal(got, ref)
    assert r == 0.0 and exact, f"rel={r:.3e} bitexact={exact}（搬运件不许有任何数值差）"
    dy = q(torch.randn(8, 32), dt)
    g = letter_readout_asc.backward(16, ids, dy, "ascend")
    acc = torch.zeros(16, 32, dtype=torch.float32)
    acc.index_add_(0, ids.long(), dy.float())
    rb = rel(g, acc)
    assert rb < 1e-6, f"scatter_add rel={rb:.3e}（fp32 累加顺序差允许 <1e-6）"
    return r, TOL["readout"][dtn], f"(8,32)←(16,32) bitexact={exact}；backward(scatter_add) rel={rb:.1e} " \
        f"重复行号累加口径同 index_add_", key


def row_add_ln(dtn):
    dt = DT[dtn]
    th, ty = 1e-5, TOL["add_ln"][dtn]
    torch.manual_seed(3)
    x, res = q(torch.randn(33, 64), dt), q(torch.randn(33, 64) + 4.0, dt)   # 大共模/小波动：贴生产残差流量级
    w, b = torch.randn(64), torch.randn(64)
    y, h = add_ln_asc.forward(x, res, w, b, 1e-5, dt, "ascend")
    key_f = route("add_ln[ascend]")
    v = x.float() + res.float()
    dev = v - v.mean(1, keepdim=True)
    rs_b = torch.rsqrt((dev * dev).mean(1, keepdim=True) + 1e-5)            # 有偏（kernel 口径）
    rs_u = torch.rsqrt((dev * dev).sum(1, keepdim=True) / 63 + 1e-5)        # 无偏（错误口径）
    r_h, r_y = rel(h, v), rel(y, dev * rs_b * w + b)
    r_unb = rel(y, dev * rs_u * w + b)
    assert r_h < th, f"h rel={r_h:.3e}（fp32 残差路应逐位同；~2e-4 即降了位宽）"
    assert r_y < ty, f"Y rel={r_y:.3e}（仿真底噪 1.5e-5；Y 全零则 rel≈1）"
    assert r_y < r_unb - 2e-4, f"有偏/无偏判别失败 r_y={r_y:.3e} r_unb={r_unb:.3e}"
    dy = q(torch.randn(33, 64), dt)
    dx, dg, db = add_ln_asc.backward(h, w, dy, 1e-5, "ascend")
    tdx, tdg, tdb = add_ln_asc._ln_bwd_torch(h, w, dy.float(), 1e-5)        # 尺子取 torch 闭式路（另一实现）
    r_b = max(rel(dx, tdx), rel(dg, tdg), rel(db, tdb))
    assert r_b < 1e-4, f"ln_bwd rel={r_b:.3e}"
    key_b = route("ln_bwd[ascend]")
    return max(r_h, r_y), ty, (f"r_h={r_h:.2e}(<1e-5) r_y={r_y:.2e}(<3e-4) r_unb={r_unb:.2e} 有偏判别过 "
                               f"| ln_bwd rel={r_b:.2e}"), f"{key_f} & {key_b}"


def _gdn_data(dt, seed=4):
    torch.manual_seed(seed)
    h_, s_, dk_, dv_ = 2, 8, 16, 16
    return (q(torch.randn(h_, s_, dk_) * .3, dt), q(torch.randn(h_, s_, dk_) * .3, dt),
            q(torch.randn(h_, s_, dv_), dt), q(-torch.rand(h_, s_, dk_), dt),
            q(torch.rand(h_, s_) * .5, dt))


def row_gdn_fwd(dtn):
    dt, tol = DT[dtn], TOL["gdn_fwd"][dtn]
    qq, kk, vv, gg, bb = _gdn_data(dt)
    got = gdn_asc.forward(qq, kk, vv, gg, bb, dt, "ascend")
    key = route("gdn[ascend]")
    st = torch.zeros(2, 16, 16, dtype=torch.float32)
    ref = torch.empty(2, 8, 16, dtype=torch.float32)
    for i in range(8):                                       # verify_gdn:212 的独立尺子（bmm 形）
        st = st * gg[:, i].float().exp().unsqueeze(-1)
        pred = torch.bmm(st.transpose(1, 2), kk[:, i].float().unsqueeze(-1)).squeeze(-1)
        upd = (vv[:, i].float() - pred) * bb[:, i].float().unsqueeze(-1)
        st = st + torch.bmm(kk[:, i].float().unsqueeze(-1), upd.unsqueeze(1))
        ref[:, i] = torch.bmm(st.transpose(1, 2), qq[:, i].float().unsqueeze(-1)).squeeze(-1)
    r = rel(got, ref)
    assert r < tol, f"rel={r:.3e} ≥ {tol:g}"
    assert float(ref.abs().max()) > 0, "参考全零 ⇒ 空测"
    return r, tol, f"h2s8k16v16 rel={r:.2e} max_abs={mabs(got, ref):.2e}", key


def row_gdn_bwd(dtn):
    dt, tol = DT[dtn], TOL["gdn_bwd"][dtn]
    qq, kk, vv, gg, bb = _gdn_data(dt)
    torch.manual_seed(9)
    dy = q(torch.randn(2, 8, 16), dt)
    lv = [t.detach().clone().float().requires_grad_(True) for t in (qq, kk, vv, gg, bb)]
    st = torch.zeros(2, 16, 16, dtype=torch.float32)
    o = []
    for i in range(8):                                       # 同一条递推的 autograd 图＝独立尺子
        s2 = st * lv[3][:, i].exp().unsqueeze(-1)
        pred = torch.einsum("hcd,hc->hd", s2, lv[1][:, i])
        upd = (lv[2][:, i] - pred) * lv[4][:, i].unsqueeze(-1)
        st = s2 + lv[1][:, i].unsqueeze(-1) * upd.unsqueeze(1)
        o.append(torch.einsum("hcd,hc->hd", st, lv[0][:, i]))
    ref5 = torch.autograd.grad(torch.stack(o, 1), lv, dy.float(), retain_graph=False)
    got5 = gdn_asc.backward(qq, kk, vv, gg, bb, dy, "ascend")
    key = route("gdnb[ascend]")
    worst, parts = 0.0, []
    for nm, a, b in zip(gdn_asc.BWD_GRADES, got5, ref5):
        assert a.dtype == torch.float32, f"{nm} 出口降位了：{a.dtype}（纪律＝归约 fp32、出口不降位）"
        r = rel(a, b)                                        # 五梯出口恒 fp32（不降位）⇒ 参考不量化
        worst = max(worst, r)
        parts.append(f"{nm}={r:.1e}")
        assert r < tol, f"{nm} rel={r:.3e} ≥ {tol:g}"
    return worst, tol, f" ".join(parts) + f" 出口 dtype=fp32（接口 {dt} 入 / fp32 梯度出，已断言）", key


def row_rope(dtn):
    dt, tol = DT[dtn], TOL["rope"][dtn]
    torch.manual_seed(5)
    tokens, heads, dim = 33, 2, 16
    half = dim // 2
    qkv = q(torch.randn(tokens, 3, heads, dim), dt)
    # 生产形状：train_step._build_params 的 cos/sin 出自 torch.cos(angle) ⇒ **fp32**，主干 qkv 是 fp16
    cos, sin = torch.cos(torch.rand(tokens, half) * 3), torch.sin(torch.rand(tokens, half) * 3)
    keep2 = qkv[:, 2].clone()
    got = rope_kernel.forward(qkv.clone(), cos, sin, rope_kernel.DEFAULT_SLOTS, "ascend")
    key_f = route("rope[ascend]")

    def rot(src, sign):                                      # 独立尺子：fp32 累加、split-half
        x1 = src[:, :2, :, :half].float().clone()
        x2 = src[:, :2, :, half:].float().clone()
        c = cos.float().view(tokens, 1, 1, half)
        s = (sign * sin).float().view(tokens, 1, 1, half)
        o = src.float().clone()
        o[:, :2, :, :half] = x1 * c - x2 * s
        o[:, :2, :, half:] = x2 * c + x1 * s
        return o

    ref = rot(qkv, 1)
    r, mad = rel(got, ref), mabs(got, ref)
    slot2 = torch.equal(got[:, 2], keep2)
    assert r < tol, f"rel={r:.3e} max_abs={mad:.3e} ≥ {tol:g}"
    assert slot2, "槽位 2（值）被改动 ⇒ 就地语义越界"
    gb = rope_kernel.backward(got.clone(), cos, sin, rope_kernel.DEFAULT_SLOTS, "ascend")
    rb = rel(gb, rot(got, -1))
    assert rb < tol, f"反向 rel={rb:.3e} ≥ {tol:g}（角取负回转，尺子独立于正文）"
    nmod = len([x for x in ascend_env.compiled_keys() if "rope[ascend]" in x])
    bwd_note = "内核" if nmod > 1 else "落 eager（rope.backward 无 ascend 分支＝既有路由缺口，非本格 dtype 面）"
    return r, tol, (f"tok{tokens}h{heads}d{dim} rel={r:.2e} max_abs={mad:.2e} 槽位2逐位不动={slot2} "
                    f"| 反向 rel={rb:.2e} {bwd_note}"), key_f


def row_attn_sw(dtn):
    dt, tol = DT[dtn], TOL["attn_sw"][dtn]
    torch.manual_seed(6)
    heads, seq, dim, win = 2, 16, 32, 8
    qq, kk, vv = (q(torch.randn(heads, seq, dim), dt) for _ in range(3))
    got = attn_sw_asc.forward(qq, kk, vv, win, None, dt, "ascend")
    key = route("attn_sw[ascend]")
    sc = qq.float() @ kk.float().transpose(-1, -2) * (dim ** -0.5)
    idx = torch.arange(seq)
    delta = idx[:, None] - idx[None, :]
    allow = (delta >= 0) & (delta < win)
    w = torch.softmax(sc.masked_fill(~allow, float("-inf")), -1)
    ref = w @ vv.float()
    r = rel(got, ref)
    assert r < tol, f"rel={r:.3e} max_abs={mabs(got, ref):.3e} ≥ {tol:g}"
    assert float(got[:, 0].abs().max()) > 0, "首位置输出全零 ⇒ 正文没真跑"
    assert float(w[:, 0, 1:].abs().sum()) == 0.0, "位置 0 的未来权重非零 ⇒ 因果判别失效"
    return r, tol, (f"h{heads}s{seq}d{dim}w{win} rel={r:.2e} max_abs={mabs(got, ref):.2e} "
                    f"窗外严格不参与(参考侧未来权重=0)"), key


def row_lora(dtn):
    dt, tol = DT[dtn], TOL["lora"][dtn]
    torch.manual_seed(7)
    m, k, n, rk = 16, 32, 64, 16
    x, a = q(torch.randn(m, k), dt), q(torch.randn(rk, k), dt)
    b = q(torch.randn(n, rk), dt)
    got = lora_asc.apply(x, a, b, 0.5, None, dt, "ascend")
    key = route("gemm[ascend]", want_n=2)                   # 两次 _lin ⇒ 两副模具
    ref = (x.float() @ a.float().T) @ (b.float() * 0.5).T
    r = rel(got, ref)
    assert r < tol, f"apply rel={r:.3e} ≥ {tol:g}"
    torch.manual_seed(8)
    dy = q(torch.randn(m, n), dt)
    dx, da, db = lora_asc.backward(dy, x, a, b, 0.5, "ascend")
    u = x.float() @ a.float().T
    du = dy.float() @ (b.float() * 0.5)
    # 口径照 lora_asc.backward：dx=dU@A、dA=dUᵀ@x、dB=(dyᵀ@U)·s（s 在件外再乘一次）
    r_dx, r_da = rel(dx, du @ a.float()), rel(da, du.T @ x.float())
    r_db = rel(db, (dy.float().T @ u) * 0.5)
    rw = max(r_dx, r_da, r_db)
    assert rw < tol, f"backward rel={rw:.3e} ≥ {tol:g}"
    return r, tol, (f"apply rel={r:.2e} max_abs={mabs(got, ref):.2e} | backward dx={r_dx:.1e} "
                    f"da={r_da:.1e} db={r_db:.1e}（梯度出口 fp32）"), key


CELLS = [("linear", row_linear), ("dw", row_dw), ("readout", row_readout), ("add_ln", row_add_ln),
         ("gdn_fwd", row_gdn_fwd), ("gdn_bwd", row_gdn_bwd), ("rope", row_rope),
         ("attn_sw", row_attn_sw), ("lora", row_lora)]


def run_cell(name, fn, dtn):
    alias()
    try:
        relv, tol, detail, key = fn(dtn)
        st = "PASS"
    except AssertionError as exc:
        msg = str(exc)
        st = "UNSUPPORTED" if (dtn == "bf16" and "NOKEY" in msg) else "FAIL"
        relv, tol, key, detail = float("nan"), "-", "-", msg[:300]
    except Exception as exc:  # noqa: BLE001
        st = "UNSUPPORTED" if dtn == "bf16" else "FAIL"
        relv, tol, key = float("nan"), "-", "-"
        detail = f"{type(exc).__name__}: {str(exc).splitlines()[0][:260]}"
    log(f"C2-{name}-{dtn}-{st} rel={relv:.3e} tol={tol} key={key} | {detail}")
    _ROWS.append(dict(row=name, dtype=dtn, status=st, rel=relv, tol=tol, key=key, note=detail))


# ── 位宽门负例（“静默错数据比崩更危险”的显式拒防线）────────────────────────
def run_neg():
    log("### 负例/门判别（plan 期显式拒＋缓存键位宽隔离）")
    alias()
    m, n, k = 16, 64, 32
    rows = []
    for dtn in ("fp32", "fp16", "bf16"):
        dt = DT[dtn]
        A, W, b = q(torch.randn(m, k), dt), q(torch.randn(n, k), dt), torch.randn(n)
        C = torch.empty(m, n, dtype=dt)
        sp = gemm_asc.plan(A, W, b, C, "none", "ascend")
        rows.append((f"gemm.plan[{dtn}]", sp, "idt"))
        dY, aa = q(torch.randn(m, n), dt), q(torch.randn(m, k), dt)
        GW = torch.empty(n, k, dtype=torch.float32)
        rows.append((f"dw.plan[{dtn}]", gemm_bwd_dw_asc.plan(dY, aa, GW, "ascend"), "idt"))
        for adtn, adt in (("tab_fp32", torch.float32), (f"tab_{dtn}", dt)):
            if dtn == "fp32" and adtn == "tab_fp32":
                continue                                     # 与上一轮同行，去重免误读
            qkv, cs, sn = q(torch.randn(8, 3, 2, 16), dt), q(torch.randn(8, 8), adt), q(torch.randn(8, 8), adt)
            rows.append((f"rope.plan[{dtn}/{adtn}]", rope_asc._plan(qkv, cs, sn, 2, 1, "ascend"), "qdt"))
    bad = rope_asc._plan(q(torch.randn(8, 3, 2, 16), torch.float16), q(torch.randn(8, 8), torch.float16),
                         q(torch.randn(8, 8), torch.float32), 2, 1, "ascend")
    rows.append(("rope.plan[fp16/cos=fp16,sin=fp32 混宽]", bad, "qdt"))
    for nm, sp, tag in rows:
        kw = (sp or {}).get("kwargs", {})
        log(f"C2-NEG-{nm} accept={sp is not None} {tag}={kw.get(tag, '-')} key={(sp or {}).get('key', '-')}")
    alias()                                                # 缓存键位宽隔离：同形状 fp32/fp16 须两副模具
    torch.manual_seed(0)
    A32, W32, b32 = torch.randn(m, k), torch.randn(n, k), torch.randn(n)
    gemm_kernel.forward(A32, W32, b32, "none", torch.float32, "ascend")
    k1 = set(ascend_env.compiled_keys())
    g2 = gemm_kernel.forward(A32.half(), W32.half(), b32, "none", torch.float16, "ascend")
    newk = [x for x in sorted(set(ascend_env.compiled_keys()) - k1) if "gemm[ascend]" in x]
    log(f"C2-NEG-key隔离 新增模具={newk} fp16 rel="
        f"{rel(g2, A32.half().float() @ W32.half().float().T + b32):.2e}（参考含 bias）")

    assert len(newk) == 1 and newk[0] not in k1, "fp16 未产生独立模具键（位宽没进缓存键 ⇒ 陈旧产物复用）"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtypes", default="fp32,fp16")
    ap.add_argument("--rows", default=",".join(n for n, _ in CELLS))
    ap.add_argument("--neg", action="store_true", help="附带位宽门负例（改动后才有意义）")
    args = ap.parse_args()
    want = [r for r in CELLS if r[0] in args.rows.split(",")]
    for dtn in args.dtypes.split(","):
        log(f"### dtype={dtn}（cpu-alias 轨：target=ascend 的路由/键名/正文；方言层垫片见模块 docstring）")
        for name, fn in want:
            run_cell(name, fn, dtn)
    if args.neg:
        run_neg()
    by = {}
    for r in _ROWS:
        by.setdefault(r["dtype"], [0, 0])
        by[r["dtype"]][0] += 1
        by[r["dtype"]][1] += (r["status"] == "PASS")
    npass = sum(1 for r in _ROWS if r["status"] == "PASS")
    log("C2-SUMMARY " + " ".join(f"{d}:{p}/{t}" for d, (t, p) in sorted(by.items()))
        + f" 总计 {npass}/{len(_ROWS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
