#!/usr/bin/env python3
"""attempts/E/_dbg_e.py — golden 自检指标伪影的分离取证件（保留不删，防误删）。

背景：check_attn 首跑 naive-rel=8.17e-05 超 2e-5 预算。本件证明这是**度量伪影**
（近零输出分量的相对放大），不是 expf 软件件或语义误差：
  1) sim32(复刻 expf) 与 true-exp fp32 两路对 f64 真值的误差**逐位同级**
     （abs 同为 2.757e-07）→ expf 贡献 ≤ sw-vs-true 差 (1.19e-07, 1ulp@~4)；
  2) 分母垫 1%|ref|max 地板后 rel 回到 3.6e-06 量级。
跑法（容器 e，纯 CPU）：python3 /tmp/_dbg_e.py
"""
import sys

import numpy as np

sys.path.insert(0, "/tmp")  # 容器内 e_golden.py 落位
import e_golden as G  # noqa: E402

NEG = G.NEG
rng = np.random.default_rng(7)
q = rng.standard_normal((4, 24, 16)).astype(np.float32)
k = rng.standard_normal((4, 24, 16)).astype(np.float32)
v = rng.standard_normal((4, 24, 16)).astype(np.float32)
SCALE = 16 ** -0.5
ref = G.attn_ref_f64(q, k, v, 8, SCALE)


def attn_row_true(q, k, v, hh, i, window, scale):
    """与 G.attn_row 同构，仅 expf_sw 换成真 exp（fp64→fp32）。"""
    dim = q.shape[-1]
    lo = max(0, i - window + 1)
    n = i - lo + 1
    sc = [np.float32(scale) * np.float32(np.dot(q[hh, i], k[hh, lo + j]))
          if j < n else NEG for j in range(window)]
    m = max(sc)
    ps = [np.float32(np.exp(np.float64(np.float32(x - m)))) for x in sc]
    l = np.float32(0.0)
    for p in ps:
        l = np.float32(l + p)
    return np.array([np.float32(np.float32(sum(np.float32(ps[j] * v[hh, lo + j, c])
                                               for j in range(window))) / l)
                     for c in range(dim)], np.float32)


sim = np.zeros((4, 24, 16), np.float32)
sim2 = np.zeros_like(sim)
for hh in range(4):
    for i in range(24):
        o, _, _ = G.attn_row(q, k, v, hh, i, 8, SCALE)
        sim[hh, i] = o
        sim2[hh, i] = attn_row_true(q, k, v, hh, i, 8, SCALE)
floor = np.maximum(np.abs(ref), 1e-2 * np.abs(ref).max())
e1 = np.abs(sim.astype(np.float64) - ref)
e2 = np.abs(sim2.astype(np.float64) - ref)
print("swexp :  abs=%.3e relFloor=%.3e relNaive=%.3e"
      % (e1.max(), (e1 / floor).max(), (e1 / np.maximum(np.abs(ref), 1e-12)).max()))
print("trueexp: abs=%.3e relFloor=%.3e" % (e2.max(), (e2 / floor).max()))
print("sw-vs-true abs=%.3e  (expf 软件件在整条 attn 通路上的净贡献)"
      % np.abs(sim.astype(np.float64) - sim2.astype(np.float64)).max())
print("E-DBG-SEPARATION-OK" if (e1.max() == e2.max() or e1.max() < 5e-7) else "E-DBG-CHECK-FAIL")
