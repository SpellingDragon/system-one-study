"""add_ln 探针：前向两出口（y/h）与反向三项（dx/dg/db）各对 fp32 torch 尺子，并确认内核真跑。"""
import torch

from ascend.kernels import add_ln_asc, ascend_env

torch.manual_seed(1)
R, D = 33, 16
x = torch.randn(R, D)
res = torch.randn(R, D)
g = torch.randn(D)
b = torch.randn(D)
eps = add_ln_asc.DEFAULT_EPS

y, h = add_ln_asc.forward(x, res, g, b, eps, torch.float32, "cpu")
href = x + res
yref = add_ln_asc._ln(href, g, b, eps)
print("h_err", float((h - href).abs().max()), "y_err", float((y - yref).abs().max()))

dy = torch.randn(R, D)
dx, dg, db = add_ln_asc.backward(h, g, dy, eps, "cpu")
rx, rg, rb = add_ln_asc._ln_bwd_torch(href, g, dy, eps)
print("dx_err", float((dx - rx).abs().max()), "dg_err", float((dg - rg).abs().max()),
      "db_err", float((db - rb).abs().max()))

x8, r8 = x[:8].clone(), res[:8].clone()
y2, h2 = add_ln_asc.forward(x8, r8, g, b, eps, torch.float32, "cpu")
print("reuse_err", float((h2 - (x8 + r8)).abs().max()),
      "reuse_y_err", float((y2 - add_ln_asc._ln(x8 + r8, g, b, eps)).abs().max()))

dx2, dg2, db2 = add_ln_asc.backward(h2, g, dy[:8].clone(), eps, "cpu")
rx2, rg2, rb2 = add_ln_asc._ln_bwd_torch(x8 + r8, g, dy[:8].clone(), eps)
print("reuse_bwd_err", float((dx2 - rx2).abs().max()), float((dg2 - rg2).abs().max()),
      float((db2 - rb2).abs().max()))

fb = add_ln_asc.forward(x.double(), res.double(), g, b, eps, torch.float32, "cpu")
print("fallback_err", float((fb[1] - href).abs().max()))
print("nc", ascend_env.compile_count(), "blk", ascend_env.blockers(),
      "keys", sorted(ascend_env.compiled_keys()))
