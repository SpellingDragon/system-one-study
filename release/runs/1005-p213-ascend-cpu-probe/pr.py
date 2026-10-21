import torch
from ascend.kernels import gemm_asc as G
def go(m):
    A, W, b = torch.rand(m, 32), torch.rand(16, 32), torch.rand(16)
    C = torch.zeros(m, 16)
    s = G.plan(A, W, b, C, "relu", "cpu")
    G.run(A, W, b, C, s)
    r = (A @ W.T + b).clamp_min(0)
    d = (C - r).abs()
    print("m", m, "err", round(float(d.max()), 5), "badrows",
