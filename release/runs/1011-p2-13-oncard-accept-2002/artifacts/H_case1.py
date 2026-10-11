import sys, os
sys.path.insert(0, "/tmp/sys1/release")
sys.path.insert(0, "/tmp/sys1/release/ascend/port910b")
sys.argv = ["run_numerics.py", "--only", "linear_prod", "--track", "card"]
import numpy as np, torch
import run_numerics as R
from ascend.kernels import gemm_kernel, ascend_env

VARIANTS = [
    (16, 96, 48, "none", "CTRL same multi-block shape relu OFF"),
    (16, 32, 16, "relu", "CTRL single bn+bk block relu ON"),
    (16, 96, 32, "relu", "CTRL n multi-block k=32 relu ON"),
    (16, 64, 64, "relu", "CTRL n=64 k=64 relu ON"),
    (1, 64, 64, "relu", "CTRL LIN_SHAPES 5th case relu ON"),
    (16, 96, 48, "relu", "REPRO the failing sub-case"),
]
ONLY = os.environ.get("H_ONLY", "")
dev, restore, skip, box = R._prod_setup()
print("H-CASE1 dev=%s track=%s skip=%s REL_TOL=%g" % (dev, R._track(), skip, R.REL_TOL_LIN), flush=True)
exc = 0
try:
    torch.manual_seed(0)
    for m, n, k, act, note in VARIANTS:
        if ONLY and ONLY != "%dx%dxdx%s" % (m, n, k, act):
            continue
        A = torch.randn(m, k, device=dev, dtype=torch.float32)
        W = torch.randn(n, k, device=dev, dtype=torch.float32)
        b = torch.randn(n, device=dev, dtype=torch.float32)
        tag = "m%dn%dk%d/%s" % (m, n, k, act)
        try:
            got, _o = R._call(skip, lambda: gemm_kernel.forward(A, W, b, act, torch.float32, "ascend"))
            torch.npu.synchronize()
            pre = A @ W.T + b
            ref = pre.clamp_min(0.0) if act == "relu" else pre
            g = R._np(got); rr = R._np(ref); pp = R._np(pre)
            r = R.relerr(g, rr)
            d = np.abs(g - rr).ravel()
            wi = int(np.argmax(d)); pos = np.unravel_index(wi, g.shape)
            print("H-VAR %s rel=%.3e rel_vs_pre=%.3e nan=%d inf=%d neg_out=%d zero_out=%d zero_ref=%d worst_at=%s got=%.5f ref=%.5f compiled=%d | %s" % (
                tag, r, R.relerr(g, pp), int(np.isnan(g).sum()), int(np.isinf(g).sum()),
                int((g < 0).sum()), int((g == 0).sum()), int((rr == 0).sum()),
                str(tuple(int(x) for x in pos)), float(g[pos[0], pos[1]]), float(rr[pos[0], pos[1]]),
                ascend_env.compile_count(), note), flush=True)
        except Exception as e:
            exc += 1
            print("H-VAR %s EXC %s: %s | %s" % (tag, type(e).__name__, str(e)[:220], note), flush=True)
finally:
    restore()
print("H-CASE1-DONE exceptions=%d" % exc, flush=True)
