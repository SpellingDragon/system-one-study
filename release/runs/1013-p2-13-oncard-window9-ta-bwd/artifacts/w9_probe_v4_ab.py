"""w9 probe v4: config-plane A/B on the REAL gdn_bwd body (min shape 1,1,1,8,8 fp32).

v3 verdict: the real kernel dies at the SMALLEST shape — size/L/dtype are all
irrelevant; the fault is in the kernel's form. v1/v2 cleared every micro-form,
so the remaining untested plane is the JIT config: the six-entry out_idx and
the workspace_idx. Same body, three configs:

  run1: original          out_idx=[9..14] ws=[7,8]   (known FAIL)
  run2: workspace manual  out_idx=[9..14] ws=None     (host allocates SWS/LAM)
  run3: single output     out_idx=[9]     ws=[7,8]   (host passes the rest)

run2 PASS => workspace auto-alloc is the killer; run3 PASS => multi-output
argument marshalling is the killer; both FAIL => the fault is in the body and
we go statement-level delta next.
"""
import sys

sys.path.insert(0, "/tmp")
import torch
import torch_npu  # noqa: F401

src = open("/tmp/gdn_bwd.py").read()
assert "workspace_idx=[7, 8]," in src and "out_idx=[9, 10, 11, 12, 13, 14]," in src
open("/tmp/gdn_bwd_v4a.py", "w").write(src.replace("    workspace_idx=[7, 8],\n", ""))
open("/tmp/gdn_bwd_v4b.py", "w").write(src.replace("out_idx=[9, 10, 11, 12, 13, 14]", "out_idx=[9]"))

from gdn_bwd import gdn_bwd as g_orig  # noqa: E402
from gdn_bwd_v4a import gdn_bwd as g_nows  # noqa: E402
from gdn_bwd_v4b import gdn_bwd as g_oneout  # noqa: E402

B, H, L, DK, DV = 1, 1, 1, 8, 8


def tensors():
    q = torch.randn(B, H, L, DK).npu()
    k = torch.randn(B, H, L, DK).npu()
    v = torch.randn(B, H, L, DV).npu()
    g = -torch.rand(B, H, L).npu()
    beta = torch.rand(B, H, L).npu()
    h0 = torch.randn(B, H, DV, DK).npu()
    do = torch.randn(B, H, L, DV).npu()
    return q, k, v, g, beta, h0, do


def leg(name, fn):
    try:
        fn()
        print(f"V4 {name} PASS")
    except Exception as e:  # noqa: BLE001
        print(f"V4 {name} FAIL {type(e).__name__}: {str(e)[:100]}")


def run1():
    q, k, v, g, beta, h0, do = tensors()
    ker = g_orig(B, H, L, DK, DV, dtype="float32", accum_dtype="float")
    outs = ker(q, k, v, g, beta, h0, do)
    [o.cpu() for o in outs]


def run2():
    q, k, v, g, beta, h0, do = tensors()
    sws = torch.zeros(B, H, L, DV, DK).npu()
    lam = torch.zeros(B, H, L + 1, DV, DK).npu()
    ker = g_nows(B, H, L, DK, DV, dtype="float32", accum_dtype="float")
    outs = ker(q, k, v, g, beta, h0, do, sws, lam)
    [o.cpu() for o in outs]


def run3():
    q, k, v, g, beta, h0, do = tensors()
    dkg = torch.zeros(B, H, L, DK).npu()
    dvg = torch.zeros(B, H, L, DV).npu()
    dbeta = torch.zeros(B, H, L).npu()
    dg = torch.zeros(B, H, L).npu()
    dh0 = torch.zeros(B, H, DV, DK).npu()
    ker = g_oneout(B, H, L, DK, DV, dtype="float32", accum_dtype="float")
    outs = ker(q, k, v, g, beta, h0, do, dkg, dvg, dbeta, dg, dh0)
    [o.cpu() for o in outs]


leg("run1-orig-6out-ws", run1)
leg("run2-6out-nows", run2)
leg("run3-1out-ws", run3)
print("V4-DONE")
