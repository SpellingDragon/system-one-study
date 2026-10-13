"""w9 probe v3: shape-ladder on the REAL gdn_bwd kernel (no golden, exception-only).

v1/v2 verdict: all eight micro-kernel forms PASS on-card, so the 507015 fault
is either size/dimension-dependent or in a form only the real kernel combines.
This driver ramps the real kernel through a shape ladder; the first rung that
dies names the offending dimension.

Run: python3 w9_probe_v3_ladder.py   (on-card leg; /tmp/gdn_bwd.py required)
"""
import sys

sys.path.insert(0, "/tmp")
import torch
import torch_npu  # noqa: F401

from gdn_bwd import gdn_bwd  # noqa: E402

LADDER = [
    (1, 1, 1, 8, 8, "float32"),
    (1, 1, 4, 8, 8, "float32"),
    (1, 1, 64, 8, 8, "float32"),
    (1, 1, 64, 64, 64, "float32"),
    (1, 2, 256, 64, 64, "float32"),
    (1, 1, 64, 64, 64, "float16"),
    (1, 2, 256, 64, 64, "float16"),
]


def rung(B, H, L, DK, DV, dt):
    td = torch.float16 if dt == "float16" else torch.float32
    q = torch.randn(B, H, L, DK, dtype=td)
    k = torch.randn(B, H, L, DK, dtype=td)
    v = torch.randn(B, H, L, DV, dtype=td)
    g = -torch.rand(B, H, L, dtype=torch.float32)
    beta = torch.rand(B, H, L, dtype=torch.float32)
    h0 = torch.randn(B, H, DV, DK, dtype=td)
    do = torch.randn(B, H, L, DV, dtype=td)
    ker = gdn_bwd(B, H, L, DK, DV, dtype=dt, accum_dtype="float")
    outs = ker(q.npu(), k.npu(), v.npu(), g.npu(), beta.npu(), h0.npu(), do.npu())
    _ = [o.cpu() for o in outs]
    return True


def main():
    for B, H, L, DK, DV, dt in LADDER:
        tag = f"B={B} H={H} L={L} DK={DK} DV={DV} dt={dt}"
        try:
            rung(B, H, L, DK, DV, dt)
            print(f"LADDER PASS {tag}")
        except Exception as e:  # noqa: BLE001
            print(f"LADDER FAIL {tag} :: {type(e).__name__}: {str(e)[:100]}")
    print("LADDER-DONE")


if __name__ == "__main__":
    main()
