import sys
sys.path.insert(0, "/tmp/sys1/release")
import torch
from ascend import train_step as TS
from ascend.kernels import ascend_env

ops = TS.ops
SEQ = [0]
FIRSTBAD = [None]

def stats(t):
    if not torch.is_tensor(t) or t.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        return None
    tf = t.detach().float()
    if tf.numel() == 0:
        return (0, 0, 0.0, str(t.dtype))
    return (int(torch.isnan(tf).sum()), int(torch.isinf(tf).sum()), float(tf.abs().max()), str(t.dtype))

def report(kind, vals):
    nan = sum(v[0] for v in vals); inf = sum(v[1] for v in vals)
    mx = max([v[2] for v in vals] or [0.0])
    dts = ",".join(sorted(set(v[3] for v in vals)))
    return "%s nan=%d inf=%d absmax=%.5g dt=%s" % (kind, nan, inf, mx, dts)

def wrap(name):
    orig = getattr(ops, name)
    def f(*a, **k):
        SEQ[0] += 1
        i = SEQ[0]
        ins = [s for s in (stats(x) for x in a) if s]
        out = orig(*a, **k)
        outs = list(out) if isinstance(out, (tuple, list)) else [out]
        os_ = [s for s in (stats(x) for x in outs) if s]
        bad = (sum(v[0] + v[1] for v in ins) > 0) or (sum(v[0] + v[1] for v in outs if True) > 0) if False else ((sum(v[0] + v[1] for v in ins) > 0) or (sum(v[0] + v[1] for v in os_) > 0))
        flag = "FIRST_BAD" if bad and FIRSTBAD[0] is None else ("BAD" if bad else "ok")
        if bad and FIRSTBAD[0] is None:
            FIRSTBAD[0] = name
        print("H-OP %03d %s in[%s] out[%s] => %s" % (i, name, report("in", ins) if ins else "none", report("out", os_) if os_ else "none", flag), flush=True)
        return out
    f.__name__ = name
    setattr(ops, name, f)

names = [n for n in dir(ops) if not n.startswith("_") and callable(getattr(ops, n))]
for n in names:
    wrap(n)
print("H-CASE2 wrapped ops: %s" % ",".join(names), flush=True)
try:
    rc = TS.main(["--target", "ascend", "--dtype", "fp16"])
    print("H-CASE2-RC=%s first_bad_op=%s ops_called=%d" % (rc, FIRSTBAD[0], SEQ[0]), flush=True)
except SystemExit as e:
    print("H-CASE2-SYSTEMEXIT=%s first_bad_op=%s ops_called=%d" % (e.code, FIRSTBAD[0], SEQ[0]), flush=True)
except Exception as e:
    print("H-CASE2-EXC %s: %s first_bad_op=%s" % (type(e).__name__, str(e)[:300], FIRSTBAD[0]), flush=True)
try:
    v = TS.reduction_violations()
    print("H-CASE2 fp16_acc_violations=%d sample=%s" % (len(v), str(v[:3])[:300]), flush=True)
except Exception as e:
    print("H-CASE2 guard read EXC %s" % e, flush=True)
