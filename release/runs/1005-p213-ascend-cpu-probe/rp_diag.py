"""诊断 rope CPU 内核：是否真的就地改动了 QKV，以及生成了什么源码。"""
import torch

import tilelang

from ascend.kernels import rope_asc, ascend_env

torch.manual_seed(0)
T, H, D = 17, 3, 8
half = D // 2
theta = torch.randn(T, half)
cos, sin = torch.cos(theta), torch.sin(theta)
q = torch.randn(T, 3, H, D)
before = q.clone()

spec = rope_asc._plan(q, cos, sin, 2, 1, "cpu")
print("spec", spec["key"], sorted(spec["kwargs"].items()))
ok = rope_asc._run(q, cos, sin, spec)
print("run_ok", ok, "changed", float((q - before).abs().max()))

k = ascend_env.get_compiled(
    spec["key"], lambda: None, "cpu")
src = k.get_kernel_source()
keep = [ln for ln in src.splitlines() if "QKV" in ln or "for " in ln or "if " in ln]
print("SRC_LINES", len(src.splitlines()))
for ln in keep[:24]:
    print("  |", ln.strip()[:110])
