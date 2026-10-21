"""attn_sw 探针：CPU 内核对 fp32 全量 softmax 尺子，并与 P1 torch_ref 交叉对拍。"""
import torch

from ascend.kernels import attn_sw_asc, attn_sw_kernel, ascend_env
from sys1.testing.torch_ref import attn_sw_ref

torch.manual_seed(5)
HEADS, SEQ, DIM, WINDOW = 2, 33, 16, 8
q = torch.randn(HEADS, SEQ, DIM)
k = torch.randn(HEADS, SEQ, DIM)
v = torch.randn(HEADS, SEQ, DIM)
scale = DIM ** -0.5

out = attn_sw_asc.forward(q, k, v, WINDOW, scale, torch.float32, "cpu")
ref = attn_sw_asc._eager(q, k, v, WINDOW, scale)
print("kernel_err", float((out - ref).abs().max()))

p1 = attn_sw_ref(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), WINDOW, scale, torch.float32)[0]
print("torch_ref_err", float((out - p1).abs().max()))

w = attn_sw_kernel.forward(q, k, v, WINDOW)
print("entry_fp16", w.dtype, float((w.float() - ref).abs().max()))

wt = attn_sw_asc.forward_weights(q, k, WINDOW)
allow = attn_sw_asc._mask(torch.zeros(HEADS, SEQ, SEQ), SEQ, WINDOW) >= 0
print("zero_weight_max", float(wt.masked_fill(~allow, 0.0).abs().max()))
print("rowsum_err", float((wt.sum(-1) - 1.0).abs().max()))

q2, k2, v2 = torch.randn(3, 16, 16), torch.randn(3, 16, 16), torch.randn(3, 16, 16)
o2 = attn_sw_asc.forward(q2, k2, v2, 64, None, torch.float32, "cpu")
r2 = attn_sw_asc._eager(q2, k2, v2, 64, 16 ** -0.5)
print("fullcausal_err", float((o2 - r2).abs().max()))
print("nc", ascend_env.compile_count(), "blk", ascend_env.blockers(),
      "keys", sorted(ascend_env.compiled_keys()))
