"""GDN 与 LoRA 探针：GDN 前向对 fp32 递推尺子（反向标 partial），LoRA 三条链对自动微分尺子。"""
import torch

from ascend.kernels import ascend_env, gdn_asc, gdn_kernel, lora_asc

torch.manual_seed(7)
HEADS, SEQ, DK, DV = 2, 24, 8, 8
q = torch.nn.functional.normalize(torch.randn(HEADS, SEQ, DK), p=2, dim=-1)
k = torch.nn.functional.normalize(torch.randn(HEADS, SEQ, DK), p=2, dim=-1)
v = torch.randn(HEADS, SEQ, DV)
g = -torch.rand(HEADS, SEQ, DK) * 0.2          # 对数衰减：负值 = 每步淡掉一点
beta = torch.rand(HEADS, SEQ) * 0.9 + 0.05

out = gdn_asc.forward(q, k, v, g, beta, torch.float32, "cpu")
ref = gdn_asc._eager(q, k, v, g, beta)
print("gdn_fwd_err", float((out - ref).abs().max()))

o4 = gdn_asc.forward(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), g.unsqueeze(0),
                     beta.unsqueeze(0), torch.float32, "cpu")
print("gdn_batch_err", float((o4[0] - ref).abs().max()), "shape", tuple(o4.shape))

print("gdn_bwd_status", gdn_kernel.BWD_STATUS)
dy = torch.randn(HEADS, SEQ, DV)
gr = gdn_asc.backward(q, k, v, g, beta, dy, "cpu")
leaf = [t.clone().requires_grad_(True) for t in (q, k, v, g, beta)]
o_ref = gdn_asc._eager(*leaf)
ga = torch.autograd.grad(o_ref, leaf, dy)
print("gdn_bwd_errs", [round(float((a - b).abs().max()), 6) for a, b in zip(gr, ga)])

M, N, KK, R, S = 12, 10, 16, 4, 0.5
x = torch.randn(M, KK)
a = torch.randn(R, KK) * 0.2
b = torch.randn(N, R) * 0.2
base = torch.randn(M, N)
delta = lora_asc.apply(x, a, b, S, None, torch.float32, "cpu")
print("lora_fwd_err", float((delta - lora_asc._eager(x, a, b, S)).abs().max()))
wbase = lora_asc.apply(x, a, b, S, base, torch.float32, "cpu")
print("lora_base_err", float((wbase - (base + lora_asc._eager(x, a, b, S))).abs().max()))

dyL = torch.randn(M, N)
dx, da, db = lora_asc.backward(dyL, x, a, b, S, "cpu")
rx, ra, rb = lora_asc._eager_grad(dyL, x, a, b, S)
print("lora_bwd_errs", round(float((dx - rx).abs().max()), 7),
      round(float((da - ra).abs().max()), 7), round(float((db - rb).abs().max()), 7))

dW = lora_asc.merge(a, b, S, torch.float32, "cpu")
print("lora_merge_err", float((dW - (b @ a * S)).abs().max()))
print("apply_vs_merge", float((lora_asc.apply(x, a, b, S, None, torch.float32, "cpu")
                               - (x @ dW.T)).abs().max()))
print("nc", ascend_env.compile_count(), "blk", ascend_env.blockers(),
      "keys", sorted(ascend_env.compiled_keys()))
