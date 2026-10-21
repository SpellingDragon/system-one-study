"""rope 探针：内核路 vs 独立 torch 尺子（前向/反向/值槽不碰/恒等往返/回退路/入口层）。"""
import torch

from ascend.kernels import rope_asc, rope_kernel, ascend_env

torch.manual_seed(0)
T, H, D = 17, 3, 8
half = D // 2
theta = torch.randn(T, half)
cos, sin = torch.cos(theta), torch.sin(theta)


def want(q, s_sign):
    """独立参考：从原件算，绝不就地；s_sign=-1 即逆旋转。"""
    out = q.clone()
    c = cos.unsqueeze(1)
    s = (s_sign * sin).unsqueeze(1)
    for slot in (0, 1):
        x1 = q[:, slot, :, :half]
        x2 = q[:, slot, :, half:]
        out[:, slot, :, :half] = x1 * c - x2 * s
        out[:, slot, :, half:] = x2 * c + x1 * s
    return out


def fresh():
    return torch.randn(T, 3, H, D)


ref = fresh()
a = rope_asc.forward(ref.clone(), cos, sin, target="cpu")
b = want(ref, 1)
print("fwd_err", float((a - b).abs().max()))
print("vslot", float((a[:, 2] - b[:, 2]).abs().max()))

same = rope_asc.forward(ref, cos, sin, target="cpu")
print("inplace_same_obj", same is ref)

q0 = fresh()
r = rope_asc.backward(rope_asc.forward(q0, cos, sin, target="cpu"), cos, sin, target="cpu")
print("roundtrip", float((r - q0).abs().max()))

g = fresh()
a = rope_asc.backward(g.clone(), cos, sin, target="cpu")
print("bwd_err", float((a - want(g, -1)).abs().max()))

q1 = fresh()
base = q1.clone()
e = rope_asc._eager(q1, cos, sin, (0, 1), 1)
print("eager_err", float((e - want(base, 1)).abs().max()), "eager_same_obj", e is q1)

q2 = fresh()
base = q2.clone()
y = rope_kernel.forward(q2, cos, sin, target="cpu")
print("entry_err", float((y - want(base, 1)).abs().max()),
      "nc", ascend_env.compile_count(), "blk", ascend_env.blockers(),
      "keys", sorted(ascend_env.compiled_keys()))
