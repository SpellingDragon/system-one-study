"""修 gdn_asc.backward：torch.autograd.grad 只"返回"梯度，不会写回 leaf.grad。

原写法 `tuple(t.grad.contiguous() ...)` 在 .grad 为 None 时直接崩
（AttributeError: 'NoneType' object has no attribute 'contiguous'），
这是 partial 回退路径的真实缺陷，必须用返回值本身。
"""
from pathlib import Path

p = Path("ascend/kernels/gdn_asc.py")
src = p.read_text()

old = '''    leaves = [t.detach().clone().float().requires_grad_(True) for t in (q, k, v, g, beta)]
    out = _eager(*leaves)
    torch.autograd.grad(out, leaves, dout.float(), retain_graph=False)
    return tuple(t.grad.contiguous() for t in leaves)'''
new = '''    leaves = [t.detach().clone().float().requires_grad_(True) for t in (q, k, v, g, beta)]
    out = _eager(*leaves)
    # 注意口径：torch.autograd.grad 只把梯度**返回**给调用方，不会写回 leaf.grad，
    # 所以这里必须用返回值，不能照 backward() 的习惯去读 t.grad（读到的永远是 None）。
    grads = torch.autograd.grad(out, leaves, dout.float(), retain_graph=False)
    return tuple(t.contiguous() for t in grads)'''

assert src.count(old) == 1, src.count(old)
p.write_text(src.replace(old, new, 1))
print("patched 1")
