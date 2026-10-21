from pathlib import Path

p = Path("ascend/kernels/lora_asc.py")
src = p.read_text()

pairs = [
    ('''from typing import Any

import torch''', 'import torch'),
    # 收紧契约：与 P1 的 linear 入口同口径，只吃 (m,k) 的二维批；更高维由上游自己 reshape
    ('''    """契约：x 是 (m,k) 或 (...,k)、A(r,k)、B(n,r)，三者的 k/r 轴要对齐，base 与输出同形。"""
    if x.dim() < 2 or a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"需 x 至少二维、A/B 二维，实得 {x.dim()}/{a.dim()}/{b.dim()} 维")''',
     '''    """契约：x 是 (m,k)、A(r,k)、B(n,r)，三者的 k/r 轴要对齐，base 与增量同形。

    与 P1 linear 入口同口径：本件只吃二维批，(B,T,K) 这类高维由上游先 reshape 成 (B*T,K)。
    """
    if x.dim() != 2 or a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"需 x/A/B 都是二维，实得 {x.dim()}/{a.dim()}/{b.dim()} 维")'''),
    ('''    if base is not None and base.shape[-1] != b.size(0):
        raise ValueError(f"base 的末维需等于 B 的行数 {b.size(0)}，实得 {base.shape[-1]}")''',
     '''    if base is not None and base.shape != (x.size(0), b.size(0)):
        raise ValueError(f"base 需与增量同形 {(x.size(0), b.size(0))}，实得 {tuple(base.shape)}")'''),
    # 反向的形状校验同步收紧
    ('''    _check(x, a, b, None)
    if dy.shape[-1] != b.size(0):
        raise ValueError(f"dy 的末维需等于 B 的行数 {b.size(0)}，实得 {dy.shape[-1]}")''',
     '''    _check(x, a, b, None)
    if dy.shape != (x.size(0), b.size(0)):
        raise ValueError(f"dy 需与增量同形 {(x.size(0), b.size(0))}，实得 {tuple(dy.shape)}")'''),
    # 自动微分尺子不需要再 reshape（已是二维）
    ('''    delta = _eager(xl, al, bl, scaling)
    flat = delta.reshape(-1, delta.size(-1))
    g = torch.autograd.grad(flat, leaves, dy.reshape(-1, delta.size(-1)).float())
    return g[0].reshape(x.shape), g[1], g[2]''',
     '''    delta = _eager(xl, al, bl, scaling)
    g = torch.autograd.grad(delta, leaves, dy.float())
    return g[0], g[1], g[2]'''),
    ('''def _eager_grad(dy: torch.Tensor, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
               scaling: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:''',
     '''def _eager_grad(dy: torch.Tensor, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                scaling: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:'''),
]
for old, new in pairs:
    assert src.count(old) == 1, (src.count(old), old[:60])
    src = src.replace(old, new, 1)
p.write_text(src)
print("patched", len(pairs))
