"""LoRA 注入件：训练期走"双 gemm 旁路"（不落地合并权重），推理期可一次性合并进底座。

【做什么】三个入口：① `apply(x, a, b, scaling, base, out_dtype, target)` 交回旁路增量
`delta = scaling · (x @ Aᵀ) @ Bᵀ`（可选再叠到 base 上）；② `backward(dy, x, a, b, scaling)` 交回
(dx, dA, dB)；③ `merge(a, b, scaling)` 交回 (N,K) 的等效权重增量，供推理期直接加到底座权重上。
【怎么做】本件**不新写任何乘加方言正文**，全部由已验证的 linear 两件组合出来：
   ① 前向两次 `lin(X, W) = X @ Wᵀ`：先把 x 降到 (m, r)（W=A），再把 (m, r) 升到 (m, n)（W=s·B）；
      尺度系数**折进 B**（B 是 (n,r) 的小矩阵，乘一遍只花 O(n·r)），这样第二次乘加不需要额外的
      alpha 参数，也不用改 gemm 件的判据；
   ② 反向三条链各自还是一次乘加：dU = dy @ (sB)（lin，W=(sB)ᵀ）、dx = dU @ A（lin，W=Aᵀ）、
      dA = dUᵀ @ x 与 dB = s · (dyᵀ @ U)（都走 dW 件 `gemm_bwd_dw`）；
   ③ 所有转置都只作用在 (r,·) 这类小矩阵上，由入口层用 torch 做一次连续化，代价相对双 gemm 可忽略。
【为什么】被否方案一：把 A、B 先合并成 (N,K) 再走一次 gemm——训练期每步都要重算 O(N·K·r) 的合并，
   等于把 LoRA 省下的算力又全花回去，还多一份要维护的临时权重；合并只留给推理期（`merge`）。
   被否方案二：给 gemm 件加 alpha 缩放参数——为一个旁路口径污染主件的形式签名（P1 的
   `gemm_kernel.forward` 没有这个参数），本域红线是"同名同签名"，故把尺度折进 B 而不是扩接口。
   组合件的正确性天然继承 linear 件的对拍结论，本件的测试只需证明"接线没接错"。
"""
import torch

from ascend.kernels import ascend_env, gemm_bwd_dw_kernel, gemm_kernel

#: 交付位宽默认值与其余入口件一致
DEFAULT_OUT_DTYPE = torch.float16


def _lin(x: torch.Tensor, w: torch.Tensor, target: str) -> torch.Tensor:
    """一次线性乘加：`x @ wᵀ`，全程复用 linear 入口件（fp32 累加，不截断）。

    白话：借用已经验过的那台乘加机，一次把一批行向量按着一排权重算出结果；本件不自己开模具。
    """
    return gemm_kernel.forward(x, w, None, "none", torch.float32, target)


def apply(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scaling: float = 1.0,
          base: torch.Tensor | None = None, out_dtype: torch.dtype = DEFAULT_OUT_DTYPE,
          target: str | None = None) -> torch.Tensor:
    """前向旁路：`delta = scaling · (x @ Aᵀ) @ Bᵀ`，给了 base 就交回 `base + delta`。

    白话：先把这批行降到低秩的小空间里走一遭，再升回来；升回来时顺手乘上这轮的下笔力度。
    底座结果先摆着，把这一小笔增量添上去就是交货内容——中间那份小矩阵从不落地成大权重。

    :raises ValueError: 形状不满足 x(m,k)/A(r,k)/B(n,r)，或 base 与增量不同形时抛出。
    """
    _check(x, a, b, base)
    name = ascend_env.normalize_target(target)
    if ascend_env.active_backend(name, x.device) == ascend_env.TILELANG:
        x32 = x.float().contiguous()
        a32 = a.float().contiguous()
        bs = (b.float() * float(scaling)).contiguous()
        u = _lin(x32, a32, name)
        delta = _lin(u, bs, name)
        out = delta if base is None else base.float() + delta
        return out.to(out_dtype)
    delta = _eager(x, a, b, scaling)
    out = delta if base is None else base.float() + delta
    return out.to(out_dtype)


def backward(dy: torch.Tensor, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
             scaling: float = 1.0,
             target: str | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """反向：交回 (dx, dA, dB)，三条链都复用 linear 两件（dx/dU 走前向件，dA/dB 走 dW 件）。

    白话：倒着走同一条旁路——先知道"升回来之后该挪多少"，除一下力度回到低秩空间，再分别问
    "这批行该挪多少""降的那排权重该挪多少""升的那排权重该挪多少"，每一问都还是一次乘加。

    :raises ValueError: 形状不满足契约时抛出。
    """
    _check(x, a, b, None)
    if dy.shape != (x.size(0), b.size(0)):
        raise ValueError(f"dy 需与增量同形 {(x.size(0), b.size(0))}，实得 {tuple(dy.shape)}")
    name = ascend_env.normalize_target(target)
    s = float(scaling)
    if ascend_env.active_backend(name, dy.device) == ascend_env.TILELANG:
        x32, a32 = x.float().contiguous(), a.float().contiguous()
        dy32 = dy.float().contiguous()
        bs = (b.float() * s).contiguous()
        u = _lin(x32, a32, name)                       # 前向留下的低秩中间量
        du = _lin(dy32, bs.t().contiguous(), name)     # dU = dy @ (sB)
        dx = _lin(du, a32.t().contiguous(), name)      # dx = dU @ A
        da = gemm_bwd_dw_kernel.backward(du, x32, out_dtype=torch.float32, target=name)
        db = gemm_bwd_dw_kernel.backward(dy32, u, out_dtype=torch.float32, target=name)
        return dx, da, db * s
    return _eager_grad(dy, x, a, b, s)


def merge(a: torch.Tensor, b: torch.Tensor, scaling: float = 1.0,
          out_dtype: torch.dtype = torch.float32,
          target: str | None = None) -> torch.Tensor:
    """推理期合并：交回 (N,K) 的等效权重增量 `scaling · B @ A`，可直接加到底座权重上。

    白话：训练时那两排小权重是分着摆的；上线不再需要旁路，就把它们合成一块和底座一样大的补丁，
    一次贴上去，之后走的就是普通线性层。

    :raises ValueError: A/B 的低秩轴对不上时抛出。
    """
    if a.dim() != 2 or b.dim() != 2 or a.size(0) != b.size(1):
        raise ValueError(f"需 A(r,k)、B(n,r) 且低秩轴一致，实得 {tuple(a.shape)} / {tuple(b.shape)}")
    name = ascend_env.normalize_target(target)
    if ascend_env.active_backend(name, a.device) == ascend_env.TILELANG:
        bs = (b.float() * float(scaling)).contiguous()
        return _lin(bs, a.t().contiguous(), name).to(out_dtype)  # (n,r) @ (r,k)
    return (b.float() @ a.float() * float(scaling)).to(out_dtype)


def _check(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
           base: torch.Tensor | None) -> None:
    """契约：x 是 (m,k)、A(r,k)、B(n,r)，三者的 k/r 轴要对齐，base 与增量同形。

    与 P1 linear 入口同口径：本件只吃二维批，(B,T,K) 这类高维由上游先 reshape 成 (B*T,K)。
    """
    if x.dim() != 2 or a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"需 x/A/B 都是二维，实得 {x.dim()}/{a.dim()}/{b.dim()} 维")
    if x.size(-1) != a.size(1):
        raise ValueError(f"x 的末维需等于 A 的列数 {a.size(1)}，实得 {x.size(-1)}")
    if b.size(1) != a.size(0):
        raise ValueError(f"B 的列数需等于 A 的行数（低秩轴）{a.size(0)}，实得 {b.size(1)}")
    if base is not None and base.shape != (x.size(0), b.size(0)):
        raise ValueError(f"base 需与增量同形 {(x.size(0), b.size(0))}，实得 {tuple(base.shape)}")


def _eager(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scaling: float) -> torch.Tensor:
    """回退路径：与旁路同口径的普通写法（fp32 累加），供对拍当尺子。

    白话：没有模具时按同一本账手工算：先降再升，最后乘上力度。
    """
    return (x.float() @ a.float().T) @ (b.float() * float(scaling)).T


def _eager_grad(dy: torch.Tensor, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                scaling: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """回退路径的反向：对同一条旁路做 torch 自动微分（口径与组合链完全一致）。"""
    leaves = [t.detach().clone().float().requires_grad_(True) for t in (x, a, b)]
    xl, al, bl = leaves
    delta = _eager(xl, al, bl, scaling)
    g = torch.autograd.grad(delta, leaves, dy.float())
    return g[0], g[1], g[2]


__all__ = ["apply", "ascend_env", "backward", "merge"]
