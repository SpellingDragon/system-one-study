"""rope 参考实现：packed-qkv 就地 rotate-half，cos/sin 分表存 fp32、按 fp32 参与运算。

【做什么】给"按位置"的旋转记号做一份标准答案：同一句话里越靠后的词，它的查询向量和键向量
要被转过的角度越大，这样两个位置一减就能看出相隔多远。打包存放的三份数据里只转前两份
（查询与键），第三份（值）一个数都不许动。
【怎么做】① 角度只依赖位置 pos 与维度对下标 i：angle = pos / theta^(2i/D)，i ∈ [0, D/2)；
② 先把 cos/sin 各算成 (tokens, D/2) 的 fp32 两张表（**分表**，不拼在一起也不塞进角度本身），
   表在 fp32 里预计算，之后内核直接查表；③ 对每个 (token, 头) 把 D 维拆成前后两半
   x1 = 前半、x2 = 后半，做 rotate-half：前半 ← x1·cos − x2·sin，后半 ← x2·cos + x1·sin，
   全程 fp32 计算、算完再降回原来的半精度位宽就地写回。
【为什么】分表 + fp32 预计算是为精度：角度直接在内核里算要引入三角函数与幂运算，Metal 方言
的超越函数通路未验证（B1 探针结论），查表把它挪到 torch 侧最稳；被否方案一：把 cos/sin 拼成
一张 (tokens, D) 表复用后半——省一次索引，但会让"前半/后半共用同一角度"这条不变式变成隐式
约定，学生版补 _cuda 时极易读错步长；被否方案二：cos/sin 也存 fp16——省内存，但 D/2=32 时
单个角度的量化误差在长位置上可累积到 1e-3 级，直接吃掉对拍预算的一半。
"""
from __future__ import annotations

import torch

DEFAULT_THETA = 10000.0


def rope_angle_tables(
    seq_len: int,
    half: int,
    theta: float = DEFAULT_THETA,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """预计算 (seq_len, half) 的 cos/sin 两张 fp32 表：第 pos 行第 i 列 = cos/sin(pos·freq_i)。

    频率 freq_i = theta^(-2i/D)，D = 2·half。位置轴取 0..seq_len-1；内核与参考实现都只查这两
    张表，不再各自算三角函数——"角度只有一处真源"是对拍能收敛的前提。

    白话：先做两张小抄，一张记"每个位置每一对要转多少的横向分量"，另一张记纵向分量；
    位置越往后，抄上的数字转得越多，后面所有计算都只翻这两张抄，不再现场算。
    """
    if half <= 0:
        raise ValueError(f"half 必须为正（head_dim 需为偶数），实得 {half}")
    freq = torch.tensor([theta ** (-2.0 * i / (2.0 * half)) for i in range(half)], dtype=torch.float32, device=device)
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    angle = pos * freq.unsqueeze(0)
    return torch.cos(angle), torch.sin(angle)


def rotate_half_ref(x: torch.Tensor) -> torch.Tensor:
    """把最后一维对半切开并交换成 (-后半, 前半) 的形状，供旋转公式复用。

    白话：一行数字从中间剪开、两半换个位置放；旋转时它就是"另一半"的角色。
    """
    half = x.size(-1) // 2
    x1 = x[..., :half].to(torch.float32)
    x2 = x[..., half:].to(torch.float32)
    return torch.cat((-x2, x1), dim=-1)


def rope_ref(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotate_slots: tuple[int, ...] = (0, 1),
) -> torch.Tensor:
    """对打包的 qkv 做 rope，返回新副本（参考实现不改调用方的数据，方便对拍反复用同一输入）。

    形状：qkv 为 (tokens, 3, heads, dim)，slot 0=查询、1=键、2=值；cos/sin 为 (tokens, dim//2)。
    只有 rotate_slots 里的槽位会被转（默认只转查询与键），其余槽位必须逐位等于输入。

    白话：三摞材料摞在同一个盒子里，第一摞和第二摞按位置转一下，第三摞原封不动；转的时候
    每摞内部前半截和后半截互相搭一手——前半减去后半乘纵向抄，后半加上前半乘横向抄。
    """
    if qkv.dim() != 4 or qkv.size(1) != 3:
        raise ValueError(f"qkv 需为 (tokens, 3, heads, dim)，实得 {tuple(qkv.shape)}")
    tokens, _, _, dim = qkv.shape
    if dim % 2:
        raise ValueError(f"dim 必须为偶数才能对半旋转，实得 {dim}")
    if cos.shape != (tokens, dim // 2) or sin.shape != (tokens, dim // 2):
        raise ValueError(
            f"cos/sin 需各为 (tokens={tokens}, dim//2={dim // 2})，实得 {tuple(cos.shape)}/{tuple(sin.shape)}"
        )

    out = qkv.clone()
    # 角度轴对齐到"每一对"的最后一维，头轴留 1 让所有头共用同一角度 → (tokens, 1, half)
    c = cos.to(torch.float32).unsqueeze(1)
    s = sin.to(torch.float32).unsqueeze(1)
    half = dim // 2
    for slot in rotate_slots:
        # 先物化两半新值再写回：fp32 输入时 part 是 out 的视图（.to(fp32) 对已是 fp32 的张量
        # 返回自身），若边算边写，后半旋转会用到"刚被覆盖的前半"造成二次旋转。
        part = out[:, slot].to(torch.float32)
        x1 = part[..., :half]
        x2 = part[..., half:]
        new1 = (x1 * c - x2 * s).to(out.dtype)
        new2 = (x2 * c + x1 * s).to(out.dtype)
        out[:, slot, :, :half] = new1
        out[:, slot, :, half:] = new2
    return out
