"""layout — HF 权重布局 → 自研栈布局的纯换算（不碰磁盘、不碰模型实例）。

【做什么】
    给四个动作：挑出查询里的门控列、把三条投影拼成一摞打包 qkv、给出旋转配对的坐标置换、
    把 HF 的半截角表扩成整头宽度（没旋转的那些位配成恒等角）。每个动作都只认形状与顺序。

【怎么做】
    全部用"下标列子"表达：`rope_pair_permutation()` 算出 perm（长度 head_dim），规定
    "自研栈的第 c 个坐标 = HF 的第 perm[c] 个坐标"；重排行与重缩放都由这一个列子派生，
    因此逐元素对拍只需断言列子本身，不必有真权重。角表扩展走同一口径：前 rotary_dim/2 位
    用 HF 的真角，其余位补 cos=1/sin=0——恒等角配上"未旋转坐标排在后半"的置换，
    两者一道把内核的固定配对套到部分旋转的模型上。

【为什么】
    自研栈的 rope 内核只认"前后两半互相搭一手"这一种配对（`sys1/kernels/rope_kernel.py` 的
    契约），而 Qwen3.5 系每个头只转前 1/4 的宽度；两头都对不上时，改数据比改内核便宜——
    内核保持最简，方言件与参考实现才有机械对照的可能。
    被否方案一：给内核加"配对半径"参数（每种半径都要重开一份编译产物，缓存键爆炸）；
    被否方案二：只在 Python 里手写一个等价 rope 跑对拍（那验的是我的写法，不是内核的契约，
    p2-13 上真件时一切推倒）。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

#: 打包 qkv 的三个槽位号（0/1/2 = 查询/键/值）——与 sys1.model.CausalAttention 的视图口径一致
SLOT_Q, SLOT_K, SLOT_V = 0, 1, 2


@dataclass(frozen=True)
class HeadLayout:
    """一个注意力层的头布局：多少大头、几套小头复用、每头多宽、其中多宽参与旋转。"""

    heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int                     # 每个头里真正被旋转的宽度（head_dim × partial 比例）

    @property
    def group(self) -> int:
        """一个大头摊到几个小头上（分组注意力的复用倍数；MHA 时为 1）。

        白话：问的一方有八个头、答的一方只备了两套，那就让每套顶着四家用；这个"顶几回"
        的倍数从这里算。除不尽说明这副身板的头数配不成组，当场说清，不留含糊。"""
        if self.heads % self.kv_heads:
            raise ValueError(f"heads={self.heads} 不是 kv_heads={self.kv_heads} 的整数倍")
        return self.heads // self.kv_heads

    @property
    def half(self) -> int:
        """内核配对的另一半起点（head_dim 的一半）。

        白话：一个头里的坐标排成前后两排坐，前排第几位与后排同列那位结对子；这里交出
        的就是后排的起点，也就是头宽的一半。头宽是奇数就两两配不上对，直接说清。"""
        if self.head_dim % 2:
            raise ValueError(f"head_dim 需为偶数才能前后两半配对，实得 {self.head_dim}")
        return self.head_dim // 2

    @classmethod
    def from_config(cls, text_config, head_dim: int | None = None) -> "HeadLayout":
        """从 HF 的文本 config 造布局（头宽与部分旋转比例都按 config 的口径取）。

        白话：身板数据从出厂说明单上抄——有几个头、几个共用一套、每份多宽、其中多宽要转角，
        一张单子读完，后面所有换算都照这个来，不再各自猜。
        """
        heads = int(text_config.num_attention_heads)
        kv_heads = int(text_config.num_key_value_heads)
        hd = int(head_dim if head_dim is not None else getattr(text_config, "head_dim",
                                                                text_config.hidden_size // heads))
        rope_params = getattr(text_config, "rope_parameters", None) or {}
        partial = float(rope_params.get("partial_rotary_factor", 1.0))
        return cls(heads=heads, kv_heads=kv_heads, head_dim=hd, rotary_dim=int(hd * partial))


def rope_pair_permutation(layout: HeadLayout) -> list[int]:
    """给出"自研栈坐标 → HF 坐标"的置换列子（长度 = head_dim）。

    白话：每个头里的位置要重新排座次。内核只认"前半第 j 个和后半第 j 个结对子"这一种坐法，
    而模型出厂时是"前一小段里 i 与 i+半段 结对子、剩下的一长条根本不转"。于是把要转的对子
    挨个放进前半/后半的位置上，不转的两位两位往后面塞——排完座次，两种坐法算的是同一件事。

    :param layout: 头布局（用到 head_dim 与 rotary_dim）。
    :returns: 长度 head_dim 的置换；`perm[c]` = 自研栈第 c 位对应的 HF 位。
    :raises ValueError: rotary_dim 出格（超过 head_dim、不是偶数、或未转宽度为奇数）。
    """
    hd, rd = layout.head_dim, layout.rotary_dim
    if rd <= 0 or rd > hd or rd % 2:
        raise ValueError(f"rotary_dim 需为 (0, {hd}] 内的偶数，实得 {rd}")
    if (hd - rd) % 2:
        raise ValueError(f"未旋转宽度 {hd - rd} 必须是偶数，才能两位一组排进后半段")
    half, r_half = layout.half, rd // 2
    perm = [0] * hd
    for j in range(r_half):                    # 真转角的对子：HF 的 (j, j+r_half) → 我们的 (j, j+half)
        perm[j] = j
        perm[half + j] = r_half + j
    cursor = rd                                # 剩下的坐标按 (a, a+1) 相邻两位补到后半段空位上
    for j in range(r_half, half):
        perm[j] = cursor
        perm[half + j] = cursor + 1
        cursor += 2
    if sorted(perm) != list(range(hd)):        # 必须是双射：漏一位或重一位等于把权重读丢
        raise AssertionError(f"置换列子不是 0..{hd - 1} 的双射")
    return perm


def expand_rope_tables(cos: torch.Tensor, sin: torch.Tensor, layout: HeadLayout) -> tuple[torch.Tensor, torch.Tensor]:
    """把 HF 的角表扩成整头宽度：真角在前、恒等角（cos=1/sin=0）补到 head_dim/2。

    白话：模型只给了要转的那一小段的角度，而且给法是"同一串念两遍"；内核要的是"每个结对子
    的位置一份角度"。于是先取它念的第一遍，再把不转的那些位配成"乘一加零"——不转的位子
    配上这份角度，转了也等于没转。

    :param cos: `(..., rotary_dim)`，HF 交出的余弦表（前后两半重复）。
    :param sin: `(..., rotary_dim)`，HF 交出的正弦表。
    :param layout: 头布局。
    :returns: 两张 `(..., head_dim//2)` 的 fp32 表。
    """
    r_half = layout.rotary_dim // 2
    need = layout.half
    if cos.shape[-1] != sin.shape[-1]:
        raise ValueError(f"cos/sin 末维需相等，实得 {cos.shape[-1]}/{sin.shape[-1]}")
    if cos.shape[-1] < r_half:
        raise ValueError(f"角表末维 {cos.shape[-1]} 不足 rotary_dim/2={r_half}")
    c = cos[..., :r_half].to(torch.float32)
    s = sin[..., :r_half].to(torch.float32)
    pad = need - r_half
    if pad:
        ones = torch.ones(*c.shape[:-1], pad, dtype=torch.float32, device=c.device)
        zeros = torch.zeros(*s.shape[:-1], pad, dtype=torch.float32, device=s.device)
        c, s = torch.cat([c, ones], dim=-1), torch.cat([s, zeros], dim=-1)
    return c, s


def split_gated_q(q_weight: torch.Tensor, layout: HeadLayout) -> tuple[torch.Tensor, torch.Tensor]:
    """把带门的 q_proj 拆成"纯查询"与"出口门"两半（每头 head_dim×2 宽 → 各 head_dim）。

    白话：这一列投影一次交出两样东西：每个头先取一段当查询、后一段当阀门。这里按头把它们
    分回两摞，各摞还是原来的行主序，谁也没混进谁。

    :raises ValueError: 行数与 `heads × head_dim × 2` 不符（说明这层没开门或口径不同）。
    """
    want = layout.heads * layout.head_dim * 2
    if q_weight.shape[0] != want:
        raise ValueError(f"q_proj 行数应为 {want}（每头 head_dim×2），实得 {q_weight.shape[0]}")
    per_head = q_weight.view(layout.heads, 2, layout.head_dim, q_weight.shape[1])
    q = per_head[:, 0].reshape(-1, q_weight.shape[1]).contiguous()
    gate = per_head[:, 1].reshape(-1, q_weight.shape[1]).contiguous()
    return q, gate


def rotated_row_index(heads: int, layout: HeadLayout, source_heads: int) -> torch.Tensor:
    """造"打包行号 → 源投影行号"列子（换座次版）：先按头补齐小头，再在每头内部换座次。

    白话：模型只备了两套键值，外层要八个头一起用；补齐的规矩是相邻的几个大头共用同一套小头
    （连着分组的），所以点名是 0,0,0,0,1,1,1,1 这样的连号；点完头再在每头内部按内核的坐次
    重新排位。

    :param heads: 目标头数（打包后一律是 heads 个大头）。
    :param layout: 头布局。
    :param source_heads: 源投影的头数（键/值传 kv_heads；查询传 heads）。
    :returns: 长度 `heads × head_dim` 的 long 列子。
    """
    hd = layout.head_dim
    owner = torch.arange(heads) // max(1, heads // source_heads)
    if int(owner.max()) >= source_heads:
        raise ValueError(f"补齐倍数算歪了：heads={heads} source_heads={source_heads}")
    rows = owner[:, None] * hd + torch.arange(hd)[None, :]
    perm_t = torch.tensor(rope_pair_permutation(layout), dtype=torch.long)
    return rows.index_select(1, perm_t).reshape(-1)


def plain_row_index(heads: int, layout: HeadLayout, source_heads: int) -> torch.Tensor:
    """只补小头、不换座次的行号列子（值投影用；也用于"原始坐次"的对照实验）。

    白话：跟换了座次那条只差一步：这里只把"两套顶八家用"的连号点名做完（0,0,0,0,1,1,1,1
    这样），每个头内部一个座次都不动。不参与转角的那一路本来就该照原样坐，拿来当对照也合适。"""
    hd = layout.head_dim
    owner = torch.arange(heads) // max(1, heads // source_heads)
    return (owner[:, None] * hd + torch.arange(hd)[None, :]).reshape(-1)


def pack_qkv(
    q_rows: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    layout: HeadLayout,
    *,
    perm_qk: bool = True,
) -> torch.Tensor:
    """三条投影拼成自研栈的打包权重 `(3×heads×head_dim, K)`，行序 = (槽位, 头, 坐标)。

    白话：三样材料按"先查询八头、再键八头、最后值八头"的顺序摞成一本，每头内部按内核认得的
    坐次排好；查询与键坐同一个座次（不然两边搭不上手），值不参与转角就照原样坐。

    :param q_rows: `(heads×head_dim, K)` 纯查询投影（门控已拆掉）。
    :param k_rows: `(kv_heads×head_dim, K)` 键投影（本函数负责补齐到大头数）。
    :param v_rows: `(kv_heads×head_dim, K)` 值投影。
    :param layout: 头布局。
    :param perm_qk: False 时查询/键也不换座次（与 HF 原始口径做对照用）。
    :returns: `(3×heads×head_dim, K)` 打包权重。
    """
    hidden = q_rows.shape[1]
    if k_rows.shape != v_rows.shape:
        raise ValueError(f"键/值投影形状需相同，实得 {tuple(k_rows.shape)}/{tuple(v_rows.shape)}")
    idx = rotated_row_index if perm_qk else plain_row_index
    q_idx = idx(layout.heads, layout, layout.heads)
    k_idx = idx(layout.heads, layout, layout.kv_heads)
    q = q_rows.index_select(0, q_idx.to(q_rows.device))
    k = k_rows.index_select(0, k_idx.to(k_rows.device))
    v = v_rows.index_select(0, plain_row_index(layout.heads, layout, layout.kv_heads).to(v_rows.device))
    out = torch.cat([q, k, v], dim=0)
    if out.shape[0] != 3 * layout.heads * layout.head_dim or out.shape[1] != hidden:
        raise AssertionError(f"打包形状出格：{tuple(out.shape)}")
    return out.contiguous()


def permute_norm(norm_weight: torch.Tensor, layout: HeadLayout) -> torch.Tensor:
    """把逐头缩放系数（q_norm/k_norm 那种 head_dim 长向量）按同一座次重排。

    白话：每个头里每个坐标配一个缩放数；坐标换了座次，配的数也得跟着搬，不然张冠李戴。
    逐坐标的缩放与"对全体求均方再缩放"这一步不关心座次先后，所以换完座次数值仍然等价。
    """
    heads = norm_weight.shape[0] // layout.head_dim
    base = norm_weight.view(heads, layout.head_dim)
    perm_t = torch.tensor(rope_pair_permutation(layout), dtype=torch.long, device=base.device)
    return base.index_select(1, perm_t).reshape(-1).contiguous()


def attention_bias_pack(
    q_bias: torch.Tensor | None,
    k_bias: torch.Tensor | None,
    v_bias: torch.Tensor | None,
    layout: HeadLayout,
) -> torch.Tensor | None:
    """把三条投影的偏置拼成打包口径（与 `pack_qkv` 同序）；三条都没有时返回 None。

    白话：有的模型每行还带一份加成。这里把三份加成按同样的坐次摞成一叠；一份都没有就如实
    交回 None，让上层不必再猜"是全零还是压根没有"。
    """
    if q_bias is None and k_bias is None and v_bias is None:
        return None
    if q_bias is None or k_bias is None or v_bias is None:
        raise ValueError("偏置要么三条都有，要么三条都没有，实得混合形态")
    parts = [
        q_bias.index_select(0, rotated_row_index(layout.heads, layout, layout.heads)),
        k_bias.index_select(0, rotated_row_index(layout.heads, layout, layout.kv_heads)),
        v_bias.index_select(0, plain_row_index(layout.heads, layout, layout.kv_heads)),
    ]
    return torch.cat(parts, dim=0).contiguous()


def letter_row_indices(vocab: int, letter_ids: list[int]) -> torch.Tensor:
    """26 个字母在输出层里的行号（读出层的取行依据；与布局无关，放这里给 loader 复用）。

    白话：整本对照表里，二十六个大写字母各占一行；把它们的行号列成一张条子，取答案那一步
    就只翻这二十六页，别的一页都不看。
    """
    bad = [i for i in letter_ids if not 0 <= i < vocab]
    if bad:
        raise ValueError(f"字母编号越界 {bad}（词表行数 {vocab}）")
    return torch.tensor(list(letter_ids), dtype=torch.long)


def fold_unit_shift(norm_weight: torch.Tensor) -> torch.Tensor:
    """把 RMSNorm 的缩放系数折算成"直接相乘"的那份：交回 `1 + w`。

    白话：Qwen3/3.5 这系身板的音量旋钮出厂装在"零"上，用的时候拧的是"一格加读数"——
    也就是先垫一个 1 再乘。自研栈的融合件只认"读数直接乘"，所以在搬完座次之后把那个
    垫底的 1 折进读数里：折完的数比原读数大 1，形状一位没动，乘的效果一模一样。
    """
    return torch.ones_like(norm_weight) + norm_weight
