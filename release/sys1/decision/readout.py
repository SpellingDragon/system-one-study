"""末位字母读出：只吃"逐位置的数字串"与"输出层权重"，交出每个候选的分数与份额。

【做什么】
    一段提示走完模型后，每个位置留下一串可学的数字。本模块取"最后一个真位置"的那串
    数字，跟对照表里 26 个字母各自的输出行做点积，得到候选分数；再按题类型缩放、折成份额。
    全程不生成任何符号——读点位之后本来也不存在任何东西。

【怎么做】
    两件事撑起全部功能。第一件是"取对行"：一批序列右对齐补空白时，每行的真正末位是
    `lengths[i] - 1` 而不是张量的最后一列，所以按行用一个下标列子把它们挑出来——这样
    "单独算一行"与"混在补了空白的批次里算同一行"必然同值（因果性由模型侧保证，本层
    只负责别把点位取错）。第二件是"取对列"：字母行 = `head_weight.index_select(0, ids)`，
    行序与渲染层的字母序一一对应，`order[i]` 就是第 i 列代表的候选代号。
    折份额走 `apply_temperature` 之后的一次单调折算，倍数缺失时退成 1.0 并在结果里
    如实标 `uncalibrated`。校验全部前置：维度不对、字母下标越界或重复、lengths 出格，
    一律当场抛错，绝不带着错形状往下算。

【为什么】
    这一层是全项目最重要的解耦点：它对模型的全部所知只有"给我 (B,T,d) 的数字与
    (vocab,d) 的输出层权重"两样，一阶段的随机小 GPT 与二阶段的预训练 backbone 共用同一份
    读出代码。被否方案一：直接调模型把整张对照表的分数都算出来再挑字母列——白算
    vocab−26 倍，且把"读哪几行"的决定权交回模型侧，换 backbone 时就要改本层。
    被否方案二：用批次张量的 `[:, -1]` 当末位——右补空白时那正是空白位，训练与评测会
    在看不见的地方漂移，spec 的"右 padding 不变性"场景就是专门钉这条的。
    被否方案三：在本层内做贪心解码取字母——那是生成循环，`output_tokens = 0` 是红线，
    tests 里有静态断言把 `.generate(` 挡在门外。
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .render import MAX_OPTIONS
from .temperature import apply_temperature, temperatures_for


@dataclass(frozen=True)
class Readout:
    """一次读出的全部产出：分数、份额，以及每行用的倍数与"这行到底调没调过"。"""

    scores: torch.Tensor          # (B, k) 字母行点积结果
    probs: torch.Tensor           # (B, k) 每行和为 1 的份额
    temperatures: tuple[float, ...]
    uncalibrated: tuple[bool, ...]

    def to_list(self) -> list[list[float]]:
        """摊成普通 list（写 jsonl 预测行用；张量不进产物文件）。

        白话：把数字表变成"一列列普通小数"，好直接写进结果文件里，读的人不用装工具
        也能打开看。
        """
        return [[float(v) for v in row] for row in self.probs]


def lengths_from_mask(attn_mask: torch.Tensor) -> list[int]:
    """从 (B,T) 的 0/1 掩码算每行真符号的个数（右补空白约定下即"真末位 + 1"）。

    白话：掩码里 1 代表真有内容、0 代表垫出来的空洞。一行里有几个 1，就读到第几个
    位置；空洞都排在后面，所以数一数就知道该在哪一格收答案。
    """
    if attn_mask.dim() != 2:
        raise ValueError(f"attn_mask 需为 (B,T) 两维，实得 {tuple(attn_mask.shape)}")
    counts = attn_mask.to(torch.bool).sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("存在整行都是空洞的样本（长度 0 没有可读位置）")
    return [int(c) for c in counts]


def last_positions(lengths: Sequence[int] | None, batch: int, seq_len: int) -> torch.Tensor:
    """把每行长度换算成"该读哪一格"的下标列子；None 表示整批都没垫空洞。

    白话：每行真正写完的最后一个格子在哪个位置，就按那个格子去收。不补空洞时
    就是最右一格；补了空洞时，第 3 行有 7 个字就读第 7 格，绝不读后面的空位。
    """
    if lengths is None:
        return torch.full((batch,), seq_len - 1, dtype=torch.long)
    values = list(lengths)
    if len(values) != batch:
        raise ValueError(f"lengths 长度 {len(values)} 与批大小 {batch} 不符")
    for i, n in enumerate(values):
        if not isinstance(n, int) or isinstance(n, bool) or n < 1 or n > seq_len:
            raise ValueError(
                f"第 {i} 行的长度 {n!r} 出格：需在 [1, {seq_len}] 之间（右补空白约定下读第 {n - 1} 格）"
            )
    return torch.tensor(values, dtype=torch.long) - 1


def option_scores(
    last_hidden: torch.Tensor,
    head_weight: torch.Tensor,
    letter_ids: Sequence[int],
    lengths: Sequence[int] | None = None,
) -> torch.Tensor:
    """末位数字 × 字母行 = `(B, k)` 的候选分数（本模块唯一碰模型产物的地方）。

    白话：每行先取出"写完最后一字时留下的那串数字"，再分别跟 26 个字母各自的那串
    数字比一比、像配钥匙一样打个赞同度；打得越像，分越高。这里只算字母那几列，
    整张对照表其余行一律不碰。

    :param last_hidden: `(B, T, d)`——模型前向交出的逐位置数字。
    :param head_weight: `(vocab, d)`——输出层权重，按行就是每个符号的分数方向。
    :param letter_ids:  候选字母的符号编号，顺序即渲染层的字母序（A 在第一个）。
    :param lengths:     每行真符号数；None 表示按整行最右一格读。
    :returns:           `(B, k)` float32 分数。
    """
    ids = _check_inputs(last_hidden, head_weight, letter_ids)
    batch, seq_len = last_hidden.shape[0], last_hidden.shape[1]
    pos = last_positions(lengths, batch, seq_len).to(last_hidden.device)
    rows = last_hidden[torch.arange(batch, device=last_hidden.device), pos]      # (B, d)
    letter_rows = head_weight.index_select(0, ids.to(head_weight.device))            # (k, d)
    # 校验层产出的下标张量落在 CPU；GPU/MPS 上必须随权重同设备，否则 index_select 跨设备崩。
    return rows.float() @ letter_rows.float().transpose(0, 1)                     # (B, k)


def to_probs(scores: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """把一行的分数折成和为 1 的份额（温度只缩放、不改名次）。

    白话：几个候选各自报个胆量分，先同除一个正数调调陡缓，再摊成一锅汤看各分几成；
    分到最多的那个就是答案，那几成也就是我们的确信程度。
    """
    scaled = scores if temperature == 1.0 else scores / temperature
    return torch.softmax(scaled.float(), dim=-1)


def readout(
    last_hidden: torch.Tensor,
    head_weight: torch.Tensor,
    letter_ids: Sequence[int],
    lengths: Sequence[int] | None = None,
    *,
    qtypes: Sequence[str | None] | str | None = None,
    table: dict[str, float] | None = None,
) -> Readout:
    """完整一次读出：取末位 → 打字母分 → 按类型缩放 → 折份额。

    白话：把上面几步串起来，一趟就交出两样东西——每个候选分到几成，以及每一行究竟
    是按哪个倍数折的、那个倍数是调过的还是临时按原样算的。后者要如实写出来，
    报告里不能把"没调过"混在"调过"里。
    """
    scores = option_scores(last_hidden, head_weight, letter_ids, lengths)
    values, calibrated = temperatures_for(qtypes, table, scores.shape[0])
    probs = torch.softmax(apply_temperature(scores, qtypes, table).float(), dim=-1)
    return Readout(scores=scores, probs=probs, temperatures=values, uncalibrated=tuple(not c for c in calibrated))


# ---------------------------------------------------------------- 入参体检
def _check_inputs(
    last_hidden: torch.Tensor,
    head_weight: torch.Tensor,
    letter_ids: Sequence[int],
) -> torch.Tensor:
    """形状/取值/重复一次查清，返回可直接 index_select 的字母下标张量。"""
    if not isinstance(last_hidden, torch.Tensor):
        raise TypeError(f"last_hidden 需为 torch.Tensor（逐位置数字），实得 {type(last_hidden).__name__}")
    if last_hidden.dim() != 3:
        raise ValueError(f"last_hidden 需为 (B,T,d) 三维，实得形状 {tuple(last_hidden.shape)}")
    if not isinstance(head_weight, torch.Tensor):
        raise TypeError(f"head_weight 需为输出层权重张量本身 (vocab,d)，实得 {type(head_weight).__name__}")
    if head_weight.dim() != 2:
        raise ValueError(f"head_weight 需为 (vocab,d) 两维，实得形状 {tuple(head_weight.shape)}")
    batch, seq_len, width = last_hidden.shape
    if seq_len < 1 or width < 1:
        raise ValueError(f"逐位置数字的形状出格：{(batch, seq_len, width)}")
    if head_weight.shape[1] != width:
        raise ValueError(
            f"输出层宽度 {head_weight.shape[1]} 与逐位置数字的 {width} 不符，点积无法对齐"
        )
    return _letter_id_tensor(letter_ids, int(head_weight.shape[0]))


def _letter_id_tensor(letter_ids: Sequence[int], vocab: int) -> torch.Tensor:
    """字母下标的体检：非空、不超 26、逐个是 [0, vocab) 内的整数且不重复。"""
    ids = list(letter_ids)
    if not ids:
        raise ValueError("letter_ids 为空：一个候选都没有就没有可读的答案")
    if len(ids) > MAX_OPTIONS:
        raise ValueError(f"一轮最多读 {MAX_OPTIONS} 个字母候选，实得 {len(ids)}；超出请走 wide.py 分组票选")
    for i, n in enumerate(ids):
        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError(f"letter_ids[{i}] 必须是 int 符号编号，实得 {type(n).__name__}")
        if n < 0 or n >= vocab:
            raise ValueError(f"letter_ids[{i}] = {n} 越出对照表行数 {vocab}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"字母编号必须两两不同，否则读出的列对不上代号：{ids}")
    return torch.tensor(ids, dtype=torch.long)


__all__ = [
    "Readout",
    "last_positions",
    "lengths_from_mask",
    "option_scores",
    "readout",
    "to_probs",
]
