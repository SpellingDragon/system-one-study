"""超过 26 个候选的分组票选：几轮"字母位"投票拼回一张完整分布。

【做什么】
    一道选择题给出 50 个候选时，一轮只有 26 个字母代号可用。本模块把候选切成几小组
    各问一轮，每组前若干名进决赛再问一轮，最后把两轮结果合成一张"每个候选各分几成"
    的表——列数等于候选数，加起来正好是 1。

【怎么做】
    1. `plan_groups()` 近等分：组数 = ⌈候选数 / 每组上限⌉，余数轮流摊到前几组，于是
       每组真实候选 ≤ 25，腾出第 26 个字母位给"其他（以上都不是）"这枚**残差槽**。
    2. 每组调用一次注入的 `vote(候选列)`，拿到与该列等长的一列份额；残差槽那一格代表
       "组内谁都不像"的质量，按组内真实候选的比例摊回去（组内相对名次不变），使每组
       和恒为 1，全部组的和也为 1。整组质量都落进残差槽时退化成组内平均，绝不除零。
    3. 每组取前 keep 名进决赛（每组人数不足时自动压小名额，确保决赛一定比全集小）；
       决赛名单一律按**字母序**重排（与候选原序一致），再问一轮。决赛名单仍超过一组
       上限时递归走同样的流程。
    4. 合成：决赛者拿 `决赛份额 × (1 − 残差留量)`，落选者共同拿 `残差留量`，按"一半照
       首轮比例、一半均分"分配（首轮份额下溢成绝对零时也不至于一分不得）；最后整体
       摊平到和恰为 1。
    本模块只做分组与算术，一次也不碰模型：谁来出那一列份额由调用方注入，因此一阶段的
    小 GPT 与二阶段 backbone 共用同一段代码。

【为什么】
    为什么不干脆把 50 个候选硬压进一轮：字母只有 26 个，多出来的候选没有可读的代号位；
    为什么不逐对比较（锦标赛式）：那要问 O(k²) 轮，端侧开销承受不起，而分组票选只需
    ⌈k/25⌉ + 1 轮。被否方案一：落选者份额直接归零——一张"50 选 1"的分布里其余 47 项
    全是 0，任何 soft target 训练都会把这种硬零当监督信号学进去，故保留 `residual`
    留量让落选者仍有非零份额。被否方案二：残差槽的质量直接丢弃不归一——组内和就不再
    是 1，合成后全局和也不是 1，spec 的"五十选一分布合法"场景当场破。被否方案三：
    在本模块里直接调用读出函数——那会把模型接口拖进票选算术，二阶段换 backbone 时
    这段干净算术也要跟着改，同构桥梁就断了。
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence

from .render import OTHER_KEY

GROUP_SIZE = 25           # 每组真实候选上限：第 26 个字母位留给残差槽
KEEP_TOP = 3              # 每组进决赛的名次（与参考实现的 wide.keep 同口径）
RESIDUAL_SHARE = 1e-3     # 留给落选者的总份额（与参考实现的 wide.residual 同口径）


class WideError(ValueError):
    """票选前提不成立（候选重复、份额长度或取值不对、决赛无法收敛）时抛出。"""


# 一"轮"的注入形式：给一列候选代号，回一列等长的份额（顺序与入参一一对应）。
VoteRound = Callable[[list[str]], Sequence[float]]


def plan_groups(keys: Sequence[str], group_size: int = GROUP_SIZE) -> list[list[str]]:
    """把候选近等分成若干组：组数尽量少、组间长度差不超过 1，且都不超过 group_size。

    白话：五十个人要分成几桌打牌，先算最少要几桌，再把人尽量摊匀——头几桌多一个人
    补余数，任何一桌都不许多到坐不下；桌子数是由座位上限倒推出来的，不是拍脑袋定的。
    """
    if group_size < 1:
        raise WideError(f"group_size 必须为正，实得 {group_size}")
    names = list(keys)
    if not names:
        raise WideError("候选清单为空，没有可票选的对象")
    n_groups = -(-len(names) // group_size)          # 向上取整，避免多出一轮空问
    size, extra = divmod(len(names), n_groups)
    groups: list[list[str]] = []
    start = 0
    for g in range(n_groups):
        end = start + size + (1 if g < extra else 0)
        groups.append(names[start:end])
        start = end
    return groups


def wide_vote(
    keys: Sequence[str],
    vote: VoteRound,
    *,
    keep: int = KEEP_TOP,
    residual: float = RESIDUAL_SHARE,
    group_size: int = GROUP_SIZE,
) -> dict[str, float]:
    """对 k > 一组上限的候选做多轮票选，返回 `代号 → 份额`（项数 = k，和为 1）。

    白话：一轮问不完就先分组各问一次，每组跑得最快的几名进决赛再问一次；决赛的人
    拿大头，没进决赛的人也不会被抹成零——他们按第一轮的成绩分一小笔留底，最后所有
    份额加回一整，谁多谁少都摆在同一张表上。

    :param keys:     候选代号列，顺序即渲染层的字母序（重复代号直接拒绝）。
    :param vote:     注入的一轮问答：收到一列候选（可能末尾多一枚"其他"残差槽），
                     回一列等长、非负、有限的份额。
    :param keep:     每组进决赛的名次数。
    :param residual: 留给落选者的总份额比例（0 表示不留，须 < 1）。
    :param group_size: 每组真实候选上限（默认 25，留一个字母位给残差槽）。
    :returns:        与 `keys` 同项数的字典，键序保持入参顺序（字母序）。
    """
    names = _check_keys(keys)
    if keep < 1:
        raise WideError(f"keep 必须为正，实得 {keep}")
    if not 0.0 <= residual < 1.0:
        raise WideError(f"residual 需落在 [0, 1) 内，实得 {residual}")

    if len(names) <= group_size:                     # 一轮问完：不存在分组与决赛
        return dict(zip(names, _round(vote, names)))

    groups = plan_groups(names, group_size)
    first = _first_round(groups, vote)
    position = {name: i for i, name in enumerate(names)}
    # 每组能装几个人，就最多只能取前几名——否则下一轮的全集不比这一轮小，递归不会收口。
    finalists = _finalists(groups, first, _effective_keep(groups, keep), position)
    if len(finalists) >= len(names):
        raise WideError(
            f"每组上限 {group_size} 容不下 keep={keep} 的收缩，决赛名单 {len(finalists)} 人"
            f"不少于全集 {len(names)} 人，票选无法收敛"
        )

    if len(finalists) <= group_size:
        final = dict(zip(finalists, _round(vote, finalists)))
    else:                                            # 决赛仍超一组上限：原样递归
        final = wide_vote(finalists, vote, keep=keep, residual=residual, group_size=group_size)

    rest = [name for name in names if name not in set(finalists)]
    out = {name: final[name] * (1.0 - residual) for name in finalists}
    out.update(_loser_share(rest, first, residual))
    return dict(zip(names, _normalize(names, out)))


# ---------------------------------------------------------------- 内部算术（私有）
def _check_keys(keys: Iterable[str]) -> list[str]:
    """候选去重体检：必须是非空字符串且互不重复（重复代号会让份额对不上人）。"""
    names = list(keys)
    if not names:
        raise WideError("候选清单为空，没有可票选的对象")
    if any(not isinstance(n, str) or not n for n in names):
        raise WideError(f"候选代号必须是非空字符串，出现 {names!r}")
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise WideError(f"候选代号重复，份额将无法对应到人：{dup}")
    return names


def _round(vote: VoteRound, candidates: list[str]) -> list[float]:
    """跑一轮并把回值收成合法份额列：长度对齐、非负、有限，然后摊平到和为 1。"""
    raw = list(vote(list(candidates)))
    if len(raw) != len(candidates):
        raise WideError(f"一轮回答回了 {len(raw)} 个数，候选有 {len(candidates)} 个，对不上")
    for name, value in zip(candidates, raw):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise WideError(f"候选 {name!r} 的份额不是数字：{type(value).__name__}")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise WideError(f"候选 {name!r} 的份额非法（需非负有限数）：{value!r}")
    return _unit([float(v) for v in raw])


def _first_round(groups: list[list[str]], vote: VoteRound) -> dict[str, float]:
    """逐组问一轮：残差槽的质量按比例摊回本组候选，保证每组和为 1（注意：返回的首轮结果
    全局总和 = 组数 G，不是 1——全局守恒由决赛合成与落选残差归并后的最终归一保证）。"""
    first: dict[str, float] = {}
    for group in groups:
        shares = _round(vote, list(group) + [OTHER_KEY])
        real = shares[:-1]
        total = sum(real)
        if total <= 0.0:
            # 全组质量都落进残差槽（模型拒绝表态）：退化成组内平均，绝不除零、也不留空组。
            even = 1.0 / len(group)
            first.update(zip(group, [even] * len(group)))
            continue
        first.update(zip(group, [s / total for s in real]))
    return first


def _effective_keep(groups: list[list[str]], keep: int) -> int:
    """把"每组取前几名"压到最小组也还剩一名落选：保证决赛一定比全集小。

    白话：一桌只有三个人的时候，"每桌前三名胜出"等于全桌晋级，下一轮人数一个没少，
    再问一百轮也是白问；所以名额要跟着桌子大小收紧——三个人就只取两名，让每一轮都
    确实刷掉一些人。桌子只剩一个人时再怎么收也刷不掉人，只能由调用方报错。
    """
    shortest = min(len(g) for g in groups)
    return min(keep, max(1, shortest - 1))


def _finalists(
    groups: list[list[str]],
    first: dict[str, float],
    keep: int,
    position: dict[str, int],
) -> list[str]:
    """每组取前 keep 名（同分按字母序取先），合并后的决赛名单一律按字母序重排。"""
    picked: list[str] = []
    for group in groups:
        ranked = sorted(group, key=lambda name: (-first[name], position[name]))
        picked.extend(ranked[:keep])
    return sorted(set(picked), key=lambda name: position[name])


def _loser_share(rest: list[str], first: dict[str, float], residual: float) -> dict[str, float]:
    """落选者共同分走 residual：一半按首轮份额比例给，另一半平均给，于是人人非零。

    白话：没进决赛的人也要有一份留底。这份留底主要照第一轮的成绩排（分高的多拿一点），
    另外压一半平均的钱进去兜底——因为第一轮算出来的可能是浮点下溢的绝对零，纯按比例
    就会让某些人一分不得；一张带硬零的表会被训练当成"这项绝不可能"的监督信号学进去。
    两半各自加起来都是 1，所以整份留量仍然恰好等于 residual，不会多给也不会少给。
    """
    if not rest:
        return {}
    if residual == 0.0:
        return {name: 0.0 for name in rest}
    weights = [max(float(first[name]), 0.0) for name in rest]
    mass = sum(weights)
    even = 1.0 / len(rest)
    proportional = [w / mass if mass > 0.0 else even for w in weights]   # 全零时退化成平均
    return {name: residual * (0.5 * part + 0.5 * even) for name, part in zip(rest, proportional)}


def _normalize(names: list[str], table: dict[str, float]) -> list[float]:
    """按候选原序取值并摊平到和恰为 1；总和为 0 说明票选退化，直接拒绝。"""
    values = [float(table[n]) for n in names]
    total = sum(values)
    if total <= 0.0:
        raise WideError(f"票选结果为全零（和 {total}），无法摊成分布")
    return [v / total for v in values]


def _unit(values: list[float]) -> list[float]:
    """把一列非负数摊成和为 1 的份额；全零时按平均分配（数值兜底，绝不产生 NaN）。"""
    total = sum(values)
    if total <= 0.0:
        even = 1.0 / len(values)
        return [even] * len(values)
    return [v / total for v in values]


__all__ = [
    "GROUP_SIZE",
    "KEEP_TOP",
    "RESIDUAL_SHARE",
    "VoteRound",
    "WideError",
    "plan_groups",
    "wide_vote",
]
