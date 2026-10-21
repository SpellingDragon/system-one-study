"""类型温度层：把"每题一档缩放倍数"的标定结果作用到候选分数上，且不改变答案次序。

【做什么】
    给定一张 `qtype → 倍数` 的表和一批候选分数，按题的类型把分数同除一个正数，得到
    更"说到做到"的份额；表里查不到该类型时一律用 1.0（等于不缩放）并显式打上
    `uncalibrated` 标注，绝不用一个偷偷生效的默认值冒充标定过。

【怎么做】
    三层递进。`clamp_temperature()` 把任意入参收成一次合法决定：必须是有限正数、
    落进 [0.2, 5.0] 区间（越界就贴到边界），不是数字、是布尔、是 NaN/Inf 都退回 1.0
    并记为未标定；区间两端与 `sys1/calibrate.py` 的网格边界同源，测试里逐值钉住，
    两边想各自漂移就有一头先红。`temperatures_for()` 按题类型（或逐行类型列表）查表，
    产出"倍数序列 + 是否标定序列"。`apply_temperature()` 只做一件事：把分数同除倍数——
    同除正数是严格单调变换，谁排第一、谁并列，都不会因为除过一个正数而改变。
    数据流：查表 → 夹界 → 缩放，份额折算是调用方（`readout.py`）的事，本模块不碰。

【为什么】
    温度必须在分数上作用、而不是在已经折好的份额上作用：对份额做非均匀拉伸会翻答案，
    单调保证直接作废——这一点 p1-05 的 docstring 里已列为被否方案，本层照同一口径落地。
    被否方案一：缺表时用池化温度或上一次的值顶上——"看起来更聪明"，但报告会误以为
    这一类标定过；宁可返回 1.0 加 `uncalibrated`，把没做过的事如实写出来。
    被否方案二：把倍数夹界交给调用方自觉——上游拟合出来的数值本就落在网格里，但表可以
    被人工编辑、被 json 读歪，边界不守在这里就会传到 softmax 里把分布打成尖针。
    被否方案三：为每行单独构造倍数张量再逐行乘——与"整表同除一个标量"相比多一次内存
    分配，收益只在混型批里才出现，故统一走"按行组装一列倍数、一次广播除"。
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

# 与 sys1/calibrate.py 的 GRID_LO / GRID_HI 同源（那是拟合网格的边界，这里是作用边界）。
# 刻意不 import 那个模块：本域只依赖"末位那串数字 + 输出层权重"，多一条跨域依赖就多一处
# 二阶段复用障碍；一致性由 tests/test_decision.py 逐值断言守住。
TEMP_MIN = 0.2
TEMP_MAX = 5.0

# 缺表时的落点：1.0 = 不缩放。UNCALIBRATED 是要写进 decision_config / run 记录的标注字样。
DEFAULT_TEMPERATURE = 1.0
UNCALIBRATED = "uncalibrated"


@dataclass(frozen=True)
class Temperature:
    """一次"该题该除多少"的决定：数值 + 是否真的标定过。"""

    value: float
    calibrated: bool

    @property
    def uncalibrated(self) -> bool:
        """与 `.calibrated` 互补的读法：报告里更愿意直接说"这一类没标定"。

        白话：同一个事实的两个问法。一个问"调过没有"，一个问"是不是没调"；这里答的是
        后一个，翻成大白话就是"这题的倍数是我临时按原样用的，不是数据里拟出来的"。
        """
        return not self.calibrated

    @property
    def label(self) -> str:
        """给配置与报告用的一个词标注：标定过写 calibrated，否则写 `uncalibrated`。

        白话：把"调过 / 没调过"这件事压成一个词，好直接抄进设置单和实验记录里；
        没调过就得白纸黑字写明没调过，不许让看报告的人以为所有数字同样可靠。
        """
        return "calibrated" if self.calibrated else UNCALIBRATED


def clamp_temperature(raw: Any) -> Temperature:
    """把一个原始倍数收成合法决定：有限正数夹进 [0.2, 5.0]，其余一律 1.0 且记未标定。

    白话：表里写的这个数能不能当真，先过三道问：它是不是个货真价实的数（不是文字、
    不是真假的开关、也不是那种除下去会把结果搞坏的怪值）、是不是正的、有没有大到离谱
    或小到离谱；越界的贴到边界上，答不上来的就当没这回事，按原样不动。

    布尔被明确拒掉：`True` 会被 float 成 1.0 冒充"拟好的倍数"，是静默错误的一类。
    """
    if isinstance(raw, bool) or raw is None:
        return Temperature(DEFAULT_TEMPERATURE, calibrated=False)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return Temperature(DEFAULT_TEMPERATURE, calibrated=False)
    if not math.isfinite(value) or value <= 0.0:
        return Temperature(DEFAULT_TEMPERATURE, calibrated=False)
    return Temperature(min(max(value, TEMP_MIN), TEMP_MAX), calibrated=True)


def temperature_for(qtype: str | None, table: Mapping[str, float] | None) -> Temperature:
    """按题类型查表；表缺失、键缺失或值不合法都退成 1.0 并标注未标定。

    白话：拿着题目的类别去那张对照表里找一个数。表没给、表里没有这一类、或者那一格
    写的是不能用的东西，就当作"这一类还没调过"，老老实实按原样出数并把这话记下来。
    """
    if not isinstance(qtype, str) or not isinstance(table, Mapping):
        return Temperature(DEFAULT_TEMPERATURE, calibrated=False)
    if qtype not in table:
        return Temperature(DEFAULT_TEMPERATURE, calibrated=False)
    return clamp_temperature(table[qtype])


def temperatures_for(
    qtypes: Sequence[str | None] | str | None,
    table: Mapping[str, float] | None,
    rows: int,
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    """把"整批同一类型"或"逐行不同类型"展平成两列长度 = rows 的序列。

    白话：一批题可能都是同一种类，也可能一行一个种类。这里把它们都拉成一列数字、
    再拉成一列"这行调过没有"的记号，缩放那步只认列子，不用去猜每一行是什么类。
    """
    keys = _row_keys(qtypes, rows)
    decided = [temperature_for(k, table) for k in keys]
    return tuple(t.value for t in decided), tuple(t.calibrated for t in decided)


def apply_temperature(
    scores: torch.Tensor,
    qtypes: Sequence[str | None] | str | None = None,
    table: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """按类型把分数同除一个正数（1-D 视为单行）；答案次序与并列关系原样保留。

    白话：每一行都除以一个不小于 0.2、不大于 5 的正数。除的是同一个正数，所以谁高
    谁低不会互换，本来一样高的还是并列；变的只是"各分几成"里的陡与缓——倍数大就平、
    倍数小就尖。

    :param scores: `(k,)` 或 `(B, k)` 的分数；dtype 原样保留（不偷偷升精度）。
    :param qtypes: 整批共用的一个类型，或长度为 B 的类型序列；None 视为未标定。
    :param table:  `qtype → 倍数` 表；None / 缺项 → 该行按 1.0 处理。
    :returns:      与入参同形状、同 dtype 的缩放结果。
    """
    if not isinstance(scores, torch.Tensor):
        raise TypeError(f"apply_temperature 需要 torch.Tensor，实得 {type(scores).__name__}")
    squeeze = scores.dim() == 1
    rows = scores.view(-1, scores.shape[-1]) if squeeze else scores
    values, _ = temperatures_for(qtypes, table, rows.shape[0])
    factor = torch.tensor(values, dtype=rows.dtype, device=rows.device).unsqueeze(1)
    out = rows / factor
    return out.squeeze(0) if squeeze else out


def _row_keys(qtypes: Sequence[str | None] | str | None, rows: int) -> list[str | None]:
    """把 qtypes 归一成逐行序列：None → 全 None；字符串 → 整批复用；序列 → 长度必须等于行数。"""
    if qtypes is None:
        return [None] * rows
    if isinstance(qtypes, str):
        return [qtypes] * rows
    keys = list(qtypes)
    if len(keys) != rows:
        raise ValueError(f"qtypes 长度 {len(keys)} 与行数 {rows} 不符（逐行类型必须一一对应）")
    return keys


__all__ = [
    "DEFAULT_TEMPERATURE",
    "TEMP_MAX",
    "TEMP_MIN",
    "UNCALIBRATED",
    "Temperature",
    "apply_temperature",
    "clamp_temperature",
    "temperature_for",
    "temperatures_for",
]
