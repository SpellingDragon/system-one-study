"""sys1.decision — 决策程序（两阶段同构桥梁）。

对外只交四件事：`from_systemone/render`（渲染）、`readout`（末位字母读出）、
`apply_temperature`（类型温度）、`wide_vote`（>26 候选分组票选），外加渲染版号
`RENDER_VERSION`。本目录对模型的全部所知只有两样——前向交出的逐位置数字
`(B, T, d)` 与输出层权重 `(vocab, d)`；一阶段小 GPT 与二阶段预训练 backbone
共用同一份代码，换 backbone 时本目录零改动（父 design D1 的落地形态，
tests/test_model.py 与 tests/test_decision.py 各有静态门守着）。

零生成约定：本目录任何文件都不许出现生成/解码循环（`output_tokens = 0`），
由 `tests/test_decision.py::test_no_generate_*` 静态断言。
"""
from __future__ import annotations

from .readout import Readout, last_positions, lengths_from_mask, option_scores, readout, to_probs
from .render import (
    EMPTY_STATE,
    LETTERS,
    MAX_OPTIONS,
    OTHER_KEY,
    OTHER_TEXT,
    RENDER_VERSION,
    SYSTEM_LINE,
    THINK_OFF_SUFFIX,
    RenderError,
    from_systemone,
    option_lines,
    option_order,
    prompt_text,
    render,
    subrow,
    user_content,
)
from .temperature import (
    DEFAULT_TEMPERATURE,
    TEMP_MAX,
    TEMP_MIN,
    UNCALIBRATED,
    Temperature,
    apply_temperature,
    clamp_temperature,
    temperature_for,
    temperatures_for,
)
from .wide import GROUP_SIZE, KEEP_TOP, RESIDUAL_SHARE, VoteRound, WideError, plan_groups, wide_vote

__all__ = [
    "DEFAULT_TEMPERATURE",
    "EMPTY_STATE",
    "GROUP_SIZE",
    "KEEP_TOP",
    "LETTERS",
    "MAX_OPTIONS",
    "OTHER_KEY",
    "OTHER_TEXT",
    "RENDER_VERSION",
    "RESIDUAL_SHARE",
    "SYSTEM_LINE",
    "TEMP_MAX",
    "TEMP_MIN",
    "THINK_OFF_SUFFIX",
    "UNCALIBRATED",
    "Readout",
    "RenderError",
    "Temperature",
    "VoteRound",
    "WideError",
    "apply_temperature",
    "clamp_temperature",
    "from_systemone",
    "last_positions",
    "lengths_from_mask",
    "option_lines",
    "option_order",
    "option_scores",
    "plan_groups",
    "prompt_text",
    "readout",
    "render",
    "subrow",
    "temperature_for",
    "temperatures_for",
    "to_probs",
    "user_content",
    "wide_vote",
]
