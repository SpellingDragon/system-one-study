"""sys1.runs — 实验记录本（runs/ lab notebook）的对外门面。

【做什么】把一次实验的四份记录（参数单、机器卡、曲线流水、三行笔记）集中到一个句柄上，
让"报告里每个数字都能指回一个 run 目录"这条铁律有落地的地方。

【怎么做】实现全在 context.py，本文件只做 re-export：new_run 开目录、RunContext 记曲线与结论、
三个异常类供调用方精确捕获。逻辑放一个文件是因为四件套的写入顺序与命名是强耦合的一套契约。

【为什么】曾考虑把每个动作拆成 config.py / metrics.py / notes.py 三个模块（方案否）——
它们共享同一套路径与命名约定，拆开后约定会散落在三处，改一处忘两处反而更容易破坏契约。
"""
from __future__ import annotations

from .context import (
    MetricsConflictError,
    MissingConclusionError,
    RunContext,
    RunNotebookError,
    default_runs_root,
    new_run,
)

__all__ = [
    "MetricsConflictError",
    "MissingConclusionError",
    "RunContext",
    "RunNotebookError",
    "default_runs_root",
    "new_run",
]
