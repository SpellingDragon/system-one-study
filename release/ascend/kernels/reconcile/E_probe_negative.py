"""E_probe_negative.py — 负例自证（R14）：新加的秩契约钉与结构判据**必须真响**（agent E）。

做法：把 `sys1.layers.attention` 的 `attention_masked` 临时换回**修复前**的实现（逐字照抄
git 改前两行：`masked_fill(~window_band(...))` + `reshape(lead_shape(q) + (dim,))`），再调用
新用例本体，看它是否判否。零写入产品件/测试件，只在进程内打桩。

用法（release 目录下）：
    .venv/bin/python ascend/kernels/reconcile/E_probe_negative.py
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]      # release/
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from sys1.layers import attention as LA


def masked_prefix(q, k, v, *, window=None, scale=None) -> LA.AttnOut:
    """修复前的 `attention_masked` 原样副本（attention.py:320/322 旧文本）。"""
    q3, k3, v3 = LA._flatten_heads(q, k, v)
    seq, dim = q3.size(1), q3.size(-1)
    scale = dim ** -0.5 if scale is None else float(scale)
    score = q3.to(torch.float32) @ k3.to(torch.float32).transpose(-1, -2) * scale
    score = score.masked_fill(~LA.window_band(seq, window, device=q.device), LA.NEG_INF)
    out = torch.softmax(score, dim=-1) @ v3.to(torch.float32)
    return LA.AttnOut(out.reshape(LA.lead_shape(q) + (dim,)) if q.dim() > 3 else out, {"route": "masked"})


def band_prefix(q, k, v, window, *, chunk=6, route="local", scale=None) -> LA.AttnOut:
    """修复前的 `attention_band` 出口 reshape 原样副本（attention.py:391 旧文本；数值无所谓，只看秩）。"""
    q3, _, _ = LA._flatten_heads(q, k, v)
    out = torch.zeros_like(q3, dtype=torch.float32)
    return LA.AttnOut(out.reshape(LA.lead_shape(q) + (q3.size(-1),)) if q.dim() > 3 else out, {"route": route})


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def expect_fail(label: str, fn, *args) -> None:
    """跑一次：期望它**响**（AssertionError 或 RuntimeError）；不响就是判据是摆设。"""
    try:
        fn(*args)
    except (AssertionError, RuntimeError, IndexError) as e:
        first = (traceback.format_exception_only(type(e), e)[0] or "").strip()
        print(f"  [RESPONDED] {label}: {type(e).__name__}: {first[:150]}")
        return
    print(f"  [SILENT!!]  {label}: 没响 —— 判据是摆设，不许合入")


def main() -> None:
    shapes = load("t_attention_shapes", ROOT / "tests" / "test_attention_shapes.py")
    longctx = load("t_longctx", ROOT / "tests" / "test_longctx.py")
    print(f"[env] torch={torch.__version__}  桩：把 attention_masked / attention_band 换回修复前实现")

    real_masked, real_band = LA.attention_masked, LA.attention_band

    # ---- 阶段 1：两条路同时打回修复前 → 秩契约钉必须条声响
    LA.attention_masked, LA.attention_band = masked_prefix, band_prefix
    try:
        for rank in (2, 3, 4):
            expect_fail(f"test_attention_routes_preserve_rank[{rank}] (pre-fix 两条路)",
                        shapes.test_attention_routes_preserve_rank, rank)
        for route in ("masked", "band_kernel", "band_local"):
            expect_fail(f"test_batched_4d_matches_per_sample_loop[{route}] (pre-fix 两条路)",
                        shapes.test_batched_4d_matches_per_sample_loop, route)
    finally:
        LA.attention_masked, LA.attention_band = real_masked, real_band
    print("  [N/A]       test_masked_fill_band_is_same_rank_as_scores: 说明性钉（torch 广播事实与正确喂法），"
          "不受产品桩影响，故不入『必须真响』清单")

    # ---- 阶段 2：只把被报告的 `attention_masked` 打回修复前 → 结构判据用例也要响
    #      （响在哪一条很关键：4 维 vs 3 维在 argmax 比较处炸开，说明形状缺陷会把结构判据的
    #       报错指错地方——这正是"先修形状、再谈判据"的顺序理由。）
    LA.attention_masked = masked_prefix
    try:
        expect_fail("test_sliding_routes_agree_with_mask_reference (pre-fix 掩码路)",
                    longctx.test_sliding_routes_agree_with_mask_reference)
        print("  [N/A]       test_sliding_visible_set_criterion_detects_reversed_window: 它是负例发生器"
              "（检验 A1 是不是摆设），产品桩不该让它响；pre-fix 下 A1 仍判『正向带 ≠ 反向带』成立"
              "→ A1 的判别力与形状缺陷无关")
    finally:
        LA.attention_masked = real_masked

    print("[对照] 桩全部撤掉后，同一批用例必须全绿：")
    for rank in (2, 3, 4):
        shapes.test_attention_routes_preserve_rank(rank)
    for route in ("masked", "band_kernel", "band_local"):
        shapes.test_batched_4d_matches_per_sample_loop(route)
    shapes.test_masked_fill_band_is_same_rank_as_scores()
    longctx.test_sliding_routes_agree_with_mask_reference()
    longctx.test_sliding_visible_set_criterion_detects_reversed_window()
    print("  [PASS] 9/9 现用例（3 秩 + 3 批 + 1 广播形状 + 2 结构判据）全绿")


if __name__ == "__main__":
    main()
