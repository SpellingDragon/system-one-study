"""R-P0-2 训测分离守卫：训练题源必须挂 train 轴，永不碰评测考卷。

背景（audit-remediation-1010 评审 2.4.1，实证于当时 HEAD）：
    `sft.py` 默认与 `production/configs/*.yaml` 全部 5 份曾把训练 axis 写成
    `quality`——而 quality 轴主力集（typed-decisions / intern-decision）在 registry
    登记的就是 `split="test"`。按现配置点火＝在评测考卷上训练＝假分。
本文件是防这条雷复发的机器守卫：
    ① 逐条核验 sft.py 默认配置与全部 `production/configs/*.yaml` 的训练 axis：
       不得为 quality、不得属于任何评测轴、且该轴在 registry 上登记的副本必须
       全为 train 档。轴→split 映射唯一取自 `sys1.eval.registry.REGISTRY`，
       本文件不硬编码任何集名——registry 改登记，守卫跟着变，这才叫守卫。
    ② R14 注错自证：临时造一份 `axis: quality` 的配置（tmp 文件 / monkeypatch
       默认档），守卫必须变红。永远不会红的守卫等于没有守卫。

运行方式（与 test_prod_sft.py 同层）：
    cd release && .venv/bin/python -m pytest tests/test_train_axis_guard.py -q
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from production import sft
from sys1.eval import registry

RELEASE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = RELEASE_ROOT / "production" / "configs"
TRAIN_SPLIT = "train"      # train 轴上唯一合法的 split 档位值（档位名，不是集名）


# ── 尺子：轴→split 映射一律现读 registry ─────────────────────────────────────
def axis_split_map() -> dict[str, set[str]]:
    """从 registry 读「每个轴上登记了哪些 split 档」，一个集名都不抄。"""
    m: dict[str, set[str]] = {}
    for _pid, pin in registry.REGISTRY.items():
        m.setdefault(pin.axis, set()).add(pin.split)
    return m


def eval_axes(amap: dict[str, set[str]]) -> set[str]:
    """评测轴＝凡挂了非 train 档副本的轴（quality/calibration/longctx/multimodal…）。"""
    return {ax for ax, splits in amap.items() if splits - {TRAIN_SPLIT}}


def assert_training_axis_ok(axis: str, *, amap: dict[str, set[str]] | None = None) -> None:
    """校验单个训练轴：非 quality、不在评测轴集合内、已登记且轴上副本全为 train 档。"""
    amap = amap if amap is not None else axis_split_map()
    if not isinstance(axis, str) or not axis.strip():
        raise AssertionError(f"训练 axis 配置为空/非串：{axis!r}")
    if axis == "quality":
        raise AssertionError("训练 axis 指向 quality 评测轴＝训测同集（audit-remediation-1010 2.4.1 的雷）")
    if axis not in amap:
        raise AssertionError(f"训练 axis {axis!r} 未在 registry 登记（无轴即无题源，来路不明）")
    if axis in eval_axes(amap):
        raise AssertionError(f"训练 axis {axis!r} 是评测轴（轴上含非 train 档考卷），训练不得触及")
    if amap[axis] != {TRAIN_SPLIT}:
        raise AssertionError(f"训练 axis {axis!r} 上的副本档位为 {sorted(amap[axis])}，只许纯 train 轴")


def configured_axes(*, extra_configs: list[Path] | None = None) -> list[tuple[str, str]]:
    """收集（来源, 训练 axis）：sft.py 默认档 + 全部 yaml；yaml 没写 axis 即沿用默认档。"""
    out: list[tuple[str, str]] = [("production/sft.py::default_cfg", str(sft.default_cfg()["axis"]))]
    paths = sorted(CONFIG_DIR.glob("*.yaml")) + [Path(p) for p in (extra_configs or [])]
    for p in paths:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if "axis" in data:
            name = str(p.relative_to(RELEASE_ROOT)) if p.is_relative_to(RELEASE_ROOT) else str(p)
            out.append((name, str(data["axis"])))
    return out


def check_all(*, extra_configs: list[Path] | None = None) -> list[str]:
    """跑一遍全量核验，返回违规清单（空表＝全绿）；带来源名，红了能直接定位是哪份配置。"""
    amap = axis_split_map()
    return [f"{src}: {e}" for src, axis in configured_axes(extra_configs=extra_configs)
            for e in [_exc_or_none(axis, amap)] if e]


def _exc_or_none(axis: str, amap: dict[str, set[str]]) -> str | None:
    try:
        assert_training_axis_ok(axis, amap=amap)
        return None
    except AssertionError as e:
        return str(e)


# ── ① 正向守卫：默认档与全部正式/冒烟配置的训练轴都不得触及考卷 ──────────────
def test_registry_train_axis_is_pure_train():
    """前提自查：registry 的 train 轴本身必须全为 train 档，否则后面无从谈起。"""
    amap = axis_split_map()
    assert amap.get(registry.TRAIN_AXIS) == {TRAIN_SPLIT}, \
        f"train 轴上混入了 {sorted(amap.get(registry.TRAIN_AXIS) or set())}——registry 台账被污染"


def test_default_and_all_prod_configs_are_train_isolated():
    bad = check_all()
    assert not bad, "训练配置指向了评测轴：\n" + "\n".join(bad)
    axes = {axis for _, axis in configured_axes()}
    assert axes, "一份训练配置都没读到——configs/ 目录或 sft.py 默认档失踪"


# ── ①b 反向枚举：registry 认定的每一根评测轴，拿来当训练轴都必须红 ────────────
@pytest.mark.parametrize("bad_axis", ["quality", "unregistered-axis-demo"])
def test_guard_reds_on_named_bad_axes(bad_axis: str):
    with pytest.raises(AssertionError):
        assert_training_axis_ok(bad_axis)


def test_guard_reds_on_every_eval_axis():
    for ax in sorted(eval_axes(axis_split_map())):
        with pytest.raises(AssertionError, match=ax):
            assert_training_axis_ok(ax)


# ── ② R14 注错自证：守卫对真实代码路径必须能红 ───────────────────────────────
def test_r14_injected_yaml_turns_red(tmp_path: Path):
    """临时目录造一份 axis: quality 的配置——扫描必须报红；真实 configs/ 一个字节不动。"""
    poison = tmp_path / "poison.yaml"
    poison.write_text("axis: quality\nlimit: 8\n", encoding="utf-8")
    bad = check_all(extra_configs=[poison])
    assert bad, "注入 quality 轴后守卫仍全绿＝守卫失效"
    assert any("poison.yaml" in row and "quality" in row for row in bad), bad


def test_r14_monkeypatch_default_turns_red(monkeypatch: pytest.MonkeyPatch):
    """monkeypatch 把默认档暂时打回旧值 quality——守卫必须红；monkeypatch 自动还原，不落盘。"""
    real = sft.default_cfg
    monkeypatch.setattr(sft, "default_cfg", lambda: {**real(), "axis": "quality"})
    bad = check_all()
    assert any("sft.py::default_cfg" in row for row in bad), f"默认档注入 quality 后守卫未红：{bad}"
