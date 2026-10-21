"""校准数学域单测（p1-05：五场景 + 口径守卫 + 基线暴力核对）。

【做什么】把 calibration spec 的五个场景逐条变成可执行断言，并额外锁死三条容易作弊的边界：
温度不许翻答案（含并列）、样本不足必须老实标注池化、分桶基线不许把 k=3 与 k=20 混成一桶。
【怎么做】合成数据全部无随机或固定种子（可复现）：完美校准数据按"每桶频率==置信"手工构造，
改善场景用已知真温（3.0）反推数据再验拟合值回收；基线用全枚举暴力算一遍对公式。
【为什么】这些数字将来要进报告与合格线，若只测"能跑通"就等于没测；曾考虑随机数据跑一遍
看不报错——否，随机数据既能掩盖公式错、也会让 ECE<1e-9 这类硬阈值变成运气。
"""
from __future__ import annotations

import itertools

import pytest
import torch

from sys1.calibrate import (
    DEFAULT_BINS,
    DEFAULT_GRID,
    ECE,
    FitReport,
    bucket_baselines,
    fit_temperature,
    make_grid,
)

# ============================ 构造器 ============================


def _perfect_probs_targets(bins: int = DEFAULT_BINS, k: int = 40):
    """构造"频率==置信"的完美校准数据（spec 场景：ECE < 1e-9）。

    第 i 桶置信 c_i = (2i+1)/(2*bins)，每桶放 2*bins 个样本、其中 c_i*2*bins = 2i+1 个命中，
    于是桶内平均置信与实际命中率在算术上恒等。k 选项里 top 项之外均分余量，
    要求余量严格小于 c_i（k=40 时 (29/30)/39 ≈ 0.0248 < c_0 ≈ 0.0333），否则 max 取错项。
    """
    per_bin = 2 * bins
    probs, targets = [], []
    for i in range(bins):
        conf = (2 * i + 1) / (2 * bins)
        rest = (1.0 - conf) / (k - 1)
        assert rest < conf, "构造前提：非 top 项必须严格小于 top，否则置信定义漂移"
        row = [conf] + [rest] * (k - 1)
        hits = 2 * i + 1
        for j in range(per_bin):
            probs.append(row)
            targets.append([1.0] + [0.0] * (k - 1) if j < hits else [0.0, 1.0] + [0.0] * (k - 2))
    return (torch.tensor(probs, dtype=torch.float64), torch.tensor(targets, dtype=torch.float64))


def _overconfident_rows(n: int = 900, k: int = 3, seed: int = 0):
    """过自信数据：真把握 = softmax(z/3)（软目标），但分值按 z/1 报（比真实陡 3 倍）。"""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, k, generator=g, dtype=torch.float64) * 2.5
    p_true = torch.softmax(z / 3.0, dim=-1)          # 施加拟合温度后恰好复现此分布 → 最优 T=3.0
    return z, p_true


def _sample_labels(p_true: torch.Tensor, seed: int = 0) -> list[int]:
    """按真实分布逐行抽样出硬标签（用局部生成器，不碰全局随机种子，保证跨用例可复现）。"""
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(p_true.shape[0], 1, generator=g, dtype=torch.float64)
    k = p_true.shape[1]
    return (p_true.cumsum(dim=-1) < u).sum(dim=-1).clamp(max=k - 1).tolist()


# ============================ 场景 1：温度不换 argmax（含 tie-break） ============================


def test_fit_temperature_argmax_unchanged_with_ties():
    """spec『拟合不换 argmax』：逐 qtype 拟合后，温度不改排序，并列（tie-break）也原样并列。"""
    g = torch.Generator().manual_seed(7)
    # 两类、两种候选宽度（3 与 20）混排：不等宽只能走列表入口（矩阵装不下），顺带压住
    # "同一 qtype 内 k 可变"这条真实数据性质（choice 既有三选也有二十选）
    narrow = torch.randn(220, 3, generator=g, dtype=torch.float64)
    wide = torch.randn(220, 20, generator=g, dtype=torch.float64)
    ties = torch.tensor([
        [1.5, 1.5, 0.2], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [-1.0, -1.0, 3.0], [7.0, 7.0, 7.25],
    ], dtype=torch.float64)
    rows = narrow.tolist() + wide.tolist() + ties.tolist()
    labels = [row.index(max(row)) for row in rows]     # 硬目标取当前榜首；并列取首位
    qtypes = ["choice"] * 225 + ["score"] * 220       # 两组各 ≥200 → 走独立拟合分支
    rep = fit_temperature(rows, labels, qtypes=qtypes)

    assert rep.total_samples == 445 and set(rep.temperatures) == {"choice", "score"}
    assert not rep.by_type["choice"].pooled and not rep.by_type["score"].pooled
    for qtype, temp in rep.temperatures.items():
        assert temp in rep.grid, f"{qtype} 的温度必须来自网格"
        assert DEFAULT_GRID[0] <= temp <= DEFAULT_GRID[-1]
    for width in {len(r) for r in rows}:              # 等宽一批做批量对拍（cat 装不下参差）
        mat = torch.tensor([r for r in rows if len(r) == width], dtype=torch.float64)
        raw_top = mat.argmax(dim=-1)
        for temp in rep.temperatures.values():
            assert torch.equal((mat / temp).argmax(dim=-1), raw_top), f"T={temp} 改了 argmax"
            assert torch.equal(torch.softmax(mat / temp, dim=-1).argmax(dim=-1),
                               torch.softmax(mat, dim=-1).argmax(dim=-1))
    # tie-break 专测：并列集合本身在缩放前后逐元素一致（比"首位相同"更强的结论）
    for row in ties:
        for temp in rep.temperatures.values():
            before = torch.isclose(row, row.max())
            after = torch.isclose(row / temp, (row / temp).max())
            assert torch.equal(before, after), f"并列集合被改变: {row} / {temp}"


# ============================ 场景 2：小样本池化 ============================


def test_fit_temperature_pools_small_qtype_120():
    """spec『小样本池化』：某 qtype 有效样本 120 → 用池化温度且 fit_report 标 pooled=true。"""
    z, p_true = _overconfident_rows(n=420, k=4, seed=1)
    labels = _sample_labels(p_true, seed=11)
    qtypes = ["choice"] * 300 + ["noul"] * 120        # noul 恰 120 样本（<200）
    rep = fit_temperature(z, labels, qtypes=qtypes)

    assert set(rep.temperatures) == {"choice", "noul"}
    assert rep.by_type["noul"].n == 120
    assert rep.by_type["noul"].pooled is True, "120 样本 MUST 池化"
    assert rep.by_type["noul"].temperature == rep.pooled_temperature
    assert rep.by_type["choice"].pooled is False, "300 样本不该被池化"
    assert rep.pooled_types == ("noul",)              # 诚实标注：报告自己说谁被池化了
    assert rep.min_samples == 200 and rep.total_samples == 420
    assert isinstance(rep, FitReport)
    # 阈值边界：恰好 200 不池化（规则是 `>=`，写死防后来者"顺手放宽"）
    rep2 = fit_temperature(z[:400], labels[:400], qtypes=["choice"] * 200 + ["noul"] * 200)
    assert rep2.pooled_types == () and not rep2.by_type["noul"].pooled


# ============================ 场景 3：完美校准 ECE 为 0 ============================


def test_ece_perfect_calibration_below_1e9():
    """spec『完美校准 ECE 为 0』：频率与置信完全一致 → ECE < 1e-9（默认 bins=15）。"""
    probs, targets = _perfect_probs_targets()
    ece = ECE(probs, targets, weighted=False)
    assert ece < 1e-9, f"完美校准应为 0，实得 {ece!r}"
    # one-hot 目标下质量口径与硬口径必须收敛同一结论（两口径守卫的正面用例）
    assert ECE(probs, targets, weighted=True) < 1e-9
    # 反向对照：同一置信、全数答错 → 每桶误差 = 置信本身，ECE 必显著大于 0
    wrong = torch.zeros_like(targets)
    wrong[:, 1] = 1.0
    assert ECE(probs, wrong, weighted=False) > 0.4


def test_ece_hard_gap_matches_hand_computed_value():
    """手算对照：全部样本置信 0.99、实际命中一半 → 单桶 ECE 恰为 |0.99-0.50| = 0.49。"""
    k = 5
    probs = torch.full((200, k), (1 - 0.99) / (k - 1), dtype=torch.float64)
    probs[:, 0] = 0.99
    targets = torch.zeros(200, k, dtype=torch.float64)
    targets[:100, 0] = 1.0                            # 前一半全押榜首（命中）
    targets[100:, 1] = 1.0                            # 后一半全押次项（未命中）
    ece = ECE(probs, targets, weighted=False)
    assert abs(ece - 0.49) < 1e-12, ece
    assert ECE(probs, targets.argmax(dim=-1), weighted=False) == pytest.approx(ece)  # 整数标签等价
    with pytest.raises(ValueError, match="probs"):     # 非法概率行直接拒
        ECE(probs * 2.0, targets, weighted=False)
    with pytest.raises(ValueError, match="宽度"):
        ECE(probs, targets[:, :3], weighted=False)


# ============================ 场景 3 续：soft 质量口径与混报守卫 ============================


def test_ece_soft_weighted_guard_against_mixed_reporting():
    """design『两口径禁止混报』：soft 目标 + weighted=False → 报错并提示正确用法。"""
    probs, targets = _perfect_probs_targets()
    soft = targets * 0.7 + 0.3 / targets.shape[1]     # 掺均匀底噪 → top 上的质量变成小数
    with pytest.raises(ValueError, match="weighted=True"):
        ECE(probs, soft, weighted=False)
    assert ECE(soft, soft, weighted=True) >= 0.0      # 质量口径可算，不抛
    # 均匀目标 + 均匀预测：质量口径下把握==质量 → 应为 0；硬口径必须报错（不许把 1/k 当命中）
    uni = torch.full((50, 6), 1 / 6, dtype=torch.float64)
    assert ECE(uni, uni, weighted=True) < 1e-12
    with pytest.raises(ValueError, match="one-hot"):
        ECE(uni, uni, weighted=False)


# ============================ 场景 4：校准前后改善（记录 before/after 与 Δ） ============================


def test_fit_records_ece_improvement_before_after():
    """spec『校准前后改善』：真温 3.0 的过自信数据 → 报告含 before/after/Δ 且 Δ≤0、after<before。"""
    n, k = 2000, 5
    z, p_true = _overconfident_rows(n=n, k=k, seed=0)
    labels = _sample_labels(p_true, seed=0)           # 硬标签按真实条件分布抽样 → 最优温度回收 3.0
    rep = fit_temperature(z, labels, qtypes=["choice"] * n)
    fit = rep.by_type["choice"]

    assert not fit.pooled and fit.n == n
    assert 2.5 <= fit.temperature <= 3.5, f"拟合温度应回收真温 3.0，实得 {fit.temperature}"
    assert fit.ece_before > 0.05, f"原始过自信应有可见误差，实得 {fit.ece_before}"
    assert fit.ece_after < fit.ece_before, (fit.ece_before, fit.ece_after)
    assert fit.ece_delta <= 0.0 and fit.improved              # DoD③ 的验收式
    assert fit.ece_after == pytest.approx(fit.ece_before + fit.ece_delta, abs=1e-12)
    assert fit.nll_after <= fit.nll_before                    # 网格含 T=1.0 ⇒ 最小值不可能更差
    # as_dict 是 p1-09/p1-10 的取数面：before/after/Δ 与 pooled 标注都要在里面
    d = rep.as_dict()
    assert set(d) >= {"temperatures", "pooled_temperature", "by_type", "grid", "min_samples", "total_samples"}
    assert set(d["by_type"]["choice"]) >= {"ece_before", "ece_after", "ece_delta", "pooled", "n"}
    assert d["by_type"]["choice"]["ece_delta"] <= 0.0
    assert d["temperatures"] == rep.temperatures


# ============================ 网格与入参守卫（A1 附带） ============================


def test_fit_grid_bounds_and_make_grid_defaults():
    """网格契约：默认 [0.2,5.0] 步长 0.05（97 点、端点精确、1.0 在格上）；非法网格即拒。"""
    grid = make_grid()
    assert grid == DEFAULT_GRID and len(grid) == 97
    assert grid[0] == 0.2 and grid[-1] == 5.0 and 1.0 in grid
    assert all(round(b - a, 10) == 0.05 for a, b in itertools.pairwise(grid))
    assert make_grid(0.3, 1.0, 0.2) == (0.3, 0.5, 0.7, 0.9, 1.0)   # 不整除时补右端点且不越过
    with pytest.raises(ValueError):
        make_grid(0.0, 1.0)
    with pytest.raises(ValueError):
        make_grid(5.0, 0.2)

    z, p_true = _overconfident_rows(n=250, k=3, seed=2)
    lab = _sample_labels(p_true, seed=2)
    # 自定义网格（可越出默认区间，由 p1-07 的 clamp 兜底）：拟合温度必须是格点之一
    rep = fit_temperature(z, lab, grid=(0.7, 1.0, 2.0), qtypes=["choice"] * 250)
    assert rep.by_type["choice"].temperature in (0.7, 1.0, 2.0)
    assert rep.grid == (0.7, 1.0, 2.0)
    with pytest.raises(ValueError):
        fit_temperature(z, lab, grid=(2.0, 0.5, 1.0))             # 非升序 → 并列取小规则失效，拒
    with pytest.raises(ValueError):
        fit_temperature(z, lab, grid=(0.0, 1.0))                  # 非正温度
    with pytest.raises(ValueError, match="宽度"):
        fit_temperature(z, [[1.0, 0.0]] * 250)                    # 目标宽度与候选数不符
    with pytest.raises(ValueError, match="qtypes"):
        fit_temperature(z, lab, qtypes=["choice"] * 249)


def test_fit_temperature_soft_targets_and_no_qtype_path():
    """无 qtypes 的退化路径（单桶 "all"）+ soft 目标可用性（质量参与加权）。"""
    z, p_true = _overconfident_rows(n=300, k=4, seed=3)
    rep = fit_temperature(z, p_true)                    # 默认 qtypes=None → 单桶 "all"
    assert list(rep.temperatures) == ["all"] and rep.by_type["all"].n == 300
    assert 2.5 <= rep.by_type["all"].temperature <= 3.5
    assert rep.by_type["all"].ece_after < rep.by_type["all"].ece_before
    # 列表式参差输入（各题候选数不同）也必须能拟：3/2/4 宽混排
    rows = [[2.0, 0.0, 0.0], [1.0, 3.0], [0.5, 0.1, 0.1, 0.9]]
    soft = [[0.8, 0.1, 0.1], [0.4, 0.6], [0.25, 0.25, 0.25, 0.25]]
    small = fit_temperature(rows * 70, soft * 70, qtypes=["choice"] * 210)
    assert small.by_type["choice"].n == 210 and not small.by_type["choice"].pooled
    with pytest.raises(ValueError):
        fit_temperature(rows * 70, soft * 70 + [1], qtypes=["choice"] * 210)   # 数量不符


# ============================ 场景 5：分桶随机基线 ============================


def test_bucket_baselines_20_and_3_are_separate_buckets():
    """spec『二十选一基线』：k=20 → 0.05，与 k=3（0.333…）分列不混报。"""
    b = bucket_baselines()
    assert b["choice"][20].value == pytest.approx(0.05)
    assert b["choice"][3].value == pytest.approx(1 / 3)
    assert b["choice"][20].k == 20 and b["choice"][3].k == 3     # 桶身份由 (qtype, k) 共同决定
    assert b["choice"][20].qtype == b["choice"][3].qtype == "choice"
    assert b["choice"][20].value != b["choice"][3].value, "两档必须不同值，否则就是混成一桶"
    assert all(bl.metric == "accuracy" for bl in b["choice"].values())
    assert {20, 3} <= set(b["choice"]), "默认桶集需同时含 k=3 与 k=20（供报告分列）"


def test_bucket_baselines_formulas_brute_force():
    """基线公式全枚举核对（防"背错公式"）：choice=1/k、noul=0.5、score MAE=(k²−1)/(3k)。"""
    b = bucket_baselines()
    assert b["noul"][2].value == 0.5 and set(b["noul"]) == {2}, "noul 只有二选一这一个合法桶"
    assert b["noul"][2].metric == "accuracy"
    for k, bl in b["choice"].items():
        # 随机选一个、真值均匀：命中概率 = 全枚举里的对角占比（自证而非套公式）
        hits = sum(1 for pick in range(k) for truth in range(k) if pick == truth)
        assert bl.value == pytest.approx(hits / (k * k)), k
    for k, bl in b["score"].items():
        diffs = [abs(i - j) for i in range(k) for j in range(k)]     # 均匀瞎猜两档位的偏差全集
        assert bl.metric == "mae"
        assert bl.value == pytest.approx(sum(diffs) / len(diffs)), k
        assert bl.value == pytest.approx((k * k - 1) / (3 * k)), k
    big = bucket_baselines(ks={"choice": [20], "score": [20]})       # 二十档与三档不同尺度
    assert big["choice"][20].value == pytest.approx(0.05)
    assert big["score"][20].value == pytest.approx(399 / 60)


def test_bucket_baselines_custom_ks():
    """ks 形态：None=默认全套；序列=各 qtype 共用；dict=按 qtype 指定（noul 仍锁 k=2）。"""
    b = bucket_baselines(ks=[3, 20])
    assert set(b["choice"]) == {3, 20} and set(b["score"]) == {3, 20} and set(b["noul"]) == {2}
    b2 = bucket_baselines(ks={"choice": [2, 4, 8], "score": [5]})
    assert set(b2) == {"choice", "score"} and set(b2["choice"]) == {2, 4, 8}
    assert b2["choice"][4].value == pytest.approx(0.25) and b2["score"][5].value == pytest.approx((25 - 1) / 15)
    with pytest.raises(ValueError):
        bucket_baselines(ks=[0])


if __name__ == "__main__":  # 无 pytest 时的手动模式（与 tests/test_ci_smoke.py 同构）
    import sys

    fails = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"  ok  {_name}")
            except Exception as _e:  # noqa: BLE001
                fails += 1
                print(f" FAIL {_name}: {type(_e).__name__}: {_e}")
    sys.exit(1 if fails else 0)
