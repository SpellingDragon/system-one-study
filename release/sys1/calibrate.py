"""校准数学域（p1-05）：类型温度拟合、ECE、分桶随机基线。

【做什么】给模型报出的"胆量分"找一个合适的缩放倍数，使它对选项的把握程度尽量说到做到；
量出"把握几成、结果几成"之间的差距（ECE）；并算出各类题目瞎猜能得多少分当作及格线，
防止一个笨模型蒙混过关。

【怎么做】倍数用网格穷举而非求导：把允许的区间 [0.2, 5.0] 按 0.05 切成 97 格（见 make_grid），
逐格把每道题的原始分值同除以该数、再算负对数似然，取最小者；候选数不同的题按宽度分组批量计算，
全程 float64 保证同输入必同输出。某 qtype 有效样本 <200 时不单独定倍数，改用全体样本拟出的
池化倍数，并在 FitReport 对应条目里如实写 pooled=True（小样本硬拟出来的只是噪声，报告不掩盖）。
ECE 走 top-label 口径：取每行最大那一项的数值当"把握"，按它落进 15 个等宽区间归桶，
桶内比较"平均把握"与"实际命中份额"，再按桶占比加权求和；若目标本身是带概率质量的分布，
走 weighted=True 分支——命中份额取目标压在获胜项上的质量、桶权重取该行总质量。
分桶基线是纯组合恒等式：单选 1/k、非真即假 0.5、评分档均匀猜测的平均偏差 (k²−1)/(3k)。

【为什么】网格法胜过梯度优化（L-BFGS 之类）：不多一套学习率/迭代数超参、结果可复现，
且与 StartLux-Decision/finetune/calibrate.py 同构便于逐数对照。倍数选"同除一个正数"，
是因为它对分值序列是严格单调变换——谁排第一不会因为同除正数而改变（含并列也原样并列），
这一点写进单测锁死而非注释里许愿。曾否掉两个方案：① 把倍数作用在已折算好的份额上
（等于对各候选做非均匀拉伸）——会翻答案，单调保证直接作废；② 样本不足也照每类单独拟——
把噪声当规律，跨次运行数字乱跳，故改为池化并在报告里标注。三函数互不依赖他域，
只用 torch/numpy 的数值内核（零依赖使本域可在 W0 与其他四域并发落地）。
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

# 契约常量：p1-07 的 clamp 区间与本域的网格边界 MUST 同源（改这里 = 同步改其 spec，工程红线）
GRID_LO = 0.2
GRID_HI = 5.0
GRID_STEP = 0.05
DEFAULT_MIN_SAMPLES = 200       # 低于此样本量的 qtype 一律池化（GUIDE §6 / design 规则）
DEFAULT_BINS = 15               # typed-decisions 上游口径：top-label ECE 固定 15 桶
_EPS = 1e-9                     # 浮点判等容差：one-hot 的"是不是 1"用绝对误差判

__all__ = [
    "DEFAULT_BINS",
    "DEFAULT_GRID",
    "DEFAULT_KS",
    "DEFAULT_MIN_SAMPLES",
    "ECE",
    "GRID_HI",
    "GRID_LO",
    "GRID_STEP",
    "Baseline",
    "FitReport",
    "TypeFit",
    "bucket_baselines",
    "fit_temperature",
    "make_grid",
]

# 一阶段各 qtype 的典型选项数（GUIDE §5：choice 3/4/5/…/20，score 星级与 0–9 档，noul 恒二选一）
DEFAULT_KS: dict[str, tuple[int, ...]] = {
    "choice": (2, 3, 4, 5, 6, 8, 10, 20),
    "noul": (2,),
    "score": (2, 3, 4, 5, 10),
}


def make_grid(lo: float = GRID_LO, hi: float = GRID_HI, step: float = GRID_STEP) -> tuple[float, ...]:
    """生成升序温度候选网格（默认 [0.2, 5.0] 步长 0.05 → 97 个点，首尾严格取 lo/hi）。

    白话：把允许的倍数范围切成一小格一小格排成队，回头逐格试一遍挑最好的——不搞连续求导，
    全靠穷举，所以同一份数据在任何人任何机器上跑出来的数都一模一样。

    末点处理：(hi-lo)/step 不整除时只铺到不超过 hi 的最后一格，再补一个 hi，保证区间右端点
    一定参与穷举（否则真实最优落在 hi 附近时会被网格系统性截断）。
    """
    # 显式拒非法参数：温度必须为正，否则 z/T 会翻掉单调性、整个域的前提崩掉
    if lo <= 0 or hi <= 0 or step <= 0:
        raise ValueError(f"网格端点与步长必须为正（got lo={lo}, hi={hi}, step={step}）")
    if hi < lo:
        raise ValueError(f"网格上界不得小于下界（lo={lo}, hi={hi}）")
    # 用地板除而不是四舍五入：round 会在 (hi-lo)/step 恰为 x.5 时多铺一格越过 hi
    n = int((hi - lo) // step)
    # round(...,10) 只为消掉 0.2+0.05*k 的二进制尾噪，让 T=1.0 这类关键刻度精确命中
    vals = [round(lo + i * step, 10) for i in range(n + 1) if lo + i * step <= hi + 1e-12]
    if vals[-1] < hi:
        vals.append(hi)
    return tuple(vals)


DEFAULT_GRID: tuple[float, ...] = make_grid()


@dataclass(frozen=True)
class TypeFit:
    """单个 qtype 的拟合结果；pooled/ece_* 字段是"报告不掩盖"的落点（design 池化规则）。"""

    qtype: str
    n: int
    temperature: float
    pooled: bool                # True = 样本不足、跟随全局池化温度
    nll_before: float           # T=1.0 的负对数似然
    nll_after: float            # 拟合温度的负对数似然（网格最小值）
    ece_before: float
    ece_after: float
    ece_delta: float            # after - before；验收要求 ≤ 0（GUIDE §6 校准轴）

    @property
    def improved(self) -> bool:
        """ECE 是否持平或改善（Δ≤0 即算合格，允许温度在噪声内不改结论）。

        白话：一句话结论——调完之后"说到做到"的程度有没有变差；没变差就算合格，
        因为有时数据本来就挺准，硬调反而只是跟着噪声晃。"""
        return self.ece_delta <= 0.0


@dataclass(frozen=True)
class FitReport:
    """fit_temperature 的返回：qtype→温度表 + 逐类明细 + 池化温度与网格元信息。"""

    temperatures: dict[str, float]      # 供 p1-07 直接写 decision_config.json 的表
    pooled_temperature: float           # 全局池化温度（样本不足类的归属）
    by_type: dict[str, TypeFit]
    grid: tuple[float, ...]
    min_samples: int
    total_samples: int

    @property
    def pooled_types(self) -> tuple[str, ...]:
        """被池化（样本不足、跟随大盘）的 qtype 清单——空元组表示逐类都独立拟了。

        白话：把"题太少、没资格自己定倍数、只能跟着全体走"的那些类点名列出来，
        空列表就说明每一类都是自己定的——不让读者误以为所有数字都同样可靠。"""
        return tuple(q for q, f in self.by_type.items() if f.pooled)

    def as_dict(self) -> dict[str, Any]:
        """摊平成可 JSON 落盘的报告（p1-09 写 run 记录、p1-10 calibration 轴直接消费）。

        白话：把上面这些数字拍成一个普通的文本清单，好塞进配置文件和实验记录里，
        事后不用读代码就能核对每一类用了多少倍数、前后把握差了多少。

        刻意不预舍入：浮点保真交调用方按显示需要 round(…, 4)（StartLux 参考实现在写盘时才舍）。
        """
        out = asdict(self)                    # 嵌套 dataclass（by_type）由 asdict 一并转 dict
        out["grid"] = list(self.grid)         # 转 list：tuple 虽 JSON 可序列化，但下游数值处理更认 list
        return out


@dataclass(frozen=True)
class Baseline:
    """一个 qtype×k 桶的随机基线；metric 随 qtype 变（choice/noul 看准确率，score 看平均偏差）。"""

    qtype: str
    k: int
    metric: str
    value: float


def _rows_of(x: Any, name: str) -> list[torch.Tensor]:
    """统一输入为 [1-D float64 行]，同时支持等宽矩阵与参差列表（选择题候选数天然不等宽）。"""
    if isinstance(x, torch.Tensor):
        t = x.detach()
        if t.dim() == 0:
            raise ValueError(f"{name} 不能是标量")
        if t.dim() == 1:
            t = t.unsqueeze(0)                # 单样本一维 → 一行，省去调用方手工 reshape
        if t.dim() != 2:
            raise ValueError(f"{name} 需为 [样本数, 候选数]，实际 {tuple(t.shape)}")
        return [row.to(torch.float64) for row in t]
    if isinstance(x, np.ndarray):
        arr = np.atleast_2d(x)
        return [_row_to_tensor(r, name) for r in arr]
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError(f"{name} 为空，无法计算")
        # 列表元素可为标量（硬标签）或变长序列（soft 目标 / 参差分值），逐个转换
        return [_row_to_tensor(item, name) for item in x]
    raise TypeError(f"{name} 需为 torch.Tensor / np.ndarray / 序列，实际 {type(x)}")


def _row_to_tensor(item: Any, name: str) -> torch.Tensor:
    """把单个元素转成一维 float64 张量（标量视作长度 1）；不用 stack 以容忍参差宽度。"""
    t = item if isinstance(item, torch.Tensor) else torch.as_tensor(item)
    return t.detach().reshape(-1).to(torch.float64)


def _value_rows(logits: Any, name: str) -> list[torch.Tensor]:
    """取原始分值行并做基本校验（有限性；float64 使网格搜索结果跨设备可复现）。"""
    rows = _rows_of(logits, name)
    for i, r in enumerate(rows):
        if r.numel() < 1 or not bool(torch.isfinite(r).all()):
            raise ValueError(f"{name} 第 {i} 行含 NaN/Inf 或为空")
    return rows


def _is_hard_label(v: Any) -> bool:
    """该元素是否是"第几个选项"式的整数标签（bool 与整值浮点也算，noul 常写成 0/1）。"""
    if isinstance(v, (bool, np.bool_)):
        return True
    if isinstance(v, (int, np.integer)):
        return True
    if isinstance(v, (float, np.floating)):
        return float(v).is_integer()
    return False


def _target_rows(targets: Any, widths: Sequence[int]) -> list[torch.Tensor]:
    """目标 → 概率质量行（float64）。整数标签转 one-hot；分布行保留原始质量、不做归一化。

    分布行保持原样是有意的：weighted 口径要靠"质量"本身做权重与命中份额，归一化会抹掉它。
    一维浮点张量按单样本分布解读（多样本请传二维张量或整数标签列），宽度校验会兜住误用。
    """
    n = len(widths)
    if isinstance(targets, torch.Tensor):
        t = targets.detach()
        if t.is_floating_point():
            rows = _rows_of(t, "targets")
        else:
            rows = [_hard_row(int(v), w) for v, w in zip(t.reshape(-1).tolist(), widths)]
    elif isinstance(targets, np.ndarray):
        arr = np.asarray(targets)
        if arr.ndim == 1 and arr.dtype.kind in ("i", "u", "b"):
            rows = [_hard_row(int(v), w) for v, w in zip(arr.reshape(-1).tolist(), widths)]
        else:
            rows = _rows_of(arr, "targets")
    elif isinstance(targets, (list, tuple)):
        if len(targets) != n:
            raise ValueError(f"targets 数量 {len(targets)} 与样本数 {n} 不符")
        hard = [v for v in targets if _is_hard_label(v)]
        if len(hard) == len(targets):
            rows = [_hard_row(int(v), w) for v, w in zip(targets, widths)]
        elif hard:
            raise ValueError("targets 不可硬标签与概率分布行混用（两口径会互相污染）")
        else:
            rows = [_row_to_tensor(v, "targets") for v in targets]
    else:
        raise TypeError(f"targets 需为整数标签或概率分布行，实际 {type(targets)}")

    if len(rows) != n:
        raise ValueError(f"targets 样本数 {len(rows)} 与 logits 样本数 {n} 不符")
    for i, (row, w) in enumerate(zip(rows, widths)):
        if row.numel() != w:
            raise ValueError(f"targets 第 {i} 行宽度 {row.numel()} 与 logits 候选数 {w} 不符")
        if bool((row < -_EPS).any()):
            raise ValueError(f"targets 第 {i} 行出现负质量")
    return rows


def _hard_row(label: int, width: int) -> torch.Tensor:
    """整数标签 → one-hot 质量行（和恰为 1，使 hard/weighted 两口径在此输入上收敛同一值）。"""
    if not 0 <= label < width:
        raise ValueError(f"标签 {label} 越界 [0, {width})")
    row = torch.zeros(width, dtype=torch.float64)
    row[label] = 1.0
    return row


def _by_width(rows: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """按候选数分桶堆成批量张量：一次 log_softmax 算一整桶，避免逐样本 Python 循环。"""
    grouped: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for z, t in rows:
        grouped.setdefault(z.numel(), []).append((z, t))
    return [(torch.stack([z for z, _ in v]), torch.stack([t for _, t in v])) for v in grouped.values()]


def _nll_at(rows: Sequence[tuple[torch.Tensor, torch.Tensor]], temp: float) -> float:
    """给定温度的平均负对数似然（soft 目标按行概率质量加权：质量小的行本就不该左右结论）。"""
    num = 0.0
    den = 0.0
    for Z, Tg in _by_width(rows):
        # log_softmax 而非 log(softmax)：数值稳定，温度很大时也不会出现 log(0)
        num += float((Tg * torch.log_softmax(Z / temp, dim=-1)).sum())
        den += float(Tg.sum())                # 分母是总质量而非常数样本数 → 支持非归一 soft 目标
    if den <= 0:
        raise ValueError("所有目标行的概率质量为 0，无法拟合")
    return -num / den


def _best_temperature(rows: Sequence[tuple[torch.Tensor, torch.Tensor]], grid: Sequence[float]) -> float:
    """网格穷举取 NLL 最小者；并列时取较小温度（min 返回首个最小 → 升序网格即最保守）。"""
    curve = [_nll_at(rows, t) for t in grid]
    return grid[min(range(len(curve)), key=lambda i: (curve[i], i))]


def fit_temperature(
    logits: Any,
    targets: Any,
    grid: Sequence[float] | None = None,
    qtypes: Sequence[str] | None = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    bins: int = DEFAULT_BINS,
) -> FitReport:
    """逐 qtype 在 NLL 网格上拟合标量温度；样本 <min_samples 的类池化到全局温度。

    白话：每一类题目各自找一个把胆量分同除的倍数，让"我几成把握"最贴"实际几成对"；
    题目太少的类不自己定夺，跟着全体算出来的那个数走，并在报告里写明它是跟来的。

    参数次序以 calibration spec 的 `fit_temperature(logits, targets, grid)` 为准（grid 第三），
    qtypes 作第四参；缺省视为单一桶 "all"（无类型信息时仍然可用）。温度表按单调性 MUST 不换
    argmax；逐类记录 nll/ece 的 before/after 与 Δ，供 GUIDE §6 校准轴与变更级 DoD③ 直接取用。
    """
    grid_vals = DEFAULT_GRID if grid is None else tuple(float(g) for g in grid)
    if not grid_vals:
        raise ValueError("grid 不能为空")
    if any(t <= 0 for t in grid_vals) or sorted(grid_vals) != list(grid_vals):
        raise ValueError("grid MUST 为升序正数序列（并列取最小温度依赖该约定）")
    if min_samples < 1:
        raise ValueError(f"min_samples MUST ≥1，实际 {min_samples}")

    z_rows = _value_rows(logits, "logits")
    t_rows = _target_rows(targets, [r.numel() for r in z_rows])
    n = len(z_rows)
    if n == 0:
        raise ValueError("fit_temperature 需要至少一个样本")

    if qtypes is None:
        keys = ["all"] * n                    # 退化路径：无类型标注时全体进同一桶 "all"
    else:
        keys = [str(q) for q in qtypes]
        if len(keys) != n:
            raise ValueError(f"qtypes 数量 {len(keys)} 与样本数 {n} 不符")

    pairs = list(zip(z_rows, t_rows))
    # 池化温度先在全体样本上拟一次：既是小样本类的归属，也是"全局趋势"的本身记录
    pooled_temp = _best_temperature(pairs, grid_vals)

    groups: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for key, pair in zip(keys, pairs):
        groups.setdefault(key, []).append(pair)

    by_type: dict[str, TypeFit] = {}
    temperatures: dict[str, float] = {}
    for qtype in sorted(groups):
        rows = groups[qtype]
        if len(rows) >= min_samples:
            temp, pooled = _best_temperature(rows, grid_vals), False
        else:
            temp, pooled = pooled_temp, True  # 样本不足：硬拟只是拟合噪声，跟随大盘并如实标注
        temperatures[qtype] = temp
        by_type[qtype] = _type_fit(qtype, rows, temp, pooled, bins)

    return FitReport(
        temperatures=temperatures,
        pooled_temperature=pooled_temp,
        by_type=by_type,
        grid=grid_vals,
        min_samples=min_samples,
        total_samples=n,
    )


def _type_fit(qtype: str, rows: Sequence[tuple[torch.Tensor, torch.Tensor]], temp: float,
              pooled: bool, bins: int) -> TypeFit:
    """组装单类明细：nll/ece 的 T=1 与 T=temp 两组数 + Δ（before 恒取未施加温度的 1.0）。"""
    t_rows = [t for _, t in rows]
    soft = _has_soft_mass(t_rows)             # 目标带质量 → 逐类 ECE 走质量口径
    before = _ece_rows([_probs_row(z, 1.0) for z, _ in rows], t_rows, bins, soft)
    after = _ece_rows([_probs_row(z, temp) for z, _ in rows], t_rows, bins, soft)
    return TypeFit(
        qtype=qtype, n=len(rows), temperature=temp, pooled=pooled,
        nll_before=_nll_at(rows, 1.0), nll_after=_nll_at(rows, temp),
        ece_before=before, ece_after=after, ece_delta=after - before,
    )


def _mass_on_top(t_rows: Sequence[torch.Tensor], top: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """取每行"目标压在获胜项上的质量"与"该行总质量"。

    等宽时一次 gather 批量算完；不等宽（同一 qtype 里三选与二十选并存是常态）退回逐行索引，
    不构造 0 填充矩阵——填充会污染总质量，进而污染 weighted 口径的桶权重。
    """
    if len({r.numel() for r in t_rows}) == 1:
        mat = torch.stack(list(t_rows))
        return mat.gather(1, top.unsqueeze(1)).squeeze(1), mat.sum(dim=1)
    mass_top = torch.stack([row[int(i)] for row, i in zip(t_rows, top.tolist())])
    return mass_top, torch.stack([row.sum() for row in t_rows])


def _probs_row(z: torch.Tensor, temp: float) -> torch.Tensor:
    """原始分值 → 该温度下的份额（float64，与 argmax 判读同源）。"""
    return torch.softmax(z / temp, dim=-1)


def _has_soft_mass(t_rows: Sequence[torch.Tensor]) -> bool:
    """探测目标是否带非 0/1 的概率质量（决定 ECE 走 hard 还是 weighted 口径）。"""
    return any(bool(((t > _EPS) & (t < 1.0 - _EPS)).any()) for t in t_rows)


# 函数名 ECE 由 calibration spec 锁定（与指标缩写一致），非风格疏忽
def ECE(
    probs: Any,
    targets: Any,
    bins: int = DEFAULT_BINS,
    *,
    weighted: bool,
) -> float:
    """top-label 口径的期望校准误差：bins 个等宽区间，按把握程度分堆后逐桶加权。

    白话：先看每道题里最有把握的那一项有多把握，照这个把握高低把题分到小格里；
    每格里比一比"平均报了几成把握"和"实际命中几成"，差多少就记多少，再按格子大小合起来。
    数字越小，说明它嘴上说几成就真有几成。

    bins 默认 15（上游 typed-decisions 口径）。weighted 是强制的关键字参数（design：两口径
    禁止混报）：False 走硬命中，目标 MUST 是 one-hot，否则报错提醒；True 走概率质量口径——
    命中份额取目标压在获胜项上的质量、桶权重取该行总质量，soft 目标 MUST 用它。
    """
    if bins < 1:
        raise ValueError(f"bins MUST ≥1，实际 {bins}")
    p_rows = _rows_of(probs, "probs")
    for i, r in enumerate(p_rows):
        if not bool(torch.isfinite(r).all()) or bool((r < -_EPS).any()) or bool((r > 1.0 + _EPS).any()):
            raise ValueError(f"probs 第 {i} 行非法（需为 [0,1] 内的有限数值）")
    t_rows = _target_rows(targets, [r.numel() for r in p_rows])
    return _ece_rows(p_rows, t_rows, bins, weighted)


def _ece_rows(p_rows: Sequence[torch.Tensor], t_rows: Sequence[torch.Tensor],
              bins: int, weighted: bool) -> float:
    """ECE 内核（fit 路径复用，绕开一次公共校验；p_rows 与 t_rows 必须同序等长）。"""
    conf = torch.stack([p.max() for p in p_rows])                     # top-label 置信
    top = torch.stack([p.argmax() for p in p_rows])                   # 并列取首位（与 argmax 约定一致）
    mass_top, row_mass = _mass_on_top(t_rows, top)

    if weighted:
        w, hit = row_mass.clamp(min=0.0), mass_top.clamp(min=0.0)     # 质量口径：权重与命中同源
    else:
        # 硬口径遇 soft 目标：把"部分命中"强行当 0 或 1 就是混报，宁可报错也不给可疑的数
        fractional = bool(((mass_top > _EPS) & (mass_top < 1.0 - _EPS)).any())
        off_unit = bool((row_mass - 1.0).abs().max() > 1e-6)
        if fractional or off_unit:
            raise ValueError(
                "ECE(weighted=False) 只接受 one-hot 目标：检测到带概率质量的目标分布。"
                "soft 目标请显式传 weighted=True（两口径禁止混报，见 p1-05 design.md）"
            )
        w = torch.ones_like(conf)
        hit = (mass_top >= 1.0 - _EPS).to(torch.float64)              # 命中=目标是否全押在获胜项

    total = float(w.sum())
    if total <= 0:
        raise ValueError("目标总概率质量为 0，ECE 无定义")
    idx = torch.clamp((conf * bins).to(torch.int64), min=0, max=bins - 1)  # conf==1.0 归最后一桶
    ece = 0.0
    for b in range(bins):
        sel = idx == b
        if not bool(sel.any()):
            continue
        wb = float(w[sel].sum())
        if wb <= 0:
            continue                                                  # 该桶无质量 → 不贡献误差
        c_bar = float((w[sel] * conf[sel]).sum() / wb)                 # 桶内加权平均把握
        a_bar = float((w[sel] * hit[sel]).sum() / wb)                  # 桶内加权实际命中
        ece += (wb / total) * abs(c_bar - a_bar)
    return ece


def bucket_baselines(ks: Sequence[int] | dict[str, Sequence[int]] | None = None) -> dict[str, dict[int, Baseline]]:
    """输出 {qtype: {k: Baseline}} 的分桶随机基线（choice 1/k、noul 0.5、score (k²−1)/(3k)）。

    白话：先把"闭眼乱选能得多少分"这条底线算清楚当及格线——三个里蒙一个有一分之一的机会，
    二十个里蒙一个只剩二十分之一，评分题乱打分的平均偏差也有固定算法。
    这两档必须分开列，否则蒙对三选一的成绩会被拿去冒充二十选一的合格线。

    参数 ks：None 用 DEFAULT_KS；序列=对全部 qtype 套用同一组 k；dict=按 qtype 指定。
    noul 恒为二选一：传入的其他 k 被忽略，保证它只有一个合法桶。
    """
    if isinstance(ks, dict):
        spec = {str(q): tuple(int(v) for v in seq) for q, seq in ks.items()}
    elif ks is None:
        spec = {q: tuple(seq) for q, seq in DEFAULT_KS.items()}
    else:
        spec = {q: tuple(int(v) for v in ks) for q in DEFAULT_KS}

    out: dict[str, dict[int, Baseline]] = {}
    for qtype in sorted(spec):
        buckets: dict[int, Baseline] = {}
        for k in sorted(set(spec[qtype])):
            if k < 1:
                raise ValueError(f"{qtype} 的桶宽 k MUST ≥1，实际 {k}")
            if qtype == "noul" and k != 2:
                continue                    # 非真即假只有二选一，别的 k 不属本域语义
            buckets[k] = Baseline(qtype=qtype, k=k, metric=_baseline_metric(qtype), value=_baseline_value(qtype, k))
        if qtype == "noul" and not buckets:
            buckets[2] = Baseline(qtype="noul", k=2, metric="accuracy", value=0.5)
        out[qtype] = buckets
    return out


def _baseline_metric(qtype: str) -> str:
    """该 qtype 用什么尺子对照基线：score 看平均偏差，其余看准确率（GUIDE §6 表）。"""
    return "mae" if qtype == "score" else "accuracy"


def _baseline_value(qtype: str, k: int) -> float:
    """随机参考值：choice=1/k、score=均匀猜测 MAE=(k²−1)/(3k)、noul=0.5。

    score 公式来历（写在这里防"背公式"）：X、Y 独立均匀取 {0,…,k−1}，
    E|X−Y| = (1/k²)·2·Σ_{d=1}^{k−1} d(k−d) = (k²−1)/(3k)，tests 用全枚举暴力核对同一值。
    """
    if qtype == "score":
        return (k * k - 1) / (3.0 * k)
    if qtype == "noul":
        return 0.5                          # k=2 的均匀准确率，与 1/2 同值但语义单列
    return 1.0 / k


if __name__ == "__main__":  # 极简自检：分桶基线逐桶分列（k=3 与 k=20 不同值即未混报）
    rep = bucket_baselines()
    print("choice:", {k: round(b.value, 4) for k, b in sorted(rep["choice"].items())})
    print("score :", {k: round(b.value, 4) for k, b in sorted(rep["score"].items())})
    print("noul  :", {k: b.value for k, b in rep["noul"].items()})
