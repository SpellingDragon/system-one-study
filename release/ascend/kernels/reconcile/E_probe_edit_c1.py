"""E_probe_edit_c1.py — 一次性 splice 工具（agent E，缺陷二）：把 `tests/test_longctx.py` C1 段里
"位等判据 + KNOWN-ENV 叙述"整块换成**平台无关结构判据**，并留一份可复跑的判据原文（本文件的
`NEW_BLOCK` 就是落盘内容，读它即知判据长什么样，不必再翻 diff）。

为什么需要一个脚本而不是手改：本 agent 的纪律要求"每步附可复跑命令"，把替换写成
`assert old in s` 的定点替换，可证明**只动了这一块**（改后 `git diff --stat` 只报该文件）。

用法（release 目录下）：
    .venv/bin/python ascend/kernels/reconcile/E_probe_edit_c1.py --check     # 只验位点，不写
    .venv/bin/python ascend/kernels/reconcile/E_probe_edit_c1.py --apply     # 落盘替换
"""
from __future__ import annotations

import pathlib
import sys

TARGET = pathlib.Path(__file__).resolve().parents[3] / "tests" / "test_longctx.py"

OLD_BLOCK = '''# ================================================================== C1 因果滑窗
# KNOWN-ENV（R-P1-4 归因，先证后判，**故不 skip**）：本用例是纯计算（compare_routes 固定 seed
# 合成张量，零资产依赖），host arm64 绿、ubuntu x86 CPU 轮红在 `kernel_vs_masked_max_abs == 0.0`
# 的**逐位相等**断言上——fp32 matmul/softmax 归约顺序随 BLAS/指令集而变，位等口径本身平台敏感，
# 属数值环境伪影而非滑窗路由回归。放宽容差动的是 sys1 侧验收口径，超出本变更白名单，
# 保留红 + 台账入档（docs/ci_baseline_triage.md §3），由编排者定夺。
def test_sliding_routes_agree_with_mask_reference():
    rep = LA.compare_routes(200, 32, chunk=48)
    assert rep["kernel_allclose"] and rep["local_allclose"]
    assert rep["kernel_vs_masked_max_abs"] == 0.0, "内核带状路与掩码路必须逐位相同"
    assert rep["local_vs_masked_max_abs"] < 1e-5
'''

NEW_BLOCK = '''# ================================================================== C1 因果滑窗
# 判据重建（缺陷二，2026-10-11 agent E；证据链 release/ascend/kernels/reconcile/D_determinism.md）。
# 旧口径 `kernel_vs_masked_max_abs == 0.0` 作废的事实依据 [实测]：带状 `route="kernel"` 在 CPU 上
# **三道闸必回落** `attn_sw_kernel._eager()`（① `backends.active_backend(cpu)` 只放 mps 设备过关；
# ② `attn_sw_mps.plan` 第一门槛要 fp16，而 compare_routes 造的是 fp32；③ band_len=chunk+window-1=79
# 不被 BLOCK_Q=16 整除）——所以这条断言比的从来不是"内核 vs 掩码"，而是**两份 eager 实现、不同归约
# 长度（200 列 vs 79 列）之间的位等**：构造上无保证（D_determinism §1.2/§3.1），ubuntu 6 run 2 红 4 绿
# 就是它的表现，host 侧姊妹路（只差一个 M 形状）在同机已有 10/20 组形状出末位差。
# 新口径：主判据是**平台无关的结构判据**三条（A1 非零可见位集合逐格相等、A2 argmax 一致、
# A3 同进程自比），量级容差只作附列。**这不是"放宽容差当止血"**：结构判据对目标错误类（滑窗方向
# 抄反 / 分块边界切错 / 窗宽 off-by-one）的守护力严格强于位等：写反方向改变的是**非零位集合**，
# 位等把它混在"末位差"里——末位差导致误报（ubuntu 2 红），而一旦为消误报放宽容差又漏报真错。
# 结构判据把"方向/边界"与"量级"分成两面各判各的，误报与漏报同时收住（负例见
# test_sliding_visible_set_criterion_detects_reversed_window，R14 守卫必须真响）。
# 位等仍合法，但只用在两侧同形状同分派处（A3 自比；以及 test_sliding_short_sequence_… 的 torch.equal）。
ROUTE_TOL = 1e-5                                       # 仓内既有量级口径（同文件姊妹断言同款）
PROBE_SEQ, PROBE_WIN, PROBE_CHUNK, PROBE_HEADS = 48, 12, 16, 2
PROBE_SEED = 20261011


def _probe_qkv() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """造 v=I 的探针三元组：`P@I` 每行只剩一个非零加数 ⇒ 出口最后一维**逐位就是**该路的权重 P。

    白话：想知道"这一路到底看了谁"，就把内容换成单位矩阵——加权平均出来的数字直接就是份额本身，
    非零位集合因此是可逐格比的结构事实，而不是随末位差漂移的数值。抽取过程与 BLAS 分派无关
    （只有一个非零加数，任何加法次序都得同一个数）。
    """
    gen = torch.Generator(device="cpu").manual_seed(PROBE_SEED)
    q = torch.randn(PROBE_HEADS, PROBE_SEQ, PROBE_SEQ, generator=gen)
    k = torch.randn(PROBE_HEADS, PROBE_SEQ, PROBE_SEQ, generator=gen)
    v = torch.eye(PROBE_SEQ).expand(PROBE_HEADS, PROBE_SEQ, PROBE_SEQ).contiguous()
    return q, k, v


def _route_weights(route: str | None) -> torch.Tensor:
    """(heads,T,T) 的注意力权重 P：route=None 走掩码路，"kernel"/"local" 走带状路（CPU 上都是 eager）。"""
    q, k, v = _probe_qkv()
    if route is None:
        return LA.attention_masked(q, k, v, window=PROBE_WIN).out
    return LA.attention_band(q, k, v, PROBE_WIN, chunk=PROBE_CHUNK, route=route).out


_ROUTE_LABEL = {None: "掩码路",
                "kernel": '带状 route="kernel" 路（CPU 上执行体是 attn_sw_kernel._eager）',
                "local": '带状 route="local" 路（就地 eager）'}


def test_sliding_routes_agree_with_mask_reference():
    """eager 带状路与掩码参考"看得见地一致"：可见位集合 + argmax + 同进程自比（量级只作附列）。"""
    from sys1.kernels import backends

    # 措辞事实钉：本用例的带状 "kernel" 路在 CPU 上必回落 _eager()，据此把措辞写成 eager 带状路。
    assert backends.active_backend("cpu") == backends.TORCH_EAGER, (
        'CPU 上带状 route="kernel" 现在不再回落 _eager() 了：本用例的措辞与判据前提要一起改——'
        "真进内核时另立『内核路 vs 掩码路』判据（fp16 + band_len 整除 BLOCK_Q），别沿用这里的 eager 口径")

    truth = LA.window_band(PROBE_SEQ, PROBE_WIN)[0, 0]          # (T,T)：与各路出口 (heads,T,T) 同秩可展
    weights = {route: _route_weights(route) for route in (None, "kernel", "local")}
    ref = weights[None]

    # 结构判据为何平台无关：两条前提自证（不是装饰，破了就说明这套判据在该数据上会随平台失稳）
    min_visible = float(ref[truth.expand_as(ref)].min())
    assert min_visible > 1e-6, (
        f"可见位最小权重 {min_visible:.3e} 逼近 fp32 下溢：『非零位』集合会有平台敏感的 0/非0 抖动，"
        "换种子或换窗宽把权重拉开再判")
    ordered = ref.sort(dim=-1).values
    margin = float((ordered[..., -1] - ordered[..., -2]).min())
    probe_diff = max(float((w - ref).abs().max()) for w in weights.values())
    gap_floor = 100 * max(probe_diff, 10 * torch.finfo(torch.float32).eps)
    assert margin > gap_floor, (
        f"argmax 判据的前提破了：top1-top2 最小间距 {margin:.3e} 与本平台实测跨路末位差 {probe_diff:.3e} "
        f"只差不到两个数量级（门槛 {gap_floor:.3e}）——近并列时 argmax 会随末位差翻转，那才是平台敏感面。"
        "先换 seed/形状把间距拉开再用这条判据，别把它当位等那样蒙着用")

    # A1 主判据：非零可见位集合逐格相等（顺带卡死"未来位严格 0"与"对角自见不除 NaN"）
    for route, w in weights.items():
        assert torch.equal(w != 0, truth.expand_as(w)), (
            f"{_ROUTE_LABEL[route]} 的非零可见位集合与 window_band({PROBE_SEQ}, {PROBE_WIN}) 不逐格相等："
            "滑窗方向 / 分块边界 / 窗宽 off-by-one 之一被写改了（位等抓不住这类错，结构判据抓得住）")
        assert bool((w.triu(1) == 0).all()), f"{_ROUTE_LABEL[route]} 泄漏未来位：因果半边破了"
        assert bool((w.diagonal(dim1=-2, dim2=-1) > 0).all()), (
            f"{_ROUTE_LABEL[route]} 对角不自见 → 首行可能被整行涂黑，softmax 除出 NaN")

    # A2 主判据：argmax 一致（spec 原生验收：滑窗正确性 → 选项 argmax 一致；平台无关）
    base_idx = ref.argmax(-1)
    for route, w in weights.items():
        assert torch.equal(w.argmax(-1), base_idx), (
            f"{_ROUTE_LABEL[route]} 与掩码参考 argmax 不一致：间距 {margin:.3e} ≫ 容差 {ROUTE_TOL:.0e}，"
            "翻转即真回归（不是末位差能解释的）")

    # A3 主判据：同进程自比——每条路与**自身重跑**位等（分配历史扰动后）。这才是可平台无关钉住的
    # "确定性"契约：同形状、同分派、同线程档 ⇒ 位等有构造保证；不成立就是真非确定，与 BLAS/ISA 无关。
    for route, first in weights.items():
        junk = torch.randn(3, 517, 733) @ torch.randn(3, 733, 611)   # 扰动分配历史/指针对齐
        del junk
        assert torch.equal(_route_weights(route), first), (
            f"{_ROUTE_LABEL[route]} 同进程两次结果位等不同 → 真·非确定回归（本条不许随平台让步）")

    # 附列（不作主判据）：被测口径 (T=200, W=32, chunk=48) 的量级差，只卡仓内既有口径
    rep = LA.compare_routes(200, 32, chunk=48)
    assert rep["kernel_allclose"] and rep["local_allclose"]
    assert rep["kernel_vs_masked_max_abs"] <= ROUTE_TOL, (
        f"eager 带状 kernel 路与掩码路量级差 {rep['kernel_vs_masked_max_abs']!r} 超出仓内既有口径 "
        f"{ROUTE_TOL}（两侧归约长度 200 vs 79 天然不同 → 位等非产品不变量，D_determinism.md §3.1）")
    assert rep["local_vs_masked_max_abs"] <= ROUTE_TOL
    print(f"[sliding-criterion] probe(T={PROBE_SEQ},W={PROBE_WIN},chunk={PROBE_CHUNK},H={PROBE_HEADS}) "
          f"pattern_eq=True argmax_eq=True self_bitwise_eq=True | 附列 max_abs kernel="
          f"{rep['kernel_vs_masked_max_abs']!r} local={rep['local_vs_masked_max_abs']!r} "
          f"min_visible={min_visible:.3e} argmax_margin={margin:.3e} probe_diff={probe_diff:.3e} gap_floor={gap_floor:.3e}")


def test_sliding_visible_set_criterion_detects_reversed_window():
    """负例自证（R14：守卫必须真响）：A1 那条集合逐格判据对"方向抄反/窗宽 off-by-one"必须判否。"""
    truth = LA.window_band(PROBE_SEQ, PROBE_WIN)[0, 0]
    idx = torch.arange(PROBE_SEQ)
    delta = idx[:, None] - idx[None, :]
    reversed_band = (delta <= 0) & (delta > -PROBE_WIN)         # 把"只许回头看"抄成"只许向前看"
    assert not torch.equal(reversed_band, truth), "负例构造失败：反向带与正向带逐格相同（探针形状退化了）"
    w = _route_weights(None)
    assert not torch.equal(w != 0, reversed_band.expand_as(w)), (
        "A1 判据对『滑窗方向写反』不响 → 它只是个摆设，换掉的位等判据不成立")
    over = LA.window_band(PROBE_SEQ, PROBE_WIN + 1)[0, 0]        # 窗宽 off-by-one（多带一格）
    wl = _route_weights("local")
    assert not torch.equal(wl != 0, over.expand_as(wl)), (
        "A1 判据对『窗宽 off-by-one』不响 → 分块边界那类错就没人抓得住")
'''


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    s = TARGET.read_text()
    hits = s.count(OLD_BLOCK)
    print(f"[locate] {TARGET.name}: OLD 位点命中 {hits} 处；NEW 已落盘 {s.count(NEW_BLOCK)} 处")
    if hits != 1:
        print("[skip] 位点不唯一或已被替换——不动文件")
        return 1
    if mode == "--apply":
        TARGET.write_text(s.replace(OLD_BLOCK, NEW_BLOCK))
        print(f"[apply] 已替换 {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
