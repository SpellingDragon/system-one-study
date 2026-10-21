"""甲路 P2-prep：ascend 可微层与混合栈一步冒烟的 host-cpu 验收件。

【做什么】把 `ascend/autograd_asc.py` 与 `ascend/train_step.py` 钉在两条硬判据上：
① `test_one_step_reduces_loss`——target=cpu 跑完整一步（前向→backward→Adam.step→
再前向），loss1 < loss0 且全部参数 grad 非 None；② `test_gradcheck_through_wrapper`
——用 fp64 数值微分对 `attn_sw / readout / gdn_conv / add_ln` 四个门面跑
`torch.autograd.gradcheck`，覆盖 `_AttnSwFn` 与 `_GdnConvFn` 这两条**入口件无反向、
本波靠闭式兜底**的反向公式，同时把 `_AddLnFn` 走 fp64 同口径分支时残差直传 + LN
闭式回账的合并逻辑一并验一遍。

【怎么做】① 判据一律"看真输出"——`test_one_step_reduces_loss` 直接调 `run(target="cpu")`
并解包 `(l0, l1, missing)`，缺 None 或 loss 不降都当场炸；② gradcheck 走 fp64：门面
`_ReadoutFn / _AttnSwFn / _AddLnFn / _GdnConvFn` 在 fp64 输入下都改走"同口径纯 torch"
分支（避免内核里硬编码的 fp32 累加器截位），保证反向公式与数值微分逐位可比；③
`_GdnFn / _LoraFn` 的 fp64 反向本波**不叠加 gradcheck**：前者内核已在 test_ascend_gradcheck
里拿自动微分尺子对拍（`gdn_backward[cpu]` blocker 在册），后者是 gemm 两件的组合、与
`_LinearFn` 同一条闭式，重复覆盖不带来新证据。**登记不假绿**（R14）。

【为什么】此前 ascend 域里"能算 loss 反着回来"这件事从没被端到端跑通过；单件反向对拍
是有的（test_ascend_gradcheck 20 绿），但"若干件串起来、loss 真的下降、每个可训参数
都拿到梯度"这条接线正确性没有独立证据。本件就是把这条证据钉成可执行文档。被否方案一：
把 gradcheck 塞进 test_ascend_gradcheck.py 里——那个文件专攻"内核数值"（fp32 对 fp32
参考），混进 fp64 反向公式验会让两件事的失败原因互相掩盖；被否方案二：只写 train_step
不写 gradcheck——`_AttnSwFn/_GdnConvFn` 的闭式反向前向同件写入，一处笔误就"看起来 loss
也会降"（因为梯度是"从错的反向里来的、又被同一个错的前向用掉"，自洽地骗过下降判据）。
"""
from __future__ import annotations

import pytest
import torch

from ascend import autograd_asc as ops
from ascend.train_step import run as train_run

#: 本文件全部用例统一走 CPU target（Ascend 卡窗另开，翻 `--target ascend` 即可）
TARGET = "cpu"


# --------------------------------------------------------------------------- 一步下降
def test_one_step_reduces_loss():
    """target=cpu：一步 Adam 更新后 loss 严格下降，且全部参数 grad 非 None（接线到位）。

    白话：`train_step.run` 已经做了"任一参数 grad=None 就抛"的硬检查；能返回就代表
    梯度一路串到底都接上了。loss 判据要**更新后再前向一次**才算，光看梯度量级不算
    ——小的 loss 未必是"更新走了对的方向"，也可能只是初始化就小。
    """
    l0, l1, missing = train_run(target=TARGET, device=torch.device("cpu"), lr=1e-2)
    assert not missing, f"以下参数 grad=None：{missing}"
    assert l1 < l0, f"一步未下降：loss0={l0} → loss1={l1}"


def test_train_step_cli_shape():
    """CLI 判决行必须能被抓到：`P2-STEP-<target> loss0=<x> loss1=<y>` 的三件套格式。

    白话：这一条不测数值，只测"人能不能一眼看懂"——把 run 输出重定向、抓 stdout 首行、
    按前缀切开对齐。日后接编排层（`apply-orchestration` 四查）就靠这条把格式钉住。
    """
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train_run(target=TARGET, device=torch.device("cpu"), lr=1e-2, verbose=True)
    line = buf.getvalue().strip().splitlines()[-1]
    assert line.startswith("P2-STEP-cpu "), f"判决行前缀不对：{line!r}"
    for key in ("loss0=", "loss1=", "reduced=", "params_grad_ok="):
        assert key in line, f"判决行缺字段 {key}：{line!r}"


# --------------------------------------------------------------------------- 反向公式（fp64 gradcheck）
def _lin_gradcheck_inputs(seq=8, m=8, n=8, k=8):
    """linear 门面用的 fp64 输入：A(M,K)、W(N,K)、bias(N)；W/A 都做 requires_grad。"""
    torch.manual_seed(0)
    A = torch.randn(m, k, dtype=torch.float64, requires_grad=True)
    W = torch.randn(n, k, dtype=torch.float64, requires_grad=True)
    b = torch.randn(n, dtype=torch.float64, requires_grad=True)
    return A, W, b


def _attn_inputs(heads=2, seq=12, dim=8, window=6):
    torch.manual_seed(1)
    q = torch.randn(heads, seq, dim, dtype=torch.float64, requires_grad=True)
    k = torch.randn(heads, seq, dim, dtype=torch.float64, requires_grad=True)
    v = torch.randn(heads, seq, dim, dtype=torch.float64, requires_grad=True)
    return q, k, v


def _add_ln_inputs(rows=16, dim=8):
    torch.manual_seed(2)
    x = torch.randn(rows, dim, dtype=torch.float64, requires_grad=True)
    r = torch.randn(rows, dim, dtype=torch.float64, requires_grad=True)
    g = torch.randn(dim, dtype=torch.float64, requires_grad=True)
    b = torch.randn(dim, dtype=torch.float64, requires_grad=True)
    return x, r, g, b


def _readout_inputs(row_count=10, dim=8, picks=6):
    torch.manual_seed(3)
    rows = torch.randn(row_count, dim, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([0, 3, 7, 3, 9, 1][:picks], dtype=torch.int64)
    return rows, ids


def _conv_inputs(channels=6, length=10, kernel=4):
    torch.manual_seed(4)
    x = torch.randn(channels, length, dtype=torch.float64, requires_grad=True)
    w = torch.randn(channels, kernel, dtype=torch.float64, requires_grad=True) * 0.3
    b = torch.randn(channels, dtype=torch.float64, requires_grad=True) * 0.1
    return x, w, b


# ---- gradcheck 门面：attn_sw / readout / gdn_conv / add_ln（**入口件无反向或走 fp64 同口径**）

def test_gradcheck_through_wrapper_attn_sw():
    """`_AttnSwFn` 的 fp32 闭式反向 = softmax-反 + 因果滑窗：fp64 数值微分对齐。

    白话：入口件里没有 attn 反向模具，本波反向靠**手推的闭式**（ds=p·(dp−Σdp·p)）。
    闭式最容易"看起来对，其实差一步链式法则"——fp64 gradcheck 是唯一能把它钉死的尺子；
    数值微分不看公式，只看扰动前后的差。
    """
    q, k, v = _attn_inputs()
    out = torch.autograd.gradcheck(
        lambda a, b, c: ops.attn_sw(a, b, c, window=6, out_dtype=torch.float64, target=TARGET),
        (q, k, v), eps=1e-6, atol=1e-4, rtol=1e-3,
    )
    assert out, "attn_sw 闭式反向与数值微分不一致"


def test_gradcheck_through_wrapper_readout():
    """`_ReadoutFn` fp64 分支：gather/scatter 是精确搬运，反向数值微分零误差对齐。

    白话：读出=按行号抽取，退回=按同一张表加回原表。本波里"接线最纯粹"的一件——
    如果连这条都过不了，就说明 Function 参数顺序或 None 占位有错。
    """
    rows, ids = _readout_inputs()
    out = torch.autograd.gradcheck(
        lambda r: ops.readout(r, ids, out_dtype=torch.float64, target=TARGET),
        (rows,), eps=1e-6, atol=1e-5, rtol=1e-3,
    )
    assert out, "readout 反向与数值微分不一致"


def test_gradcheck_through_wrapper_gdn_conv():
    """`_GdnConvFn`：入口件无 conv_backward，本波 fp32 闭式借 torch 图回传——fp64 对齐。

    白话：短卷积的反向在 dialect home 里没登记入口，闭式那一路（pad→conv1d→silu 现场
    重算 + torch 自动微分）是**接线正确性**的关键：写反了 pad 方向或忘了 silu 的链式，
    loss 一步照样降，但梯度会指歪；gradcheck 是唯一能当场抓住的路径。
    """
    x, w, b = _conv_inputs()
    out = torch.autograd.gradcheck(
        lambda a, c, e: ops.gdn_conv(a, c, e, out_dtype=torch.float64, target=TARGET),
        (x, w, b), eps=1e-6, atol=1e-4, rtol=1e-3,
    )
    assert out, "gdn_conv 闭式反向与数值微分不一致"


def test_gradcheck_through_wrapper_add_ln():
    """`_AddLnFn`：LN 闭式回账 + 上游残差直传的合并（只走 y 时的 dh_next=None 分支）。

    白话：本件返回 (y, h)，训练里"只用 y"（h 只作为残差旁路）是最常见用法；此时
    autograd 传进 backward 的 dh_next 是 None——这条分支如果没兜住，反向会炸；
    如果误当"上游直传 0"和"残差旁路"是同一件事，梯度会算重；fp64 gradcheck 把两种
    错都能抓出来。
    """
    x, r, g, b = _add_ln_inputs()

    def f(x_, r_, g_, b_):
        y, _h = ops.add_ln(x_, r_, g_, b_, eps=1e-5, out_dtype=torch.float64, target=TARGET)
        return y  # 只用 y，让 dh_next 走 None 分支

    assert torch.autograd.gradcheck(f, (x, r, g, b), eps=1e-6, atol=1e-4, rtol=1e-3)


# ---- 组合面：一次前向里同时用 readout + attn_sw，验门面之间可串（跨件接线）
def test_wrappers_compose_in_one_graph():
    """两个门面串起来跑一次 gradcheck：attn_sw → readout，证明**门面之间**也能接。

    白话：单件 gradcheck 只证明"这一件内部对"；两件串起来才证明"输出的 grad_fn 类型
    能被下一件的输入接住"——`Function` 的输出如果不小心被 mark 成 non-differentiable，
    下一件的图就断了，单件测试看不出来。
    """
    torch.manual_seed(5)
    heads, seq, dim, row_count = 1, 8, 8, 8
    q = torch.randn(heads, seq, dim, dtype=torch.float64, requires_grad=True)
    k = torch.randn(heads, seq, dim, dtype=torch.float64, requires_grad=True)
    v = torch.randn(heads, seq, dim, dtype=torch.float64, requires_grad=True)
    ids = torch.arange(row_count, dtype=torch.int64)

    def f(a, b, c):
        o = ops.attn_sw(a, b, c, window=seq, out_dtype=torch.float64, target=TARGET)
        # 把 attn 输出 reshape 成 (row_count, dim) 再喂 readout（跨件接线）
        flat = o.reshape(seq * heads, dim)
        # readout 需 rows 行数 ≥ ids.max()+1
        flat = flat[:row_count]
        return ops.readout(flat, ids, out_dtype=torch.float64, target=TARGET)

    assert torch.autograd.gradcheck(f, (q, k, v), eps=1e-6, atol=1e-4, rtol=1e-3)


# --------------------------------------------------------------------------- 反向缺失硬门（防假绿）
def test_grad_none_would_fail_train_run():
    """把某个门面的反向手工"拔线"，train_run 应抛 RuntimeError 而不是悄悄通过。

    白话：R14 反假绿的最硬一条——如果 grad 缺失时不炸，loss 也可能"看起来降了"（因为
    Adam 对 grad=None 的参数干脆不动，其它参数把 loss 拽下来了）。本例把 `_LinearFn.backward`
    的返回全替换成 None，看主循环有没有抓住。
    """
    orig_bwd = ops._LinearFn.backward
    try:
        def _broken(ctx, dC):
            n = 5  # A, W, bias, out_dtype, target
            return tuple([None] * n)
        ops._LinearFn.backward = staticmethod(_broken)
        with pytest.raises((RuntimeError, Exception)) as exc_info:
            train_run(target=TARGET, device=torch.device("cpu"), lr=1e-2, verbose=False)
        msg = str(exc_info.value)
        assert "grad=None" in msg or "反向接线未到位" in msg, f"报错未点明 grad 缺失：{msg}"
    finally:
        ops._LinearFn.backward = staticmethod(orig_bwd.__func__ if hasattr(orig_bwd, "__func__") else orig_bwd)


# --------------------------------------------------------------------------- lora / rope / linear 反向覆盖说明
# 这三件的 fp64 gradcheck 本波不叠加：
#   · `_LoraFn` 的反向是 `lora_asc.backward` = gemm 两件的组合，与 `_LinearFn` 同源；
#     `test_ascend_gradcheck::test_lora_cpu_backward_matches_autograd` 已用 fp32 自动微分交叉。
#   · `_RopeFn` 的反向 = 负角回转，`test_ascend_gradcheck::test_rope_cpu_negative_angle_roundtrip_is_identity`
#     已把"前向→反向→回到原件"这条自反性钉成断言（比 gradcheck 更硬）。
#   · `_LinearFn` 的反向 = 前向件喂转置 + dW 方言件；`test_ascend_gradcheck::test_gemm_bwd_dw_cpu_matches_torch_ruler`
#     已对 dW 独立尺子对齐。
# 若日后需要 fp64 三件叠加，直接扩本文件，不动 kernel。
