"""p2-13 R2 对拍测试：七类方言件在 cpu target 上逐件对齐 fp32 torch 参考（ascend-only 用例如打 skip）。

【做什么】对 linear（前向 + dW 反向）、rope、add_ln、letter_readout、attn_sw、GDN、LoRA 逐件
交回一份"内核结果 vs 独立 torch 尺子"的最大偏差断言，并额外验三件事：① 内核**真发射**（编译
产物表里出现该件的 key，且没被登记成阻塞点）；② 同一形状换批次**只编译一次**（compile_count
不变）；③ 语义不变量（rope 的值槽不碰与负角往返恒等、attn 的不可见权重严格为 0、GDN 的反向
边界被如实标成 partial）。
【怎么做】① 全部用例显式传 `target="cpu"`，走 TileLang 的 C 后端在 CPU 上做**语义**对拍——这
是昇腾卡被占用期间的唯一可行验收路（数值语义与后端无关，后端差异留给云端 C1）；② 尺子优先
复用 P1 冻结件 `sys1/testing/torch_ref`（gemm_ref / rope_ref / add_ln_ref / attn_sw_ref），P1
没有对应件的（letter_readout、GDN、LoRA）在本文件里另写一份**独立**的 torch 实现，不去调方言
件自带的 `_eager` 回退——否则回退路径自己给自己打分，缺陷会被两处同源掩盖；③ 位宽一律 fp32
进 fp32 出，容差 rtol=1e-4/atol=1e-5，只覆盖归约顺序差，不放过真错；④ 需要昇腾硬件的用例
统一打 `@pytest.mark.npu` 并在缺卡时 skip，函数名故意不含 "cpu"，因此
`-k cpu` 这条验收命令不会把它们算进通过数。
【为什么】被否方案一：只测 `forward` 的输出数值——那会让"回退到 torch 也算过"的假绿混进来
（R14），所以每件都追加 `_assert_kernel_ran` 这一硬证据；被否方案二：用 `_eager` 当尺子——
`_eager` 是生产回退路径，测试必须站在它外面才能同时抓到"内核错"和"回退也错"；被否方案三：
在测试里探测 torch_npu 后自动切换 target——同一条用例换后端等于换被测物，失败时无法归因，
故后端由参数写死，昇腾侧另开用例。
"""
import math

import pytest
import torch

from ascend.kernels import (
    add_ln_kernel,
    ascend_env,
    attn_sw_kernel,
    gemm_bwd_dw_kernel,
    gemm_kernel,
    gdn_kernel,
    letter_readout_kernel,
    lora_kernel,
    rope_kernel,
)
from sys1.testing.torch_ref import (
    add_ln_ref,
    attn_sw_ref,
    gemm_ref,
    rope_angle_tables,
    rope_ref,
)

#: 本文件全部用例统一走 CPU 语义对拍（昇腾实编留给云端 C1）
TARGET = "cpu"
#: fp32 对 fp32 的容差：只容忍归约顺序与低精度中间量的差别，不容忍算式错
TOL = {"rtol": 1e-4, "atol": 1e-5}


def _assert_kernel_ran(fragment: str) -> None:
    """硬证据：该件在 cpu target 上确实编译过，并且没被记进阻塞账（防"回退冒充通过"）。

    白话：光看结果对不对不够，还得看这台机器是不是真的开了模具跑过——只在架子上找得到
    这件的产物、并且它不在"没走通"的小账本上，才算它真的干过活。
    """
    keys = [k for k in ascend_env.compiled_keys() if f"[{TARGET}]" in k and fragment in k]
    assert keys, f"未见 {fragment} 的编译产物，实际键 = {sorted(ascend_env.compiled_keys())}"
    blocked = [k for k in ascend_env.blockers() if fragment in k]
    assert not blocked, f"{fragment} 被登记为阻塞并回退了：{blocked}"


def _npu_ready() -> bool:
    """昇腾硬件与 torch_npu 是否齐备（缺任一即让 ascend-only 用例 skip）。"""
    try:
        import torch_npu  # noqa: F401  只为触发设备注册，本身不被使用
    except ImportError:
        return False
    return bool(getattr(torch, "npu", None) is not None and torch.npu.is_available())


# --------------------------------------------------------------------------- linear
def test_gemm_cpu_matches_torch_ref():
    """前向乘加：与 P1 gemm_ref 逐元素对齐（无激活 / relu / 带偏置三种口径）。"""
    torch.manual_seed(1)
    a = torch.randn(33, 32)          # 行数故意不整除块高 16，验边界守卫
    w = torch.randn(16, 32)          # N/K 取 16/32：落在 BLOCK_LADDER 的支持窗内
    bias = torch.randn(16)
    for act, use_bias in (("none", False), ("none", True), ("relu", True)):
        b = bias if use_bias else None
        got = gemm_kernel.forward(a, w, b, act, torch.float32, TARGET)
        want = gemm_ref(a, w, b, act, torch.float32)
        torch.testing.assert_close(got, want, **TOL)
    _assert_kernel_ran("gemm[")


def test_gemm_cpu_reuses_single_compilation():
    """同一形状的第二个批次必须命中缓存：compile_count 不涨才是"同一份产物服务多批"。"""
    torch.manual_seed(2)
    w = torch.randn(16, 32)
    x1, x2 = torch.randn(8, 32), torch.randn(64, 32)
    base = ascend_env.compile_count()
    first = gemm_kernel.forward(x1, w, None, "none", torch.float32, TARGET)
    mid = ascend_env.compile_count()
    second = gemm_kernel.forward(x2, w, None, "none", torch.float32, TARGET)
    assert ascend_env.compile_count() == mid, "M 轴是动态维，换批不应该再开一次模具"
    assert mid >= base
    torch.testing.assert_close(first, gemm_ref(x1, w, None, "none", torch.float32), **TOL)
    torch.testing.assert_close(second, gemm_ref(x2, w, None, "none", torch.float32), **TOL)
    assert second.shape == (64, 16)


def test_gemm_bwd_dw_cpu_matches_torch_ruler():
    """权重梯度 dW = dYᵀ @ A：与 torch 直写对齐，并覆盖"多批累加"这一常见误写。"""
    torch.manual_seed(3)
    dy = torch.randn(33, 16)          # 行数 33 不整除块宽，正是要验的动态轴
    a = torch.randn(33, 32)
    got = gemm_bwd_dw_kernel.backward(dy, a, out_dtype=torch.float32, target=TARGET)
    torch.testing.assert_close(got, dy.T @ a, **TOL)
    assert got.shape == (16, 32)
    _assert_kernel_ran("gemm_dw[")


# --------------------------------------------------------------------------- rope
def test_rope_cpu_matches_torch_ref_and_keeps_value_slot():
    """rope 前向：与 P1 rope_ref 对齐；值槽（slot 2）必须逐位不动，且交回同一个对象。"""
    torch.manual_seed(4)
    tokens, heads, dim = 33, 2, 16
    cos, sin = rope_angle_tables(tokens, dim // 2)
    qkv = torch.randn(tokens, 3, heads, dim)
    original = qkv.clone()               # 尺子站在内核外面：拿没被动过的原件算参考
    got = rope_kernel.forward(qkv, cos.float(), sin.float(), rope_kernel.DEFAULT_SLOTS, TARGET)
    assert got is qkv, "rope 的口径是就地改写并交回同一个盒子"
    want = rope_ref(original, cos.float(), sin.float(), rope_kernel.DEFAULT_SLOTS)
    torch.testing.assert_close(got, want, **TOL)
    torch.testing.assert_close(got[:, 2], original[:, 2], rtol=0, atol=0)
    _assert_kernel_ran("rope[")


def test_rope_cpu_negative_angle_roundtrip_is_identity():
    """负角反向必须把前向转回去：转两遍要回到原件（旋转的逆就是 sin 取负）。"""
    torch.manual_seed(5)
    tokens, heads, dim = 24, 3, 16
    cos, sin = rope_angle_tables(tokens, dim // 2)
    original = torch.randn(tokens, 3, heads, dim)
    buf = original.clone()
    rope_kernel.forward(buf, cos.float(), sin.float(), rope_kernel.DEFAULT_SLOTS, TARGET)
    assert not torch.equal(buf, original), "先证明前向真的动了数据，否则往返恒等是假绿"
    rope_kernel.backward(buf, cos.float(), sin.float(), rope_kernel.DEFAULT_SLOTS, TARGET)
    torch.testing.assert_close(buf, original, **TOL)


# --------------------------------------------------------------------------- add_ln
def test_add_ln_cpu_matches_torch_ref_forward():
    """前向 (y, h)：与 P1 add_ln_ref 对齐；h 必须是 fp32 残差流，不许被降位宽。"""
    torch.manual_seed(6)
    rows, dim = 33, 32
    x = torch.randn(rows, dim)
    res = torch.randn(rows, dim)
    weight = torch.randn(dim)
    bias = torch.randn(dim)
    y, h = add_ln_kernel.forward(x, res, weight, bias, add_ln_kernel.DEFAULT_EPS,
                                 torch.float32, TARGET)
    y_ref, h_ref = add_ln_ref(x, res, weight, bias, add_ln_kernel.DEFAULT_EPS, torch.float32)
    torch.testing.assert_close(y, y_ref, **TOL)
    torch.testing.assert_close(h, h_ref, **TOL)
    assert h.dtype == torch.float32, "残差流降位宽是被明令禁止的口径"
    _assert_kernel_ran("add_ln[")


def test_add_ln_cpu_closed_form_backward_matches_autograd():
    """闭式反向 (dx, dW, db)：拿自动微分当独立尺子，验证"两个均值"那条闭式没抄错。"""
    torch.manual_seed(7)
    rows, dim = 33, 32
    eps = add_ln_kernel.DEFAULT_EPS
    h = torch.randn(rows, dim, requires_grad=True)
    weight = torch.randn(dim, requires_grad=True)
    bias = torch.randn(dim, requires_grad=True)
    dy = torch.randn(rows, dim)
    y, _ = add_ln_ref(h, torch.zeros_like(h), weight, bias, eps, None)
    grad_h, grad_w, grad_b = torch.autograd.grad(y, (h, weight, bias), dy)
    dx, dg, db = add_ln_kernel.backward(h.detach(), weight.detach(), dy, eps, TARGET)
    torch.testing.assert_close(dx, grad_h, **TOL)
    torch.testing.assert_close(dg, grad_w, **TOL)
    torch.testing.assert_close(db, grad_b, **TOL)
    _assert_kernel_ran("ln_bwd[")


# --------------------------------------------------------------------------- letter_readout
def test_letter_readout_cpu_gather_matches_indexing():
    """读出 out[i] = rows[ids[i]]：允许重复行号，逐位精确（纯搬运不该有数值误差）。"""
    torch.manual_seed(8)
    rows = torch.randn(12, 16)
    ids = torch.tensor([3, 0, 7, 3, 11, 5], dtype=torch.int64)
    got = letter_readout_kernel.forward(rows, ids, torch.float32, TARGET)
    torch.testing.assert_close(got, rows[ids], rtol=0, atol=0)
    assert got.shape == (6, 16)
    _assert_kernel_ran("gather[")


def test_letter_readout_cpu_scatter_matches_index_add_and_autograd():
    """退回大表：重复行号必须累加而不是覆盖，并与自动微分对同一次读出的梯度交叉。"""
    torch.manual_seed(9)
    row_count, dim = 12, 16
    ids = torch.tensor([3, 0, 7, 3, 11, 5], dtype=torch.int64)
    dy = torch.randn(int(ids.numel()), dim)
    got = letter_readout_kernel.backward(row_count, ids, dy, TARGET)
    want = torch.zeros(row_count, dim)
    want.index_add_(0, ids, dy)
    torch.testing.assert_close(got, want, **TOL)
    assert float(got[3].sub(dy[0] + dy[3]).abs().max()) == 0.0, "同一行退两次就是加两次"

    big = torch.randn(row_count, dim, requires_grad=True)
    loss = big[ids]                                  # 用 autograd 反查同一条搬运链
    grad = torch.autograd.grad(loss, big, dy)[0]
    torch.testing.assert_close(got, grad, **TOL)
    _assert_kernel_ran("scatter_add[")


# --------------------------------------------------------------------------- attn_sw
def test_attn_sw_cpu_matches_torch_ref():
    """因果滑窗注意力：与 P1 attn_sw_ref 交叉（序列长度故意不整除块宽）。"""
    torch.manual_seed(10)
    heads, seq, dim, window = 2, 33, 16, 8
    q = torch.randn(heads, seq, dim)
    k = torch.randn(heads, seq, dim)
    v = torch.randn(heads, seq, dim)
    got = attn_sw_kernel.forward(q, k, v, window, dim ** -0.5, torch.float32, TARGET)
    want = attn_sw_ref(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), window,
                       dim ** -0.5, None)[0]
    torch.testing.assert_close(got, want, **TOL)
    assert got.shape == q.shape
    _assert_kernel_ran("attn_sw[")


def test_attn_sw_cpu_keeps_invisible_weights_exactly_zero():
    """可见性不变量：窗口外与未来的位置，分到的权重必须**严格**是 0（不是"接近 0"）。"""
    torch.manual_seed(11)
    heads, seq, dim, window = 2, 24, 16, 6
    q = torch.randn(heads, seq, dim)
    k = torch.randn(heads, seq, dim)
    scale = dim ** -0.5
    wt = attn_sw_kernel.forward_weights(q, k, window, scale)
    idx = torch.arange(seq)
    allow = (idx[None, :] <= idx[:, None]) & (idx[:, None] - idx[None, :] < window)
    hidden = wt.masked_select(~allow)                # 只挑"不该看见的"那些格
    assert float(hidden.abs().max()) == 0.0, f"不可见位置偷分了：{float(hidden.abs().max())}"
    live = wt.masked_select(allow)
    assert live.numel() > 0 and float(live.min()) > 0.0, "可见位置的份额必须真的分到了东西"
    torch.testing.assert_close(wt.sum(-1), torch.ones(heads, seq), **TOL)


def test_attn_sw_cpu_full_window_equals_causal_attention():
    """窗口开到不截断时，必须退化成普通因果注意力（这是"窗口方向没抄反"的最硬证据）。"""
    torch.manual_seed(12)
    heads, seq, dim = 1, 20, 16
    q = torch.randn(heads, seq, dim)
    k = torch.randn(heads, seq, dim)
    v = torch.randn(heads, seq, dim)
    got = attn_sw_kernel.forward(q, k, v, seq, dim ** -0.5, torch.float32, TARGET)
    score = q.to(torch.float32) @ k.to(torch.float32).transpose(-1, -2) * dim ** -0.5
    causal = torch.full_like(score, float("-inf")).triu_(1)
    want = torch.softmax(score + causal, dim=-1) @ v.to(torch.float32)
    torch.testing.assert_close(got, want, **TOL)


# --------------------------------------------------------------------------- GDN
def _gdn_torch_ruler(q, k, v, g, beta):
    """独立写一遍 Gated DeltaRule 递推（fp32，矩阵写法），只当尺子，不 import 方言件的回退。

    白话：五步一循环——黑板按比例淡掉、照键把已有内容读出来当预期、真值减预期乘力度补进去、
    照查询读一遍就是这一步的输出。这里用 matmul 而不是逐格乘加，故意与内核写法不同源。
    """
    heads, seq, dk = q.shape
    dv = v.size(-1)
    state = torch.zeros(heads, dk, dv, dtype=torch.float32)
    out = torch.empty(heads, seq, dv, dtype=torch.float32)
    for t in range(seq):
        state = state * g[:, t].exp().unsqueeze(-1)
        pred = torch.bmm(state.transpose(1, 2), k[:, t].unsqueeze(-1)).squeeze(-1)
        upd = (v[:, t] - pred) * beta[:, t].unsqueeze(-1)
        state = state + torch.bmm(k[:, t].unsqueeze(-1), upd.unsqueeze(1))
        out[:, t] = torch.bmm(state.transpose(1, 2), q[:, t].unsqueeze(-1)).squeeze(-1)
    return out


def test_gdn_cpu_forward_matches_independent_recursion():
    """GDN 前向（本域最重件）：对独立 torch 递推；带前导批维也要走同一条路。"""
    torch.manual_seed(13)
    heads, seq, dk, dv = 2, 24, 8, 8
    q = torch.nn.functional.normalize(torch.randn(heads, seq, dk), p=2, dim=-1)
    k = torch.nn.functional.normalize(torch.randn(heads, seq, dk), p=2, dim=-1)
    v = torch.randn(heads, seq, dv)
    g = -torch.rand(heads, seq, dk) * 0.2            # 对数衰减：负值 = 每步淡掉一点
    beta = torch.rand(heads, seq) * 0.9 + 0.05
    want = _gdn_torch_ruler(q, k, v, g, beta)
    got = gdn_kernel.forward(q, k, v, g, beta, torch.float32, TARGET)
    torch.testing.assert_close(got, want, **TOL)
    batched = gdn_kernel.forward(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                                 g.unsqueeze(0), beta.unsqueeze(0), torch.float32, TARGET)
    torch.testing.assert_close(batched[0], want, **TOL)
    assert batched.shape == (1, heads, seq, dv)
    _assert_kernel_ran("gdn[")


def test_gdn_backward_ascend_kernelized_cpu_ruler_grads_correct():
    """反向边界（P1-4 合流后）：昂腾反向已内核化（BWD_STATUS=kernelized），但 cpu target 故意走 torch 尺子并登记 gdn_backward[cpu] 阻塞；回退那份梯度要经得起自动微分。

    同时守两件事：接口面如实反映"昂腾已内核化、cpu 留尺子路"的边界；且回退路梯度正确——
    云端真机换 ascend 件时数值口径不漂移。
    """
    assert gdn_kernel.BWD_STATUS == "kernelized", "P1-4 合流起 gdn 反向已内核化（attempts/F 六梯）"
    torch.manual_seed(14)
    heads, seq, dk, dv = 2, 12, 8, 8
    q = torch.nn.functional.normalize(torch.randn(heads, seq, dk), p=2, dim=-1)
    k = torch.nn.functional.normalize(torch.randn(heads, seq, dk), p=2, dim=-1)
    v = torch.randn(heads, seq, dv)
    g = -torch.rand(heads, seq, dk) * 0.2
    beta = torch.rand(heads, seq) * 0.9 + 0.05
    dy = torch.randn(heads, seq, dv)

    leaves = [t.clone().requires_grad_(True) for t in (q, k, v, g, beta)]
    want = torch.autograd.grad(_gdn_torch_ruler(*leaves), leaves, dy)
    got = gdn_kernel.backward(q, k, v, g, beta, dy, TARGET)
    assert len(got) == 5
    for name, got_one, want_one in zip(("dq", "dk_", "dv", "dg", "dbeta"), got, want):
        torch.testing.assert_close(got_one, want_one, **TOL, msg=f"{name} 与自动微分不一致")
    assert any("gdn_backward" in key for key in ascend_env.blockers()), \
        "cpu 回退必须留下机器可见的阻塞记录（昂腾路已内核化、不走本 cpu 测）"


# --------------------------------------------------------------------------- LoRA
def test_lora_cpu_apply_matches_direct_expression():
    """旁路增量 delta = s·(x @ Aᵀ) @ Bᵀ：与直写表达式对齐，并验 base 叠加。"""
    torch.manual_seed(15)
    m, n, kk, r, s = 12, 32, 16, 16, 0.5   # n/kk/r 全取块宽整数倍，旁路两次乘加才真进内核
    x = torch.randn(m, kk)
    a = torch.randn(r, kk) * 0.2
    b = torch.randn(n, r) * 0.2
    delta = (x @ a.T) @ (b * s).T
    got = lora_kernel.apply(x, a, b, s, None, torch.float32, TARGET)
    torch.testing.assert_close(got, delta, **TOL)
    base = torch.randn(m, n)
    torch.testing.assert_close(lora_kernel.apply(x, a, b, s, base, torch.float32, TARGET),
                               base + delta, **TOL)
    _assert_kernel_ran("gemm[")                 # 组合件不产自己的 key，复用 linear 两件


def test_lora_cpu_backward_matches_autograd():
    """反向三条链 (dx, dA, dB)：与同一条旁路的自动微分交叉。"""
    torch.manual_seed(16)
    m, n, kk, r, s = 12, 32, 16, 16, 0.5
    x = torch.randn(m, kk)
    a = torch.randn(r, kk) * 0.2
    b = torch.randn(n, r) * 0.2
    dy = torch.randn(m, n)
    leaves = [t.clone().requires_grad_(True) for t in (x, a, b)]
    xl, al, bl = leaves
    want = torch.autograd.grad((xl @ al.T) @ (bl * s).T, leaves, dy)
    got = lora_kernel.backward(dy, x, a, b, s, TARGET)
    for name, got_one, want_one in zip(("dx", "dA", "dB"), got, want):
        torch.testing.assert_close(got_one, want_one, **TOL, msg=f"{name} 与自动微分不一致")
    _assert_kernel_ran("gemm_dw[")              # dA/dB 两条链确实是 dW 件算出来的


def test_lora_cpu_merge_equals_apply_and_scales_once():
    """推理期合并 s·B @ A：与旁路结果等价（合并只乘一次尺度，不许重复打折）。"""
    torch.manual_seed(17)
    m, n, kk, r, s = 12, 32, 16, 16, 0.5
    x = torch.randn(m, kk)
    a = torch.randn(r, kk) * 0.2
    b = torch.randn(n, r) * 0.2
    d_w = lora_kernel.merge(a, b, s, torch.float32, TARGET)
    torch.testing.assert_close(d_w, b @ a * s, **TOL)
    assert d_w.shape == (n, kk)
    torch.testing.assert_close(lora_kernel.apply(x, a, b, s, None, torch.float32, TARGET),
                               x @ d_w.T, **TOL)


def test_gemm_cpu_falls_back_outside_block_window():
    """支持窗外的形状：必须**数值仍对**且**不新增编译产物**（走回退是设计，不是失败）。

    白话：这块模具只认得 16/32/64 的块宽；碰上 10 这种尺寸时不硬凑，改用手算，但账目不能错。
    这条用例把"什么时候进内核、什么时候回退"钉成可执行文档，免得日后误以为回退=内核坏了。
    """
    torch.manual_seed(19)
    a = torch.randn(12, 10)           # K=10 不在 BLOCK_LADDER 的任何块宽整除关系里
    w = torch.randn(10, 10)
    before = set(ascend_env.compiled_keys())
    got = gemm_kernel.forward(a, w, None, "none", torch.float32, TARGET)
    torch.testing.assert_close(got, gemm_ref(a, w, None, "none", torch.float32), **TOL)
    assert set(ascend_env.compiled_keys()) == before, "窗外形状不该留下编译产物"


# --------------------------------------------------------------------------- 契约与形状
def test_entry_contract_rejects_bad_shapes_cpu():
    """入口判据要主动报错而不是静默回退：形状不对必须 raise，免得脏数据流到下游。"""
    with pytest.raises(ValueError):
        gemm_kernel.forward(torch.randn(4, 8), torch.randn(5, 7), None, "none",
                            torch.float32, TARGET)
    with pytest.raises(ValueError):
        add_ln_kernel.forward(torch.randn(4, 8), torch.randn(4, 9), torch.randn(8),
                              torch.randn(8), 1e-5, torch.float32, TARGET)
    with pytest.raises(ValueError):
        attn_sw_kernel.forward(torch.randn(2, 4, 8), torch.randn(2, 4, 8),
                               torch.randn(2, 4, 8), 0, None, torch.float32, TARGET)
    with pytest.raises(ValueError):
        # beta 需为 (heads, seq)=(2, 4)，这里故意给 (2, 5)
        gdn_kernel.forward(torch.randn(2, 4, 8), torch.randn(2, 4, 8), torch.randn(2, 4, 8),
                           torch.randn(2, 4, 8), torch.randn(2, 5), torch.float32, TARGET)


def test_target_switch_cpu_and_ascend_are_whitelisted():
    """`--target ascend|cpu` 开关口径：缺省读环境变量，显式传值必须优先，且两值都能归一。"""
    assert ascend_env.normalize_target("cpu") == "cpu"
    assert ascend_env.normalize_target("ascend") == "ascend"
    assert ascend_env.normalize_target(None) in ("cpu", "ascend")
    with pytest.raises(ValueError):
        ascend_env.normalize_target("cuda")


@pytest.mark.skipif(not _npu_ready(), reason="需要 Ascend NPU（torch_npu + 真卡），云端 C1 验")
@pytest.mark.npu
def test_attn_sw_ascend_target_semantics_on_npu():
    """ascend-only：昇腾 target 的前向必须与已验的 CPU 语义逐元素对齐（本地不实编，留云端 C1）。

    函数名故意不含 "cpu"，因此 `-k cpu` 这条验收命令不会把缺卡跳过的用例算进通过数。
    """
    heads, seq, dim, window = 2, 32, 16, 8
    torch.manual_seed(18)
    q = torch.randn(heads, seq, dim).npu()
    k, v = torch.randn(heads, seq, dim).npu(), torch.randn(heads, seq, dim).npu()
    want = attn_sw_kernel.forward(q.cpu(), k.cpu(), v.cpu(), window, dim ** -0.5,
                                 torch.float32, TARGET)
    got = attn_sw_kernel.forward(q, k, v, window, dim ** -0.5, torch.float32, "ascend")
    rel = float((got.float().cpu() - want).abs().max() / want.abs().max())
    assert rel < 2 ** -8, f"昇腾与 CPU 语义偏离过大：{rel}（口径阈值 1/256，bf16 级容差）"
    assert math.isfinite(rel)
