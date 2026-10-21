"""torch 参考实现属性测试（p1-04 A1-A4，纯 CPU，无设备标记）。

【做什么】把四算子参考实现的口径逐条钉死：gemm 的 fp32 累加纪律、add_ln 的"残差流保 fp32"、
rope 的 rotate-half 与分表角度、attn_sw 的**因果半窗**（未来与窗口外的份额必须严格等于零）。
【怎么做】全部用构造数据 + 闭式性质（行和为一、配对模长不变、与 PyTorch 官方算子互校）来断言，
不依赖手写期望值表；同时把"错口径"也显式算一遍（例如逐 K 降位宽的朴素累加），用来证明
验收阈值不是随手挑的。
【为什么】这些参考实现是内核对拍唯一的生命线，它自己错了就全红转绿、绿转红都无从判断。
被否方案：只测"能跑不抛异常"——否，形状对但口径反（把因果写成双向）正是本项目最贵的事故。
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from sys1.testing.torch_ref import (
    DEFAULT_EPS,
    activate_ref,
    add_ln_ref,
    attn_sw_ref,
    attn_sw_weights_ref,
    causal_window_mask,
    gemm_ref,
    layer_stats_ref,
    rope_angle_tables,
    rope_ref,
    rotate_half_ref,
)

# ============================ A1 gemm ============================


def _fp64_gemm(A, W, bias=None):
    """用 fp64 独立算一遍"数学真值"的乘加部分（只在测试里当参照，不进实现）。"""
    acc = A.to(torch.float64) @ W.to(torch.float64).transpose(0, 1)
    if bias is not None:
        acc = acc + bias.to(torch.float64).unsqueeze(0)
    return acc


def _fp64_activation(acc: torch.Tensor, act: str) -> torch.Tensor:
    """在 fp64 域按参考实现同款定义过激活（gelu 同样用 tanh 近似式）。"""
    if act == "none":
        return acc
    if act == "relu":
        return torch.clamp(acc, min=0)
    if act == "gelu":
        return 0.5 * acc * (1.0 + torch.tanh(0.7978845608028654 * (acc + 0.044715 * acc.pow(3))))
    if act == "silu":
        return acc * torch.sigmoid(acc)
    raise ValueError(act)


def test_gemm_shapes_and_dynamic_m():
    """任意 M（含 1 / 7 / 16 三档）都直接可算：动态 M 的责任在内核侧，参考侧不许有对齐假设。"""
    W = torch.randn(32, 24, dtype=torch.float16)
    for m in (1, 7, 16):
        A = torch.randn(m, 24, dtype=torch.float16)
        out = gemm_ref(A, W)
        assert out.shape == (m, 32)
        assert out.dtype == torch.float16


@pytest.mark.parametrize("act", ["none", "relu", "gelu", "silu"])
def test_gemm_act_matches_fp64_truth(act):
    """四种激活的参考值与 fp64 真值之差须在 fp32 累加误差量级内（远小于 2e-2 验收线）。"""
    torch.manual_seed(0)
    A = torch.randn(8, 24, dtype=torch.float16)
    W = torch.randn(16, 24, dtype=torch.float16)
    b = torch.randn(16, dtype=torch.float32)
    got = gemm_ref(A, W, b, act=act, out_dtype=None)
    truth = _fp64_activation(_fp64_gemm(A, W, b), act)
    assert (got.to(torch.float64) - truth).abs().max().item() < 1e-4


def test_gemm_fp32_accumulation_beats_naive_fp16():
    """钉死"fp32 累加"这条纪律：逐 K 降位宽的朴素累加误差须显著大于本实现（本实现<1e-4）。"""
    torch.manual_seed(0)
    A = torch.randn(4, 256, dtype=torch.float16)
    W = torch.randn(4, 256, dtype=torch.float16)
    truth = _fp64_gemm(A, W)
    ref_err = (gemm_ref(A, W, act="none", out_dtype=None).to(torch.float64) - truth).abs().max().item()
    naive = torch.zeros(4, 4, dtype=torch.float16)
    for kk in range(256):  # 每加一项就把结果压回 fp16，模拟"累加器位宽不足"
        naive = (naive + A[:, kk : kk + 1] * W[:, kk].unsqueeze(0)).to(torch.float16)
    naive_err = (naive.to(torch.float64) - truth).abs().max().item()
    assert ref_err < 1e-4, f"fp32 累加本身就不该有 {ref_err} 的偏差"
    assert naive_err > 5 * max(ref_err, 1e-9), f"朴素 fp16 累加误差 {naive_err} 未体现精度地板"


def test_gemm_relu_clamps_and_bias_is_per_n():
    """偏置沿 N 轴广播（每列一个），relu 后不许出现负值。"""
    A = torch.ones(3, 4, dtype=torch.float16)
    W = torch.full((5, 4), -0.5, dtype=torch.float16)
    b = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
    got = gemm_ref(A, W, b, act="relu", out_dtype=None)
    assert torch.all(got >= 0)
    expect = torch.tensor([[-1.0, 0.0, 1.0, 2.0, 3.0]], dtype=torch.float32).expand(3, 5).clamp(min=0)
    assert torch.allclose(got, expect, atol=1e-6)


def test_gemm_rejects_bad_contract():
    """K 轴不匹配、bias 长度不等于 N、未知激活名都要当场报错（不许静默出错误形状）。"""
    A = torch.randn(2, 6, dtype=torch.float16)
    with pytest.raises(ValueError, match="K 轴"):
        gemm_ref(A, torch.randn(4, 5, dtype=torch.float16))
    with pytest.raises(ValueError, match="bias"):
        gemm_ref(A, torch.randn(4, 6, dtype=torch.float16), bias=torch.randn(3))
    with pytest.raises(ValueError, match="二维"):
        gemm_ref(torch.randn(6, dtype=torch.float16), torch.randn(4, 6, dtype=torch.float16))
    with pytest.raises(ValueError, match="未知激活"):
        activate_ref(A, act="tanh")


# ============================ A2 add_ln ============================


def test_add_ln_residual_stream_is_fp32():
    """硬约束：交给下一层的残差 h 必须是 fp32，且逐位等于 x+residual 的 fp32 和。"""
    torch.manual_seed(0)
    x = torch.randn(5, 64, dtype=torch.float16)
    res = torch.randn(5, 64, dtype=torch.float16)
    y, h = add_ln_ref(x, res, torch.ones(64), torch.zeros(64))
    assert h.dtype == torch.float32
    assert y.dtype == torch.float16
    assert torch.equal(h, x.to(torch.float32) + res.to(torch.float32))


def test_add_ln_matches_official_layer_norm():
    """与 PyTorch 官方层内整形互校（fp32 域），偏差须在 fp16 出口舍入量级内。"""
    torch.manual_seed(1)
    x = torch.randn(7, 128, dtype=torch.float16)
    res = torch.randn(7, 128, dtype=torch.float16)
    w = torch.randn(128, dtype=torch.float32)
    b = torch.randn(128, dtype=torch.float32)
    y, h = add_ln_ref(x, res, w, b)
    official = F.layer_norm(h, (128,), w, b, DEFAULT_EPS)
    assert (y.to(torch.float32) - official).abs().max().item() < 2e-2


def test_add_ln_stats_are_biased_variance():
    """统计口径钉死：有偏方差（除以 D），与官方 LayerNorm 一致而不是样本方差。"""
    torch.manual_seed(2)
    h = torch.randn(4, 50, dtype=torch.float32)
    mu, var = layer_stats_ref(h)
    assert torch.allclose(mu, h.mean(dim=-1, keepdim=True), atol=1e-6)
    assert torch.allclose(var, h.var(dim=-1, unbiased=False, keepdim=True), atol=1e-6)
    assert not torch.allclose(var, h.var(dim=-1, unbiased=True, keepdim=True), atol=1e-9)


def test_add_ln_unit_stats_after_transform():
    """weight=1/bias=0 时**整形输出**应"均值≈0、方差≈1"；多维输入形状须原样保留、残差不变。"""
    torch.manual_seed(3)
    x = torch.randn(2, 3, 96, dtype=torch.float16) * 4 + 7
    res = torch.randn(2, 3, 96, dtype=torch.float16)
    y, h = add_ln_ref(x, res, torch.ones(96), torch.zeros(96), out_dtype=None)
    assert y.shape == x.shape == h.shape and y.dtype == torch.float32
    assert torch.equal(h, x.to(torch.float32) + res.to(torch.float32))  # 残差流不被整形污染
    assert y.mean(dim=-1).abs().max().item() < 1e-4
    assert (y.var(dim=-1, unbiased=False) - 1.0).abs().max().item() < 1e-2


def test_add_ln_rejects_shape_mismatch():
    x = torch.randn(3, 8, dtype=torch.float16)
    with pytest.raises(ValueError, match="形状"):
        add_ln_ref(x, torch.randn(4, 8, dtype=torch.float16), torch.ones(8), torch.zeros(8))
    with pytest.raises(ValueError, match="weight/bias"):
        add_ln_ref(x, x, torch.ones(7), torch.zeros(8))


# ============================ A3 rope ============================


def test_rope_angle_tables_are_unit_circle_and_split():
    """分表形状 (seq, half)、fp32，且每格 cos²+sin²==1（角度唯一真源）。"""
    cos, sin = rope_angle_tables(seq_len=16, half=16)
    assert cos.shape == sin.shape == (16, 16)
    assert cos.dtype == sin.dtype == torch.float32
    assert (cos.pow(2) + sin.pow(2) - 1.0).abs().max().item() < 1e-6
    assert torch.allclose(cos[0], torch.ones(16), atol=1e-6)  # 位置 0 不转
    assert not torch.allclose(cos[5], cos[4], atol=1e-6)  # 位置递增角度递增


def test_rope_preserves_pair_norm_and_leaves_value_slot():
    """旋转是保模长的：每对 (前, 后) 的平方和不变；第三摞（值）一个数都不许动。"""
    torch.manual_seed(4)
    tokens, heads, dim = 12, 3, 32
    qkv = torch.randn(tokens, 3, heads, dim, dtype=torch.float16)
    cos, sin = rope_angle_tables(tokens, dim // 2)
    out = rope_ref(qkv, cos, sin)
    half = dim // 2
    for slot in (0, 1):
        before = qkv[:, slot].to(torch.float32)
        after = out[:, slot].to(torch.float32)
        n0 = before[..., :half].pow(2) + before[..., half:].pow(2)
        n1 = after[..., :half].pow(2) + after[..., half:].pow(2)
        assert (n0 - n1).abs().max().item() < 2e-2
    assert torch.equal(out[:, 2], qkv[:, 2])


def test_rope_matches_classical_full_table_formula():
    """与经典写法 out = x·cos_full + rotate_half(x)·sin_full 等价（分表↔全表的换算）。"""
    torch.manual_seed(5)
    qkv = torch.randn(9, 3, 2, 24, dtype=torch.float16)
    cos, sin = rope_angle_tables(9, 12)
    got = rope_ref(qkv, cos, sin)
    full_c = torch.cat([cos, cos], dim=-1).unsqueeze(1)  # (tokens, 1, dim) 广播到每个头
    full_s = torch.cat([sin, sin], dim=-1).unsqueeze(1)
    for slot in (0, 1):
        x = qkv[:, slot].to(torch.float32)
        expect = x * full_c + rotate_half_ref(x) * full_s
        assert (got[:, slot].to(torch.float32) - expect).abs().max().item() < 2e-2


def test_rope_fp32_input_no_double_rotation_regression():
    """回归：fp32 输入时 `.to(fp32)` 返回视图，旧写法边算边写会让后半用到已转的前半。

    判据：与原始输入的全表公式在 fp32 下 ≤1e-5；阳性对照用错误顺序就地复算，
    与正确结果必须显著不同（>1e-3），否则此门本身是假绿。"""
    torch.manual_seed(7)
    qkv = torch.randn(8, 3, 2, 16, dtype=torch.float32)
    cos, sin = rope_angle_tables(8, 8)
    got = rope_ref(qkv, cos, sin)
    full_c = torch.cat([cos, cos], dim=-1).unsqueeze(1)
    full_s = torch.cat([sin, sin], dim=-1).unsqueeze(1)
    for slot in (0, 1):
        x = qkv[:, slot]
        expect = x * full_c + rotate_half_ref(x) * full_s
        assert (got[:, slot] - expect).abs().max().item() < 1e-5
    # 阳性对照：复现旧 bug 的写入顺序（先写前半污染视图，再算后半）
    bad = qkv.clone()
    half, c, s = 8, cos.unsqueeze(1), sin.unsqueeze(1)
    for slot in (0, 1):
        part = bad[:, slot]
        x1, x2 = part[..., :half], part[..., half:]
        part[..., :half] = x1 * c - x2 * s
        part[..., half:] = x2 * c + x1 * s
    assert (bad - got).abs().max().item() > 1e-3


def test_rope_zero_angle_is_identity_and_slot_subset():
    """角度全零时旋转应恒等；只转指定槽位时另一槽位必须逐位不变。"""
    qkv = torch.randn(6, 3, 2, 16, dtype=torch.float16)
    zeros, ones = torch.zeros(6, 8), torch.ones(6, 8)
    assert torch.equal(rope_ref(qkv, ones, zeros), qkv)
    cos, sin = rope_angle_tables(6, 8)
    only_q = rope_ref(qkv, cos, sin, rotate_slots=(0,))
    assert torch.equal(only_q[:, 1], qkv[:, 1]) and torch.equal(only_q[:, 2], qkv[:, 2])
    assert not torch.equal(only_q[:, 0], qkv[:, 0])


def test_rope_rejects_odd_dim_and_bad_table():
    qkv = torch.randn(4, 3, 2, 16, dtype=torch.float16)
    with pytest.raises(ValueError, match="偶数"):
        rope_ref(torch.randn(4, 3, 2, 15, dtype=torch.float16), torch.ones(4, 7), torch.zeros(4, 7))
    with pytest.raises(ValueError, match="cos/sin"):
        rope_ref(qkv, torch.ones(4, 4), torch.zeros(4, 4))
    with pytest.raises(ValueError, match="qkv 需为"):
        rope_ref(torch.randn(4, 4, 2, 16, dtype=torch.float16), torch.ones(4, 8), torch.zeros(4, 8))
    with pytest.raises(ValueError, match="half"):
        rope_angle_tables(4, 0)


# ============================ A4 attn_sw（因果半窗）============================


def test_causal_window_mask_shape_semantics():
    """掩码口径：下标差 0<=i-j<W 才点亮；上三角（未来，即 j>i）整片为假，对角恒为真。"""
    m = causal_window_mask(8, 3)
    assert m.shape == (8, 8) and m.dtype == torch.bool
    delta = torch.arange(8).unsqueeze(1) - torch.arange(8).unsqueeze(0)  # delta[i, j] = i - j
    assert not m[delta < 0].any(), "未来位置（j>i）被点亮——方向抄反"
    assert torch.diagonal(m).all()
    assert torch.equal(m.sum(dim=-1), torch.tensor([1, 2, 3, 3, 3, 3, 3, 3]))
    assert causal_window_mask(8, None).triu(1).logical_not().all()  # 纯因果也不看未来


@pytest.mark.parametrize("window", [8, 16, 64])
def test_attn_sw_weights_zero_outside_window(window):
    """份额矩阵硬断言：未来与超出 W 的过去位置**严格等于 0**，每行和为 1，自位份额为正。"""
    torch.manual_seed(6)
    seq, heads, dim = 32, 2, 16
    q = torch.randn(1, heads, seq, dim, dtype=torch.float16)
    k = torch.randn(1, heads, seq, dim, dtype=torch.float16)
    w = attn_sw_weights_ref(q, k, window=window)
    assert w.shape == (1, heads, seq, seq) and w.dtype == torch.float32
    delta = torch.arange(seq).unsqueeze(1) - torch.arange(seq).unsqueeze(0)
    assert (w[..., delta < 0] == 0).all(), "未来位置份额非零"
    assert (w[..., delta >= window] == 0).all(), "窗口外份额非零"
    assert torch.isfinite(w).all()
    assert (w.sum(dim=-1) - 1.0).abs().max().item() < 1e-5
    assert (w.diagonal(dim1=-2, dim2=-1) > 0).all()


def test_attn_sw_window_one_reduces_to_identity():
    """W=1 时每个位置只看自己：份额应退化成单位阵，输出逐位等于值本身。"""
    torch.manual_seed(7)
    seq, dim = 10, 8
    q = torch.randn(1, 2, seq, dim, dtype=torch.float16)
    k = torch.randn(1, 2, seq, dim, dtype=torch.float16)
    v = torch.randn(1, 2, seq, dim, dtype=torch.float16)
    w = attn_sw_weights_ref(q, k, window=1)
    eye = torch.eye(seq).unsqueeze(0).unsqueeze(0)
    assert (w - eye).abs().max().item() < 1e-6
    out = attn_sw_ref(q, k, v, window=1, out_dtype=None)
    assert (out - v.to(torch.float32)).abs().max().item() < 1e-3


def test_attn_sw_full_window_matches_official_causal_attention():
    """W>=seq 时等价于官方因果缩放点积注意力（把"因果"这条口径与成熟实现绑定）。"""
    torch.manual_seed(8)
    seq, heads, dim = 24, 3, 16
    q = torch.randn(2, heads, seq, dim, dtype=torch.float16)
    k = torch.randn(2, heads, seq, dim, dtype=torch.float16)
    v = torch.randn(2, heads, seq, dim, dtype=torch.float16)
    mine = attn_sw_ref(q, k, v, window=seq, out_dtype=None)
    official = F.scaled_dot_product_attention(
        q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), is_causal=True
    )
    assert (mine - official).abs().max().item() < 1e-4


def test_attn_sw_right_alignment_when_query_shorter():
    """Sq<Skv 的解码口径：查询 i 的真实位置是 Skv-Sq+i，只回看 W 个、绝不看未来。"""
    torch.manual_seed(9)
    skv, sq, dim, window = 12, 3, 8, 4
    q = torch.randn(1, 1, sq, dim, dtype=torch.float16)
    k = torch.randn(1, 1, skv, dim, dtype=torch.float16)
    w = attn_sw_weights_ref(q, k, window=window)
    offset = skv - sq  # 行 i 的查询位置 = offset + i；可见键区间 = (pos-W, pos]
    for i in range(sq):
        pos = offset + i
        assert (w[..., i, pos + 1 :] == 0).all(), f"行 {i} 看见了未来键"
        assert (w[..., i, : max(0, pos - window + 1)] == 0).all(), f"行 {i} 回看超出 W"
        assert w[..., i, pos] > 0, f"行 {i} 连自己都没看见"
    assert (w.sum(dim=-1) - 1.0).abs().max().item() < 1e-5


def test_attn_sw_default_scale_is_reciprocal_sqrt_dim():
    """默认缩放必须等于 1/sqrt(head_dim)：显式传同值应与默认逐位相同。"""
    torch.manual_seed(10)
    q = torch.randn(1, 2, 8, 16, dtype=torch.float16)
    k = torch.randn(1, 2, 8, 16, dtype=torch.float16)
    default = attn_sw_weights_ref(q, k, window=8)
    explicit = attn_sw_weights_ref(q, k, window=8, scale=1.0 / math.sqrt(16))
    assert (default - explicit).abs().max().item() < 1e-9


def test_attn_sw_scale_override_sharpens_last_row():
    """放大 scale（降温）让份额更集中：最后一行（可见键最多）的最大份额必增。"""
    torch.manual_seed(11)
    q = torch.randn(1, 2, 8, 16, dtype=torch.float16)
    k = torch.randn(1, 2, 8, 16, dtype=torch.float16)
    v = torch.randn(1, 2, 8, 16, dtype=torch.float16)
    mild = attn_sw_weights_ref(q, k, window=8)
    sharp = attn_sw_weights_ref(q, k, window=8, scale=1.0)
    assert mild[..., -1, :].max().item() < sharp[..., -1, :].max().item()
    out_mild = attn_sw_ref(q, k, v, window=8, out_dtype=None)
    out_scaled = attn_sw_ref(q, k, v, window=8, scale=1.0, out_dtype=None)
    assert (out_mild - out_scaled).abs().max().item() > 1e-4


def test_attn_sw_rejects_bad_contract():
    q = torch.randn(1, 2, 8, 16, dtype=torch.float16)
    with pytest.raises(ValueError, match="Sq"):
        attn_sw_weights_ref(q, torch.randn(1, 2, 4, 16, dtype=torch.float16))
    with pytest.raises(ValueError, match="四维"):
        attn_sw_weights_ref(q.squeeze(0), q.squeeze(0))
    with pytest.raises(ValueError, match="v 与 k"):
        attn_sw_ref(q, q, torch.randn(1, 2, 7, 16, dtype=torch.float16))
    with pytest.raises(ValueError, match="seq"):
        causal_window_mask(0, 4)
    with pytest.raises(ValueError, match="window"):
        causal_window_mask(4, 0)
