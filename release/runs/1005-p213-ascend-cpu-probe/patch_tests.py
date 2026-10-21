"""修 tests/test_ascend_gradcheck.py 的四处问题（写测试时自己踩的坑，逐条对应）。

1. `_assert_kernel_ran("gemm")` 的片段会同时命中 gemm_dw 的键，改为 "gemm[" 精确匹配；
2. 复用性用例里 `gemm_ref(first.new_tensor(first), ...)` 是我写糊的废话（拿结果当输入），
   改为直接对 gemm_ref 比对；
3. rope 用例里 `got.data_ptr() == snapshot.data_ptr()` 逻辑反了（snapshot 是 clone，必然不等），
   改为断言"交回的就是原对象"；
4. attn 权重可见性用 `wt[allow.expand_as(wt)]`（布尔索引形状易踩坑），改 masked_select；
5. 契约用例里 gdn 的第 6 个位置参数是 out_dtype，我误传了张量（只会 TypeError，不是目标
   ValueError），改为把 beta 的形状写错。
"""
from pathlib import Path

p = Path("tests/test_ascend_gradcheck.py")
src = p.read_text()

pairs = [
    # 1) 片段精确化：线性前向件的键是 "gemm[cpu]"，dW 件是 "gemm_dw[cpu]"
    ('''    _assert_kernel_ran("gemm")


def test_gemm_cpu_reuses_single_compilation''',
     '''    _assert_kernel_ran("gemm[")


def test_gemm_cpu_reuses_single_compilation'''),
    ('''    got = gemm_bwd_dw_kernel.backward(dy, a, out_dtype=torch.float32, target=TARGET)
    torch.testing.assert_close(got, dy.T @ a, **TOL)
    assert got.shape == (10, 16)
    _assert_kernel_ran("gemm_dw")''',
     '''    got = gemm_bwd_dw_kernel.backward(dy, a, out_dtype=torch.float32, target=TARGET)
    torch.testing.assert_close(got, dy.T @ a, **TOL)
    assert got.shape == (10, 16)
    _assert_kernel_ran("gemm_dw[")'''),
    ('''    torch.testing.assert_close(first, gemm_ref(first.new_tensor(first), w, None, "none",
                                               torch.float32), **TOL)
    assert second.shape == (64, 10)''',
     '''    torch.testing.assert_close(first, gemm_ref(first, w, None, "none", torch.float32), **TOL)
    torch.testing.assert_close(second, gemm_ref(second, w, None, "none", torch.float32), **TOL)'''),
    # 2) rope：断言交回的是同一个对象（就地语义），尺子用未动的原件
    ('''    qkv = torch.randn(tokens, 3, heads, dim)
    snapshot = qkv.clone()
    got = rope_kernel.forward(qkv, cos.float(), sin.float(), rope_kernel.DEFAULT_SLOTS, TARGET)
    assert got.data_ptr() == snapshot.data_ptr(), "rope 的口径是就地改写并交回同一个盒子"
    want = rope_ref(snapshot, cos.float(), sin.float(), rope_kernel.DEFAULT_SLOTS)
    torch.testing.assert_close(got, want, **TOL)
    torch.testing.assert_close(got[:, 2], snapshot[:, 2], rtol=0, atol=0)
    _assert_kernel_ran("rope")''',
     '''    qkv = torch.randn(tokens, 3, heads, dim)
    original = qkv.clone()               # 尺子站在内核外面：拿没被动过的原件算参考
    got = rope_kernel.forward(qkv, cos.float(), sin.float(), rope_kernel.DEFAULT_SLOTS, TARGET)
    assert got is qkv, "rope 的口径是就地改写并交回同一个盒子"
    want = rope_ref(original, cos.float(), sin.float(), rope_kernel.DEFAULT_SLOTS)
    torch.testing.assert_close(got, want, **TOL)
    torch.testing.assert_close(got[:, 2], original[:, 2], rtol=0, atol=0)
    _assert_kernel_ran("rope[")'''),
    ('''    _assert_kernel_ran("add_ln")''', '''    _assert_kernel_ran("add_ln[")'''),
    ('''    _assert_kernel_ran("ln_bwd")''', '''    _assert_kernel_ran("ln_bwd[")'''),
    ('''    assert got.shape == (6, 16)
    _assert_kernel_ran("gather")''',
     '''    assert got.shape == (6, 16)
    _assert_kernel_ran("gather[")'''),
    ('''    torch.testing.assert_close(got, grad, **TOL)
    _assert_kernel_ran("scatter_add")''',
     '''    torch.testing.assert_close(got, grad, **TOL)
    _assert_kernel_ran("scatter_add[")'''),
    ('''    assert got.shape == q.shape
    _assert_kernel_ran("attn_sw")''',
     '''    assert got.shape == q.shape
    _assert_kernel_ran("attn_sw[")'''),
    # 4) 布尔掩码统一用 masked_select（会广播，不需要手工 expand）
    ('''    hidden = wt.masked_fill(allow, 0.0)              # 把"该看见的"抹掉，剩下的必须全零
    assert float(hidden.abs().max()) == 0.0, f"不可见位置偷分了：{float(hidden.abs().max())}"
    live = wt[allow.expand_as(wt)]
    assert float(live.min()) > 0.0, "可见位置的份额必须真的分到了东西"''',
     '''    hidden = wt.masked_select(~allow)                # 只挑"不该看见的"那些格
    assert float(hidden.abs().max()) == 0.0, f"不可见位置偷分了：{float(hidden.abs().max())}"
    live = wt.masked_select(allow)
    assert live.numel() > 0 and float(live.min()) > 0.0, "可见位置的份额必须真的分到了东西"'''),
    ('''    assert batched.shape == (1, heads, seq, dv)
    _assert_kernel_ran("gdn")''',
     '''    assert batched.shape == (1, heads, seq, dv)
    _assert_kernel_ran("gdn[")'''),
    ('''    torch.testing.assert_close(lora_kernel.apply(x, a, b, s, base, torch.float32, TARGET),
                               base + delta, **TOL)
    _assert_kernel_ran("gemm")                  # 组合件不产自己的 key，复用 linear 两件''',
     '''    torch.testing.assert_close(lora_kernel.apply(x, a, b, s, base, torch.float32, TARGET),
                               base + delta, **TOL)
    _assert_kernel_ran("gemm[")                 # 组合件不产自己的 key，复用 linear 两件'''),
    ('''    for name, got_one, want_one in zip(("dx", "dA", "dB"), got, want):
        torch.testing.assert_close(got_one, want_one, **TOL, msg=f"{name} 与自动微分不一致")''',
     '''    for name, got_one, want_one in zip(("dx", "dA", "dB"), got, want):
        torch.testing.assert_close(got_one, want_one, **TOL, msg=f"{name} 与自动微分不一致")
    _assert_kernel_ran("gemm_dw[")              # dA/dB 两条链确实是 dW 件算出来的'''),
    # 5) 契约用例：gdn 的第 6 个位置参数是 out_dtype，改成把 beta 形状写错来触发 ValueError
    ('''    with pytest.raises(ValueError):
        gdn_kernel.forward(torch.randn(2, 4, 8), torch.randn(2, 4, 8), torch.randn(2, 4, 8),
                           torch.randn(2, 4, 8), torch.randn(2, 4), torch.randn(2, 4, 8), TARGET)''',
     '''    with pytest.raises(ValueError):
        # beta 需为 (heads, seq)=(2, 4)，这里故意给 (2, 5)
        gdn_kernel.forward(torch.randn(2, 4, 8), torch.randn(2, 4, 8), torch.randn(2, 4, 8),
                           torch.randn(2, 4, 8), torch.randn(2, 5), torch.float32, TARGET)'''),
]
for old, new in pairs:
    assert src.count(old) == 1, (src.count(old), old[:70])
    src = src.replace(old, new, 1)
p.write_text(src)
print("patched", len(pairs))
