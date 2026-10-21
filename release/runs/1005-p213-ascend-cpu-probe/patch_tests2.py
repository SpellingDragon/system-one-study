"""修 tests/test_ascend_gradcheck.py 里 linear/LoRA 五条用例的形状选择。

根因（不是内核 bug，是我挑错了被测形状）：本域 linear 两件与 P1 同口径，N/K 轴必须能被
`BLOCK_LADDER = (64, 32, 16)` 里的某个块宽整除，否则 `plan()` 如实返回 None、入口层落回
torch 写法。我原先写的 n=10 / r=4 全在这个支持窗之外，于是数值对拍"通过"了但内核根本没发射
——`_assert_kernel_ran` 正好把这种"回退冒充通过"的假绿抓住（这就是它存在的意义）。

同时修掉复用性用例里我自己写糊的一行（把内核输出当输入喂给 gemm_ref，会 K 轴不匹配报错）。
"""
from pathlib import Path

p = Path("tests/test_ascend_gradcheck.py")
src = p.read_text()

pairs = [
    # ---- ① 前向：n=16、k=32 都在支持窗内；M=33 故意不整除块高 16（M 是动态维，正是要验的点）
    ('''    torch.manual_seed(1)
    a = torch.randn(33, 16)          # 行数故意不整除块高 16，验边界守卫
    w = torch.randn(10, 16)
    bias = torch.randn(10)''',
     '''    torch.manual_seed(1)
    a = torch.randn(33, 32)          # 行数故意不整除块高 16，验边界守卫
    w = torch.randn(16, 32)          # N/K 取 16/32：落在 BLOCK_LADDER 的支持窗内
    bias = torch.randn(16)'''),
    # ---- ② 复用性：输入要各自有名，且 W 与被测件同口径；原来把输出当输入喂参考实现是笔误
    ('''    torch.manual_seed(2)
    w = torch.randn(10, 16)
    base = ascend_env.compile_count()
    first = gemm_kernel.forward(torch.randn(8, 16), w, None, "none", torch.float32, TARGET)
    mid = ascend_env.compile_count()
    second = gemm_kernel.forward(torch.randn(64, 16), w, None, "none", torch.float32, TARGET)
    assert ascend_env.compile_count() == mid, "M 轴是动态维，换批不应该再开一次模具"
    assert mid >= base
    torch.testing.assert_close(first, gemm_ref(first, w, None, "none", torch.float32), **TOL)
    torch.testing.assert_close(second, gemm_ref(second, w, None, "none", torch.float32), **TOL)''',
     '''    torch.manual_seed(2)
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
    assert second.shape == (64, 16)'''),
    # ---- ③ dW：N/K 同样进支持窗
    ('''    torch.manual_seed(3)
    dy = torch.randn(33, 10)
    a = torch.randn(33, 16)''',
     '''    torch.manual_seed(3)
    dy = torch.randn(33, 16)          # 行数 33 不整除块宽，正是要验的动态轴
    a = torch.randn(33, 32)'''),
    ('''    torch.testing.assert_close(got, dy.T @ a, **TOL)
    assert got.shape == (10, 16)''',
     '''    torch.testing.assert_close(got, dy.T @ a, **TOL)
    assert got.shape == (16, 32)'''),
    # ---- ④⑤ LoRA：低秩轴 r 必须 ≥16 且能整除块宽，否则旁路两次乘加都落在支持窗外（只回退）
    ('''    torch.manual_seed(15)
    m, n, kk, r, s = 12, 10, 16, 4, 0.5''',
     '''    torch.manual_seed(15)
    m, n, kk, r, s = 12, 32, 16, 16, 0.5   # n/kk/r 全取块宽整数倍，旁路两次乘加才真进内核'''),
    ('''    torch.manual_seed(16)
    m, n, kk, r, s = 12, 10, 16, 4, 0.5''',
     '''    torch.manual_seed(16)
    m, n, kk, r, s = 12, 32, 16, 16, 0.5'''),
    ('''    torch.manual_seed(17)
    m, n, kk, r, s = 12, 10, 16, 4, 0.5''',
     '''    torch.manual_seed(17)
    m, n, kk, r, s = 12, 32, 16, 16, 0.5'''),
    # ---- ⑥ 契约用例里 gemm 的非法形状改成"K 轴对不上"（原来 (4,8)/(5,7) 也能报错，留着即可）
    # 追加一条：支持窗外的形状必须"数值仍对、但不产生编译产物"，把这个口径钉成文档
    ('''# --------------------------------------------------------------------------- 契约与形状''',
     '''def test_gemm_cpu_falls_back_outside_block_window():
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


# --------------------------------------------------------------------------- 契约与形状'''),
]
for old, new in pairs:
    assert src.count(old) == 1, (src.count(old), old[:70])
    src = src.replace(old, new, 1)
p.write_text(src)
print("patched", len(pairs))
