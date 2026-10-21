"""p1-04 MPS 内核对拍（`@pytest.mark.mps`）：B1 方言冒烟 + B2~B5 逐算子对拍 + 组装决策一致。

本文件的定位是**方言真跑**的证据链，因此有一条铁律：任何用例都不许把"回退到 torch 写法"
当成内核通过。每条对拍用例都用两把尺子卡住这一点——
① `backends.blockers()` 必须为空（用例开始前先隔离掉别的文件留下的阻塞账）；
② 该算子的编译键必须已经出现在 `backends.compiled_keys()` 里（回退路径不会在缓存里留痕）。
方言不可用（没装 tilelang / 没 MPS 设备 / metal 后端探测失败）时整模块 skip，
由 `tests/test_backends.py` 的回退用例负责证明"回退路径本身正确"。

数值门与 spec 一致：逐元素 `max|err| ≤ 2e-2`（fp16 出口）。
"""
from __future__ import annotations

import pytest
import torch

from sys1.kernels import (
    add_ln_kernel,
    attn_sw_kernel,
    attn_sw_mps,
    backends,
    gemm_kernel,
    gemm_mps,
    rope_kernel,
    rope_mps,
)
from sys1.testing.torch_ref import (
    add_ln_ref,
    attn_sw_ref,
    gemm_ref,
    rope_angle_tables,
    rope_ref,
)

try:  # 方言入口只在设备门用例里需要；缺失时本模块整体 skip
    import tilelang
    import tilelang.metal.language as T
except Exception:  # noqa: BLE001  # pragma: no cover - 方言装不上/加载失败都只意味着本模块整体 skip
    tilelang = None
    T = None

DEV = torch.device("mps")
TOL = 2e-2  # spec 的 fp16 对拍门限


def _dialect_ready() -> bool:
    """本机是否真能把方言发出去（三件事都得成：MPS 在、tilelang 在、metal 后端在）。"""
    return tilelang is not None and torch.backends.mps.is_available() and backends.active_backend(DEV) == backends.TILELANG


pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(not _dialect_ready(), reason="TileLang Metal 方言或 MPS 设备不可用（回退路径见 test_backends.py）"),
]


if T is not None:
    # B1 冒烟内核放在模块级：tilelang 要靠 inspect 取源码，嵌套函数取不到干净的一手源码。
    @tilelang.jit(target="metal", execution_backend="tvm_ffi")
    def smoke_impl(X, Y, cols: int):
        """y = 2x + 1：最小可用的 elementwise 内核，行数是动态符号。"""
        n_rows = T.dynamic("n_rows")
        X: T.Tensor((n_rows, cols), T.float16)
        Y: T.Tensor((n_rows, cols), T.float16)
        with T.Kernel(T.ceildiv(n_rows, 16), threads=64) as bx:
            blk = T.alloc_shared((16, cols), T.float16)
            T.copy(X[bx * 16:(bx + 1) * 16, :], blk)
            for i, j in T.Parallel(16, cols):
                blk[i, j] = T.cast(T.cast(blk[i, j], T.float32) * 2.0 + 1.0, T.float16)
            T.copy(blk, Y[bx * 16:(bx + 1) * 16, :])


@pytest.fixture(autouse=True)
def _clean_blocker_ledger(monkeypatch):
    """每条用例从"阻塞账为空"开始，但保留编译缓存（否则每条都要重编，慢到没法跑）。"""
    monkeypatch.setattr(backends, "_blockers", {})


def _maxerr(a: torch.Tensor, b: torch.Tensor) -> float:
    """两条路径的逐元素最大绝对偏差（统一升到 fp32 再比）。"""
    return (a.to(torch.float32) - b.to(torch.float32)).abs().max().item()


def _assert_ran_dialect(fragment: str) -> None:
    """"内核真的跑过"的硬证：阻塞账为空 且 缓存里能看到带这个片段的编译键。"""
    assert backends.blockers() == {}, f"方言被记账回退了：{backends.blockers()}"
    keys = backends.compiled_keys()
    assert any(fragment in k for k in keys), f"编译缓存里没有 {fragment!r}（现有：{keys}）——这条用例没走内核"


# ---------------------------------------------------------------------------------------------
# B1 方言冒烟
# ---------------------------------------------------------------------------------------------


def test_smoke_metal_language_entry_exists():
    """B1 前置：`tilelang.metal.language` 这个入口在 0.1.15 里确实存在且够用。"""
    assert T is not None
    needed = (
        "Kernel", "Tensor", "const", "dynamic", "alloc_shared", "alloc_var", "copy", "clear", "fill",
        "gemm", "Parallel", "serial", "cast", "max", "min", "if_then_else", "exp2", "rsqrt", "ceildiv",
        "floordiv", "get_thread_binding", "sync_threads", "float16", "float32",
    )
    missing = [n for n in needed if not hasattr(T, n)]
    assert not missing, f"metal 方言缺符号：{missing}"


@pytest.mark.parametrize("rows", [1, 16, 33])
def test_smoke_elementwise_kernel_compiles_and_runs(rows):
    """B1 冒烟：最小 elementwise 内核能编译、能在 MPS 上发射、行数不为整数块也对。

    这一条是整个 B 系列的前置探针：动态行数 + `T.copy` 区域切片的边界谓词在这里先验证掉，
    后面四个算子才敢都用同一套写法（33 行故意不是块高 16 的整数倍）。三档行数共用一份产物。
    """
    before = backends.compile_count()
    x = torch.randn(rows, 16, dtype=torch.float16, device=DEV)
    y = torch.zeros_like(x)
    smoke_impl(x, y, 16)
    torch.mps.synchronize()
    want = (x.to(torch.float32) * 2.0 + 1.0).to(torch.float16)
    assert _maxerr(y, want) == 0.0, f"rows={rows} 冒烟失败：{_maxerr(y, want):.3e}"
    assert backends.compile_count() == before, "冒烟内核不该走 backends 缓存却动了计数"


# ---------------------------------------------------------------------------------------------
# B2 gemm
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("m", "n", "k"), [(1, 32, 64), (7, 64, 128), (16, 128, 256), (40, 96, 64), (128, 32, 32)])
def test_gemm_parity_fp16(m, n, k):
    """gemm 对拍：relu + bias 口径，fp32 累加、fp16 出口，逐元素 err ≤ 2e-2。"""
    A = torch.randn(m, k, dtype=torch.float16, device=DEV)
    W = torch.randn(n, k, dtype=torch.float16, device=DEV)
    bias = torch.randn(n, dtype=torch.float32, device=DEV)
    got = gemm_kernel.forward(A, W, bias, act="relu")
    want = gemm_ref(A, W, bias, act="relu")
    assert got.dtype == torch.float16 and got.shape == (m, n)
    err = _maxerr(got, want)
    assert err <= TOL, f"gemm m={m} n={n} k={k} err={err:.3e}"
    _assert_ran_dialect(f"gemm|n={n}|k={k}|")


def test_gemm_dynamic_m_reuses_one_compilation():
    """spec"多 batch 复用"：M=1,7,16 三档只编译一次，三档都过误差门。

    缓存键里不含 M（M 是 `T.dynamic`），所以计数增量必须正好是 1；用一对只在本用例出现的
    (N,K) 组合，避免被别的用例先编译掉。
    """
    before = backends.compile_count()
    W = torch.randn(64, 64, dtype=torch.float16, device=DEV)
    bias = torch.randn(64, dtype=torch.float32, device=DEV)
    for m in (1, 7, 16):
        A = torch.randn(m, 64, dtype=torch.float16, device=DEV)
        got = gemm_kernel.forward(A, W, bias, act="relu")
        assert _maxerr(got, gemm_ref(A, W, bias, act="relu")) <= TOL, f"M={m}"
    assert backends.compile_count() - before == 1, "同一 (N,K) 不同 M 触发了重复编译"
    _assert_ran_dialect("gemm|n=64|k=64|")


def test_gemm_act_none_and_no_bias_match_ref():
    """不加偏置、不过截断的口径同样走内核（内部补一条零 bias，与"没有加成"等价）。"""
    A = torch.randn(16, 64, dtype=torch.float16, device=DEV)
    W = torch.randn(32, 64, dtype=torch.float16, device=DEV)
    got = gemm_kernel.forward(A, W, None, act="none")
    assert _maxerr(got, gemm_ref(A, W, None, act="none")) <= TOL
    # act_mode=0 是本用例独有的编译键尾段（其余 gemm 用例均为 relu 的 act_mode=1），
    # 用它防止"别的形状先编译过、本用例静默回退却因泛化前缀命中而假绿"。
    _assert_ran_dialect("act_mode=0|")


@pytest.mark.parametrize("act", ["gelu", "silu"])
def test_gemm_unsupported_act_falls_back_to_eager(act):
    """方言只落地 none/relu：其余口径必须回退，且回退结果与参考实现等价（数值纪律不松）。"""
    A = torch.randn(16, 64, dtype=torch.float16, device=DEV)
    W = torch.randn(32, 64, dtype=torch.float16, device=DEV)
    got = gemm_kernel.forward(A, W, None, act=act)
    assert _maxerr(got, gemm_ref(A, W, None, act=act)) <= TOL
    assert gemm_mps.plan(A, W, None, torch.empty((16, 32), dtype=torch.float16, device=DEV), act) is None


# ---------------------------------------------------------------------------------------------
# B3 add_ln
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("rows", "dim"), [(16, 32), (40, 32), (48, 64), (33, 16)])
def test_add_ln_parity_and_fp32_residual(rows, dim):
    """add_ln 对拍：整形输出走内核，残差出口 h 必须是 fp32（补齐行数在入口完成）。"""
    x = torch.randn(rows, dim, dtype=torch.float16, device=DEV)
    res = torch.randn(rows, dim, dtype=torch.float16, device=DEV)
    w = torch.randn(dim, dtype=torch.float32, device=DEV)
    b = torch.randn(dim, dtype=torch.float32, device=DEV)
    y, h = add_ln_kernel.forward(x, res, w, b)
    wy, wh = add_ln_ref(x, res, w, b)
    assert h.dtype == torch.float32, "残差流被降位宽了"
    assert _maxerr(y, wy) <= TOL
    assert _maxerr(h, wh) <= 1e-6, f"fp32 残差不等：{_maxerr(h, wh):.3e}"
    _assert_ran_dialect("add_ln|")


def test_add_ln_accepts_fp32_residual_stream():
    """第二层起的残差就是上一层交出的 fp32：这条路径也必须走内核，不能被迫降回 fp16。"""
    x = torch.randn(32, 32, dtype=torch.float16, device=DEV)
    h_prev = torch.randn(32, 32, dtype=torch.float32, device=DEV)
    w = torch.randn(32, dtype=torch.float32, device=DEV)
    b = torch.randn(32, dtype=torch.float32, device=DEV)
    y, h = add_ln_kernel.forward(x, h_prev, w, b)
    wy, wh = add_ln_ref(x, h_prev, w, b)
    assert _maxerr(y, wy) <= TOL and _maxerr(h, wh) <= 1e-6
    _assert_ran_dialect("res_fp32=1")


# ---------------------------------------------------------------------------------------------
# B4 rope
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tokens", [1, 8, 24])
def test_rope_parity_and_value_slot_untouched(tokens):
    """rope 对拍：就地旋转 packed-qkv，值槽必须逐位等于输入（一个 bit 都不许动）。"""
    cos, sin = rope_angle_tables(tokens, 16, device=DEV)
    qkv = torch.randn(tokens, 3, 4, 32, dtype=torch.float16, device=DEV)
    got = rope_kernel.forward(qkv.clone(), cos, sin)
    want = rope_ref(qkv, cos, sin)
    assert _maxerr(got, want) <= TOL
    assert torch.equal(got[:, 2], qkv[:, 2]), "值槽被改动了"
    _assert_ran_dialect("rope|heads=4|dim=32")


def test_rope_is_in_place_and_reuses_one_compilation():
    """就地语义 + 动态 token 数：返回同一对象，且换长度不重编。"""
    before = backends.compile_count()
    for tokens in (8, 24):
        qkv = torch.randn(tokens, 3, 3, 16, dtype=torch.float16, device=DEV)
        cos, sin = rope_angle_tables(tokens, 8, device=DEV)
        snapshot = qkv.clone()  # 就地改写会覆盖原值，参考实现必须吃改动前的那份
        returned = rope_kernel.forward(qkv, cos, sin)
        assert returned is qkv, "就地改写没有返回同一对象"
        assert _maxerr(returned, rope_ref(snapshot, cos, sin)) <= TOL
    assert backends.compile_count() - before == 1, "换 token 数触发了重复编译"
    _assert_ran_dialect("rope|heads=3|dim=16")


def test_rope_non_default_slots_fall_back_but_stay_in_place():
    """非默认槽位组合走回退：语义仍要求原地改写、且其余槽位分毫不动。"""
    tokens = 8
    qkv = torch.randn(tokens, 3, 4, 32, dtype=torch.float16, device=DEV)
    cos, sin = rope_angle_tables(tokens, 16, device=DEV)
    snapshot = qkv.clone()
    returned = rope_kernel.forward(qkv, cos, sin, rotate_slots=(1,))
    assert returned is qkv
    assert torch.equal(returned[:, 0], snapshot[:, 0]) and torch.equal(returned[:, 2], snapshot[:, 2])
    assert _maxerr(returned, rope_ref(snapshot, cos, sin, rotate_slots=(1,))) <= TOL
    assert rope_mps.plan(snapshot, cos, sin) is not None  # 方言本身只认默认前两槽


# ---------------------------------------------------------------------------------------------
# B5 attn_sw
# ---------------------------------------------------------------------------------------------


def test_attn_compile_on_ladder_length():
    """B5a：阶梯长度上的因果滑窗内核能编出来、能发射，并且误差过门。"""
    q = torch.randn(1, 2, 32, 32, dtype=torch.float16, device=DEV)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    got = attn_sw_kernel.forward(q, k, v, 16)
    assert got.shape == q.shape and got.dtype == torch.float16
    assert _maxerr(got, attn_sw_ref(q, k, v, 16)) <= TOL
    _assert_ran_dialect("attn_sw|")


@pytest.mark.parametrize(("seq", "window"), [(64, 64), (128, 64), (128, 128), (256, 64), (256, 128), (256, 256)])
def test_attn_parity_multi_window(seq, window):
    """B5b 对拍：多档 W（64/128/256）× 多档 seq，fp16 逐元素 err ≤ 2e-2。"""
    q = torch.randn(1, 2, seq, 32, dtype=torch.float16, device=DEV)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    got = attn_sw_kernel.forward(q, k, v, window)
    want = attn_sw_ref(q, k, v, window)
    err = _maxerr(got, want)
    assert err <= TOL, f"seq={seq} W={window} err={err:.3e}"
    _assert_ran_dialect(f"attn_sw|heads=2|seq={seq}|dim=32|window={window}|")


def test_attn_parity_causal_direction_ignores_invisible_keys():
    """B5b 因果纪律（内核侧）：改动"未来"与"窗口外"的键值，该行输出必须逐位不变。

    这是把 A4 在参考实现上断言的方向性，落到真正生成的 MSL 上再验一次——蓝本是双向滑窗，
    抄反方向只会让对拍误差碰巧变大，只有"扰动不可见位置、结果分毫不差"才是硬证据。
    """
    seq, window, row = 64, 16, 40
    q = torch.randn(2, seq, 32, dtype=torch.float16, device=DEV)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    o_base = attn_sw_kernel.forward(q, k, v, window, out_dtype=torch.float32)

    lo = row - window + 1  # 行 row 能看见的最老键：[lo, row]
    k2, v2 = k.clone(), v.clone()
    k2[:, :lo] = torch.randn_like(k2[:, :lo])  # 窗口外（太老的过去）
    v2[:, :lo] = torch.randn_like(v2[:, :lo])
    k2[:, row + 1:] = torch.randn_like(k2[:, row + 1:])  # 未来
    v2[:, row + 1:] = torch.randn_like(v2[:, row + 1:])
    o2 = attn_sw_kernel.forward(q, k2, v2, window, out_dtype=torch.float32)
    assert torch.equal(o_base[:, row], o2[:, row]), "不可见位置被算进来了：因果/窗口方向有问题"

    k3 = k.clone()
    k3[:, row] = k3[:, row] * 8.0  # 只动可见范围内的一条键
    o3 = attn_sw_kernel.forward(q, k3, v, window, out_dtype=torch.float32)
    assert not torch.equal(o_base[:, row], o3[:, row]), "可见键被改动而输出不变：结果可能是假通过"


@pytest.mark.parametrize("seq", [17, 20])
def test_attn_non_ladder_falls_back_to_eager(seq):
    """非阶梯长度（不被块宽整除）plan 判不可表达 → 回退 torch 写法，结果仍与参考实现一致。"""
    q = torch.randn(1, 2, seq, 32, dtype=torch.float16, device=DEV)
    k, v = torch.randn_like(q), torch.randn_like(q)
    got = attn_sw_kernel.forward(q, k, v, 16)
    assert _maxerr(got, attn_sw_ref(q, k, v, 16)) <= TOL
    q3 = q.reshape(-1, seq, 32).contiguous()
    out3 = torch.empty(q3.shape, dtype=torch.float16, device=DEV)
    assert attn_sw_mps.plan(q3, q3, q3, out3, 16, 32**-0.5) is None


# ---------------------------------------------------------------------------------------------
# B5c 组装小前向：决策 argmax 与 torch 路径 100% 一致
# ---------------------------------------------------------------------------------------------

# 小前向的固定开本：D=64、2 头 × 32、32 个 token、窗口 16、候选词表 64
D_MODEL = 64
N_HEADS = 2
HEAD_DIM = 32
N_TOKENS = 32
WINDOW = 16
VOCAB = 64
OPTIONS = ((0, 7), (1, 23), (5, 41))


def _mini_params(device: torch.device) -> dict[str, torch.Tensor]:
    """造一套固定的小前向参数（CPU 上用固定种子生成，再搬到目标设备，保证两条路径同源）。"""
    gen = torch.Generator(device="cpu").manual_seed(20261004)

    def h16(*shape: int, scale: float = 0.1) -> torch.Tensor:
        return (torch.randn(*shape, generator=gen) * scale).to(torch.float16).to(device)

    def f32(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen).to(torch.float32).to(device)

    cos, sin = rope_angle_tables(N_TOKENS, HEAD_DIM // 2, device=device)
    return {
        "w_qkv": h16(3 * N_HEADS * HEAD_DIM, D_MODEL),
        "b_qkv": f32(3 * N_HEADS * HEAD_DIM),
        "cos": cos,
        "sin": sin,
        "w_o": h16(D_MODEL, N_HEADS * HEAD_DIM),
        "ln_a_w": f32(D_MODEL),
        "ln_a_b": f32(D_MODEL),
        "w_up": h16(2 * D_MODEL, D_MODEL),
        "w_down": h16(D_MODEL, 2 * D_MODEL),
        "ln_b_w": f32(D_MODEL),
        "ln_b_b": f32(D_MODEL),
        "w_logits": h16(VOCAB, D_MODEL),
    }


def _mini_forward(p: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """把四算子串成一层"注意力 + 小 MLP"的前向，返回最后一个位置的候选分数（fp32）。

    第二次的残差加法刻意吃上一层交出的 fp32 残差，用来验证残差流一路高精度不降位宽。
    """
    qkv = gemm_kernel.forward(x, p["w_qkv"], p["b_qkv"], act="none").view(N_TOKENS, 3, N_HEADS, HEAD_DIM)
    rope_kernel.forward(qkv, p["cos"], p["sin"])
    q, k, v = (qkv[:, slot].transpose(0, 1).contiguous() for slot in range(3))
    ctx = attn_sw_kernel.forward(q, k, v, WINDOW).transpose(0, 1).reshape(N_TOKENS, N_HEADS * HEAD_DIM)
    att = gemm_kernel.forward(ctx, p["w_o"], None, act="none")
    y1, h1 = add_ln_kernel.forward(att, x, p["ln_a_w"], p["ln_a_b"])
    up = gemm_kernel.forward(y1, p["w_up"], None, act="relu")
    down = gemm_kernel.forward(up, p["w_down"], None, act="none")
    y2, _h2 = add_ln_kernel.forward(down, h1, p["ln_b_w"], p["ln_b_b"])
    logits = gemm_kernel.forward(y2, p["w_logits"], None, act="none")
    return logits[N_TOKENS - 1].to(torch.float32)


def _decide(score: torch.Tensor) -> int:
    """把候选分数按选项分组求和，取分数最高的选项编号（一阶段的决策口径）。"""
    return int(max(range(len(OPTIONS)), key=lambda i: sum(score[j].item() for j in OPTIONS[i])))


def test_attn_argmax_assembled_forward_matches_torch_path():
    """B5c：同一批决策输入，方言路径与普通 torch 路径的选项 argmax 必须 100% 一致。"""
    p = _mini_params(DEV)
    gen = torch.Generator(device="cpu").manual_seed(7)
    cases = [
        (torch.randn(N_TOKENS, D_MODEL, generator=gen) * 0.5).to(torch.float16).to(DEV)
        for _ in range(8)
    ]

    backends.set_backend_preference("auto")
    dialect_scores = [_mini_forward(p, x) for x in cases]
    _assert_ran_dialect("gemm|")
    _assert_ran_dialect("rope|")
    _assert_ran_dialect("attn_sw|")
    _assert_ran_dialect("add_ln|")
    _assert_ran_dialect("res_fp32=1")

    backends.set_backend_preference("torch")
    try:
        eager_scores = [_mini_forward(p, x) for x in cases]
    finally:
        backends.set_backend_preference("auto")

    diffs = [_maxerr(a, b) for a, b in zip(dialect_scores, eager_scores)]
    assert max(diffs) <= TOL, f"逐元素偏差超门：{max(diffs):.3e}"
    same = [_decide(a) == _decide(b) for a, b in zip(dialect_scores, eager_scores)]
    assert all(same), f"决策换人：{same}，偏差={diffs}"
