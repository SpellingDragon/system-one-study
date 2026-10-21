"""C1 backends 分发与回退（纯 CPU 可测，不打设备标记）。

覆盖 spec 的"方言阻塞回退"场景：三种阻塞来源都要**流程不中断**——
① 环境缺失（import 不了 tilelang / 本机没有 MPS / env 强制普通写法）；
② 编译失败（builder 抛异常）：必须发一次显式 warning、记进阻塞账、返回 None 让调用方回退，
   并且同一 key 不反复重试；
③ 覆盖不全（形状/位宽表达不了）：plan 判 None，入口照样算出与参考实现等价的结果。
每条用例前后都 reset 全局状态，避免计数与阻塞账在用例间串味。
"""
from __future__ import annotations

import sys
import warnings

import pytest
import torch

from sys1.kernels import add_ln_kernel, attn_sw_kernel, backends, gemm_kernel, rope_kernel
from sys1.testing.torch_ref import add_ln_ref, attn_sw_ref, gemm_ref, rope_angle_tables, rope_ref

CPU = torch.device("cpu")
TOL = 2e-2


# 设备门（R-P1-4 归因，先证后改）：下面四条用例走 get_compiled 的"方言路"——backends.py:146
# 的第一道闸 `tilelang_available()` 三段判据含"本机真有 MPS"（backends.py:88），ubuntu 无 MPS
# 时 builder 压根不被调用：计数 0、警告不发、失败记忆无从登记。这是**设备条件用例缺门**
# （环境伪影），不是回退逻辑回归——门照 test_assets.py:41 的 MPS 纪律口径设 skip，
# 不改产品代码语义；skip 只认"真没有 MPS"，host（有 MPS）照跑不豁免。
need_dialect_host = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="方言路用例需真有 MPS 的机器（get_compiled 的 tilelang_available() 判据含 MPS）；"
           "ubuntu 无 MPS 时 builder 不被调用，红的是探测口径而非回退逻辑（R-P1-4，见 docs/ci_baseline_triage.md §2）",
)


@pytest.fixture(autouse=True)
def _clean_compile_cache():
    """缓存行为用例必须从干净态起步（R16 教训的 CI 实锤：本机用例顺序侥幸干净≠任何环境干净）。

    get_compiled 的编译缓存/失败记忆是模块级单例；不隔离则别组用例先行写入 key 时，
    "per-key 计数""失败不重试"等断言在收集顺序变化后（CI 装包版本差异即可触发）必炸。
    """
    backends.reset()
    yield
    backends.reset()


def _maxerr(a: torch.Tensor, b: torch.Tensor) -> float:
    """逐元素最大绝对偏差（统一升 fp32 再比）。"""
    return (a.to(torch.float32) - b.to(torch.float32)).abs().max().item()


def _boom(exc: BaseException):
    """造一个"必然抛出 exc"的构造器，用来模拟编译期失败。"""

    def builder():
        raise exc

    return builder


# ---------------------------------------------------------------- 设备探测


def test_resolve_device_prefers_explicit_choice():
    """点名要的设备优先于自动探测。"""
    assert backends.resolve_device("cpu") == CPU
    assert backends.resolve_device(torch.device("cpu")) == CPU


def test_resolve_device_auto_is_mps_or_cpu():
    """自动档只会给出 mps 或 cpu（教师版裁掉了 cuda/npu）。"""
    dev = backends.resolve_device()
    assert dev.type in ("mps", "cpu")
    if not torch.backends.mps.is_available():
        assert dev.type == "cpu"


def test_active_backend_cpu_never_takes_dialect(monkeypatch):
    """即使方言可用，cpu 设备也必须判成 torch_eager——Metal 产物只能在 MPS 上发射。"""
    monkeypatch.setattr(backends, "_probe", {"available": True})
    assert backends.active_backend(CPU) == backends.TORCH_EAGER
    assert backends.active_backend("mps") == backends.TILELANG


# ---------------------------------------------------------------- env 强制与普通写法


def test_env_force_torch_disables_dialect():
    """env 说"用普通写法"时，探测直接判否且不记账（主动选择不是阻塞）。"""
    backends.set_backend_preference("torch")
    assert backends.tilelang_available() is False
    assert backends.active_backend("mps") == backends.TORCH_EAGER
    assert backends.blockers() == {}


def test_env_force_tilelang_still_needs_real_probe():
    """env 点名要方言时仍要真探测：本机没有 MPS 就判否——强制不等于伪造。"""
    backends.set_backend_preference("tilelang")
    if not torch.backends.mps.is_available():
        assert backends.tilelang_available() is False
        assert backends.active_backend("mps") == backends.TORCH_EAGER


# ---------------------------------------------------------------- 编译缓存与计数


@need_dialect_host
def test_get_compiled_builds_once_and_caches():
    """同一 key 只构建一次，第二次直接命中缓存且不动计数。"""
    calls = []

    def builder():
        calls.append(1)
        return object()

    first = backends.get_compiled("k1", builder)
    second = backends.get_compiled("k1", builder)
    assert first is second and len(calls) == 1
    assert backends.compile_count() == 1
    assert backends.compiled_keys() == ("k1",)


@need_dialect_host
def test_get_compiled_counts_per_key():
    """不同 key 各自编译，计数如实增长。"""
    backends.get_compiled("a", lambda: 1)
    backends.get_compiled("b", lambda: 2)
    assert backends.compile_count() == 2
    assert set(backends.compiled_keys()) == {"a", "b"}


def test_get_compiled_skips_builder_when_unavailable():
    """方言不可用时压根不该去调 builder：直接给回退信号，也不发警告。"""
    backends.set_backend_preference("torch")
    calls = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert backends.get_compiled("k", lambda: calls.append(1)) is None
    assert calls == [] and not caught


# ---------------------------------------------------------------- 编译失败回退（spec 场景）


@need_dialect_host
def test_compile_failure_warns_records_and_returns_none():
    """builder 抛异常 → 一次显式 RuntimeWarning + 登记阻塞点 + 返回 None（异常不外泄给业务）。"""
    with pytest.warns(RuntimeWarning, match="回退") as caught:
        got = backends.get_compiled("gemm|n=8|k=8", _boom(RuntimeError("MSL 不支持")))
    assert got is None
    assert len(caught) == 1
    blockers = backends.blockers()
    assert "gemm|n=8|k=8" in blockers
    assert "MSL 不支持" in blockers["gemm|n=8|k=8"]


@need_dialect_host
def test_failed_key_is_not_retried_and_warns_only_once():
    """失败过的 key 不再重试、不再重复告警（否则每层每步都要刷一条警告）。"""
    calls = []

    def builder():
        calls.append(1)
        raise RuntimeError("boom")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            assert backends.get_compiled("rope|heads=4|dim=32", builder) is None
    assert len(calls) == 1, "失败的 key 被反复重试"
    assert len(caught) == 1, f"告警发了 {len(caught)} 次"
    assert backends.compile_count() == 0


def test_record_blocker_text_is_reusable_for_run_notes():
    """阻塞账里要能直接抄出"哪一步、为什么"，这是 run notes 的原始素材。"""
    with pytest.warns(RuntimeWarning):
        backends.record_blocker("attn_sw|x", ValueError("缺原语"))
    block = backends.blockers()["attn_sw|x"]
    assert "ValueError" in block and "缺原语" in block


def test_missing_tilelang_probe_records_blocker(monkeypatch):
    """环境缺失（import 不了 tilelang）也被登记成阻塞点，并按"不可用"处理。"""
    monkeypatch.setitem(sys.modules, "tilelang", None)
    with pytest.warns(RuntimeWarning, match="tilelang_probe"):
        assert backends.tilelang_available() is False
    assert "tilelang_probe" in backends.blockers()
    assert backends.active_backend("mps") == backends.TORCH_EAGER


# ---------------------------------------------------------------- 回退结果必须正确


def test_all_ops_fallback_matches_torch_ref_on_cpu():
    """CPU 上四算子全走回退：结果必须与参考实现等价，"内核不阻主线"才是真的成立。"""
    torch.manual_seed(0)

    A = torch.randn(8, 32, dtype=torch.float16, device=CPU)
    W = torch.randn(16, 32, dtype=torch.float16, device=CPU)
    bias = torch.randn(16, dtype=torch.float32, device=CPU)
    assert _maxerr(gemm_kernel.forward(A, W, bias, act="relu"), gemm_ref(A, W, bias, act="relu")) <= TOL
    assert _maxerr(gemm_kernel.forward(A, W, None, act="gelu"), gemm_ref(A, W, None, act="gelu")) <= TOL

    x = torch.randn(6, 4, 32, dtype=torch.float16, device=CPU)
    res = torch.randn(6, 4, 32, dtype=torch.float32, device=CPU)
    g = torch.randn(32, dtype=torch.float32, device=CPU)
    b = torch.randn(32, dtype=torch.float32, device=CPU)
    y, h = add_ln_kernel.forward(x, res, g, b)
    wy, wh = add_ln_ref(x, res, g, b)
    assert h.dtype == torch.float32 and y.shape == x.shape
    assert _maxerr(y, wy) <= TOL and _maxerr(h, wh) <= 1e-6

    qkv = torch.randn(5, 3, 2, 16, dtype=torch.float16, device=CPU)
    cos, sin = rope_angle_tables(5, 8, device=CPU)
    snapshot = qkv.clone()
    got = rope_kernel.forward(qkv, cos, sin)
    assert got is qkv and torch.equal(got[:, 2], snapshot[:, 2])
    assert _maxerr(got, rope_ref(snapshot, cos, sin)) <= TOL

    q = torch.randn(2, 3, 24, 16, dtype=torch.float16, device=CPU)
    k, v = torch.randn_like(q), torch.randn_like(q)
    assert _maxerr(attn_sw_kernel.forward(q, k, v, 8), attn_sw_ref(q, k, v, 8)) <= TOL
    assert backends.compile_count() == 0 and backends.compiled_keys() == ()


def test_odd_block_sizes_fall_back_without_warning():
    """N/K 不能被任何候选块宽整除 → plan 判不可表达 → 静默回退（不发警告、不记阻塞）。"""
    A = torch.randn(8, 12, dtype=torch.float16, device=CPU)
    W = torch.randn(20, 12, dtype=torch.float16, device=CPU)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = gemm_kernel.forward(A, W, None, act="relu")
    assert _maxerr(got, gemm_ref(A, W, None, act="relu")) <= TOL
    assert not caught, "回退是设计内行为，不该发警告"
    assert backends.blockers() == {}


def test_fp32_inputs_fall_back_and_stay_stable_across_calls():
    """fp32 输入（方言只吃 fp16）走回退，且反复调用给出逐位一致的结果——主线可以一路跑到底。"""
    A = torch.randn(4, 16, dtype=torch.float32, device=CPU)
    W = torch.randn(8, 16, dtype=torch.float32, device=CPU)
    outs = [gemm_kernel.forward(A, W, None, act="relu") for _ in range(3)]
    assert all(torch.equal(o, outs[0]) for o in outs)
    assert backends.compile_count() == 0
