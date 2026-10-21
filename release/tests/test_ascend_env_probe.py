"""tests/test_ascend_env_probe.py — P1-1g G-gate 守卫：ascend 探针的 910B 形态不回潮。

【做什么】盯死 ascend_env._compile_probe 的四条线：① 静态守卫——ascend 分支源码里零
SimtVF/T.Parallel（950 SIMT 载体在 910B(dav-2201) 必编不出，wave2 因此把九件内核全挡成
torch eager，train_step "no grad_fn"），并锚定"手工挂即评 __annotations__ 再过 prim_func"
的构造形态（本模块有 future annotations 前导，装饰器+字符串注解在 py3.10 会在构造期就
TypeError——两个病灶都必须守得住）；② cpu 探针在 host 依然可用（backend_available("cpu")
判真、不登记探针阻塞）——这条是"cpu 路一行没动"的直接凭据；③ 探针在双方言都走得通构造
路：cpu 真构造出 PrimFunc，ascend 在不带 C++ ascend 算子注册的 host 上只许败在
"ascend_copy is not registered" 这一环境事实上，**不许**败在载体或构造语法上（带 ascend
注册的环境则要求构造全绿且 IR 无 SIMT 面）；④ 锚定 host 无卡行为：_npu_present() 为假时
ascend 必须因**设备门**判假——本次修的是"有卡也编不过"，没卡的结论不许变。
【怎么做】① 用 ast 解析 inspect.getsource(_compile_probe)，按行号切出
`if target == TARGET_ASCEND:` 分支体做子串断言——纯静态，CI host 档常驻；③ 直接调
_compile_probe 并 monkeypatch tilelang.compile 拦下发件：cpu 检查 PrimFunc 与编译参数，
ascend 按环境两条路各验其形。真编译判决（容器 target=ascend PASS）另档留
reconcile/T_RESULT.md，本文件不重复。
【为什么】被否方案一：只在容器实编当守卫——host 测试跑不到 CANN 容器，防回潮必须日常
常驻；被否方案二：只查"分支里有没有 SimtVF 字样"——旧探针在容器连构造都没进（py3.10 的
字符串注解被 typing._type_check 拒），单防载体回潮会漏掉构造形态回潮，故正向锚死
`probe_asc.__annotations__` 与 `t.prim_func(probe_asc)` 两个要件。
"""
import ast
import inspect
import re

import pytest

from ascend.kernels import ascend_env

_SRC = inspect.getsource(ascend_env._compile_probe)
#: 950 SIMT 面载体黑名单：出现在 ascend 分支即视为 G-gate 回潮（R14 防假绿）
_BANNED = ("SimtVF", "Parallel")


def _ascend_branch(source: str) -> str:
    """从 _compile_probe 源码里切出 `if target == TARGET_ASCEND:` 分支体的原文。

    白话：不猜行号——用 ast 找到那个 if 节点，取它的 body 起止行，回到原文切片。
    分支体从 def 起，所以分支上方的说明注释（含病灶名）不会误伤黑名单。
    """
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)  # getsource 的函数体本就从第 0 列起，可直接 parse
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and getattr(node.test.left, "id", "") == "target"
                and getattr(node.test.comparators[0], "id", "") == "TARGET_ASCEND"):
            start = node.body[0].lineno - 1
            end = node.body[-1].end_lineno
            return "".join(lines[start:end])
    raise AssertionError("未找到 `if target == TARGET_ASCEND:` 分支——探针结构被改了？")


# ---------------------------------------------------------------- ① 静态守卫（不依赖 tilelang）
def test_probe_ascend_branch_has_no_simt_carrier():
    """ascend 分支零 SimtVF/T.Parallel：950 载体回潮即红（P1-1g 主守卫）。"""
    branch = _ascend_branch(_SRC)
    for token in _BANNED:
        assert token not in branch, f"ascend 探针回潮 950 载体 [{token}]：910B 必编不出，门会挡死全部内核"


def test_probe_ascend_branch_is_910b_legal_shape():
    """ascend 分支必须是已判决的 910B 形态与构造要件，缺一即红。"""
    branch = _ascend_branch(_SRC)
    assert "t.serial(" in branch, "ascend 探针缺 T.serial 标量循环（910B 合法形态要件）"
    assert branch.count("t.copy(") >= 2, "ascend 探针需一次搬入、一次搬出（与 cpu 探针同构）"
    assert "t.alloc_shared" in branch, "ascend 探针需经 UB（alloc_shared）中转"
    # 构造形态锚：future-annotations 前导下，装饰器+def 处注解=字符串注解，py3.10 构造期必炸；
    # 只许"先 def、手工挂即评 __annotations__、再显式过 prim_func"这一形态。
    assert "probe_asc.__annotations__" in branch, "ascend 探针丢了即评注解挂载（py3.10 构造地雷回潮）"
    assert "t.prim_func(probe_asc)" in branch, "ascend 探针未走显式 prim_func 应用"
    assert "@t.prim_func" not in branch, "ascend 分支不许再用装饰器形态（字符串注解在 py3.10 会被拒）"


# ---------------------------------------------------------------- ② cpu 探针仍可用
def test_cpu_probe_still_available():
    """cpu 探针在 host 判真且探针路零阻塞——'TARGET_CPU 路一行未动'的直接凭据。"""
    ascend_env.reset()
    try:
        assert ascend_env.backend_available("cpu") is True, \
            f"cpu 探针失效：{ascend_env.blockers()}"
        assert "cpu:probe" not in ascend_env.blockers()
        assert ascend_env.active_backend("cpu", device="cpu") == ascend_env.TILELANG
    finally:
        ascend_env.reset()


# ---------------------------------------------------------------- ③ 双方言构造（不实编）
def test_cpu_probe_constructs_and_dispatches_compile(monkeypatch):
    """cpu 探针真构造出 PrimFunc 并按 compile_kwargs("cpu") 递交（不实编）。"""
    tilelang = pytest.importorskip("tilelang")
    captured = []
    monkeypatch.setattr(tilelang, "compile",
                        lambda prog, **kw: captured.append((prog, kw)))
    ascend_env._compile_probe(ascend_env.TARGET_CPU)  # 不许抛
    assert captured, "cpu 探针没走到 tilelang.compile"
    _prog, kwargs = captured[-1]
    assert kwargs == {"out_idx": [1], **ascend_env.compile_kwargs(ascend_env.TARGET_CPU)}


def test_ascend_probe_never_fails_on_carrier_or_syntax(monkeypatch):
    """ascend 探针在 host 只许败在 op 注册这一环境事实，不许败在载体/构造语法。

    白话：不带 ascend C++ 算子注册的机器（本机）上，`ascend_copy is not registered` 是
    已知环境账（模块 docstring 自陈），构造必须已经"越过"语法与载体两关才碰到它；
    带注册的环境（云端 NPU 机/容器）则要求直接构造成功、IR 干净。真编译判决在
    reconcile/T_RESULT.md（容器 T-PROBE-ASC-COMPILE-PASS）。
    """
    tilelang = pytest.importorskip("tilelang")
    pytest.importorskip("tilelang.ascend.language")
    captured = []
    monkeypatch.setattr(tilelang, "compile",
                        lambda prog, **kw: captured.append((prog, kw)))
    try:
        ascend_env._compile_probe(ascend_env.TARGET_ASCEND)
    except Exception as exc:  # noqa: BLE001  host 无 ascend op 注册是预期环境事实
        msg = f"{type(exc).__name__}: {exc}"
        for token in _BANNED:
            assert token not in msg, f"ascend 探针败因碰到 950 载体：{msg[:200]}"
        assert "not registered" in msg or "ascend" in msg.lower(), \
            f"ascend 探针败因不是已知环境事实（op 未注册），需排查：{msg[:300]}"
        return
    # 能构造成功的环境：必须走到了 compile，且件里没有 SIMT 面载体
    assert captured, "ascend 探针构造成功却没递交 compile"
    prog, kwargs = captured[-1]
    assert kwargs == {"out_idx": [1], **ascend_env.compile_kwargs(ascend_env.TARGET_ASCEND)}
    ir = str(prog).lower()
    assert "simtvf" not in ir and "parallel" not in ir, f"ascend 探针 IR 含并行/SIMT 载体：{ir[:300]}"


# ---------------------------------------------------------------- ④ host 无卡行为锚
def test_host_without_npu_ascend_still_false():
    """host 无 npu 时 backend_available("ascend") 必须仍为 False，且败因是设备门。

    白话：这次修的是"有卡也编不过"；没卡的机器上结论不许变，也不许变成 True——
    阻塞账里记的那句必须是"本机没有可用昇腾设备"，而不是探针编译失败。
    """
    if ascend_env._npu_present():
        pytest.skip("本机存在 npu，设备门不在本用例射程（有卡判决见 reconcile/T_RESULT.md）")
    ascend_env.reset()
    try:
        assert ascend_env.backend_available("ascend") is False
        assert ascend_env.active_backend("ascend", device="cpu") == ascend_env.TORCH_EAGER
        why = ascend_env.blockers().get("ascend:probe", "")
        assert re.search("没有可用昇腾设备|is_available", why), \
            f"ascend 判假但原因不是设备门：{why!r}"
    finally:
        ascend_env.reset()
