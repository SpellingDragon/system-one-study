"""ascend_env：p2-13 方言件的运行底座——target 开关、方言导入、编译缓存与"真走了内核"自证。

【做什么】给 ascend/kernels/ 下所有方言件统一回答四个问题："这一版按哪种方言写"、"提交给
tilelang.compile 该带哪些参数"、"这台机器能不能真跑该后端"、"某个模具是编好了还是没编成"。
【怎么做】① 对外只承认两个 target 名："ascend" 与 "cpu"。"cpu" 是对外口径，对内翻译成 TileLang
   CPU 后端的真实写法 target="c" + target_host="c" + execution_backend="cython"（实测字符串
   "cpu" 会被 determine_target 直接拒：Target kind "cpu" is not defined）；② dialect(target)
   按名字惰性返回对应语言模块，各算子件据此拿 T；③ get_compiled(key, builder) 命中缓存直返、
   未命中才真编译并把 compile_count 加一，builder 抛异常则登记阻塞点、发一次 warning、返回
   None，调用方随即落回 torch 写法；④ backend_available() 除设备探测外还要做一次最小件
   lowering 自证，因为"能 import 方言"与"算子已注册"是两回事。
【为什么】把"后端到底可不可用"这件环境不确定性与算子逻辑解耦，缺件只是环境事实，不该让整条
   前向/反向链断掉。语义与 sys1/kernels/backends.py 对齐（同名四量：blockers/compile_count/
   compiled_keys/get_compiled），但对齐是"照抄口径"不是"照抄代码"：本域是三栈与推理的公共
   底座，反向依赖 sys1 会把底座吊在算子层上，故自带一份。被否方案一：直接 import
   sys1.kernels.backends 复用——底座域依赖使用方，层次倒挂且会把 MPS 占用状态混进来；被否
   方案二：不设 lowering 自证，只看 import 成不成功——本机正是这种误判的受害者（方言能
   import，一编译就报 ascend_copy is not registered），少了这一步就会把"环境没装对"误报成
   "内核写错了"。
"""
from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from typing import Any

import torch

#: 强制后端选择的环境变量；取值 auto / tilelang / torch（CI 用 torch 跑纯语义回归）
ENV_BACKEND = "DMLAYA_ASCEND_BACKEND"
#: 默认编译目标环境变量；取值 ascend / cpu，未设置时按 cpu（本地无卡，落在安全侧）
ENV_TARGET = "DMLAYA_ASCEND_TARGET"
TILELANG = "tilelang"
TORCH_EAGER = "torch_eager"

TARGET_ASCEND = "ascend"
TARGET_CPU = "cpu"
TARGETS = (TARGET_ASCEND, TARGET_CPU)

# 进程级状态：可用性结论（按 target 分开记）、编译产物缓存、阻塞账、计数。测试用 reset() 清零。
_probe: dict[str, bool | None] = {TARGET_CPU: None, TARGET_ASCEND: None}
_compiled: dict[str, Any] = {}
_blockers: dict[str, str] = {}
_counts: dict[str, int] = {"compile": 0}


def normalize_target(raw: str | None) -> str:
    """把外部传来的 target 名收进白名单，其余一律报错，不做"看起来像昇腾"的猜测。

    白话：只认两种写法——昇腾和普通 CPU。别名（asc、npu、c）在表里对得上就翻译过来，
    对不上就直接退回，免得一个拼错的字符串悄悄改了整条链的编译目标。

    :raises ValueError: 名字不在白名单内时抛出，并列出可用取值。
    """
    key = (raw or os.environ.get(ENV_TARGET) or TARGET_CPU).strip().lower()
    aliases = {
        "ascend": TARGET_ASCEND, "asc": TARGET_ASCEND, "npu": TARGET_ASCEND,
        "cpu": TARGET_CPU, "c": TARGET_CPU, "host": TARGET_CPU,
    }
    if key not in aliases:
        raise ValueError(f"未知 target={raw!r}，只支持 {'/'.join(TARGETS)}（别名 asc/npu、c/host）")
    return aliases[key]


def compile_kwargs(target: str | None) -> dict[str, Any]:
    """给出 tilelang.compile 的关键字参数；昇腾走 target="ascend"，CPU 走 c 后端三件套。

    白话：把"交给编译器的那句配置"集中写在一处。昇腾那侧只有一个 target 名；CPU 那侧必须
    连给三个值（自己是 c、宿主是 c、执行用 cython 直调），少一个就编不出能在本机跑的产物。
    """
    if normalize_target(target) == TARGET_ASCEND:
        # 口径来源：tilelang/ascend/target.py（kind="ascend"）与 examples/ascend/*.py 的
        # 统一写法 tilelang.compile(prog, target="ascend", out_idx=-1)。
        return {"target": "ascend"}
    return {"target": "c", "target_host": "c", "execution_backend": "cython"}


def dialect(target: str | None):
    """返回该 target 对应的 TileLang 语言模块（各算子件里的 T 就从这里来）。

    白话：两种方言各有各的"工具箱"，这里按需把它们取出来，避免在同一个文件里把两套同名
    原语混在一起，编出来的东西到底是哪种方言也就一目了然。
    """
    name = normalize_target(target)
    if name == TARGET_ASCEND:
        import tilelang.ascend.language as t_asc

        return t_asc
    import tilelang.cpu.language as t_cpu

    return t_cpu


def _npu_present() -> bool:
    """本机是否存在可用的昇腾设备（torch 侧判据；TileKernels 用 /dev/davinci_manager 是另一套）。

    白话：先问框架"有没有卡能用"。装了 torch_npu 但没插卡、或者插了卡但驱动不对，这里都会
    答"没有"，于是后面就不去白费力气编译昇腾产物。
    """
    return hasattr(torch, "npu") and bool(torch.npu.is_available())


def _backend_pref() -> str:
    """读取 env 里的后端偏好；torch 表示强制走普通写法（供 CI 与排障用）。

    白话：允许外部一句话命令"别用自写内核"，这样在没装好环境或想单独看回退行为时不用改代码。
    """
    raw = (os.environ.get(ENV_BACKEND) or "auto").strip().lower()
    if raw in ("torch", "eager", TORCH_EAGER):
        return TORCH_EAGER
    if raw == TILELANG:
        return TILELANG
    return "auto"


def backend_available(target: str | None) -> bool:
    """该 target 在本机是否真可用（结论按 target 缓存一次；env 强制 torch 时一律判否）。

    判据分两段：先问"这台机器有没有对应的设备"，再真编译一个两行的小件做 lowering 自证。
    第二段是关键——本机实测能 import tilelang.ascend.language，但一提交编译就报
    Operator tl.tileop.ascend_copy is not registered；只看 import 会把"环境没装对"误判成
    "内核写错了"，也会把"编不出"的真相推到云端开机那天。

    白话：不猜，直接试一小块活儿能不能干成；试过一次就记住结论，不反复试探。
    """
    name = normalize_target(target)
    if _backend_pref() == TORCH_EAGER:
        return False
    if _probe[name] is not None:
        return bool(_probe[name])
    if name == TARGET_ASCEND and not _npu_present():
        record_blocker(f"{name}:probe", "本机没有可用昇腾设备（torch.npu.is_available()=False）")
        _probe[name] = False
        return False
    ok = True
    try:
        import tilelang  # noqa: F401  只做可导入性探测，真编译在下面
        dialect(name)
        _compile_probe(name)
    except Exception as exc:  # noqa: BLE001  缺件表现说不准：登记后按不可用处理，不带崩主线
        record_blocker(f"{name}:probe", exc)
        ok = False
    _probe[name] = ok
    return ok


def _compile_probe(target: str) -> None:
    """编译一个极小件做 lowering 自证；失败就抛，由调用方登记成阻塞点。

    白话：真正试一次"能不能把小模具打出来"。打得出来才说明这套方言在本机真能用；打不出来
    就把编译器原话留着，方便判断是版本不对还是装错了包。
    """
    import tilelang

    t = dialect(target)
    if target == TARGET_ASCEND:
        @t.prim_func
        def probe_asc(A: t.Tensor((64,), "float32"), C: t.Tensor((64,), "float32")):
            """昇腾方言的最小自证件：一次搬入、一次并行乘二、一次搬出。

        白话：不测任何业务算式，只看这台机器能不能把"搬进来、每人算一格、搬出去"这条路
        编译成产物；能过就说明方言与后端是齐的，之后再谈算子写得对不对。
        
            """
            with t.Kernel(1) as bx:  # noqa: F841  方言要求块索引存在，探针里不用它
                a = t.alloc_shared((64,), "float32")
                c = t.alloc_shared((64,), "float32")
                t.copy(A[0:64], a)
                with t.SimtVF(threads=64):
                    for i in t.Parallel(64):
                        c[i] = a[i] * 2
                t.copy(c, C[0:64])

        prog = probe_asc
    else:
        @t.prim_func
        def probe_cpu(A: t.Tensor((64,), "float32"), C: t.Tensor((64,), "float32")):
            """CPU(c) 后端的最小自证件：与昇腾探针同一套动作，串行逐格乘二。

        白话：先证明"能编译、能真跑、结果逐位对"这三件事在 CPU 上成立，语义对拍才有地基；
        它只当环境体检用，不参与任何业务算式的验收。
        
            """
            with t.Kernel(1) as bx:  # noqa: F841
                al = t.alloc_local((64,), "float32")
                t.copy(A[0], al)
                for i in t.serial(64):
                    al[i] = al[i] * 2
                t.copy(al, C[0])

        prog = probe_cpu
    tilelang.compile(prog, out_idx=[1], **compile_kwargs(target))


def active_backend(target: str | None = None, device: torch.device | str | None = None) -> str:
    """给出该 target + 设备上应走的实现："tilelang" 或 "torch_eager"。

    昇腾产物只能在 npu 设备上发射，CPU 产物只在 cpu 设备上跑：设备与 target 不搭时判成
    torch_eager，避免"编译成功、发射到错的设备"这种更难查的失败。

    白话：先看在哪儿跑，再问那套手写工具在这台机器上能不能用，两个都对上才用自写内核。
    """
    name = normalize_target(target)
    if device is not None:
        want = "npu" if name == TARGET_ASCEND else "cpu"
        if torch.device(device).type != want:
            return TORCH_EAGER
    return TILELANG if backend_available(name) else TORCH_EAGER


def record_blocker(key: str, err: BaseException | str) -> None:
    """登记一个阻塞点并发一次性 warning——回退必须"看得见"，不许静默换实现（R14 防假绿）。

    白话：把"哪一步没走通、为什么"记到一本小账上，并大声提醒一次，同一件事不重复啰嗦。
    """
    text = f"{type(err).__name__}: {err}" if isinstance(err, BaseException) else str(err)
    first = key not in _blockers
    _blockers[key] = (text.splitlines()[0] if text else "")[:400]
    if first:
        warnings.warn(f"方言后端 [{key}] 不可用，已回退 torch eager：{text}",
                      RuntimeWarning, stacklevel=2)


def blockers() -> dict[str, str]:
    """已登记的阻塞点快照（key → 首次错误摘要），供运行记录与排障直接引用。

    白话：把那本小账摊开给人看，一眼知道哪几处没走通、分别卡在哪一步，不用回头翻日志。
    """
    return dict(_blockers)


def make_key(target: str | None, key: str) -> str:
    """给缓存键加 target 前缀，返回 "ascend|xxx" / "cpu|xxx" 形态。

    白话：同一套模具在两种后端下是两个不同的东西，名字前面带上后端标识，货架上就不会拿错。
    """
    return f"{normalize_target(target)}|{key}"


def get_compiled(key: str, builder: Callable[[], Any], target: str | None = None) -> Any | None:
    """按 key 取编译产物；未命中才真编译并计数；不可用或失败返回 None（= 该走回退了）。

    key 一律经 make_key 带 target 前缀，否则同形状的 cpu/ascend 两份模具会在缓存里互相顶掉。
    **同一 key 只编译一次**：多批复用同一句柄就靠 compile_count() 不变来验收。

    白话：做好一次的活不重复做；做不动的活记下来、提醒一次，然后换条普通路子继续往前走。
    """
    name = normalize_target(target)
    full = make_key(name, key)
    if full in _compiled:
        return _compiled[full]
    if full in _blockers or not backend_available(name):
        return None
    try:
        kernel = builder()
    except Exception as exc:  # noqa: BLE001  编译失败：登记 + 回退，主线不中断
        record_blocker(full, exc)
        return None
    _compiled[full] = kernel
    _counts["compile"] += 1
    return kernel


def compile_count() -> int:
    """进程内真实发生过的编译次数（缓存命中不计入）。

    白话：只数"真的动手做过"的次数，拿现成成品不算——用它就能证明同一份东西没被反复重做。
    """
    return _counts["compile"]


def compiled_keys() -> tuple[str, ...]:
    """已缓存的编译产物键（只读快照，带 target 前缀）。

    对拍测试用它确认"内核真跑过"：只有编译成功并缓存过的 key 才出现在这里，回退路径不留痕迹，
    所以没人能把回退伪装成内核通过。

    白话：把"哪些模具已开好、正摆在架子上"列一张清单给人核对，没开成功的一个也不虚列。
    """
    return tuple(_compiled)


def default_target() -> str:
    """当前默认编译目标（env 未设置时为 cpu）。

    白话：没人点名的时候按哪个后端来。本机没卡，所以默认落在能真跑对拍的 CPU 侧。
    """
    return normalize_target(None)


def set_backend_preference(value: str | None, target: str | None = None) -> None:
    """运行期改写 env 偏好并清探测缓存（测试与脚本用）；target 非空时同时改默认目标。

    白话：临时改一句"接下来用哪种写法、按哪种方言编"，改完把之前记下的答案作废重问。
    """
    if value is None:
        os.environ.pop(ENV_BACKEND, None)
    else:
        os.environ[ENV_BACKEND] = value
    if target is not None:
        os.environ[ENV_TARGET] = normalize_target(target)
    for name in TARGETS:
        _probe[name] = None


def reset() -> None:
    """清空全部进程级状态（探测结论、编译缓存、阻塞账、计数、env 偏好）。

    单测之间必须互不污染：计数与阻塞账都是全局量，不留复位口就会互相串。

    白话：把小账、成品架子、计数器全部归零，让下一次判断从头开始，不被上一条用例带偏。
    """
    for name in TARGETS:
        _probe[name] = None
    _compiled.clear()
    _blockers.clear()
    _counts["compile"] = 0
    os.environ.pop(ENV_BACKEND, None)
    os.environ.pop(ENV_TARGET, None)
