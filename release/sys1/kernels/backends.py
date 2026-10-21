"""backends：设备探测 + tilelang/torch 双路径分发 + 编译失败回退（教师版只有 MPS 一条后端）。

【做什么】集中回答三个问题——"这批数据该在哪个设备上跑"、"这台机器能不能用 TileLang 的 Metal
方言"、"某个内核编译不出来时怎么办"。四个算子入口只负责问，不负责判断。
【怎么做】① resolve_device()：按 mps → cpu 的顺序探测（cuda/npu 的判据留成注释扩展位）；
② tilelang_available()：先认环境变量 DMLAYA_KERNEL_BACKEND（torch=强制回退，供 CI 与排障），
   再探 tilelang 与 metal 方言能否导入、本机是否有 MPS，结论按进程缓存一次；
③ get_compiled(key, builder)：命中缓存直接返回，未命中才真正编译并把 compile_count 加一
   （"一次编译多批复用"就靠这个计数自证）；builder 抛异常时登记阻塞点、发一次 RuntimeWarning、
   返回 None，调用方随即走 torch eager；同一 key 的失败只登记与告警一次，不反复重试；
④ active_backend()：给入口层一个二元结论（"tilelang" 或 "torch_eager"）。
【为什么】把"方言能不能用"这个环境不确定性与算子逻辑解耦：缺原语只是环境事实，不该让整条前向
链断掉。被否方案一：四个算子各自 try/except 编译——四处重复、warning 刷屏、阻塞点也没法集中
登记；被否方案二：探测失败直接抛错退出——等于把 W0 的方言风险升级成主线阻塞，违反父 design
D3"内核不阻主线"。
"""
from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from typing import Any

import torch

#: 强制后端选择的环境变量；取值 auto / tilelang / torch（CI 用 torch 跑纯 CPU 回归）
ENV_BACKEND = "DMLAYA_KERNEL_BACKEND"
TILELANG = "tilelang"
TORCH_EAGER = "torch_eager"

# 进程级状态：探测结论、编译产物缓存、失败登记、编译计数。测试用 reset() 清零。
_probe: dict[str, Any] = {"available": None}
_compiled: dict[str, Any] = {}
_blockers: dict[str, str] = {}
_counts: dict[str, int] = {"compile": 0}


class KernelUnavailable(RuntimeError):
    """方言不可用时抛出的类型；本模块默认**不**抛它（回退优先），仅保留给强制模式的调用方。"""


def _preference() -> str:
    """读取环境变量里的后端偏好，未设置或取值非法都按 auto 处理。

    白话：允许外部一句话指定"别用自写内核，用普通写法"，方便在没有相应硬件的机器上做回归。
    """
    raw = (os.environ.get(ENV_BACKEND) or "auto").strip().lower()
    if raw in ("torch", "eager", TORCH_EAGER):
        return TORCH_EAGER
    if raw in ("tilelang", TILELANG):
        return TILELANG
    return "auto"


def resolve_device(preferred: str | torch.device | None = None) -> torch.device:
    """给出计算设备：显式指定优先，其次 mps，最后 cpu。

    探测顺序在教师版里被刻意裁短。TODO(学生版)：按 cuda → mps → npu 的完整顺序补齐，昇腾判据
    用 `os.path.exists("/dev/davinci_manager")`，并在此收敛设备能力（核数/显存/可用位宽白名单）。

    白话：先看有没有点名要的地方，没有就挑本机最快的那个；都没有就退回普通 CPU 写法。
    """
    if preferred is not None and str(preferred) not in ("", "auto"):
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tilelang_available() -> bool:
    """本机是否具备可用的 TileLang Metal 方言（结论缓存一次；env 强制 torch 时直接判否）。

    三段判据缺一不可：能 import tilelang、能 import metal 方言入口、本机真有 MPS 设备。
    任何一段失败都会记进 blockers()，方便运行记录里直接抄出阻塞点原文。

    白话：先问一句"这台机器上那套手写内核工具到底能不能用"，问过就记住答案，不反复试探。
    """
    if _probe["available"] is not None:
        return bool(_probe["available"])
    if _preference() == TORCH_EAGER:
        _probe["available"] = False
        return False
    ok: bool
    try:
        import tilelang  # noqa: F401  仅探测可导入性，真正编译在各 _mps 模块里做
        from tilelang.metal.target import check_metal_availability

        ok = bool(torch.backends.mps.is_available()) and bool(check_metal_availability())
    except Exception as exc:  # noqa: BLE001  缺件的表现形式说不准：登记后按不可用处理，绝不把主线带崩
        record_blocker("tilelang_probe", exc)
        ok = False
    _probe["available"] = ok
    return ok


def active_backend(device: torch.device | str | None = None) -> str:
    """返回该设备上应走的实现："tilelang" 或 "torch_eager"。

    只允许 mps 设备走方言（Metal 后端产出的二进制只能在 MPS 上发射）；设备是 cpu、或探测
    不可用、或被 env 强制，都判成 torch_eager。

    白话：先看在哪儿跑，再问方言能不能用，两个条件都满足才用自写内核，否则用普通写法。
    """
    if device is not None and torch.device(device).type != "mps":
        return TORCH_EAGER
    return TILELANG if tilelang_available() else TORCH_EAGER


def record_blocker(key: str, err: BaseException | str) -> None:
    """登记一个阻塞点并发出一次性 warning（回退必须"看得见"，不许静默换实现）。

    白话：把"哪一步没走通、为什么"记到一本小账上，并大声提醒一次，同一件事不重复啰嗦。
    """
    text = f"{type(err).__name__}: {err}" if isinstance(err, BaseException) else str(err)
    first = key not in _blockers
    _blockers[key] = text
    if first:
        warnings.warn(
            f"TileLang MPS 后端 [{key}] 不可用，已回退 torch-MPS eager：{text}",
            RuntimeWarning,
            stacklevel=2,
        )


def blockers() -> dict[str, str]:
    """已登记的阻塞点快照（key → 首次错误摘要），供运行记录与排障直接引用。

    白话：把那本小账摊开给人看，一眼就知道是哪几处没走通、分别卡在哪一步，不用回头翻日志。
    """
    return dict(_blockers)


def get_compiled(key: str, builder: Callable[[], Any]) -> Any | None:
    """按 key 取编译产物；未命中则调 builder() 编译一次并计数；不可用或失败返回 None。

    返回 None 就是"该走回退了"的信号，调用方不需要理解方言细节。**同一 key 只编译一次**——
    多批不同长度复用同一句柄，spec 的"多批复用"场景正是用 compile_count() 不变来验收的。
    失败过的 key 记进 _blockers 并直接判不可用，不再逐次重试编译。

    白话：做好一次的活不重复做；做不动的活记下来、提醒一次，然后换条普通路子继续往前走。
    """
    if key in _compiled:
        return _compiled[key]
    if key in _blockers:
        return None
    if not tilelang_available():
        return None
    try:
        kernel = builder()
    except Exception as exc:  # noqa: BLE001  编译/预热失败：登记 + 回退，主线不中断
        record_blocker(key, exc)
        return None
    _compiled[key] = kernel
    _counts["compile"] += 1
    return kernel


def compile_count() -> int:
    """进程内真实发生过的编译次数（缓存命中不计入）。

    白话：只数"真的动手做过"的次数，拿现成成品不算——用它就能证明同一份东西没被反复重做。
    """
    return _counts["compile"]


def compiled_keys() -> tuple[str, ...]:
    """已缓存的编译产物键（只读快照）。

    对拍测试用它确认"内核真的跑过"：只有真正编译成功并缓存过的 key 才会出现在这里，回退路径
    不会留下任何痕迹，所以没人能把回退伪装成内核通过。

    白话：把"哪些模具已经开好、正摆在架子上"列一张清单给人核对，没开成功的一个也不虚列上去。
    """
    return tuple(_compiled)


def set_backend_preference(value: str | None) -> None:
    """运行期改写 env 偏好并清掉探测缓存（测试与脚本用，等价于设 DMLAYA_KERNEL_BACKEND）。

    白话：临时改一句"接下来用哪种写法"，改完就把之前记下的答案作废重问。
    """
    if value is None:
        os.environ.pop(ENV_BACKEND, None)
    else:
        os.environ[ENV_BACKEND] = value
    _probe["available"] = None


def reset() -> None:
    """清空全部进程级状态（探测结论、编译缓存、阻塞账、计数、env 偏好）。

    单测之间必须互不污染：计数与阻塞账都是全局量，不留复位口就会互相串。

    白话：把小账、成品架子、计数器全部归零，让下一次判断从头开始，不被上一条用例留下的痕迹带偏。
    """
    _probe["available"] = None
    _compiled.clear()
    _blockers.clear()
    _counts["compile"] = 0
    set_backend_preference(None)
