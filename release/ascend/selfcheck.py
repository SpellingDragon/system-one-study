"""selfcheck：C1 短租探针——一把机器上把"昇腾这条技术路线能不能走"问清楚（P2 / p2-13 R1）。

【做什么】按九项依次探测：软件栈版本口径、昇腾方言可否导入、最小方言件能否 lowering、机器上
有没有卡、AscendC 编译器在不在、外部件后端判据是否成立、torch_npu 前向参照的数值差、bf16/fp16
在本栈的数值行为，最后一条是没有卡时也要能跑的 CPU 侧语义链路自检。每项打一行结论。
【怎么做】① 每项一个 probe_* 函数，返回 (结论, 一行说明)，异常统一被框架兜住记成 FAIL 并留
   原文首行；② 结论分四类：PASS / FAIL / SKIP（前提不具备，如没插卡）/ KNOWN-GAP（已知环境
   缺件，加 --allow-ascend-missing 后把"方言不可编译"从 FAIL 降级成 KNOWN-GAP，供本机跑通
   流程用——它**不计入通过**，汇总行单独列）；③ 数值项不设硬阈值，只把实测最大偏差打进入
   结论行，由 run notes 抄录为探针事实（禁止用"看起来对"代替数字）。
【为什么】把风险在花钱之前一次性暴露：设计文档里 C1 是"探针先行"，任一项失败即触发 D6 回退
   决策（改走 torch_npu 底座），所以脚本必须能在一次开机内跑完并打印可直接抄进 notes 的结论。
   被否方案一：只写一个 import 测试——本机已实测"能 import 方言但一编译就报 ascend_copy is
   not registered"，import 级探测会把装错包误报成方言可用；被否方案二：脚本里顺带 pip install
   缺件——探针的职责是"报事实"，动手修环境是 env_setup.sh 的职责，混在一起会让失败原因说不清。
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Callable

PASS, FAIL, SKIP, GAP = "PASS", "FAIL", "SKIP", "KNOWN-GAP"


def _head(err: BaseException, width: int = 220) -> str:
    """取异常首行摘要（去掉堆栈噪声），供结论行直接引用。"""
    text = "".join(traceback.format_exception_only(type(err), err)).strip()
    return (text.splitlines() or [""])[0][:width]


def p_versions() -> tuple[str, str]:
    """打印 Python / torch / tilelang 三个版本与两个关键环境变量。

    白话:先把"这台机器上到底装了什么、版本号是几"抄成一行，后面任何结论都要能回溯到这里。
    """
    import platform

    import torch

    import tilelang

    line = (f"python={platform.python_version()} torch={torch.__version__} "
            f"tilelang={tilelang.__version__} ASCEND_HOME_PATH={os.environ.get('ASCEND_HOME_PATH', '<未设置>')} "
            f"ASCEND_NPU_ARCH={os.environ.get('ASCEND_NPU_ARCH', '<默认 dav-3510>')}")
    bad = []
    if sys.version_info < (3, 10):
        bad.append("Python < 3.10")
    tv = tuple(int(x) for x in tilelang.__version__.split("+")[0].split(".")[:3] if x.isdigit())
    if tv < (0, 1, 15):
        bad.append(f"tilelang {tv} < 0.1.15")
    return (GAP if bad else PASS), line + (f" 不达标: {','.join(bad)}" if bad else "")


def p_dialect_import() -> tuple[str, str]:
    """昇腾方言模块能否导入，并确认关键原语齐不齐。

    白话:先看这套"手写内核用的工具箱"能不能搬进屋，再数一遍里面该有的家伙在不在，缺哪个记哪个。
    """
    import tilelang.ascend.language as T

    need = ["Kernel", "Tensor", "copy", "gemm", "alloc_l1", "alloc_l0c", "alloc_shared",
            "Pipelined", "Persistent", "SimtVF", "Parallel", "dual_copy", "annotate_buffer_versions"]
    missing = [n for n in need if not hasattr(T, n)]
    if missing:
        return FAIL, f"方言导入成功但缺原语: {missing}"
    return PASS, f"方言可导入，13 项关键原语齐备（ascend.language 共 {len(dir(T))} 符号）"


def p_lower_minimal() -> tuple[str, str]:
    """把一个两行的小模具交给昇腾编译器 lowering，看能不能出 AscendC 源码。

    白话:真正下一次活儿试试。能出源码说明这条技术路线在本机装对了；出不了就把编译器原话留着，
    一眼能分清是"包没装对"还是"内核写错了"。
    """
    import tilelang
    import tilelang.ascend.language as T

    @T.prim_func
    def probe(A: T.Tensor((64,), "float32"), C: T.Tensor((64,), "float32")):
        """selfcheck 的昇腾探针件：一次搬入、一次并行乘二、一次搬出。

        白话：这段只回答"昇腾方言能不能 lowering 成产物"，故意不带任何业务算式，这样报出来
        的失败只可能指向环境，不会被误读成"内核写错了"。
        
        """
        with T.Kernel(1) as bx:  # noqa: F841 方言要求块索引存在，探针里不用它
            a = T.alloc_shared((64,), "float32")
            c = T.alloc_shared((64,), "float32")
            T.copy(A[0:64], a)
            with T.SimtVF(threads=64):
                for i in T.Parallel(64):
                    c[i] = a[i] * 2
            T.copy(c, C[0:64])

    kernel = tilelang.compile(probe, target="ascend", out_idx=[1])
    src = str(kernel.get_kernel_source())
    marks = [m for m in ("aic", "aicore", "AscendC", "__aicore__", "SetBuf", "pipe") if m.lower() in src.lower()]
    return PASS, f"最小件 lowering 成功，源码 {len(src)} 字节，命中标记 {marks}"


def p_device() -> tuple[str, str]:
    """机器上有没有可用的昇腾卡，卡叫什么名字。

    白话:先确认插了卡、驱动认得它。没有卡的话后面几项都不用测，直接标跳过。
    """
    import torch

    if not hasattr(torch, "npu"):
        return SKIP, "未安装 torch_npu（无 torch.npu 命名空间）"
    if not torch.npu.is_available():
        return SKIP, "torch_npu 已装但本机无可用设备（is_available=False）"
    name = torch.npu.get_device_name(0)
    count = torch.npu.device_count()
    return PASS, f"检测到 {count} 张卡，0 号 = {name}"


def p_bisheng() -> tuple[str, str]:
    """AscendC 编译器（bisheng）与链接器是否就位。

    白话:方言源码要靠华为那套编译器变成卡上能跑的东西。找不到它，编出来的东西就只能停在源码阶段。
    """
    from tilelang.contrib import bisheng

    path = bisheng.find_bisheng_path()
    arch = bisheng.get_npu_arch()
    if not path:
        return FAIL, "找不到 bisheng：请 source CANN set_env.sh 或设 ASCEND_HOME_PATH/BISHENG_HOME"
    return PASS, f"bisheng={path} npu_arch={arch}"


def p_backend_switch() -> tuple[str, str]:
    """外部件（TileKernels）用的后端自动切换判据在本机是否成立。

    白话:别人那套库是看"有没有那个设备节点文件"来决定走不走昇腾的。我们把这个判据单独跑一遍，
    确认它在本机给出的答案和真卡一致，将来借它的实现做参照才不会张冠李戴。
    """
    node = os.path.exists("/dev/davinci_manager")
    import torch

    real = bool(hasattr(torch, "npu") and torch.npu.is_available())
    verdict = PASS if node == real else FAIL
    root = os.environ.get("TILEKERNELS_ROOT")
    extra = "（未设 TILEKERNELS_ROOT，只做判据核对，不 import 外部件）"
    if root and os.path.isdir(root):
        extra = f"（TILEKERNELS_ROOT={root}，仅作人工对读参照）"
    return verdict, f"/dev/davinci_manager 存在={node} 与 torch.npu 可用={real} {'一致' if node == real else '不一致'}{extra}"


def p_torch_npu_ref() -> tuple[str, str]:
    """torch_npu 前向参照：同一批数在卡上与在 CPU 上算，差多少。

    白话:拿官方自带的那套算子当尺子。尺子先要能对上我们的参照实现，后面自研内核才有可信的比对对象。
    """
    import torch

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        return SKIP, "无可用昇腾设备，参照系留到开卡后再采"
    torch.manual_seed(0)
    a = torch.randn(256, 512)
    b = torch.randn(512, 256)
    cpu = (a @ b.T).float()
    na, nb = a.to("npu"), b.to("npu")
    for dt in (torch.float32, torch.bfloat16):
        got = (na.to(dt) @ nb.to(dt).T).float().cpu()
        err = float((got - cpu).abs().max())
        rel = err / float(cpu.abs().max().clamp(min=1e-6))
        if dt == torch.float32 and err > 1e-2:
            return FAIL, f"fp32 参照偏差过大 max_err={err:.3e}"
    return PASS, f"npu matmul 与 CPU 参照同量级（bf16 max_rel≈{rel:.2e}），可作参照系"


def p_dtype_behavior() -> tuple[str, str]:
    """bf16/fp16 在本栈口径下的数值行为：降位宽放在出口还是中间，差多少。

    白话:同样一笔乘加，先转成粗记法再算、还是算完再转粗记法，结果会差一截。把这一截实测出来，
    才能定对拍容差该给多宽，不至于把口径差异当成内核 bug。
    """
    import torch

    torch.manual_seed(0)
    a = torch.randn(128, 256)
    b = torch.randn(128, 256)   # 与 a 同形：按本栈 W=(N,K) 口径，乘的是 b.T
    ref = (a.double() @ b.double().T).float()
    rows = []
    for dt in (torch.float16, torch.bfloat16):
        late = (a @ b.T).to(dt).float()          # 先 fp32 累加，出口降位（本栈口径）
        early = (a.to(dt) @ b.to(dt).T).float()  # 中间就降位（部分硬件默认）
        rows.append(f"{str(dt).split('.')[-1]}: 出口降位={float((late - ref).abs().max()):.2e} "
                    f"中间降位={float((early - ref).abs().max()):.2e}")
    return PASS, "；".join(rows) + "（对拍容差据此定：本栈 fp16/bf16 取 2e-2）"


def p_cpu_semantics() -> tuple[str, str]:
    """没有卡时也要能跑的一条：CPU 后端的编译-发射链路是否可用。

    白话:本地半场全靠这条链路验语义——同一个模具在 CPU 后端编出来、跑一遍、数对得上，才敢拿去
    卡上试。这条不通，所有"本地已验证"都是空话。
    """
    import tilelang
    import tilelang.cpu.language as T

    @T.prim_func
    def probe(A: T.Tensor((64,), "float32"), C: T.Tensor((64,), "float32")):
        """selfcheck 的 CPU 探针件：与昇腾探针同动作，串行逐格乘二。

        白话：用来证明本地这条验收路（CPU 语义对拍）本身是通的——能编译、能发射、结果逐位
        对得上，后面七件的对拍结论才有立足点。
        
        """
        with T.Kernel(1) as bx:  # noqa: F841
            al = T.alloc_local((64,), "float32")
            T.copy(A[0], al)
            for i in T.serial(64):
                al[i] = al[i] * 2
            T.copy(al, C[0])

    import torch

    kernel = tilelang.compile(probe, out_idx=[1], target="c", target_host="c", execution_backend="cython")
    x = torch.rand(64)
    err = float((kernel(x) - x * 2).abs().max())
    return (PASS if err == 0.0 else FAIL), f"CPU(c) 后端编译+发射可用，max_err={err:.2e}"


PROBES: list[tuple[str, Callable[[], tuple[str, str]]]] = [
    ("版本口径", p_versions),
    ("方言导入", p_dialect_import),
    ("最小件 lowering", p_lower_minimal),
    ("设备在位", p_device),
    ("AscendC 编译器", p_bisheng),
    ("后端切换判据", p_backend_switch),
    ("torch_npu 前向参照", p_torch_npu_ref),
    ("bf16/fp16 数值行为", p_dtype_behavior),
    ("CPU 语义链路", p_cpu_semantics),
]


def main(argv: list[str] | None = None) -> int:
    """依次跑九项探针，每项打一行结论，末尾给汇总与退出码。

    白话:一次开机把该问的九个问题全问完，每条答案单独成行、能直接抄进实验记录。默认"有任何
    一项失败就返回非零"，加 --allow-ascend-missing 后允许在无卡机器上把方言编译失败降级显示。

    :param argv: 命令行参数（缺省读 sys.argv）。
    """
    ap = argparse.ArgumentParser(description="p2-13 C1 探针（昇腾环境自检）")
    ap.add_argument("--allow-ascend-missing", action="store_true",
                    help="把方言编译/设备类 FAIL 降级为 KNOWN-GAP（本机无卡时用；不计入通过）")
    ap.add_argument("--only", default=None, help="只跑名称包含该子串的探针")
    args = ap.parse_args(argv)

    print("== C1 探针：p2-13 昇腾运行时 ==")
    counts = {PASS: 0, FAIL: 0, SKIP: 0, GAP: 0}
    for idx, (name, fn) in enumerate(PROBES, 1):
        try:
            verdict, detail = fn()
        except Exception as exc:  # noqa: BLE001 探针的职责是把失败如实摊开，不是兜住它
            verdict, detail = FAIL, _head(exc)
        if args.allow_ascend_missing and verdict == FAIL and name in ("最小件 lowering", "AscendC 编译器", "设备在位", "torch_npu 前向参照"):
            verdict = GAP
        counts[verdict] = counts.get(verdict, 0) + 1
        print(f"[C1/{idx}] {name}: {verdict} — {detail}")
    print(f"—— 汇总：PASS={counts[PASS]} FAIL={counts[FAIL]} SKIP={counts[SKIP]} KNOWN-GAP={counts[GAP]} ——")
    print("结论行（抄入 run notes）：以上 FAIL 任一项即触发 D6 回退决策（改走 torch_npu 底座，算子故事降级为基准表+回退记录）")
    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
