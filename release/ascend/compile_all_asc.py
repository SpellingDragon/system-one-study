#!/usr/bin/env python3
"""compile_all_asc.py — 910B **单一判决入口**：九件产品接口 +（`--products`）产品件/对照件双轨。

【判决策 A2-P2 合并（承 Y 待裁决项④）】本件原是"九件接口腿"清单，产品件 `*_prod` 与
performance-track `*_intrinsic` 双轨清单在 `ascend/port910b/run_numerics.py`。两套清单=两处漂移，
现**以本件为唯一编排入口**：`run_numerics.py` 的 case 正身、判据正文与阈值**一行未改、文件不删**
（卡窗 `run_*_card.py` / `run_delta_card.py` / `fixup_card.sh` / `bundle_ondemand.sh` 仍直接依赖），
本件只做 **import + 逐 case 转调 + 统一前缀与汇总**。转调手法：`run_numerics.ARGS` 是 import 期
解析的，逐 case 换 `sys.argv` 不生效 ⇒ 直接改写其 Namespace 的 `only/track/no_run` 三量后直呼 case
函数（不改其任何判据正文）。

卡窗一行拿全部判决（含 E-3 判别所需 intrinsic 对照）：
      python3 ascend/compile_all_asc.py --target ascend --products
host 回归门（九件语义面 + 产品件 cpu-alias 数值）：
      python3 ascend/compile_all_asc.py --target cpu
      python3 ascend/compile_all_asc.py --target cpu --products --prod-cases prod

对 `ascend/kernels/*_asc.py` 九件，在 **patched 910B tilelang + 无设备** 下驱动其**对外入口件**
（`gemm_kernel` / `gemm_bwd_dw_kernel` / `rope_kernel` / `add_ln_kernel` / `attn_sw_kernel` /
`gdn_kernel`（递推 + 短卷积两枚）/ `letter_readout_kernel` / `lora_kernel`），把**编译判决**与
**发射**隔离开：真跑 `tilelang.compile` 拿成功/失败，用 dummy 顶掉 kernel 调用（卡上才需要真发射）。
判据来自 ascend_env 的 `compiled_keys()`（真编译并缓存过才算走内核，R14 防假绿）与 `blockers()`
（编译失败原文首行）。

【P1-1i 两处修法（元缺陷：判决件量的不是产品代码）】
  ① `_drive_gemm` 原先手搭 `gemm_asc.plan(A, W, bias, C, n=…, k=…)`，与现签名
     `plan(A, W, bias, C, act, target)` 不符 ⇒ `TypeError: plan() got an unexpected keyword
     argument 'n'` ⇒ gemm 腿长期 PRECOMPILE-ERR（S 报告）。现在**九条腿一律只走入口件**
     （`*_kernel.forward/backward`，与 `ascend.autograd_asc`/`train_step.py` 同一批调用行）——
     harness 通过 = 产品接口通过，harness 不再自带第二套接口口径（第二套口径本身就是错的来源）。
  ② 原先 target 硬写 "ascend"：host（无 ascend 后端）上只能看见后端错，看不见语义面。现在
     target 由 `--target` / `DMLAYA_ASCEND_TARGET` 决定，**缺省 cpu**（与 `ascend_env` 的
     "env 未设置时按 cpu，本地无卡落在安全侧"同口径），判决行与判据前缀都跟着 target 走。

一行/件：`ASC-<op> PASS .o=<bytes> keys=<n>` / `ASC-<op> FAIL <head>` /
        `ASC-<op> PRECOMPILE-ERR <head>`（= 没达到 tilelang.compile，多为形状/plan 拒/入口回退）。
`--products` 追加行（**判决原文照抄 run_numerics 的 case 返回值**，只挂上腿别标签）：
        `ASC-PROD-<case> PASS|FAIL <run_numerics 判决原文>`   产品件轨 ⇒ **计入门禁**
        `ASC-PERF-<case> PASS|FAIL|SKIP <原文>`               performance-track 对照 ⇒ 默认**不计
            门禁**（Y 口径：`NUM-*` 是 §12/intrinsic 对照，不可当正确性判决引用；要计入加
            `--perf-blocking`）。PASS/FAIL 取判决原文里第一个 `-PASS|-FAIL` 标记；**没有标记一律
            按 FAIL**，不许空转算过。
退出码：门禁腿（九件接口 + 三件产品，`--perf-blocking` 时再加对照腿）任一非 PASS→1；全 PASS→0。

products 腿的 track 随 `--target` 派生：
  · `ascend` → `card`：真设备在场⇒真数值；无设备有 bisheng（容器）⇒自动 compile-only（`no_run`），
    要强制可加 `--prod-run` / `--prod-no-run`。**ascend 编译面不在场**（host 误跑 `--target ascend
    --products`）时整条腿打 `ASC-PROD-skipped FAIL` 并计一条门禁失败——判据是**真编一枚最小件**
    （`ascend_env._compile_probe`），不是"桩说可用"：注意不能用 `backend_available` 当这个门禁，
    它先要求 `_npu_present()`，容器无卡必 False，而 run_numerics 的 card 轨靠自己在 `_prod_setup`
    里桩设备门来判编译，用它会把容器整条腿误杀。
  · `cpu` → `cpu-alias`：host 用 cpu 方言别名强跑**生产 ascend 正文**的真数值（Y 口径的"无卡最强
    证据"）；对照腿（intrinsic 正文直编 `target="ascend"`，host 无后端必挂）按设计打 `SKIP` 行，
    不计门禁也不冒充 PASS——要看它们请在容器跑 `--target ascend --products`。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import sys
import traceback

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
HERE = os.path.dirname(os.path.abspath(__file__))          # .../release/ascend
RELEASE_ROOT = os.path.dirname(HERE)                        # .../release
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)                        # 使 `ascend` 包可导入

import torch  # noqa: E402
import tilelang  # noqa: E402

from ascend.kernels import ascend_env  # noqa: E402

_AP = argparse.ArgumentParser()
_AP.add_argument("--target", default=None, help="ascend | cpu（缺省读 DMLAYA_ASCEND_TARGET，再缺省 cpu）")
_AP.add_argument("--products", action="store_true",
                 help="本件兼作唯一入口：九件之后转调 run_numerics 的产品件轨(*_prod)与"
                      "performance-track 对照轨(*_intrinsic 等)")
_AP.add_argument("--prod-cases", default="all",
                 help="products 腿子集：all=三件产品+八条对照 | prod=只三件产品 | 逗号列表（含弃用别名）")
_AP.add_argument("--prod-track", default=None, choices=(None, "card", "cpu-alias"),
                 help="products 腿的 track（缺省随 --target：ascend→card，cpu→cpu-alias）")
_AP.add_argument("--prod-run", action="store_true", help="强制真发射（卡窗有设备数值判决）")
_AP.add_argument("--prod-no-run", action="store_true", help="强制只判编译（发射桩掉）")
_AP.add_argument("--perf-blocking", action="store_true",
                 help="把 performance-track 对照腿也计入退出码（默认只打印不计门禁）")
_AP.add_argument("ops", nargs="*", help="只判这些接口件（缺省九件全判；不影响 --products 的腿）")
_ARGS = _AP.parse_args()
#: 本次判决面。`normalize_target(None)` 的取值链 = `--target` 之外唯一的入口，与产品件同源。
_TARGET = _ARGS.target or ascend_env.default_target()


class _LaunchSkip(Exception):
    """kernel 被"发射"时抛它：本 harness 只判编译，不判发射（无设备）。"""


#: 被桩掉的四量与 tilelang.compile 的原身；`_restore_stubs` 逐件还原
_SAVED: dict[str, object] = {}


def _install_stubs():
    """强制走 tilelang 后端 + 把真编译产物换成"可编译、调用即跳发射"的替身。"""
    for _n in ("backend_available", "_npu_present", "active_backend", "compile_kwargs"):
        _SAVED[_n] = getattr(ascend_env, _n)
    ascend_env.backend_available = lambda *a, **k: True      # 绕 npu 在场门
    ascend_env._npu_present = lambda: True
    # 各代理一致上报：active_backend(target, cpu张量.device) 因 want=npu≠cpu 直判 eager，
    # 只桩上两量仍走不到 compile → 一并桩掉设备-目标匹配门（判据仍锁 `<target>|` 键）。
    ascend_env.active_backend = lambda *a, **k: ascend_env.TILELANG
    real_compile = tilelang.compile
    _SAVED["tilelang.compile"] = real_compile

    def _compile(*a, **k):
        kern = real_compile(*a, **k)                         # 真编译；失败原样抛，由 get_compiled 记 blocker
        nbytes = len(str(kern.get_kernel_source())) if hasattr(kern, "get_kernel_source") else 0

        class _Dummy:
            def __init__(self): self._n = nbytes
            def __call__(self, *aa, **kk): raise _LaunchSkip()
            def __getattr__(self, n): return lambda *aa, **kk: 0
        d = _Dummy()
        _size_slot["n"] = nbytes
        return d

    tilelang.compile = _compile


def _restore_stubs():
    """还原接口腿的桩并清空 ascend_env 账本（`--products` 前必做：两套桩不能同框）。

    run_numerics 的 `_install_launch_skip` 自己 capture/替换 `tilelang.compile`；若本件的替身还挂着，
    它会 capture 到替身，且两边 `_LaunchSkip` 是两个不同类 ⇒ 产品腿会把接口腿的抛出当未知异常报 FAIL。
    撤桩 + reset 之后，产品腿看到的是干净的原始环境。
    """
    for _n, _v in _SAVED.items():
        if _n == "tilelang.compile":
            tilelang.compile = _v
        else:
            setattr(ascend_env, _n, _v)
    _SAVED.clear()
    ascend_env.reset()


_size_slot = {"n": 0}


def _f16(*shape): return torch.randn(*shape, dtype=torch.float16)
def _f32(*shape): return torch.randn(*shape, dtype=torch.float32)


# ── 九条腿：全部按**产品接口**取判决（形状取够小的代表值，位宽按入口件白名单给）─────────
# 注意 fp32：`gemm_kernel`/`gemm_bwd_dw_kernel` 的入口在 cpu target 会把输入升 fp32，而
# `rope_asc.forward` 在 ascend target 只在 **fp32** 时走内核（P1-4 路由修复的位宽门），
# 故各腿一律喂 fp32：两 target 都能真达到 tilelang.compile，不靠回退蒙过判据。
def _drive_gemm():
    """`gemm_kernel.forward(A, W, bias, act, out_dtype, target)`：C = A@Wᵀ + bias。"""
    from ascend.kernels import gemm_kernel
    A, W, bias = _f32(16, 32), _f32(32, 32), _f32(32)
    return gemm_kernel.forward(A, W, bias, "none", torch.float32, _TARGET)


def _drive_gemm_bwd_dw():
    """`gemm_bwd_dw_kernel.backward(dY, A, out_dtype=, target=)`：dW = dYᵀ @ A，出口 (N,K)。"""
    from ascend.kernels import gemm_bwd_dw_kernel
    dY, A = _f32(16, 32), _f32(16, 32)
    return gemm_bwd_dw_kernel.backward(dY, A, out_dtype=torch.float32, target=_TARGET)


def _drive_add_ln():
    """`add_ln_kernel.forward` + `.backward`（前向 add+LN 融合、反向 ln_bwd 是两份正文，都判）。"""
    from ascend.kernels import add_ln_kernel
    x, res = _f32(16, 64), _f32(16, 64)
    g, b = _f32(64), _f32(64)
    y, h = add_ln_kernel.forward(x, res, g, b, target=_TARGET)
    return add_ln_kernel.backward(h, g, y.float(), target=_TARGET)


def _drive_rope():
    """`rope_kernel.forward`：packed (tokens,3,heads,dim)，cos/sin 是 (tokens, dim//2)。"""
    from ascend.kernels import rope_kernel
    qkv = _f32(8, 3, 2, 8)                                    # (tokens,3,heads,dim)，dim 偶
    cos = _f32(8, 4)
    sin = _f32(8, 4)
    return rope_kernel.forward(qkv, cos, sin, target=_TARGET)


def _drive_attn_sw():
    """`attn_sw_kernel.forward`：因果滑窗注意力（heads, seq, dim）；dim 必须被 16 整除。"""
    from ascend.kernels import attn_sw_kernel
    q, k, v = _f32(2, 8, 16), _f32(2, 8, 16), _f32(2, 8, 16)
    return attn_sw_kernel.forward(q, k, v, 8, target=_TARGET)


def _drive_gdn():
    """`gdn_kernel.forward`：GDN 递推（g 允许 (heads,seq) 的标量门）。"""
    from ascend.kernels import gdn_kernel
    q, k = _f32(2, 8, 16), _f32(2, 8, 16)
    v = _f32(2, 8, 8)
    g = _f32(2, 8)
    beta = _f32(2, 8)
    return gdn_kernel.forward(q, k, v, g, beta, torch.float32, _TARGET)


def _drive_gdn_conv():
    """`gdn_kernel.conv_forward`：GDN 短卷积（x 是 (..., C, L) 通道优先，w 是 (C,4)）。"""
    from ascend.kernels import gdn_kernel
    x = _f32(2, 16, 32)                                        # (batch, chans, length)
    w = _f32(16, 4)
    bias = _f32(16)
    return gdn_kernel.conv_forward(x, w, bias, torch.float32, _TARGET)


def _drive_letter_readout():
    """`letter_readout_kernel.forward` + `.backward`：gather 与 scatter_add 两份正文都判。"""
    from ascend.kernels import letter_readout_kernel
    rows = _f32(16, 64)
    ids = torch.arange(4, dtype=torch.long) % 16               # 0,1,2,3
    out = letter_readout_kernel.forward(rows, ids, torch.float32, _TARGET)
    dy = _f16(int(ids.numel()), 64).float()
    return letter_readout_kernel.backward(16, ids, dy, _TARGET)


def _drive_lora():
    """`lora_kernel.apply`：delta = scaling·(x@Aᵀ)@Bᵀ，两次都该走 gemm 内核（r/k/n 均需被 16 整除）。"""
    from ascend.kernels import lora_kernel
    x = _f32(16, 32)                                           # (m, k)
    a = _f32(16, 32)                                           # (r, k)  r=16
    b = _f32(32, 16)                                           # (n, r)  n=32
    return lora_kernel.apply(x, a, b, 1.0, None, torch.float32, _TARGET)


DRIVERS = {
    "gemm": _drive_gemm,
    "gemm_bwd_dw": _drive_gemm_bwd_dw,
    "add_ln": _drive_add_ln,
    "rope": _drive_rope,
    "attn_sw": _drive_attn_sw,
    "gdn": _drive_gdn,
    "gdn_conv": _drive_gdn_conv,
    "letter_readout": _drive_letter_readout,
    "lora": _drive_lora,
}


def _target_keys() -> list[str]:
    """本 target 下真编译并缓存过的键（判据的唯一来源：回退路径不留痕迹）。"""
    pre = f"{_TARGET}|"
    return [k for k in ascend_env.compiled_keys() if k.startswith(pre)]


def _verdict_for(op: str) -> tuple[str, str]:
    """据 compiled_keys/blockers 判：真编过=PASS；有 `<target>|` blocker=FAIL(编译错)；都没=PRECOMPILE-ERR。"""
    keys = _target_keys()
    bl = {k: v for k, v in ascend_env.blockers().items() if k.startswith(f"{_TARGET}|")}
    if keys:
        return "PASS", f".o~{_size_slot['n']}B keys={len(keys)} first={keys[0]}"
    if bl:
        return "FAIL", next(iter(bl.values()))[:200]
    return "PRECOMPILE-ERR", "未达 tilelang.compile（形状/plan 拒，或入口件静默落了 eager）"


def bl_hit(_exc) -> bool:
    return any(k.startswith(f"{_TARGET}|") for k in ascend_env.blockers())


def run_one(op: str) -> tuple[str, str]:
    ascend_env.reset()
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda *a, **k: ascend_env.TILELANG
    _size_slot["n"] = 0
    try:
        DRIVERS[op]()
    except _LaunchSkip:
        pass                                                 # 发射被桩掉 = 本 harness 期望的路径
    except Exception as exc:  # noqa: BLE001
        head = (traceback.format_exception_only(type(exc), exc) or [""])[0].strip()
        v, d = _verdict_for(op)
        if v == "PASS":
            return v, d
        return ("FAIL" if v == "FAIL" else "PRECOMPILE-ERR"), (d if bl_hit(exc) else head)
    return _verdict_for(op)


# ────────────────── --products：转调 run_numerics（判决策 A2-P2 合并）──────────────────
_VERDICT_RE = re.compile(r"-(PASS|FAIL)\b")


def _npu_in_place() -> bool:
    """**真** 设备在场（用撤桩后的 `ascend_env._npu_present` 原身问，不信桩）。"""
    orig = _SAVED.get("_npu_present") or ascend_env._npu_present
    try:
        return bool(orig())
    except Exception:  # noqa: BLE001
        return False


def _ascend_compile_face() -> bool:
    """ascend **编译面**是否在场：容器（有 bisheng 无设备）True / host False。

    判据 = 真编一枚最小件（`ascend_env._compile_probe`，即 `backend_available` 第二段那个 lowering
    自证），**不是** `backend_available` 本身：它先要求 `_npu_present()`，容器无卡必 False，而
    run_numerics 的 card 轨正是靠自己在 `_prod_setup` 里桩掉设备门来只判编译的——用它当门禁会把
    容器整条 products 腿误杀。探测本身失败一律按不在场处理（宁可 SKIP 也不交假绿）。
    """
    if _npu_in_place():
        return True
    try:
        ascend_env._compile_probe(ascend_env.TARGET_ASCEND)
        return True
    except Exception:  # noqa: BLE001
        return False


def _prod_track() -> str:
    return _ARGS.prod_track or ("card" if _TARGET == "ascend" else "cpu-alias")


def _prod_no_run(has_dev: bool) -> bool:
    if _ARGS.prod_no_run:
        return True
    if _ARGS.prod_run:
        return False
    return not has_dev          # 无设备⇒只判编译；有设备⇒真数值判决


def _import_run_numerics():
    """import run_numerics（其 `ARGS` 在 import 期解析 ⇒ 先把 argv 收成空选项，再逐 case 改 Namespace）。"""
    rn_dir = os.path.join(RELEASE_ROOT, "ascend", "port910b")
    if rn_dir not in sys.path:
        sys.path.insert(0, rn_dir)
    saved = sys.argv
    sys.argv = ["run_numerics.py"]
    try:
        import run_numerics as R      # noqa: E402  判决主体在 __main__ 里，import 不跑 case
    finally:
        sys.argv = saved
    return R


def _prod_names(R) -> list[str]:
    """`--prod-cases` → case 名（all=三件产品+八条对照；prod=只三件；逗号表原样，含弃用别名）。"""
    spec = (_ARGS.prod_cases or "all").strip()
    if spec == "all":
        return list(R.PROD_TARGETS) + list(R.PERF_TARGETS)
    if spec == "prod":
        return list(R.PROD_TARGETS)
    want = [c.strip() for c in spec.split(",") if c.strip()]
    unknown = [c for c in want if c not in R.TARGETS and c not in R.DEPRECATED]
    if unknown:
        print(f"ASC-products FAIL 未知 case {unknown}；可用：{','.join(R.TARGETS)}", flush=True)
        return []
    return want


def _leg_of(R, case: str) -> str:
    """腿别：产品件轨计门禁，performance-track 对照默认不计（Y 口径：NUM-* 不可当正确性判决）。"""
    return "ASC-PERF" if case in R.PERF_TARGETS else "ASC-PROD"


def run_products() -> tuple[int, int, int, int, int]:
    """逐 case 直呼 run_numerics 的函数，输出统一 `ASC-PROD-*`/`ASC-PERF-*` 行。

    :returns: `(产品PASS, 产品FAIL, 对照PASS, 对照FAIL, 对照按设计SKIP)`；
              "整条腿没环境跑" 折进 `产品FAIL`（计门禁），设计性 SKIP 不计门禁也不算 PASS。
    """
    track = _prod_track()
    try:
        R = _import_run_numerics()
    except Exception as exc:  # noqa: BLE001  入口没聚合成 = 如实报，不静默
        print(f"ASC-products FAIL import(run_numerics) {type(exc).__name__}: {str(exc)[:200]}",
              flush=True)
        return 0, 1, 0, 0, 0
    names = _prod_names(R)
    if not names:
        return 0, 1, 0, 0, 0

    has_dev = _npu_in_place() if track == "card" else False
    no_run = _prod_no_run(has_dev)
    if track == "card" and not _ascend_compile_face():
        print("ASC-PROD-skipped FAIL ascend 编译面不在场（真编最小件失败）⇒ products 腿整条未跑；"
              "请在有 bisheng 的容器/卡上跑 `--target ascend --products`"
              "（source set_env.sh + BISHENG_HOME + ASCEND_NPU_ARCH=dav-2201）", flush=True)
        return 0, 1, 0, 0, 0

    print(f"—— products 腿 track={track} no_run={no_run} cases={len(names)}"
          f"（case 正身与阈值全在 run_numerics.py，本件不复制判据）——", flush=True)
    pp = pf = cp = cf = dsk = 0
    for name in names:
        canon = R.DEPRECATED.get(name, name)
        fn = R.TARGETS.get(canon)
        leg = _leg_of(R, canon)
        if fn is None:
            print(f"{leg}-{name} FAIL unknown case（run_numerics.TARGETS 无此项）", flush=True)
            cf += 1 if leg == "ASC-PERF" else 0
            pf += 1 if leg == "ASC-PROD" else 0
            continue
        if leg == "ASC-PERF" and track != "card":
            print(f"{leg}-{name} SKIP track={track} 不量 intrinsic 正文（其 tilelang.compile("
                  f"target='ascend') 在无 ascend 后端的 host 必挂）⇒ 见 --target ascend --products",
                  flush=True)
            dsk += 1
            continue
        R.ARGS.only, R.ARGS.track, R.ARGS.no_run = name, track, no_run
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                line = fn()
        except Exception as exc:  # noqa: BLE001  同 run_numerics.__main__：记 FAIL 继续，不中断整窗
            line = f"{type(exc).__name__}: {str(exc)[:300]}"
            verdict = "FAIL"
        else:
            m = _VERDICT_RE.search(str(line))
            verdict = m.group(1) if m else "FAIL"      # 无 PASS/FAIL 标记 ⇒ 不算过（R14 非空转）
        txt = " ".join(str(line).split())
        print(f"{leg}-{name} {verdict} {txt}", flush=True)
        for extra in buf.getvalue().splitlines():        # 自证/note 行如实带出，不吞信息
            if "-PASS" not in extra and "-FAIL" not in extra:
                print(f"    · {extra}", flush=True)
        if verdict == "PASS":
            cp += 1 if leg == "ASC-PERF" else 0
            pp += 1 if leg == "ASC-PROD" else 0
        else:
            cf += 1 if leg == "ASC-PERF" else 0
            pf += 1 if leg == "ASC-PROD" else 0
    return pp, pf, cp, cf, dsk


def main(argv=None):
    ops = _ARGS.ops or list(DRIVERS)
    unknown = [o for o in ops if o not in DRIVERS]
    if unknown:
        print(f"ASC-usage FAIL 未知件 {unknown}；可用：{','.join(DRIVERS)}")
        return 1
    _install_stubs()
    print(f"=== compile_all_asc target={_TARGET} tilelang={tilelang.__version__} ops={len(ops)}"
          f"{' products=on' if _ARGS.products else ''} ===")
    n_fail = 0
    for op in ops:
        try:
            verdict, detail = run_one(op)
        except Exception as exc:  # noqa: BLE001
            verdict, detail = "PRECOMPILE-ERR", (traceback.format_exception_only(type(exc), exc) or [""])[0][:160]
        if verdict != "PASS":
            n_fail += 1
        print(f"ASC-{op} {verdict} {detail}", flush=True)
    pp = pf = cp = cf = dsk = 0
    if _ARGS.products:
        _restore_stubs()                 # 接口腿的桩先撤，再让 run_numerics 自己装配运行面
        pp, pf, cp, cf, dsk = run_products()
        print(f"—— 产品腿 PASS={pp} FAIL={pf} | 对照腿(performance-track) PASS={cp} FAIL={cf} "
              f"SKIP={dsk} [{'blocking' if _ARGS.perf_blocking else 'non-blocking'}] ——", flush=True)
    gating = n_fail + (pf if _ARGS.products else 0) \
        + (cf if (_ARGS.products and _ARGS.perf_blocking) else 0)
    print(f"—— 汇总 target={_TARGET} 接口腿 PASS={len(ops)-n_fail} FAIL={n_fail}"
          + (f" 产品腿 PASS={pp} FAIL={pf} 对照腿 PASS={cp} FAIL={cf} SKIP={dsk}"
             if _ARGS.products else "")
          + f" —— 门禁失败数={gating}")
    return 1 if gating else 0


if __name__ == "__main__":
    sys.exit(main())
