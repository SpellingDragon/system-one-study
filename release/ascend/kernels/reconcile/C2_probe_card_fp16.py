#!/usr/bin/env python3
"""reconcile/C2_probe_card_fp16.py — P1-1m 的 **910B target=ascend 编译判决**（fp16 三件＋bf16 一手取证）。

【判什么的】host cpu-alias 轨（C2_probe_dtype_matrix.py）判的是"数值对不对、路由真不真"；本脚本判
的是"真 ascend 方言下，位宽参数化后的正文编不编得出产物"，两轨合起来才是闭环：
  A 组（接口路径，编译判决）：gemm/dw/rope 三件各跑 fp32 与 fp16（rope 再加一档"qkv fp16 + 角度表
     fp32"的生产混宽），判据＝`compiled_keys()` 出 `...dfloat16` / `qfloat16afloat32` 副键、
     `blockers()` 无 `ascend|` 项、桩真见过 ≥1 次成功编译且拿到非空生成码。
  B 组（bf16 一手取证）：**绕过 plan 白名单**直接按 `idt/qdt="bfloat16"` 组正文并真编译，把 bisheng
     的原文报错抓下来（这就是三件 `_ASC_DTYPES` 不收 bf16 的依据；白名单在 plan 期显式拒，见
     C2_probe_dtype_matrix.py --neg）。B 组**不是**支持性判决，是缺件取证。

【桩与隔离】配方同 reconcile/verify_dw.py：绕设备门（backend_available/_npu_present/active_backend
三处都得绕，最后这个不绕则 cpu 张量＋target=ascend 被判 torch_eager、根本走不到编译）＋把
`tilelang.compile` 换成"真编译后发射 no-op"的替身（无设备 ⇒ 本轨不判数值，GW/出口空表绝不当证据读）。
B 组不用桩：直接 `tilelang.compile(prog, target="ascend")`，异常原样抓。

用法（容器内，release 挂 /work；跑前先同步 trunk 的 port910b_compat.h）：
  source /usr/local/Ascend/cann-8.5.0/set_env.sh
  export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201
  cd /work && TILELANG_CACHE_DIR=$(mktemp -d) python3 ascend/kernels/reconcile/C2_probe_card_fp16.py
判决行：`C2-CARD-<件>-<位宽>-COMPILE-PASS keys=[...]` / `C2-CARD-<件>-bf16-UNSUPPORTED err=<原文>`，
末行 `C2-CARD-SUMMARY A=<n>/6 B=<n>/3`。注意：**不能**加 `from __future__ import annotations`。
"""
import os
import sys
import traceback

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

import torch           # noqa: E402
import tilelang        # noqa: E402
from ascend.kernels import (ascend_env, gemm_asc, gemm_bwd_dw_asc,   # noqa: E402
                            gemm_bwd_dw_kernel, gemm_kernel, rope_asc, rope_kernel)

SRC = {"acc": "", "n": 0, "per": {}}
CUR = {"tag": "?"}                                        # 本次编译归属（判决/位宽字样按它分桶）
A_PASS, A_FAIL, B_DONE, B_FAIL = [], [], [], []


def _stub():
    """真编译 + 发射 no-op 替身（形态同 verify_dw.py，编译失败仍原样抛给 get_compiled 记 blocker）。"""
    real = tilelang.compile

    def _fake(*a, **k):
        kern = real(*a, **k)
        src = str(kern.get_kernel_source()) if hasattr(kern, "get_kernel_source") else ""
        SRC["acc"] += src
        SRC["per"].setdefault(CUR["tag"], "")
        SRC["per"][CUR["tag"]] += src
        SRC["n"] += 1

        class _Dummy:
            def __call__(self, *aa, **kk):
                return None

        return _Dummy()

    tilelang.compile = _fake
    return lambda: setattr(tilelang, "compile", real)


def _gate():
    ascend_env.reset()
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda target=None, device=None: ascend_env.TILELANG


def _verdict(piece, tag, want):
    keys = [x for x in ascend_env.compiled_keys() if x.startswith("ascend|") and piece in x]
    bl = {x: v for x, v in ascend_env.blockers().items() if x.startswith("ascend|")}
    if keys and not bl and any(want in k for k in keys):
        print(f"C2-CARD-{piece}-{tag}-COMPILE-PASS keys={keys} compiled={SRC['n']} "
              f"src={len(SRC['acc'])}B blockers=无", flush=True)
        A_PASS.append(f"{piece}/{tag}")
    else:
        print(f"C2-CARD-{piece}-{tag}-COMPILE-FAIL keys={keys} 期望位宽段={want} compiled={SRC['n']} "
              f"blockers={ {k: str(v)[:220] for k, v in bl.items()} }", flush=True)
        A_FAIL.append(f"{piece}/{tag}")
    return keys


def group_a():
    """接口路径的 fp32/fp16 编译判决（三件六位）。"""
    restore = _stub()
    try:
        _gate()
        SRC["acc"], SRC["n"] = "", 0
        torch.manual_seed(0)
        m, n, k = 32, 64, 64
        for dtn, dt in (("fp32", torch.float32), ("fp16", torch.float16)):
            A, W = torch.randn(m, k, dtype=dt), torch.randn(n, k, dtype=dt)
            b = torch.randn(n, dtype=torch.float32)
            CUR["tag"] = f"gemm/{dtn}"
            gemm_kernel.forward(A, W, b, "none", dt, "ascend")
            _verdict("gemm[ascend]", dtn, "dfloat16" if dtn == "fp16" else "dfloat32")
        _gate()
        for dtn, dt in (("fp32", torch.float32), ("fp16", torch.float16)):
            dY, A = torch.randn(m, n, dtype=dt), torch.randn(m, k, dtype=dt)
            CUR["tag"] = f"gemm_dw/{dtn}"
            gemm_bwd_dw_kernel.backward(dY, A, out_dtype=dt, target="ascend")
            _verdict("gemm_dw[ascend]", dtn, "dfloat16" if dtn == "fp16" else "dfloat32")
        _gate()
        tokens, heads, dim = 32, 2, 16
        for tag, qdt, adt in (("fp32", torch.float32, torch.float32),
                              ("fp16+fp16表", torch.float16, torch.float16),
                              ("fp16+fp32表(生产混宽)", torch.float16, torch.float32)):
            qkv = torch.randn(tokens, 3, heads, dim, dtype=qdt)
            cos, sin = (torch.randn(tokens, dim // 2, dtype=adt) for _ in range(2))
            CUR["tag"] = f"rope/{tag}"
            rope_kernel.forward(qkv, cos, sin, rope_kernel.DEFAULT_SLOTS, "ascend")
            _verdict("rope[ascend]", tag, "qfloat16" if qdt == torch.float16 else "qfloat32")
    finally:
        restore()


def _diag(exc):
    """从异常里挑出判决用的原文行（bisheng 的 fatal/error 行优先，否则末行）。"""
    txt = "".join(traceback.format_exception_only(type(exc), exc))
    lines = [x.strip() for x in (txt + str(exc)).splitlines() if x.strip()]
    hit = [x for x in lines if "not support" in x or "fatal error" in x or "error:" in x]
    return (hit or lines)[:3]


def group_b():
    """bf16 一手取证：绕开 plan 白名单，直接把正文按 bfloat16 组出来真编译。"""
    m, n, k, tokens, heads, dim = 32, 64, 64, 32, 2, 16
    cases = [
        ("gemm", lambda: gemm_asc.gemm_asc_impl(
            None, None, None, None, n=n, k=k, bm=gemm_asc.BLOCK_M, bn=64, bk=32, act_mode=0,
            idt="bfloat16", odt="bfloat16")),
        ("gemm_dw", lambda: gemm_bwd_dw_asc.dw_asc_impl(
            None, None, None, n=n, k=k, tc=gemm_bwd_dw_asc.BLOCK_TC, bn=64, bk=32,
            idt="bfloat16")),
        ("rope", lambda: rope_asc.rope_asc_impl(
            None, None, None, heads=heads, dim=dim, slots=1, sign=1,
            qdt="bfloat16", adt="bfloat16")),
    ]
    for name, build in cases:
        try:
            kern = tilelang.compile(build(), target="ascend")
            src = str(kern.get_kernel_source()) if hasattr(kern, "get_kernel_source") else ""
            print(f"C2-CARD-{name}-bf16-COMPILED(!) src={len(src)}B —— 与预期不符，须复核白名单",
                  flush=True)
            B_FAIL.append(name)
        except Exception as exc:  # noqa: BLE001  取证本就是要抓原文
            d = _diag(exc)
            print(f"C2-CARD-{name}-bf16-UNSUPPORTED err={d}", flush=True)
            if not any("bf16" in x.lower() or "bfloat16" in x.lower() for x in d):
                print(f"C2-CARD-{name}-bf16-RAW {str(exc)[:400]!r}", flush=True)
            B_DONE.append(name)


def dtype_tokens():
    """每份生成码的位宽字样（旁证：fp16 轨真出 half 面，而非 fp32 模具被复用）。

    口径：不猜 token——把 `__global__` 那行**原样**交出来（GM 指针位宽就在里面），并数 fp32 字样
    `float ` 的出现次数（累加面/UB 标量应恒 fp32，即“主干 fp16、归约 fp32”在生成码上的投影）。
    """
    for tag in sorted(SRC["per"]):
        c = SRC["per"][tag]
        sig = [x.strip() for x in c.splitlines() if "__global__" in x][:1]
        print(f"C2-CARD-SRC {tag} size={len(c)}B float_word={c.count('float ')} "
              f"half_word={c.count('half')} SimtVF={'SimtVF' in c} "
              f"Cube/mad={'Cube' in c or ' mad' in c} 签名={sig}", flush=True)


def main():
    group_a()
    dtype_tokens()
    group_b()
    print(f"C2-CARD-SUMMARY A={len(A_PASS)}/{len(A_PASS) + len(A_FAIL)} "
          f"B(取证到手)={len(B_DONE)}/3 "
          f"A_FAIL={A_FAIL or '无'} B_UNEXPECTED={B_FAIL or '无'}", flush=True)
    return 1 if (A_FAIL or B_FAIL) else 0


if __name__ == "__main__":
    sys.exit(main())
