#!/usr/bin/env python3
"""reconcile/F2_probe_card_fp16_step.py — p2-13 F2：**真 ascend 方言**下的全图 fp16 编译腿。

【判什么的】host 两轨（--dtype fp16 直跑 / cpu-alias 真编真跑）判的是"数值与路由"；本腿判
"target=ascend 路由的全图训练步，各 op 在 bisheng 真编下出不出得了产物、op 键含不含位宽段"。
桩口径与 `C2_probe_card_fp16.py` 完全一致（同配方两代）：
  · 绕设备门三处（backend_available/_npu_present/active_backend——最后这个不绕则 cpu 张量
    被判 torch_eager、根本走不到编译）；
  · `tilelang.compile` 换"真编译后发射 no-op"替身：**无设备 ⇒ 本腿不判数值**，桩下空表
    （未初始化输出的 loss/grad）绝不当证据读。训练步照常走完（两 fwd 一 bwd 一 step），
    判的只有编译与键账。
【位宽段判据】三原生件必须出 fp16 段（`gemm…dfloat16`、`gemm_dw…dfloat16`、
`rope…qfloat16afloat32`＝生产混宽）；六物化件（add_ln/ln_bwd/attn_sw/gdn/gdnb/gather/
scatter_add）键无位宽段是 C2 §5 的**既定形态**（入口 fp16/内核 fp32），分型如实打印。
每 tag 另交 `__global__` 签名原文与 `float `/`half` 字样计数（生成码位宽旁证，同 C2 口径）。

用法（容器内，release 挂 /work；跑前同步 trunk compat `ca023c2e`）：
  docker exec cann910b-h bash -lc 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \
    export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201; \
    cd /work && TILELANG_CACHE_DIR=/dev/shm/f2cache python3 ascend/kernels/reconcile/F2_probe_card_fp16_step.py'
判决行：F2-CARD-<用例>-PASS/FAIL；末行 F2-CARD-SUMMARY n/m。
注意：**不能**加 `from __future__ import annotations`。
"""
import os
import sys
import traceback

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if RELEASE not in sys.path:
    sys.path.insert(0, RELEASE)

import torch           # noqa: E402
import tilelang        # noqa: E402
from ascend.kernels import ascend_env  # noqa: E402

SRC = {"acc": "", "n": 0, "per": {}, "fail": []}
CUR = {"tag": "?"}
RESULTS = []


def _stub():
    """真编译 + 发射 no-op 替身（编译失败原样抛给 get_compiled 记 blocker，不吞）。"""
    real = tilelang.compile

    def _fake(*a, **k):
        try:
            kern = real(*a, **k)
        except Exception as exc:  # noqa: BLE001
            SRC["fail"].append(f"{CUR['tag']}: {type(exc).__name__} "
                               f"{str(exc).splitlines()[0][:200]}")
            raise
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


def verdict(name, ok, detail=""):
    print(f"F2-CARD-{name}-{'PASS' if ok else 'FAIL'} {detail}", flush=True)
    RESULTS.append(bool(ok))


def main():
    restore = _stub()
    step_err = None
    try:
        _gate()
        from ascend import train_step as ts
        CUR["tag"] = "trainstep/fp16"
        try:
            ts.run(target="ascend", device=torch.device("cpu"),
                   dtype=torch.float16, verbose=True)
        except Exception as exc:  # noqa: BLE001  桩下数值不可信：捕获只为继续判键
            step_err = f"{type(exc).__name__}: {str(exc)[:160]}"
    finally:
        restore()

    keys = sorted(ascend_env.compiled_keys())
    bl = {k: str(v)[:200] for k, v in ascend_env.blockers().items()}

    # 1) 编译腿本体：全图 ≥13 件真编、生成码非空
    verdict("COMPILE", SRC["n"] >= 13 and len(SRC["acc"]) > 1000 and not SRC["fail"],
            f"compiled={SRC['n']} src={len(SRC['acc'])}B 编译失败={SRC['fail'] or '无'}")

    # 2) 件齐（真 ascend 方言，无回落）
    want = ["gemm[ascend]", "gemm_dw[ascend]", "rope[ascend]", "add_ln[ascend]",
            "ln_bwd[ascend]", "attn_sw[ascend]", "gdn[ascend]", "gdnb[ascend]",
            "gather[ascend]", "scatter_add[ascend]"]
    absent = [p for p in want if not [k for k in keys if p in k]]
    verdict("KEYS-ROUTE", not absent and not bl,
            f"出件={len(want) - len(absent)}/{len(want)} 缺={absent or '无'} "
            f"blockers={bl or '无'}")

    # 3) 位宽段：原生三件 fp16 段（rope 生产混宽 qfloat16afloat32）
    g16 = [k for k in keys if "gemm[ascend]" in k and "dfloat16" in k]
    w16 = [k for k in keys if "gemm_dw[ascend]" in k and "dfloat16" in k]
    r16 = [k for k in keys if "rope[ascend]" in k and "qfloat16afloat32" in k]
    verdict("KEYS-DTYPE", bool(g16) and bool(w16) and bool(r16),
            f"gemm×{len(g16)} dw×{len(w16)} rope×{len(r16)} :: {g16[:1] + w16[:1] + r16[:1]}")

    # 4) 物化件键分型如实打印（无位宽段＝C2 §5 既定形态，不作 FAIL）
    for p in want:
        print(f"F2-CARD-KEYTYPE {p} keys={[k for k in keys if p in k]}", flush=True)

    # 5) 生成码位宽旁证（__global__ 签名原样 + fp32 字样计数，同 C2 口径）
    for tag in sorted(SRC["per"]):
        c = SRC["per"][tag]
        sig = [x.strip() for x in c.splitlines() if "__global__" in x][:1]
        print(f"F2-CARD-SRC {tag} size={len(c)}B float_word={c.count('float ')} "
              f"half_word={c.count('half')} 签名={sig}", flush=True)

    # 6) 位宽账（组装面 guard）：桩下数值不可信（step_err 如实打印），但 Python 层归约
    #    纪律与桩无关——违规=0 才算 PASS。
    from ascend import train_step as ts
    audit = dict(ts.LAST_AUDIT)
    line = ts.dtype_trace_line(audit, fallback_events=0)
    verdict("AUDIT", audit.get("fp16") == 16 and not audit.get("violations"),
            f"{line} step_err={step_err or '无'}（桩轨不判数值，见 docstring）")

    n_ok = sum(RESULTS)
    print(f"F2-CARD-SUMMARY {n_ok}/{len(RESULTS)}", flush=True)
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(99)
