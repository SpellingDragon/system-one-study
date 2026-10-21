#!/usr/bin/env python3
"""E2_probe_mold.py — 模具位宽自证探针（E2 专用，容器/卡上都可跑，**不计时**）。

【为什么要它】`bench_card.py` 的报价表里 `linear`/`dW` 各有 `bf16>fp32` 与 `fp16>fp16` 两档。
入口件的编译缓存键（`gemm_asc.plan` 的 `key`）只到 (n, k, 块宽, act)，**不含位宽** ⇒ 第二档位宽
会复用第一档的编译产物。我在 BENCH 行尾用 `same-mold-as=<首个位宽>` 把这件事标了出来，但"复用
的到底是同一份源码，还是只是同名不同物"必须实证——否则 fp16 那一行的吞吐到底是谁的速度说不清。

【它量什么】对同一形状，逐档位宽**各自冷开一次进程内缓存**（`ascend_env.reset()`），把
`tilelang.compile` 真产物的源码取哈希/字节数/位宽 token 计数并排打出来：
    MOLD op=linear device=cpu dtype=bf16 key=ascend|gemm[ascend]|... src_bytes=3118 sha=ab12cd34
         bf16_tok=0 half_tok=0 fp32_tok=7
两档 sha 相同 ⇒ 模具位宽无关（报价表的 dtype 列只能读作"接口位宽"，模具位宽是另一回事）；
不同 ⇒ 模具确实按位宽分化，只是缓存键没带上它（此时**必须**分两趟进程跑，行尾标注也照样打）。

用法（容器）：
    docker exec cann910b-j bash -lc 'source /usr/local/Ascend/cann-8.5.0/set_env.sh;
      export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201;
      cd /work && TILELANG_CACHE_DIR=$(mktemp -d) \\
        python3 ascend/kernels/reconcile/E2_probe_mold.py --target ascend'
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))    # .../release

import torch  # noqa: E402
import tilelang  # noqa: E402

from ascend import bench_card  # noqa: E402  复用同一套桩（不另造判据）
from ascend.kernels import (  # noqa: E402
    add_ln_kernel,
    ascend_env,
    gemm_bwd_dw_kernel,
    gemm_kernel,
)

_DTYPES = ("fp32", "bf16", "fp16")
_DT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def _spy():
    """包一层 `tilelang.compile`：真编译 + 抓源码指纹 + 把发射桩掉（无设备也能跑完编译）。"""
    real = tilelang.compile
    rec: list[tuple] = []

    class _NoLaunch:
        def __init__(self, kern):
            self._k = kern

        def __call__(self, *a, **k):
            raise RuntimeError("probe: 只编译，不发射")

        def __getattr__(self, n):
            return getattr(self._k, n)

    def wrap(*a, **k):
        kern = real(*a, **k)
        src = str(kern.get_kernel_source())
        rec.append((len(src), hashlib.sha1(src.encode("utf-8", "replace")).hexdigest()[:12],
                    src.count("bfloat16"), src.count("half"), src.count("float")))
        return _NoLaunch(kern)

    tilelang.compile = wrap

    def restore():
        tilelang.compile = real

    return restore, rec


def _one(op, dtype, target, m, n, k):
    """一趟：清缓存 → 桩开设备门 → 调一次入口件（编译必发生，发射必被桩掉）→ 读指纹与键。"""
    ascend_env.reset()
    restore_compile, rec = _spy()
    restore_gate = bench_card._stub_device_gate()
    key = "n/a"
    try:
        tdt = _DT[dtype]
        if op == "linear":
            A = torch.randn(m, k, dtype=tdt)
            W = torch.randn(n, k, dtype=tdt)
            b = torch.randn(n, dtype=torch.float32)
            fn = lambda: gemm_kernel.forward(A, W, b, "none", torch.float32, target)  # noqa: E731
            sub = "gemm["
        elif op == "dW":
            dY = torch.randn(m, n, dtype=tdt)
            X = torch.randn(m, k, dtype=tdt)
            fn = lambda: gemm_bwd_dw_kernel.backward(dY, X, out_dtype=torch.float32,
                                                     target=target)  # noqa: E731
            sub = "gemm_dw["
        else:
            x = torch.randn(m, n, dtype=torch.float32)
            r = torch.randn(m, n, dtype=torch.float32)
            g = torch.ones(n, dtype=torch.float32)
            bs = torch.zeros(n, dtype=torch.float32)
            fn = lambda: add_ln_kernel.forward(x, r, g, bs, 1e-5, torch.float32, target)  # noqa: E731
            sub, dtype = "add_ln[", "fp32"
        try:
            fn()
        except Exception as exc:  # noqa: BLE001  发射被桩掉属预期；编译失败也在这里显形
            if "只编译" not in str(exc):
                return f"MOLD-{op} dtype={dtype} FAIL {_first(exc)}"
        keys = [x for x in ascend_env.compiled_keys() if x.startswith(f"{target}|") and sub in x]
        key = keys[-1] if keys else "NOT-COMPILED"
        if not rec:
            return f"MOLD-{op} dtype={dtype} key={key} note=没抓到编译产物（走了回退或缓存命中）"
        size, sha, nb, nh, nf = rec[-1]
        return (f"MOLD-{op} dtype={dtype} target={target} key={key} src_bytes={size} sha={sha} "
                f"tok_bf16={nb} tok_fp16={nh} tok_fp32={nf} compiles={len(rec)}")
    finally:
        restore_gate()
        restore_compile()


def _first(exc):
    return f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"


def main():
    ap = argparse.ArgumentParser(description="模具位宽自证（只编译，不计时）")
    ap.add_argument("--target", default="ascend", choices=("ascend", "cpu"))
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--k", type=int, default=512)
    args = ap.parse_args()

    print(f"MOLD-ENV target={args.target} shape={args.m}x{args.n}x{args.k} torch={torch.__version__}")
    for op in ("linear", "dW", "add_ln"):
        dts = _DTYPES if op in ("linear", "dW") else ("fp32",)
        for dt in dts:
            print(_one(op, dt, args.target, args.m, args.n, args.k), flush=True)
    print("MOLD-NOTE 同一 (op,shape) 的两档位宽若 sha 相同 ⇒ 编译产物与位宽无关（入口件的缓存键"
          "不含位宽，bench_card 用 same-mold-as= 标注）；若不同 ⇒ 模具按位宽分化，报价表的这两档"
          "必须**分两趟进程**各取各的模具。本探针不计时，不产性能结论。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
