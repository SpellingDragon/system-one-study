#!/usr/bin/env python3
"""reconcile/M2_probe_card_fp16_dma.py — 容器（真 ascend 后端，compile-only）里**直接读生成码**的 DMA 档判决。

【为什么要它】host 的 cpu-alias 轨结构上没有 MTE2 跨步装填面（`T.copy` 走 c 后端 lowering），所以
"改 bk 之后到底编成哪一档"这句话**只能在真 ascend 后端的生成码里判**。本探针在容器里真跑
`tilelang.compile`（`target=ascend`/dav-2201/bisheng），把 `kern.get_kernel_source()` 里每条
`asc_copy_gm2ub_align(...)` 的实参按 `M2_probe_dma_cfg.parse_copy/classify` 分档 ⇒ 得到与归档
真机件同口径的档位判决（**编译判决＋生成码直读，不发射、不需要设备**）。

【判据】对每个 (m,n,k,位宽)：
  ① 出键（`ascend|gemm[ascend]|…`）且无 blocker ⇒ COMPILE-PASS；
  ② 生成码里所有 gm2ub 调用都是 `SINGLE`（`blockCount==1` 或 `src_stride==burst`）⇒ 好档；
     出现 `STRIDED-SMALL`（多突发且突发 <128B）⇒ **FAIL**（P1-1p/P1-1q 的实测坏档形态）；
  ③ 顺手把突发字节是否 32 的倍数（`TL910B_BYTES_TO_BLK` 整除截断风险）与签名位宽打出来。
工装配方（`_stub`/`_gate`）逐字同 `C2_probe_card_fp16.py`（该件已在本容器闭环 7/7）。

用法（容器内，跑前先 docker cp trunk 并核 md5=ca023c2e）：
  docker exec cann910b-i bash -lc 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \\
    export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201; \\
    cd /work && TILELANG_CACHE_DIR=$(mktemp -d) python3 ascend/kernels/reconcile/M2_probe_card_fp16_dma.py'
host 侧期望 SKIP（没有 ascend 后端），本件不为 host 设计。
判决行：`M2-CARD <case> COMPILE-PASS|FAIL ... 档=<..> copies=<..>`
末行：  `M2-CARD-DMA-SUMMARY pass=<n>/<N> strided_small=<cases> fail=<cases>`
"""
from __future__ import annotations

import os
import sys

import torch
import tilelang

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if RELEASE not in sys.path:
    sys.path.insert(0, RELEASE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ascend.kernels import ascend_env, gemm_kernel          # noqa: E402
from M2_probe_dma_cfg import classify, parse_copy           # noqa: E402

SRC = {"per": {}, "n": 0}
CUR = {"tag": "?"}
CASES = [(16, 32, 96, "fp16"), (16, 32, 96, "fp32"),      # P1-1p 的 dA 形状（step0 反向唯一跨步件）
         (16, 96, 48, "fp32"), (16, 96, 48, "fp16"),      # P1-1q 的 m16n96k48 非对齐 K 档
         (16, 32, 48, "fp32"), (16, 32, 48, "fp16"),
         (16, 96, 32, "fp16"), (16, 32, 32, "fp16")]      # 对照：ladder 本来就等于 k 的两档

_OK, _FAIL, _SMALL = [], [], []


def _stub():
    """真编译 + 发射 no-op 替身（同 C2_probe_card_fp16._stub），将生成码按 tag 收下来。"""
    real = tilelang.compile

    def _fake(*a, **k):
        kern = real(*a, **k)
        src = str(kern.get_kernel_source()) if hasattr(kern, "get_kernel_source") else ""
        SRC["per"][CUR["tag"]] = SRC["per"].get(CUR["tag"], "") + src
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


def readout(src):
    """逐条 gm2ub/ub2gm 调用分档；返回 (档列表, 是否见多突发跨步小档)。"""
    rows, small = [], False
    for i, ln in enumerate(src.splitlines(), 1):
        if "asc_copy_gm2ub_align" not in ln and "asc_copy_ub2gm_align" not in ln:
            continue
        parsed = parse_copy(ln)
        if parsed is None:
            rows.append(("PARSE-MISS", i, ln.strip()[:70]))
            continue
        d, nb, burst, ss, ds = parsed
        cls, why = classify(nb, burst, ss, ds)
        nums = [v for v in (burst, ss, ds) if isinstance(v, int) and v]
        mult32 = "BAD" if any(v % 32 for v in nums) else ("ok" if nums else "n/a")
        if cls == "STRIDED-SMALL":
            small = True
        rows.append((f"{d}:{cls}", i, f"{why} mult32={mult32}"))
    return rows, small


def main() -> int:
    if not hasattr(torch, "npu") and "dav" not in os.environ.get("ASCEND_NPU_ARCH", ""):
        print("M2-CARD-SKIP 本机无 ascend 编译面（ASCEND_NPU_ARCH 未设）——本件只在容器里判", flush=True)
        return 3
    restore = _stub()
    try:
        _gate()
        torch.manual_seed(0)
        for m, n, k, dtn in CASES:
            dt = torch.float16 if dtn == "fp16" else torch.float32
            A = torch.randn(m, k, dtype=dt)
            W = torch.randn(n, k, dtype=dt)
            tag = f"{m}x{n}x{k}/{dtn}"
            CUR["tag"] = tag
            before = SRC["n"]
            gemm_kernel.forward(A, W, None, "none", dt, "ascend")
            want = "dfloat16" if dtn == "fp16" else "dfloat32"
            keys = [x for x in ascend_env.compiled_keys() if f"n{n}k{k}b" in x and want in x]
            bl = {x: v for x, v in ascend_env.blockers().items() if x.startswith("ascend|")}
            src = SRC["per"].get(tag, "")
            rows, small = readout(src)
            sig = [x.strip() for x in src.splitlines() if "__global__" in x][:1]
            compiled = SRC["n"] > before
            if not compiled:
                print(f"M2-CARD {tag} COMPILE-FAIL 没有新产物（keys={keys} blockers="
                      f"{ {k2: str(v)[:150] for k2, v in bl.items()} }）", flush=True)
                _FAIL.append(tag)
                continue
            if bl or not keys:
                print(f"M2-CARD {tag} COMPILE-FAIL keys={keys} blockers="
                      f"{ {k2: str(v)[:200] for k2, v in bl.items()} }", flush=True)
                _FAIL.append(tag)
                continue
            if small:
                print(f"M2-CARD {tag} DMA-FAIL（生成码里仍有 STRIDED-SMALL）", flush=True)
                _SMALL.append(tag)
            else:
                _OK.append(tag)
            for cls, line, why in rows:
                print(f"M2-CARD {tag} line={line} {cls} {why}", flush=True)
            print(f"M2-CARD {tag} COMPILE-PASS keys={keys} copies={len(rows)} src={len(src)}B "
                  f"float_word={src.count('float ')} half_word={src.count('half')} "
                  f"{'无DMA(纯标量)' if not rows else ''} 签名={sig}", flush=True)
    finally:
        restore()
    print(f"M2-CARD-DMA-SUMMARY pass={len(_OK)}/{len(CASES)} strided_small={_SMALL or 'NONE'} "
          f"fail={_FAIL or 'NONE'} compiled={SRC['n']}", flush=True)
    return 0 if len(_OK) == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
