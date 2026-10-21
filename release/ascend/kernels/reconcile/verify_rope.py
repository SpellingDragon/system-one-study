#!/usr/bin/env python3
"""reconcile/verify_rope.py — 接口 home `rope_asc` 的 910B target=ascend 编译判决（代理 I）。

【做什么】驱动**真接口路径**判两件事：① `rope_asc` 的昇腾正文在 patched 910B tilelang 下能不能
出 .o；②（`--golden`）那段正文的**循环结构**在数值上是否还是同一个口径。链路是
rope_asc._plan(target="ascend") → rope_asc._run → ascend_env.get_compiled →
tilelang.compile(..., target="ascend")，本件不复制算式进 DSL、不改动被测件。
【怎么做】① 编译判决与发射隔离：真跑 tilelang.compile，成功后换成"调用即跳"的 dummy 顶掉
kernel 发射（本机无 NPU，本波只判编译，数值 rel 待卡窗）；② 判据取 ascend_env 的
`compiled_keys()`（真编过并缓存过才登记，R14 防假绿），失败侧取 `blockers()` 的原文首行；
③ 顺带对 codegen 产物做体检：SIMT 线程载体/threadIdx 应为 False（证明昇腾路径已从 950 载体
合流到标量面）、`gm_bypass/ubuf` 记实测形态；④ 前向（sign=1）与反向（sign=-1）各判一次——
它们是接口 `forward/backward` 的两个真实入口，共用同一条 _plan/_run 路；⑤ `--golden` 打的是
**numpy 影子件**：按昇腾正文同一套"每核 bx 起、步长 NUM_BLOCKS、行程数 ceildiv(tokens-bx,
NUM_BLOCKS)"的领行方式与"先物化 x1/x2 再双写"的顺序重算一遍，尺子取 P1 冻结件
`sys1.testing.torch_ref.rope_ref`（不用本模块自带的 `_eager`，免得自己给自己打分）。影子件校的是
**跨步领行是否恰好覆盖 [0,tokens) 且不重不漏**、就地双写是否互不污染、槽位 2 是否逐位不动、
反向是否等于角取负——它不证明 DSL 与 compat 件的数值（那要上卡），如实划界。
【为什么】判决口径与 ascend/compile_all_asc.py 完全一致（同一套 stub、同一个 compiled_keys
判据），这样"per-op 精修"与"九件总判决"两套工装不会各说各话；plan 拒与编译失败分开打印，
避免把"形状没过判据"误报成"方言编不出"。

一行判决：`I-ROPE-ASC-COMPILE-PASS sign=+1 key=... .o_src~<NB>`（前向/反向各一行）
影子件：`I-ROPE-ASC-GOLDEN-PASS sign=+1 max_abs=...`
失败：`I-ROPE-ASC-COMPILE-FAIL <blocker 首行>`；plan 拒：`I-ROPE-ASC-PLAN-REJECT`。

用法（容器 cann910b-i，release 挂到 /work）：
  docker exec cann910b-i bash -c 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \\
    export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201; \\
    cd /work && TILELANG_CACHE_DIR=$(mktemp -d) timeout 300 \\
      python3 ascend/kernels/reconcile/verify_rope.py --golden'
可选：--tokens 32 --heads 8 --dim 64 --sign both|1|-1 --dtype float32 --dump <path> --golden
缓存纪律：每次判决必须换新 `TILELANG_CACHE_DIR=$(mktemp -d)`（tilelang 按 kernel 源码哈希
命中缓存、不含模板内容，不换就是陈旧 PASS——LOCAL_RUN.md/B/C/E 实证）。
注意：**不能** 在本文件加 `from __future__ import annotations`（tilelang eager builder 的
get_type_hints 会把闭包变量当模块全局名 → NameError，C 波实测）。
"""
import argparse
import os
import re
import sys
import traceback

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")

HERE = os.path.dirname(os.path.abspath(__file__))              # .../ascend/kernels/reconcile
RELEASE_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))   # .../release
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)                            # 使 `ascend` 包可导入

import numpy as np   # noqa: E402
import torch         # noqa: E402
import tilelang      # noqa: E402

from ascend.kernels import ascend_env, rope_asc   # noqa: E402

PASSED = "I-ROPE-ASC-COMPILE-PASS"
FAILED = "I-ROPE-ASC-COMPILE-FAIL"
REJECT = "I-ROPE-ASC-PLAN-REJECT"
GOLDEN = "I-ROPE-ASC-GOLDEN-PASS"
GOLDEN_FAIL = "I-ROPE-ASC-GOLDEN-FAIL"

#: 领行/形状串：默认档 + 尾行档（tokens 不被 NUM_BLOCKS 整除）+ slots=1 档
SHAPES = [(32, 8, 64, 2), (33, 8, 64, 2), (8, 4, 32, 2), (5, 2, 16, 1)]


class _LaunchSkip(Exception):
    """kernel 被"发射"时抛它：本件只判编译，不判发射（无设备）。"""


_slots = {"n": 0, "src": ""}


def _install_stubs():
    """绕设备门（无 npu 也要走到真编译）+ 把真编译产物换成"可编译、调用即跳发射"的替身。"""
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda target=None, device=None: (
        ascend_env.TILELANG if ascend_env.normalize_target(target) == ascend_env.TARGET_ASCEND
        else ascend_env.TORCH_EAGER)
    real_compile = tilelang.compile

    def _compile(*a, **k):
        kern = real_compile(*a, **k)            # 真编译；失败原样抛，由 get_compiled 记 blocker
        src = str(kern.get_kernel_source()) if hasattr(kern, "get_kernel_source") else ""
        _slots["n"], _slots["src"] = len(src.encode()), src

        class _Dummy:
            def __call__(self, *aa, **kk):
                raise _LaunchSkip()

            def __getattr__(self, name):
                return lambda *aa, **kk: 0

        return _Dummy()

    tilelang.compile = _compile


def scan_source(src: str) -> dict:
    """codegen 产物体检（与 attempts/E 同格式，供"昇腾路径已无 950 载体"这一结论溯源）。"""
    return {
        "lines": src.count("\n"),
        "has_Simt_thread_carrier": ("SimtVF" in src) or ("asc_vf_call" in src) or ("threadIdx" in src),
        "has_ASC_IS_": "ASC_IS_" in src,
        "has_mix": "__mix__" in src,
        "has_asc_sync": bool(re.search(r"\basc_sync", src)),
        "has_mte_copy": bool(re.search(r"\basc_copy_(gm2l1|l12gm|nd2nz)", src)),
        "has_gm_bypass": "gm_bypass_dcache" in src,
        "has_ubuf": "__ubuf__" in src,
        "math": sorted(set(re.findall(
            r"\b(expf|exp2f|logf|sqrtf|rsqrtf|sinf|cosf|tanhf|__cce_[a-z0-9_]+)\s*\(", src))),
        "asc_*": sorted(set(re.findall(r"\basc_[A-Za-z0-9_]+", src))),
        "tl::*": sorted(set(re.findall(r"\btl::[A-Za-z0-9_]+", src))),
    }


def _tables(tokens: int, half: int, seed: int = 7):
    """造一对 cos/sin 表（host 预计算，与生产 rope_angle_tables 同一形态：真实角度、可正可负）。"""
    rng = np.random.default_rng(seed)
    ang = np.arange(tokens, dtype=np.float64)[:, None] * rng.random(half)[None, :] * 0.7
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def _drive(sign: int, shape, args) -> tuple[str, str]:
    """一次判决：_plan → _run（dummy 抛 _LaunchSkip 即"编译已过、发射不判"）→ 读 compiled_keys。"""
    tokens, heads, dim, slots = shape
    _slots["n"] = 0
    _slots["src"] = ""
    ascend_env.reset()
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    dtype = getattr(torch, args.dtype)
    half = dim // 2
    qkv = torch.randn(tokens, 3, heads, dim, dtype=dtype).contiguous()
    cos, sin = (torch.from_numpy(x).contiguous() for x in _tables(tokens, half))
    if dtype != torch.float32:                    # _plan 要求 qkv.dtype == cos.dtype
        cos, sin = cos.to(dtype), sin.to(dtype)

    spec = rope_asc._plan(qkv, cos, sin, slots, sign, "ascend")
    if spec is None:
        return REJECT, f"plan 拒：qkv={tuple(qkv.shape)} dt={dtype} cos={tuple(cos.shape)} sign={sign}"
    try:
        rope_asc._run(qkv, cos, sin, spec)
    except _LaunchSkip:
        pass
    except Exception as exc:  # noqa: BLE001
        head = (traceback.format_exception_only(type(exc), exc) or [""])[0].strip()
        return _verdict(head, args)
    return _verdict("", args)


def _verdict(exc_head: str, args) -> tuple[str, str]:
    keys = [k for k in ascend_env.compiled_keys() if k.startswith("ascend|")]
    bl = {k: v for k, v in ascend_env.blockers().items() if k.startswith("ascend|")}
    if keys:
        src = _slots["src"]
        if src and args.dump:
            with open(args.dump, "w") as f:
                f.write(src)
        feats = scan_source(src) if src else {}
        return PASSED, (f"key={keys[0]} .o_src~{_slots['n']}B " +
                        " ".join(f"{k}={feats.get(k)}" for k in sorted(feats)))
    if bl:
        return FAILED, f"keys={keys} blocker={next(iter(bl.values()))[:220]}"
    return FAILED, f"未达 compiled_keys；exc={exc_head[:220]} blocker_cnt={len(bl)}"


def _twin(qkv, cos, sin, slots: int, sign: int) -> np.ndarray:
    """昇腾正文的 numpy 影子件：**逐字照搬**那段循环结构（领行方式、就地顺序、槽位范围）。

    与 `rope_asc_impl` 的对应关系：`for r in T.serial(ceildiv(tokens - bx, NUM_BLOCKS))` →
    下面的 `for r in range(-(-(tokens - bx) // NUM_BLOCKS))`；`t = bx + r * NUM_BLOCKS`；
    `for s in serial(slots)` / `for hh in serial(heads)` / `for j in serial(half)`；
    x1/x2 先各物化再双写（就地顺序）；slot 2 不碰。sign=-1 走"sn 取负"那一条编译期分支。
    """
    tokens, _, heads, dim = qkv.shape
    half = dim // 2
    nb = rope_asc.NUM_BLOCKS
    out = qkv.astype(np.float64).copy()
    for bx in range(nb):
        for r in range(-(-(tokens - bx) // nb)):          # ceildiv(tokens - bx, nb)，含 0 行程
            t = bx + r * nb
            for s in range(slots):
                for hh in range(heads):
                    x1 = out[t, s, hh, :half].copy()      # 先物化（就地不污染）
                    x2 = out[t, s, hh, half:].copy()
                    c = cos[t].astype(np.float64)
                    sn = sin[t].astype(np.float64) * (1.0 if sign > 0 else -1.0)
                    out[t, s, hh, :half] = x1 * c - x2 * sn
                    out[t, s, hh, half:] = x2 * c + x1 * sn
    return out.astype(np.float32)


def _golden(args) -> list[tuple[str, str]]:
    """影子件 vs P1 冻结尺子 rope_ref（覆盖/就地/槽位 2/反向=角取负 四件事一起校）。"""
    from sys1.testing.torch_ref import rope_ref

    rows = []
    for tokens, heads, dim, slots in SHAPES:
        half = dim // 2
        cos_np, sin_np = _tables(tokens, half)
        qkv_np = (np.random.default_rng(11).standard_normal((tokens, 3, heads, dim))
                  .astype(np.float32))
        qkv_t = torch.from_numpy(qkv_np).contiguous()
        cos_t, sin_t = torch.from_numpy(cos_np), torch.from_numpy(sin_np)
        for sign in (1, -1):
            got = _twin(qkv_np, cos_np, sin_np, slots, sign)
            # 尺子：rope_ref 是"照抄表转一遍、返回新副本"；反向=把 sin 取负，与本件同一代数事实
            want = rope_ref(qkv_t, cos_t, -sin_t if sign < 0 else sin_t,
                            tuple(range(slots))).numpy()
            diff = float(np.max(np.abs(got - want)))
            slot2 = bool(np.array_equal(got[:, 2], qkv_np[:, 2]))
            ok = diff < 1e-5 and slot2
            rows.append(((GOLDEN if ok else GOLDEN_FAIL),
                         f"tokens={tokens} heads={heads} dim={dim} slots={slots} sign={sign:+d} "
                         f"max_abs={diff:.2e} slot2_逐位不动={slot2}"))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=int, default=None)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--sign", default="both", choices=["both", "1", "-1"])
    ap.add_argument("--dump", default=os.environ.get("TL_I_DUMP", ""))
    ap.add_argument("--golden", action="store_true", help="额外跑 numpy 影子件对 P1 尺子")
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))

    shapes = SHAPES if args.tokens is None else [(args.tokens, args.heads, args.dim, args.slots)]
    n_fail = 0
    print(f"# verify_rope  NUM_BLOCKS={rope_asc.NUM_BLOCKS} ROTATE_SLOTS={rope_asc.ROTATE_SLOTS} "
          f"shapes={shapes} dtype={args.dtype}")
    if args.golden:
        for verdict, detail in _golden(args):
            if verdict != GOLDEN:
                n_fail += 1
            print(f"{verdict} {detail}")
    _install_stubs()
    signs = [1, -1] if args.sign == "both" else [int(args.sign)]
    for shape in shapes:
        for sign in signs:
            try:
                verdict, detail = _drive(sign, shape, args)
            except Exception as exc:  # noqa: BLE001
                verdict, detail = FAILED, (traceback.format_exception_only(type(exc), exc) or [""])[0][:220]
            if verdict != PASSED:
                n_fail += 1
            print(f"{verdict} shape=t{shape[0]}h{shape[1]}d{shape[2]}s{shape[3]} "
                  f"sign={'+' if sign > 0 else '-'}1 {detail}")
    total = (2 * len(SHAPES) if args.golden else 0) + len(shapes) * len(signs)
    print(f"—— rope 判决 PASS={total - n_fail}/{total} ——")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
