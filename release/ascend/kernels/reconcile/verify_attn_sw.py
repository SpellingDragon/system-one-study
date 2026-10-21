#!/usr/bin/env python3
"""reconcile/verify_attn_sw.py — 接口 home `attn_sw_asc` 的 910B target=ascend 编译判决（代理 I）。

【做什么】驱动**真接口路径**判两件事：① `attn_sw_asc` 的昇腾正文在 patched 910B tilelang 下能不能
出 .o；②（`--golden`）那段正文的**循环结构**是否还是同一个因果滑窗口径。链路是
attn_sw_asc.plan(target="ascend") → attn_sw_asc.run → ascend_env.get_compiled →
tilelang.compile(..., target="ascend")，本件不复制算式进 DSL、不改动被测件。
【怎么做】① 编译判决与发射隔离：真跑 tilelang.compile，成功后换成"调用即跳"的 dummy 顶掉
kernel 发射（本机无 NPU，本波只判编译；真机数值 rel=1.41e-07 是 attempts/E 的 **stage** 形态在
P1-1c 收的，本件合流的正是那一形态，所以数值面待卡只需复验"动态 seq 领行 + 下标夹位"这两处新增）；
② 判据取 `compiled_keys()`（真编过并缓存过才算，R14 防假绿），失败侧取 `blockers()` 原文首行；
③ 一次跑一串形状，把"动态 seq"这条合流新增的歧义面逐个钉死：默认形状（照 attempts/E 的
heads4/seq64/dim32/window8）、seq 不被 NUM_BLOCKS 整除、seq 不被块宽整除、`window == seq`、
`window > seq`（下标夹位那条路）；同一 key 两次不同 seq 都编过 = "换长度不换产物"仍在；
④ 顺带对 codegen 产物体检：SIMT 线程载体应为 False、`expf`（compat §11 软件件）应出现、被访问
的 UB 数组应落在 `buf_dyn_shmem` 动态池且不重叠（G-C8 反例形态是"恰好 1 个被访问的 shared"）；
⑤ `--golden` 打的是**numpy 影子件**：按昇腾正文同一套"每 (头,核) 隔 NUM_BLOCKS 领行、窗内
打分→行最大→exp→加权→除分母（含 FLOOR）"的结构与 `min(lo+j, seq-1)` 的夹位重算一遍，尺子取
P1 冻结件 `sys1.testing.torch_ref.attn_sw_ref`（不用本模块自带的 `_eager`，免得自己给自己打分）。
影子件校四件事：**领行不重不漏**（覆盖整条 seq）、**可见窗两头都拦**（与尺子逐位级同值）、
**窗外严格 0**（把看不见的键整列扰动，那些行必须逐位不动；同时看得见那些键的行必须真的变，
免得判据空转）、**夹位不改数值**。它不证明 DSL 与 expf 软件件的数值（那要上卡），如实划界。
【为什么】判决口径与 ascend/compile_all_asc.py 一致（同一套 stub、同一个 compiled_keys 判据），
per-op 精修与九件总判决两套工装不会各说各话；plan 拒与编译失败分开打印，避免把"形状没过判据"
误报成"方言编不出"。

一行判决：`I-ATTNSW-ASC-COMPILE-PASS case=<形状> key=... .o_src~<NB>`（每个形状一行）
影子件：`I-ATTNSW-ASC-GOLDEN-PASS case=<形状> max_abs=... 窗外扰动逐位不动=True 该变的真变了=True`
失败：`I-ATTNSW-ASC-COMPILE-FAIL <blocker 首行>`；plan 拒：`I-ATTNSW-ASC-PLAN-REJECT`。
（编排口径里写的 `I-ATtnSW-ASC-COMPILE-PASS` 与本行是同一判决，只是大小写笔误。）

用法（容器 cann910b-i，release 挂到 /work）：
  docker exec cann910b-i bash -c 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \\
    export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201; \\
    cd /work && TILELANG_CACHE_DIR=$(mktemp -d) timeout 600 \\
      python3 ascend/kernels/reconcile/verify_attn_sw.py --golden'
只跑一个形状：`--heads 4 --seq 64 --dim 32 --window 8`（给了 --seq 就只跑该单例）
缓存纪律：每次判决必须换新 `TILELANG_CACHE_DIR=$(mktemp -d)`（陈旧 PASS 之坑见 LOCAL_RUN.md）。
注意：**不能** 在本文件加 `from __future__ import annotations`（eager builder 的
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

import numpy as np    # noqa: E402
import torch          # noqa: E402
import tilelang       # noqa: E402

from ascend.kernels import ascend_env, attn_sw_asc   # noqa: E402

PASSED = "I-ATTNSW-ASC-COMPILE-PASS"
FAILED = "I-ATTNSW-ASC-COMPILE-FAIL"
REJECT = "I-ATTNSW-ASC-PLAN-REJECT"
GOLDEN = "I-ATTNSW-ASC-GOLDEN-PASS"
GOLDEN_FAIL = "I-ATTNSW-ASC-GOLDEN-FAIL"

# 形状串 (heads, seq, dim, window)：dim 需被 16 整除（plan 硬门槛），seq 一律取动态轴的刁钻值。
CASES = [
    (4, 64, 32, 8),      # 照 attempts/E 默认形状（P1-1c 真机 rel=1.41e-07 的那一档）
    (4, 33, 32, 8),      # seq 不被 NUM_BLOCKS 整除：跨步领行的尾行（与上一档同 key！）
    (2, 70, 16, 16),     # seq 不被块宽整除 + 最小合法 dim
    (1, 64, 64, 64),     # window == seq（整窗因果口径）
    (2, 24, 16, 64),     # window > seq：窗外格的下标夹位路
]


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
    """codegen 产物体检（与 attempts/E 同格式）：证明昇腾路径已无 950 载体、且吃的是软件 expf。"""
    return {
        "lines": src.count("\n"),
        "has_Simt_thread_carrier": ("SimtVF" in src) or ("asc_vf_call" in src) or ("threadIdx" in src),
        "has_ASC_IS_": "ASC_IS_" in src,
        "has_mix": "__mix__" in src,
        "has_mte_copy": bool(re.search(r"\basc_copy_(gm2l1|l12gm|nd2nz)", src)),
        "has_gm_bypass": "gm_bypass_dcache" in src,
        "has_ubuf": "__ubuf__" in src,
        "ub_named_null_handles": sorted(set(re.findall(
            r"__ubuf__\s+\w+\s*\*\s*(?!buf_dyn_shmem)(\w+)\s*=\s*\(__ubuf__\s+\w+\s*\*\)\s*0", src))),
        "math": sorted(set(re.findall(
            r"\b(expf|exp2f|logf|sqrtf|rsqrtf|sinf|cosf|tanhf|__cce_[a-z0-9_]+)\s*\(", src))),
        "asc_*": sorted(set(re.findall(r"\basc_[A-Za-z0-9_]+", src))),
        "tl::*": sorted(set(re.findall(r"\btl::[A-Za-z0-9_]+", src))),
    }


def _drive(case, args) -> tuple[str, str]:
    """一次判决：plan → run（dummy 抛 _LaunchSkip 即"编译已过、发射不判"）→ 读 compiled_keys。"""
    heads, seq, dim, window = case
    _slots["n"] = 0
    _slots["src"] = ""
    ascend_env.reset()
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True

    def mk():
        return torch.randn(heads, seq, dim, dtype=torch.float32).contiguous()

    q, k, v = mk(), mk(), mk()
    o = torch.empty((heads, seq, dim), dtype=torch.float32)
    scale = dim ** -0.5                       # 与入口默认口径一致
    spec = attn_sw_asc.plan(q, k, v, o, window, scale, "ascend")
    if spec is None:
        return REJECT, f"plan 拒：shape=(h{heads},s{seq},d{dim}) w{window}"
    try:
        attn_sw_asc.run(q, k, v, o, spec)
    except _LaunchSkip:
        pass
    except Exception as exc:  # noqa: BLE001
        return _verdict((traceback.format_exception_only(type(exc), exc) or [""])[0].strip(), args)
    return _verdict("", args)


def _verdict(exc_head: str, args) -> tuple[str, str]:
    keys = [kk for kk in ascend_env.compiled_keys() if kk.startswith("ascend|")]
    bl = {kk: vv for kk, vv in ascend_env.blockers().items() if kk.startswith("ascend|")}
    if keys:
        src = _slots["src"]
        if src and args.dump:
            with open(args.dump, "w") as f:
                f.write(src)
        feats = scan_source(src) if src else {}
        return PASSED, (f"key={keys[0]} .o_src~{_slots['n']}B " +
                        " ".join(f"{kk}={feats.get(kk)}" for kk in sorted(feats)))
    if bl:
        return FAILED, f"keys={keys} blocker={next(iter(bl.values()))[:240]}"
    return FAILED, f"未达 compiled_keys；exc={exc_head[:240]} blocker_cnt={len(bl)}"


def _twin(q, k, v, window: int, scale: float) -> np.ndarray:
    """昇腾正文的 numpy 影子件：**逐字照搬**那段循环结构（领行方式、夹位、四趟顺序）。

    与 `attn_sw_asc_impl` 的对应关系：`T.Kernel(heads * NUM_BLOCKS)` → `for bx in range(...)`，
    `hh = bx // NUM_BLOCKS` / `core = bx % NUM_BLOCKS`；
    `for rr in T.serial(ceildiv(seq - core, NUM_BLOCKS))` → 下面的 `range(-(-(seq-core)//NB))`；
    `i = core + rr*NB`、`lo = max(i-win+1, 0)`、`n = i-lo+1`、`kk = min(lo+j, seq-1)`；
    趟1 分数（窗外写 NEG）→ 趟2 行最大 → 趟3 exp 与分母 → 趟4 加权 V 除以 `max(l, FLOOR)`。
    """
    heads, seq, dim = q.shape
    win, nb = int(window), attn_sw_asc.NUM_BLOCKS
    out = np.zeros_like(q, dtype=np.float64)
    for bx in range(heads * nb):
        hh, core = divmod(bx, nb)
        for rr in range(-(-(seq - core) // nb)):
            i = core + rr * nb
            lo = max(i - win + 1, 0)
            n = i - lo + 1
            idx = np.minimum(lo + np.arange(win), seq - 1)      # 夹位：与 DSL 的 T.min 同式
            sc = (q[hh, i] @ k[hh, idx].T) * scale              # 趟1
            sc = np.where(np.arange(win) < n, sc, attn_sw_asc.NEG)   # 窗外写哨兵
            m = float(np.max(sc))                               # 趟2
            ps = np.exp(sc - m)                                 # 趟3（expf 换 numpy exp，只校结构）
            l = float(ps.sum())
            out[hh, i] = (ps @ v[hh, idx]) / max(l, attn_sw_asc.FLOOR)   # 趟4
    return out.astype(np.float32)


def _golden() -> list[tuple[str, str]]:
    """影子件 vs P1 冻结尺子 attn_sw_ref：覆盖 / 可见窗 / 窗外严格 0 / 夹位不改数值。"""
    from sys1.testing.torch_ref import attn_sw_ref

    rows = []
    for heads, seq, dim, window in CASES:
        rng = np.random.default_rng(seq * 7 + heads)
        q, k, v = (rng.standard_normal((heads, seq, dim)).astype(np.float32) for _ in range(3))
        scale = dim ** -0.5
        got = _twin(q, k, v, window, scale)
        want = attn_sw_ref(torch.from_numpy(q)[None], torch.from_numpy(k)[None],
                           torch.from_numpy(v)[None], window, scale, None)[0].numpy()
        diff = float(np.max(np.abs(got - want)))
        # 窗外严格 0：把 p0 之后的键整列放大 3 倍。看不见它们的行（i < p0）必须逐位不动，
        # 看得见的那些行必须真的变——两头一起打，判据才不会空转。
        p0 = min(seq - 1, window)
        k2 = k.copy()
        k2[:, p0:] *= 3.0
        pert = _twin(q, k2, v, window, scale)
        quiet = np.arange(seq) < p0
        blind_ok = bool(np.array_equal(pert[:, quiet], got[:, quiet])) if quiet.any() else True
        seen = ~quiet
        seen_ok = bool(not np.array_equal(pert[:, seen], got[:, seen])) if seen.any() else True
        ok = diff < 1e-4 and blind_ok and seen_ok
        rows.append((GOLDEN if ok else GOLDEN_FAIL,
                     f"case=h{heads}s{seq}d{dim}w{window} max_abs={diff:.2e} "
                     f"窗外扰动逐位不动={blind_ok} 该变的真变了={seen_ok}"))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--seq", type=int, default=None)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--dump", default=os.environ.get("TL_I_DUMP", ""))
    ap.add_argument("--golden", action="store_true", help="额外跑 numpy 影子件对 P1 尺子")
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))

    cases = CASES if args.seq is None else [(args.heads or 4, args.seq, args.dim, args.window)]
    n_fail = 0
    print(f"# verify_attn_sw  NUM_BLOCKS={attn_sw_asc.NUM_BLOCKS} NEG={attn_sw_asc.NEG} "
          f"FLOOR={attn_sw_asc.FLOOR} 形状串={cases}")
    if args.golden:
        for verdict, detail in _golden():
            if verdict != GOLDEN:
                n_fail += 1
            print(f"{verdict} {detail}")
    _install_stubs()
    for case in cases:
        try:
            verdict, detail = _drive(case, args)
        except Exception as exc:  # noqa: BLE001
            verdict, detail = FAILED, (traceback.format_exception_only(type(exc), exc) or [""])[0][:240]
        if verdict != PASSED:
            n_fail += 1
        print(f"{verdict} case=h{case[0]}s{case[1]}d{case[2]}w{case[3]} {detail}")
    total = (len(CASES) if args.golden else 0) + len(cases)
    print(f"—— attn_sw 判决 PASS={total - n_fail}/{total} ——")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
