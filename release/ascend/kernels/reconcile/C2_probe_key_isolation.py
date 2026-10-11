#!/usr/bin/env python3
"""reconcile/C2_probe_key_isolation.py — 位宽**缓存键隔离**判别（fp32 模具不得被 fp16 形状复用）。

【为什么要这一格】矩阵每格各自 `alias()`（内含 `ascend_env.reset()`）重置缓存，所以它**证明不了**
同进程里“先跑 fp32、再拿同形状跑 fp16”会拿到一副匹配的模具。而编译缓存按 key 命中：若 key 不含
位宽段，fp16 形状就会命中 fp32 的产物——编译门对位宽错配是瞎的（`reconcile/I_RESULT.md §⑥-3`），
剩下的唯一拦截是 cython 的 `_check_buffer_dtype`（发射期 ValueError，生产直崩）；runtime 一旦跳过
校验（adapter 有 `skip_tensor_validation` 形参）就是静默错数据。

【判据】同一进程内、每格**只重置一次**，先 fp32 再 fp16（同形状同种子）：
  · 位宽参数化三件（linear/dw/rope）：fp16 档必须**新增**带位宽段的模具（`dfloat16`/`qfloat16`），
    且两档数值各过自己的阈值；
  · 入口物化六件（readout/add_ln/gdn_fwd/gdn_bwd/attn_sw/lora）：入口 `.float()` 升宽后内核恒 fp32
    ⇒ 模具与位宽无关，**复用是正确构造**（若复用真会错数据，这里必然 rel 爆掉或发射期崩）。
判决行：`C2-ISO-<row>-<dt>-PASS rel=<v> tol=<t> 本档新增模具=<[...]`；末行 `C2-ISO-SUMMARY ...`。

用法（缓存纪律照旧：换新 TILELANG_CACHE_DIR）：
  TILELANG_CACHE_DIR=$(mktemp -d) .venv/bin/python ascend/kernels/reconcile/C2_probe_key_isolation.py
  可选：位置参数挑格，如 `... linear dw rope`
坑（本探针首版踩过）：**必须**复用矩阵探针里的 `ascend_env` 对象（它就是产品件 import 的那一个）；
若另行 `import ascend_env`（顶层名）会拿到**另一个模块实例**、其 `_compiled` 永远为空 ⇒ 判别全盲。
注意：**不能**加 `from __future__ import annotations`（PEP 563 破坏 prim_func 注解闭包）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from C2_probe_dtype_matrix import CELLS, alias, ascend_env  # noqa: E402

#: 位宽参数化件（正文按 dtype 声明 ⇒ 缓存键必须带位宽段）
PARAM = {"linear": "dfloat16", "dw": "dfloat16", "rope": "qfloat16"}
#: 每格在缓存键里的定位串（列模具用）
MARK = {"linear": "gemm[ascend]", "dw": "gemm_dw[ascend]", "readout": "gather[ascend]",
        "add_ln": "add_ln[ascend]", "gdn_fwd": "gdn[ascend]", "gdn_bwd": "gdnb[ascend]",
        "rope": "rope[ascend]", "attn_sw": "attn_sw[ascend]", "lora": "gemm[ascend]"}
FAILS = []


def molds(mark):
    return sorted(k for k in ascend_env.compiled_keys() if k.startswith("ascend|") and mark in k)


def one(name, fn):
    alias()                                                 # 每格**只重置一次**，两档共用同一缓存
    mark = MARK[name]
    seen = {}
    for dtn in ("fp32", "fp16"):
        before = set(molds(mark))
        try:
            relv, tol, detail, key = fn(dtn)
        except Exception as exc:  # noqa: BLE001
            print(f"C2-ISO-{name}-{dtn}-FAIL {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
            FAILS.append(f"{name}/{dtn}")
            return
        now, new = set(molds(mark)), sorted(set(molds(mark)) - before)
        seen[dtn] = now
        print(f"C2-ISO-{name}-{dtn}-PASS rel={relv:.3e} tol={tol} 本档新增模具={new} "
              f"该格全部模具={sorted(now)}")
    extra = seen["fp16"] - seen["fp32"]
    if name in PARAM:
        tag = PARAM[name]
        assert any(tag in k for k in extra), (
            f"{name} fp16 未产生新模具（位宽没进缓存键 ⇒ fp32 产物被复用）；新增={sorted(extra)}")
        print(f"C2-ISO-{name}-键隔离=新模具（含 {tag}，共 {len(extra)} 副）")
    else:                                              # 入口物化件：模具与位宽无关＝复用是正确构造
        print(f"C2-ISO-{name}-键隔离=复用 fp32 模具（入口 .float() 物化、内核恒 fp32；两档均 PASS "
              f"即证复用无害）fp16 档新增={sorted(extra)}")


def main():
    rows = sys.argv[1:]
    for name, fn in CELLS:
        if rows and name not in rows:
            continue
        one(name, fn)
    print(f"C2-ISO-SUMMARY {len(CELLS) - len(FAILS)}/{len(CELLS)} 格双档闭环 失败={FAILS or '无'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
