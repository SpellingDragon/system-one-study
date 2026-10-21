#!/usr/bin/env python3
"""attempts/B/dump_final.py — 把两个交付 kernel 的**生成码**落盘，供 attempts/B/ 留证

抓码要绕过两层缓存，否则 compile_ascend 根本不会被调用（第一版就是这么空手而归的）：
  ① tilelang.jit / tilelang.cache 的磁盘缓存：把 TILELANG_CACHE_DIR 指到一次性目录；
  ② tilelang/ascend/backend.py:24-38 的 AscendBinaryCache：命中即返回缓存 ELF，
     所以把 AscendBinaryCache.load 打成恒 None（强制真编译）。
然后 monkeypatch tilelang.contrib.bisheng.compile_ascend（backend.py 里是 `bisheng.compile_ascend(...)`
属性查找，替换有效）抓 .asc；再 runpy 跑交付脚本本体，保证留证码与判决那次同源同路径。
"""
import os

os.environ["ASCEND_NPU_ARCH"] = "dav-2201"
os.environ["TILELANG_CACHE_DIR"] = "/tmp/agentB/cache_dump"  # ① 一次性缓存目录
os.makedirs("/tmp/agentB/gen", exist_ok=True)

import runpy
import sys

from tilelang.cache.ascend_binary_cache import AscendBinaryCache
from tilelang.contrib import bisheng as bs

AscendBinaryCache.load = staticmethod(lambda *a, **kw: None)  # ② 强制 cache miss

OUT_DIR = "/tmp/agentB/gen"
_orig = bs.compile_ascend
CAPTURED = []


def spy(code, *a, **kw):
    CAPTURED.append(code)
    return _orig(code, *a, **kw)


bs.compile_ascend = spy

TARGETS = [("b_addln_910b.py", f"{OUT_DIR}/final_addln.asc"),
           ("b_readout_910b.py", f"{OUT_DIR}/final_readout.asc")]
rc = 0
for script, dst in TARGETS:
    CAPTURED.clear()
    sys.argv = [script]
    try:
        runpy.run_path(f"/tmp/agentB/{script}", run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            rc = 1
    if CAPTURED:
        with open(dst, "w") as f:
            f.write(CAPTURED[-1])
        print(f"SAVED {dst} bytes={len(CAPTURED[-1])}")
    else:
        print(f"NO-GENCODE for {script}（未走到设备编译）")
print("DUMP-FINAL-DONE" if rc == 0 else "DUMP-FINAL-WARN(某 kernel 未 PASS)")
