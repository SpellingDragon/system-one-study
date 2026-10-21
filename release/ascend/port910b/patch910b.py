#!/usr/bin/env python3
"""Patch a tilelang src tree for the 910B (no-device) compile face.
Idempotent; point at any tl_templates/ascend root (repo or pip package).
Usage: patch910b.py <ascend_templates_dir> [compat_src]"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
p = root / "common.h"
if not p.exists():
    p = next(root.glob("**/tl_templates/ascend/common.h"))
s = p.read_text("utf-8")
if "TL_ASCEND_SIMT" not in s:
    s = s.replace(
        '#include "c_api/asc_simd.h"',
        '#if defined(TL_ASCEND_SIMT)\n#include "c_api/asc_simd.h"', 1)
    for h in ('#include "simt_api/asc_bf16.h"',
              '#include "simt_api/asc_fp16.h"',
              '#include "simt_api/asc_fp8.h"',
              '#include "simt_api/cooperative_groups.h"',
              '#include "simt_api/asc_simt.h"'):
        s = s.replace(h, '#if defined(TL_ASCEND_SIMT)\n' + h, 1)
    s = s.replace(
        '#include "tl_templates/ascend/simd_inst.h"',
        '#endif\n#if defined(TL_ASCEND_SIMT)\n#include "tl_templates/ascend/simd_inst.h"', 1)
    if "port910b_compat" not in s:
        s = s.replace(
            '#include "tl_templates/ascend/numeric_limits.h"',
            '#else\n#include "tl_templates/ascend/port910b_compat.h"\n#endif\n#include "tl_templates/ascend/numeric_limits.h"', 1)
    p.write_text(s)
    print("patched", p)
else:
    print("already patched", p)
compat_src = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path(
    "/tilelang/src/tl_templates/ascend/port910b_compat.h")
if compat_src.exists():
    (p.parent / "port910b_compat.h").write_text(compat_src.read_text())
    print("compat placed at", p.parent)
