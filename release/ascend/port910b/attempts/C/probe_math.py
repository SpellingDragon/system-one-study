#!/usr/bin/env python3
"""attempts/C/probe_math.py — 910B(dav-2201) 标量数学面可用性探针。

做法：直接调 tilelang.contrib.bisheng.compile_ascend 走 **和 tilelang codegen
完全相同的一条 bisheng 通路**（同 .asc 语言、同 target_format=aibin、同 include
与 -DTL_PORT910B_NATIVE_TYPES），每个候选写法一个最小 kernel，打印
`PROBE <name>: PASS/FAIL(+首条 error)`。aibin 会把链接期缺符号也一并判出来
（`__builtin_expf` 这类"编得过、链不上"的写法在本机上就是不可用）。

为什么需要：GDN 短卷积的 SiLU 要 exp；后续 delta-rule 件要 rcp/sqrt/log/tanh。
本机无 NPU → 只能这样在编译+链接面判定"哪个符号真的存在"，缺的进 compat。
"""
import os
import re
import sys

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")

from tilelang.contrib import bisheng  # noqa: E402
from tilelang.env import TILELANG_TEMPLATE_PATH  # noqa: E402

ROOT = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest") + "/aarch64-linux"
BASE_INC = ["-I" + TILELANG_TEMPLATE_PATH, "-DTL_PORT910B_NATIVE_TYPES",
            "-I%s/asc/impl" % ROOT, "-I%s/asc/include" % ROOT]

HDR = '#include <tl_templates/ascend/common.h>\n'

# (名字, 额外 include, 表达式)  —— 表达式里 x 是 __gm__ float* 读出的 float
CASES = [
    # 1) codegen_ascend 实际发射的名字（intrin_rule_ascend.cc: AscendMath → name+'f'）
    ("expf",          "", "expf(-x)"),
    ("sqrtf",         "", "sqrtf(x)"),
    ("rsqrtf",        "", "rsqrtf(x)"),
    ("fabsf",         "", "fabsf(-x)"),
    ("floorf",        "", "floorf(x)"),
    ("log2f",         "", "log2f(x)"),
    ("logf",          "", "logf(x)"),
    ("tanhf",         "", "tanhf(x)"),
    ("powf",          "", "powf(x, 2.0f)"),
    ("exp10f",        "", "exp10f(x)"),
    # 2) 不带 f 的 double 面（clang 对 sqrtf 建议过 'sqrt'，验证是否真可用）
    ("sqrt_d",        "", "sqrt(x)"),
    ("fabs_d",        "", "fabs(-x)"),
    ("exp_d",         "", "exp(-x)"),
    # 3) clang 内建（可能折成指令、也可能退化成 libcall→链不上）
    ("__builtin_fabsf", "", "__builtin_fabsf(-x)"),
    ("__builtin_sqrtf", "", "__builtin_sqrtf(x)"),
    ("__builtin_floorf", "", "__builtin_floorf(x)"),
    ("__builtin_ceilf", "", "__builtin_ceilf(x)"),
    ("__builtin_expf", "", "__builtin_expf(-x)"),
    ("__builtin_exp2f", "", "__builtin_exp2f(x)"),
    ("__builtin_log2f", "", "__builtin_log2f(x)"),
    ("__builtin_rsqrtf", "", "__builtin_rsqrtf(x)"),
    # 4) 头文件路线（已知会把 host libstdc++ 拖进来，留证据）
    ("expf_math_h",   "#include <math.h>", "expf(-x)"),
    # 5) 纯四则基线（对照组：证明探针本身没别的问题）
    ("arith_baseline", "", "x * 2.0f + 1.0f"),
]


def build(expr, extra_inc):
    return HDR + extra_inc + (
        '\nextern "C" __global__ __vector__ void probe(__gm__ float *out, __gm__ float *in) {\n'
        '  float x = in[0];\n'
        '  out[0] = %s + x;\n'
        '}\n' % expr)


def main():
    only = set(sys.argv[1:]) or None
    fails = 0
    for name, extra_inc, expr in CASES:
        if only and name not in only:
            continue
        try:
            bisheng.compile_ascend(build(expr, extra_inc), target_format="aibin",
                                   npu_arch="dav-2201", options=BASE_INC)
            print("PROBE %-18s PASS   (%s)" % (name, expr))
        except Exception as e:  # noqa: BLE001
            fails += 1
            s = str(e)
            m = re.search(r"[^\n]*(error|undefined symbol)[^\n]*", s)
            first = (m.group(0) if m else s.splitlines()[0])[:170]
            print("PROBE %-18s FAIL   (%s) :: %s" % (name, expr, first.strip()))
    print("PROBE-SUMMARY fails=%d" % fails)


if __name__ == "__main__":
    main()
