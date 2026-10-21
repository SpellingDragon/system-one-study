#!/usr/bin/env python3
"""attempts/G3_probe_compile_v2.py — G3 编译面自证：双因子提案 v2 打上后**可编译**。

目的（任务 (b) 自证②）：证明 `attempts/S3_compat_proposal_v2.diff` 打上后 **可编译**，且对
trunk **零改动**。手法与 `reconcile/S3_probe_compile_patch.py` 同：把打过提案的
`port910b_compat.h` 作为**临时副本**放在 `<inc>/tl_templates/ascend/`（盘紧故用 /dev/shm），
编译最小 TU 时用 `-I<inc>` 前置覆盖；**从不改 install 里的 trunk 头**。

本文件不改任何产品语义：只编一个最小 `__vector__` TU（aibin 全链，与 codegen 同一条 bisheng
通路），调用 codegen 真实发射的三种形态（k=48 多突发跨步 10 参 / k=32 单发 10 参 / ub2gm 7 参）。

跑法（容器 cann910b-h 内）：
    source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null 2>&1
    python3 /work/ascend/port910b/attempts/G3_probe_compile_v2.py --inc /dev/shm/g3_v2  # v2 → S3-V2-COMPILE PASS
    python3 /work/ascend/port910b/attempts/G3_probe_compile_v2.py --inc /dev/shm/g3_v1  # v1 对照
    python3 /work/ascend/port910b/attempts/G3_probe_compile_v2.py                       # install 对照
    python3 /work/ascend/port910b/attempts/G3_probe_compile_v2.py --inc /dev/shm/g3_v2 --dump
        # ↑ 额外打印 bisheng -E 预处理原文（含 DMA 实参行 / vector_dup 行）供 grep 取证
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")

from tilelang.contrib import bisheng            # noqa: E402
from tilelang.env import TILELANG_TEMPLATE_PATH  # noqa: E402

ROOT = os.environ.get(
    "ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest") + "/aarch64-linux"

TU = r'''
#include <tl_templates/ascend/common.h>
extern "C" __global__ __vector__ void s3_probe(__gm__ float *out, __gm__ float *in) {
  __ubuf__ uint8_t *ub = (__ubuf__ uint8_t *)0;
  /* codegen 真实形态①：k=48 多突发跨步（blockCount=32, len=64B, srcStride=192B, dstStride=64B） */
  asc_copy_gm2ub_align(ub, (__gm__ uint8_t *)in, 32, 64, 0, 0, 0,
                       static_cast<asc_load_l2_cache_mode>(0), 192, 64);
  /* codegen 真实形态②：k=32 单发连续（blockCount=1, len=4096B, stride=4096B） */
  asc_copy_gm2ub_align(ub, (__gm__ uint8_t *)in, 1, 4096, 0, 0, 0,
                       static_cast<asc_load_l2_cache_mode>(0), 4096, 4096);
  /* codegen 真实形态③：ub2gm 7 参 */
  asc_copy_ub2gm_align((__gm__ uint8_t *)out, ub, 1, 2048,
                       static_cast<asc_store_l2_cache_mode>(4), 2048, 2048);
  out[0] = in[0];
}
'''


def header_md5(inc_root):
    p = os.path.join(inc_root, "tl_templates", "ascend", "port910b_compat.h")
    if not os.path.isfile(p):
        return p, "<absent>"
    return p, hashlib.md5(open(p, "rb").read()).hexdigest()


def includes(inc):
    out = ["-I" + TILELANG_TEMPLATE_PATH, "-DTL_PORT910B_NATIVE_TYPES",
           "-I%s/asc/impl" % ROOT, "-I%s/asc/include" % ROOT]
    if inc:
        out = ["-I" + inc] + out
    return out


def dump_preprocessed(inc):
    """bisheng -E：把 TU 与其 include 的头**预处理展开**，露出 DMA 实参行（= 编译器真正看到的码）。"""
    import tempfile
    code = tempfile.NamedTemporaryFile(suffix=".asc", delete=False, mode="w")
    code.write(TU)
    code.close()
    cmd = [bisheng.find_bisheng_path(), "--npu-arch=dav-2201", "-E"] + includes(inc) + [code.name]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    sys.stdout.write(proc.stdout.decode("utf-8", "replace"))
    os.unlink(code.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inc", default="", help="前置 include 根（打过提案的副本），空=用 install")
    ap.add_argument("--dump", action="store_true", help="额外打印 bisheng -E 预处理原文")
    ap.add_argument("--tag", default="S3-V2", help="判决行标签前缀，如 S3-V2 / S3-V1")
    a = ap.parse_args()

    hp, hmd5 = header_md5(a.inc or TILELANG_TEMPLATE_PATH)
    print("%s-COMPILE-HEADER %s md5=%s" % (a.tag, hp, hmd5))

    if a.dump:
        print("%s-DUMP-BEGIN inc=%s" % (a.tag, a.inc or "<install>"), flush=True)
        dump_preprocessed(a.inc)
        print("%s-DUMP-END" % a.tag, flush=True)

    try:
        bisheng.compile_ascend(TU, target_format="aibin", npu_arch="dav-2201", options=includes(a.inc))
        print("%s-COMPILE PASS inc=%s" % (a.tag, a.inc or "<install>"))
        return 0
    except Exception as e:  # noqa: BLE001
        s = str(e)
        errs = re.findall(r"[^\n]*(?:error:|undefined symbol|Fatal Err)[^\n]*", s)
        first = errs[0][:220].strip() if errs else s.splitlines()[-1][:220]
        print("%s-COMPILE FAIL inc=%s :: %s" % (a.tag, a.inc or "<install>", first))
        return 1


if __name__ == "__main__":
    sys.exit(main())
