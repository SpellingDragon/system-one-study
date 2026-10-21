#!/usr/bin/env python3
"""reconcile/S3_probe_compile_patch.py — S3 编译面自证（P1-1s 提案 patch 可编译）。

目的（任务书交付物 4）：证明 `attempts/S3_compat_proposal.diff` 打上后 **可编译**，
且对 trunk **零改动**。手法：把打过补丁的 `port910b_compat.h` 作为**临时副本**放在
`/dev/shm/s3_inc/tl_templates/ascend/`（盘紧故用 /dev/shm），编译最小 TU 时用
`-I/dev/shm/s3_inc` 前置覆盖；**从不改 install 里的 trunk 头**。

不改任何产品语义：本文件只编一个最小 `__vector__ ` TU（aibin 全链，与 codegen 同一条
bisheng 通路），调用 codegen 真实发射的三种形态（k=48 多突发跨步 10 参 / k=32 单发 10 参 /
ub2gm 7 参）。

跑法（容器 cann910b-h 内）：
    source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null 2>&1
    python3 /tmp/S3_probe_compile_patch.py --inc /dev/shm/s3_inc   # 打过提案的副本 → 期望 PASS
    python3 /tmp/S3_probe_compile_patch.py                          # 原装 install 对照 → 期望 PASS
"""
import argparse
import hashlib
import os
import re
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inc", default="", help="前置 include 根（打过补丁的副本），空=用 install")
    a = ap.parse_args()

    inc = ["-I" + TILELANG_TEMPLATE_PATH, "-DTL_PORT910B_NATIVE_TYPES",
           "-I%s/asc/impl" % ROOT, "-I%s/asc/include" % ROOT]
    if a.inc:
        inc = ["-I" + a.inc] + inc

    hp, hmd5 = header_md5(a.inc or TILELANG_TEMPLATE_PATH)
    print("S3-COMPILE-HEADER %s md5=%s" % (hp, hmd5))

    try:
        bisheng.compile_ascend(TU, target_format="aibin", npu_arch="dav-2201", options=inc)
        print("S3-COMPILE PASS inc=%s" % (a.inc or "<install>"))
        return 0
    except Exception as e:  # noqa: BLE001
        s = str(e)
        errs = re.findall(r"[^\n]*(?:error:|undefined symbol|Fatal Err)[^\n]*", s)
        first = errs[0][:220].strip() if errs else s.splitlines()[-1][:220]
        print("S3-COMPILE FAIL inc=%s :: %s" % (a.inc or "<install>", first))
        return 1


if __name__ == "__main__":
    sys.exit(main())
