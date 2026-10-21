#!/usr/bin/env python3
"""attempts/D/patch_compat_D.py — GAP-D1b：就地改写 pip 副本 compat 的 asc_copy_l12l0b_transpose。

缺口（实测证据）：d_dw_910b.py --variant l0tr 首挂于
    <tilelang>/src/tl_templates/ascend/port910b_compat.h:907:29: error: parameters too many
即真源 compat 里 `asc_copy_l12l0b_transpose` 体内对原生件发的是 **6 参**：
    load_cbuf_to_cb_transpose(dst, src, mStart, mStep, srcStride, (uint8_t)0);
而 910B（dav_c220）官方 Cal 层的真实形态是 **8 参**（三处独立证据）：
    asc/impl/basic_api/dav_c220/kernel_operator_mm_impl.h:152
        load_cbuf_to_cb_transpose(dst, src, startIndex, repeatTimes, srcStride, dstGap, inc, dstFracGap)
    asc/impl/basic_api/dav_c220/kernel_operator_cube_others_impl.h:388
        LoadCbufToCbTranspose(dst, src, indexID, repeatTime, srcStride, dstStride, addrmode, dstFracStride)
    asc/include/pto/npu/a2a3/TExtract.hpp:154
        load_cbuf_to_cb_transpose(dstAddr, srcAddr, startIdx0 + i*srcRowNum, dstRowNum, 1, dstGap, false, 0)
⇒ 6 参调用匹配不到重载（clang 报 "parameters too many"，即只匹配到 4 参 packed-config 形）。

为什么这个坑到今天才炸：真源 compat 的 b_transpose 只被 `gemm.h` 模板体的 **TRANS_B=false
（NN）分支** 调用，而 A案立起来的用例全是 `TRANS_B=true`（NT）——`if constexpr` 的死分支
从不实例化，所以这行**从未被真实编过**。本波 l0tr（L1→L0B 转置装填）是它第一次上场。

处置（最小、可回退）：**只改函数体，签名一字不动**；位序沿用族内既有约定
（indexID=发射槽 3 mStart，repeatTime=发射槽 5 mStep，与已证真的非转置 asc_copy_l12l0a
同一取法），addrmode/dstFracStride 依官方形给 false/0，并在行尾留 VERIFY V2' 标记。
幂等：已改写则 no-op；每次 env_setup_D.sh 先 cp 真源重置副本，故改写必然重新生效。

用法：python3 patch_compat_D.py            # 自动定位 pip 树 compat
"""
import glob
import os
import re
import sys

ANCHOR = re.compile(
    r"(asc_copy_l12l0b_transpose[^\n]*\n[^\n]*\n[^\n]*\n)"      # 签名三行
    r"(\s*\(void\)kStart; \(void\)kStep;\n)"
    r"(\s*\(void\)dstStride;[^\n]*\n)"
    r"(\s*load_cbuf_to_cb_transpose\([^;]*\);)", re.M)
MARKER = "GAP-D1b"


def find_compat():
    pats = [
        "/usr/local/lib/python3*/dist-packages/tilelang/src/tl_templates/ascend/port910b_compat.h",
        "/usr/local/lib/python3*/site-packages/tilelang/src/tl_templates/ascend/port910b_compat.h",
        os.path.expanduser("~/.local/lib/python3*/site-packages/tilelang/src/tl_templates/ascend/port910b_compat.h"),
    ]
    for pt in pats:
        for hit in sorted(glob.glob(pt)):
            if os.path.isfile(hit):
                return hit
    try:
        import tilelang
        return os.path.join(os.path.dirname(tilelang.__file__), "src", "tl_templates",
                            "ascend", "port910b_compat.h")
    except Exception as e:  # noqa: BLE001
        raise SystemExit("D-PATCH-FAIL: compat not found (%s)" % e)


def main():
    p = find_compat()
    s = open(p).read()
    if MARKER in s:
        print("compat GAP-D1b already applied (%s)" % p)
        return 0
    m = ANCHOR.search(s)
    if not m:
        raise SystemExit("D-PATCH-FAIL: anchor not found in %s (真源漂移?)" % p)
    new_body = (m.group(1) + m.group(2) +
                "  (void)dstStride;  // VERIFY V2'（与 asc_copy_l12l0a_transpose 同族取法）\n"
                "  // GAP-D1b: 6 参形态在 910B 不存在（官方三处证据均 8 参）→ 改 8 参\n"
                "  load_cbuf_to_cb_transpose(dst, src, (uint16_t)mStart, (uint8_t)mStep,\n"
                "                            (uint16_t)srcStride, (uint16_t)dstStride,\n"
                "                            false, (uint16_t)0);  // VERIFY V2' 位序\n")
    s2 = s[:m.start()] + new_body + s[m.end():]
    open(p, "w").write(s2)
    print("D-PATCH-COMPAT-D1b APPLIED %s (%d -> %d lines)"
          % (p, s.count("\n"), s2.count("\n")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
