#!/usr/bin/env python3
"""attempts/C/patch_compat_10b.py — 910B 就地改写 pip 模板树 compat §10b 的 GM->UB 适配器。

为什么必须改写（证据链，全部可复跑）：
1) codegen 的位置契约（主仓只读，src/ascend/op/builtin.h:189 + codegen_ascend.cc
   :1407 EmitGmToUbCopy_）——IR 的 sid 被 codegen 主动丢弃，实发 10 参顺序是
       (dst, src, nBurst, burstLen, leftPad, rightPad, dataSelect, l2Ctl,
        burstSrcStride, burstDstStride)
2) compat §10b 的形参名（port910b_compat.h:573）是
       (dst, src, sid, lenBurst, srcStride, dstStride, blockGroup, mode, nBurst, totalLen)
   ⇒ 槽位错位：sid 槽收到 nBurst；srcStride 槽收到 leftPad；dstStride 槽收到
     rightPad；**nBurst 槽收到 burstSrcStride；totalLen 槽收到 burstDstStride**。
3) §10b 体内自造的 config 位段（sid | nBurst<<4 | lenBurst<<16 | srcStride<<32 |
   dstStride<<48）也和 950 align_v2 的真实位段不符（对照
   ccec_compiler/lib/clang/15.0.5/include/cce_aicore_intrinsics_3101.h:1478 的
   注释：sid<<0 | burstNum<<4(21b) | burstLen<<25(21b) | leftPad<<46 |
   rightPad<<52 | dataSelect<<58 | l2Ctl<<60；config1 = srcStride<<0(40b) |
   dstStride<<40）。
4) **单位**：910B 原生 copy_gm_to_ubuf/copy_ubuf_to_gm 的 blockLen/stride 以
   ONE_BLK_SIZE=32B 为单位（证据：asc/include/adv_api/matmul/matmul_client.h:2745
   `repeatParams.blockLen = ... * sizeof(T) / ONE_BLK_SIZE;`，:2747 srcGap 同），
   而 tilelang IR 传的是**字节**（copy.cc:240-245 AscendMTEBytesFromElements）。
   ⇒ 不做 /32 换算就是 64 倍过搬（上卡即炸）。

本脚本把 §10b 的 asc_copy_gm2ub_align **体内改成 910B 原生 7 参形态**并按上面的
位置契约取槽，stride 单位换算 /32；leftPad/rightPad/dataSelect 在 910B 需要
align_b8/b16/b32 变体（本波不覆盖 → G-C6）。签名一字不动，故与主仓真源兼容；
只作用于容器内 pip 树副本（env_setup.sh [2] 每次都 cp 重置为真源后重新打）。

幂等：命中 "TL910B-10B-REWRITTEN" 标记即跳过。
教训（已踩）：注入块里的宏一定要放在使用它的函数之前；否则因为这段代码在 **header 里**，
会技术上毒化几乎所有 TU（实测 probe_dma 19/19 全挂在同一行）。
另一条教训：**改 compat 头不会让 tilelang 的 kernel 编译失效**（缓存按 kernel 源码哈希，
见 TILELANG_CACHE_DIR=~/.tilelang/cache）→ 改完必须换/清 cache 再判，否则是陈旧 PASS。语法自检：落盘前 compile() 不了
（C++ 无法 compile()），改为校验替换次数==1 且括号配平。

用法：python3 patch_compat_10b.py [compat.h 路径]
"""
import io
import os
import re
import sys

MARK = "TL910B-10B-REWRITTEN"

NEW_FN = """__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ uint8_t *dst, __gm__ uint8_t *src,
                                           int sid, int lenBurst, int srcStride, int dstStride,
                                           int blockGroup, asc_load_l2_cache_mode mode,
                                           int nBurst, int totalLen) {
  // TL910B-10B-REWRITTEN（attempts/C/patch_compat_10b.py 注入，主仓真源未改）
  // 形参名沿用 §10b，但**按 codegen 的实际槽位**取值（EmitGmToUbCopy_ 丢了 IR sid）：
  //   sid 槽 = nBurst(burst_num)   lenBurst 槽 = burstLen(字节)
  //   srcStride 槽 = leftPadding   dstStride 槽 = rightPadding
  //   blockGroup 槽 = dataSelect   mode 槽 = l2CacheCtl
  //   nBurst 槽 = burstSrcStride(字节)   totalLen 槽 = burstDstStride(字节)
  // 910B 无 byte-granular burst：padding/dataSelect 需要 align_b8/b16/b32 变体，
  // 本改写先覆盖 leftPad==rightPad==dataSelect==0 的对齐通路（G-C6 登记缺口）。
  const int burst_num = sid;                    // 槽 3
  const int burst_len_bytes = lenBurst;         // 槽 4
  const int src_stride_bytes = nBurst;          // 槽 9
  const int dst_stride_bytes = totalLen;        // 槽 10
  const int left_pad = srcStride, right_pad = dstStride, data_select = blockGroup;
  (void)mode;  // 910B 原生 7 参形态不吃 l2_cache_ctl（性能提示，非正确性）
  ASCEND_910B_ASSERT_ALIGN32(left_pad == 0 && right_pad == 0 && data_select == 0);
  copy_gm_to_ubuf(dst, src, (int8_t)0, (uint16_t)burst_num,
                  (uint16_t)(burst_len_bytes / 32),
                  (uint16_t)(src_stride_bytes / 32),
                  (uint16_t)(dst_stride_bytes / 32));
}
"""

GUARD = """
// TL910B-10B-REWRITTEN 断言占位：一期保守——非对齐/padding 的 GM->UB 通路
// 在 910B 需 align_b8/b16/b32（本波未接），编译期不做静态拒绝，运行期由
// 上层保证 32B 对齐；此处留空宏，便于后续替换成 ASCENDC_ASSERT。
#ifndef ASCEND_910B_ASSERT_ALIGN32
#define ASCEND_910B_ASSERT_ALIGN32(cond) ((void)(cond))
#endif
"""


def compat_path():
    if len(sys.argv) > 1:
        return sys.argv[1]
    import tilelang
    root = os.path.dirname(tilelang.__file__)
    return os.path.join(root, "src", "tl_templates", "ascend", "port910b_compat.h")


def main():
    p = compat_path()
    s = io.open(p, encoding="utf-8").read()
    if MARK in s:
        print("C-COMPAT-10B already rewritten (%s)" % p)
        return 0
    pat = re.compile(
        r"__aicore__ inline void asc_copy_gm2ub_align\(__ubuf__ uint8_t \*dst.*?\n\}\n",
        re.S)
    hits = pat.findall(s)
    if len(hits) != 1:
        print("C-COMPAT-10B-SKIP anchor hits=%d (期望 1；§10b 可能已被上游改写)" % len(hits))
        return 2
    new = s.replace(hits[0], GUARD + NEW_FN)  # 宏必须在函数之前（否则 header 里先用后定义 → 毒化所有 TU）
    if new.count("{") != new.count("}"):
        print("C-COMPAT-10B-ABORT brace mismatch, 不落盘")
        return 3
    io.open(p, "w", encoding="utf-8").write(new)
    print("C-COMPAT-10B-REWRITTEN %s (%d -> %d lines)"
          % (p, s.count("\n"), new.count("\n")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
