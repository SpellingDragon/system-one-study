#!/usr/bin/env python3
"""Z_patch_sec12.py — p2-13 P1-1k：把 X_sec12_proposal.md 的 P0/P1/P2 落成 trunk §12 代码。

定点替换（每个锚点断言恰命中一次），改 trunk
  $WORKSPACE_ROOT/tilelang/src/tl_templates/ascend/port910b_compat.h
并同步镜像
  release/ascend/port910b/patches/port910b_compat.h（md5 必须一致）

用法：python3 Z_patch_sec12.py            # 应用
      python3 Z_patch_sec12.py --dry     # 只验锚点
"""
import hashlib
import sys

TRUNK = "$WORKSPACE_ROOT/tilelang/src/tl_templates/ascend/port910b_compat.h"
MIRROR = "$REPO_ROOT/release/ascend/port910b/patches/port910b_compat.h"

# ── 块1：§12 头部「建材 / 上卡待证真点」→ 考古定论 + U# 指向 ─────────────────
OLD1_S = "// 建材（全部 raw 指针，无需 TPipe/LocalTensor；a11/a16/a17 实测签名）："
OLD1_E = "//   V3 l0c2gm 的 nSize/mSize/srcStride 分块步距\n"
NEW1 = """// 建材（全部 raw 指针，无需 TPipe/LocalTensor；a11/a16/a17 实测签名）：
//   copy_gm_to_cbuf_multi_nd2nz_b8/b16/b32s —— dav_c220 11 参 ND2NZ 件（P1-1k 写侧；
//     sema D1/D5/D6 rc=0。旧 8 参 copy_gm_to_cbuf = DataCopyGM2L1Impl，ND 线性块拷贝
//     不产 NZ，本路弃用——X_FRACTAL_EVIDENCE §5.4/§6）
//   LoadData2DL12L0ACal/B2Cal    —— asc/impl/basic_api/dav_c220/kernel_operator_mm_impl.h
//   MmadCal<T,U,S>(cc,ca,cb,MmadParams) —— bf16/fp16/fp32/int8 组合官方支持
//   FixpipeL0C2UBImpl + copy_ubuf_to_gm —— L0C→GM 两段式（fix 不支持 32bit 直出 GM）
// VERIFY 状态（P1-1k 落 X 取证成套，2026-10-11；编号= X_FRACTAL_EVIDENCE §8）：
//   V1 已关闭为定论：GM/L1 侧 stride/len 单位 = 32B 块（DEFAULT_C0_SIZE=32，地址模型
//     kernel_check_data_copy_overflow.h:505-520）；GM→L1 必须走 ND2NZ 件；padFuncMode 一支
//     作废（2201 ND2NZ 件无该槽）。
//   V2 方向已定论（§5.2 四条硬规则）：startIndex≡0（二维起点折进 src 指针）、
//     repeatTimes=kStep（u8）、srcStride=row16 块数、M 向外层 for 拆趟、sid≡0（含 P0
//     止血：sid 槽不再吃 IR m_start）。dst 侧步进/gap 未定 → 各件 U2/U3。
//   V3 l0c2gm 的 nSize/mSize/srcStride 分块步距 —— 本轮未取证，保留。
//   未定论 U1–U6 总表见 X_FRACTAL_EVIDENCE §8，随真机 E-1/E-2/E-4 收口。
"""

# ── 块2：asc_copy_gm2l1_nd2nz / dn2nz（P1） ────────────────────────────────
OLD2_S = "// GM→L1（ND/DN 首版同路：行块连续搬运。V1：NZ 语义在 padFuncMode 与块步距上卡定）"
OLD2_E = "  asc_copy_gm2l1_nd2nz(dst, src, rowBytes, mode, nRows, cols, pad1, pad2);  // VERIFY V1\n}\n"
NEW2 = """// GM→L1（P1-1k 落提案 §2）：910B 官方 Matmul 写 L1 只有 ND2NZ/NZ2NZ 两路
//   （data_copy_wrapper_nd.h:41-115 / data_copy_wrapper_nz.h:39-80），不存在“ND 直写
//   L1 喂 cube”。旧形 8 参 copy_gm_to_cbuf 与 DataCopyGM2L1Impl 逐字同款
//   （kernel_operator_data_copy_impl.h:83-108）→ L1 落行主序 ND，而读侧 V1 的
//   startIndex 文档语义 = Fractal matrix ID 按分形块寻址（kernel_operator_mm_intf.h:26-36），
//   且 (void)cols 丢了 IR d_value、asc_set_gm2l1_nz_para 空实现 → c0 pitch 从未生成
//   = 布局侧根因（X_FRACTAL_EVIDENCE §6）。改走 11 参 ND2NZ 件，按 sizeof(ST) 选族。
// 字段单位（官方文档 kernel_operator_data_copy_intf.h:44-56 + 地址模型
//   kernel_check_data_copy_overflow.h:505-520）：nValue/dValue/srcNdMatrixStride/
//   srcDValue=元素；dstNzC0Stride/dstNzNStride=**32B 块**；dstNzMatrixStride=元素。
// 取值照抄官方 tiling（data_copy_wrapper_nd.h:74-97）：ndNum=1、srcNdMatrixStride=0、
//   dstNzC0Stride=Ceil(height,16)*16、dstNzNStride=1、dstNzMatrixStride=0；
//   sid 传字面 0（data_copy_impl.h:275；D2 探针证 sid 同族 4-bit [0,15]）。
// IR 口径（copy.cc:401-410 + codegen_ascend.cc:1458 逐位透传）：rowBytes=args[3]=
//   src_row_stride_bytes（GM **全行宽**，恰是官方 gCol 的语义）、nRows=args[5]=n_value
//   （height）、cols=args[6]=d_value（width）。
// VERIFY U1：dstNzC0Stride(32B 块) 与读侧 srcStride(row16 分形块) 是同一物理量的两种
//   单位——不变式 `codegen 的 L1 outer1 == Ceil(nRows,16)`（bf16：srcStride*16 块 == c0s）。
//   §12 感知不到 codegen buffer extent，static 一致性检查需 codegen 同窗（本窗禁改）→
//   以注释钉死该不变式，随 E-1（位型回读）闭合。
template <typename DT, typename ST>
__aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ DT *dst, __gm__ ST *src,
                                            int rowBytes, asc_load_l2_cache_mode mode,
                                            int nRows, int cols, int pad1, int pad2) {
  (void)mode; (void)pad1; (void)pad2;  // l2_cache_ctrl 在 2201 ND2NZ 件无对应槽；pad 区由
                                       // AscendInsertOOBPadding 预处理 / asc_fill_l1 收尾（U4）
  __cbuf__ ST *const d = (__cbuf__ ST *)(uintptr_t)dst;
  __gm__ ST *const s = (__gm__ ST *)(uintptr_t)src;
  uint32_t const gColW = (uint32_t)((unsigned)rowBytes / (unsigned)sizeof(ST));  // 官方 gCol（元素）
  uint16_t const h = (uint16_t)nRows;   // nValue（行/高，单位=元素）
  uint16_t const w = (uint16_t)cols;    // dValue（宽，单位=元素）
  uint16_t const c0s = (uint16_t)((((unsigned)nRows + 15u) / 16u) * 16u);  // dstNzC0Stride(32B 块)
  if (gColW < 0xffffU) {  // 官方 else 支（wrapper_nd.h:108）：单发，srcDValue=gCol
    uint16_t const gCol = (uint16_t)gColW;
    if constexpr (sizeof(ST) == 1) {
      copy_gm_to_cbuf_multi_nd2nz_b8((__cbuf__ int8_t *)(uintptr_t)d, (__gm__ int8_t *)(uintptr_t)s,
          (int8_t)0, (uint16_t)1, h, w, (uint16_t)0, gCol, c0s, (uint16_t)1, (uint16_t)0);
    } else if constexpr (sizeof(ST) == 2) {
      copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ bfloat16_t *)(uintptr_t)d,
                                      (__gm__ bfloat16_t *)(uintptr_t)s,
          (int8_t)0, (uint16_t)1, h, w, (uint16_t)0, gCol, c0s, (uint16_t)1, (uint16_t)0);
    } else {
      copy_gm_to_cbuf_multi_nd2nz_b32s((__cbuf__ float *)(uintptr_t)d, (__gm__ float *)(uintptr_t)s,
          (int8_t)0, (uint16_t)1, h, w, (uint16_t)0, gCol, c0s, (uint16_t)1, (uint16_t)0);
    }
  } else {
    // 官方 2201 越界兜底（data_copy_wrapper_nd.h:99-107 逐语义复刻）：全部字段 u16
    //   （kernel_struct_data_copy.h:185-192），gCol >= UINT16_MAX 时 M 向**逐行拆趟**：
    //   nValue=1、srcDValue=dValue=width（gCol 不再进任何槽 → 无截断）；dst 每趟前进
    //   1 个 32B 块（= 32/sizeof(ST) 个元素）、src 每趟前进 gCol 个元素。代价 = nRows 条
    //   指令（纯性能面，语义不损，官方同款）。
    int const c0e = 32 / (int)sizeof(ST);  // DEFAULT_C0_SIZE=32 → 每块元素数
    for (int i = 0; i < nRows; ++i) {
      if constexpr (sizeof(ST) == 1) {
        copy_gm_to_cbuf_multi_nd2nz_b8(d + (int64_t)i * c0e, s + (int64_t)i * gColW,
            (int8_t)0, (uint16_t)1, (uint16_t)1, w, (uint16_t)0, w, c0s, (uint16_t)1, (uint16_t)0);
      } else if constexpr (sizeof(ST) == 2) {
        copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ bfloat16_t *)(uintptr_t)(d + (int64_t)i * c0e),
                                        (__gm__ bfloat16_t *)(uintptr_t)(s + (int64_t)i * gColW),
            (int8_t)0, (uint16_t)1, (uint16_t)1, w, (uint16_t)0, w, c0s, (uint16_t)1, (uint16_t)0);
      } else {
        copy_gm_to_cbuf_multi_nd2nz_b32s((__cbuf__ float *)(uintptr_t)(d + (int64_t)i * c0e),
                                         (__gm__ float *)(uintptr_t)(s + (int64_t)i * gColW),
            (int8_t)0, (uint16_t)1, (uint16_t)1, w, (uint16_t)0, w, c0s, (uint16_t)1, (uint16_t)0);
      }
    }
  }
}
template <typename DT, typename ST>
__aicore__ inline void asc_copy_gm2l1_dn2nz(__cbuf__ DT *dst, __gm__ ST *src,
                                            int rowBytes, asc_load_l2_cache_mode mode,
                                            int nRows, int cols, int pad1, int pad2) {
  // dn2nz 本窗**不改**（提案 §2 不改项）：2201 另有 copy_gm_to_cbuf_multi_dn2nz
  //   （cce_aicore_intrinsics.h:980），其 Nd2NzParams 同族字段文档为 dn 版
  //   （srcDnMatrixStride/srcDValue），与 nValue/dValue 的互换关系需 compile+真机一次判别
  //   （X_FRACTAL_EVIDENCE §8-E1 的 dn 扩展）→ 保持委托。注意：P1 落地后本委托=以 IR
  //   互换过的 n_value/d_value（copy.cc:399-402：transpose 路 n=cols、d=rows）发 ND2NZ
  //   件——与官方 B 侧 tiling 口径同向，但 dn 专属件差异未证 → VERIFY U5/E-1(dn)。
  asc_copy_gm2l1_nd2nz(dst, src, rowBytes, mode, nRows, cols, pad1, pad2);  // VERIFY U5
}
"""

# ── 块3：asc_copy_l12l0a / l12l0b（P2，含 P0 sid≡0） ───────────────────────
OLD3_S = "// P1-1d 修复（2026-10-10）：官方 LoadData2DL12L0ACal（mm_impl.h:32/35）发 **9 参**"
OLD3_E = "                  (__cce_scalar::addr_cal_mode_t)0);  // 官方 9 参位序（对称 LoadData2DL12L0BCal）\n}\n"
NEW3 = """// P1-1d（2026-10-10）遗留结论：官方 LoadData2DL12L0ACal（mm_impl.h:32/35）发 **9 参**
//   (dst,src,startIndex,repeatTimes,srcStride,dstGap,sid,transpose(0/1),inc) —— **选族正确**，
//   P1-1d 补齐 9 参后仍崩，说明病灶在槽位（考古 P1-1j/X 结案）。
// P1-1k（2026-10-11，落提案 §1/§3）逐槽纠正（X_FRACTAL_EVIDENCE §7.2 错位表）：
//   ・第 7 槽 sid = 4-bit（sema 实证 [0,15]，**运行期值不检查** §2.4 RT3 rc=0）；旧形把
//     codegen args[2]（= IR **m_start**，row16 块起点，0..127 量级，copy.cc:457/535-538）
//     回穿进 sid → 越界即非法 L1 分区 = 507015 头号嫌疑。本窗 sid≡0（官方 Matmul 恒 0，
//     §5.2 规则 4），m_start 的 M 向语义改由本件指针+拆趟承担（P0 止血并入 P2，P0 的
//     “钉 0 判别”对照件保留在 X_probe_proposal.py::P0_bleed_sid0，供下窗 E-3 二选一）。
//   ・startIndex 官方语义 = Fractal matrix ID，承载不了二维起点 → 折进 src 指针（规则 1）；
//     repeatTimes(u8) 的正确轴向 = kStep（旧 (uint8_t)mStep 轴选反，且 IR m_step>255 还
//     8-bit 静默截断）。2201 无 2DV2（LoadData2DParamsV2 运行期 NOT_SUPPORT，§5.3）→
//     V2 语义只能“折指针 + 拆趟循环”手工降级，不存在等价单条指令。
//   形参 `sid` → `mStart` 更名：第 3 实参一直是 IR m_start，旧名是错位放大器（§7.2 末）。
template <typename T>
__aicore__ inline void asc_copy_l12l0a(__ca__ T *dst, __cbuf__ T *src, int mStart, int kStart,
                                       int mStep, int kStep, int srcStride, int dstStride) {
  // P2（按官方 V1 模板 load_to_l0a_load2d.h:41-110 降级，§5.2 四条硬规则）：
  //   (1) 二维起点折进指针，startIndex≡0；(2) repeatTimes 沿 K/C0 = kStep；
  //   (3) srcStride = L1 的 row16 块数（= IR src_stride，口径本就一致，§7.1）；
  //   (4) M 向外层 for 拆 mStep 趟，每趟 src 前进 1 个分形块、dst 前进 kStep 个分形块。
  (void)dstStride;  // VERIFY U2：假定 L0 目的紧凑（dst 侧 pitch == kStep 个分形块）；非紧凑
                    //   须改用 dstGap 或再拆趟——官方 dstOffset=blockUseK*CUBE_MAX_SIZE/factor_，
                    //   CUBE_MAX_SIZE 口径未抓到（X_FRACTAL_EVIDENCE §8-U2，随 E-2/E-4 标定）。
                    // VERIFY U3：dstGap≡0（官方 L0A 路该槽 0；L0B 路为 nFraC0-1，§8-U3 未定论）。
  int const blkElems = 16 * (int)(32 / (int)sizeof(T));  // 分形块元素数（bf16: 256；16×32B 恒等）
  __cbuf__ T *base = (__cbuf__ T *)(uintptr_t)src + (int64_t)(kStart * srcStride + mStart) * blkElems;
  __ca__ T *dbase = (__ca__ T *)(uintptr_t)dst;
  if (kStep == 1) {  // 官方同款退化（blockUseK==1 支）：repeat 改沿 M、srcStride=1
    load_cbuf_to_ca(dbase, base, (uint16_t)0, (uint8_t)mStep, (uint16_t)1, (uint16_t)0,
                    (uint8_t)0, false, (__cce_scalar::addr_cal_mode_t)0);
  } else {  // repeatTimes=u8：kStep>255 静默截断（§2.3 E2 实证类型）——本窗不设 ICHECK，
            // 910B tile 的 C0 组数远低于 255，越界属真机 E-2 判别面
    for (int i = 0; i < mStep; ++i) {  // 块线性序 block_id = k_block*pitch + m_block（VERIFY U1）
      load_cbuf_to_ca(dbase + (int64_t)i * kStep * blkElems,
                      base + (int64_t)i * blkElems,
                      (uint16_t)0, (uint8_t)kStep, (uint16_t)srcStride, (uint16_t)0,
                      (uint8_t)0, false, (__cce_scalar::addr_cal_mode_t)0);  // mm_impl.h:35 位序
    }
  }
}
template <typename T>
__aicore__ inline void asc_copy_l12l0b(__cb__ T *dst, __cbuf__ T *src, int mStart, int kStart,
                                       int mStep, int kStep, int srcStride, int dstStride) {
  // P2-B（提案 §3）：官方 L0B 非转置路 repeatTimes=blockUseN、srcStride=Ceil(bL1K,16)；
  //   转置由 L1 写侧 dn2nz 承担而非 load 侧 transpose 位（load_to_l0b_load2d.h:38-92、
  //   load_to_l0b_basic.h:111-119：cfg{startIndex=0, repeat=l0bRepeat, srcStride=l0bSrcstride,
  //   sid=0, dstGap=l0bDststride, transpose=0, addrmode=0} + 外层 for 拆趟）。
  //   本轮 §12 的 B 侧几何是 K-major 单趟形态，先按 A 式同款降级；N 向拆趟与 dstGap
  //   （=nFraC0-1）留 U3/E-4 卡定。
  (void)mStart; (void)mStep; (void)dstStride;  // VERIFY U3（§8-U3/E-4：B 侧 N 向拆趟、dstGap 真值）
  int const blkElems = 16 * (int)(32 / (int)sizeof(T));
  __cbuf__ T *base = (__cbuf__ T *)(uintptr_t)src + (int64_t)(kStart * srcStride) * blkElems;
  load_cbuf_to_cb((__cb__ T *)(uintptr_t)dst, base, (uint16_t)0, (uint8_t)kStep,
                  (uint16_t)srcStride, (uint16_t)0, (uint8_t)0, false,
                  (__cce_scalar::addr_cal_mode_t)0);  // 对称 LoadData2DL12L0BCal（mm_impl.h）
}
"""

# ── 块4/5：transpоse 两件——代码不动，只把旧 V 标记改指 U5/E-5（提案“不改项”） ──
OLD4 = "  (void)dstStride;  // VERIFY V2\n"
NEW4 = "  (void)dstStride;  // VERIFY U5（§8-U5/E-5：transpose 件为双 cfg 字打包、位序未取证——本窗原样保持）\n"
OLD5A = "  (void)dstStride;  // VERIFY V2'：与 asc_copy_l12l0b_transpose 同款取舍\n"
NEW5A = "  (void)dstStride;  // VERIFY U5（§8-U5/E-5：与 b_transpose 同款——位序未取证，本窗原样保持）\n"
OLD5B = "  load_cbuf_to_ca_transpose(dst, src, (uint16_t)mStart, (uint8_t)mStep, (uint16_t)srcStride,\n                            (uint16_t)dstStride, false, (uint16_t)0);  // VERIFY V2' 位序\n"
NEW5B = "  load_cbuf_to_ca_transpose(dst, src, (uint16_t)mStart, (uint8_t)mStep, (uint16_t)srcStride,\n                            (uint16_t)dstStride, false, (uint16_t)0);  // VERIFY U5/E-5 位序（ hivmc 名字表示双 cfg 字，非 V1 同族）\n"

# ── 块6：asc_set_gm2l1_nz_para 注释尾（V1' → 关闭口径），代码不改 ────────────
OLD6 = "// c0_stride 语义由 copy 侧固定块式搬运承担；V1' 上卡数值证真。"
NEW6 = "// c0_stride 语义已由 P1-1k 写侧自算 dstNzC0Stride 承担（提案“不改项”：保持空实现；\n// 若改用 codegen args[10] 需加形参 = 接口扩张，须 codegen 同窗，本窗禁改）。"


def cut(text, start_marker, end_marker, new_block, label):
    i = text.find(start_marker)
    assert i != -1, "锚1未命中: " + label
    assert text.find(start_marker, i + 1) == -1, "锚1多命中: " + label
    j = text.find(end_marker, i)
    assert j != -1, "锚2未命中: " + label
    j2 = text.find(end_marker, j + 1)
    assert j2 == -1 or j2 > i + len(text), "锚2多命中: " + label
    j_end = j + len(end_marker)
    return text[:i] + new_block + text[j_end:]


def once_replace(text, old, new, label):
    assert text.count(old) == 1, "唯一性失败: " + label + " count=%d" % text.count(old)
    return text.replace(old, new)


def main():
    dry = "--dry" in sys.argv
    with open(TRUNK, "r", encoding="utf-8") as f:
        t = f.read()
    before_lines = t.count("\n")
    t = cut(t, OLD1_S, OLD1_E, NEW1, "block1-头部VERIFY")
    t = cut(t, OLD2_S, OLD2_E, NEW2, "block2-gm2l1")
    t = cut(t, OLD3_S, OLD3_E, NEW3, "block3-l12l0")
    t = once_replace(t, OLD4, NEW4, "block4-btranspose注释")
    t = once_replace(t, OLD5A, NEW5A, "block5a-atranspose注释")
    t = once_replace(t, OLD5B, NEW5B, "block5b-atranspose位序注释")
    t = once_replace(t, OLD6, NEW6, "block6-nz_para注释")
    print("行数: %d -> %d" % (before_lines, t.count("\n")))
    print("#if 系: %d   #endif: %d" % (t.count("\n#if") + t.count("\n#ifndef"), t.count("\n#endif")))
    for probe in ("asc_fill_l1", "load_cbuf_to_ca(dbase", "copy_gm_to_cbuf_multi_nd2nz_b16",
                  "(uint8_t)0, false", "asc_copy_l12l0a(__ca__ T *dst, __cbuf__ T *src, int mStart"):
        print("锚点 %-45s hits=%d" % (probe, t.count(probe)))
    if dry:
        print("[dry] 锚点全部命中，未写盘")
        return
    with open(TRUNK, "w", encoding="utf-8") as f:
        f.write(t)
    with open(MIRROR, "w", encoding="utf-8") as f:
        f.write(t)
    print("md5(trunk) ", hashlib.md5(t.encode()).hexdigest())
    print("md5(mirror)同笔写入，天然一致")


if __name__ == "__main__":
    main()
