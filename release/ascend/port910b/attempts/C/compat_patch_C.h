// ════════════════════════════════════════════════════════════════════════════
// attempts/C/compat_patch_C.h — GAP-C：910B 标量数学面（p2-13 甲路 P1-1 波 C）
//
// 真源应落在 /tilelang/src/tl_templates/ascend/port910b_compat.h 的 §11；本波硬
// 边界禁改主仓，故以「附加块」形式由 attempts/C/env_setup.sh [2b] 注入容器 pip 树
// 副本（追加在文件尾，自带罩，不动 §0-§10 的任何行）。
//
// 依据（可复跑）：attempts/C/probe_math.py @ cann910b-c（CANN 8.5.0 / bisheng
// clang 15.0.5 / dav-2201 / target_format=aibin，与 tilelang codegen 同一条编译通路）
//   · 未声明（编译期即 error）：expf sqrtf rsqrtf fabsf floorf logf log2f tanhf powf exp10f
//   · 编得过但链不上（ld.lld: undefined symbol）：__builtin_expf __builtin_exp2f
//     __builtin_sqrtf __builtin_floorf __builtin_ceilf __builtin_log2f
//   · 原生可用：sqrt（重载含 float）、abs、__builtin_fabsf（纯指令，无 libcall）、四则
//   · #include <math.h> → 把 host libstdc++ 拖进来，stl_iterator.h 直接崩，禁路
//   ⇒ codegen_ascend 的 math 发射面（intrin_rule_ascend.cc: AscendMath 对 float32
//     一律 name+'f'）在 910B **整条不存在**，凡带超越函数的件（GDN SiLU、softmax、
//     RMSNorm 的 rsqrt、delta-rule 的 log/decay）都必须走 compat。
//   · 另证（GAP-C 自身踩坑）：compat §1 的 tl910b_bit_cast 是**无 __aicore__ 限定**的
//     `inline constexpr` 模板，从 __aicore__ 函数里调用会被 CCE 的重载集判成
//     "no matching function for call to 'tl910b_bit_cast'"（且插件把真实诊断吞掉，
//     只留 Fatal Err）。所以本块内部的位重解释一律就地写在 __aicore__ 函数里，
//     不调 §1/§5 的任何 helper。→ 已登记为缺口 G-C0（compat 侧缺一个 __aicore__ 版
//     bit_cast 入口）。
//
// 退让：单个名字还给上游 → -DTL_PORT910B_SKIP_expf / -DTL_PORT910B_SKIP_fabsf；
//       整块关掉 → 不注入本文件即可（§0-§10 不依赖这里的任何名字）。
// ════════════════════════════════════════════════════════════════════════════
#ifndef TL_ASCEND_SIMT
#ifdef TL_PORT910B_NATIVE_TYPES
#ifndef TL_PORT910B_COMPAT_GAP_C_H
#define TL_PORT910B_COMPAT_GAP_C_H

// float <-> uint32 位视图（就地、__aicore__，不借 compat §1 模板）
__aicore__ inline unsigned int tl910b_gap_f2u(float x) {
  float q = x;
  return *reinterpret_cast<unsigned int *>(&q);
}
__aicore__ inline float tl910b_gap_u2f(unsigned int bits) {
  unsigned int u = bits;
  return *reinterpret_cast<float *>(&u);
}

// expf：910B 标量核无 SFU 入口 → 范围规约 + 6 阶多项式 + 指数域拼 2^n。
//   x = n*ln2 + r（n=round(x*log2e)，|r| <= ln2/2），e^r 用 6 阶 Taylor，
//   e^x = e^r * 2^n。|r|<=ln2/2 时尾项 ~1.2e-7（相对），与 fp32 eps 同量级；
//   e^r 可能落到 [0.707,1) → 左移一次规格化到 [1,2) 后直接加指数域。
//   边界：x<=-104 → 0（sigmoid 语义 exp(-z)→0 正确）；x>=88 → FLT_MAX（不产 inf）。
__aicore__ inline float tl910b_expf(float x) {
  if (x <= -104.0f) return 0.0f;
  if (x >= 88.0f) return 3.4028234663852886e38f;
  const float kLog2e = 1.44269504088896340736f;
  const float kLn2 = 0.69314718055994530942f;
  const float t = x * kLog2e;
  int n = static_cast<int>(t >= 0.0f ? t + 0.5f : t - 0.5f);
  float r = x - static_cast<float>(n) * kLn2;
  float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666666666666666f +
              r * (0.041666666666666664f + r * (0.008333333333333333f +
              r * 0.001388888888888889f)))));
  if (p < 1.0f) {
    p += p;  // *2
    n -= 1;
  }
  // 指数域拼接：p∈[1,2) 时 f2u(p) 的指数域**已是 127**，所以只能再叠 n（不是 n+127！）。
  // 注：本波无卡，这个双重加 127 的错是由 c_gdn_golden.py 的逐行复刻抓到的
  // （exp(0) 会算成 1.7e38、exp(1) 直接 nan）→ 两边必须同步维护。
  if (n < -126) return 0.0f;
  if (n > 127) return 3.4028234663852886e38f;
  const unsigned int mant = tl910b_gap_f2u(p);
  return (n < 0)
      ? tl910b_gap_u2f(mant - (static_cast<unsigned int>(-n) << 23))
      : tl910b_gap_u2f(mant + (static_cast<unsigned int>(n) << 23));
}

// fabsf：让位给 clang 纯指令内建（探针实证 __builtin_fabsf PASS，无 libcall）。
__aicore__ inline float tl910b_fabsf(float x) { return __builtin_fabsf(x); }

// codegen 发射的是无修饰全局名，用函数式宏接走（宏而非重载：上游若哪天补了真
// 声明也不会撞「同名不同种类符号」）。
#ifndef TL_PORT910B_SKIP_expf
#define expf(x) tl910b_expf(x)
#endif
#ifndef TL_PORT910B_SKIP_fabsf
#define fabsf(x) tl910b_fabsf(x)
#endif

#endif  // TL_PORT910B_COMPAT_GAP_C_H
#endif  // TL_PORT910B_NATIVE_TYPES
#endif  // TL_ASCEND_SIMT

// ════════════════════════════════════════════════════════════════════════════
// GAP-C 第二块：UB<->GM 搬运面（compat §10b 的 store 侧补齐）— p2-13 甲路 P1-1 波 C
// （自带守卫，可独立追加在 §0-§10 之后；不动原有任何一行）
//
// 依据（可复跑）：attempts/C/probe_dma.py @ cann910b-c（dav-2201 / aibin 全链）
//   PASS: copy_gm_to_ubuf(dst,src,uint64 cfg) / copy_gm_to_ubuf(dst,src,sid,
//         burst_num,burst_len,srcStride,dstStride) / copy_ubuf_to_gm 两形态 /
//         asc_copy_gm2ub_align(10 参) / asc_copy_ub2gm_align(10 参)
//   FAIL: copy_data / copy_data_align64（910B AIV 面无此重载）;
//         asc_store_l2_cache_mode（unknown type name）;
//         asc_copy_ub2gm_align 的 **7 参** 形态（与 10 参声明 arity 不符）
//
// 事实核对（主仓，只读）：
//   * codegen_ascend.cc:1420 EmitUbufToGmCopy_ 只发 7 参：
//       asc_copy_ub2gm_align(dst, src, burst_num, burst_len,
//                           static_cast<asc_store_l2_cache_mode>(ctl),
//                           burst_dst_stride, burst_src_stride)
//     （IR op 是 8 参，arg2=sid 被 codegen 主动丢弃）
//   * port910b_compat.h:571 只定义了 asc_load_l2_cache_mode；store 侧枚举**从未
//     被定义过**，§10b 的 ub2gm 适配器按 10 参写 → codegen 的 7 参发射体从未可编。
//   ⇒ 两个纯 compat 缺口：G-C2（缺 asc_store_l2_cache_mode 枚举）、
//      G-C3（ub2gm arity 与 codegen 发射不符）。本块补齐，重载与 §10b 原行共存。
//
// 910B 原生选型：走 **7 参形态**（AscendC 自己在 dav_c220/
// kernel_operator_data_copy_impl.h 的 DataCopyUB2GMImpl 里就是这么调的），
// 而非 §10b 那种 packed-config 三参（950 风格，本探针也证可编）。
// l2_cache_ctl：910B 该原生形态不吃此参数 → 与 §10b 一致地忽略（性能提示，非正确性）。
// ════════════════════════════════════════════════════════════════════════════
#ifndef TL_ASCEND_SIMT
#ifdef TL_PORT910B_NATIVE_TYPES
#ifndef TL_PORT910B_COMPAT_GAP_C_DMA_H
#define TL_PORT910B_COMPAT_GAP_C_DMA_H

// G-C2：codegen 硬引用该枚举（值域 [0,15]，见 ascend/op/copy.cc L2CacheCtrlOr）。
// 固定底层类型：C++11 起有 fixed underlying type 时枚举可取底层类型全域，
// 故 static_cast<asc_store_l2_cache_mode>(4) 合法。
enum asc_store_l2_cache_mode : unsigned char {
  ASC_STORE_L2_CACHE_ALLOC = 0,
  ASC_STORE_L2_CACHE_NO_ALLOC = 1,
  ASC_STORE_L2_CACHE_NORMAL = 4
};

// G-C3：与 codegen 发射体逐参数对齐的 7 参重载。
// 语义映射（照抄 AscendC dav_c220 DataCopyUB2GMImpl 的实参顺序）：
//   sid=0（codegen 不发 sid）、blockCount=burst_num、blockLen=burst_len、
//   srcStride=burst_src_stride、dstStride=burst_dst_stride。
// 单位：已按上述证据做 bytes->32B-blocks 换算（G-C4 已定性，仍需上卡数值证真）。
__aicore__ inline void asc_copy_ub2gm_align(__gm__ uint8_t *dst, __ubuf__ uint8_t *src,
                                            int burst_num, int burst_len,
                                            asc_store_l2_cache_mode mode,
                                            int burst_dst_stride,
                                            int burst_src_stride) {
  (void)mode;  // 910B 原生 7 参形态不吃 l2_cache_ctl
  // 单位换算：形参是**字节**（codegen/IR 约定，见 copy.cc AscendMTEBytesFromElements），
  // 910B 原生 blockLen/stride 以 ONE_BLK_SIZE=32B 为单位（证据：asc/include/adv_api/
  // matmul/matmul_client.h:2745 即 blockLen = N * sizeof(T) / ONE_BLK_SIZE）。
  // 同一件事在 AscendC dav_c220 DataCopyUB2GMImpl 里也是直接透传 params 的块单位。
  // 不换算 = 64 倍过搬。非 32B 倍数的尾段需 align_b8/b16/b32 变体 → G-C6。
  copy_ubuf_to_gm(dst, src, (int8_t)0, (uint16_t)burst_num,
                  (uint16_t)(burst_len >> 5), (uint16_t)(burst_src_stride >> 5),
                  (uint16_t)(burst_dst_stride >> 5));
}

#endif  // TL_PORT910B_COMPAT_GAP_C_DMA_H
#endif  // TL_PORT910B_NATIVE_TYPES
#endif  // TL_ASCEND_SIMT
