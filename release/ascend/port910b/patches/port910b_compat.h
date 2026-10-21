#pragma once

// __aicore__ 仲裁（compat 居 include 链首位，先于 debug.h 展开）：
// 910B bisheng 的 aicore 上下文语法是 `[aicore]` 限定符（CCE 函数仅此上下文注入）；
// host/950 面维持空。
#ifdef TL_PORT910B_NATIVE_TYPES
#define __aicore__ [aicore]
#else
#ifndef __aicore__
#define __aicore__
#endif
#endif

// ════════════════════════════════════════════════════════════════════════════
// port910b_compat.h — TileLang 昇腾后端 910B(dav-2201) 兼容层草稿
//
// 域 p2-13-ascend-runtime / 孙任务 P0-1 产物。只兑现 INVENTORY.md §3-B 的
// **B 类 20 个符号**（bisheng 类型系统第 5 层墙）；C 类 250 个 SIMT/SIMD 方言
// 符号不在本层解决，由 `TL_ASCEND_SIMT` 罩屏蔽；D 类 44 个符号的存在性由
// P0-2 探针（probe910b.sh）裁决。接入方式同目录 README.md。
//
// 覆盖符号（与 inventory_symbols.md 的 B 段一一对应）：
//   B1-TYPE(12) bfloat16_t bfloat16x2_t half half_t half2 float2 float4
//               make_float2 make_float4 make_int2 make_longlong4 make_ulonglong4
//   B2-CVT(5)   __bfloat162float __bfloat1622float2 __float22bfloat162_rn
//               __half22float2 __float22half2_rn   （+ __float2bfloat16_rn 供
//                codegen_ascend.cc:3355 发射，附表因词法合并未单列）
//   B3-LOCK(2)  ASC_LOCK_BLOCK ASC_LOCK_NON_BLOCK（枚举值；asc_lock/unlock 本体在 D2）
//   B4(1)       __SIMT_DEVICE_FUNCTIONS_DECL__
//
// 设计约束（举证自移植对象，全部为静态取证）：
//   1) numeric_limits.h:19-29 用 `TL_BIT_CAST(half|bfloat16_t, uint16_t{...})`
//      且声明为 `inline constexpr` → 标量类型必须 **2 字节、平凡可复制**，
//      且允许 constexpr 位重解释。
//   2) debug.h:209,213,221,227 用 C 风格 `(float)val` → 必须提供
//      `explicit operator float()`（C 风格转换可调用显式转换函数）。
//   3) codegen_ascend.cc:3594,3601,3630,3638 发射
//      `((bfloat16x2_t*)(&(v)))[i/2].x` / `((half2*)(&v))[i/2].y` /
//      `((ushort2*)(&v))[i/2].x` → 元组类型必须是 **成员为 .x/.y 的 4 字节
//      可取址 struct**（不能是 clang 向量类型，否则 `[]`+`.x` 组合语义不同）。
//   4) codegen_ascend.cc:1078 发射 `*(unsigned long long*)&make_float2(v, v)`
//      → make_* 返回的临时量必须可按地址重解释为 8 字节整数。
//   5) codegen_ascend.cc:2723-2725 注释（上游取证）："bisheng rejects them
//      outside a VF body" → bisheng **自身可能带 make_float2/make_int2 内建**。
//      故本层每个名字都可单独退让：`-DTL_PORT910B_SKIP_<name>`（见 §0.3）。
// ════════════════════════════════════════════════════════════════════════════

// ── 0. 开关与总罩 ────────────────────────────────────────────────────────────
// 0.1 本层只在 910B 面生效；950 面（TL_ASCEND_SIMT）由真头提供这些名字。
#if !defined(TL_ASCEND_SIMT)

#include <stdint.h>
#include <type_traits>
// 910B: .asc 模式 bisheng 自动注入 __clang_cce_* 头（CCE 函数/pipe_t 直接可见）。
// 插件注入的 kernel 注册段引用官方 TLV 元数据结构链——直接对齐官方宏头（比逐结构复刻稳）：
#if defined(TL_PORT910B_NATIVE_TYPES)
#include "basic_api/utils/kernel_utils_macros.h"
#endif

// 0.2 后端选择：编译器若提供原生 16 位浮点 builtin 就直接用，否则退到自带
//     storage struct（零依赖，行为等价但算术走 float 提升）。
#ifndef TL_PORT910B_BF16_BUILTIN
#if defined(__BFLT16_MANT_DIG__) || defined(__CLANG_BF16__) || defined(TL_PORT910B_NATIVE_TYPES)
#define TL_PORT910B_BF16_BUILTIN 1
#else
#define TL_PORT910B_BF16_BUILTIN 0
#endif
#endif

#ifndef TL_PORT910B_HALF_BUILTIN
#if defined(__FLT16_MANT_DIG__) || defined(TL_PORT910B_NATIVE_TYPES)
#define TL_PORT910B_HALF_BUILTIN 1
#else
#define TL_PORT910B_HALF_BUILTIN 0
#endif
#endif

// 0.3 逐名字退让：上卡遇到 "redefinition of 'float2'" 这类冲突时，
//     加 -DTL_PORT910B_SKIP_float2 即可把该名字让给 bisheng，不动本文件。
// cce (.asc) compilation: clang driver injects __clang_cce_types.h which provides
// NATIVE bfloat16_t (typedef __bf16) and half (__cce_half). Our proxies must stand down
// there, or `using`/typedef conflicts arise (verified 2026-10-08 local Docker env).
#if defined(TL_PORT910B_NATIVE_TYPES)  // passed by codegen for cce (.asc) compiles
#define TL_PORT910B_SKIP_bfloat16_t
#define TL_PORT910B_SKIP_half
#endif

#define TL910B_EMIT(name) !defined(TL_PORT910B_SKIP_##name)

namespace tl910b {

// ── 1. 位工具 ────────────────────────────────────────────────────────────────
// bit_cast 走 __builtin_bit_cast（与 numeric_limits.h:8 的 TL_BIT_CAST 同源，
// B1 已证实 bisheng clang15 无 std::bit_cast 但有该内建）。
template <typename D, typename S> inline constexpr D tl910b_bit_cast(const S &x) {
  static_assert(sizeof(D) == sizeof(S), "tl910b_bit_cast requires equal size");
  return __builtin_bit_cast(D, x);
}

// bf16 <-> f32：截尾 + round-to-nearest-even（`_rn` 语义），NaN 保持 quiet。
inline constexpr uint16_t tl910b_f32_to_bf16_bits(float f) {
  const uint32_t u = tl910b_bit_cast<uint32_t, float>(f);
  if (((u >> 23) & 0xffu) == 0xffu && (u & 0x7fffffu) != 0u) {
    return static_cast<uint16_t>((u >> 16) | 0x40u); // NaN -> quiet NaN
  }
  const uint32_t lsb = (u >> 16) & 1u;
  const uint32_t rounded = u + 0x7fffu + lsb; // RNE
  return static_cast<uint16_t>(rounded >> 16);
}

inline constexpr float tl910b_bf16_bits_to_f32(uint16_t b) {
  return tl910b_bit_cast<float, uint32_t>(static_cast<uint32_t>(b) << 16);
}

// fp16 <-> f32：软件路径只在 TL_PORT910B_HALF_BUILTIN=0 时用于转换内建；
// 与 IEEE 754 binary16 一致（含 subnormal 与 inf/NaN）。
inline constexpr float tl910b_f16_bits_to_f32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1fu;
  const uint32_t man = h & 0x3ffu;
  if (exp == 0u) {
    if (man == 0u) {
      return tl910b_bit_cast<float, uint32_t>(sign); // +-0
    }
    uint32_t e = 0x7fu + 1u; // 规格化 subnormal
    uint32_t m = man << (23 - 10);
    while ((m & (1u << 23)) == 0u) {
      m <<= 1;
      --e;
    }
    m &= (1u << 23) - 1u;
    return tl910b_bit_cast<float, uint32_t>(sign | (e << 23) | m);
  }
  if (exp == 0x1fu) {
    return tl910b_bit_cast<float, uint32_t>(
        sign | 0x7f800000u | (man << (23 - 10))); // inf / NaN
  }
  const uint32_t e = exp + (0x7fu - 0x0fu);
  return tl910b_bit_cast<float, uint32_t>(sign | (e << 23) | (man << (23 - 10)));
}

inline constexpr uint16_t tl910b_f32_to_f16_bits(float f) {
  const uint32_t u = tl910b_bit_cast<uint32_t, float>(f);
  const uint32_t sign = (u >> 16) & 0x8000u;
  const int32_t expf = static_cast<int32_t>((u >> 23) & 0xffu);
  const uint32_t man = u & 0x7fffffu;
  if (expf == 0xff) { // inf / NaN（NaN 一律归为 quiet）
    return static_cast<uint16_t>(sign | 0x7c00u | (man ? 0x200u : 0u));
  }
  const int32_t e = expf - 127 + 15; // fp16 有偏指数（全部用有符号比较！）
  if (e >= 31) { // 指数 31 为 inf/NaN 保留 -> 溢出（65520 以上，含恰好 2^16）
    return static_cast<uint16_t>(sign | 0x7c00u);
  }
  if (e >= 1) { // 规格数：丢 13 bit，RNE = +(0x0fff + 保留最低位)
    const uint32_t lsb = (man >> 13) & 1u;
    const uint32_t rounded = (man + 0x0fffu + lsb) >> 13;
    if (rounded == 0x400u) { // 舍入进位到下一指数；e=30 时正好落到 inf(0x7C00)
      return static_cast<uint16_t>(sign | (static_cast<uint32_t>(e + 1) << 10));
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) |
                                 rounded);
  }
  if (e < -10) { // 连最小次正规 2^-24 都够不着 -> 带符号 0
    return static_cast<uint16_t>(sign);
  }
  const uint32_t m = man | 0x800000u; // 补回隐含 1，得 24 bit 有效数
  const uint32_t shift = static_cast<uint32_t>(14 - e);
  const uint32_t lsb = (m >> shift) & 1u;
  const uint32_t rounded = (m + ((1u << (shift - 1)) - 1u) + lsb) >> shift;
  return static_cast<uint16_t>(sign | rounded);
}

// ── 2. B1-TYPE：标量类型 ─────────────────────────────────────────────────────
#if TL910B_EMIT(bfloat16_t)
#if TL_PORT910B_BF16_BUILTIN
  // builtin 路径：`__bf16` 是 2 字节标量浮点类型，直接 typedef 即满足
  // TL_BIT_CAST / (float) / 指针运算；(float) 与算术由 clang 负责提升。
  typedef __bf16 bfloat16_t;
#else
  // storage 路径：自带 POD，行为自洽（数值面走 float 提升）。
  struct bfloat16_t {
    uint16_t data;

    bfloat16_t() = default;
    constexpr explicit bfloat16_t(uint16_t bits) : data(bits) {}
    constexpr bfloat16_t(float v)
        : data(tl910b_f32_to_bf16_bits(v)) {} // codegen 常以 float 初始化

    explicit constexpr operator float() const {
      return tl910b_bf16_bits_to_f32(data);
    }
    constexpr explicit operator double() const {
      return static_cast<double>(static_cast<float>(*this));
    }
    constexpr explicit operator bool() const {
      return (data & 0x7fffu) != 0u;
    }
  };
  // 比较与算术：全部转 float 计算再回 bf16（910B 上标量核本就无 bf16 原生 ALU）。
  constexpr bfloat16_t operator+(bfloat16_t a, bfloat16_t b) {
    return bfloat16_t(static_cast<float>(a) + static_cast<float>(b));
  }
  constexpr bfloat16_t operator-(bfloat16_t a, bfloat16_t b) {
    return bfloat16_t(static_cast<float>(a) - static_cast<float>(b));
  }
  constexpr bfloat16_t operator*(bfloat16_t a, bfloat16_t b) {
    return bfloat16_t(static_cast<float>(a) * static_cast<float>(b));
  }
  constexpr bfloat16_t operator/(bfloat16_t a, bfloat16_t b) {
    return bfloat16_t(static_cast<float>(a) / static_cast<float>(b));
  }
  constexpr bool operator==(bfloat16_t a, bfloat16_t b) {
    return a.data == b.data;
  }
  constexpr bool operator!=(bfloat16_t a, bfloat16_t b) {
    return a.data != b.data;
  }
  constexpr bool operator<(bfloat16_t a, bfloat16_t b) {
    return static_cast<float>(a) < static_cast<float>(b);
  }
  constexpr bool operator>(bfloat16_t a, bfloat16_t b) {
    return static_cast<float>(a) > static_cast<float>(b);
  }
  constexpr bool operator<=(bfloat16_t a, bfloat16_t b) {
    return static_cast<float>(a) <= static_cast<float>(b);
  }
  constexpr bool operator>=(bfloat16_t a, bfloat16_t b) {
    return static_cast<float>(a) >= static_cast<float>(b);
  }
  // codegen_ascend.cc:3419 等以 `bfloat16_t` 作模板实参，需要可默认构造。
  struct bfloat16x2_t;
#endif
#endif // TL910B_EMIT(bfloat16_t)

#if TL910B_EMIT(half)
#if TL_PORT910B_HALF_BUILTIN
  typedef _Float16 half;
#else
  struct half {
    uint16_t data;

    half() = default;
    constexpr explicit half(uint16_t bits) : data(bits) {}
    constexpr half(float v) : data(tl910b_f32_to_f16_bits(v)) {}

    explicit constexpr operator float() const {
      return tl910b_f16_bits_to_f32(data);
    }
    constexpr explicit operator double() const {
      return static_cast<double>(static_cast<float>(*this));
    }
    constexpr explicit operator bool() const { return (data & 0x7fffu) != 0u; }
  };
  constexpr half operator+(half a, half b) {
    return half(static_cast<float>(a) + static_cast<float>(b));
  }
  constexpr half operator-(half a, half b) {
    return half(static_cast<float>(a) - static_cast<float>(b));
  }
  constexpr half operator*(half a, half b) {
    return half(static_cast<float>(a) * static_cast<float>(b));
  }
  constexpr half operator/(half a, half b) {
    return half(static_cast<float>(a) / static_cast<float>(b));
  }
  constexpr bool operator==(half a, half b) { return a.data == b.data; }
  constexpr bool operator!=(half a, half b) { return a.data != b.data; }
  constexpr bool operator<(half a, half b) {
    return static_cast<float>(a) < static_cast<float>(b);
  }
  constexpr bool operator>(half a, half b) {
    return static_cast<float>(a) > static_cast<float>(b);
  }
  constexpr bool operator<=(half a, half b) {
    return static_cast<float>(a) <= static_cast<float>(b);
  }
  constexpr bool operator>=(half a, half b) {
    return static_cast<float>(a) >= static_cast<float>(b);
  }
#endif
#endif // TL910B_EMIT(half)

#if TL910B_EMIT(half_t)
  // gemm.h:36 文档里以 half_t 指称 fp16 元素类型；与 tilelang CUDA 面同名习惯一致。
  typedef half half_t;
#endif

// ── 3. B1-TYPE：2 元组（.x/.y 可取址，供 ((T*)(&v))[i].x 用法） ──────────────
#if TL910B_EMIT(bfloat16x2_t)
  // 两个后端下形状一致：4 字节、.x/.y 成员、可 ((bfloat16x2_t*)&v)[i] 下标。
  struct bfloat16x2_t {
    bfloat16_t x, y;
  };
#endif

#if TL910B_EMIT(half2)
  struct half2 {
    half x, y;
  };
#endif

#if TL910B_EMIT(float2)
  struct float2 {
    float x, y;
  };
#endif

#if TL910B_EMIT(float4)
  struct float4 {
    float x, y, z, w;
  };
#endif

#if TL910B_EMIT(int2)
  struct int2 {
    int x, y;
  };
#endif

#if TL910B_EMIT(ushort2)
  // codegen_ascend.cc:3601,3638 用 `(ushort2*)(&v))[i/2].x` 访问 >4 lane 的
  // vector_bool；common.h:27 的 ushort typedef 在本头之后才出现，故此处用原生长度。
  struct ushort2 {
    unsigned short x, y;
  };
#endif

#if TL910B_EMIT(longlong4)
  struct longlong4 {
    long long x, y, z, w;
  };
#endif

#if TL910B_EMIT(ulonglong4)
  struct ulonglong4 {
    unsigned long long x, y, z, w;
  };
#endif

// ── 4. B1-TYPE：make_* 构造器（聚合初始化，避免向量 builtin 依赖） ───────────
#if TL910B_EMIT(make_float2)
  constexpr float2 make_float2(float x, float y) { return float2{x, y}; }
#endif
#if TL910B_EMIT(make_float4)
  constexpr float4 make_float4(float x, float y, float z, float w) {
    return float4{x, y, z, w};
  }
#endif
#if TL910B_EMIT(make_int2)
  constexpr int2 make_int2(int x, int y) { return int2{x, y}; }
#endif
#if TL910B_EMIT(make_longlong4)
  constexpr longlong4 make_longlong4(long long x, long long y, long long z,
                                     long long w) {
    return longlong4{x, y, z, w};
  }
#endif
#if TL910B_EMIT(make_ulonglong4)
  constexpr ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y,
                                       unsigned long long z,
                                       unsigned long long w) {
    return ulonglong4{x, y, z, w};
  }
#endif

// ── 5. B2-CVT：转换内建（发射点 codegen_ascend.cc:3333,3355,3403,3411,3419,3428） ─
// 语义与 CUDA/AscendC 同名件对齐：`_rn` = round-to-nearest-even；成对版拆两次标量。
#if TL910B_EMIT(__bfloat162float)
  constexpr float __bfloat162float(const bfloat16_t &x) {
#if defined(TL_PORT910B_NATIVE_TYPES)
    // bisheng BFloat16 proxy 无转换算子：位级取出后软件还原（bf16 是 f32 高位截断）
    return tl910b_bf16_bits_to_f32(tl910b_bit_cast<uint16_t, bfloat16_t>(x));
#elif TL_PORT910B_BF16_BUILTIN
    return static_cast<float>(x);
#else
    return tl910b_bf16_bits_to_f32(x.data);
#endif
  }
#endif

#if TL910B_EMIT(__float2bfloat16_rn)
  constexpr bfloat16_t __float2bfloat16_rn(float x) {
#if TL_PORT910B_BF16_BUILTIN || defined(TL_PORT910B_NATIVE_TYPES)
    // builtin 下无 RNE 保证（实现定义），显式走位路径再位重解释。
    return tl910b_bit_cast<bfloat16_t, uint16_t>(tl910b_f32_to_bf16_bits(x));
#else
    return bfloat16_t(tl910b_f32_to_bf16_bits(x));
#endif
  }
#endif

#if TL910B_EMIT(__bfloat1622float2)
  constexpr float2 __bfloat1622float2(const bfloat16x2_t &x) {
#if defined(TL_PORT910B_NATIVE_TYPES)
    // proxy 对类型无 .x/.y：按 2x16bit 位拆包各自还原
    uint32_t bits = tl910b_bit_cast<uint32_t, bfloat16x2_t>(x);
    return float2{tl910b_bf16_bits_to_f32(static_cast<uint16_t>(bits & 0xFFFFu)),
                  tl910b_bf16_bits_to_f32(static_cast<uint16_t>(bits >> 16))};
#else
    return float2{__bfloat162float(x.x), __bfloat162float(x.y)};
#endif
  }
#endif

#if TL910B_EMIT(__float22bfloat162_rn)
  constexpr bfloat16x2_t __float22bfloat162_rn(const float2 &x) {
#if defined(TL_PORT910B_NATIVE_TYPES)
    uint32_t lo = tl910b_f32_to_bf16_bits(x.x), hi = tl910b_f32_to_bf16_bits(x.y);
    return tl910b_bit_cast<bfloat16x2_t, uint32_t>(lo | (hi << 16));
#else
    return bfloat16x2_t{__float2bfloat16_rn(x.x), __float2bfloat16_rn(x.y)};
#endif
  }
#endif

#if TL910B_EMIT(__half22float2)
  constexpr float2 __half22float2(half2 x) {
    return float2{static_cast<float>(x.x), static_cast<float>(x.y)};
  }
#endif

#if TL910B_EMIT(__float22half2_rn)
  constexpr half2 __float22half2_rn(float2 x) {
    return half2{half(x.x), half(x.y)};
  }
#endif

} // namespace tl910b

// ── 6. 全局名字注入（逐名字可退让；`using` 而非宏，保持重载/模板能力） ───────
#define TL910B_USING(name) using tl910b::name
#if TL910B_EMIT(bfloat16_t)
TL910B_USING(bfloat16_t);
#endif
#if TL910B_EMIT(half)
TL910B_USING(half);
#endif
#if TL910B_EMIT(half_t)
TL910B_USING(half_t);
#endif
#if TL910B_EMIT(bfloat16x2_t)
TL910B_USING(bfloat16x2_t);
#endif
#if TL910B_EMIT(half2)
TL910B_USING(half2);
#endif
#if TL910B_EMIT(float2)
TL910B_USING(float2);
#endif
#if TL910B_EMIT(float4)
TL910B_USING(float4);
#endif
#if TL910B_EMIT(int2)
TL910B_USING(int2);
#endif
#if TL910B_EMIT(ushort2)
TL910B_USING(ushort2);
#endif
#if TL910B_EMIT(longlong4)
TL910B_USING(longlong4);
#endif
#if TL910B_EMIT(ulonglong4)
TL910B_USING(ulonglong4);
#endif
#if TL910B_EMIT(make_float2)
TL910B_USING(make_float2);
#endif
#if TL910B_EMIT(make_float4)
TL910B_USING(make_float4);
#endif
#if TL910B_EMIT(make_int2)
TL910B_USING(make_int2);
#endif
#if TL910B_EMIT(make_longlong4)
TL910B_USING(make_longlong4);
#endif
#if TL910B_EMIT(make_ulonglong4)
TL910B_USING(make_ulonglong4);
#endif
#if TL910B_EMIT(__bfloat162float)
TL910B_USING(__bfloat162float);
#endif
#if TL910B_EMIT(__float2bfloat16_rn)
TL910B_USING(__float2bfloat16_rn);
#endif
#if TL910B_EMIT(__bfloat1622float2)
TL910B_USING(__bfloat1622float2);
#endif
#if TL910B_EMIT(__float22bfloat162_rn)
TL910B_USING(__float22bfloat162_rn);
#endif
#if TL910B_EMIT(__half22float2)
TL910B_USING(__half22float2);
#endif
#if TL910B_EMIT(__float22half2_rn)
TL910B_USING(__float22half2_rn);
#endif

// ── 7. B3-LOCKMODE：asc_lock/asc_unlock 的模式实参 ───────────────────────────
// 发射点 codegen_ascend.cc:1855-1857,1869-1871（`asc_lock(PIPE_X, id, ASC_LOCK_BLOCK)`）
// 与 rewrite_flag_to_buf.cc:20-23。注意：本组只解决"枚举值可编译"，
// `asc_lock/asc_unlock` 本体在 910B 是否存在属 D2-AIC-API（探针 P3）。
#ifndef ASC_LOCK_BLOCK
#define ASC_LOCK_BLOCK 0
#endif
#ifndef ASC_LOCK_NON_BLOCK
#define ASC_LOCK_NON_BLOCK 1
#endif

// ── 8. B4：__SIMT_DEVICE_FUNCTIONS_DECL__（ascend_fp8.h:9 的 TL_DEVICE 前缀） ─
// 950 上它是 SIMT device 函数声明限定；910B 面无 SIMT，空展开即可。
#ifndef __SIMT_DEVICE_FUNCTIONS_DECL__
#define __SIMT_DEVICE_FUNCTIONS_DECL__
#endif

// ── 8b. debug 打印的 float 化辅助（proxy 类型无 (float) 算子）─────────────────
inline float tl910b_debug_as_float(const bfloat16_t &v) {
  return tl910b::tl910b_bf16_bits_to_f32(tl910b::tl910b_bit_cast<uint16_t, bfloat16_t>(v));
}
template <typename T> inline float tl910b_debug_as_float(T v) { return static_cast<float>(v); }

// ── 10. AIC 适配器：tilelang codegen 裸 emitted 符号 → 910B CCE 原生（P1-1 首件）
#if defined(TL_PORT910B_NATIVE_TYPES)
// codegen 生成 `extern "C" __global__ __mix__(1,2) ...` 与 `if ASC_IS_AIC {...} else {...}`
// 910B cce 原生即有 __mix__(cube,vec)=core_ratio 属性宏（w9/w10 实证）——让位，不覆盖。
// 仅 host/无 cce 环境自检时补一个空定义保 TU 可析。
#ifndef __mix__
#define __mix__(a, b)
#endif
#define ASC_IS_AIC (true)                  // 一期 AIC 优先；AIV 分支保留可编（死代码消除前）

// pipe 事件常量：与 codegen_ascend 的 PIPE_* 对齐即可（值仅传参用）
enum tl910b_pipe_id { TL910B_PIPE_M = 0, TL910B_PIPE_MTE1, TL910B_PIPE_MTE2,
                      TL910B_PIPE_MTE3, TL910B_PIPE_V, TL910B_PIPE_S, TL910B_PIPE_ICACHE };

__aicore__ inline void asc_init() {}                       // 910B mad 无显式初始化
// FFTS 调度表基址：与官方同名对齐；框架链在场（ASCENDC_MODULE_UTILS_H）时让位
#ifndef ASCENDC_MODULE_UTILS_H
__gm__ void *g_sysFftsAddr = nullptr;
#endif
// （TLV 枚举 FuncMetaType/KernelType/BinaryMetaType 与注册结构由官方 kernel_utils_macros.h 提供）


#define TL910B_ASC_MMAD_OVERLOAD(TY)                                          \
  __aicore__ inline void asc_mmad(__ubuf__ float *cc, __ubuf__ TY *ca,             \
                                  __ubuf__ TY *cb, uint16_t m, uint16_t k,        \
                                  uint16_t n, int unit_flag_ctrl, bool gemv_ctrl, \
                                  bool btbuf_ctrl, bool zero_c_ctrl) {            \
    /* 910B mad(cc, BTAddr, ca, cb, M, K, N, unit_flag, gemv, BTbuf, zero_C)；    \
       地址空间指针经整数中转（C-style 整→空间指针 CCE 允许）*/                   \
    mad((__cc__ float *)(uintptr_t)cc, (__ca__ TY *)(uintptr_t)ca,                 \
        (__cb__ TY *)(uintptr_t)cb, m, k, n, (uint8_t)unit_flag_ctrl,               \
        gemv_ctrl, btbuf_ctrl, zero_c_ctrl);                                       \
  }
TL910B_ASC_MMAD_OVERLOAD(bfloat16_t)
TL910B_ASC_MMAD_OVERLOAD(half)
TL910B_ASC_MMAD_OVERLOAD(float)
// (int8 形态 910B mad 走 fp8 家族参数，非一期范围，暂缓)

// 跨 pipe 同步：一期保守全序列化（正确性优先，后续换 SetFlag/WaitFlag 精细化）
__aicore__ inline void asc_sync_intra_wait(int pipe, int event) { (void)pipe; (void)event; __cce_pipe_barrier(PIPE_V); }
__aicore__ inline void asc_sync_intra_set(int pipe, int event)  { (void)pipe; (void)event; }
__aicore__ inline void asc_lock(int pipe, int id)   { (void)pipe; (void)id; __cce_pipe_barrier(PIPE_V); }
__aicore__ inline void asc_lock(int pipe, int id, int mode) { (void)mode; asc_lock(pipe, id); }
__aicore__ inline void asc_unlock(int pipe, int id, int mode) { (void)pipe; (void)id; (void)mode; }

// ── 10b. AIV/索引/搬运面（一期 if(0) 分支符号可解析；数值待 host launch 绑定后精化）
#define ASC_IS_AIV (0)
struct tl910b_dim3 { uint32_t x, y, z; };
__aicore__ inline tl910b_dim3 tl910b_blockIdx_holder() { uint32_t i = get_block_idx(); return tl910b_dim3{0, i, 0}; }
#define blockIdx tl910b_blockIdx_holder()
__aicore__ inline int32_t asc_get_sub_block_id() { return (int32_t)get_subblockid(); }

enum asc_load_l2_cache_mode { ASC_LOAD_L2_CACHE_ALLOC = 0, ASC_LOAD_L2_CACHE_NORMAL = 1 };
// ── 10b-GAPB（P1-1 执行代理 B 于 cann910b-b 实测补齐；证据见 attempts/B/compat_gap_B.md）
#ifndef TL910B_GAPB_APPLIED
#define TL910B_GAPB_APPLIED

// G3: store 侧 L2 cache 模式枚举。发射点 codegen_ascend.cc:1430 以
// `static_cast<asc_store_l2_cache_mode>(n)` 传值（n=4 来自 copy.cc:316 kStoreDefaultL2CacheCtrl），
// 910B 的 GM<-UB 搬运原生件无 L2 ctrl 形参，本值在此层被丢弃。
enum asc_store_l2_cache_mode : unsigned char {  // merged B(G3)+C(G-C2): fixed underlying
  // type required for static_cast<...>(4); 2201 native copy drops the hint anyway
  ASC_STORE_L2_CACHE_ALLOC = 0,
  ASC_STORE_L2_CACHE_NO_ALLOC = 1,
  ASC_STORE_L2_CACHE_FREE = 2,
  ASC_STORE_L2_CACHE_NORMAL = 4,  // kStoreDefaultL2CacheCtrl @ src/ascend/op/copy.cc:316
};

// G1: GM<->UB 搬运的 910B 正确形态（7 参原生件，长度/步长以 32B 为单位）。
// codegen 侧实发形态（src/ascend/codegen/codegen_ascend.cc:1405-1433 + src/ascend/op/copy.cc:300-328）：
//   asc_copy_gm2ub_align(dst, src, n_rows, row_bytes, 0, 0, 0, load_mode, src_stride_bytes, dst_stride_bytes)
//   asc_copy_ub2gm_align(dst, src, burst_num, burst_len, store_mode, burst_dst_stride, burst_src_stride)  // 7 参!
// 910B 原生件（asc/impl/basic_api/dav_c220/kernel_operator_data_copy_impl.h:51/124/221）：
//   copy_gm_to_ubuf(__ubuf__ void*, __gm__ void*, sid, blockCount, blockLen, srcStride, dstStride)
//   copy_ubuf_to_gm(__gm__  void*, __ubuf__ void*, sid, blockCount, blockLen, srcStride, dstStride)
//   —— 官方 :221 传 `tensorSize*sizeof(T)/32` 作 blockLen，故单位=32B；名字见
//      cce_aicore_intrinsics.h:992/1042（namespace __cce_scalar，无 arch guard）。
#define TL910B_BYTES_TO_BLK(x) ((uint16_t)(((int)(x)) / 32))

__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ uint8_t *dst, __gm__ uint8_t *src,
                                            int blockCount, int blockLenB, int rsvd0,
                                            int rsvd1 /*right_pad*/, int rsvd2,
                                            asc_load_l2_cache_mode mode, int srcStrideB,
                                            int dstStrideB) {
  (void)rsvd0; (void)rsvd1; (void)rsvd2; (void)mode;
  __cce_scalar::copy_gm_to_ubuf((__ubuf__ void *)dst, (__gm__ void *)src, (uint8_t)0,
                                (uint16_t)blockCount, TL910B_BYTES_TO_BLK(blockLenB),
                                TL910B_BYTES_TO_BLK(srcStrideB), TL910B_BYTES_TO_BLK(dstStrideB));
}

// (B 版 ub2gm 已由 C 版取代：同 codegen 契约、注释与 32B 单位换算更严谨 @ orchestration)

// G2: 跨 pipe 事件对。发射点：T.copy(gm->ub) 后 `asc_sync_notify(PIPE_MTE2, PIPE_S,
// static_cast<event_t>(0))`，首个消费点 asc_sync_wait(...)；ub->gm 前为 (PIPE_S, PIPE_MTE3)
// （见 gen_vec.asc / gen_ub2gm.asc）。910B 依据：AscendC dav_c220 官方搬运即
//   SetFlag<HardEvent::MTE2_S>/S_MTE3 成对使用（kernel_operator_data_copy_impl.h:167-168,
//   kernel_event.h:57 MTE2_S）；toolchain 自身样板 __clang_cce_aicore_functions.h:2152-2159。
// **必须用宏、不能用 inline 函数**：__cce_scalar::set_flag/wait_flag 是
//   `__attribute__((clang_builtin_alias(...))) void set_flag(...)`（cce_aicore_intrinsics.h:2072/2742），
//   CCE 前端校验要求 pipe 实参为字面枚举常量；实测（attempts/B/probe_setflag.sh）：
//     V1 字面量直调        COMPILE-OK
//     V2 inline 函数转调   COMPILE-FAIL "the 1st parameter maybe need a type 'pipe_t'"
//     V3/V4 宏（带/不带括号）COMPILE-OK
//     V5 static_cast<event_t>(3) COMPILE-OK（event 可非 0）
//   这也解释了 toolchain 的 __cce_set_flag(:2796) 为何 (void)p;(void)tp 丢弃实参、硬编码
//   PIPE_M->PIPE_V —— 它无法转发变量形 pipe 实参。我们靠宏保住真实的 MTE2->S / S->MTE3 语义。
#ifndef asc_sync_notify
#define asc_sync_notify(from, to, evt) __cce_scalar::set_flag(from, to, evt)
#endif
#ifndef asc_sync_wait
#define asc_sync_wait(from, to, evt) __cce_scalar::wait_flag(from, to, evt)
#endif

// G4: 910B(dav-2201) **标量面没有 rsqrtf**。一符号一 TU 实测（attempts/B/probe_math_symbols.sh）：
//     sqrt(float)=OK；rsqrtf / sqrtf / expf / exp / logf / log / powf / fabs / floorf /
//     ceilf / tanhf / __nv_rsqrtf / __nv_expf 全 MISSING；__builtin_sqrt FAIL（aicore 函数禁 double）。
// tilelang 的 T.rsqrt 在 vector 面落码为标量 `rsqrtf(float)`（见 gen_rsqrt.asc），故补最小 shim。
// 数值口径：1/sqrt(x)，走已验证的标量 sqrt；**不等价于 RCRS/RSQRT 近似指令**，
//   编译面等价、上卡后与 AscendC 原生 rsqrt 可能有 ~ulp 级差异（RESULT.md 已申报）。
__aicore__ inline float rsqrtf(float x) { return 1.0f / sqrt(x); }

#endif // TL910B_GAPB_APPLIED
__aicore__ inline void asc_sync_intra_arrive(int pipe, int event) { (void)pipe; (void)event; }
#endif // TL_PORT910B_NATIVE_TYPES (AIC adapters)

#if !defined(TL_PORT910B_NATIVE_TYPES)  // 自检仅对自带类型有意义；NATIVE 下类型由 cce 原生保证

// ── 9. 自检（P9 探针即"编一个 TU"，这些断言在编译期即给出等价性真值） ────────
#if defined(TL_PORT910B_NATIVE_TYPES)
static_assert(sizeof(bfloat16_t) == 2, "native bf16 must be 2 bytes");
static_assert(sizeof(half) == 2, "native half must be 2 bytes");
#else
static_assert(sizeof(tl910b::bfloat16_t) == 2, "bf16 must be 2 bytes for TL_BIT_CAST");
static_assert(sizeof(tl910b::half) == 2, "fp16 must be 2 bytes for TL_BIT_CAST");
#endif
static_assert(sizeof(tl910b::bfloat16x2_t) == 4, "bf16x2 must be 4 bytes");
static_assert(sizeof(tl910b::half2) == 4, "half2 must be 4 bytes");
static_assert(sizeof(tl910b::float2) == 8, "float2 must be 8 bytes");
static_assert(sizeof(tl910b::float4) == 16, "float4 must be 16 bytes");
static_assert(sizeof(tl910b::ushort2) == 4, "ushort2 must be 4 bytes");
static_assert(sizeof(tl910b::ulonglong4) == 32, "ulonglong4 must be 32 bytes");
static_assert(__is_trivially_copyable(tl910b::bfloat16_t), "must be trivially copyable");
static_assert(__is_trivially_copyable(tl910b::bfloat16x2_t), "must be trivially copyable");

// 位视图与 codegen 的 `((T*)(&v))[i].x` 习惯等价：4 个 bf16 lane 装进 ulonglong4 的低 8 字节
static_assert(tl910b::tl910b_f32_to_bf16_bits(1.0f) == 0x3F80u, "bf16(1.0)");
static_assert(tl910b::tl910b_bf16_bits_to_f32(0x3F80u) == 1.0f, "f32 from bf16");
static_assert(tl910b::tl910b_f32_to_f16_bits(1.0f) == 0x3C00u, "fp16(1.0)");
static_assert(tl910b::tl910b_f16_bits_to_f32(0x3C00u) == 1.0f, "f32 from fp16");
static_assert(tl910b::tl910b_f32_to_f16_bits(-2.0f) == 0xC000u, "fp16(-2.0)");
static_assert(tl910b::tl910b_f32_to_bf16_bits(0.5f) == 0x3F00u, "bf16(0.5)");

#endif // !TL_PORT910B_NATIVE_TYPES (self-check)
#endif // !TL_ASCEND_SIMT

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
// (enum merged into GAP-B block at orchestration; identical values)

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

// ════════════════════════════════════════════════════════════════════════════
// §12 shared.l1 gemm path — 910B backend (P1-1 A案). Anchor: TL910B_SEC12_L1GEMM
// codegen `tl.ascend_gemm_l1` lowering（alloc_l1/alloc_l0c + transpose_B）发射面：
//   asc_set_gm2l1_nz_para / asc_copy_gm2l1_nd2nz / ascend_gemm_l1<...>(模板体)
//   asc_copy_l12l0a/b(_transpose) / asc_mmad / asc_lock/unlock / asc_copy_l0c2gm
// 建材（全部 raw 指针，无需 TPipe/LocalTensor；a11/a16/a17 实测签名）：
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
// ════════════════════════════════════════════════════════════════════════════
#ifndef TL_ASCEND_SIMT
#ifdef TL_PORT910B_NATIVE_TYPES
#ifndef TL910B_SEC12_L1GEMM
#define TL910B_SEC12_L1GEMM

// ── D2 判别器（p2-13 / P1-1n）：E-1..E-4 最小判别实验的编译期可选变体 ────────────
// 三个**互相正交**的开关，默认全 OFF ⇒ 本文件展开后的 token 流与 Z 落地态等价；
// 实证入口：attempts/D2_probe_matrix.py（pristine f56432b0 vs 本文件，同 TU 同参数编 .o
// → sha256 必须相同）+ attempts/D2_probe_legs.py（真生成码 gemm_l1_intrinsic /
// dw_intrinsic 的 device_kernel.asc 与 kernel.aibin 同基线）。
// 注入面：-DTL910B_VARIANT_x（tilelang 侧唯一入口 = contrib/bisheng.py 的 result 行与
//   tilelang_callback_ascend_compile 的 options；窗内配方见 attempts/D2_variants.md §2）。
//   -DTL910B_VARIANT_A  写侧 GM→L1 退回 8 参 **ND 线性**拷贝（P1 之前）→ 答 E-1/裁决②
//   -DTL910B_VARIANT_B  读侧 sid 槽重新吃 IR m_start（P0 之前）        → 答 E-3/裁决③
//   -DTL910B_VARIANT_C  读侧退单趟 repeatTimes=mStep + 不折指针（P2 之前）→ 答 E-2/E-4/U2/U3
// ⚠ 三个开关都是**判别载荷，不是修法**：开启后可能拿静默错值也可能拿非法访问——这正是判据。
// ⚠ 覆盖关系：`asc_copy_l12l0a/b_transpose`（§8-U5，Z 未改该件）既无 sid 槽、也从未走过
//   P2 的折指针/拆趟 ⇒ **B、C 对只发 _transpose 件的腿（dw_intrinsic，其生成码里读侧只有
//   asc_copy_l12l0a_transpose/l12l0b_transpose）无效**，该腿只对 A 有反应；勿读成"变体失效"。
// sid 槽取值（VARIANT-B 的唯一改动点；默认展开 = `(uint8_t)0`，与 Z 文本逐 token 同）
// ── VARIANT-B ══ 读侧第 7 槽 sid 回归控制（E-3；X_FRACTAL_EVIDENCE §7.2 表第 7 行 +
//   §2.2 S1/S2（sid=4-bit，sema 对**编译期常量**查 [0,15]）+ §2.4 RT3（**运行期值不查**，
//   rc=0 → 越界只能真机暴露 = 507015 头号嫌疑的成因）。旧形（P0 之前）把 codegen
//   args[2]＝IR m_start（row16 块起点，0..127 量级，copy.cc:457/535-538）回穿进 sid 槽。
//   开它回答 **E-3 二选一**：507015 是不是 sid 越界触发的。
// ⚠ 本窗实测（attempts/D2_probe_matrix.py 表② -D B 行 + D2_probe_legs.py [+B vs default]）：
//     sid 槽在本窗**两条真腿上都拿不到非零值** —— gemm_l1_intrinsic 走 gemm.h:68/74 的
//     mStart **字面 0**；dw_intrinsic 读侧只发 `_transpose` 件（该件无 sid 槽，Z 未改）。
//     ⇒ B 在这两腿 .o/.so 指纹 SAME = **空开关**，E-3 对这两腿**零卡即可结案（否）**；
//     B 的判别力只存在于 mStart≠0 的载荷（矩阵 M3/M5 已证 DIFF）——若日后 codegen 直发
//     tl.ascend_load_cbuf_to_ca 且 m_start 为运行期表达式，本开关即刻可用，无需再改 §12。
//     `-DTL910B_VARIANT_B_SID=<n>`（默认 0/未定义）是把 sid 钉成**编译期常量**的加强支：
//     n>15 时 bisheng **编译期即拦**（本窗实测错误原文
//     `error: the range of 7th parameter must be [0, 15]`，报在 `asc_copy_l12l0a`/`l12l0b`
//     里 `load_cbuf_to_ca`/`load_cbuf_to_cb` 的第 7 实参行）⇒ 它只能作 sema 面的旁证，
//     **不能**当真机载荷用（行号随本文件改动漂移，故不给死行号；见 D2_variants.md §6.2）。
#if defined(TL910B_VARIANT_B)
#ifdef TL910B_VARIANT_B_SID  // B 的强制值支：sid 给一个**编译期已知越界**的常量（可选）
#define TL910B_L1_SID(mStart) (uint8_t)(TL910B_VARIANT_B_SID)
#else
#define TL910B_L1_SID(mStart) (uint8_t)(mStart)  // 回归控制：4-bit sid 槽重新吃 IR m_start
#endif
#else
#define TL910B_L1_SID(mStart) (uint8_t)0  // 默认（P0 止血）：sid≡0（§5.2 规则 4）
#endif

// b7 探针 5/5 USABLE（compat 链内，无框架 include）：load_cbuf_to_ca、load_gm_to_ca、
// mad(官方 10 参)、fix_matrix_cc_to_cbufubuf_dualout、copy_gm_to_cbuf —— 全 native 组装。
// codegen 发射的 mode 枚举（存在性面，值域保守）
enum asc_unit_flag_mode : unsigned char { ASC_UF_MODE_DEFAULT = 0 };
enum asc_quant_mode : unsigned char { ASC_QUANT_MODE_DEFAULT = 0 };
enum asc_relu_pre_mode : unsigned char { ASC_RELU_MODE_OFF = 0, ASC_RELU_MODE_ON = 1 };
#ifndef ASC_LOCK_BLOCK
#define ASC_LOCK_BLOCK 0
#endif

// NZ 参数：CCE 禁可变成体全局（a2k3 实证）→ 一期 set 空实现，
// c0_stride 语义已由 P1-1k 写侧自算 dstNzC0Stride 承担（提案“不改项”：保持空实现；
// 若改用 codegen args[10] 需加形参 = 接口扩张，须 codegen 同窗，本窗禁改）。
__aicore__ inline void asc_set_gm2l1_nz_para(int en, int rchg, uint16_t c0_stride, int pad) {
  (void)en; (void)rchg; (void)c0_stride; (void)pad;
}
__aicore__ inline void asc_set_l0c_copy_nz_para(int en, int a, int b) { (void)en; (void)a; (void)b; }
__aicore__ inline void asc_set_copy_pad_val(int v) { (void)v; }

// GM→L1（P1-1k 落提案 §2）：910B 官方 Matmul 写 L1 只有 ND2NZ/NZ2NZ 两路
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
#if defined(TL910B_VARIANT_A)
  // ══ VARIANT-A ══ 写侧退回 8 参 ND 线性拷贝（逐字取自 X_sec12_proposal.md §2 的 "-" 侧
  //   = 官方 DataCopyGM2L1Impl，kernel_operator_data_copy_impl.h:83-108）。行号证据：
  //   X_FRACTAL_EVIDENCE §2.5 D7（8 参形在 2201 **能编** ⇒ 不是编译期病灶）+ §6 证据
  //   1/2/4（startIndex 文档语义=Fractal matrix ID ⇒ L1 必须是 NZ；官方 Matmul 写 L1
  //   只有 ND2NZ/NZ2NZ 两路；旧形丢 d_value、nz_para 空实现 ⇒ c0 pitch 从未生成）。
  //   开它回答：**E-1 / 裁决②（ND vs NZ 布局是不是主因）**——默认(NZ) 与 A(ND) 各一趟，
  //   `gemm_l1_intrinsic` 红绿翻转即坐实布局主因；同色则布局被排除，主嫌转读侧（C）。
  (void)mode; (void)cols; (void)pad1; (void)pad2;
  uint16_t const blk = (uint16_t)((unsigned)rowBytes >> 5);  // 32B 块口径（§5.1 定论）
  // 官方 8 参（data_copy_impl.h:92）：sid,blockCount,blockLen,srcStride,dstStride,pad
  copy_gm_to_cbuf((__cbuf__ void *)(uintptr_t)dst, (__gm__ void *)(uintptr_t)src, (int8_t)0,
                  (uint16_t)nRows, blk, blk, blk, (pad_t)0);
#else
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
#endif  // TL910B_VARIANT_A（写侧；dn2nz 委托同一开关覆盖）
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

// P1-1d（2026-10-10）遗留结论：官方 LoadData2DL12L0ACal（mm_impl.h:32/35）发 **9 参**
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
#if defined(TL910B_VARIANT_C)
  // ══ VARIANT-C ══ 读侧退单条指令 + 不折指针（逐字取自 X_sec12_proposal.md §3 的 "-"
  //   侧，即 P2 之前的 trunk）。逐槽错位（X_FRACTAL_EVIDENCE §7.2 表）——
  //   〔3 startIndex〕←kStart（官方恒 0，二维起点要折进指针）、〔4 repeatTimes〕←mStep
  //   （轴选反，应=kStep；u8 还会静默截断 m_step>255，§2.3 E2）、〔6 dstGap〕←dstStride
  //   （U3 未定论）、(void)kStep（丢掉官方唯一合法的 repeat 值）、**无 M 向拆趟**。
  //   开它回答：**E-2/E-4（U1 块线性序 / U2 L0 目的步进 / U3 dstGap 真值）**。
  (void)kStep;
  load_cbuf_to_ca(dst, src, (uint16_t)kStart, (uint8_t)mStep, (uint16_t)srcStride,
                  (uint16_t)dstStride, TL910B_L1_SID(mStart), false,
                  (__cce_scalar::addr_cal_mode_t)0);  // 官方 9 参位序（mm_impl.h:35）
#else
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
                    TL910B_L1_SID(mStart), false, (__cce_scalar::addr_cal_mode_t)0);
  } else {  // repeatTimes=u8：kStep>255 静默截断（§2.3 E2 实证类型）——本窗不设 ICHECK，
            // 910B tile 的 C0 组数远低于 255，越界属真机 E-2 判别面
    for (int i = 0; i < mStep; ++i) {  // 块线性序 block_id = k_block*pitch + m_block（VERIFY U1）
      load_cbuf_to_ca(dbase + (int64_t)i * kStep * blkElems,
                      base + (int64_t)i * blkElems,
                      (uint16_t)0, (uint8_t)kStep, (uint16_t)srcStride, (uint16_t)0,
                      TL910B_L1_SID(mStart), false, (__cce_scalar::addr_cal_mode_t)0);  // mm_impl.h:35 位序
    }
  }
#endif  // TL910B_VARIANT_C（读侧单趟/不折指针回退）
}
template <typename T>
__aicore__ inline void asc_copy_l12l0b(__cb__ T *dst, __cbuf__ T *src, int mStart, int kStart,
                                       int mStep, int kStep, int srcStride, int dstStride) {
#if defined(TL910B_VARIANT_C)
  // ══ VARIANT-C ══ 同 L12L0A：退单条指令、repeatTimes←mStep、startIndex←kStart（不折
  //   指针）、dstGap←dstStride、(void)kStep。行号证据同上（§7.2 逐槽表，B 侧对称）。
  //   开它回答 E-2/E-4 的 **B 侧**（U3：官方 load_to_l0b_basic.h:112 的 dstGap=
  //   l0bDststride=nFraC0-1 而非 0，与 L0A 路不同）。
  (void)kStep;
  load_cbuf_to_cb(dst, src, (uint16_t)kStart, (uint8_t)mStep, (uint16_t)srcStride,
                  (uint16_t)dstStride, TL910B_L1_SID(mStart), false,
                  (__cce_scalar::addr_cal_mode_t)0);  // 对称 LoadData2DL12L0BCal
#else
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
                  (uint16_t)srcStride, (uint16_t)0, TL910B_L1_SID(mStart), false,
                  (__cce_scalar::addr_cal_mode_t)0);  // 对称 LoadData2DL12L0BCal（mm_impl.h）
#endif  // TL910B_VARIANT_C（读侧单趟/不折指针回退）
}
template <typename T>
__aicore__ inline void asc_copy_l12l0b_transpose(__cb__ T *dst, __cbuf__ T *src, int mStart,
                                                 int kStart, int mStep, int kStep, int srcStride,
                                                 int dstStride) {
  (void)kStart; (void)kStep;
  (void)dstStride;  // VERIFY U5（§8-U5/E-5：transpose 件为双 cfg 字打包、位序未取证——本窗原样保持）
  load_cbuf_to_cb_transpose(dst, src, (uint16_t)mStart, (uint8_t)mStep, (uint16_t)srcStride,
                            (uint16_t)dstStride, false, (uint16_t)0);  // D1b 8 参（mm_impl.h:152）
}

// cube 计算：910B 官方 Cal（MmadParams(m,n,k,unitFlag,cmatrixSource,cmatrixInitVal)）
template <typename CT, typename AT, typename BT>
__aicore__ inline void asc_mmad(CT *cc, AT *ca, BT *cb, uint16_t m, uint16_t k, uint16_t n,
                                int unit_flag_ctrl, bool gemv_ctrl, bool btbuf_ctrl,
                                bool zero_c_ctrl) {
  (void)gemv_ctrl; (void)btbuf_ctrl;
  // 官方形（b7 实证）：mad(c,a,b,m,k,n,unitFlag,kDirAlign,cmatrixSource,cmatrixInitVal)
  mad(cc, ca, cb, m, k, n, (uint8_t)unit_flag_ctrl, false, false, zero_c_ctrl);
}

// L0C→GM：fix 不支持 32bit 直 GM（native static_assert 实证）→ L0C→UB→GM 两段流水
// 发射形态：asc_copy_l0c2gm(gm, cc, mRows?, nCols?, m?, n?, store_mode, uf, quant, relu, a,b,c,d)
__aicore__ inline void asc_copy_l0c2gm(__gm__ float *dst, __cc__ float *src, int mRows,
                                      int nCols, int srcM, int srcN,
                                      asc_store_l2_cache_mode smode, asc_unit_flag_mode uf,
                                      asc_quant_mode q, asc_relu_pre_mode r, int deqLow,
                                      int nz2nd, int chSplit, int d) {
  (void)srcM; (void)srcN; (void)smode; (void)q; (void)r; (void)deqLow; (void)chSplit; (void)d;
  // VERIFY V3：16 行分块（L0C 行粒度=16），行内 nCols 连续
  // A-4 破局（fixpipe_v2_impl.h:433 官方 Cal 形 + cce_aicore_intrinsics.h:1020
  // 变长 alias builtin）：L0C→GM 直出件 = copy_matrix_cc_to_gm(12 参)。
  // 官方位序：(dst, src, rsvd0, nSize, mSize, dstStride, srcStride, unitFlag,
  //           quant, relu, isChannelSplit, nz2ndEn)
  // VERIFY V3：stride 单位（元素 vs 32B 块）与 codegen 14 参的位映射上卡定。
  (void)srcM; (void)srcN; (void)smode; (void)q; (void)r;
  copy_matrix_cc_to_gm(dst, src, (uint8_t)0, (uint16_t)nCols, (uint16_t)mRows,
                       (uint16_t)nCols, (uint16_t)nCols, (uint8_t)uf,
                       QuantMode_t::NoQuant, static_cast<uint8_t>(0), false,
                       nz2nd /*arg12=1 ROW_MAJOR*/);
}

#endif  // TL910B_SEC12_L1GEMM
#endif  // TL_PORT910B_NATIVE_TYPES
#endif  // !TL_ASCEND_SIMT (§12)

// ════════════════════════════════════════════════════════════════════════════
// §12 补件：asc_fill_l1 — 910B 后端（p2-13 甲路 P1-4b）. Anchor: TL910B_SEC12_FILL_L1
//
// 挡路事实（前波 K 定位，本波复证）：codegen 在每个 asc_copy_gm2l1_nd2nz 之后对
//   未填满的 L1 尾区发射（codegen_ascend.cc:1534-1555 EmitFillL1_，7 参 IR op）：
//     asc_fill_l1((__cbuf__ uint16_t|uint32_t*)((__cbuf__ uint8_t*)base + byte_off),
//                 (uint32_t)value, { .repeat=…, .blk_num=…, .dst_gap=… });
//   §12 无此件 → dW(l0tr)/lora 报 `use of undeclared identifier 'asc_fill_l1'`。
//
// 语义（ascend/op/fill.cc:99-116 LowerL1Fill + ascend/op/builtin.h:230-241）：
//   · blk_num / dst_gap / dst_pitch_blocks 一律以 **32B 块**计（byte_offset=
//     (c0_start*pitch + row_start)*32 亦据此）；repeat = c0_range.extent / C0（列组数）。
//   · 地址递推：趟 i 起点 = byte_offset + i*(blk_num+dst_gap)*32 字节，每趟写 blk_num 块。
//   · value **恒为 0**（fill.cc:87 ICHECK CanProveEqual(fill_value, 0)，原文
//     "Ascend L1 fill currently supports only zero"）→ 位型落在哪个字段数值上都等价。
//   · fill_word_bits ∈ {16,32} → 发射类型 uint16_t/uint32_t 恰与官方
//     SET_L1_2D.b16 / SET_L1_2D.b32 两条编码一一对应（无需手工选重载）。
//
// 后端与 config 位打包的**权威出处**（均可在容器内复跑）：
//   (1) CCE intrinsic：cce_aicore_intrinsics_3101.h:4233-4263（#3982-#3991）
//         void set_l1_2d(__cbuf__ T *dst, int64_t config);
//         ASM: SET_L1_2D.b16|.b32 [dst],config   Pipe: PIPE_MTE2
//         .b16 ← half/bfloat16_t/int16_t/uint16_t；.b32 ← float/int32_t/uint32_t
//       （cce_aicore_intrinsics.h:2162 / _3101.h:830 另有同 builtin 的变长 alias 声明）
//   (2) AscendC 薄封装：asc/impl/basic_api/dav_c220/kernel_operator_cube_others_impl.h:579
//         template <typename T> SetL12D(__cbuf__ T *dst, int64_t config){set_l1_2d(...);}
//   (3) **位段唯一实证**＝官方 set_l1_2d 的 HiVM 实现体。取自
//       tools/bisheng_compiler/.../bishengir/bin/hivmc 内嵌的 dav-c220-cube bitcode
//       （按 `BC\xc0\xde` 魔数切块 → llvm-link → bisheng -x ir -S -emit-llvm 反汇编），
//       _mlir_ciface_set_l1_2d_{int8_t,half,float,bfloat16_t} 四条 dtype 路形态一致：
//         store i64 1, ... %repeat_time                      ; repeat = 1
//         %blk = udiv i64 (size*sizeof(T) + 31), 32          ; blk_num = ceil(字节/32)
//         %cfg = or i64 (shl i64 %blk, 16), %repeat_time     ; config = (blk_num<<16)|repeat
//         call void @llvm.hivm.CREATE.CBUF.MATRIX.{v2.u16|f16.h|f32.h|bf16.ori}(
//              ptr addrspace(2) %dst, i64 %cfg, <16bit %value>) ; !asan.cce.api.name=set_l1_2d
//       ⇒ 实证：**cfg[15:0]=repeat_time，cfg[31:16]=blk_num（32B 块数）**；value 在 HiVM
//          侧是**独立操作数**（16 位位型；float 路先 fptrunc→half，int8 路字节 4 路复制）。
//
// 本件三处不确定各标一处 VERIFY（数值域待上卡；本波只判"编得出 + 不回归"）：
//   V4 value 位段：HiVM 有独立 value 操作数，而 CCE 形只 (dst,config) 两参 ⇒ 位型必须
//      并入 cfg 高 32 位 [63:32]。**此步系推断**（hivm→llvm 降级取不到 asm 位表）。
//      现网 value 恒 0 ⇒ [63:32] 无论解释成何字段都是 0，数值无害；若日后放开非 0 填充，
//      必须上卡重定该位段。
//   V5 dst_gap：官方 2D-set 的 cfg **没有 gap 字段**（带 src_gap<<32|dst_gap<<48 的是同族
//      别的 packed-cfg，见 cce_aicore_intrinsics_3101.h:1067-1068/2147-2148）⇒ 本件**拆趟**：
//      每趟只发一条 "repeat=1 + blk_num=blk_num" 的官方实证形指令，地址手推
//      (blk_num+dst_gap)*32 字节。代价＝指令数从 1 变 repeat（纯性能面，语义不损）。
//   V6 repeat 步距：官方以 repeat=1 铺满连续 blk_num 块，说明 repeat>1 时趟间距即
//      blk_num（gap 语义只能靠拆趟表达）；故 gap=0 时拆趟与单条 repeat=r 等价。待卡证真。
// ════════════════════════════════════════════════════════════════════════════
#ifndef TL_ASCEND_SIMT
#ifdef TL_PORT910B_NATIVE_TYPES
#ifndef TL910B_SEC12_FILL_L1
#define TL910B_SEC12_FILL_L1

// codegen 以 designated-init 聚合体传第 3 参（-std=c++20，实证见 contrib/bisheng.py:111）
// ⇒ 成员声明序**必须**与 .repeat/.blk_num/.dst_gap 同序，否则 C++20 拒绝。
struct asc_fill_l1_params {
  uint64_t repeat;   // 趟数（C0 列组数）
  uint64_t blk_num;  // 每趟写的 32B 块数
  uint64_t dst_gap;  // 相邻趟跳过的 32B 块数
};

// SET_L1_2D 的 config 打包（位段依据见块头 (3)：[15:0]=repeat、[31:16]=blk_num）。
// 16 位截断在此安全：L1=1MB ⇒ blk_num<=32768，repeat 为 C0 组数（远小于 2^16）。
__aicore__ inline int64_t tl910b_fill_l1_cfg(uint32_t value, uint64_t blk_num, uint64_t repeat) {
  uint64_t const v = static_cast<uint64_t>(value);
  return static_cast<int64_t>(((v & 0xffffffffULL) << 32)     // VERIFY V4：位段系推断（value 恒 0）
                              | ((blk_num & 0xffffULL) << 16)  // 实证：ceil(bytes/32) 块数
                              | (repeat & 0xffffULL));         // 实证：趟数
}

// dst 元素类型由 codegen 按 fill_word_bits 给定（uint16_t→SET_L1_2D.b16 /
// uint32_t→SET_L1_2D.b32，官方 #3988/#3990），故直接透传给 dtype 重载即可。
// __cbuf__ 不可标量解引用（K 波实测 "only __ubuf__/__gm__/local can be dereferenced"）
// ⇒ 唯一可行路就是这条 DMA 填塞件；地址推进沿用 §12 的 (uintptr_t) 中转手法。
template <typename T>
__aicore__ inline void asc_fill_l1(__cbuf__ T *dst, uint32_t value, asc_fill_l1_params p) {
  if (p.repeat == 0 || p.blk_num == 0) return;  // 空尾区不发射（codegen 可能给 0）
  __cbuf__ uint8_t *const raw = (__cbuf__ uint8_t *)(uintptr_t)dst;
  uint64_t const step_bytes = (p.blk_num + p.dst_gap) * 32ULL;  // 32B 块单位（见块头语义）
  int64_t const cfg = tl910b_fill_l1_cfg(value, p.blk_num, 1);  // 每趟 repeat=1：官方实证形
  for (uint64_t i = 0; i < p.repeat; ++i) {                     // VERIFY V5/V6：拆趟代 gap
    set_l1_2d((__cbuf__ T *)(uintptr_t)(raw + i * step_bytes), cfg);
  }
}

#endif  // TL910B_SEC12_FILL_L1
#endif  // TL_PORT910B_NATIVE_TYPES
#endif  // !TL_ASCEND_SIMT (§12 asc_fill_l1)

#ifndef TL_PORT910B_COMPAT_GAP_D_H
#define TL_PORT910B_COMPAT_GAP_D_H

#ifndef TL_ASCEND_SIMT  // 与 §12 同域：仅 910B native 面（非 SIMT）装配
#ifdef TL_PORT910B_NATIVE_TYPES

template <typename T>
__aicore__ inline void asc_copy_l12l0a_transpose(__ca__ T *dst, __cbuf__ T *src, int mStart,
                                                 int kStart, int mStep, int kStep, int srcStride,
                                                 int dstStride) {
  (void)kStart; (void)kStep;
  (void)dstStride;  // VERIFY U5（§8-U5/E-5：与 b_transpose 同款——位序未取证，本窗原样保持）
  load_cbuf_to_ca_transpose(dst, src, (uint16_t)mStart, (uint8_t)mStep, (uint16_t)srcStride,
                            (uint16_t)dstStride, false, (uint16_t)0);  // VERIFY U5/E-5 位序（ hivmc 名字表示双 cfg 字，非 V1 同族）
}

#endif  // TL_PORT910B_NATIVE_TYPES
#endif  // !TL_ASCEND_SIMT
#endif  // TL_PORT910B_COMPAT_GAP_D_H

