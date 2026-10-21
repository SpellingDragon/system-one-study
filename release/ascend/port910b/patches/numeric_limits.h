#pragma once

#include <bit>
#include <cstdint>

// 910B compat (system-one p2-13 B1): bisheng's C++20 stdlib lacks std::bit_cast;
// union-based constexpr reinterpretation is behavior-equivalent here.
#define TL_BIT_CAST(D, x) __builtin_bit_cast(D, x)  // clang intrinsic, the very impl std::bit_cast delegates to

namespace tl {
namespace limits {

// --- float (fp32) ---
inline constexpr float kFloatInf = TL_BIT_CAST(float, 0x7F800000u);
inline constexpr float kFloatNInf = TL_BIT_CAST(float, 0xFF800000u);
inline constexpr float kFloatNaN = TL_BIT_CAST(float, 0x7FC00000u);

// --- half (fp16) ---
inline constexpr half kHalfInf = TL_BIT_CAST(half, uint16_t{0x7C00u});
inline constexpr half kHalfNInf = TL_BIT_CAST(half, uint16_t{0xFC00u});
inline constexpr half kHalfNaN = TL_BIT_CAST(half, uint16_t{0x7E00u});

// --- bfloat16 ---
inline constexpr bfloat16_t kBf16Inf =
    TL_BIT_CAST(bfloat16_t, uint16_t{0x7F80u});
inline constexpr bfloat16_t kBf16NInf =
    TL_BIT_CAST(bfloat16_t, uint16_t{0xFF80u});
inline constexpr bfloat16_t kBf16NaN =
    TL_BIT_CAST(bfloat16_t, uint16_t{0x7FC0u});

} // namespace limits
} // namespace tl
