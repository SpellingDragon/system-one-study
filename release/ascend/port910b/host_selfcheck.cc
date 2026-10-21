// host_selfcheck.cc — port910b_compat.h 的主机侧等价性自检（**非 ascend 目标**）
//
// 【做什么】在 x86/arm 主机上编译并运行，验证兼容层的 B 类实现满足三条取证约束：
//           ① 位宽/平凡可复制（numeric_limits.h 的 TL_BIT_CAST 前提）
//           ② 数值语义（bf16/f16 的 round-to-nearest-even 与主机原生转换一致）
//           ③ codegen 发射的形状习惯（((T*)(&v))[i].x 取址-下标-成员）
// 【怎么做】硬编码已知位型表校验 bf16；用主机 `_Float16` 作 fp16 的转换神谕
//           （random sweep + 边界值）；用 memcpy 位视图复现 codegen 的 lane 访问。
// 【为什么】本机无 NPU/CANN，兼容层是唯一可离线闭环的部分（INVENTORY.md §3-B），
//           故必须在主机上把"B 类可解"这一判决从推断级提升为实证级；上卡后
//           probe910b.sh 的 P9 只是重复本文件已做的形状检查在 bisheng 前端是否成立。
//
// 编译（两种后端都要过）：
//   clang++ -std=c++17 -DTL_PORT910B_BF16_BUILTIN=0 -DTL_PORT910B_HALF_BUILTIN=0 host_selfcheck.cc && ./a.out
//   clang++ -std=c++17 -DTL_PORT910B_BF16_BUILTIN=1 -DTL_PORT910B_HALF_BUILTIN=1 host_selfcheck.cc && ./a.out
//   clang++ -std=c++17 host_selfcheck.cc && ./a.out   # 自动探测

#include "port910b_compat.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

// TL_BIT_CAST 的本地复刻：与移植对象 numeric_limits.h:8 完全同一表达式，
// 目的是证明"我们的类型能被同一个内建 constexpr 位重解释"。
#define HOST_BIT_CAST(D, x) __builtin_bit_cast(D, x)

static int g_fail = 0;

// 白话: 断言小工具，失败就打印带期望/实际的行，最后统一以非零码退出。
static void check(bool ok, const char *what, long got, long want) {
  if (!ok) {
    ++g_fail;
    std::printf("FAIL %s: got=0x%lx want=0x%lx\n", what, got, want);
  }
}

// 白话: bf16 的已知位型表（RNE 语义，位型可在任意 IEEE754 参考实现复算）。
static void test_bf16_table() {
  struct {
    const char *name;
    float f;
    uint16_t bits;
  } tbl[] = {
      {"1.0", 1.0f, 0x3F80u},       {"0.5", 0.5f, 0x3F00u},
      {"1.5", 1.5f, 0x3FC0u},       {"-2.0", -2.0f, 0xC000u},
      {"pi", 3.14159265f, 0x4049u}, {"0.1", 0.1f, 0x3DCDu},
      {"-0.0", -0.0f, 0x8000u},     {"inf", HUGE_VALF, 0x7F80u},
  };
  for (const auto &c : tbl) {
    bfloat16_t got = __float2bfloat16_rn(c.f);
    uint16_t raw = 0;
    std::memcpy(&raw, &got, sizeof(raw));
    check(raw == c.bits, c.name, raw, c.bits);
    // 反向：位型 -> 浮点必须与正向自洽（inf 除外，比较用 isinf）
    float back = __bfloat162float(got);
    if (std::isinf(c.f)) {
      check(std::isinf(back), "bf16 inf round-trip",
            static_cast<long>(back > 0), 1);
    } else {
      // 反向自洽：同一位型经 bit_cast 构造后转换值必须一致
      bfloat16_t ref = HOST_BIT_CAST(bfloat16_t, c.bits);
      check(back == __bfloat162float(ref), "bf16 reverse bits",
            static_cast<long>(back * 256), static_cast<long>(c.bits));
    }
  }
}

#if defined(__FLT16_MANT_DIG__)
// 白话: 用主机原生 _Float16（clang 保证 RNE）当神谕，随机 sweep 我们的软件转换，
//       两个后端（storage struct / _Float16 alias）都必须逐位一致。
static void test_f16_against_host_oracle() {
  std::mt19937 rng(20261007u);
  std::uniform_real_distribution<float> dist(-1000.f, 1000.f);
  int mismatch = 0;
  for (int i = 0; i < 200000; ++i) {
    float f = dist(rng);
    uint16_t want = 0;
    _Float16 h = static_cast<_Float16>(f);
    std::memcpy(&want, &h, sizeof(want));
    half_t got = half(f);
    uint16_t raw = 0;
    std::memcpy(&raw, &got, sizeof(raw));
    if (raw != want) {
      ++mismatch;
      if (mismatch <= 5) {
        std::printf("FAIL f16 sweep f=%.9g got=0x%04x want=0x%04x\n", f, raw,
                    want);
      }
    }
  }
  // 边界：subnormal、inf、NaN、溢出
  const struct {
    float f;
    uint16_t bits;
  } edge[] = {
      {0.0f, 0x0000u},    {1.0f, 0x3C00u},     {-2.0f, 0xC000u},
      {65504.0f, 0x7BFFu}, {70000.0f, 0x7C00u}, {1e-8f, 0x0000u},
      {5.96e-8f, 0x0001u},
  };
  int edge_fail = 0;
  for (const auto &e : edge) {
    uint16_t raw = 0;
    half_t h = half(e.f);
    std::memcpy(&raw, &h, sizeof(raw));
    if (raw != e.bits) {
      ++edge_fail;
      ++mismatch;
      std::printf("FAIL f16 edge f=%g got=0x%04x want=0x%04x\n", e.f, raw,
                  e.bits);
    }
  }
  std::printf("f16 sweep: 200000 samples + %d edges, mismatch=%d\n", 7,
              mismatch);
  check(mismatch == 0, "f16 vs host oracle", mismatch, 0);
  (void)edge_fail;
}
#endif

// 白话: 复现 codegen_ascend.cc:3594/3601/3630/3638 的取址-下标-成员写法，
//       确认我们的 struct 与 codegen 发射的 lane 访问形状严格兼容。
static void test_codegen_lane_shape() {
  bfloat16_t v[4];
  v[0] = bfloat16_t(1.0f);
  v[1] = bfloat16_t(2.0f);
  v[2] = bfloat16_t(3.0f);
  v[3] = bfloat16_t(4.0f);
  for (int i = 0; i < 4; ++i) {
    // 完全照抄发射形式：((bfloat16x2_t*)(&(vec)))[i/2].access[i%2]
    const bfloat16_t &lane =
        i % 2 == 0 ? ((bfloat16x2_t *)(&(v)))[i / 2].x
                   : ((bfloat16x2_t *)(&(v)))[i / 2].y;
    float got = __bfloat162float(lane);
    check(got == static_cast<float>(i + 1), "bf16 lane view",
          static_cast<long>(got), static_cast<long>(i + 1));
  }
  // 写侧同样成立（codegen 的 per-lane store 形态）
  ((bfloat16x2_t *)(&(v)))[1].x = __float2bfloat16_rn(9.0f);
  check(__bfloat162float(v[2]) == 9.0f, "bf16 lane store", 0, 0);

  half hv[4];
  hv[0] = half(1.0f);
  hv[1] = half(2.0f);
  hv[2] = half(4.0f);
  hv[3] = half(8.0f);
  for (int i = 0; i < 4; ++i) {
    float got = i % 2 == 0 ? (float)((half2 *)(&(hv)))[i / 2].x
                           : (float)((half2 *)(&(hv)))[i / 2].y;
    check(got == static_cast<float>(1 << i), "f16 lane view",
          static_cast<long>(got), static_cast<long>(1 << i));
  }

  // ushort2：vector_bool 的 >4 lane 访问形态（codegen:3601,3638）
  unsigned short b[4] = {1, 0, 1, 1};
  check(((ushort2 *)(&(b)))[1].y == 1, "ushort2 lane view", 0, 1);

  // codegen_ascend.cc:1078 的 `*(unsigned long long*)&make_float2(v, v)`
  float f = 0.5f;
  float2 tmp = make_float2(f, f); // 具名局部量（C++ 禁止对临时量取地址）
  unsigned long long bits = *(unsigned long long *)&tmp;
  unsigned long long want = 0;
  float pair[2] = {0.5f, 0.5f};
  std::memcpy(&want, pair, sizeof(want));
  check(bits == want, "make_float2 addressable", 0, 0);
}

// 白话: 复现 numeric_limits.h:19-29 的 constexpr 位重解释常量 + debug.h 的 (float)val
//       C 风格转换，两者都是 B 类判决成立的硬前提。
static void test_bit_cast_constants_and_print_cast() {
  constexpr bfloat16_t kBf16Inf = HOST_BIT_CAST(bfloat16_t, uint16_t{0x7F80u});
  constexpr bfloat16_t kBf16NInf = HOST_BIT_CAST(bfloat16_t, uint16_t{0xFF80u});
  constexpr bfloat16_t kBf16NaN = HOST_BIT_CAST(bfloat16_t, uint16_t{0x7FC0u});
  constexpr half kHalfInf = HOST_BIT_CAST(half, uint16_t{0x7C00u});
  constexpr half kHalfNaN = HOST_BIT_CAST(half, uint16_t{0x7E00u});

  check(std::isinf((float)kBf16Inf), "kBf16Inf is inf", 0, 1);
  check((float)kBf16Inf > 0, "kBf16Inf sign", 0, 1);
  check(std::isinf((float)kBf16NInf) && (float)kBf16NInf < 0, "kBf16NInf", 0, 1);
  check(std::isnan((float)kBf16NaN), "kBf16NaN", 0, 1);
  check(std::isinf((float)kHalfInf), "kHalfInf", 0, 1);
  check(std::isnan((float)kHalfNaN), "kHalfNaN", 0, 1);

  // debug.h:209,221 形态：printf("%f", (float)val) —— C 风格转换必须可用
  bfloat16_t one = bfloat16_t(1.0f);
  std::printf("  (debug-print shape) dtype=bfloat16 value=%f\n", (float)one);
}

// 白话: 校验 B3/B4 组的名字确实可用（值本身无语义，只保证 codegen 发射能编）。
static void test_lockmode_and_decl_macro() {
  int mode_block = ASC_LOCK_BLOCK;
  int mode_non = ASC_LOCK_NON_BLOCK;
  check(mode_block != mode_non, "ASC_LOCK_* distinct", 0, 1);
#ifdef __SIMT_DEVICE_FUNCTIONS_DECL__
  check(true, "__SIMT_DEVICE_FUNCTIONS_DECL__ defined", 0, 0);
#else
  check(false, "__SIMT_DEVICE_FUNCTIONS_DECL__ defined", 0, 1);
#endif
}

int main() {
  std::printf("== port910b_compat.h host self-check ==\n");
  std::printf("   backend: bf16=%d half=%d\n", TL_PORT910B_BF16_BUILTIN,
              TL_PORT910B_HALF_BUILTIN);
  test_bit_cast_constants_and_print_cast();
  test_bf16_table();
  test_codegen_lane_shape();
  test_lockmode_and_decl_macro();
#if defined(__FLT16_MANT_DIG__)
  test_f16_against_host_oracle();
#else
  std::printf("  (skip f16 oracle: no _Float16 on this host)\n");
#endif
  if (g_fail == 0) {
    std::printf("SELFCHECK PASS: 0 failure (B 类 20 符号的等价性在主机成立)\n");
    return 0;
  }
  std::printf("SELFCHECK FAIL: %d failure\n", g_fail);
  return 1;
}
