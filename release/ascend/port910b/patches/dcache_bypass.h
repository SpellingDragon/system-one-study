#pragma once

#include <stdint.h>

namespace tl {

// Scalar GM access that bypasses the per-core DCache. AscendC::{Read,Write}
// GmByPassDCache only accept unsigned integer pointers, so we bit-cast through
// the matching-width uintN_t and reinterpret on the way out. This is the
// scalar-core fallback for GM loads/stores whose addresses may not be aligned
// to the 512B DCache line.

template <typename T>
#if defined(TL_PORT910B_NATIVE_TYPES)
// 910B cce face: plain __gm__ pointer access (no _dev dcache-bypass builtins);
// the pass choosing this file already requested bypass semantics at IR level.
#define TL_GM_LOAD(ptr) (*(ptr))
#define TL_GM_STORE(ptr, val) (*(ptr) = (val))
#else
#define TL_GM_LOAD(ptr) asc_load_dev(ptr)
#define TL_GM_STORE(ptr, val) asc_store_dev(ptr, val)
#endif

__aicore__ inline T read_gm_bypass_dcache(__gm__ T *addr) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 ||
                    sizeof(T) == 8,
                "read_gm_bypass_dcache only supports 1/2/4/8-byte scalar "
                "types");
  if constexpr (sizeof(T) == 8) {
    uint64_t bits = TL_GM_LOAD(reinterpret_cast<__gm__ uint64_t *>(addr));
    return *reinterpret_cast<T *>(&bits);
  } else if constexpr (sizeof(T) == 4) {
    uint32_t bits = TL_GM_LOAD(reinterpret_cast<__gm__ uint32_t *>(addr));
    return *reinterpret_cast<T *>(&bits);
  } else if constexpr (sizeof(T) == 2) {
    uint16_t bits = TL_GM_LOAD(reinterpret_cast<__gm__ uint16_t *>(addr));
    return *reinterpret_cast<T *>(&bits);
  } else {
    uint8_t bits = TL_GM_LOAD(reinterpret_cast<__gm__ uint8_t *>(addr));
    return *reinterpret_cast<T *>(&bits);
  }
}

template <typename T>
__aicore__ inline void write_gm_bypass_dcache(__gm__ T *addr, T value) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 ||
                    sizeof(T) == 8,
                "write_gm_bypass_dcache only supports 1/2/4/8-byte scalar "
                "types");
  if constexpr (sizeof(T) == 8) {
    uint64_t bits = *reinterpret_cast<uint64_t *>(&value);
    TL_GM_STORE(reinterpret_cast<__gm__ uint64_t *>(addr), bits);
  } else if constexpr (sizeof(T) == 4) {
    uint32_t bits = *reinterpret_cast<uint32_t *>(&value);
    TL_GM_STORE(reinterpret_cast<__gm__ uint32_t *>(addr), bits);
  } else if constexpr (sizeof(T) == 2) {
    uint16_t bits = *reinterpret_cast<uint16_t *>(&value);
    TL_GM_STORE(reinterpret_cast<__gm__ uint16_t *>(addr), bits);
  } else {
    uint8_t bits = *reinterpret_cast<uint8_t *>(&value);
    TL_GM_STORE(reinterpret_cast<__gm__ uint8_t *>(addr), bits);
  }
}

} // namespace tl
