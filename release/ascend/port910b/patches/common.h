#pragma once

#include <stdio.h>

#include "acl/acl.h"
#if defined(TL_ASCEND_SIMT)
// 950-generation SIMD/SIMT surface (c_api/simt_api headers + fp8 casts).
// Absent on 910B / CANN 8.5.x: AIC-path kernels (bf16/fp32) build without
// it; -DTL_ASCEND_SIMT=1 is injected for dav-3510+ targets by the driver
// (upstream consistency work tracked in system-one p2-13 B1).
#include "c_api/asc_simd.h"
#include "simt_api/asc_bf16.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_fp8.h"
#include "simt_api/asc_simt.h"
#include "tl_templates/ascend/ascend_fp8.h"
#else
#include "tl_templates/ascend/port910b_compat.h"  // 910B: half=_Float16 / bf16=u16 storage (P0-1 B-class)
#endif
#include "tl_templates/ascend/numeric_limits.h"
#if defined(TL_ASCEND_SIMT)
#include "tl_templates/ascend/reduce.h"  // SIMT reduce 实现（910B 无对应，罩除）
#endif
#if defined(TL_ASCEND_SIMT)
#include "tl_templates/ascend/simd_inst.h"  // 950 SIMD intrinsic 宏组（128b vec/cce attr 冲突 910B clang15，见 p2-13 B1）
#endif

// Use standard unsigned types instead of macros to avoid conflicts with system
// headers
typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;

#define TILELANG_CHECK(stmt)                                                   \
  do {                                                                         \
    aclError __err = (stmt);                                                   \
    if (__err != ACL_SUCCESS) {                                                \
      snprintf(error_buf, ERROR_BUF_SIZE, "%s:%d: aclError %d", __FILE__,      \
               __LINE__, (int)__err);                                          \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define TILELANG_CHECK_LAST_ERROR(kernel_name)                                 \
  do {                                                                         \
    auto __err = aclrtGetLastError(ACL_RT_THREAD_LEVEL);                       \
    if (__err) {                                                               \
      snprintf(error_buf, ERROR_BUF_SIZE, kernel_name ": aclError %d",         \
               (int)__err);                                                    \
      return -1;                                                               \
    }                                                                          \
  } while (0)
