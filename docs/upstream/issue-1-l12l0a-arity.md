# [Ascend] dav-2201 L1→L0A/B load is a 9-argument V1 family — a shorter `load_cbuf_to_ca` arity mis-maps `sid`/`transpose` and surfaces only as `aicore` exception 507015

> Draft for `tile-ai/tilelang`. **Not yet filed.** Status: root-cause finding + self-contained
> compile/sema repro; on-card confirmation pending (see *What we have NOT determined*).
> All line numbers below were opened and re-read by the author against CANN 8.5.0 inside
> `cann910b-k` on 2026-10-11; the sema results in *Minimal repro* were re-run by the author
> (not transcribed).

## Target / HW

- Ascend 910B (dav-2201), aarch64
- CANN 8.5.0 (docker image `cann910b`) and CANN 8.5.2 (ModelArts instance image)
- bisheng clang 15.0.5 — `.../ccec_compiler/bin/bisheng`, `clang version 15.0.5`, `Target: aarch64-unknown-linux-gnu`

## Summary

Bringing the TileLang Ascend DSL path (`target="ascend"`) to 910B requires lowering
`tl.ascend_load_cbuf_to_ca` / `tl.ascend_load_cbuf_to_cb` (the 950-generation L1→L0A/L0B load).
On dav-2201 there is **no 2DV2/explicit-form overload**: the only accepted lowering is the
**V1 `LoadData2D` family, 9 arguments**, i.e.

```
load_cbuf_to_ca(dst, src, startIndex:u16, repeatTimes:u8, srcStride:u16, dstGap:u16, sid, transpose, addr_cal_mode)
```

`sid` is a **4-bit** SMMU-L1 partition id, `transpose` **must be a compile-time constant**, and
the tail `addr_cal_mode` is `__cce_scalar::addr_cal_mode_t`.

Two traps make this painful to discover from outside:

1. **A shorter arity does not fail as a "wrong arity" error.** With 8 arguments the front-end
   resolves a *different, non-existent-on-2201* overload and reports
   `... does not support the given target feature` — the error text names the 3101 explicit form,
   which sends you chasing the wrong arch. Our first port emitted a **7-argument** call
   (documented in our own reconciliation notes `C3_closure_audit.md` §3 and
   `X_sec12_proposal.md` §3; the pre-fix source text is no longer on disk in that form) — the
   tail slots (`sid`, `transpose`, `addr_cal_mode`) were therefore mis-populated, leaving the
   L1→L0 load descriptor field-misaligned.
2. **`sid` is only range-checked when it is a compile-time constant.** A runtime `sid` value
   `> 15` compiles with zero diagnostics (verified below), so `m_start` (a row16 block index that
   can reach 0..127) silently lands in the 4-bit `sid` slot and produces an illegal L1 partition
   number. This is exactly the shape of an `aicore exception 507015` (illegal DMA access) rather
   than a build failure.

We are not filing this as "your code is broken": upstream has no 910B path today (see #3448).
The ask is (a) documenting that the dav-2201 surface is the V1 9-argument family and (b) whether
a 910B port that re-derives this lowering would be welcome (see *Proposed fix*).

## Minimal repro

Self-contained; **does not depend on our fork**. Only CANN is needed. Put the following in
`probe.asc`:

```cpp
#include <cstdint>
#include "kernel_operator.h"

extern "C" __global__ __cube__ void probe(__gm__ bfloat16_t *g) {
  __ca__   bfloat16_t *d = (__ca__ bfloat16_t *)(uintptr_t)0;
  __cbuf__ bfloat16_t *c = (__cbuf__ bfloat16_t *)(uintptr_t)0x10000;
  // (A) legal V1 9-arg form
  load_cbuf_to_ca(d, c, (uint16_t)0, (uint8_t)1, (uint16_t)1, (uint16_t)0,
                  (uint8_t)0, false, (__cce_scalar::addr_cal_mode_t)0);
  g[0] = (bfloat16_t)0;
}
```

Compile:

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
CANNR=/usr/local/Ascend/cann-8.5.0/aarch64-linux
INC=/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/lib/clang/15.0.5/include
bisheng -std=c++20 -O2 --npu-arch=dav-2201 -I$CANNR/asc/include -I$CANNR/include \
        --cce-aicore-only -c probe.asc -o probe.o
```

Then vary the call. Results reproduced by the author on `cann910b-k` (exit codes are the only
verdict; the message text is quoted verbatim):

| variant | change | exit | message |
|---|---|---|---|
| A | 9-arg V1, `sid=0` | **0** | — |
| B | A but `sid = (uint8_t)105` | **1** | `probe.asc:6:85: error: the range of 7th parameter must be [0, 15]` |
| C | A but **8 args** (omit `addr_cal_mode`) | **1** | `error: function type 'void (__ca__ __bf16 *, __cbuf__ __bf16 *, unsigned short, unsigned char, unsigned short, unsigned short, bool, unsigned int) noexcept' of 'load_cbuf_to_ca' does not support the given target feature` |
| D | A but 4th arg `1.5f` | **1** | `error: the 4th parameter maybe need a type 'unsigned char'` |
| E | A but `sid = (uint8_t)g[4]` (**runtime** value) | **0** | — (no diagnostics; the 4-bit range is unchecked for runtime values) |
| F | A but `transpose = (bool)g[5]` (**runtime** value) | **1** | `error: function type 'void (__ca__ __bf16 *, __cbuf__ __bf16 *, unsigned short, unsigned short, unsigned char, unsigned char, short, unsigned short, bool) noexcept' of 'load_cbuf_to_ca' does not support the given target feature` |

Variant F is instructive: a non-constant `transpose` falls through to the **3101 explicit-form**
signature (note the `unsigned short, unsigned short, unsigned char, unsigned char, short, unsigned
short, bool` shape) and is rejected as an unsupported target feature, i.e. the error text is
misleading about which form is actually expected.

## Observed vs Expected

### Expected (official dav-2201 implementation, quoted verbatim)

`asc/impl/basic_api/dav_c220/kernel_operator_mm_impl.h` (CANN root `.../aarch64-linux/`), the
`LoadData2DL12L0ACal` body — signature at **line 24**, two call sites verified at **lines 32-33**
(the `ifTranspose` branch; its `if` is at line 31) and **lines 35-36** (the else branch; its `else`
is at line 34), with the `int4b_t` s4 path at **lines 28-29**:

```cpp
__aicore__ inline void LoadData2DL12L0ACal(__ca__ T* dst, __cbuf__ T* src, const LoadData2DParams& loadDataParam)  // :23-24
{
    if ASCEND_IS_AIC {
        if constexpr (IsSameType<T, int4b_t>::value) {
            load_cbuf_to_ca_s4((__ca__ void *)dst, (__cbuf__ void *)src, loadDataParam.startIndex,
                loadDataParam.repeatTimes, loadDataParam.srcStride, loadDataParam.dstGap, loadDataParam.sid, 0, inc);   // :28-29
        } else {
            if (loadDataParam.ifTranspose) {
                load_cbuf_to_ca(dst, src, loadDataParam.startIndex, loadDataParam.repeatTimes, loadDataParam.srcStride,
                    loadDataParam.dstGap, loadDataParam.sid, 1, inc);                                                 // :32-33
            } else {
                load_cbuf_to_ca(dst, src, loadDataParam.startIndex, loadDataParam.repeatTimes, loadDataParam.srcStride,
                    loadDataParam.dstGap, loadDataParam.sid, 0, inc);                                                 // :35-36
            }
        }
    }
}
```

So the official lowering is exactly the **9-argument** order; the tail slot is `inc` (a member of
the enclosing enum `__cce_scalar::addr_cal_mode_t`), and the sema names that slot
`addr_cal_mode_t`.

The field order is defined by `asc/include/basic_api/kernel_struct_mm.h` (**verified lines 22-43**):

```cpp
struct LoadData2DParams {                                                             // :22
    __aicore__ LoadData2DParams(const uint16_t startIndexIn, const uint8_t repeatTimesIn,
        const uint16_t srcStrideIn, const uint8_t sidIn, const uint16_t dstGapIn,
        const bool ifTransposeIn, const uint8_t addrModeIn)                            // :25-26
        : startIndex(...), repeatTimes(...), srcStride(...), sid(...),
          dstGap(...), ifTranspose(...), addrMode(...) {}                              // :27-33
    uint16_t startIndex = 0;   // :36
    uint16_t dstGap     = 0;   // :37
    uint16_t srcStride  = 0;   // :38
    bool     ifTranspose = 0;  // :39
    uint8_t  repeatTimes = 0;  // :40
    uint8_t  sid         = 0;  // :42
    uint8_t  addrMode    = 0;  // :43
};
```

Note the declaration order of the members (`startIndex, dstGap, srcStride, ifTranspose,
repeatTimes, sid, addrMode`) is **different** from the constructor/call order
(`startIndex, repeatTimes, srcStride, sid, dstGap, ifTranspose, addrMode`) — a second easy way
to mis-slot the descriptor if one copies the member list instead of the constructor list.

### What the intrinsic actually is on dav-2201

`cce_aicore_intrinsics.h` (in
`/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/lib/clang/15.0.5/include/`) declares these
as variadic builtin aliases, **not** as C++ overloads — which is why the signature is enforced in
the *builtin sema* (verified lines):

```cpp
974:  __attribute__((clang_builtin_alias(__builtin_cce_copy_gm_to_cbuf))) void copy_gm_to_cbuf(...);
992:  __attribute__((clang_builtin_alias(__builtin_cce_copy_gm_to_ubuf))) void copy_gm_to_ubuf(...);
1384: __attribute__((clang_builtin_alias(__builtin_cce_load_cbuf_to_ca))) void load_cbuf_to_ca(...);
```

and `__cce_scalar::addr_cal_mode_t` is (verified 442-475):

```cpp
namespace __cce_scalar {                                             // :442
  typedef enum { inc = 0, dec = 1, } addr_cal_mode_t;                // :463-466
}
```

The 3101 explicit form is arch-gated out on 2201 — `__clang_cce_types.h:231` is
`#if (__NPU_ARCH__ == 3101)` guarding `#include "cce_aicore_intrinsics_3101.h"` (verified).

## Impact

- Any 910B kernel that lowers an L1→L0A/L0B load through the wrong arity derives a
  field-misaligned L1→L0 descriptor. Because the `sid` range is unchecked for runtime values
  (repro E), the failure mode is **not a build error** but a runtime `aicore` exception
  `507015` (illegal DMA/L1 access) — or, worse, silent wrong L1 addressing.
- This is not hypothetical for a DSL: the row16 block start (`m_start`, 0..127) is a natural
  thing for a `T.gemm` lowering to pass in the third slot; if that lands in the 4-bit `sid` slot
  the kernel dies with 507015 (as observed in our `gemm_l1` / `dw_intrinsic` legs).
- The "7 vs 9 vs 8 arguments" ambiguity plus the `LoadData2DParams` member-vs-ctor order
  mismatch make this trap easy to fall into even for someone reading the official headers.

## Proposed fix / reference patch

Our port lowers to the V1 family directly (fold the 2-D start into the `src` pointer;
`startIndex ≡ 0`; `repeatTimes` along K/C0; a leading `for` over the M blocks; `sid ≡ 0` unless a
real L1 partition is used). This is validated **at compile level only** (bisheng, dav-2201);
on-device judgement is pending.

- Reference (not upstreamed, in our repo): the `asc_copy_l12l0a` / `asc_copy_l12l0b`
  wrappers in our `port910b_compat.h`, and the ISA evidence file `X_FRACTAL_EVIDENCE.md`
  (per-slot table). This is a *port shim*, not a patch to any upstream file.
- The change that a maintainer would actually want is the `tl.ascend_load_cbuf_to_ca` /
  `tl.ascend_load_cbuf_to_cb` **910B lowering**, which has no upstream implementation today.

**Could this be an issue rather than a PR?** — yes: at this time the actionable request is
documentation/roadmap (see #3448 for the same question at the backend level). If a maintainer
wants the lowering upstreamed, we can prepare a PR that adds it behind an opt-in
`dav-2201`/`TL_PORT910B_*` target guard, zero impact on the 950 path.

## What we have NOT determined

- Whether the *specific* `aicore 507015` seen in our `gemm_l1` leg is caused **solely** by the
  arity/slot misalignment. The structural fix (9-arg order + `sid` re-routing) was applied, and
  the leg still fails with 507015 on-device; a second, independent axis (the GM→L1 NZ layout and
  the strided GM→UB stride/gap semantics — see the companion drafts) is also in play. Arity
  correctness is *necessary*, we have not shown it is *sufficient*.
- The exact pre-fix 7-argument source text is **no longer present on disk**; it is recorded only
  as prose in our notes (`C3_closure_audit.md` §3, `X_sec12_proposal.md` §3). Treat "7 args" as
  our own historical statement, not an artifact you can grep.
- The absolute line numbers in `cce_aicore_intrinsics.h` / `kernel_operator_mm_impl.h` come from
  CANN 8.5.0; 8.5.2 may shift them. The declarations/semantics were not checked against 8.5.2.
- The exact V1 config-word bit packing for 2201 was not recovered bit-for-bit (only the field
  order, via the official `LoadData2DL12L0ACal` call and the sema slot types).
