# [Ascend][dav-2201] `copy_gm_to_ubuf` / `copy_ubuf_to_gm` stride slots are inter-burst *gaps*, not row *pitch* — multi-burst strided GM↔UB copies (`bk < k`) load the wrong data

> Draft for `tile-ai/tilelang`. **Not yet filed.** Status: **partially localised** — the ISA
> semantics below are first-hand and cross-checked; the on-device effect is real (an archived
> on-card verdict exists), but the full error picture is *not* explained by a single geometric
> model (see *What we have NOT determined*). All quotes were opened and re-read by the author
> against CANN 8.5.0 in container `cann910b-h`.

## Target / HW

- Ascend 910B (dav-2201), aarch64
- CANN 8.5.0 (docker) / 8.5.2 (instance); bisheng clang 15.0.5
- Triggering TileLang shape: a `T.copy` GM→UB of a 2-D region whose **row length ≠ row pitch**
  (`blockCount > 1 && srcStride ≠ blockLen`), i.e. `bk < k` for a `T.gemm` A/W tile.

## Summary

On dav-2201 the native MTE intrinsics `copy_gm_to_ubuf` / `copy_ubuf_to_gm` take, in slots 6 and
7, an **inter-burst gap**, *not* an address-to-address pitch. Three independent CANN sources say
so (below). TileLang's Ascend copy planner (`src/ascend/op/copy.cc`, `plan_single_row`) computes
and forwards an inter-row **pitch** in those slots, and only collapses to a single burst when
`stride == row_bytes` on both sides. So for `bk < k` multi-burst strided copies the hardware
advances by `blockLen + pitch` per burst instead of `pitch` — the loaded rows are misaligned and
the tail bursts run past the intended region.

We are reporting this against upstream `copy.cc` because the planner's naming/contract
(`src_stride` / `dst_stride`, pitch semantics) is the locus that does not survive the port to
dav-2201's gap semantics. It is *not* a claim that the 950 path is wrong — we have not observed
950 hardware.

## Minimal repro

### (a) Reproducing the geometry on dav-2201 (self-contained, CANN only)

```cpp
#include <cstdint>
#include "kernel_operator.h"
extern "C" __global__ __vector__ void k(__gm__ float *g) {
  __ubuf__ float *u = (__ubuf__ float *)(uintptr_t)0;
  // 32 bursts of 64B each, source rows 192B apart, destination rows 64B apart (K=48, bk=16, fp32)
  __cce_scalar::copy_gm_to_ubuf((__ubuf__ void *)u, (__gm__ void *)g, (uint8_t)0,
                                (uint16_t)32, (uint16_t)2, (uint16_t)6, (uint16_t)2);
  g[0] = u[0];
}
```

compiled with the same `bisheng -std=c++20 -O2 --npu-arch=dav-2201 --cce-aicore-only -c` line as
the companion issue. This compiles clean (exit 0) — i.e. the call shape is *accepted*; the slots
are block-counted (here 2 = 64B, 6 = 192B, 2 = 64B). **The defect is semantic, not a compile
error**, and cannot be demonstrated without a device (see *What we have NOT determined*).

### (b) The TileLang-level repro (what the planner emits)

`T.copy` of a GM region with row length 64B and row pitch 192B, `blockCount = 32`. The archived
generated kernel (from our own on-card run; `sys1_kernels.tgz` md5
`f2ba9a2080de73d1d04d4a309dc8496e`) contains, at **`device_kernel.asc:31`**:

```c
// 5d732a8f…/device_kernel.asc:31  (fp32 gemm, m16 n96 k48)
asc_copy_gm2ub_align((__ubuf__ uint8_t*)(…), (__gm__ uint8_t*)(…),
                     32, 64, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 192, 64);
//                   ^blockCount ^blockLenB                          ^srcStrideB ^dstStrideB
```

that is, the planner supplied `srcStride=192B` and `dstStride=64B` while wrapping a 64B row.
The same geometry appears on the fp16 side
(`ea1059e6…/device_kernel.asc:31`, `…, 32, 64, 0, 0, 0, mode0, 192, 64`).

## Observed vs Expected

### Evidence A — `srcStride`/`dstStride` are union aliases of `srcGap`/`dstGap`

`asc/include/basic_api/kernel_struct_data_copy.h` (**verified lines 45-58**):

```cpp
uint16_t blockCount = DEFAULT_DATA_COPY_NBURST;                     // :45
uint16_t blockLen = 0;                                              // :46
// srcStride and dstStride will be deprecated, use srcGap and dstGap instead.   // :47
union { uint16_t srcGap = 0; /* srcStride will be deprecated, use srcGap instead */ uint16_t srcStride; };  // :48-52
union { uint16_t dstGap = 0; /* dstStride will be deprecated, use dstGap instead */ uint16_t dstStride; };  // :54-58
```

### Evidence B — the bounds formula is "lengths plus gaps", not "pitch"

`asc/impl/basic_api/utils/kernel_check_data_copy_util.h` (**verified lines 344-352**):

```cpp
uint16_t nBurst = repeatParams.blockCount;                          // :344
uint16_t lenBurst = repeatParams.blockLen;                          // :345
uint16_t srcStride = repeatParams.srcStride;                        // :346
uint16_t dstStride = repeatParams.dstStride;                        // :347
// unit byte
srcMaxOffset =                                                      // :349
    static_cast<uint64_t>(nBurst) * lenBurst * srcBurstLenUnit + (nBurst - 1) * srcStride * srcStrideUnit;   // :350
dstMaxOffset =                                                      // :351
    static_cast<uint64_t>(nBurst) * lenBurst * dstBurstLenUnit + (nBurst - 1) * dstStride * dstStrideUnit;   // :352
```

If the slot were a pitch the span would be `(n-1)*pitch + len`; instead it is
`n*len + (n-1)*stride`, i.e. every burst is `len`, and `stride` is the empty space between
consecutive bursts ⇒ **gap**.

### Evidence C — the official 910B PTO template computes `gap = stride − len`

`include/pto/npu/a2a3/TLoad.hpp` (**verified lines 69-72**), feeding the `copy_gm_to_ubuf_align_*`
family (`:20/:22/:24/:26`, `TLoadInstrGm2ub` at `:15-28`):

```cpp
uint64_t gmGapValue = (gStride3 - gShape4) * sizeof(typename TileData::DType);   // :69  gap = (rowStride - rowLen)*elemBytes
uint32_t gmGap = (uint32_t)gmGapValue;                                            // :70
uint32_t ubGapElement = (TileData::Cols - validCol);                              // :71
uint32_t ubGap = (ubGapElement * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;  // :72
...
copy_gm_to_ubuf_align_b16(dst, src, 0, nBurst, lenBurst, 0, ubPad, gmGap, ubGap);       // :22
```

The official template itself converts **pitch → gap** before the intrinsic. A caller that
forwards a raw pitch contradicts this.

### Expected vs observed

- **Expected:** for the emitted tuple `(blockCount=32, blockLen=64B, srcStride=192B, dstStride=64B)`
  the hardware should advance the source by 192B and the destination by 64B per burst. With the
  gap semantics this is only true if those slots receive `192-64 = 128B` and `64-64 = 0`.
- **Observed:** the slots receive the pitch (`192`, `64`) ⇒ the hardware advances by
  `64+192 = 256B` (source, should be 192B) and `64+64 = 128B` (destination, should be 64B). The
  destination therefore only lands data in every other 64B block and walks into the neighbouring
  `c_ub` region; the source over-strides.

### On-device corroboration

On-card, the fp32 `k=48` linear failed with a wrong-value (not-just-noise) signature
`rel=1.150e+00, got=15.24771 ref=-15.37733` (`runs/1011-p2-13-oncard-accept-2002/notes.md` §②-1),
while the `k=32`/`k=64` shapes were green (`1.16e-07` …). The fp16 training step went
`loss0=0.975374 → nan` on the *backward*, and the only fp16 kernel in the archive that hits the
multi-burst strided path is the one shown above.

## Impact

Every GM↔UB copy with `blockCount > 1` **and** `row length ≠ row pitch` is affected — in our
archive that is 4 kernels out of ~46 (`k=48` fp32, `k=96` fp32, `k=96` fp16, and a bf16
matmul with `blockCount=2048`). The other 40+ kernels are all `blockCount == 1` (`gap = pitch −
len = 0`), i.e. **byte-identical** under either semantics, so there is no small-shape regression.
The `bk < k` condition is exactly the "odd-K gemm" case, so this shows up as the *only* failing
shape class for a gemm-heavy model.

## Proposed fix / reference patch

Two options, smallest first:

- **Option A (compat/adapter layer)** — normalise pitch→gap at the point where the TileLang
  stride slots meet the native intrinsic: `gap = max(pitch − blockLen, 0)`. Compile-level
  self-check passes; `blockCount == 1` and `pitch == blockLen` are unchanged (gap 0), so it is
  regression-safe. **Validated at compile level only; on-device judgement pending.**
- **Option B (planner layer, in `copy.cc`)** — emit a `blockCount == 1` contiguous DMA per row
  (or a scalar loop) for the strided case, so the stride slots do not participate at all. This is
  semantics-independent (correct under *any* of pitch / gap / dstStride-ignored) but costs
  instruction count (×32 MTE2 for our `blockCount=32` case).

Reference patch (Option A) in our repo: `attempts/S3_compat_proposal.diff` — it is a *proposal*,
**not landed** (our compat header md5 is still `ca023c2e…`, unchanged). It also mirrors the fix
on `asc_copy_ub2gm_align`, which is *inferred* rather than observed (the offending kernel's exit
is a scalar `write_gm_bypass_dcache`, so the ub→gm hunk was never exercised) and must be verified
on-card alongside the gm→ub hunk.

**Issue vs PR:** we would file this as an **issue** first, because the fix location depends on the
maintainer's intent — an adapter normalisation (Option A) is a one-file change, but a planner
fix (Option B) changes the generated DMA shape and we would want agreement before proposing it.

## What we have NOT determined — read this before acting on the above

1. **A single geometric rule does not explain the observed data.** An offline exhaustive search
   over `(srcStride, dstStride, globalOffset)` of **308,945 combinations found zero** that
   reproduce the observed error map (minimum L1 residual 167; see
   `reconcile/S3_STRIDED_SECOND_FACTOR.md` §5). So *at least one non-geometric factor* remains
   undetermined — our best (unproven) hypothesis is **stale/uninitialised UB**: the `w_ub` buffer
   is never cleared, and the "holes" left by a gap-mis-strided DMA are filled with whatever a
   previous launch left at that address. This has **not** been directly measured (a
   pre-clear experiment is designed but not yet run).
2. **We explicitly retract an earlier conclusion.** A previous internal window concluded "the
   hardware ignores `dstStride`" from a probe in which the winning candidate simply submitted
   fewer finite cells; the scoring never penalised "observed finite but candidate NaN", so a
   candidate that predicts a large NaN region wins by construction. That verdict was a **metric
   artifact** and is **withdrawn**. **Do not write "the hardware ignores dstStride"** — it is not
   established by our data, and the mechanism narrated in that window ("hardware writes each
   burst contiguously") did not even match the candidate model that supposedly won.
3. **Direction is not closed.** The evidence above shows the natives' slots are gaps; whether the
   planner *thinks* it is passing a pitch (our reading of `copy.cc`) or something else is a
   naming question we have not resolved past the code comments. Treat "pitch passed where gap is
   expected" as the leading hypothesis, not a verdict.
4. **No device reproduction of the minimal repro.** We have an archived on-card failure and a
   clean model, but the two have not been closed by a single controlled on-card experiment
   (the one-shot discriminating probe is authored, not yet run). Local `cpu`/`cpu-alias` lanes
   are structurally blind to this defect (the strided DMA shape is only produced for real
   `ascend` lowering).
5. `asc_copy_ub2gm_align` counterpart is **inferred**, not observed (see above).
