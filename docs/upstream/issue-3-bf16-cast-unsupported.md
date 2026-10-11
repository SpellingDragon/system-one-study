# [Ascend][dav-2201] bisheng 15.0.5 backend rejects the `bfloat16_t` ↔ `float` cast (`not support bf16 type cast`, exit 70) — bf16 kernels that accumulate or store in fp32 cannot be built

> Draft for `tile-ai/tilelang`. **Not yet filed.** This is a **capability question**, not a bug
> report against TileLang: the wall is inside the vendored bisheng backend, and the ask is
> "is this planned / what is the recommended workaround". Status: reproduced first-hand at
> compile level; no on-device step is involved (the failure is a frontend/backend abort, so it
> never reaches the device).
> All bisheng results below were re-run by the author inside container `cann910b-k` on
> 2026-10-11 (not transcribed); the `Source:` line quotes our own generated code.

## Target / HW

- Ascend 910B (dav-2201), aarch64
- CANN 8.5.0 (docker) / 8.5.2 (instance)
- bisheng clang 15.0.5 — `.../ccec_compiler/bin/bisheng`, `clang version 15.0.5`, `Target: aarch64-unknown-linux-gnu`
- `--npu-arch=dav-2201` (the failure is arch-dependent: 2201 rejects the cast)

## Summary

On dav-2201, the bisheng backend aborts on **any scalar cast between `bfloat16_t` and `float`**.
A `T.gemm`/reduction lowering for bf16 needs exactly this cast (load bf16 from GM → promote to
fp32 → accumulate → store), so the whole "bf16 kernel that accumulates in fp32" pattern is
unbuildable on 910B through this toolchain. The diagnostic is a **backend abort with no source
location**:

```
fatal error: error in backend: not support bf16 type cast
bisheng: error: clang frontend command failed with exit code 70 (use -v to see invocation)
```

The cast — not the `bfloat16_t` type itself — is what fails: a *pure* bf16 copy (no cast to
`float`) compiles cleanly. This distinguishes "bf16 is unsupported" from "bf16 *cast* is
unsupported", and it is why our own attempt to enable a bf16 track surfaced as an obscure
`RuntimeError: Ascend device compilation failed` rather than a type error.

TileLang's Ascend backend (`target="ascend"`) has no bf16 lowering today, so this is not a
regression in upstream; it is a hard dependency we would hit if a bf16 path were added. We are
asking (a) whether bf16 scalar cast support is on the roadmap for dav-2201, and (b) whether
there is a supported intrinsic we missed (we only exercised the plain C++ cast path).

## Minimal repro

Self-contained; **does not depend on our fork**. Only CANN + bisheng are needed. Put the
following in `probe.asc`:

```cpp
#include <cstdint>
#include "kernel_operator.h"

extern "C" __global__ __vector__ void k(__gm__ bfloat16_t *a, __gm__ float *o) {
  float f = (float)a[0];        // <-- the only line that matters
  o[0] = f;
}
```

Compile (include paths are the standard CANN asc/impl ones; `source set_env.sh` first so the
bisheng plugin finds `ASCEND_HOME_PATH`):

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
CANNR=/usr/local/Ascend/cann-8.5.0/aarch64-linux
BS=/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler
$BS/bin/bisheng -std=c++20 -O2 --npu-arch=dav-2201 \
    -I$CANNR/asc/include -I$CANNR/asc/impl/basic_api -I$CANNR/asc/include/basic_api -I$CANNR/include \
    --cce-aicore-only -c probe.asc -o probe.o
```

Then vary the body. Results re-run by the author on `cann910b-k` (exit code is the verdict;
message text quoted verbatim):

| variant | kernel body | exit | message |
|---|---|---|---|
| A | `float f = (float)a[0];` (bf16→fp32), `__vector__` | **1** | `fatal error: error in backend: not support bf16 type cast` + `clang frontend command failed with exit code 70` |
| B | `bfloat16_t v = a[0]; float f = (float)v;` (via a variable) | **1** | same `not support bf16 type cast` |
| C | A, but `__cube__` face (body also issues `load_cbuf_to_ca`) | **1** | same `not support bf16 type cast` |
| D | `__gm__ bfloat16_t*` → `__gm__ bfloat16_t*`, **no cast** (`o[0]=a[0];`) | **0** | — |
| E | `float f = (float)a[0];` with `__gm__ half*` (fp16, not bf16) | **0** | — |
| F | reinterpret `__gm__ bfloat16_t*`→`__gm__ uint16_t*`, then `union{uint32_t u; float f;} v; v.u=((uint32_t)p[0])<<16;` | **0** | — |
| G | `(__gm__ bfloat16_t*)a` cast to `const void*` (to feed `memcpy`) | **1** | `error: casting '__gm__ bfloat16_t *' (aka '__gm__ __bf16 *') to type 'const void *' is not allowed in aicore function` |

Takeaways: **A/B/C** show the cast is rejected on *either* core face (so it is a backend cast
limitation, not a vector-unit one). **D** shows the `bfloat16_t` type drills through a pure
copy. **E/F** are the two compile-clean workarounds we found (use fp16, or bit-reinterpret).
**G** shows the obvious `memcpy`-style escape hatch is aicore-illegal.

## Observed vs Expected

- **Expected** (per any reasonable reading of the CANN headers): a scalar
  `bfloat16_t`→`float` conversion should lower to the usual `<<16` widen, as it does on other
  Ascend generations, and `-DTL_PORT910B_NATIVE_TYPES` code that loads fp32 accumulation
  operands from a bf16 buffer should build.
- **Observed:** the backend aborts before emitting code, with no file/line (bisheng gives only
  the `fatal error: error in backend:` line). The minimum anchor is the `bfloat16_t` GM pointer
  in the kernel signature plus the scalar cast in the body.

Our own captured instance (from our notes, `reconcile/C2_dtype_matrix.md` §8.1) is reproduced
verbatim; there the compiler was invoked by TileLang and the error surfaced as:

```
RuntimeError: Ascend device compilation failed.
Command: .../ccec_compiler/bin/bisheng -O2 -fPIC -std=c++20 -DTL_PORT910B_NATIVE_TYPES \
         --npu-arch=dav-2201 -cce-aicore-only /tmp/<x>/tl_kernel.asc -o /tmp/<x>/tl_kernel.rel.o
Compiler output:
  fatal error: error in backend: not support bf16 type cast
  bisheng: error: clang frontend command failed with exit code 70 (use -v to see invocation)
  clang version 15.0.5   Target: aarch64-unknown-linux-gnu
Source:
  extern "C" __global__ __vector__ void gemm_impl_kernel(__gm__ bfloat16_t* A, __gm__ bfloat16_t* C,
                                                          __gm__ bfloat16_t* W, __gm__ float* bias, int32_t m)
```

i.e. three of our kernels (gemm / gemm_dw / rope bf16) all died at the same backend abort, with
the minimal anchor being the `__gm__ bfloat16_t*` signature + the scalar cast the DSL inserts at
the `_ld/_st` load/store seam.

## Impact

- **Any TileLang Ascend kernel whose dtype is bf16 and whose arithmetic is fp32 is unbuildable
  on 910B** — which is the *only* sensible bf16 kernel shape (bf16 storage, fp32 accumulate).
  A "bf16 support" toggle for the Ascend backend cannot be honored on dav-2201 through this
  toolchain.
- For our port this is a **hard capability gap, not a workaround-able bug**: we explicitly gate
  bf16 off at the plan level (reject + fall back to eager/fp16) rather than pretend to support
  it. The cost is that a bf16 model cannot use the Ascend kernels at all.
- The failure is silent at the TileLang level: it appears only deep in a raised
  `Ascend device compilation failed` with a backend message, so it is easy to misattribute to a
  DSL/type-mapping bug. Documenting it saves the next porter the same detour.

## Proposed fix / reference patch

This draft is **question-first** — there is nothing for us to patch in TileLang until the
toolchain changes. Two questions for the maintainers / CANN owners:

1. **Is bf16 scalar cast support planned for dav-2201**, and if so on which CANN / bisheng
   version? (The failure is on 8.5.0; we have not checked whether a newer bisheng lifts it.)
2. **Is there a supported intrinsic we overlooked?** We only exercised the plain C++ cast and
   the `AscendC` type surface that `kernel_operator.h` exposes for this arch; we did not find a
   `ToFloat`/vector-convert that is legal on 2201 for bf16.

Until then, our posture is:
- **Primary workaround: use fp16.** The fp16→fp32 cast is accepted (repro E) and fp16 is our
  production dtype.
- **Fallback workaround: bit-reinterpret.** Reinterpret the buffer as `__gm__ uint16_t*` and
  widen with `(uint32_t)h << 16` (repro F). This avoids the bf16 type in the cast entirely and
  compiles cleanly, at the cost of losing the type in the kernel signature and hand-rolling the
  conversion at every seam. **Validated at compile level only; on-device numerics not yet
  exercised.**

**Issue vs PR:** this should be an **issue / question**, not a PR — there is no upstream code
change we are proposing; the repository change, if any, would be a documented "bf16 unsupported
on dav-2201" note.

## What we have NOT determined

1. **Whether this is really "vector-face" only.** Our own notes characterise it as the vector
   face (`reconcile/C2_dtype_matrix.md` §8), and the kernels that failed for us were `__vector__`.
   My re-run additionally reproduces the identical abort on a `__cube__`-face kernel that also
   issues `load_cbuf_to_ca` (variant C), with a clean control that omits the cast (exit 0). So the
   trigger is the **cast**, on either face — but I did not exhaustively test every combination of
   core face / cast direction / storage class.
2. **No test of the reverse direction.** We only probed `bfloat16_t`→`float`. `float`→`bfloat16_t`
   (the store seam) was not isolated; it may fail for the same or a different reason.
3. **Version matrix unknown.** Only 8.5.0 was exercised here (and the instance-side 8.5.2 in the
   original capture). We have **not** checked whether any other CANN/bisheng release on dav-2201
   accepts the cast, nor whether the "not support" is a flag-gated limitation
   (`-DTL_PORT910B_NATIVE_TYPES` was present in our invocation; the minimal repro above does not
   require it).
4. **A `SIMD`/vector conversion path we may have missed.** We tried the scalar cast and a
   `uint16_t`+union widen; we did **not** search the `AscendC` vector intrinsic surface for a
   legal bf16 convert on 2201 (e.g. a `vconv`-style builtin), so "no way exists" would be an
   over-claim.
5. **Workaround F is compile-only.** Variant F builds, but we have not run it on device to
   confirm bit-exactness of the `<<16` widen for special values (NaN/Inf/subnormal bf16).
