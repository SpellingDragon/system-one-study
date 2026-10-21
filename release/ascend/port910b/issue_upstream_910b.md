# Upstream issue (SUBMITTED 2026-10-08): https://github.com/tile-ai/tilelang/issues/3448

Title:
```
[Ascend][DSL] 950-generation intrinsic emission blocks 910B (dav-2201, CANN 8.5) — findings + working compat port
```

Body:
```markdown
## Summary

`tilelang.compile(..., target="ascend")` installs and runs its frontend fine on
Atlas 910B (aarch64, CANN 8.5.x), but the generated kernel does not compile:
the Ascend DSL codegen emits the 950-generation intrinsic surface
(`c_api/asc_simd.h`, `simt_api/asc_{bf16,fp16,fp8,simt}.h`, bare `asc_*` calls),
which does not exist for `dav-2201` in CANN 8.5's bisheng toolchain. We ported
the template layer to 910B with a conditional-compatibility shim and got a bf16
gemm (explicit `T.Cube()` block) to compile end-to-end on 910B through the DSL
path. This issue reports the dialect-gap findings and asks whether 910B
support for the DSL path is on the roadmap.

## Environment

- Atlas 900 model A / 910B4 (aarch64), CANN 8.5.0 toolkit (docker) and 8.5.2 (instance image)
- bisheng clang 15.0.5 (both `bin/bisheng` wrapper and `ccec_compiler/bin/bisheng`)
- tilelang 0.1.15 (pip), reproduced against current main templates

## What breaks on 910B (dav-2201)

1. `c_api/simt_api` headers are absent from CANN 8.5; every bare `asc_*`
   intrinsic the codegen emits (`asc_init`, `asc_mmad`, `asc_sync_intra_wait`,
   `asc_copy_gm2ub_align`, ...) is undeclared. In CANN's own headers the MMAD
   family is gated to `__NPU_ARCH__ == 5102/9201/3801/3101` — 2201 is not in
   the list.
2. `std::bit_cast` missing (clang 15 + no libstdc++ on device TU) — needs
   `__builtin_bit_cast`.
3. `bfloat16_t` in cce context is an opaque proxy type
   (`BISHENG_BUILTIN_TYPE_PROXY(BFloat16)`): deleted copy constructor, no
   conversion operators, no `.data` — pass-by-value and `(float)` casts fail.

## Findings that may be useful to maintainers

- **F1 (cube arch):** compiling cube code for 910B requires
  `--cce-aicore-arch=dav-c220-cube` (the ASCPLUGIN passes this internally for
  `.asc` inputs). A bare `--npu-arch=dav-2201` compiles the intrinsic *syntax*
  but then fails the target-feature check on `mad` at codegen time.
- **F2 (`__mix__`):** 910B cce natively defines
  `#define __mix__(cube,vec) __attribute__((core_ratio(cube, vec)))`. A shim
  should stand down and let the native macro through.
- **F3 (plugin registration):** for `.asc` inputs the plugin injects a kernel
  registration segment that references the official TLV metadata from
  `asc/impl/basic_api/utils/kernel_utils_macros.h` (`FuncMetaType`,
  `KernelType`, `g_sysFftsAddr`, ...). Templates that don't include that chain
  fail with `undeclared identifier 'F_TYPE_KTYPE'` etc.
- **F4 (single-work escape hatch):** an explicit `T.Cube()` block lowers to a
  `kCube` single-work kernel, which avoids the mix cross-core-sync registration
  checks entirely. With that + a compat shim (type proxies via bit_cast,
  `asc_*`→CCE-native adapters like `mad`, `__cce_pipe_barrier`, `copy_gm_to_ubuf`),
  a 2048^3 bf16 gemm compiles end-to-end on 910B.
- **F5 (bf16):** `__bf16` is not a supported builtin type on dav-2201; the
  proxy requires `const&` parameter passing + `__builtin_bit_cast` software
  conversion.

## Minimal repro

aarch64 docker (or any 910 instance), CANN 8.5 toolkit sourced,
`pip install tilelang`, then any `T.copy/T.gemm` kernel with
`target="ascend"`, e.g. `examples/ascend/test_gemm.py` — template inclusion
fails at `common.h` with the missing `c_api/simt_api` headers.

## Questions

1. Is 910B (dav-2201) support on the roadmap for the **DSL codegen path**? We
   are specifically interested in the DSL path rather than switching to the
   Ascend C generation path, since our stack standardizes on the tilelang DSL
   across backends.
2. Our compat layer (all behind `TL_PORT910B_NATIVE_TYPES` /
   `TL_ASCEND_SIMT` guards, zero impact on the 950 path) proves the DSL path
   can compile on 910B. Would a minimal PR adding this as an opt-in target be
   welcome once it passes on-card numeric validation?
```
