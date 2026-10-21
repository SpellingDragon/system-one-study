# Upstream feedback drafts — `docs/upstream/`

Three submission-ready drafts for `tile-ai/tilelang`, produced offline by the p2-13 upstream
track (agent R2). **Nothing here is filed, pushed, or PR'd** — this directory is the review
staging area for the decision-maker.

| draft | file | one line |
|---|---|---|
| **#1** | `issue-1-l12l0a-arity.md` | dav-2201 L1→L0A/B load is a **9-argument** V1 family; a shorter `load_cbuf_to_ca` arity mis-maps `sid`/`transpose` and only surfaces as `aicore` exception 507015. Airtight (official headers + sema repro, no device needed). |
| **#2** | `issue-2-strided-gm2ub-gap-semantics.md` | `copy_gm_to_ubuf` / `copy_ubuf_to_gm` stride slots are **inter-burst gaps**, not row pitch; the upstream `copy.cc::plan_single_row` planner forwards a pitch ⇒ `bk < k` multi-burst strided GM↔UB copies load the wrong data. **Highest value**, but **partially localised**. |
| **#3** | `issue-3-bf16-cast-unsupported.md` | bisheng 15.0.5 dav-2201 backend aborts on the `bfloat16_t`↔`float` cast (`not support bf16 type cast`, exit 70). **Capability question**, not a bug. |

All three were written to the same required structure (Title / Target-HW / Minimal repro /
Observed-vs-Expected / Impact / Proposed fix / What we have NOT determined). Every cited line
number was opened and re-read by the author against CANN 8.5.0 inside a container (the three
ISA quotes in #2 were re-opened, not transcribed from our own notes); every sema/compile verdict
in #1 and #3 was re-run, not copied.

## Submission order — recommendation

**Recommended order: #1 → #2 → #3.**

1. **#1 first.** It is the most airtight: the entire evidence chain is official CANN headers plus a
   self-contained set of pass/fail compile probes that a maintainer can reproduce in one `bisheng`
   invocation, with **no device and no private repo** required. Leading with the unambiguous one
   maximises the chance the maintainer engages; it also establishes the theme
   ("the dav-2201 lowering differs from the 950 path") that #2 then builds on.
2. **#2 second.** It is the highest-*value* item (a real, archived on-card numeric failure) and
   the only one with a genuine upstream code locus (`src/ascend/op/copy.cc`). File it after #1 so
   it can cross-link, and so its honest "partially localised" framing is read against the already
   demonstrated quality of #1. Its ask is deliberately open (adapter vs planner), which needs a
   maintainer reply before any PR.
3. **#3 last.** Orthogonal (a toolchain capability question, not a TileLang code issue). Filing it
   separately keeps it from diluting the two technical issues; order relative to #1/#2 does not
   matter much.

*Alternative:* if the maintainers triage by impact-per-issue rather than by evidence quality, swap
#1 and #2 (#2 first). We recommend #1 first because #2's own "What we have NOT determined" admits
a single geometric model does not yet explain the data, so it benefits from a credibility
predecessor.

## Issue vs PR

| draft | form | why |
|---|---|---|
| #1 | **issue** first, optional opt-in **PR** later | At the backend level the actionable ask is documentation / roadmap (same as #3448). Upstream has **no** 910B L1→L0 lowering today, so there is nothing to fix in place. If a maintainer wants it upstreamed, we can prepare a PR that adds the lowering behind an opt-in `dav-2201` / `TL_PORT910B_*` guard — zero impact on the 950 path. |
| #2 | **issue** first, **PR** only after agreement | The fix location is ambiguous: an adapter normalisation (`gap = max(pitch − blockLen, 0)`) is a one-file change, but the "correct" fix in the planner (`copy.cc`) changes the generated DMA shape. We want the maintainer's intent before proposing either. |
| #3 | **issue / question** | No upstream code change is proposed — the wall is inside the vendored toolchain. At most it becomes a documented "bf16 unsupported on dav-2201" note. |

## Fork-vs-trunk delta (measured, not from memory)

`git -C $WORKSPACE_ROOT/tilelang diff --numstat` @ trunk
`994b44eca1a83a00d19d926e5264a6c608700656` (measured 2026-10-11):

```
14  2   src/tl_templates/ascend/common.h
18  8   src/tl_templates/ascend/dcache_bypass.h
35  7   src/tl_templates/ascend/debug.h
 2  0   src/tl_templates/ascend/gemm.h
13  9   src/tl_templates/ascend/numeric_limits.h
----
82 insertions(+), 26 deletions(-)   (5 tracked files)
```

Untracked new files (all our additions):

```
1248  src/tl_templates/ascend/port910b_compat.h      (the 910B compat shim — the bulk of the port)
   7  src/c_api/asc_simd.h                          (950-header stub)
   4  src/simt_api/asc_bf16.h                       (950 SIMT-surface stub)
   4  src/simt_api/asc_fp16.h                       (950 SIMT-surface stub)
   4  src/simt_api/asc_fp8.h                        (950 SIMT-surface stub)
   4  src/simt_api/asc_simt.h                       (950 SIMT-surface stub)
```

**What each tracked change actually is** (all are `910B`-behind-a-macro guards; the whole delta
is under `src/tl_templates/ascend/` + the two new include dirs — **no Python and no
`src/ascend/op/` is touched**, i.e. `copy.cc` / `codegen_ascend.cc` are pure upstream):

| file | change | guard |
|---|---|---|
| `common.h` | splits the include surface by generation: 950 SIMT headers (`c_api/asc_simd.h`, `ascend_fp8.h`, `reduce.h`, `simd_inst.h`) are now `#if defined(TL_ASCEND_SIMT)`; the `#else` branch pulls in `port910b_compat.h`. | `TL_ASCEND_SIMT` |
| `dcache_bypass.h` | `TL_GM_LOAD` / `TL_GM_STORE` macros: on the 910B cce face use plain `__gm__` access instead of the `asc_load_dev` / `asc_store_dev` builtins (absent on 910B). | `TL_PORT910B_NATIVE_TYPES` |
| `debug.h` | empty `__aicore__` shim; `TL_ASCEND_ASSERT` = `__builtin_trap()` (no host `<cassert>` on the cce face); empty `device_printf`; `block_idx`→`0`; the SIMT debug variant gated out. Also routes `(float)val` in the fp16/bf16 print paths through `tl910b_debug_as_float(...)` — **a direct consequence of draft #3** (the raw `(float)` bf16 cast does not compile on dav-2201). | `TL_PORT910B_NATIVE_TYPES` / `TL_ASCEND_SIMT` |
| `gemm.h` | gates the MX / blockscaled gemm face (`ascend_blockscaled_gemm_l1`) out on 910B (fp8/SF types + `asc_*_mx` builtins are 950-only). | `!TL_PORT910B_NATIVE_TYPES` |
| `numeric_limits.h` | `std::bit_cast` → `__builtin_bit_cast` (bisheng's C++20 stdlib lacks `std::bit_cast`). | none (macro `TL_BIT_CAST`) |

`tl910b_debug_as_float` is defined in `port910b_compat.h:517-520` (a `bfloat16_t` overload plus a
generic `static_cast<float>` one).

## Recovery list — what our fork can drop after upstream feedback

Ordered by how cleanly each maps to one of the three drafts:

1. **Whole 910B shim — if upstream grants #3448 / lands a native dav-2201 target.**
   Delete the untracked additions (`port910b_compat.h`, `src/c_api/asc_simd.h`,
   `src/simt_api/*`) and revert the five tracked files to trunk. None of the tracked hunks do
   anything except route around the missing 910B surface, so they are pure subtraction, not
   behaviour we need to keep.
2. **`copy.cc` pitch→gap normalisation — if upstream adopts draft #2.**
   Our adapter-level fix lives in the *not-yet-landed* `attempts/S3_compat_proposal.diff`
   (targets `asc_copy_gm2ub_align` and `asc_copy_ub2gm_align`); the current compat header md5 is
   still `ca023c2e01746ca52c87f36bcea93ee9`, i.e. **not applied**. If upstream normalises
   pitch→gap in the planner instead, do **not** apply our adapter too — double-normalisation
   would subtract `blockLen` twice. Decide by which layer ends up owning the conversion.
3. **`debug.h` `tl910b_debug_as_float` — if bisheng lifts the bf16 cast (draft #3).**
   The `tl910b_debug_as_float` indirection exists only to avoid the illegal `(float)bf16` cast;
   once the backend accepts it, the print paths can revert to a plain `(float)val` and the helper
   (and its `port910b_compat.h` definition) can go.
4. **`numeric_limits.h` `TL_BIT_CAST` — independent of all three.**
   Drop when the vendored bisheng libc++ provides `std::bit_cast`; unrelated to the drafts, listed
   here only so the recovery list is complete.

## Cross-links and caveats

- **#3448** (the already-filed backend-level issue) is tracked in `STATUS.md`; #1/#2 extend its
  F-findings and #3 extends its bf16 question. Nothing in these drafts contradicts #3448.
- Draft #2 carries two load-bearing retractions in its "What we have NOT determined": the
  "hardware ignores `dstStride`" claim is **withdrawn** (metric artifact), and the 308,945-combo
  zero-hit means at least one non-geometric factor is still open. Do not let a reviewer read #2 as
  a closed verdict.
- Draft #3 records a first-hand finding that goes **slightly beyond** our notes: our notes call the
  bf16 failure "vector face", but the re-run reproduces it on a `__cube__`-face kernel too, so the
  trigger is the **cast**, not the core face. This is flagged inside #3 itself (see its
  "What we have NOT determined" #1) rather than silently asserted.
