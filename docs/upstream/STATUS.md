# STATUS — filed issue #3448 and its relationship to the three drafts

> **DECISION (decision-maker, 2026-10-11 evening): no further issues will be filed.** The PMC has
> signalled it does not intend to maintain the 910 (Ascend 910B) target. The three drafts below are
> re-scoped as **PR material, held on ice** — to be submitted as one PR package only after ALL
> mainline tasks are complete and verified. No external action from this track until then.

Last updated 2026-10-11 (agent R2, upstream feedback track). Read-only summary; **no action has
been taken on GitHub by this track** (no comment, no new issue, no PR).

## #3448 at a glance

| field | value |
|---|---|
| URL | https://github.com/tile-ai/tilelang/issues/3448 |
| Title | `[Ascend][DSL] 950-generation intrinsic emission blocks 910B (dav-2201, CANN 8.5) — findings + working compat port` |
| Filed | **2026-10-08** |
| Local copy | `release/ascend/port910b/issue_upstream_910b.md` |
| Run record | `release/runs/1008-p2-13-upstream-issue-3448-be5c/notes.md` (`结论：上游情报立此存照。PR 前置=上卡 RUNTIME-PASS（P0-2 [5]）。`) |
| Maintainer response | **none recorded** as of 2026-10-11 |

**What #3448 asks (at the backend/dialect level):** `tilelang.compile(..., target="ascend")`'s
codegen emits the **950-generation** intrinsic surface (`c_api/asc_simd.h`,
`simt_api/asc_{bf16,fp16,fp8,simt}.h`, bare `asc_*` calls) which does not exist for `dav-2201` in
CANN 8.5, so the generated kernel does not compile on 910B. It reports the dialect gap plus the
F-findings from our working compat port, and asks (Q1) whether 910B support for the **DSL codegen
path** is on the roadmap, and (Q2) whether an opt-in target PR would be welcome.

Its own findings, as filed: **F1** cube needs `--cce-aicore-arch=dav-c220-cube`; **F2** `__mix__` is
native on 910B; **F3** plugin registration needs the `kernel_utils_macros.h` TLV chain; **F4** an
explicit `T.Cube()` single-work block + compat shim compiles a 2048³ bf16 gemm; **F5** `__bf16` is
not a supported builtin on dav-2201 (proxy type; `(float)` casts fail).

## Engagement history

- **2026-10-08** — filed (`1008-…-3448-be5c`).
- **2026-10-09** — engagement window (`1009-p2-13-issue-3448-engaged-f114`): our comment
  `6077898715` was posted and then **deleted on the decision-maker's instruction**; no follow-up
  comment. Strategic read unchanged (triple-confirmed route A + the PTO-elimination measurement
  are on file). The `G-D2` bug material and the opt-in-PR intent were moved to **internal hold**
  — "对外动作留待上卡数值落地后、牌面完整时再定".
- **Gate for any follow-up:** the PR precondition recorded in `1008` is **on-card RUNTIME-PASS
  (P0-2 [5])**. The first real-device acceptance window (`1011-p2-13-oncard-accept-2002`) has since
  run and exposed the `K=48`/fp16 failure axes that drafts #1/#2 address — so the "牌面" is now
  more complete than it was on 2026-10-09, but a Go/No-Go on resuming the thread is still a
  decision-maker call, not this track's.

## How the three drafts connect to #3448

None of the drafts reopen or contradict #3448 — each is a **deeper, independent dive into a
sub-claim** or an orthogonal item, with first-hand evidence and a self-contained repro that #3448
did not carry. Suggested chaining:

| draft | #3448 relationship | overlap / boundary |
|---|---|---|
| **#1 L12L0A arity** | **extends** #3448 "What breaks #1" (the 2201 cube-intrinsic surface differs; MMAD family arch-gated). | #3448 named the *family* gap; #1 pins the **exact 9-argument V1 order** of `load_cbuf_to_ca`/`_cb` from the official `LoadData2DL12L0ACal` and shows a shorter arity only surfaces as `aicore 507015`. Disjoint from #3448's F1–F5 (those are about the 950 surface / plugin / bf16 proxy). |
| **#2 GM→UB gap semantics** | **extends** #3448 F4 (which first mentioned a `copy_gm_to_ubuf` adapter). | #3448 listed the adapter as an existence proof; #2 shows the native slots are **inter-burst gaps** (three CANN sources) and that the upstream planner forwards a pitch, with the on-card `K=48` failure as corroboration. Carries two explicit retractions — see below. |
| **#3 bf16 cast** | **refines** #3448 "What breaks #3" + F5 (bf16 proxy; `(float)` casts fail). | Same sub-claim, but #3 gives a **self-contained bisheng repro** and characterises the boundary precisely: the *cast* fails (exit 70) while a *pure* bf16 copy compiles and a bit-reinterpret workaround builds. Consistent with #3448's F5 "software conversion" note. |

**Direct claims that must stay consistent (checked):**

- #3448 F5 said the bf16 proxy "requires `const&` parameter passing + `__builtin_bit_cast`
  software conversion". Draft #3's variant F (reinterpret to `__gm__ uint16_t*`, `(uint32_t)h << 16`)
  compiles — i.e. it agrees that the plain `(float)` cast is the wall and a bit-level path works.
  No contradiction; #3 just localises the wall to the **cast** and adds the exit code.
- #3448 F4 named `copy_gm_to_ubuf` among the adapters it had to write. Draft #2 does not dispute
  that the adapter made the kernel *build*; it disputes that the adapter is *numerically correct*
  for `bk < k`. Consistent (compile vs numerics).
- #3448 "What breaks #3" said `bfloat16_t` is an opaque proxy with "deleted copy constructor".
  Draft #3's variant D shows a pure `bfloat16_t` copy (`o[0]=a[0]`) compiles on 2201 — a mild
  refinement worth noting if the thread is resumed, so the filed text's stronger "deleted copy
  ctor" wording is not quoted back as-is.

## Drafts still to clear before filing

- All three drafts are **staged only**; filing requires the decision-maker's approval and (for any
  follow-up in the #3448 thread) resolution of the current internal hold.
- Draft #2 is explicitly labelled **partially localised**; do not let it be filed or read as a
  closed verdict (two retractions live inside it).
- Draft #3 contains one first-hand finding that goes slightly beyond our own notes (the bf16 cast
  failure also reproduces on a `__cube__` face, not only `__vector__`); it is flagged inside #3
  rather than silently asserted. If the maintainer asks, we should be ready to re-run it live.
