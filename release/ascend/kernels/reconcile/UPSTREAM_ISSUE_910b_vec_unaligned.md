# Upstream issue draft — tilelang-ascend (tile-ai/tilelang-ascend, branch ascendc_pto)

> Status: DRAFT, not filed. Filed only after user approval (external actions stay frozen by default).
> Chinese ledger: release/runs/1013-p2-13-oncard-window9-ta-bwd/notes.md (window 9)

## Prior-art scan (2026-10-13, before filing — per user instruction)

- **#1307 [open, 2026-07-07] "Fix AscendC GetWithOffset allocation size alignment"** — same mechanism family: allocation SIZES not rounded to 32B. Our generated code contains `GetWithOffset<float>(1, 2400)` (4-byte capacity). Filing posture: **a new, chip-fatal trigger face of #1307** (aicore exception instead of garbage read-back), posted as a comment on #1307 or a linked issue — NOT a duplicate bug report.
- #1680 [closed 2026-10-08] column-offset BufferRegion flatten → 507015 — different trigger face (we use no column-offset slices); fix likely already in our build.
- #1641 / #1777 T.tile.broadcast GM-src silent wrong values — broadcast family but different symptom (silent wrong vs chip exception) and different src (GM-loaded).
- #1453 / #1537 compact unaligned copies — adjacent family, not our trigger.
- Pending on-card verification (next window): enlarging every sub-32B UB buffer to 32B (`[1]`→`[8]`) as a user-side workaround — **RESOLVED NEGATIVE (window 10, on-card): the aligned variant still fails with the identical 507015**, so the #1307 size-rounding gap is NOT our mechanism (worth stating upstream to rule the hypothesis out); our case stands as an independent defect, filed with references to the #1307/#1680 families.

---

## Title

[Bug] 910B (dav-2201): chip exception "The UB address accessed by the VEC instruction is not aligned" (507015, aicore) from j-axis broadcast in reduction-heavy loops — 19-point bisect matrix + generated-code diff

## Environment

- Device: Ascend 910B (dav-2201), CANN **8.5.2** (card) — kernels prebuilt in a CANN 8.5.0 container and run on-card (the window-5 GDN forward family runs green on the same package, so the ABI/interface is fine)
- tilelang-ascend @ `83b0ece6` (`ascendc_pto`), prebuilt runtime package (PYTHONPATH), pass_configs = the 5-key hybrid set (AUTO_CV_COMBINE / AUTO_CV_SYNC / AUTO_SYNC / MEMORY_PLANNING / TAIL_MASK, all True)
- torch 2.7.1 + torch_npu 2.7.1.post2, Python 3.12

## Symptom

A per-token GDN backward kernel (per-(b,h) block, `T.Kernel(B*H, threads=1)`, serial forward + serial reverse loop over L) dies at the FIRST launch with:

```
EZ9999: fftsplus aivector error, core id 24, vec error info: 0x400000056, mte error info: 0x6000077
errorStr: The UB address accessed by the VEC instruction is not aligned. subErrType:4
Kernel task happen error, retCode=0x26, [aicore exception]
rtStreamSynchronize execution failed ... runtime result = 507015
```

Reproduces at the SMALLEST shape (B=1,H=1,L=1,DK=8,DV=8), both fp32 and fp16, all six L0 cases, deterministically across launches. Shape, dtype, `out_idx`/`workspace_idx` configuration are all irrelevant (three-way config A/B cleared).

## Bisect matrix (19 on-card verdicts)

Probe: one LEVEL-gated kernel factory, each level compiles a pruned kernel; `PASS` = clean return, `FAIL` = 507015 above. Full probe + generated code attached.

| # | Form (inside `for t in T.serial(L)` + `T.Parallel(DV,DK)` bodies) | Verdict |
|---|---|---|
| 1 | length-1 region copy GM→UB `[1]` buffer | PASS |
| 2 | `T.tile.exp` on `[1]` buffer | PASS |
| 3 | `T.reduce_sum([1,8] → [1], dim=1)` | PASS |
| 4 | `T.tile.mul(dst2D, src2D, scalar_BufferLoad)` | PASS |
| 5 | 2D `reduce_sum` dim=1 / dim=0 in a flat kernel | PASS |
| 6 | GM scalar store inside `T.Scope("V")` | PASS |
| 7 | serial 64-trip copy+exp+mul density | PASS |
| 8 | fp16 staging 1D UB→UB cast copy | PASS |
| 9 | out_idx 6-vs-1, workspace auto-vs-manual (3 flips) | all irrelevant |
| 10 | 2D read + single cross-index add `lam2[j,i] = lam[j,i] + do[j]` | PASS |
| 11 | pure outer product `buf2[j,i] = do[j] * q[i]` | PASS |
| 12 | 2D read × single cross-index mul (i-axis bcast) `t[j,i] = lam2[j,i] * k[i]` + reduce dim=1 | PASS |
| 13 | i-axis bcast mul **+ reduce dim=0** | **FAIL** |
| 14 | j-axis bcast mul **+ reduce dim=1** | **FAIL** |
| 15 | bare `1D[j] × 2D` mul + reduce dim=0 (5 variants: dst buffer swap, store-target swap, parenthesised, +0.0, mirrored) | **all FAIL** |
| 16 | compound `(2D + 2D) * 1D[j]` + reduce dim=0 | **PASS** (the only immune sample) |
| 17 | same compound re-minimised (brackets / +0.0 / mirror) | **FAIL** (immunity does not transfer) |
| 18 | `T.tile.broadcast(u2, u1D)` + 2D×2D `tile.mul` + reduce dim=0 | **FAIL** |
| 19 | serial-j row accumulation (`for jj in T.serial(DV): for i in T.Parallel(DK): dk[i] += lam2[jj,i] * u[jj]`, both in-place and two-step variants) | **FAIL** |

The failing family reduces to: a 1-D vector contracted along the **first (j) Parallel axis** — i.e. `sum_j A[j,i] * v[j]` — regardless of how the broadcast is expressed (bare Parallel, compound, tile.broadcast, or serial-row accumulation). The lone PASS at #16 suggests the fault is in a specific lowering path, not in the semantics.

## Generated-code diff (PASS #16 vs FAIL #15-min, CANN 8.5.0 container compile)

For the j-broadcast the codegen inserts:

```cpp
auto do_ub_broadcast_0_data = ascend_ub.GetWithOffset<float>(64, 1728);   // extra [8,8] UB buffer
...
tl::ascend::Broadcast<float, 2, 1, false>(do_ub_broadcast_0_data[0], do_ub[0],
    broadcast_workspace_1_data[0], (uint32_t[]){8, 8}, (uint32_t[]){8, 1});
```

plus an extra `AscendC::PipeBarrier<PIPE_V>();`. The Broadcast scratch shifts every later buffer's UB offset (dv_ub 2368, scol_ub 2400, …) versus the PASS variant's layout (scol_ub 2112). We could not tie the fault to a visibly misaligned offset by hand-inspection (all sampled offsets are 32B-aligned), so we suspect the interaction between the auto-inserted Broadcast expansion and UB memory planning (or the barrier placement), rather than the source form.

## Minimal reproduction

`artifacts/w9_probe_v7_loop.py` (LEVEL factory; run `LEVEL=18` for the minimal FAIL leg — a ~30-line kernel: serial loop + one `buf2[j,i] = u[j] * lam2[j,i]` + `reduce_sum(dim=0)` + store). Driver at the bottom of the file; needs a real dav-2201 card. Generated code for L12 (PASS) / L18 / L23 (FAIL) in the same directory (`w9_gen_L*.cpp`).

## Ask

1. Is the j-axis (first Parallel axis) broadcast contracted via `reduce_sum(dim=0)` a known casualty of the #1307 size-alignment gap? Our repro would then serve as a chip-fatal variant for that PR's regression set (the existing repro is a garbage-read case).
2. Any recommended lowering for `dk_i = sum_j A[j,i] * v[j]` on 910B (vector path) that avoids the auto-Broadcast path? (GEMM/GEMV via `T.gemm_v0` is our planned fallback for the chunked version.)

---

Attachments (this repo): `release/runs/1013-p2-13-oncard-window9-ta-bwd/artifacts/{w9_probe_v7_loop.py, w9_gen_L12.cpp, w9_gen_L18.cpp, w9_gen_L23.cpp}` — trimmed copies would be attached at filing time.
- 2026-10-14：issue 已发布 → https://github.com/tile-ai/tilelang-ascend/issues/1885（用户授权；正文=21 点矩阵+生成码 diff+自足 v14 复现+workaround 证伪）
