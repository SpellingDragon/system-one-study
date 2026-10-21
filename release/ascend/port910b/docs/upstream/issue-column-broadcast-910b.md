# [Bug] 1D→2D column broadcast (`c[i,j] = b[j]`) silently uses j=0 on Ascend 910B2

**Status**: draft (not yet filed)
**Environment**
- Platform: Ascend 910B2 (A2A3), CANN 8.5.2
- Package: tilelang-ascend 0.1.4 (`ta_pkg`, VERSION=0.1.4) — its `src/transform/ascend_lower_parallel_to_vector.cc` is byte-identical to `ascendc_pto` HEAD
- Pass configs: `TL_ASCEND_AUTO_CV_COMBINE=True, TL_ASCEND_AUTO_CV_SYNC=True, TL_ASCEND_AUTO_SYNC=True` (+ 2 more, Developer mode)
- Codegen path: `codegen_ascend.cc` (AscendC), **not** PTO

## Summary

The documented 1D→2D column broadcast

```python
for (i, j) in T.Parallel(64, 64):
    c_ub[i, j] = b_ub[j] + 5.0     # b_ub is 1D, j is the SECOND dim
```

is promised by `docs/tutorials/t_parallel.md` §3.5 ("Mismatched Operand-Result
Dimensions") and `api-compute.md` (broadcast table) — but on 910B2 the `j` index
**collapses to 0**, i.e. every row becomes `b_ub[0] + 5`.

Row broadcast (`c[i,j] = b[i]`) works. Only the **second-dimension** index degrades.

## Minimal reproduction

```python
@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def bcast_probe():
    @T.prim_func
    def main(B: T.Tensor([64], "float"), C: T.Tensor([64, 64], "float")):
        with T.Kernel(1, threads=1, is_npu=True) as (cid):
            b_ub = T.alloc_ub([64], "float")
            c_ub = T.alloc_ub([64, 64], "float")
            T.copy(B[0], b_ub)
            for (i, j) in T.Parallel(64, 64):
                c_ub[i, j] = b_ub[j] + 5.0
            T.copy(c_ub, C[0, 0])
    return main

b = torch.arange(64, dtype=torch.float32)          # b[j] = j
out = bcast_probe()(b.npu()).float().cpu().numpy()  # expected C[i,j] = j + 5
```

**Actual**: every column equals `b[0] + 5 = 5` ⇒ `max_abs_err = 63`.
**Expected**: `C[i,j] = j + 5`.

## Evidence it is the same lowering family as #1194 / #1219

- `#1219` / `f6b0b921` ("Lower un-vectorizable T.Parallel to serial") is **present**
  in the package we run (file diffed line-by-line), yet the pattern above still
  degrades ⇒ the vectorisation **accepts** this loop and then emits a broadcast
  whose source index is 0. It is not the "should have fallen back to serial" case.
- `#1194` was also reported on **Ascend 910B2**, so 910B's `T.Parallel` lowering is
  known to be incomplete.
- The same defect was found in production through a Gamma matrix
  `Gam[i,j] = exp(g_i - g_j)`: the card produced `exp(g_i - g_0)` — row 10 had all
  columns equal, diagonal was `[1.0, 0.6452, 0.4969, ...]` instead of all 1.

## Suggested fix direction

In `ascend_lower_parallel_to_vector.cc`, when the RHS indexes a 1D buffer with the
**second** loop var, either
1. materialise it as a `[1, N]` view and emit `T.tile.broadcast` (the shape the
   low-level API actually requires — see `troubleshooting.md`: "broadcast 源必须
   是 2D `[M,1]` 或 `[1,N]`"), or
2. treat the pattern as un-vectorisable and lower to serial (the #1219 path), or
3. reject it with a clear diagnostic instead of producing wrong numbers.

## Impact

Any kernel computing `X[i,j] ± y[j]` (column-wise coefficient, e.g. cumsum-based
gates, positional scaling, attention decay matrices) silently produces
column-invariant results. Workaround in use: both loop vars serial
(`for i in T.serial(C): for j in T.serial(C)`), which costs AIV scalar time.

## Workaround

```python
for i in T.serial(C):          # S7g idiom (serial/serial)
    for j in T.serial(C):
        aux[i, j] = g_ub[i] - g_ub[j]
```

Note: `for i in T.serial(C): for j in T.Parallel(C)` raises **507015**
(serial-outer/parallel-inner), consistent with `api-compute.md` recommending
`range()` (python unroll) rather than `T.serial` for the outer loop.
