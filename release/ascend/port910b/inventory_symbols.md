# inventory_symbols.md — 逐符号附表

> **自动生成，勿手改**；生成命令 `./checklist.sh symbols`。
> 分类规则 = checklist.sh 的 `classify()`；class 语义见 INVENTORY.md §1.2，
> 各分组（A1..D4）的处置方案叙述在 INVENTORY.md §3。改判决请改 classify() 后重跑。

举证路径相对 tilelang 主仓根；`命中` = 该符号在主仓内的出现次数。

## A 类逐符号

| class | 组 | 符号 | 族 | 命中 | 首处举证 文件:行 | 出现文件 |
|---|---|---|---|---|---|---|
| A | A6-SCAN-ARTIFACT | `::print_buffer` | F11 | 1 | `src/tl_templates/ascend/debug.h:266` | debug.h |
| A | A6-SCAN-ARTIFACT | `::print_var` | F11 | 1 | `src/tl_templates/ascend/debug.h:260` | debug.h |
| A | A6-SCAN-ARTIFACT | `asc_aicore` | F6 | 3 | `src/tl_templates/ascend/debug.h:16` | debug.h |
| A | A6-SCAN-ARTIFACT | `asc_bf16` | F6 | 1 | `src/tl_templates/ascend/common.h:12` | common.h |
| A | A6-SCAN-ARTIFACT | `asc_fp16` | F6 | 2 | `src/tl_templates/ascend/ascend_fp8.h:5` | ascend_fp8.h, common.h |
| A | A6-SCAN-ARTIFACT | `asc_fp8` | F6 | 2 | `src/tl_templates/ascend/ascend_fp8.h:6` | ascend_fp8.h, common.h |
| A | A6-SCAN-ARTIFACT | `asc_fp8_interpretation_t` | F6 | 1 | `src/ascend/target_utils.cc:20` | target_utils.cc |
| A | A6-SCAN-ARTIFACT | `asc_loadalign_v2_impl` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:193` | simd_inst.h |
| A | A6-SCAN-ARTIFACT | `asc_pad_val` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:1524` | codegen_ascend.cc |
| A | A6-SCAN-ARTIFACT | `asc_rng_state` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:2545` | codegen_ascend.cc |
| A | A6-SCAN-ARTIFACT | `asc_simd` | F6 | 4 | `src/tl_templates/ascend/common.h:11` | common.h, debug.h, gemm.h, simd_inst.h |
| A | A6-SCAN-ARTIFACT | `asc_simt` | F6 | 3 | `src/tl_templates/ascend/common.h:15` | common.h, random_kernel_base.h, reduce.h |
| A | A6-SCAN-ARTIFACT | `asc_simt_vf` | F6 | 3 | `src/tl_templates/ascend/debug.h:16` | debug.h |
| A | A6-SCAN-ARTIFACT | `asc_target` | F6 | 2 | `tilelang/ascend/target.py:78` | target.py |
| A | A4-BITCAST | `bit>` | F9 | 1 | `src/tl_templates/ascend/numeric_limits.h:3` | numeric_limits.h |
| A | A3-BLOCKIDX-0 | `block_idx` | F3 | 47 | `src/ascend/codegen/codegen_ascend.cc:546` | codegen_ascend.cc, codegen_pto.cc, debug.h, tile_schedule.py |
| A | A1-HEAD-STUB | `c_api/asc_simd.h` | F1 | 4 | `src/tl_templates/ascend/common.h:11` | common.h, debug.h, gemm.h, simd_inst.h |
| A | A1-HEAD-STUB | `simt_api/asc_bf16.h` | F1 | 1 | `src/tl_templates/ascend/common.h:12` | common.h |
| A | A1-HEAD-STUB | `simt_api/asc_fp16.h` | F1 | 2 | `src/tl_templates/ascend/ascend_fp8.h:5` | ascend_fp8.h, common.h |
| A | A1-HEAD-STUB | `simt_api/asc_fp8.h` | F1 | 2 | `src/tl_templates/ascend/ascend_fp8.h:6` | ascend_fp8.h, common.h |
| A | A1-HEAD-STUB | `simt_api/asc_simt.h` | F1 | 3 | `src/tl_templates/ascend/common.h:15` | common.h, random_kernel_base.h, reduce.h |
| A | A4-BITCAST | `std::bit_cast` | F9 | 2 | `src/tl_templates/ascend/numeric_limits.h:6` | numeric_limits.h |
| A | A2-ATTR-ALIAS | `__aicore__` | F2 | 29 | `src/tl_templates/ascend/dcache_bypass.h:14` | dcache_bypass.h, debug.h, gemm.h, philox_rng.h, random_kernel_base.h |
| A | A6-SCAN-ARTIFACT | `__asc_aicore` | F7 | 3 | `src/tl_templates/ascend/debug.h:16` | debug.h |
| A | A6-SCAN-ARTIFACT | `__asc_fp8_interpretation_t` | F7 | 1 | `src/ascend/target_utils.cc:20` | target_utils.cc |
| A | A6-SCAN-ARTIFACT | `__asc_pad_val` | F7 | 1 | `src/ascend/codegen/codegen_ascend.cc:1524` | codegen_ascend.cc |
| A | A6-SCAN-ARTIFACT | `__asc_rng_state` | F7 | 1 | `src/ascend/codegen/codegen_ascend.cc:2545` | codegen_ascend.cc |
| A | A6-SCAN-ARTIFACT | `__asc_simt_vf` | F7 | 3 | `src/tl_templates/ascend/debug.h:16` | debug.h |
| A | A4-BITCAST | `__builtin_bit_cast` | F9 | 1 | `src/tl_templates/ascend/numeric_limits.h:8` | numeric_limits.h |
| A | A7-ATTR-PROVEN | `__ca__` | F2 | 10 | `src/ascend/codegen/codegen_ascend.cc:156` | codegen_ascend.cc, gemm.h |
| A | A7-ATTR-PROVEN | `__cbuf__` | F2 | 17 | `src/ascend/codegen/codegen_ascend.cc:162` | codegen_ascend.cc, gemm.h |
| A | A7-ATTR-PROVEN | `__cb__` | F2 | 12 | `src/ascend/codegen/codegen_ascend.cc:158` | codegen_ascend.cc, gemm.h |
| A | A7-ATTR-PROVEN | `__cc__` | F2 | 3 | `src/ascend/codegen/codegen_ascend.cc:160` | codegen_ascend.cc, gemm.h |
| A | A7-ATTR-PROVEN | `__gm__` | F2 | 75 | `src/ascend/codegen/codegen_ascend.cc:1409` | codegen_ascend.cc, codegen_pto.cc, dcache_bypass.h, debug.h |
| A | A7-ATTR-PROVEN | `__ubuf__` | F2 | 86 | `src/ascend/codegen/codegen_ascend.cc:154` | codegen_ascend.cc, codegen_pto.cc, nd2nz_copy.h, reduce.h, simd_inst.h |

## B 类逐符号

| class | 组 | 符号 | 族 | 命中 | 首处举证 文件:行 | 出现文件 |
|---|---|---|---|---|---|---|
| B | B3-LOCKMODE | `ASC_LOCK_BLOCK` | F8 | 20 | `src/ascend/codegen/codegen_ascend.cc:1855` | codegen_ascend.cc, rewrite_flag_to_buf.cc, gemm.h |
| B | B3-LOCKMODE | `ASC_LOCK_NON` | F8 | 4 | `src/ascend/codegen/codegen_ascend.cc:1856` | codegen_ascend.cc, rewrite_flag_to_buf.cc |
| B | B1-TYPE | `bfloat16x2_t` | F5 | 10 | `src/ascend/codegen/codegen_ascend.cc:3419` | codegen_ascend.cc, ascend_fp8.h |
| B | B1-TYPE | `bfloat16_t` | F5 | 36 | `src/ascend/codegen/codegen_ascend.cc:249` | codegen_ascend.cc, codegen_pto.cc, memory_detector.h, estimate_latency.cc, insert_nd2nz.cc, debug.h, gemm.h, nd2nz_copy.h, numeric_limits.h, simd_inst.h, mode.py, gemm_mad.py |
| B | B1-TYPE | `float2` | F5 | 55 | `src/ascend/codegen/codegen_ascend.cc:1013` | codegen_ascend.cc, ascend_fp8.h, philox_rng.h |
| B | B1-TYPE | `float4` | F5 | 56 | `src/ascend/codegen/codegen_ascend.cc:206` | codegen_ascend.cc, codegen_pto.cc, ascend_mte_plan.h, copy.cc, estimate_latency.cc, rewrite_fp4_to_fp4x2.cc, philox_rng.h, simd.py, gemm_mad.py, pipeline.py, __init__.py |
| B | B1-TYPE | `half` | F5 | 77 | `src/ascend/codegen/codegen_ascend.cc:247` | codegen_ascend.cc, codegen_pto.cc, builtin.h, ascend_simdvf_lower_parallel.cc, memory_detector.h, estimate_latency.cc, insert_nd2nz.cc, normalize_conflict_hints.cc, rewrite_dual_copy.cc, thread_storage_sync.cc, debug.h, nd2nz_copy.h, numeric_limits.h, simd_inst.h, copy_op.py, mode.py, schedule_hint.py, simd.py, gemm_mad.py |
| B | B1-TYPE | `half2` | F5 | 24 | `src/ascend/codegen/codegen_ascend.cc:3403` | codegen_ascend.cc, ascend_fp8.h |
| B | B1-TYPE | `half_t` | F5 | 1 | `src/tl_templates/ascend/gemm.h:36` | gemm.h |
| B | B1-TYPE | `make_float2` | F7 | 4 | `src/ascend/codegen/codegen_ascend.cc:1078` | codegen_ascend.cc, philox_rng.h |
| B | B1-TYPE | `make_float4` | F7 | 3 | `src/ascend/codegen/codegen_ascend.cc:3367` | codegen_ascend.cc, philox_rng.h |
| B | B1-TYPE | `make_int2` | F7 | 1 | `src/ascend/codegen/codegen_ascend.cc:2724` | codegen_ascend.cc |
| B | B1-TYPE | `make_longlong4` | F7 | 1 | `src/ascend/codegen/codegen_ascend.cc:1063` | codegen_ascend.cc |
| B | B1-TYPE | `make_ulonglong4` | F7 | 2 | `src/ascend/codegen/codegen_ascend.cc:1060` | codegen_ascend.cc |
| B | B2-CVT | `__bfloat1622float2` | F7 | 3 | `src/ascend/codegen/codegen_ascend.cc:3419` | codegen_ascend.cc, ascend_fp8.h |
| B | B2-CVT | `__bfloat162float` | F7 | 1 | `src/ascend/codegen/codegen_ascend.cc:3333` | codegen_ascend.cc |
| B | B2-CVT | `__float22bfloat162_rn` | F7 | 3 | `src/ascend/codegen/codegen_ascend.cc:3428` | codegen_ascend.cc, ascend_fp8.h |
| B | B2-CVT | `__float22half2_rn` | F7 | 3 | `src/ascend/codegen/codegen_ascend.cc:3411` | codegen_ascend.cc, ascend_fp8.h |
| B | B2-CVT | `__half22float2` | F7 | 3 | `src/ascend/codegen/codegen_ascend.cc:3403` | codegen_ascend.cc, ascend_fp8.h |
| B | B4-DECL-MACRO | `__SIMT_DEVICE_FUNCTIONS_DECL__` | F2 | 1 | `src/tl_templates/ascend/ascend_fp8.h:9` | ascend_fp8.h |

## C 类逐符号

| class | 组 | 符号 | 族 | 命中 | 首处举证 文件:行 | 出现文件 |
|---|---|---|---|---|---|---|
| C | C9-RNG950 | `::ALG_COUNTER_SIZE` | F11 | 4 | `src/tl_templates/ascend/philox_rng.h:19` | philox_rng.h |
| C | C9-RNG950 | `::ALG_KEY_SIZE` | F11 | 1 | `src/tl_templates/ascend/philox_rng.h:18` | philox_rng.h |
| C | C3-SIMD-TAG | `::asc_mem_bar` | F11 | 1 | `src/tl_templates/ascend/simd_inst.h:1047` | simd_inst.h |
| C | C9-RNG950 | `::BoxMullerFloat` | F11 | 2 | `src/tl_templates/ascend/philox_rng.h:80` | philox_rng.h |
| C | C3-SIMD-TAG | `::vcvt` | F11 | 5 | `src/ascend/codegen/codegen_ascend.cc:2474` | codegen_ascend.cc, nd2nz_copy.h, simd_inst.h |
| C | C3-SIMD-TAG | `::vpack` | F11 | 2 | `src/ascend/codegen/codegen_ascend.cc:2253` | codegen_ascend.cc, nd2nz_copy.h |
| C | C3-SIMD-TAG | `::vsstb` | F11 | 6 | `src/ascend/codegen/codegen_ascend.cc:2433` | codegen_ascend.cc, nd2nz_copy.h |
| C | C9-RNG950 | `ALG_COUNTER_SIZE` | F11 | 8 | `src/tl_templates/ascend/random_kernel_base.h:28` | random_kernel_base.h |
| C | C9-RNG950 | `ALG_KEY_SIZE` | F11 | 3 | `src/tl_templates/ascend/random_kernel_base.h:27` | random_kernel_base.h |
| C | C2-SIMD-REG | `AscendSimdVFLowerParallel` | F10 | 15 | `src/ascend/transform/ascend_simdvf_lower_parallel.cc:29` | ascend_simdvf_lower_parallel.cc, layout_inference.cc, reduce_op.py, pipeline.py, __init__.py |
| C | C4-MX-FP8 | `ascend_mad_mx` | F10 | 18 | `src/ascend/codegen/codegen_ascend.cc:583` | codegen_ascend.cc, codegen_pto.cc, builtin.cc, builtin.h, ascend_pipe.h, gemm_mad_blockscaled.py |
| C | C2-SIMD-REG | `asc_abs` | F6 | 4 | `src/tl_templates/ascend/simd_inst.h:479` | simd_inst.h |
| C | C2-SIMD-REG | `asc_abs_sub` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:344` | simd_inst.h |
| C | C1-SIMT | `asc_activemask` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:1845` | codegen_ascend.cc |
| C | C2-SIMD-REG | `asc_add` | F6 | 2 | `src/tl_templates/ascend/simd_inst.h:333` | simd_inst.h |
| C | C2-SIMD-REG | `asc_addc` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:379` | simd_inst.h |
| C | C2-SIMD-REG | `asc_add_scalar` | F6 | 4 | `src/tl_templates/ascend/simd_inst.h:346` | simd_inst.h |
| C | C2-SIMD-REG | `asc_and` | F6 | 3 | `src/tl_templates/ascend/simd_inst.h:257` | simd_inst.h |
| C | C2-SIMD-REG | `asc_arange` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:855` | simd_inst.h |
| C | C2-SIMD-REG | `asc_arange_descend` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:857` | simd_inst.h |
| C | C1-SIMT | `asc_atomic_` | F6 | 2 | `src/ascend/codegen/codegen_ascend.cc:2502` | codegen_ascend.cc |
| C | C2-SIMD-REG | `asc_axpy` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:420` | simd_inst.h |
| C | C1-SIMT | `asc_ballot` | F6 | 2 | `src/ascend/codegen/codegen_ascend.cc:1833` | codegen_ascend.cc |
| C | C6-ND2NZ-DUAL | `asc_copy_gm2l1_dn2nz` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:1458` | codegen_ascend.cc |
| C | C6-ND2NZ-DUAL | `asc_copy_gm2l1_nd2nz` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:1458` | codegen_ascend.cc |
| C | C6-ND2NZ-DUAL | `asc_copy_l0c2gm` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:1499` | codegen_ascend.cc |
| C | C4-MX-FP8 | `asc_copy_l12l0a_mx` | F6 | 5 | `src/ascend/codegen/codegen_ascend.cc:1586` | codegen_ascend.cc, builtin.h, utils.h, gemm.h |
| C | C4-MX-FP8 | `asc_copy_l12l0b_mx` | F6 | 3 | `src/ascend/codegen/codegen_ascend.cc:1586` | codegen_ascend.cc, gemm.h |
| C | C6-ND2NZ-DUAL | `asc_copy_ub2l1` | F6 | 4 | `src/ascend/codegen/codegen_ascend.cc:1601` | codegen_ascend.cc, nd2nz_copy.h |
| C | C2-SIMD-REG | `asc_create_mask_b` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:3179` | codegen_ascend.cc |
| C | C2-SIMD-REG | `asc_create_mask_b16` | F6 | 4 | `src/tl_templates/ascend/nd2nz_copy.h:101` | nd2nz_copy.h, simd_inst.h, simd.py |
| C | C2-SIMD-REG | `asc_create_mask_b32` | F6 | 3 | `src/tl_templates/ascend/nd2nz_copy.h:100` | nd2nz_copy.h, simd_inst.h |
| C | C2-SIMD-REG | `asc_create_mask_b8` | F6 | 3 | `src/tl_templates/ascend/simd_inst.h:95` | simd_inst.h, simd.py |
| C | C2-SIMD-REG | `asc_cumulative_histogram_bin0` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:841` | simd_inst.h |
| C | C2-SIMD-REG | `asc_cumulative_histogram_bin1` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:843` | simd_inst.h |
| C | C4-MX-FP8 | `asc_cvt_float2_to_fp8x2` | F6 | 8 | `src/ascend/codegen/codegen_ascend.cc:3442` | codegen_ascend.cc, ascend_fp8.h |
| C | C2-SIMD-REG | `asc_deintlv` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:301` | simd_inst.h |
| C | C2-SIMD-REG | `asc_deintlv_` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:818` | simd_inst.h |
| C | C5-AIC-MODE | `asc_disable_hf32` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:1877` | codegen_ascend.cc |
| C | C2-SIMD-REG | `asc_div` | F6 | 3 | `src/tl_templates/ascend/simd_inst.h:336` | simd_inst.h |
| C | C5-AIC-MODE | `asc_dual_dst_mode` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:1483` | codegen_ascend.cc |
| C | C2-SIMD-REG | `asc_duplicate` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:910` | simd_inst.h |
| C | C2-SIMD-REG | `asc_duplicate_highest` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:912` | simd_inst.h |
| C | C2-SIMD-REG | `asc_duplicate_scalar` | F6 | 5 | `src/tl_templates/ascend/simd_inst.h:449` | simd_inst.h |
| C | C5-AIC-MODE | `asc_enable_hf32` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:1879` | codegen_ascend.cc |
| C | C2-SIMD-REG | `asc_eq_scalar` | F6 | 3 | `src/tl_templates/ascend/simd_inst.h:460` | simd_inst.h |
| C | C2-SIMD-REG | `asc_exp` | F6 | 3 | `src/tl_templates/ascend/simd_inst.h:529` | simd_inst.h |
| C | C2-SIMD-REG | `asc_exp_sub` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:950` | simd_inst.h |
| C | C6-ND2NZ-DUAL | `asc_fill_l1` | F6 | 6 | `src/ascend/codegen/codegen_ascend.cc:1549` | codegen_ascend.cc, builtin.h, oob_padding.h, insert_oob_padding.cc, pipeline.py, __init__.py |
| C | C4-MX-FP8 | `asc_fp8x2_storage_t` | F6 | 4 | `src/tl_templates/ascend/ascend_fp8.h:17` | ascend_fp8.h |
| C | C2-SIMD-REG | `asc_frequency_histogram_bin0` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:831` | simd_inst.h |
| C | C2-SIMD-REG | `asc_frequency_histogram_bin1` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:833` | simd_inst.h |
| C | C2-SIMD-REG | `asc_gather` | F6 | 4 | `src/tl_templates/ascend/simd_inst.h:228` | simd_inst.h |
| C | C2-SIMD-REG | `asc_gather_datablock` | F6 | 2 | `src/tl_templates/ascend/simd_inst.h:212` | simd_inst.h |
| C | C2-SIMD-REG | `asc_get_sub_block_id` | F3,F6 | 5 | `src/ascend/codegen/codegen_ascend.cc:887` | codegen_ascend.cc, ir.cc, frame.py, kernel.py |
| C | C2-SIMD-REG | `asc_ge_scalar` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:462` | simd_inst.h |
| C | C2-SIMD-REG | `asc_gt_scalar` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:609` | simd_inst.h |
| C | C5-AIC-MODE | `asc_hf32_round_mode` | F6 | 2 | `src/ascend/codegen/codegen_ascend.cc:1882` | codegen_ascend.cc |
| C | C2-SIMD-REG | `asc_intlv` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:308` | simd_inst.h |
| C | C2-SIMD-REG | `asc_intlv_` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:812` | simd_inst.h |
| C | C7-MIXKERN | `ASC_IS_AIC` | F3 | 1 | `src/ascend/codegen/codegen_ascend.cc:869` | codegen_ascend.cc |
| C | C1-SIMT | `ASC_IS_AIV` | F3 | 1 | `src/ascend/codegen/codegen_ascend.cc:883` | codegen_ascend.cc |
| C | C2-SIMD-REG | `asc_leakyrelu` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:537` | simd_inst.h |
| C | C2-SIMD-REG | `asc_le_scalar` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:577` | simd_inst.h |
| C | C2-SIMD-REG | `asc_ln` | F6 | 3 | `src/tl_templates/ascend/simd_inst.h:523` | simd_inst.h |
| C | C2-SIMD-REG | `asc_loadalign` | F6 | 3 | `src/tl_templates/ascend/simd_inst.h:146` | simd_inst.h |
| C | C2-SIMD-REG | `asc_loadalign_brc_datablock` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:152` | simd_inst.h |
| C | C2-SIMD-REG | `asc_loadalign_brc_elem` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:147` | simd_inst.h |
| C | C2-SIMD-REG | `asc_loadalign_brc_elem2datablock` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:153` | simd_inst.h |
| C | C2-SIMD-REG | `asc_loadalign_deintlv` | F6 | 2 | `src/tl_templates/ascend/simd_inst.h:197` | simd_inst.h |
| C | C2-SIMD-REG | `asc_loadalign_downsample` | F6 | 2 | `src/tl_templates/ascend/simd_inst.h:149` | simd_inst.h |
| C | C2-SIMD-REG | `asc_loadalign_unpack` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:150` | simd_inst.h |
| C | C2-SIMD-REG | `asc_loadalign_unpack4` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:151` | simd_inst.h |
| C | C2-SIMD-REG | `asc_loadalign_upsample` | F6 | 2 | `src/tl_templates/ascend/simd_inst.h:148` | simd_inst.h |
| C | C5-AIC-MODE | `asc_load_l2_cache_mode` | F6 | 2 | `src/ascend/codegen/codegen_ascend.cc:1415` | codegen_ascend.cc |
| C | C2-SIMD-REG | `asc_lt` | F6 | 2 | `src/tl_templates/ascend/simd_inst.h:484` | simd_inst.h |
| C | C2-SIMD-REG | `asc_lt_scalar` | F6 | 2 | `src/tl_templates/ascend/simd_inst.h:608` | simd_inst.h |
| C | C2-SIMD-REG | `asc_madd` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:659` | simd_inst.h |
| C | C2-SIMD-REG | `asc_max` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:337` | simd_inst.h |
| C | C2-SIMD-REG | `asc_max_scalar` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:347` | simd_inst.h |
| C | C2-SIMD-REG | `asc_mem_bar` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:1047` | simd_inst.h |
| C | C2-SIMD-REG | `asc_min` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:338` | simd_inst.h |
| C | C2-SIMD-REG | `asc_min_scalar` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:348` | simd_inst.h |
| C | C4-MX-FP8 | `asc_mmad_mx` | F6 | 4 | `src/ascend/codegen/codegen_ascend.cc:1656` | codegen_ascend.cc, gemm.h, gemm_mad_blockscaled.py |
| C | C2-SIMD-REG | `asc_mul` | F6 | 4 | `src/tl_templates/ascend/simd_inst.h:335` | simd_inst.h |
| C | C2-SIMD-REG | `asc_mull` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:399` | simd_inst.h |
| C | C2-SIMD-REG | `asc_mul_scalar` | F6 | 10 | `src/tl_templates/ascend/simd_inst.h:349` | simd_inst.h |
| C | C2-SIMD-REG | `asc_neg` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:526` | simd_inst.h |
| C | C2-SIMD-REG | `asc_not` | F6 | 2 | `src/tl_templates/ascend/simd_inst.h:277` | simd_inst.h |
| C | C2-SIMD-REG | `asc_or` | F6 | 5 | `src/tl_templates/ascend/simd_inst.h:264` | simd_inst.h |
| C | C2-SIMD-REG | `asc_pack_to_high` | F6 | 2 | `src/tl_templates/ascend/simd_inst.h:790` | simd_inst.h |
| C | C2-SIMD-REG | `asc_pack_to_low` | F6 | 2 | `src/tl_templates/ascend/simd_inst.h:792` | simd_inst.h |
| C | C2-SIMD-REG | `asc_pair_reduce_sum` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:683` | simd_inst.h |
| C | C2-SIMD-REG | `asc_prelu` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:545` | simd_inst.h |
| C | C5-AIC-MODE | `asc_quant_mode` | F6 | 2 | `src/ascend/codegen/codegen_ascend.cc:1485` | codegen_ascend.cc |
| C | C1-SIMT | `asc_reduce_add` | F6 | 5 | `src/ascend/codegen/codegen_ascend.cc:1948` | codegen_ascend.cc, reduce.h |
| C | C1-SIMT | `asc_reduce_max` | F6 | 6 | `src/ascend/codegen/codegen_ascend.cc:1949` | codegen_ascend.cc, reduce.h, simd_inst.h |
| C | C2-SIMD-REG | `asc_reduce_max_datablock` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:747` | simd_inst.h |
| C | C1-SIMT | `asc_reduce_min` | F6 | 6 | `src/ascend/codegen/codegen_ascend.cc:1950` | codegen_ascend.cc, reduce.h, simd_inst.h |
| C | C2-SIMD-REG | `asc_reduce_min_datablock` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:754` | simd_inst.h |
| C | C2-SIMD-REG | `asc_reduce_sum` | F6 | 2 | `src/tl_templates/ascend/simd_inst.h:691` | simd_inst.h |
| C | C2-SIMD-REG | `asc_reduce_sum_datablock` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:740` | simd_inst.h |
| C | C2-SIMD-REG | `asc_relu` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:527` | simd_inst.h |
| C | C5-AIC-MODE | `asc_relu_pre_mode` | F6 | 2 | `src/ascend/codegen/codegen_ascend.cc:1486` | codegen_ascend.cc |
| C | C2-SIMD-REG | `asc_scatter` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:251` | simd_inst.h |
| C | C2-SIMD-REG | `asc_select` | F6 | 13 | `src/tl_templates/ascend/simd_inst.h:87` | simd_inst.h |
| C | C5-AIC-MODE | `asc_set_atomic_add` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:1899` | codegen_ascend.cc |
| C | C5-AIC-MODE | `asc_set_atomic_max` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:1897` | codegen_ascend.cc |
| C | C5-AIC-MODE | `asc_set_atomic_min` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:1898` | codegen_ascend.cc |
| C | C5-AIC-MODE | `asc_set_atomic_none` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:1935` | codegen_ascend.cc |
| C | C6-ND2NZ-DUAL | `asc_set_copy_pad_val` | F6 | 3 | `src/ascend/codegen/codegen_ascend.cc:1530` | codegen_ascend.cc, copy.cc, dma.py |
| C | C6-ND2NZ-DUAL | `asc_set_gm2l1_nz_para` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:1443` | codegen_ascend.cc |
| C | C5-AIC-MODE | `asc_set_hf32_round_mode` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:1881` | codegen_ascend.cc |
| C | C6-ND2NZ-DUAL | `asc_set_l0c_copy_nz_para` | F6 | 2 | `src/ascend/codegen/codegen_ascend.cc:1474` | codegen_ascend.cc |
| C | C5-AIC-MODE | `asc_set_mmad_direction_` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:1892` | codegen_ascend.cc |
| C | C1-SIMT | `asc_shfl_xor` | F6 | 7 | `src/ascend/op/reduce.cc:28` | reduce.cc, reduce.h |
| C | C2-SIMD-REG | `asc_shiftleft` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:342` | simd_inst.h |
| C | C2-SIMD-REG | `asc_shiftleft_scalar` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:350` | simd_inst.h |
| C | C2-SIMD-REG | `asc_shiftright` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:343` | simd_inst.h |
| C | C2-SIMD-REG | `asc_shiftright_scalar` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:351` | simd_inst.h |
| C | C2-SIMD-REG | `asc_sqrt` | F6 | 2 | `src/tl_templates/ascend/simd_inst.h:524` | simd_inst.h |
| C | C2-SIMD-REG | `asc_squeeze` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:761` | simd_inst.h |
| C | C2-SIMD-REG | `asc_storealign` | F6 | 5 | `src/tl_templates/ascend/simd_inst.h:181` | simd_inst.h |
| C | C2-SIMD-REG | `asc_storealign_1st` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:1004` | simd_inst.h |
| C | C2-SIMD-REG | `asc_storealign_intlv` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:1043` | simd_inst.h |
| C | C2-SIMD-REG | `asc_storealign_pack` | F6 | 3 | `src/tl_templates/ascend/simd_inst.h:189` | simd_inst.h |
| C | C2-SIMD-REG | `asc_storealign_pack_quarter` | F6 | 2 | `src/tl_templates/ascend/simd_inst.h:1028` | simd_inst.h |
| C | C2-SIMD-REG | `asc_storealign_postupdate` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:987` | simd_inst.h |
| C | C5-AIC-MODE | `asc_store_l2_cache_mode` | F6 | 2 | `src/ascend/codegen/codegen_ascend.cc:1430` | codegen_ascend.cc |
| C | C2-SIMD-REG | `asc_sub` | F6 | 2 | `src/tl_templates/ascend/simd_inst.h:334` | simd_inst.h |
| C | C2-SIMD-REG | `asc_subc` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:389` | simd_inst.h |
| C | C1-SIMT | `asc_syncthreads` | F6 | 8 | `src/ascend/codegen/codegen_ascend.cc:3557` | codegen_ascend.cc, reduce.h |
| C | C8-EVENT | `asc_sync_` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:1379` | codegen_ascend.cc |
| C | C8-EVENT | `asc_sync_notify` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:1361` | codegen_ascend.cc |
| C | C8-EVENT | `asc_sync_pipe` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:1345` | codegen_ascend.cc |
| C | C8-EVENT | `asc_sync_wait` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:1361` | codegen_ascend.cc |
| C | C1-SIMT | `asc_threadfence` | F6 | 2 | `src/ascend/codegen/codegen_ascend.cc:1814` | codegen_ascend.cc, sync.py |
| C | C5-AIC-MODE | `asc_unit_flag_mode` | F6 | 2 | `src/ascend/codegen/codegen_ascend.cc:1484` | codegen_ascend.cc |
| C | C2-SIMD-REG | `asc_unpack_lower` | F6 | 2 | `src/tl_templates/ascend/simd_inst.h:803` | simd_inst.h |
| C | C2-SIMD-REG | `asc_unpack_upper` | F6 | 2 | `src/tl_templates/ascend/simd_inst.h:801` | simd_inst.h |
| C | C2-SIMD-REG | `asc_unsqueeze` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:770` | simd_inst.h |
| C | C1-SIMT | `asc_update_addr_reg_b` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:2378` | codegen_ascend.cc |
| C | C2-SIMD-REG | `asc_update_mask_b16` | F6 | 2 | `src/tl_templates/ascend/nd2nz_copy.h:69` | nd2nz_copy.h, simd_inst.h |
| C | C2-SIMD-REG | `asc_update_mask_b32` | F6 | 2 | `src/tl_templates/ascend/nd2nz_copy.h:71` | nd2nz_copy.h, simd_inst.h |
| C | C2-SIMD-REG | `asc_update_mask_b8` | F6 | 1 | `src/tl_templates/ascend/simd_inst.h:775` | simd_inst.h |
| C | C1-SIMT | `asc_vf_call` | F6 | 3 | `src/ascend/codegen/codegen_ascend.cc:728` | codegen_ascend.cc, auto_schedule.cc |
| C | C2-SIMD-REG | `asc_xor` | F6 | 2 | `src/tl_templates/ascend/simd_inst.h:271` | simd_inst.h |
| C | C3-SIMD-TAG | `Bin_N0` | F8 | 2 | `src/tl_templates/ascend/simd_inst.h:830` | simd_inst.h |
| C | C1-SIMT | `blockIdx.` | F3 | 13 | `src/ascend/codegen/codegen_ascend.cc:3018` | codegen_ascend.cc, codegen_pto.cc, ir.cc, thread_storage_sync.cc, debug.h, kernel.py, pipeline.py |
| C | C4-MX-FP8 | `blockscaled` | F10 | 123 | `src/ascend/codegen/codegen_ascend.cc:579` | codegen_ascend.cc, codegen_pto.cc, codegen_pto.h, ascend_layouts.cc, builtin.cc, builtin.h, copy.cc, gemm_blockscaled.cc, ascend_pipe.h, estimate_latency.cc, insert_oob_padding.cc, layout_inference.cc, lower_tile_op.cc, rewrite_flag_to_buf.cc, gemm.h, __init__.py, allocate.py, gemm_op.py, gemm_mad.py, gemm_mad_blockscaled.py |
| C | C4-MX-FP8 | `cast_float_to_fp8_e4m3` | F11 | 3 | `src/tl_templates/ascend/ascend_fp8.h:15` | ascend_fp8.h |
| C | C4-MX-FP8 | `cast_float_to_fp8_e5m2` | F11 | 3 | `src/tl_templates/ascend/ascend_fp8.h:22` | ascend_fp8.h |
| C | C4-MX-FP8 | `cast_fp8_e4m3_to_float` | F11 | 2 | `src/tl_templates/ascend/ascend_fp8.h:29` | ascend_fp8.h |
| C | C4-MX-FP8 | `cast_fp8_e5m2_to_float` | F11 | 2 | `src/tl_templates/ascend/ascend_fp8.h:37` | ascend_fp8.h |
| C | C1-SIMT | `cce::dim3` | F3 | 1 | `src/ascend/codegen/codegen_ascend.cc:986` | codegen_ascend.cc |
| C | C1-SIMT | `coalesced_threads` | F3 | 4 | `src/ascend/codegen/codegen_ascend.cc:631` | codegen_ascend.cc |
| C | C1-SIMT | `cooperative_groups` | F3 | 10 | `src/ascend/codegen/codegen_ascend.cc:559` | codegen_ascend.cc, codegen_ascend.h |
| C | C6-ND2NZ-DUAL | `DATA_BLOCK_COPY` | F8 | 1 | `src/tl_templates/ascend/nd2nz_copy.h:9` | nd2nz_copy.h |
| C | C6-ND2NZ-DUAL | `dual_copy` | F10 | 20 | `src/ascend/codegen/codegen_pto.cc:4293` | codegen_pto.cc, estimate_latency.cc, infer_buffer_aliases.cc, insert_nd2nz.cc, rewrite_dual_copy.cc, __init__.py, copy_op.py, mode.py |
| C | C8-EVENT | `event_t` | F8 | 1 | `src/ascend/codegen/codegen_ascend.cc:1362` | codegen_ascend.cc |
| C | C4-MX-FP8 | `float4_e1m2x2_t` | F5 | 2 | `src/tl_templates/ascend/simd_inst.h:36` | simd_inst.h |
| C | C4-MX-FP8 | `float4_e2m1x2_t` | F5 | 11 | `src/ascend/codegen/codegen_ascend.cc:245` | codegen_ascend.cc, codegen_pto.cc, gemm.h, simd_inst.h, gemm_mad.py |
| C | C4-MX-FP8 | `float8_e4m3x2_t` | F5 | 7 | `src/ascend/codegen/codegen_ascend.cc:3485` | codegen_ascend.cc, ascend_fp8.h |
| C | C4-MX-FP8 | `float8_e4m3_t` | F5 | 7 | `src/ascend/codegen/codegen_ascend.cc:239` | codegen_ascend.cc, codegen_pto.cc, ascend_fp8.h, gemm.h, gemm_mad.py, gemm_mad_blockscaled.py |
| C | C4-MX-FP8 | `float8_e5m2x2_t` | F5 | 7 | `src/ascend/codegen/codegen_ascend.cc:3485` | codegen_ascend.cc, ascend_fp8.h |
| C | C4-MX-FP8 | `float8_e5m2_t` | F5 | 3 | `src/ascend/codegen/codegen_ascend.cc:241` | codegen_ascend.cc, codegen_pto.cc, simd_inst.h |
| C | C4-MX-FP8 | `float8_e8m0_t` | F5 | 2 | `src/ascend/codegen/codegen_ascend.cc:243` | codegen_ascend.cc, simd_inst.h |
| C | C4-MX-FP8 | `float_e4m3_t` | F5 | 11 | `src/ascend/codegen/codegen_ascend.cc:387` | codegen_ascend.cc, ascend_fp8.h |
| C | C4-MX-FP8 | `float_e5m2_t` | F5 | 8 | `src/ascend/codegen/codegen_ascend.cc:390` | codegen_ascend.cc, ascend_fp8.h |
| C | C4-MX-FP8 | `fp8_e4_2_t` | F5 | 1 | `src/tl_templates/ascend/ascend_fp8.h:136` | ascend_fp8.h |
| C | C4-MX-FP8 | `fp8_e4_4_t` | F5 | 1 | `src/tl_templates/ascend/ascend_fp8.h:137` | ascend_fp8.h |
| C | C4-MX-FP8 | `fp8_e4_8_t` | F5 | 1 | `src/tl_templates/ascend/ascend_fp8.h:138` | ascend_fp8.h |
| C | C4-MX-FP8 | `fp8_e4_t` | F5 | 5 | `src/ascend/codegen/codegen_ascend.cc:799` | codegen_ascend.cc, ascend_fp8.h, simd_inst.h |
| C | C4-MX-FP8 | `fp8_e5_2_t` | F5 | 1 | `src/tl_templates/ascend/ascend_fp8.h:140` | ascend_fp8.h |
| C | C4-MX-FP8 | `fp8_e5_4_t` | F5 | 1 | `src/tl_templates/ascend/ascend_fp8.h:141` | ascend_fp8.h |
| C | C4-MX-FP8 | `fp8_e5_8_t` | F5 | 1 | `src/tl_templates/ascend/ascend_fp8.h:142` | ascend_fp8.h |
| C | C4-MX-FP8 | `fp8_e5_t` | F5 | 3 | `src/tl_templates/ascend/ascend_fp8.h:133` | ascend_fp8.h, simd_inst.h |
| C | C4-MX-FP8 | `fp8_e8m0_t` | F5 | 3 | `src/ascend/codegen/codegen_ascend.cc:1588` | codegen_ascend.cc, gemm.h |
| C | C4-MX-FP8 | `fp8_e8_2_t` | F5 | 1 | `src/tl_templates/ascend/ascend_fp8.h:144` | ascend_fp8.h |
| C | C4-MX-FP8 | `fp8_e8_4_t` | F5 | 1 | `src/tl_templates/ascend/ascend_fp8.h:145` | ascend_fp8.h |
| C | C4-MX-FP8 | `fp8_e8_8_t` | F5 | 1 | `src/tl_templates/ascend/ascend_fp8.h:146` | ascend_fp8.h |
| C | C4-MX-FP8 | `fp8_e8_t` | F5 | 1 | `src/tl_templates/ascend/ascend_fp8.h:134` | ascend_fp8.h |
| C | C5-AIC-MODE | `hf32` | F10 | 65 | `src/ascend/codegen/codegen_ascend.cc:1871` | codegen_ascend.cc, codegen_pto.cc, codegen_pto.h, builtin.cc, builtin.h, ascend_pipe.h, auto_schedule.cc, ir_structure.h, task_analysis.h, estimate_latency.cc, estimate_latency.h, mode.py |
| C | C3-SIMD-TAG | `HIGHER` | F8 | 15 | `src/ascend/codegen/codegen_pto.cc:5181` | codegen_pto.cc, simd_inst.h, simd.py |
| C | C9-RNG950 | `IDX_2` | F11 | 8 | `src/tl_templates/ascend/philox_rng.h:30` | philox_rng.h, random_kernel_base.h |
| C | C9-RNG950 | `IDX_3` | F11 | 7 | `src/tl_templates/ascend/philox_rng.h:31` | philox_rng.h, random_kernel_base.h |
| C | C3-SIMD-TAG | `INC_ORDER` | F8 | 6 | `src/ascend/codegen/codegen_pto.cc:5351` | codegen_pto.cc, simd_inst.h, simd.py |
| C | C6-ND2NZ-DUAL | `InsertNd2Nz` | F10 | 12 | `src/ascend/transform/estimate_latency.cc:1197` | estimate_latency.cc, insert_nd2nz.cc, copy_op.py, pipeline.py, __init__.py |
| C | C8-EVENT | `kIntraCoreFlagLimit` | F12 | 4 | `src/ascend/transform/insert_sync.cc:1943` | insert_sync.cc |
| C | C1-SIMT | `laneid` | F3 | 1 | `src/ascend/codegen/codegen_ascend.cc:1827` | codegen_ascend.cc |
| C | C2-SIMD-REG | `LegalizeSimdMerging` | F10 | 12 | `src/ascend/codegen/codegen_pto.cc:1655` | codegen_pto.cc, sfu_precision.h, legalize_simd_merging.cc, simd_inst.h, pipeline.py, __init__.py |
| C | C3-SIMD-TAG | `LOWER` | F8 | 12 | `src/ascend/codegen/codegen_pto.cc:5181` | codegen_pto.cc, nd2nz_copy.h, simd.py |
| C | C7-MIXKERN | `MixedKernel` | F10 | 26 | `src/ascend/codegen/codegen_pto.cc:1827` | codegen_pto.cc, codegen_pto.h, ir.cc, scheduled_tir.h, lower_scheduled_tir.cc, rewrite_dual_copy.cc, __init__.py, kernel.py, pipeline.py |
| C | C3-SIMD-TAG | `MODE_MERGING` | F8 | 29 | `src/ascend/codegen/codegen_ascend.cc:31` | codegen_ascend.cc, codegen_pto.cc, sfu_precision.h, legalize_simd_merging.cc, simd_inst.h, simd.py, pipeline.py, __init__.py |
| C | C3-SIMD-TAG | `MODE_ZEROING` | F8 | 64 | `src/ascend/codegen/codegen_pto.cc:39` | codegen_pto.cc, codegen_pto.h, ascend_simdvf_lower_parallel.cc, nd2nz_copy.h, simd_inst.h, simd.py |
| C | C6-ND2NZ-DUAL | `Nd2Nz` | F10 | 34 | `src/ascend/codegen/codegen_ascend.cc:1610` | codegen_ascend.cc, codegen_ascend.h, codegen_pto.cc, codegen_pto.h, memory_detector.h, estimate_latency.cc, estimate_latency.h, insert_nd2nz.cc |
| C | C6-ND2NZ-DUAL | `nd2nz` | F10 | 91 | `src/ascend/codegen/codegen_ascend.cc:558` | codegen_ascend.cc, codegen_ascend.h, codegen_pto.cc, builtin.cc, builtin.h, copy.cc, copy.h, ascend_pipe.h, memory_detector.h, estimate_latency.cc, estimate_latency.h, insert_nd2nz.cc, nd2nz_copy.h, copy_op.py, dma.py |
| C | C3-SIMD-TAG | `PART_EVEN` | F8 | 6 | `src/ascend/codegen/codegen_pto.cc:5522` | codegen_pto.cc, ascend_simdvf_lower_parallel.cc, nd2nz_copy.h, simd.py |
| C | C3-SIMD-TAG | `PART_ODD` | F8 | 3 | `src/tl_templates/ascend/nd2nz_copy.h:153` | nd2nz_copy.h, simd.py |
| C | C3-SIMD-TAG | `PAT_ALL` | F8 | 21 | `src/ascend/codegen/codegen_ascend.cc:3177` | codegen_ascend.cc, codegen_pto.cc, ascend_simdvf_lower_parallel.cc, nd2nz_copy.h, simd.py |
| C | C3-SIMD-TAG | `PAT_ALLF` | F8 | 2 | `src/ascend/codegen/codegen_pto.cc:4916` | codegen_pto.cc |
| C | C3-SIMD-TAG | `PAT_H` | F8 | 3 | `src/ascend/codegen/codegen_pto.cc:4916` | codegen_pto.cc, simd.py |
| C | C3-SIMD-TAG | `PAT_M3` | F8 | 3 | `src/ascend/codegen/codegen_pto.cc:4916` | codegen_pto.cc, simd.py |
| C | C3-SIMD-TAG | `PAT_M4` | F8 | 3 | `src/ascend/codegen/codegen_pto.cc:4916` | codegen_pto.cc, simd.py |
| C | C3-SIMD-TAG | `PAT_Q` | F8 | 3 | `src/ascend/codegen/codegen_pto.cc:4916` | codegen_pto.cc, simd.py |
| C | C3-SIMD-TAG | `PAT_VL` | F8 | 5 | `src/ascend/codegen/codegen_pto.cc:1728` | codegen_pto.cc |
| C | C3-SIMD-TAG | `PAT_VL1` | F8 | 5 | `src/ascend/codegen/codegen_pto.cc:1709` | codegen_pto.cc, simd_inst.h, simd.py |
| C | C3-SIMD-TAG | `PAT_VL128` | F8 | 2 | `src/ascend/codegen/codegen_pto.cc:4915` | codegen_pto.cc, simd.py |
| C | C3-SIMD-TAG | `PAT_VL16` | F8 | 1 | `src/ascend/codegen/codegen_pto.cc:4915` | codegen_pto.cc |
| C | C3-SIMD-TAG | `PAT_VL2` | F8 | 4 | `src/ascend/codegen/codegen_pto.cc:1712` | codegen_pto.cc, simd_inst.h |
| C | C3-SIMD-TAG | `PAT_VL3` | F8 | 1 | `src/ascend/codegen/codegen_pto.cc:4914` | codegen_pto.cc |
| C | C3-SIMD-TAG | `PAT_VL32` | F8 | 1 | `src/ascend/codegen/codegen_pto.cc:4915` | codegen_pto.cc |
| C | C3-SIMD-TAG | `PAT_VL4` | F8 | 1 | `src/ascend/codegen/codegen_pto.cc:4914` | codegen_pto.cc |
| C | C3-SIMD-TAG | `PAT_VL64` | F8 | 2 | `src/ascend/codegen/codegen_pto.cc:4915` | codegen_pto.cc, nd2nz_copy.h |
| C | C3-SIMD-TAG | `PAT_VL8` | F8 | 3 | `src/ascend/codegen/codegen_pto.cc:1716` | codegen_pto.cc, simd.py |
| C | C1-SIMT | `PIPE_CONS` | F8 | 3 | `src/ascend/transform/rewrite_flag_to_buf.cc:22` | rewrite_flag_to_buf.cc |
| C | C1-SIMT | `PIPE_PROD` | F8 | 3 | `src/ascend/transform/rewrite_flag_to_buf.cc:20` | rewrite_flag_to_buf.cc |
| C | C3-SIMD-TAG | `POST_UPDATE` | F8 | 14 | `src/ascend/codegen/codegen_ascend.cc:2429` | codegen_ascend.cc, codegen_pto.cc, nd2nz_copy.h, simd.py |
| C | C3-SIMD-TAG | `POS_LOWEST` | F8 | 4 | `src/ascend/codegen/codegen_pto.cc:5405` | codegen_pto.cc, simd_inst.h, simd.py |
| C | C9-RNG950 | `RAND_2POW32_INV` | F11 | 4 | `src/tl_templates/ascend/philox_rng.h:54` | philox_rng.h, random_kernel_base.h |
| C | C9-RNG950 | `RAND_2POW32_INV_HALF` | F11 | 3 | `src/tl_templates/ascend/philox_rng.h:55` | philox_rng.h, random_kernel_base.h |
| C | C4-MX-FP8 | `RewriteFp4ToFp4x2` | F10 | 14 | `src/ascend/op/copy.cc:95` | copy.cc, rewrite_fp4_to_fp4x2.cc, pipeline.py, __init__.py |
| C | C4-MX-FP8 | `rewrite_fp4_to_fp4x2` | F10 | 1 | `src/ascend/transform/rewrite_fp4_to_fp4x2.cc:2` | rewrite_fp4_to_fp4x2.cc |
| C | C3-SIMD-TAG | `ROUND_R` | F8 | 6 | `src/ascend/codegen/codegen_pto.cc:5518` | codegen_pto.cc, ascend_simdvf_lower_parallel.cc, nd2nz_copy.h, simd.py |
| C | C3-SIMD-TAG | `RS_DISABLE` | F8 | 3 | `src/ascend/codegen/codegen_pto.cc:5519` | codegen_pto.cc, nd2nz_copy.h, simd.py |
| C | C5-AIC-MODE | `set_atomic` | F10 | 42 | `src/ascend/codegen/codegen_ascend.cc:1893` | codegen_ascend.cc, codegen_pto.cc, builtin.cc, builtin.h, ascend_pipe.h, ir_structure.h, task_analysis.h, mode.py |
| C | C2-SIMD-REG | `SimdVF` | F10 | 140 | `src/ascend/codegen/codegen_ascend.cc:403` | codegen_ascend.cc, codegen_ascend.h, codegen_pto.cc, ir.cc, builtin.h, ascend_pipe.h, ascend_simdvf_lower_parallel.cc, layout_inference.cc, materialize_schedule_units.cc, nd2nz_copy.h, vf_checker.py, __init__.py, frame.py, kernel.py, reduce_op.py, schedule_hint.py, simd.py |
| C | C1-SIMT | `SimtVF` | F10 | 109 | `src/ascend/codegen/codegen_ascend.cc:401` | codegen_ascend.cc, codegen_ascend.h, codegen_pto.cc, codegen_pto.h, ir.cc, copy.cc, target.cc, ascend_pipe.h, auto_schedule.cc, materialize_schedule_units.cc, thread_storage_sync.cc, vf_checker.py, __init__.py, frame.py, kernel.py, random.py, schedule_hint.py, pipeline.py |
| C | C1-SIMT | `simt_api/cooperative_groups.h` | F1 | 1 | `src/ascend/codegen/codegen_ascend.cc:633` | codegen_ascend.cc |
| C | C1-SIMT | `threadIdx.` | F3 | 45 | `src/ascend/codegen/codegen_ascend.cc:939` | codegen_ascend.cc, codegen_pto.cc, ir.cc, layout_inference.cc, lower_tile_op.cc, thread_storage_sync.cc, vf_regions.h, reduce.h |
| C | C4-MX-FP8 | `vector_f4e1m2x2` | F4 | 1 | `src/tl_templates/ascend/simd_inst.h:37` | simd_inst.h |
| C | C4-MX-FP8 | `vector_f4e2m1x2` | F4 | 1 | `src/tl_templates/ascend/simd_inst.h:34` | simd_inst.h |
| C | C4-MX-FP8 | `vector_f8e4m3` | F4 | 1 | `src/tl_templates/ascend/simd_inst.h:22` | simd_inst.h |
| C | C4-MX-FP8 | `vector_f8e5m2` | F4 | 2 | `src/tl_templates/ascend/simd_inst.h:25` | simd_inst.h |
| C | C4-MX-FP8 | `vector_f8e8m0` | F4 | 1 | `src/tl_templates/ascend/simd_inst.h:31` | simd_inst.h |
| C | C4-MX-FP8 | `__asc_cvt_float2_to_fp8x2` | F7 | 8 | `src/ascend/codegen/codegen_ascend.cc:3442` | codegen_ascend.cc, ascend_fp8.h |
| C | C4-MX-FP8 | `__ASC_E4M3` | F7,F8 | 4 | `src/ascend/codegen/codegen_ascend.cc:3440` | codegen_ascend.cc, ascend_fp8.h |
| C | C4-MX-FP8 | `__ASC_E5M2` | F7,F8 | 4 | `src/ascend/codegen/codegen_ascend.cc:3440` | codegen_ascend.cc, ascend_fp8.h |
| C | C4-MX-FP8 | `__asc_fp8x2_storage_t` | F7 | 4 | `src/tl_templates/ascend/ascend_fp8.h:17` | ascend_fp8.h |
| C | C4-MX-FP8 | `__ASC_SATFINITE` | F7,F8 | 7 | `src/ascend/codegen/codegen_ascend.cc:3444` | codegen_ascend.cc, ascend_fp8.h |
| C | C7-MIXKERN | `__cube__` | F2 | 1 | `src/ascend/codegen/codegen_ascend.cc:348` | codegen_ascend.cc |
| C | C4-MX-FP8 | `__e4m3x22float2` | F7 | 4 | `src/ascend/codegen/codegen_ascend.cc:3487` | codegen_ascend.cc, ascend_fp8.h |
| C | C4-MX-FP8 | `__e5m2x22float2` | F7 | 4 | `src/ascend/codegen/codegen_ascend.cc:3487` | codegen_ascend.cc, ascend_fp8.h |
| C | C7-MIXKERN | `__global__` | F2 | 4 | `src/ascend/codegen/codegen_ascend.cc:342` | codegen_ascend.cc, debug.h |
| C | C1-SIMT | `__launch_bounds__` | F2,F3 | 1 | `src/ascend/codegen/codegen_ascend.cc:980` | codegen_ascend.cc |
| C | C7-MIXKERN | `__mix__` | F2 | 1 | `src/ascend/codegen/codegen_ascend.cc:342` | codegen_ascend.cc |
| C | C2-SIMD-REG | `__simd_callee__` | F2 | 98 | `src/tl_templates/ascend/nd2nz_copy.h:24` | nd2nz_copy.h, simd_inst.h |
| C | C7-MIXKERN | `__simd_vf__` | F2 | 4 | `src/ascend/codegen/codegen_ascend.cc:729` | codegen_ascend.cc, nd2nz_copy.h |
| C | C1-SIMT | `__simt_callee__` | F2 | 52 | `src/tl_templates/ascend/debug.h:295` | debug.h, philox_rng.h, random_kernel_base.h, reduce.h |
| C | C1-SIMT | `__simt_vf__` | F2 | 3 | `src/ascend/codegen/codegen_ascend.cc:978` | codegen_ascend.cc, debug.h |

## D 类逐符号

| class | 组 | 符号 | 族 | 命中 | 首处举证 文件:行 | 出现文件 |
|---|---|---|---|---|---|---|
| D | D2-AIC-API | `asc_copy_gm2ub_align` | F6 | 2 | `src/ascend/codegen/codegen_ascend.cc:1406` | codegen_ascend.cc |
| D | D2-AIC-API | `asc_copy_l0c2ub` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:1475` | codegen_ascend.cc |
| D | D2-AIC-API | `asc_copy_l12l0a` | F6 | 4 | `src/ascend/codegen/codegen_ascend.cc:1567` | codegen_ascend.cc, gemm.h |
| D | D2-AIC-API | `asc_copy_l12l0a_transpose` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:1566` | codegen_ascend.cc |
| D | D2-AIC-API | `asc_copy_l12l0b` | F6 | 5 | `src/ascend/codegen/codegen_ascend.cc:1567` | codegen_ascend.cc, gemm.h |
| D | D2-AIC-API | `asc_copy_l12l0b_transpose` | F6 | 3 | `src/ascend/codegen/codegen_ascend.cc:1566` | codegen_ascend.cc, gemm.h |
| D | D2-AIC-API | `asc_copy_ub2gm_align` | F6 | 2 | `src/ascend/codegen/codegen_ascend.cc:1424` | codegen_ascend.cc |
| D | D2-AIC-API | `asc_init` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:711` | codegen_ascend.cc |
| D | D2-AIC-API | `asc_load_dev` | F6 | 4 | `src/tl_templates/ascend/dcache_bypass.h:20` | dcache_bypass.h |
| D | D2-AIC-API | `asc_lock` | F6 | 15 | `src/ascend/codegen/codegen_ascend.cc:1853` | codegen_ascend.cc, rewrite_flag_to_buf.cc, gemm.h, sync.py |
| D | D2-AIC-API | `asc_mmad` | F6 | 2 | `src/ascend/codegen/codegen_ascend.cc:1656` | codegen_ascend.cc, gemm.h |
| D | D2-AIC-API | `asc_store_dev` | F6 | 4 | `src/tl_templates/ascend/dcache_bypass.h:42` | dcache_bypass.h |
| D | D2-AIC-API | `asc_sync` | F6 | 1 | `src/ascend/codegen/codegen_ascend.cc:1343` | codegen_ascend.cc |
| D | D2-AIC-API | `asc_unlock` | F6 | 15 | `src/ascend/codegen/codegen_ascend.cc:1865` | codegen_ascend.cc, rewrite_flag_to_buf.cc, gemm.h, sync.py |
| D | D4-LAYOUT | `C0` | F12 | 82 | `src/ascend/codegen/codegen_pto.cc:679` | codegen_pto.cc, ascend_layouts.cc, ascend_layouts.h, builtin.h, copy.cc, fill.cc, oob_padding.h, insert_oob_padding.cc, gemm.h, numeric_limits.h, gemm_mad.py |
| D | D4-LAYOUT | `Fractal` | F12 | 137 | `src/ascend/codegen/codegen_pto.cc:3455` | codegen_pto.cc, codegen_pto.h, ascend_layouts.cc, ascend_layouts.h, copy.cc, fill.cc, oob_padding.cc, oob_padding.h, insert_oob_padding.cc, normalize_fractal_storage.cc, pipeline.py, __init__.py |
| D | D4-LAYOUT | `fractal` | F12 | 48 | `src/ascend/codegen/codegen_pto.cc:3763` | codegen_pto.cc, ascend_layouts.cc, ascend_layouts.h, builtin.h, copy.cc, fill.cc, oob_padding.cc, insert_oob_padding.cc, normalize_fractal_storage.cc, gemm.h, nd2nz_copy.h, gemm_mad.py, gemm_mad_blockscaled.py |
| D | D4-LAYOUT | `mix_aiv_count` | F12 | 4 | `src/ascend/codegen/codegen_ascend.cc:341` | codegen_ascend.cc, codegen_ascend.h |
| D | D4-LAYOUT | `nBlk` | F12 | 3 | `src/tl_templates/ascend/gemm.h:73` | gemm.h |
| D | D3-PIPEENUM | `PIPE_ALL` | F8 | 6 | `src/ascend/codegen/codegen_ascend.cc:1333` | codegen_ascend.cc, ascend_pipe.h, infer_buffer_aliases.cc, sync.py |
| D | D3-PIPEENUM | `PIPE_FIX` | F8 | 10 | `src/ascend/codegen/codegen_ascend.cc:1334` | codegen_ascend.cc, assign_core.cc, infer_buffer_aliases.cc, schedule_hint.py, sync.py |
| D | D3-PIPEENUM | `PIPE_H` | F8 | 3 | `src/ascend/transform/ascend_pipe.h:26` | ascend_pipe.h |
| D | D3-PIPEENUM | `PIPE_M` | F8 | 13 | `src/ascend/codegen/codegen_ascend.cc:1333` | codegen_ascend.cc, assign_core.cc, insert_sync.cc, gemm.h, sync.py |
| D | D3-PIPEENUM | `PIPE_MTE1` | F8 | 12 | `src/ascend/codegen/codegen_ascend.cc:1333` | codegen_ascend.cc, assign_core.cc, gemm.h, sync.py |
| D | D3-PIPEENUM | `PIPE_MTE2` | F8 | 6 | `src/ascend/codegen/codegen_ascend.cc:1334` | codegen_ascend.cc, assign_core.cc, copy_op.py, sync.py |
| D | D3-PIPEENUM | `PIPE_MTE3` | F8 | 11 | `src/ascend/codegen/codegen_ascend.cc:1334` | codegen_ascend.cc, assign_core.cc, infer_buffer_aliases.cc, sync.py |
| D | D3-PIPEENUM | `PIPE_S` | F8 | 8 | `src/ascend/codegen/codegen_ascend.cc:1334` | codegen_ascend.cc, assign_core.cc, insert_sync.cc, copy_op.py, sync.py |
| D | D3-PIPEENUM | `PIPE_V` | F8 | 7 | `src/ascend/codegen/codegen_ascend.cc:1333` | codegen_ascend.cc, ascend_pipe.h, assign_core.cc, insert_sync.cc, sync.py |
| D | D4-LAYOUT | `TRANS_B` | F12 | 6 | `src/tl_templates/ascend/gemm.h:34` | gemm.h |
| D | D1-REGTYPE | `vector_bf16` | F4 | 2 | `src/tl_templates/ascend/simd_inst.h:19` | simd_inst.h |
| D | D1-REGTYPE | `vector_bool` | F4 | 141 | `src/ascend/codegen/codegen_ascend.cc:461` | codegen_ascend.cc, builtin.h, nd2nz_copy.h, simd_inst.h |
| D | D1-REGTYPE | `vector_f16` | F4 | 2 | `src/tl_templates/ascend/simd_inst.h:14` | simd_inst.h |
| D | D1-REGTYPE | `vector_f32` | F4 | 17 | `src/ascend/codegen/codegen_ascend.cc:3093` | codegen_ascend.cc, simd_inst.h |
| D | D1-REGTYPE | `vector_s16` | F4 | 4 | `src/tl_templates/ascend/simd_inst.h:58` | simd_inst.h |
| D | D1-REGTYPE | `vector_s32` | F4 | 6 | `src/tl_templates/ascend/simd_inst.h:54` | simd_inst.h |
| D | D1-REGTYPE | `vector_s64` | F4 | 1 | `src/tl_templates/ascend/simd_inst.h:66` | simd_inst.h |
| D | D1-REGTYPE | `vector_s8` | F4 | 3 | `src/tl_templates/ascend/simd_inst.h:62` | simd_inst.h |
| D | D1-REGTYPE | `vector_u16` | F4 | 6 | `src/tl_templates/ascend/simd_inst.h:46` | simd_inst.h |
| D | D1-REGTYPE | `vector_u32` | F4 | 21 | `src/tl_templates/ascend/simd_inst.h:42` | simd_inst.h |
| D | D1-REGTYPE | `vector_u64` | F4 | 1 | `src/tl_templates/ascend/simd_inst.h:70` | simd_inst.h |
| D | D1-REGTYPE | `vector_u8` | F4 | 5 | `src/tl_templates/ascend/simd_inst.h:50` | simd_inst.h |
| D | D1-REGTYPE | `vector_uint16_t` | F4 | 1 | `src/tl_templates/ascend/simd_inst.h:1011` | simd_inst.h |
| D | D1-REGTYPE | `vector_uint32_t` | F4 | 1 | `src/tl_templates/ascend/simd_inst.h:1018` | simd_inst.h |
| D | D3-PIPEENUM | `__NPU_ARCH__` | F8 | 2 | `src/tl_templates/ascend/random_kernel_base.h:9` | random_kernel_base.h, simd_inst.h |
