#!/usr/bin/env bash
# =============================================================================
# checklist.sh — 910B 移植"950-only 依赖面"清单自证脚本（p2-13 · P0-1 产物）
#
# 作用（四件事，全部可重跑）：
#   1) dump    : 按符号族穷举 tilelang 主仓模板层/codegen 层的 950 面命中点（file:line:symbol）
#   2) tok     : 按族输出 uniq 符号集合（写清单时的取料口，也是 audit 的输入）
#   3) symbols : 生成附表 inventory_symbols.md（逐符号 分类|组|命中数|举证 文件:行）
#   4) audit   : 拿穷举结果反查 INVENTORY.md + 附表，凡"grep 到但清单没写"判 UNLISTED
#                —— UNLISTED=0 即"清单完备性"自证通过（退出码 0）。
#
# 用法:
#   ./checklist.sh                  # audit + 分类计数（默认）
#   ./checklist.sh dump [F#]        # 全量/单族命中明细
#   ./checklist.sh tok  [F#]        # 全量/单族 uniq 符号
#   ./checklist.sh symbols          # 重新生成 inventory_symbols.md（分类规则见 classify()）
#   ./checklist.sh counts           # 只打印 A/B/C/D 计数
#   REPO=/path/to/tilelang ./checklist.sh
#
# 纪律：全程只读主仓，不编译、不运行任何 ascend 目标（本机无 NPU，纯静态分析）。
# 兼容：macOS 自带 bash 3.2 —— case 分支内不得出现括号，正则只在 FAMS 表里用。
# =============================================================================
set -uo pipefail
TMPD_GAP=$(mktemp)

REPO="${REPO:-$HOME/Documents/codes/tilelang}"
[ -d "$REPO/src/tl_templates/ascend" ] || { echo "FATAL: 主仓不可达: $REPO" >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INV="$HERE/INVENTORY.md"
SYM="$HERE/inventory_symbols.md"

TPL="$REPO/src/tl_templates/ascend"
CG="$REPO/src/ascend"
PL="$REPO/tilelang/ascend"

MODE="${1:-audit}"
FILTER="${2:-}"

# ── 符号族（族号在 INVENTORY.md §2 引用；改族必须同步改清单）───────────────────
# 格式: 族号|族名|扫描范围|ERE
FAMS=(
  'F1|950 代际头 include（c_api/simt_api）|ALL|(c_api|simt_api)/[a-z0-9_]+\.h'
  'F2|device 属性与函数限定宏|TPL_CG|__simt_vf__|__simt_callee__|__simd_vf__|__simd_callee__|__aicore__|__launch_bounds__|__SIMT_DEVICE_FUNCTIONS_DECL__|__global__|__mix__|__cube__|__gm__|__ubuf__|__cc__|__cbuf__|__ca__|__cb__'
  'F3|SIMT 内建变量与核身份宏|ALL|block_idx|blockIdx\.|threadIdx\.|asc_get_sub_block_id|ASC_IS_AIV|ASC_IS_AIC|cce::dim3|laneid|cooperative_groups|coalesced_threads|__launch_bounds__'
  'F4|SIMD 寄存器类型 vector_*|TPL_CG|vector_(f32|f16|bf16|u8|u16|u32|u64|s8|s16|s32|s64|bool|uint16_t|uint32_t|f8e4m3|f8e5m2|f8e8m0|f4e2m1x2|f4e1m2x2)'
  'F5|类型代理（bf16/fp16/fp8/fp4/宽向量元组）|ALL|bfloat16_t|bfloat16x2_t|float8_e4m3x?_t|float8_e5m2x?_t|float4_e2m1x2_t|float4_e1m2x2_t|float_e4m3_t|float_e5m2_t|fp8_e4_t|fp8_e5_t|fp8_e8_t|fp8_e8m0_t|fp8_e[458]_[248]_t|float8_e[0-9a-z]*_t|half_t|__fp16|half2|half|float2|float4'
  'F6|asc_* C API（950 MicroAPI 命名空间）|ALL|asc_[a-z0-9_]+'
  'F7|双下划线内建（转换/存储/常量）|ALL|__[aA][sS][cC]_[a-zA-Z0-9_]+|__(e4m3x2|e5m2x2|bfloat162|bfloat1622float2|half22float2|float22)[a-z0-9_]*|make_float[24]|make_(u?longlong|u?int)[24]'
  'F8|硬件枚举/tag/mode 类型与事件|ALL|PAT_[A-Z0-9]+|MODE_(ZEROING|MERGING)|ROUND_R|ROUND_MODE[A-Za-z_]*|PART_(EVEN|ODD)|RS_DISABLE|POST_UPDATE|HIGHER|LOWER|POS_LOWEST|INC_ORDER|Bin_N[01]|ASC_LOCK_[A-Z]+|PIPE_[A-Z0-9]+|event_t|DATA_BLOCK_COPY|__ASC_(SATFINITE|E4M3|E5M2)|__NPU_ARCH__'
  'F9|C++ 标准库依赖（std::bit_cast / <bit>）|TPL|std::bit_cast|__builtin_bit_cast|bit>'
  'F10|950 专属语言构造与 pass 名|ALL|SimtVF|SimdVF|dual_copy|blockscaled|RewriteFp4ToFp4x2|rewrite_fp4_to_fp4x2|set_atomic|hf32|nd2nz|Nd2Nz|MixedKernel|ascend_mad_mx|InsertNd2Nz|LegalizeSimdMerging|AscendSimdVFLowerParallel'
  'F11|全局作用域 CCE 内建与 950 运行时件|TPL_CG|::(vcvt|vpack|vsstb|asc_mem_bar|print_var|print_buffer|BoxMullerFloat|ALG_KEY_SIZE|ALG_COUNTER_SIZE)|cast_float_to_fp8_[a-z0-9]+|cast_fp8_[a-z0-9]+_to_float|ALG_(KEY|COUNTER)_SIZE|IDX_[0-9]|RAND_2POW32_INV(_HALF)?'
  'F12|分形/布局与同步容量前提|ALL|C0|kIntraCoreFlagLimit|TRANS_B|mix_aiv_count|nBlk|[fF]ractal'
)

fam_field() { # $1=族串 $2=段号
  local IFS='|' f1 f2 f3 f4
  read -r f1 f2 f3 f4 <<<"$1"
  case "$2" in
    1) printf '%s' "$f1" ;; 2) printf '%s' "$f2" ;;
    3) printf '%s' "$f3" ;; 4) printf '%s' "$f4" ;;
  esac
}

scan_files() {
  case "$1" in
    TPL)     find "$TPL" -maxdepth 1 -name '*.h' ;;
    CG)      { find "$CG" -name '*.cc'; find "$CG" -name '*.h'; } ;;
    TPL_CG)  { find "$TPL" -maxdepth 1 -name '*.h'; find "$CG" -name '*.cc'; find "$CG" -name '*.h'; } ;;
    ALL)     { find "$TPL" -maxdepth 1 -name '*.h'; find "$CG" -name '*.cc'; find "$CG" -name '*.h'; find "$PL" -name '*.py'; } ;;
    *) return 1 ;;
  esac | sort
}

run_family() { # $1=族串  $2=tok→uniq / 其他→file:line:symbol
  local scope regex
  scope=$(fam_field "$1" 3); regex=$(fam_field "$1" 4)
  if [ "$2" = "tok" ]; then
    scan_files "$scope" | xargs grep -Eoh "$regex" 2>/dev/null | sort -u
  else
    scan_files "$scope" | xargs grep -Eon "$regex" 2>/dev/null | sed "s#^$REPO/##"
  fi
}

# ── 分类规则（改这里=改判决，必须同步改 INVENTORY.md §3 叙述）───────────────────
# 输入: $1=符号 $2=出现文件串 ; 输出 "CLASS|GROUP"。class 语义见 INVENTORY.md §1.2
classify() {
  local s="$1" files="$2"
  case "$s" in
    # A ─ B1 四层处置法已覆盖（补丁已落主仓工作区，编译已推进到第 5 层）
    c_api/asc_simd.h|simt_api/asc_bf16.h|simt_api/asc_fp16.h|simt_api/asc_fp8.h|simt_api/asc_simt.h) echo "A|A1-HEAD-STUB"; return;;
    __aicore__) echo "A|A2-ATTR-ALIAS"; return;;
    block_idx)  echo "A|A3-BLOCKIDX-0"; return;;
    std::bit_cast|__builtin_bit_cast|bit\>) echo "A|A4-BITCAST"; return;;
    TL_ASCEND_SIMT) echo "A|A5-GATE"; return;;
    # A7 ─ 910B/CANN8.5.2 bisheng 实勘已证存在的存储限定符，无需处置
    __gm__|__ubuf__|__cc__|__cbuf__|__ca__|__cb__) echo "A|A7-ATTR-PROVEN"; return;;
    # A ─ 扫描副产物 / tilelang 自有标识符（并非 950 外部符号）
    asc_bf16|asc_fp16|asc_fp8|asc_simd|asc_simt|asc_target|asc_aicore|__asc_aicore|asc_simt_vf|__asc_simt_vf|__asc_fp8_interpretation_t|__asc_pad_val|asc_pad_val|__asc_rng_state|asc_rng_state|asc_fp8_interpretation_t|asc_loadalign_v2_impl|::print_var|::print_buffer) echo "A|A6-SCAN-ARTIFACT"; return;;
    # B ─ 兼容层可解（port910b_compat.h）
    bfloat16_t|bfloat16x2_t|__fp16|half|half2|float2|float4|make_float2|make_float4|make_ulonglong4|make_longlong4|make_int2|half_t) echo "B|B1-TYPE"; return;;
    __bfloat162float|__float2bfloat16_rn|__bfloat1622float2|__float22bfloat162_rn|__half22float2|__float22half2_rn) echo "B|B2-CVT"; return;;
    ASC_LOCK_BLOCK|ASC_LOCK_NON|ASC_LOCK_NON_BLOCK) echo "B|B3-LOCKMODE"; return;;
    __SIMT_DEVICE_FUNCTIONS_DECL__) echo "B|B4-DECL-MACRO"; return;;
    # C ─ SIMT 执行模型（910B 无 warp/线程语义，整条路径须关闭或改道）
    __simt_vf__|__simt_callee__|threadIdx.|cce::dim3|__launch_bounds__|asc_vf_call|asc_threadfence|asc_ballot|asc_activemask|asc_shfl_xor|asc_syncthreads|asc_reduce_add|asc_reduce_max|asc_reduce_min|laneid|cooperative_groups|coalesced_threads|simt_api/cooperative_groups.h|asc_atomic_|asc_update_addr_reg_b|ASC_IS_AIV|SimtVF|PIPE_PROD|PIPE_CONS|blockIdx.) echo "C|C1-SIMT"; return;;
    # C ─ 950 SIMD 寄存器值方言 + 谓词寄存器（910B 无谓词硬件 → 改道而非改名）
    asc_loadalign*|asc_storealign*|asc_select|asc_duplicate*|asc_create_mask*|asc_update_mask*|asc_gather*|asc_scatter|asc_intlv*|asc_deintlv*|asc_pack_to_*|asc_unpack_*|asc_arange*|asc_squeeze|asc_unsqueeze|asc_pair_reduce_sum|asc_reduce_sum*|asc_and|asc_or|asc_xor|asc_not|asc_eq*|asc_ne*|asc_gt*|asc_ge*|asc_lt*|asc_le*|asc_add*|asc_sub*|asc_mul*|asc_div|asc_max*|asc_min*|asc_abs*|asc_neg|asc_relu|asc_exp*|asc_ln|asc_sqrt|asc_leakyrelu|asc_prelu|asc_axpy|asc_madd|asc_shiftleft*|asc_shiftright*|asc_mem_bar|asc_frequency_histogram*|asc_cumulative_histogram*|SimdVF|AscendSimdVFLowerParallel|LegalizeSimdMerging) echo "C|C2-SIMD-REG"; return;;
    PAT_*|MODE_ZEROING|MODE_MERGING|ROUND_R|ROUND_MODE*|PART_EVEN|PART_ODD|RS_DISABLE|POST_UPDATE|HIGHER|LOWER|POS_LOWEST|INC_ORDER|Bin_N0|::vcvt|::vpack|::vsstb|::asc_mem_bar) echo "C|C3-SIMD-TAG"; return;;
    # C ─ 950 专属硬件特性（910B 无该特性 → 关特性，不是换名可解）
    asc_mmad_mx|asc_copy_l12l0a_mx|asc_copy_l12l0b_mx|ascend_mad_mx|blockscaled|RewriteFp4ToFp4x2|rewrite_fp4_to_fp4x2|fp8_e*_t|float8_e*_t|float4_e*_t|float_e*_t|vector_f8*|vector_f4*|__ASC_E4M3|__ASC_E5M2|__ASC_SATFINITE|__asc_cvt_float2_to_fp8x2|asc_cvt_float2_to_fp8x2|__asc_fp8x2_storage_t|asc_fp8x2_storage_t|__e4m3x22float2|__e5m2x22float2|cast_float_to_fp8_*|cast_fp8_*_to_float|float4_e2m1x2_t|float4_e1m2x2_t|fp8_e8m0_t) echo "C|C4-MX-FP8"; return;;
    asc_unit_flag_mode|asc_dual_dst_mode|asc_quant_mode|asc_relu_pre_mode|asc_set_atomic_*|asc_set_hf32_round_mode|asc_disable_hf32|asc_enable_hf32|asc_hf32_round_mode|asc_set_mmad_direction_|asc_load_l2_cache_mode|asc_store_l2_cache_mode|hf32|set_atomic) echo "C|C5-AIC-MODE"; return;;
    asc_copy_gm2l1_nd2nz|asc_copy_gm2l1_dn2nz|asc_set_gm2l1_nz_para|asc_set_l0c_copy_nz_para|asc_set_copy_pad_val|asc_fill_l1|asc_copy_ub2l1|asc_copy_l0c2gm|ascend_nd2nz_scatter_callee|nd2nz|Nd2Nz|InsertNd2Nz|dual_copy|DATA_BLOCK_COPY) echo "C|C6-ND2NZ-DUAL"; return;;
    asc_get_sub_block_id|__mix__|__cube__|__global__|__simd_vf__|MixedKernel|ASC_IS_AIC) echo "C|C7-MIXKERN"; return;;
    asc_sync_notify|asc_sync_wait|asc_sync_*|event_t|kIntraCoreFlagLimit) echo "C|C8-EVENT"; return;;
    philox*|tl::philox_init|tl::philox_rand|BoxMullerFloat|::BoxMullerFloat|::ALG_KEY_SIZE|::ALG_COUNTER_SIZE|ALG_KEY_SIZE|ALG_COUNTER_SIZE|IDX_2|IDX_3|RAND_2POW32_INV|RAND_2POW32_INV_HALF|::float_e4m3_t|::float_e5m2_t) echo "C|C9-RNG950"; return;;
    # D ─ 静态不可判（910B/CANN 8.5.2 是否声明此符号，由上卡探针裁决）
    vector_f32|vector_f16|vector_bool|vector_u8|vector_u16|vector_u32|vector_u64|vector_s8|vector_s16|vector_s32|vector_s64|vector_bf16|vector_uint16_t|vector_uint32_t) echo "D|D1-REGTYPE"; return;;
    asc_init|asc_copy_gm2ub_align|asc_copy_ub2gm_align|asc_copy_l12l0a|asc_copy_l12l0b|asc_copy_l12l0a_transpose|asc_copy_l12l0b_transpose|asc_mmad|asc_lock|asc_unlock|asc_sync|asc_sync_pipe|asc_copy_l0c2ub|asc_load_dev|asc_store_dev) echo "D|D2-AIC-API"; return;;
    PIPE_ALL|PIPE_V|PIPE_M|PIPE_MTE1|PIPE_MTE2|PIPE_MTE3|PIPE_S|PIPE_FIX|PIPE_H|__NPU_ARCH__) echo "D|D3-PIPEENUM"; return;;
    C0|TRANS_B|mix_aiv_count|nBlk|fractal|Fractal) echo "D|D4-LAYOUT"; return;;
  esac
  # 兜底：按举证位置推断（只在 SIMD 方言两个头里出现 → C2）
  case "$files" in
    simd_inst.h) echo "C|C2-SIMD-REG"; return;;
    "simd_inst.h, nd2nz_copy.h"|"nd2nz_copy.h, simd_inst.h") echo "C|C2-SIMD-REG"; return;;
  esac
  echo "D|D0-UNRESOLVED"
}

# 逐符号聚合：stdin=file:line:symbol（前两段为路径与行号，其余全归符号）
aggregate() { # $1=族号
  awk -F: -v fam="$1" '
  { path=$1; ln=$2; tok=substr($0, length(path)+length(ln)+3)
    base=path; sub(/^.*\//, "", base)
    cnt[tok]++
    if (!(tok in first)) first[tok]=path ":" ln
    k=tok SUBSEP base
    if (!(k in fseen)) { fseen[k]=1
      if (tok in fl) fl[tok]=fl[tok]", "base; else fl[tok]=base }
    if (!(tok in fams)) fams[tok]=fam; else if (index(fams[tok], fam)==0) fams[tok]=fams[tok]","fam }
  END { for (t in cnt) printf "%s\t%d\t%s\t%s\t%s\n", t, cnt[t], first[t], fl[t], fams[t] }' \
  | sort -t"$(printf '\t')" -k1,1
}

MASTER_CACHE="${MASTER_CACHE:-}"
master_tsv() { # 全族聚合 + 跨族去重（同符号取首次出现的族为主族）
  local tmpd f num
  if [ -n "$MASTER_CACHE" ] && [ -s "$MASTER_CACHE" ]; then cat "$MASTER_CACHE"; return; fi
  tmpd=$(mktemp -d)
  for f in "${FAMS[@]}"; do
    num=$(fam_field "$f" 1)
    run_family "$f" dump | aggregate "$num" > "$tmpd/$num.tsv"
  done
  cat "$tmpd"/F*.tsv | awk -F'\t' '
    { t=$1
      if (!(t in seen)) { seen[t]=1; ord[++n]=t; c[t]=$2; fr[t]=$3; fl[t]=$4; fm[t]=$5 }
      else { if ($2+0 > c[t]+0) c[t]=$2
             if (index(fm[t], $5)==0) fm[t]=fm[t]","$5 } }
    END { for (i=1;i<=n;i++) { t=ord[i]; printf "%s\t%d\t%s\t%s\t%s\n", t, c[t], fr[t], fl[t], fm[t] } }' \
    | sort -f -t"$(printf '\t')" -k1,1
  rm -rf "$tmpd"
}

emit_rows() { # $1=目标 class
  master_tsv | while IFS=$'\t' read -r tok n first files fams; do
    [ -n "$tok" ] || continue
    cls=$(classify "$tok" "$files" | head -1)
    cl=${cls%%|*}; gr=${cls#*|}
    [ "$cl" = "$1" ] || continue
    first=${first#"$REPO/"}
    printf '| %s | %s | `%s` | %s | %s | `%s` | %s |\n' \
      "$cl" "$gr" "$tok" "$fams" "$n" "$first" "$files"
  done
}

counts_from_symbols() {
  [ -f "$SYM" ] || { echo "FATAL: 缺附表 $SYM，先跑 ./checklist.sh symbols" >&2; return 1; }
  for c in A B C D; do printf '%s = %s\n' "$c" "$(grep -cE "^\| *$c *\|" "$SYM")"; done
}

case "$MODE" in
  dump|tok)
    echo "# 950-only 依赖面  mode=$MODE  REPO=$REPO  (族号释义见 INVENTORY.md §2)"
    for f in "${FAMS[@]}"; do
      num=$(fam_field "$f" 1); name=$(fam_field "$f" 2)
      if [ -n "$FILTER" ] && [ "$FILTER" != "$num" ]; then continue; fi
      echo; echo "===== $num $name ====="
      run_family "$f" "$MODE"
    done
    ;;

  symbols)
    MASTER_CACHE=$(mktemp); export MASTER_CACHE
    master_tsv > "$MASTER_CACHE"
    {
      echo "# inventory_symbols.md — 逐符号附表"
      echo
      echo "> **自动生成，勿手改**；生成命令 \`./checklist.sh symbols\`。"
      echo "> 分类规则 = checklist.sh 的 \`classify()\`；class 语义见 INVENTORY.md §1.2，"
      echo "> 各分组（A1..D4）的处置方案叙述在 INVENTORY.md §3。改判决请改 classify() 后重跑。"
      echo
      echo "举证路径相对 tilelang 主仓根；\`命中\` = 该符号在主仓内的出现次数。"
      for c in A B C D; do
        echo; echo "## $c 类逐符号"
        echo
        echo '| class | 组 | 符号 | 族 | 命中 | 首处举证 文件:行 | 出现文件 |'
        echo '|---|---|---|---|---|---|---|'
        emit_rows "$c"
      done
    } > "$SYM.tmp" && mv "$SYM.tmp" "$SYM"
    rm -f "$MASTER_CACHE"
    echo "written: $SYM"
    counts_from_symbols
    grep -c '^| [ABCD] |' "$SYM" | sed 's/^/总符号行数 = /'
    ;;

  gap)
    # 残差体检：模板层里"像外部依赖、又不属于任何已定义族"的标识符（人工复核队列，非报错）
    echo "# gap: 族外残差候选（启发式，用于证明族集合覆盖率；复核结论见 INVENTORY.md §7）"
    ALLRE="("
    for f in "${FAMS[@]}"; do ALLRE="$ALLRE($(fam_field "$f" 4))|"; done
    ALLRE="${ALLRE%|})"
    find "$TPL" -maxdepth 1 -name '*.h' | xargs awk '
      BEGIN{
        stop="if else for while return void bool char int unsigned short long signed float double template typename struct class enum namespace using static const constexpr inline true false nullptr new delete case default break continue sizeof public private protected operator this switch do auto alignas alignof static_cast reinterpret_cast dynamic_cast const_cast noexcept noexcept decltype mutable extern register volatile friend virtual override final explicit virtual constexpr"
        n=split(stop, a, " "); for(i=1;i<=n;i++) K[a[i]]=1
        ext="asc vec vector_ bfloat float fp8 half make PAT MODE PIPE ASC ROUND PART RS POST POS INC Bin cast tl __ ACL acl cstdint cstdio type_traits stdint bit std string stddef assert printf uint int8 int16 int32 int64 uint8 uint16 uint32 uint64 size_t ptrdiff"
        m=split(ext, e, " "); for(i=1;i<=m;i++) E[e[i]]=1
      }
      { line=$0; sub(/\/\/.*/, "", line); sub(/"[^"]*"/, "", line)
        while (match(line, /[A-Za-z_][A-Za-z0-9_]*/)) {
          t=substr(line, RSTART, RLENGTH); line=substr(line, RSTART+RLENGTH)
          if (!(t in K) && (t in E || t ~ /^__/ || t ~ /_t$/ || t ~ /^[A-Z][A-Z0-9_]{2,}$/)) seen[t]++
        } }
      END{ for (t in seen) print t }' | sort -u > "$TMPD_GAP.res"
    grep -hEo "$ALLRE" $(scan_files TPL) 2>/dev/null | sort -u > "$TMPD_GAP.cov"
    comm -23 "$TMPD_GAP.res" "$TMPD_GAP.cov" | sed 's/^/  RESIDUE /' | head -80
    echo "  # 残差候选=$(wc -l < "$TMPD_GAP.res" | tr -d ' ') 族内已覆盖=$(comm -12 "$TMPD_GAP.res" "$TMPD_GAP.cov" | wc -l | tr -d ' ')"
    rm -f "$TMPD_GAP.res" "$TMPD_GAP.cov"
    ;;

  counts)
    counts_from_symbols
    ;;

  audit)
    [ -f "$INV" ] || { echo "FATAL: 缺 $INV" >&2; exit 2; }
    [ -f "$SYM" ] || { echo "FATAL: 缺附表 $SYM（先跑 ./checklist.sh symbols）" >&2; exit 2; }
    CAT=$(mktemp); cat "$INV" "$SYM" > "$CAT"
    echo "# audit: grep 穷举 → 反查 INVENTORY.md + inventory_symbols.md   REPO=$REPO"
    miss_total=0; uniq_total=0
    for f in "${FAMS[@]}"; do
      num=$(fam_field "$f" 1); name=$(fam_field "$f" 2)
      hits=$(run_family "$f" tok)
      n_uniq=$(grep -c . <<<"$hits")
      uniq_total=$((uniq_total+n_uniq))
      unlisted=0
      while IFS= read -r tok; do
        [ -n "$tok" ] || continue
        if grep -qF -- "$tok" "$CAT"; then :; else
          echo "  [$num] UNLISTED: $tok"; unlisted=$((unlisted+1))
        fi
      done <<<"$hits"
      miss_total=$((miss_total+unlisted))
      printf '  %-4s %-46s uniq=%-5s unlisted=%s\n' "$num" "$name" "$n_uniq" "$unlisted"
    done
    rm -f "$CAT"
    echo "# 分类计数（附表逐符号）:"
    counts_from_symbols | sed 's/^/   /'
    if [ "$miss_total" -eq 0 ]; then
      echo "AUDIT PASS: 穷举 $uniq_total 个符号 100% 见于清单（unlisted=0）"
      exit 0
    fi
    echo "AUDIT FAIL: $miss_total 个符号未入清单 → 补 classify() 规则并重跑 symbols"
    exit 1
    ;;

  *) echo "用法: $0 [audit|dump [F#]|tok [F#]|symbols|counts|gap]" >&2; exit 2 ;;
esac
