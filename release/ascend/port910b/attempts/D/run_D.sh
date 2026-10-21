#!/bin/bash
# attempts/D/run_D.sh — 宿主机一键驱动 cann910b-d 跑 dW 件判决（幂等）
# 用法：
#   bash run_D.sh --setup                 # 装配容器环境（overlay+bisheng+GAP-D1/GAP-D1b）
#   bash run_D.sh --v ntt                 # 单件判决（全新 TILELANG_CACHE_DIR）
#   bash run_D.sh --matrix                # 5 变体判决矩阵 + 建材回归（A2/e2e_cube）+ golden
#   bash run_D.sh --shapes                # 主案 l0tr 的形状/tiling 鲁棒矩阵（5 组）
#   bash run_D.sh --all                   # matrix + shapes（判决日志的完整来源）
#   bash run_D.sh --golden                # CPU fp32 golden 自检（含上卡判据标定）
#   bash run_D.sh --npz                   # golden + 产上卡对拍件并取回 attempts/D/
#   bash run_D.sh --dump /tmp/dw_ntt.asc ntt   # 指定变体并把生成码取回本目录
# 判决纪律：所有编译一律 TILELANG_CACHE_DIR=$(mktemp -d)（compat 头内容不进 key）。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PORT910B="$(cd "$HERE/../.." && pwd)"
CTR=cann910b-d

docker cp "$HERE/d_dw_910b.py"       "$CTR:/tmp/d_dw_910b.py"       >/dev/null
docker cp "$HERE/d_dw_golden.py"     "$CTR:/tmp/d_dw_golden.py"     >/dev/null
docker cp "$HERE/a2_baseline_ntt.py" "$CTR:/tmp/a2_baseline_ntt.py" >/dev/null
docker cp "$HERE/env_setup_D.sh"     "$CTR:/tmp/env_setup_D.sh"     >/dev/null
docker cp "$HERE/compat_patch_D.h"   "$CTR:/tmp/compat_patch_D.h"   >/dev/null
docker cp "$HERE/patch_compat_D.py"  "$CTR:/tmp/patch_compat_D.py"  >/dev/null
docker cp "$PORT910B/patches/patch_bisheng.py" "$CTR:/tmp/patch_bisheng.py" >/dev/null
docker cp "$PORT910B/e2e_cube.py"    "$CTR:/tmp/e2e_cube.py"        >/dev/null

RUNENV='source /usr/local/Ascend/cann-8.5.0/set_env.sh; export BISHENG_HOME=/usr/local/Ascend/cann-8.5.0/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201; cd /tmp'

run_one() {  # run_one <argv...> —— 判决纪律：每次 TILELANG_CACHE_DIR=$(mktemp -d)
  docker exec "$CTR" bash -c "$RUNENV; TILELANG_CACHE_DIR=\$(mktemp -d) timeout 300 python3 $* 2>&1 | grep -v 'TileLang begins\|TileLang completes' | tail -34"
}

do_matrix() {
  echo "===== MATRIX：5 变体 + 建材回归（每次全新 TILELANG_CACHE_DIR）====="
  for v in baseline ntt l0tr madta madl1; do
    echo "----- variant=$v -----"
    run_one "/tmp/d_dw_910b.py --variant $v" || true
  done
  echo "----- 建材回归: A2 纯 NT L1 件 (fp16) -----"
  run_one /tmp/a2_baseline_ntt.py || true
  echo "----- 建材回归: e2e_cube 直路 -----"
  run_one /tmp/e2e_cube.py || true
  echo "----- golden -----"
  run_one /tmp/d_dw_golden.py || true
}

do_shapes() {
  echo "===== SHAPES：主案 l0tr 的形状/tiling 鲁棒矩阵 ====="
  # "T N K BN BK TT"
  for s in "2048 1024 1024 128 128 64" \
           "512 512 1024 64 64 32" \
           "4096 2048 512 128 64 128" \
           "2048 512 1024 128 128 64" \
           "2048 1024 1024 256 256 64"; do
    read -r T N K BN BK TT <<< "$s"
    echo "----- l0tr T=$T N=$N K=$K BN=$BN BK=$BK TT=$TT cores=64 -----"
    run_one "/tmp/d_dw_910b.py --variant l0tr --T $T --N $N --K $K --BN $BN --BK $BK --TT $TT" || true
  done
  echo "----- 同形状 dtype 面：l0tr acc=fp32 入 fp16（生产一致）已由上列覆盖；bf16 已知后端缺陷不测 -----"
}

case "${1:-}" in
  --setup)  docker exec "$CTR" bash /tmp/env_setup_D.sh ;;
  --v)      shift; run_one "/tmp/d_dw_910b.py --variant $*" ;;
  --matrix) do_matrix ;;
  --shapes) do_shapes ;;
  --all)    do_matrix; echo; do_shapes ;;
  --dump)   shift; D="${1:-/tmp/dw_ntt.asc}"; V="${2:-ntt}"
            docker exec "$CTR" bash -c "$RUNENV; TILELANG_CACHE_DIR=\$(mktemp -d) timeout 300 python3 /tmp/d_dw_910b.py --variant $V --dump $D 2>&1 | grep -v 'TileLang begins\|TileLang completes' | tail -20"
            docker cp "$CTR:$D" "$HERE/$(basename "$D")" && echo "-> $HERE/$(basename "$D")" ;;
  --golden) run_one /tmp/d_dw_golden.py ;;
  --npz)    run_one "/tmp/d_dw_golden.py --npz /tmp/dw_pairs.npz"
            docker cp "$CTR:/tmp/dw_pairs.npz" "$HERE/dw_pairs.npz" && echo "-> $HERE/dw_pairs.npz" ;;
  *)        grep '^# ' "$HERE/run_D.sh" | head -11; exit 2 ;;
esac
