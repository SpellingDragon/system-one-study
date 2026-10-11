#!/usr/bin/env bash
# A2 摸底：run_numerics 全部 case 在容器（无设备）card --no-run 下的判决行
set -u
source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null 2>&1
export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201
cd /work/ascend/port910b
for c in linear_prod dw_prod readout_prod gemm_l1_intrinsic dw_intrinsic readout_intrinsic addln_intrinsic gemm_direct gdn rope attnsw; do
  S=$(mktemp -d)
  OUT=$(TILELANG_CACHE_DIR=$S timeout 600 python3 run_numerics.py --only "$c" --track card --no-run 2>&1)
  RC=$?
  LINE=$(echo "$OUT" | grep -E "^(Y-|NUM-)" | head -3 | tr '\n' '|')
  echo "SCAN $c rc=$RC line=${LINE:-<EMPTY>}"
done
