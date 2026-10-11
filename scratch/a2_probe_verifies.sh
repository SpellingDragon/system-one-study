#!/usr/bin/env bash
# A2 同查：attempts 侧各 verify_*.py 能否稳定打出判决行（债一同款缺口排查）
set -u
source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null 2>&1
export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201
cd /work
for f in ascend/kernels/reconcile/verify_*.py; do
  b=$(basename "$f")
  S=$(mktemp -d)
  OUT=$(TILELANG_CACHE_DIR=$S timeout 600 python3 "$f" 2>&1); RC=$?
  V=$(echo "$OUT" | grep -E "(ASC-COMPILE|CPU-SEMANTIC|VERIFY-ONLY|STRUCTURE-SANITY)" | tr '\n' '|')
  T=$(echo "$OUT" | grep -Ec "Traceback")
  echo "PROBE $b rc=$RC traceback=$T verdict=${V:-<NONE>}"
done
