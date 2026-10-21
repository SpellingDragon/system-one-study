#!/bin/bash
# card-side fixup runner (per-case subprocess isolation already via --only)
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh 2>/dev/null || source /usr/local/Ascend/cann-8.5.2/set_env.sh
CCEC=$(find /usr/local/Ascend -path "*ccec_compiler/bin/bisheng" | head -1)
export BISHENG_HOME=$(dirname $(dirname "$CCEC")) ASCEND_NPU_ARCH=dav-2201
cd /tmp/sys1 && git pull -q origin main
grep -c TORCH_DT release/ascend/port910b/run_numerics.py
for c in gemm_l1 rope attnsw dw; do
  echo "CASE-$c"
  timeout 420 python3 release/ascend/port910b/run_numerics.py --only $c 2>&1 | grep -E "NUM-|FAIL|Error:" | head -3
done
echo FIXUP-DONE
