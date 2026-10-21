#!/bin/bash
# window7_card.sh — 第七窗：M4 修复判决窗（fresh 六项 + 指纹收敛 + 修后 gradcheck + w6 捞档）
# 三臂映射：A1(预清零)＝本窗 fresh 跑（M4 已在码内）；A0 基线＝窗四归档数字；conv 预测＝仍 FAIL
set -u
T=/tmp/sys1
CR=/tmp/card_run.sh
[ -d $T/release ] || { git clone -q --depth 1 https://github.com/SpellingDragon/system-one-study.git $T || { echo W7-ABORT-CLONE; exit 1; }; }
cd $T && git pull -q origin main 2>/dev/null
cd $T/release
CARDRUN_SKIP_CLONE=1 CARDRUN_STOP_AFTER=2 bash $CR > /tmp/w7_boot.log 2>&1 || true
grep -E "OVERLAY-OK|FATAL" /tmp/w7_boot.log | tail -2
python3 -c "import tilelang" 2>/dev/null || { echo "W7-ABORT no-tilelang"; exit 1; }
source /usr/local/Ascend/cann-8.5.2/set_env.sh 2>/dev/null
export ASCEND_NPU_ARCH=dav-2201
export BISHENG_HOME=$(dirname $(dirname $(command -v bisheng)))
echo "W7-HEAD=$(git -C $T rev-parse --short HEAD)"

echo "=== [1] w6 捞档（重启后可能已没）==="
ls -la /tmp/w6_all.log 2>/dev/null && tail -6 /tmp/w6_all.log || echo "w6 log 不在（重启清了 /tmp）"

echo "=== [2] M4 修复判决：fresh 六项 ITEM（期望 dbeta 0.80 → <2e-4）==="
export TILELANG_CACHE_DIR=$(mktemp -d)
timeout 900 python3 ascend/port910b/run_delta_card.py --target ascend --bwd > /tmp/w7_deltabwd.log 2>&1
echo "W7-V2-EXIT=$?"; grep -E "^ITEM|^NUM-" /tmp/w7_deltabwd.log

echo "=== [3] D3 顺序探针（期望 none 指纹收敛到 zero 指纹）==="
export TILELANG_CACHE_DIR=$(mktemp -d)
timeout 600 python3 ascend/kernels/reconcile/D3_probe_dbeta_order.py --dev npu --variant ub --target ascend > /tmp/w7_d3ub.log 2>&1
echo "W7-V3-EXIT=$?"; grep -E "^D3-ORDER|gdnb keys|^ITEM|^DBETA-FP|dbeta-fp" /tmp/w7_d3ub.log | head -14

echo "=== [4] 修后 gradcheck（修判据后；conv 预测仍 FAIL，gdn_delta 观察）==="
export TILELANG_CACHE_DIR=$(mktemp -d)
timeout 900 python3 ascend/gradcheck_card.py --target ascend > /tmp/w7_gradcheck.log 2>&1
echo "W7-V4-EXIT=$?"; grep -E "^GRADCHECK-" /tmp/w7_gradcheck.log

echo "=== 现场保全 ==="
tar czf /tmp/w7_artifacts.tgz -C /tmp w7_boot.log w7_deltabwd.log w7_d3ub.log w7_gradcheck.log 2>/dev/null
echo "W7-TGZ size=$(wc -c </tmp/w7_artifacts.tgz)B"; echo "W7-DONE"
