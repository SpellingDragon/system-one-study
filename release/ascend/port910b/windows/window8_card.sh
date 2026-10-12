#!/bin/bash
# window8_card.sh — 第八窗：A2 预热修复判决窗（A0 对照 → A2 默认 → 修后 gradcheck → 顺序探针收敛）
set -u
T=/tmp/sys1
CR=/tmp/card_run.sh
[ -d $T/release ] || { git clone -q --depth 1 https://github.com/SpellingDragon/system-one-study.git $T || { echo W8-ABORT-CLONE; exit 1; }; }
cd $T && git pull -q origin main 2>/dev/null
cd $T/release
CARDRUN_SKIP_CLONE=1 CARDRUN_STOP_AFTER=2 bash $CR > /tmp/w8_boot.log 2>&1 || true
python3 -c "import tilelang" 2>/dev/null || { echo "W8-ABORT no-tilelang"; exit 1; }
source /usr/local/Ascend/cann-8.5.2/set_env.sh 2>/dev/null
export ASCEND_NPU_ARCH=dav-2201
export BISHENG_HOME=$(dirname $(dirname $(command -v bisheng)))
echo "W8-HEAD=$(git -C $T rev-parse --short HEAD)"

echo "=== [1] A0 对照臂（SYS1_DISABLE_WARMUP=1，期望 dbeta 复发 0.80 证明因果）==="
export TILELANG_CACHE_DIR=$(mktemp -d)
SYS1_DISABLE_WARMUP=1 timeout 900 python3 ascend/port910b/run_delta_card.py --target ascend --bwd > /tmp/w8_a0.log 2>&1
echo "W8-V1-EXIT=$?"; grep -cE "^WARMUP" /tmp/w8_a0.log | sed 's/^/WARMUP行=/'; grep -E "^ITEM dbeta|^NUM-delta_bwd" /tmp/w8_a0.log

echo "=== [2] A2 默认臂（期望 WARMUP 行 + dbeta <2e-4 六项全 PASS）==="
export TILELANG_CACHE_DIR=$(mktemp -d)
timeout 900 python3 ascend/port910b/run_delta_card.py --target ascend --bwd > /tmp/w8_a2.log 2>&1
echo "W8-V2-EXIT=$?"; grep -E "^WARMUP" /tmp/w8_a2.log | head -4; grep -E "^ITEM|^NUM-delta_bwd" /tmp/w8_a2.log

echo "=== [3] 修后 gradcheck（期望 gdn_delta/gdn_conv 转 PASS ⇒ 8/8；conv 若仍红则独立根因坐实）==="
export TILELANG_CACHE_DIR=$(mktemp -d)
timeout 900 python3 ascend/gradcheck_card.py --target ascend > /tmp/w8_gradcheck.log 2>&1
echo "W8-V3-EXIT=$?"; grep -E "^GRADCHECK-" /tmp/w8_gradcheck.log

echo "=== [4] 顺序探针（期望 none 指纹 == dirty == zero）==="
export TILELANG_CACHE_DIR=$(mktemp -d)
timeout 600 python3 ascend/kernels/reconcile/D3_probe_dbeta_order.py --dev npu --variant ub --target ascend > /tmp/w8_d3.log 2>&1
echo "W8-V4-EXIT=$?"; grep -E "dbeta-fp|^ITEM dbeta" /tmp/w8_d3.log | head -5

echo "=== 现场保全 ==="
tar czf /tmp/w8_artifacts.tgz -C /tmp w8_boot.log w8_a0.log w8_a2.log w8_gradcheck.log w8_d3.log 2>/dev/null
echo "W8-TGZ size=$(wc -c </tmp/w8_artifacts.tgz)B"; echo "W8-DONE"
