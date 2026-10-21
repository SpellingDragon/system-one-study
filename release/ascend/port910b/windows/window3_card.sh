#!/bin/bash
# window3_card.sh — 第三窗：钉死窗（五段，各独立进程；判据全部来自已就绪的判决件，不改任何阈值）
# A 验收链（含 addln_prod 产品腿 + fp16 训练步）→ B 卡上 gradcheck（P1-1y 真机执行）
# C delta_bwd 六项梯度真机判决（P1-1w）→ D D2 变体腿 X1–X7（P910B_REPO 口径已订正）
# E S3 跨步第二层因素一枪（H-b UB 预填 / H-e 拆趟，`--prefills none,zero,perm`）
set -u
T=/tmp/sys1
CR=${CR:-/tmp/card_run.sh}

echo "=== A 验收链（card_run.sh）==="
timeout 1500 bash "$CR" > /tmp/w3_cardrun.log 2>&1; echo "W3-A-EXIT=$?"
grep -E "NPU-REG|ASC-PROD|^Y-|P2-STEP|DTYPE-TRACE|KERNEL-USED|KERNELS-TGZ|RUNNER-DONE|^NUM-delta" /tmp/w3_cardrun.log | tail -30
[ -d $T/release ] || { echo "W3-ABORT no-tree"; exit 1; }
cd $T/release
source /usr/local/Ascend/cann-8.5.2/set_env.sh 2>/dev/null
export ASCEND_NPU_ARCH=dav-2201
export BISHENG_HOME=$(dirname $(dirname $(command -v bisheng)))

echo "=== B 卡上 gradcheck（8 门面 × KERNEL-USED 绑定）==="
export TILELANG_CACHE_DIR=$(mktemp -d)
timeout 900 python3 ascend/gradcheck_card.py --target ascend > /tmp/w3_gradcheck.log 2>&1; echo "W3-B-EXIT=$?"
grep -E "^GRADCHECK-" /tmp/w3_gradcheck.log
echo "--- ts ---"; date "+%F %T"

echo "=== C delta 反向六项梯度真机判决 ==="
export TILELANG_CACHE_DIR=$(mktemp -d)
timeout 900 python3 ascend/port910b/run_delta_card.py --target ascend --bwd > /tmp/w3_deltabwd.log 2>&1; echo "W3-C-EXIT=$?"
grep -E "^NUM-|FAIL|Error" /tmp/w3_deltabwd.log | head -12
echo "--- ts ---"; date "+%F %T"

echo "=== D D2 变体腿（X1–X7 真值表；REPO 口径已订正为 release/ascend）==="
export P910B_REPO=$T/release/ascend
export TILELANG_CACHE_DIR=$(mktemp -d)
( cd ascend/port910b/attempts && timeout 900 python3 -u D2_probe_legs.py --combos default,A,C --card --cache-root /tmp/d2_card ) > /tmp/w3_d2legs.log 2>&1
echo "W3-D-EXIT=$?"
grep -E "inject|SO-VERDICT|LEG-VERDICT|LEGS-SUMMARY|NUM-|FATAL|child-rc" /tmp/w3_d2legs.log | tail -24
echo "--- ts ---"; date "+%F %T"

echo "=== E 跨步第二层因素一枪（H-b 预填对照 / H-e）==="
export TILELANG_CACHE_DIR=$(mktemp -d)
timeout 900 python3 ascend/kernels/reconcile/S3_probe_nextshot.py --dev npu --prefills none,zero,perm > /tmp/w3_s3.log 2>&1; echo "W3-E-EXIT=$?"
grep -E "U2-|S3-|STALE-DIFF|VERDICT|unexpected" /tmp/w3_s3.log | tail -30
echo "--- ts ---"; date "+%F %T"

echo "=== 现场保全 ==="
tar czf /tmp/w3_artifacts.tgz -C /tmp w3_cardrun.log w3_gradcheck.log w3_deltabwd.log w3_d2legs.log w3_s3.log 2>/dev/null
echo "W3-TGZ size=$(wc -c </tmp/w3_artifacts.tgz)B"; echo "W3-DONE"
