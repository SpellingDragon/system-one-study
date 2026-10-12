#!/bin/bash
# window2_card.sh — 评审后方向调整的四处采数（卡侧，一次性；不改任何判据/阈值/产品码语义）
# A 主验收链（fp16 判决 + 生成码打包）→ B 吞吐（硬预算闸）→ C pitch/gap 一枪 → D 反事实（撤避让必复现 nan）
# 纪律：四段各自独立进程；每段 EXIT 码与判决行都打；D 结束必须还原产品件并证 md5。
set -u
SRC=https://github.com/SpellingDragon/system-one-study.git
T=/tmp/sys1
CR=${CR:-/tmp/card_run.sh}

echo "=== A 主验收链（card_run.sh，bench 腿默认关）==="
timeout 1500 bash "$CR" > /tmp/w2_cardrun.log 2>&1; echo "W2-A-EXIT=$?"
grep -E "NPU-REG|ASC-PROD|Y-.*PROD|P2-STEP|DTYPE-TRACE|KERNEL-USED|KERNELS-TGZ|RUNNER-DONE|^NUM-.*delta" /tmp/w2_cardrun.log | tail -32

[ -d $T/release ] || { echo "W2-ABORT no-tree"; exit 1; }
cd $T/release
source /usr/local/Ascend/cann-8.5.2/set_env.sh 2>/dev/null
export ASCEND_NPU_ARCH=dav-2201
export BISHENG_HOME=$(dirname $(dirname $(command -v bisheng)))

echo "=== B 吞吐/报价（E2 工装；超预算自 SKIP，不出结论只出数）==="
export TILELANG_CACHE_DIR=$(mktemp -d)
( cd ascend && timeout 420 python3 bench_card.py --device npu --quick --iters 3 --warmup 1 --budget-sec 300 ) > /tmp/w2_bench.log 2>&1
echo "W2-B-EXIT=$?"; grep -E "^BENCH-(QUOTE|ABLAT|SUMMARY|NOTE|ABORT)" /tmp/w2_bench.log | head -12
grep -E "^BENCH-" /tmp/w2_bench.log | grep -vE "QUOTE|ABLAT|SUMMARY|NOTE|ABORT" | head -24
echo "--- 卡上时间戳 ---"; date "+%F %T"

echo "=== C 跨步装填语义一枪（pitch vs gap，U2 强钉梯）==="
export TILELANG_CACHE_DIR=$(mktemp -d)
timeout 300 python3 ascend/kernels/reconcile/U2_probe_k48_dma_semantics.py --dev npu > /tmp/w2_pitchgap.log 2>&1
echo "W2-C-EXIT=$?"; grep -E "U2-" /tmp/w2_pitchgap.log | head -34
echo "--- 卡上时间戳 ---"; date "+%F %T"

echo "=== D 反事实：撤 bk 避让纪律，fp16 必须复现 nan，随后还原 ==="
timeout 200 python3 ascend/kernels/reconcile/M2_probe_patch_gemm.py --revert >/tmp/w2_revert.log 2>&1; echo "W2-D0-EXIT=$? md5_after_revert=$(md5sum ascend/kernels/gemm_asc.py | cut -c1-8)"
export TILELANG_CACHE_DIR=$(mktemp -d)
timeout 600 python3 ascend/train_step.py --target ascend --dtype fp16 > /tmp/w2_d_fp16.log 2>&1; echo "W2-D1-EXIT=$?"
grep -E "P2-STEP|DTYPE-TRACE|KERNEL-USED" /tmp/w2_d_fp16.log
timeout 200 python3 ascend/kernels/reconcile/M2_probe_patch_gemm.py >/tmp/w2_apply.log 2>&1; echo "W2-D2-restore-EXIT=$? md5_restored=$(md5sum ascend/kernels/gemm_asc.py | cut -c1-8)"
grep -cE "6ef882df" /tmp/w2_apply.log || true
echo "=== 现场保全：打包生成码与四段日志 ==="
tar czf /tmp/w2_artifacts.tgz -C /tmp w2_cardrun.log w2_bench.log w2_pitchgap.log w2_d_fp16.log w2_revert.log w2_apply.log 2>/dev/null
echo "W2-TGZ size=$(wc -c </tmp/w2_artifacts.tgz)B"; echo "W2-DONE"
