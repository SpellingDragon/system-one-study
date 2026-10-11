#!/bin/bash
# poll_window2_local.sh — 只等不回炉（A 段输出在内层 w2_cardrun.log，外层行数不动属设计使然）
K=/Users/pengweiye/Documents/codes/system-one/release/.venv/KeyPair-19e0.pem
H=ma-user@dev-modelarts-cnsouth1.huaweicloud.com
S="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ServerAliveInterval=20 -i $K -p 31955"
chmod 400 "$K" 2>/dev/null
echo "--- 续等（上限 25 分钟；每轮报内层进度）---"
PREV=-1; SAME=0
for i in $(seq 1 33); do
  R=$(ssh -T $S $H 'D=$(grep -c W2-DONE /tmp/w2_all.log 2>/dev/null); N=$(grep -c RUNNER-DONE /tmp/w2_cardrun.log 2>/dev/null); L=$(wc -l < /tmp/w2_cardrun.log 2>/dev/null); T=$(grep -oE "CASE-[A-Za-z0-9_-]+" /tmp/w2_cardrun.log 2>/dev/null | tail -1); P=$(grep -oE "=== [ABCD] " /tmp/w2_all.log 2>/dev/null | tail -1); echo "$D $N $L ${T:-?} ${P:-?} $(pgrep -fc python3 2>/dev/null || echo 0)"' 2>/dev/null)
  DONE=$(echo "$R" | awk '{print $1+0}'); INNER=$(echo "$R" | awk '{print $3+0}')
  echo "[t=$((i*45))s] 完成=$DONE 内层行=$INNER 段=$(echo "$R" | awk '{print $5}') 腿=$(echo "$R" | awk '{print $4}') py=$(echo "$R" | awk '{print $6}')"
  [ "$DONE" -ge 1 ] && { echo COMPLETED; break; }
  if [ "$INNER" = "$PREV" ]; then SAME=$((SAME+1)); else SAME=0; fi
  PREV=$INNER
  [ "$SAME" -ge 12 ] && { echo "真停滞-12轮内层无增长"; break; }
  sleep 45
done
echo "--- 四段判决全量回读 ---"
ssh -T $S $H 'echo "### A 段（内层）"; grep -E "NPU-REG|ASC-PROD|^Y-|P2-STEP|DTYPE-TRACE|KERNEL-USED|KERNELS-TGZ|RUNNER-DONE|^NUM-" /tmp/w2_cardrun.log 2>/dev/null | tail -30
echo "### B/C/D 段（外层）"; grep -nE "W2-[ABCD][0-9]?-EXIT|^BENCH-|^U2-|md5_after_revert|md5_restored|P2-STEP|DTYPE-TRACE|W2-TGZ|W2-DONE|Traceback|FATAL|not found|No such" /tmp/w2_all.log 2>/dev/null | tail -60' 2>/dev/null
