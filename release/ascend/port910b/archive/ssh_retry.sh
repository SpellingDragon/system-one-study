#!/bin/bash
# ssh_retry.sh — 网关 502 重试（每 60s×6 次；成功即停）
K=$REPO_ROOT/release/.venv/${CARD_KEY}
for i in 1 2 3 4 5 6; do
  OUT=$(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=12 -i $K -p 31955 ${CARD_SSH} 'echo SSH-OK; tail -3 /tmp/w6_all.log 2>/dev/null' 2>&1 | tail -4)
  echo "[try$i $(date +%H:%M:%S)] $OUT"
  echo "$OUT" | grep -q "SSH-OK" && { echo RECOVERED; exit 0; }
  sleep 60
done
echo STILL-502
exit 1
