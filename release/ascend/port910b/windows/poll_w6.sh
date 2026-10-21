#!/bin/bash
# poll_w6.sh — 续等窗六降级档（4096/2048）结果
K=$REPO_ROOT/release/.venv/${CARD_KEY}
S="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -i $K -p 31955"
H=${CARD_SSH}
for i in $(seq 1 25); do
  R=$(ssh -T $S $H 'echo "$(grep -c "W6-DONE" /tmp/w6_all.log 2>/dev/null)|$(grep -oE "W6-SEQ=[0-9]+ EXIT=[0-9]+" /tmp/w6_all.log | tail -1)|py=$(pgrep -fc python3 || echo 0)|hbm=$(npu-smi info 2>/dev/null | grep -m1 HBM-Usage | grep -oE "[0-9]+ / 32768")"' 2>/dev/null)
  echo "[t=$((i*60))s] $R"
  echo "$R" | grep -q "^1|" && { echo COMPLETED; break; }
  sleep 60
done
echo "--- 窗六全量判决 ---"
ssh -T $S $H 'grep -nE "W6-|P2-REAL-|^COUNTS |^SWAP |KERNEL-USED|DTYPE-TRACE|GDN-G-SPREAD" /tmp/w6_all.log | tail -70' 2>/dev/null
