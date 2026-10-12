#!/bin/bash
# run_window2_local.sh — 宿主机：上传 window2 采数序列 → setsid 起跑 → 同步等待 W2-DONE → 回读四段判决
K=/Users/pengweiye/Documents/codes/system-one/release/.venv/KeyPair-19e0.pem
D=/Users/pengweiye/Documents/codes/system-one/release/ascend/port910b
H=ma-user@dev-modelarts-cnsouth1.huaweicloud.com
S="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ServerAliveInterval=20 -i $K -p 31955"
chmod 400 "$K" 2>/dev/null
for f in card_run.sh window2_card.sh; do
  L=$(wc -c < "$D/$f"); echo "LOCAL $f=$L"
  [ "$L" -lt 500 ] && { echo "TRUNCATED-ABORT $f"; exit 1; }
done
cat "$D/card_run.sh" | ssh -T $S $H 'cat > /tmp/card_run.sh; wc -c < /tmp/card_run.sh' 2>/dev/null
cat "$D/window2_card.sh" | ssh -T $S $H 'cat > /tmp/window2_card.sh; wc -c < /tmp/window2_card.sh' 2>/dev/null
ssh -T $S $H 'P=$(pgrep -fc python3 2>/dev/null || echo 0); echo "python3=$P"; nvidia-smi -L 2>/dev/null | head -1; npu-smi info 2>/dev/null | grep -m1 "910B\|H9" || echo "NPU-SMI none"' 2>/dev/null
ssh -T $S $H ': > /tmp/w2_all.log; (setsid bash /tmp/window2_card.sh > /tmp/w2_all.log 2>&1 < /dev/null &); echo LAUNCHED; sleep 20; head -4 /tmp/w2_all.log' 2>/dev/null

echo "--- 同步等待（上限 28 分钟）---"
PREV=-1; STALL=0
for i in $(seq 1 37); do
  R=$(ssh -T $S $H 'echo "$(grep -c W2-DONE /tmp/w2_all.log 2>/dev/null) $(wc -l < /tmp/w2_all.log 2>/dev/null) $(pgrep -fc python3 2>/dev/null || echo 0)"' 2>/dev/null)
  DONE=$(echo "$R" | awk '{print $1+0}'); SZ=$(echo "$R" | awk '{print $2+0}'); PP=$(echo "$R" | awk '{print $3+0}')
  SEG=$(ssh -T $S $H 'grep -oE "=== [ABCD] [^=]*" /tmp/w2_all.log | tail -1' 2>/dev/null)
  echo "[t=$((i*45))s] done=$DONE lines=$SZ py=$PP 当前段=${SEG:-?}"
  [ "$DONE" -ge 1 ] && { echo COMPLETED; break; }
  if [ "$SZ" = "$PREV" ]; then STALL=$((STALL+1)); else STALL=0; fi
  PREV=$SZ
  [ "$STALL" -ge 7 ] && { echo "STALL-7轮无进展"; break; }
  sleep 45
done

echo "--- 四段判决回读 ---"
ssh -T $S $H 'grep -nE "W2-[ABCD][0-9]?-EXIT|W2-A-EXIT|NPU-REG|ASC-PROD|P2-STEP|DTYPE-TRACE|KERNEL-USED|KERNELS-TGZ|RUNNER-DONE|^BENCH-|^U2-|md5_after_revert|md5_restored|W2-TGZ|W2-DONE|Traceback|FATAL" /tmp/w2_all.log | tail -70
echo "──── 末 10 行 ────"; tail -10 /tmp/w2_all.log' 2>/dev/null
