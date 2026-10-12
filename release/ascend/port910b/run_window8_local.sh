#!/bin/bash
# run_window8_local.sh — 发射器（上传两脚本 → setsid → 同步等 W8-DONE → 回读）
K=/Users/pengweiye/Documents/codes/system-one/release/.venv/KeyPair-19e0.pem
D=/Users/pengweiye/Documents/codes/system-one/release/ascend/port910b
H=ma-user@dev-modelarts-cnsouth1.huaweicloud.com
S="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ServerAliveInterval=20 -i $K -p 31955"
chmod 400 "$K" 2>/dev/null
for f in card_run.sh window8_card.sh; do
  L=$(wc -c < "$D/$f"); echo "LOCAL $f=$L"
  cat "$D/$f" | ssh -T $S $H "cat > /tmp/$f; echo REMOTE=\$(wc -c < /tmp/$f)" 2>/dev/null
done
ssh -T $S $H 'rm -f /tmp/w8_*.log /tmp/w8_artifacts.tgz; : > /tmp/w8_all.log
(setsid bash /tmp/window8_card.sh > /tmp/w8_all.log 2>&1 < /dev/null &)
echo LAUNCHED; sleep 20; head -4 /tmp/w8_all.log' 2>/dev/null
echo "--- 同步等待（上限 25 分钟）---"
for i in $(seq 1 33); do
  R=$(ssh -T $S $H 'echo "$(grep -c W8-DONE /tmp/w8_all.log 2>/dev/null)|$(grep -oE "^=== \[[1-4]\]" /tmp/w8_all.log 2>/dev/null | tail -1)|py=$(pgrep -fc python3 2>/dev/null || echo 0)|L=$(cat /tmp/w8_deltabwd.log /tmp/w8_d3ub.log /tmp/w8_gradcheck.log 2>/dev/null | wc -l)"' 2>/dev/null)
  echo "[t=$((i*45))s] $R"
  echo "$R" | grep -q "^1|" && { echo COMPLETED; break; }
  sleep 45
done
echo "--- 窗七判决回读 ---"
ssh -T $S $H 'grep -nE "W7-|OVERLAY-OK|^ITEM |^NUM-|gdnb keys|DBETA-FP|dbeta-fp|^GRADCHECK-|w6 log|Traceback|FATAL" /tmp/w8_all.log | tail -60' 2>/dev/null
