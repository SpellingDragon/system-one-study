#!/bin/bash
# run_window4_local.sh — 宿主机发射器（第四窗）。与 run_window3_local.sh 刻意保持"复制而非重构"：
# 付费窗口前不动已验证过的发射链，日志前缀 w4_* 与段落标记不同，合并反而引入风险。
K=/Users/pengweiye/Documents/codes/system-one/release/.venv/KeyPair-19e0.pem
D=/Users/pengweiye/Documents/codes/system-one/release/ascend/port910b
H=ma-user@dev-modelarts-cnsouth1.huaweicloud.com
S="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ServerAliveInterval=20 -i $K -p 31955"
chmod 400 "$K" 2>/dev/null
for f in card_run.sh window4_card.sh; do
  L=$(wc -c < "$D/$f"); echo "LOCAL $f=$L"; [ "$L" -lt 2000 ] && { echo "TRUNCATED-ABORT"; exit 1; }
  cat "$D/$f" | ssh -T $S $H "cat > /tmp/$f; echo REMOTE=\$(wc -c < /tmp/$f)" 2>/dev/null
done
ssh -T $S $H 'echo "python3=$(pgrep -fc python3 2>/dev/null || echo 0)"; npu-smi info 2>/dev/null | grep -m1 "910B" || echo NPU-NOT-FOUND; ls -d /tmp/sys1 2>/dev/null && echo TREE-EXISTS（会先 pull）|| echo TREE-ABSENT（脚本内 clone）' 2>/dev/null
ssh -T $S $H 'rm -f /tmp/w4_*.log /tmp/w4_artifacts.tgz; : > /tmp/w4_all.log
(setsid bash /tmp/window4_card.sh > /tmp/w4_all.log 2>&1 < /dev/null &)
echo LAUNCHED; sleep 20; head -6 /tmp/w4_all.log' 2>/dev/null

echo "--- 同步等待（上限 30 分钟；45s 一轮，按内层日志判进度）---"
QUIET=0; LAST=""
for i in $(seq 1 40); do
  R=$(ssh -T $S $H 'A=$(grep -c W4-DONE /tmp/w4_all.log 2>/dev/null)
SEG=$(grep -oE "^=== V[1-5] " /tmp/w4_all.log 2>/dev/null | tail -1 | tr -d "= ")
P=$(pgrep -fc python3 2>/dev/null || echo 0)
S1=$(wc -l < /tmp/w4_gradcheck.log 2>/dev/null || echo 0); S2=$(wc -l < /tmp/w4_triage.log 2>/dev/null || echo 0)
S3=$(wc -l < /tmp/w4_deltabwd.log 2>/dev/null || echo 0); S4=$(cat /tmp/w4_d3_ub.log /tmp/w4_d3_pure.log 2>/dev/null | wc -l)
S5=$(cat /tmp/w4_v2_before.log /tmp/w4_v2_after.log 2>/dev/null | wc -l)
echo "$A|$SEG|$P|$S1/$S2/$S3/$S4/$S5"' 2>/dev/null)
  DONE=$(echo "$R" | cut -d"|" -f1); SEG=$(echo "$R" | cut -d"|" -f2); P=$(echo "$R" | cut -d"|" -f3); CS=$(echo "$R" | cut -d"|" -f4)
  echo "[t=$((i*45))s] done=$DONE 段=${SEG:-起} py=$P 内层行=$CS"
  [ "$DONE" = "1" ] && { echo COMPLETED; break; }
  if [ "$P" = "0" ] && [ "$CS" = "$LAST" ] && [ -n "$LAST" ]; then QUIET=$((QUIET+1)); else QUIET=0; fi
  LAST=$CS
  [ "$QUIET" -ge 10 ] && { echo "STALLED-7.5分钟全静默——如实停报"; break; }
  sleep 45
done

echo "--- 五段判决回读 ---"
ssh -T $S $H 'grep -nE "W4-ENV|W4-V[1-5][A-Z-]*-EXIT|W4-V5-(PATCHED|SKIP|RESTORED)|^GRADCHECK-|VERDICT|raise-frame|^ITEM |^NUM-|^D3-|DBETA-FP|dbeta-fp|gdnb keys|STALE-DIFF|U2-READOUT|unexpected|FATAL|Traceback|W4-TGZ|W4-DONE" /tmp/w4_all.log | tail -80
echo "──── 末 8 行 ────"; tail -8 /tmp/w4_all.log' 2>/dev/null
