#!/bin/bash
# run_window3_local.sh — 宿主机发射器：上传两脚本 → setsid 起跑 → 同步等待 W3-DONE → 回读五段判决
# 监控教训固化（窗二之鉴）：A 段输出重定向进 w3_cardrun.log，外层行数不动≠停滞；
# 故每轮同时报【外层段落标记 + 当前段内层日志行数 + python3 进程数】，停滞判据＝三者皆静默且非段间隙。
K=/Users/pengweiye/Documents/codes/system-one/release/.venv/KeyPair-19e0.pem
D=/Users/pengweiye/Documents/codes/system-one/release/ascend/port910b
H=ma-user@dev-modelarts-cnsouth1.huaweicloud.com
S="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ServerAliveInterval=20 -i $K -p 31955"
chmod 400 "$K" 2>/dev/null

for f in card_run.sh window3_card.sh; do
  L=$(wc -c < "$D/$f"); echo "LOCAL $f=$L"
  [ "$L" -lt 2000 ] && { echo "TRUNCATED-ABORT $f"; exit 1; }
  cat "$D/$f" | ssh -T $S $H "cat > /tmp/$f; echo REMOTE=\$(wc -c < /tmp/$f)" 2>/dev/null
done
ssh -T $S $H 'echo "python3=$(pgrep -fc python3 2>/dev/null || echo 0)"; npu-smi info 2>/dev/null | grep -m1 "910B" || echo NPU-NOT-FOUND' 2>/dev/null
ssh -T $S $H 'rm -f /tmp/w3_*.log /tmp/w3_artifacts.tgz; : > /tmp/w3_all.log
(setsid bash /tmp/window3_card.sh > /tmp/w3_all.log 2>&1 < /dev/null &)
echo LAUNCHED; sleep 20; head -4 /tmp/w3_all.log' 2>/dev/null

echo "--- 同步等待（上限 40 分钟；45s 一轮）---"
QUIET=0
for i in $(seq 1 53); do
  R=$(ssh -T $S $H 'A=$(grep -c W3-DONE /tmp/w3_all.log 2>/dev/null)
SEG=$(grep -oE "^=== [A-E] " /tmp/w3_all.log 2>/dev/null | tail -1 | tr -d "= ")
P=$(pgrep -fc python3 2>/dev/null || echo 0)
C1=$(wc -l < /tmp/w3_cardrun.log 2>/dev/null || echo 0)
C2=$(wc -l < /tmp/w3_gradcheck.log 2>/dev/null || echo 0)
C3=$(wc -l < /tmp/w3_deltabwd.log 2>/dev/null || echo 0)
C4=$(wc -l < /tmp/w3_d2legs.log 2>/dev/null || echo 0)
C5=$(wc -l < /tmp/w3_s3.log 2>/dev/null || echo 0)
echo "$A|$SEG|$P|$C1/$C2/$C3/$C4/$C5"' 2>/dev/null)
  DONE=$(echo "$R" | cut -d"|" -f1); SEG=$(echo "$R" | cut -d"|" -f2); P=$(echo "$R" | cut -d"|" -f3); CS=$(echo "$R" | cut -d"|" -f4)
  echo "[t=$((i*45))s] done=$DONE 段=${SEG:-起} py=$P 内层行=$CS"
  [ "$DONE" = "1" ] && { echo COMPLETED; break; }
  if [ "$P" = "0" ] && [ "$CS" = "$(echo $LAST_CS 2>/dev/null)" ] && [ -n "$LAST_CS" ]; then QUIET=$((QUIET+1)); else QUIET=0; fi
  LAST_CS=$CS
  [ "$QUIET" -ge 12 ] && { echo "STALLED-9分钟全静默（非段间隙量级）——如实停报"; break; }
  sleep 45
done

echo "--- 五段判决回读 ---"
ssh -T $S $H 'grep -nE "W3-[A-E][0-9]?-EXIT|NPU-REG|ASC-PROD|^Y-.*PROD|P2-STEP|DTYPE-TRACE|KERNEL-USED|KERNELS-TGZ|RUNNER-DONE|^GRADCHECK-|^NUM-|SO-VERDICT|LEG-VERDICT|LEGS-SUMMARY|inject|STALE-DIFF|U2-READOUT|unexpected|FATAL|W3-TGZ|W3-DONE|Traceback" /tmp/w3_all.log | tail -80
echo "──── 末 8 行 ────"; tail -8 /tmp/w3_all.log' 2>/dev/null
