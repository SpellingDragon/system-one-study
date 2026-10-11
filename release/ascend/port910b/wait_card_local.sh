#!/bin/bash
# wait_card_local.sh — 宿主机：上传 card_run.sh → setsid 起跑 → **同步等待** RUNNER-DONE → 回读判决行
# 教训固化：起跑守卫不得用 pgrep -f 'card_run.sh'（远端 bash -c 自身命令行含该串会自匹配，
# 曾导致"误判已有实例→根本没起跑→日志空"）。改判 python3 工作进程数；setsid 会 fork-exit，
# 故其 pid 不可作为存活依据，存活以"日志行数是否增长 + 是否出现 RUNNER-DONE"为准。
K=/Users/pengweiye/Documents/codes/system-one/release/.venv/KeyPair-19e0.pem
F=/Users/pengweiye/Documents/codes/system-one/release/ascend/port910b/card_run.sh
H=ma-user@dev-modelarts-cnsouth1.huaweicloud.com
S="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ServerAliveInterval=20 -i $K -p 31955"
LOG=/tmp/card_run.log
chmod 400 "$K" 2>/dev/null

L=$(wc -c < "$F"); echo "LOCAL-SIZE=$L"
[ "$L" -lt 2000 ] && { echo "LOCAL-TRUNCATED-ABORT"; exit 1; }

cat "$F" | ssh -T $S $H 'cat > /tmp/card_run.sh
R=$(wc -c < /tmp/card_run.sh); echo CARD-SIZE=$R
[ "$R" -lt 2000 ] && { echo UPLOAD-TRUNCATED; exit 1; }
P=$(pgrep -fc python3 2>/dev/null || echo 0)
if [ "$P" -gt 2 ]; then echo "PREEXISTING-python3=$P (不重复起跑，先看现况)"; else
  : > /tmp/card_run.log
  (setsid bash /tmp/card_run.sh > /tmp/card_run.log 2>&1 < /dev/null &)
  echo LAUNCHED
fi
sleep 15
echo "--- 首 10 行 ---"; head -10 /tmp/card_run.log
echo "--- 工作进程 ---"; echo "python3_procs=$(pgrep -fc python3 2>/dev/null || echo 0)"' 2>/dev/null

echo "--- 同步等待（上限 25 分钟，45s 一轮）---"
PREV=-1; STALL=0
for i in $(seq 1 34); do
  LINE=$(ssh -T $S $H 'echo "$(grep -c RUNNER-DONE /tmp/card_run.log 2>/dev/null) $(wc -l < /tmp/card_run.log 2>/dev/null) $(pgrep -fc python3 2>/dev/null || echo 0)"' 2>/dev/null)
  DONE=$(echo "$LINE" | awk '{print $1+0}'); SZ=$(echo "$LINE" | awk '{print $2+0}'); PP=$(echo "$LINE" | awk '{print $3+0}')
  echo "[t=$((i*45))s] done=$DONE loglines=$SZ python3=$PP"
  [ "$DONE" -ge 1 ] && { echo "COMPLETED"; break; }
  if [ "$SZ" = "$PREV" ]; then STALL=$((STALL+1)); else STALL=0; fi
  PREV=$SZ
  [ "$STALL" -ge 6 ] && { echo "STALLED-6轮无进展(=${STALL}x45s) — 现场已冻结，如实报"; break; }
  [ "$PP" -eq 0 ] && [ "$i" -gt 3 ] && { echo "NO-WORKER-BEFORE-DONE"; break; }
  sleep 45
done

echo "--- 判决行回读 ---"
ssh -T $S $H 'grep -nE "NPU-REG|PROD|NUM-|ASC-|P2-STEP-|DTYPE-TRACE|KERNEL-USED|KERNELS-TGZ|OVERLAY-OK|INJECT-OK|CASE-|RUNNER-DONE|FATAL|Traceback" /tmp/card_run.log | tail -60
echo "──── 末 15 行原样 ────"; tail -15 /tmp/card_run.log' 2>/dev/null
