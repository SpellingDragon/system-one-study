#!/bin/bash
# run_window6_local.sh — 窗六：真形档 0.8B 一步（B 档 8192×1；超时/失败自动降 4096，再降 2048）
K=/Users/pengweiye/Documents/codes/system-one/release/.venv/KeyPair-19e0.pem
D=/Users/pengweiye/Documents/codes/system-one/release/ascend/port910b
H=ma-user@dev-modelarts-cnsouth1.huaweicloud.com
S="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ServerAliveInterval=20 -i $K -p 31955"
chmod 400 "$K" 2>/dev/null
cat > /tmp/w6_card.sh <<'EOS'
#!/bin/bash
set -u
cd /tmp/sys1 && git pull -q origin main 2>/dev/null
cd /tmp/sys1/release
source /usr/local/Ascend/cann-8.5.2/set_env.sh 2>/dev/null
export ASCEND_NPU_ARCH=dav-2201
export BISHENG_HOME=$(dirname $(dirname $(command -v bisheng)))
for SEQ in 8192 4096 2048; do
  echo "=== W6-SEQ=$SEQ batch=1 ==="
  export TILELANG_CACHE_DIR=/tmp/w6_cache_$SEQ
  mkdir -p $TILELANG_CACHE_DIR
  timeout 1500 python3 ascend/train_step_real.py --target ascend --seq $SEQ --batch 1 > /tmp/w6_run_$SEQ.log 2>&1
  RC=$?
  echo "W6-SEQ=$SEQ EXIT=$RC"
  grep -E "^P2-REAL-|^COUNTS |^SWAP |KERNEL-USED|DTYPE-TRACE|GDN-G-SPREAD|NOFACADE" /tmp/w6_run_$SEQ.log | head -40
  [ $RC -eq 0 ] && { echo "W6-DONE-AT=$SEQ"; break; }
  echo "W6-DEGRADE from=$SEQ（tail 见下）"; tail -4 /tmp/w6_run_$SEQ.log
done
echo "W6-DONE"
EOS
cat /tmp/w6_card.sh | ssh -T $S $H 'cat > /tmp/w6_card.sh; echo REMOTE=$(wc -c < /tmp/w6_card.sh)' 2>/dev/null
ssh -T $S $H ': > /tmp/w6_all.log; (setsid bash /tmp/w6_card.sh > /tmp/w6_all.log 2>&1 < /dev/null &); echo LAUNCHED; sleep 15; head -3 /tmp/w6_all.log' 2>/dev/null
echo "--- 同步等待（上限 28 分钟；60s 一轮）---"
for i in $(seq 1 28); do
  R=$(ssh -T $S $H 'echo "$(grep -c "W6-DONE" /tmp/w6_all.log) $(grep -oE "W6-SEQ=[0-9]+ EXIT=[0-9]+" /tmp/w6_all.log | tail -1) $(ls /tmp/w6_run_*.log 2>/dev/null | wc -l) $(pgrep -fc python3 || echo 0) $(npu-smi info 2>/dev/null | grep -m1 HBM | grep -oE "[0-9]+ / 32768" | head -1)"' 2>/dev/null)
  echo "[t=$((i*60))s] $R"
  echo "$R" | grep -q "^1 " && { echo COMPLETED; break; }
  sleep 60
done
echo "--- 窗六判决回读 ---"
ssh -T $S $H 'grep -nE "W6-|P2-REAL-|COUNTS |SWAP |KERNEL-USED|DTYPE-TRACE|Traceback|Error" /tmp/w6_all.log | tail -60' 2>/dev/null
