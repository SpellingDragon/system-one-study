#!/bin/bash
# launch_card_local.sh — 宿主机侧：验连 → 上传 card_run.sh → 校验尺寸 → setsid 起跑（时钟敏感，一次过）
K=/Users/pengweiye/Documents/codes/system-one/release/.venv/KeyPair-19e0.pem
F=/Users/pengweiye/Documents/codes/system-one/release/ascend/port910b/card_run.sh
H="ma-user@dev-modelarts-cnsouth1.huaweicloud.com"
chmod 400 "$K" 2>/dev/null
S="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -i $K -p 31955"
echo "--- conn ---"
ssh -T $S $H "echo CONN-OK; npu-smi info 2>/dev/null | grep -m1 910B || true" 2>/dev/null
L=$(wc -c < "$F"); echo "LOCAL-SIZE=$L"
[ "$L" -lt 1500 ] && { echo "LOCAL-TRUNCATED-ABORT"; exit 1; }
cat "$F" | ssh -T $S $H "cat > /tmp/card_run.sh; R=\$(wc -c < /tmp/card_run.sh); echo CARD-SIZE=\$R; if [ \$R -gt 1500 ]; then (setsid bash /tmp/card_run.sh > /tmp/card_run.log 2>&1 < /dev/null &); sleep 4; echo LAUNCHED; head -3 /tmp/card_run.log; else echo UPLOAD-TRUNCATED; fi" 2>/dev/null
