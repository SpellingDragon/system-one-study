#!/bin/bash
# ssh_diag.sh — 分层诊断：TCP 通不通 → ssh 握手到哪断
K=/Users/pengweiye/Documents/codes/system-one/release/.venv/KeyPair-19e0.pem
H=dev-modelarts-cnsouth1.huaweicloud.com
P=31955
echo "== TCP 探测 =="
nc -vz -G 8 $H $P 2>&1 | tail -2
echo "== ssh verbose（只看到 banner/kex 为止）=="
ssh -v -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o BatchMode=yes -i $K -p $P ma-user@$H 'echo SSH-OK' 2>&1 | grep -E "Connecting|Connection|banner|kex|Authenticat|closed|refused|timed" | head -8
