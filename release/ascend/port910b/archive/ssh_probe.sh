#!/bin/bash
# ssh_probe.sh — 探实例是否还活着（不吞 stderr）
K=/Users/pengweiye/Documents/codes/system-one/release/.venv/KeyPair-19e0.pem
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -i $K -p 31955 ma-user@dev-modelarts-cnsouth1.huaweicloud.com 'echo SSH-OK; date "+%F %T"; tail -6 /tmp/w6_all.log 2>/dev/null; echo "py=$(pgrep -fc python3 2>/dev/null || echo 0)"'
echo "SSH_RC=$?"
