#!/bin/bash
# res_check_local.sh — 实例资源盘点（三窗前的最终预检）
K=/Users/pengweiye/Documents/codes/system-one/release/.venv/KeyPair-19e0.pem
S="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -i $K -p 31955"
H=ma-user@dev-modelarts-cnsouth1.huaweicloud.com
chmod 400 "$K" 2>/dev/null
ssh -T $S $H 'echo "=== NPU ==="; npu-smi info 2>/dev/null | head -20
echo "=== CPU/内存 ==="; lscpu 2>/dev/null | grep -E "^(Model name|Architecture|CPU\(s\)):"; free -h | head -2
echo "=== 磁盘 ==="; df -h / /tmp 2>/dev/null | grep -vE "tmpfs|Filesystem" ; df -h /dev/shm 2>/dev/null | tail -1
echo "=== python/torch 栈 ==="; python3 -V; python3 -c "import torch;print(\"torch\",torch.__version__)" 2>&1 | tail -1; python3 -c "import torch_npu;print(\"torch_npu ok\")" 2>&1 | tail -1
echo "=== 旧现场 ==="; ls -d /tmp/sys1 2>/dev/null && du -sh /tmp/sys1 2>/dev/null || echo "no /tmp/sys1"
ls /tmp/w4_*.log /tmp/w5_*.log 2>/dev/null | head -4' 2>&1 | grep -vE "^Warning"
