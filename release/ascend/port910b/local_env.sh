#!/bin/bash
# P0-2L 本地编译判决环境：Mac colima(aarch64) + Ubuntu + CANN 8.5.0 toolkit
set -e
IMG=docker.m.daocloud.io/library/ubuntu:22.04
CT=cann910b
CC=$REPO_ROOT/release/ascend/port910b/cann_cache
TL=$WORKSPACE_ROOT/tilelang

# ① 起容器（幂等：删旧建新）
docker rm -f $CT 2>/dev/null || true
docker run -d --name $CT -h cann910b \
  -v "$CC":/cann_cache -v "$TL":/tilelang \
  -w /root $IMG sleep infinity

# ② 装基础依赖 + CANN toolkit
docker exec $CT bash -c '
set -e
export DEBIAN_FRONTEND=noninteractive
sed -i "s|archive.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|; s|security.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|" /etc/apt/sources.list
apt-get update -qq && apt-get install -y -qq gcc g++ make cmake python3 python3-pip wget xz-utils zlib1g 2>&1 | tail -1
chmod +x /cann_cache/Ascend-cann-toolkit_8.5.0_linux-aarch64.run
/cann_cache/Ascend-cann-toolkit_8.5.0_linux-aarch64.run --install-path /usr/local/Ascend --quiet 2>&1 | tail -3
'

# ③ 验证 bisheng
docker exec $CT bash -c '
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || source /usr/local/Ascend/nnae/set_env.sh 2>/dev/null || find /usr/local/Ascend -name set_env.sh | head -1
B=$(find /usr/local/Ascend -name bisheng | head -1)
echo "bisheng at: $B"
$B --version 2>&1 | head -2 || echo "bisheng version 调用形式待查"
echo ENV-READY
'
