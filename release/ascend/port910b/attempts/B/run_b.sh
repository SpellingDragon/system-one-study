#!/bin/bash
# attempts/B/run_b.sh — 把 attempts/B/<file> 同步进 cann910b-b 的 /tmp/agentB 并按判决环境执行
# 用法: bash run_b.sh <file.py|file.sh> [tail_lines] [脚本参数...]
#   · .py 走 python3，.sh 走 bash
#   · 判决环境（source set_env.sh + BISHENG_HOME + ASCEND_NPU_ARCH=dav-2201）由外层统一注入
#   · scratch 目录用 /tmp/agentB（不用公共 /tmp，避免与其他并发代理同名互踩）
#   · 形状/位宽类环境变量（下列 FWD 名单）若宿主机设了就透传进容器，便于一条命令复现各种配置
set -u
f="$1"
n="${2:-16}"
shift 2 2>/dev/null || true
extra="$*"
here="$(cd "$(dirname "$0")" && pwd)"
FWD="B_ROWS B_DIM B_BM B_RES_FP32 B_OUT_FP16 B_EPS R_BATCH R_SEQ R_DIM R_K R_BM R_HDT"
fwd=""
for v in $FWD; do
  val="${!v:-}"
  [ -n "$val" ] && fwd="${fwd}export ${v}=${val}; "
done
docker exec cann910b-b mkdir -p /tmp/agentB || { echo "MKDIR-FAIL"; exit 1; }
docker exec -i cann910b-b bash -c "cat > /tmp/agentB/${f}" < "${here}/${f}" || { echo "SYNC-FAIL"; exit 1; }
case "$f" in
  *.sh) runner="bash" ;;
  *)    runner="python3" ;;
esac
docker exec cann910b-b bash -c "source /usr/local/Ascend/cann-8.5.0/set_env.sh; export BISHENG_HOME=/usr/local/Ascend/cann-8.5.0/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201; ${fwd}cd /tmp/agentB && ${runner} /tmp/agentB/${f} ${extra} 2>&1 | tail -${n}"
