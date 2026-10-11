#!/usr/bin/env bash
# A2：卡窗主命令复跑稳定性（容器，冷 TILELANG_CACHE_DIR 逐轮）
set -u
source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null 2>&1
export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201
cd /work
for i in 2 3; do
  S=$(mktemp -d)
  TILELANG_CACHE_DIR=$S timeout 900 python3 ascend/compile_all_asc.py --target ascend --products > /tmp/a2_rep$i.txt 2>&1
  RC=$?
  LINES=$(grep -c "^ASC-" /tmp/a2_rep$i.txt)
  NPASS=$(grep -c " PASS " /tmp/a2_rep$i.txt)
  echo "REP$i rc=$RC verdict_lines=$LINES pass_lines=$NPASS summary=$(tail -1 /tmp/a2_rep$i.txt)"
done
