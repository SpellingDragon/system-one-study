#!/bin/bash
# attempts/C/run.sh — 宿主侧一键：同步 C 波件 → cann910b-c → 跑 compile 判决
# 用法: bash attempts/C/run.sh [c_gdn_conv_910b.py 的参数…]
#   例: bash attempts/C/run.sh --variant scalar --silu exp
#       bash attempts/C/run.sh --variant ubstage
#       bash attempts/C/run.sh --probe            # 数学面探针 (probe_math.py)
#       bash attempts/C/run.sh --dma              # 搬运面探针 (probe_dma.py)
#       bash attempts/C/run.sh --ub               # UB 句柄形态探针 (probe_ubhandle.py)
#       bash attempts/C/run.sh --golden           # CPU 语义 golden（纯 CPU，无需卡）
#       bash attempts/C/run.sh --matrix           # 4 变体判决矩阵（RESULT.md 的凭据源）
#
# 判据纪律：tilelang 按 **kernel 源码哈希** 命中编译缓存（TILELANG_CACHE_DIR=
# ~/.tilelang/cache），改 compat **头**不会让缓存失效 → 会拿到陈旧 PASS。故每次
# 判决都用全新 cache 目录（下面容器侧 mktemp）。
set -e
C=${TL_C_CONTAINER:-cann910b-c}
D=$(cd "$(dirname "$0")" && pwd)
ENV='source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null 2>&1; export BISHENG_HOME=/usr/local/Ascend/cann-8.5.0/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201 TILELANG_CACHE_DIR=$(mktemp -d /tmp/tlc_cache.XXXXXX);'

for f in env_setup.sh patch_bisheng.py compat_patch_C.h patch_compat_10b.py \
         probe_math.py probe_dma.py probe_ubhandle.py c_gdn_conv_910b.py c_gdn_golden.py; do
  [ -f "$D/$f" ] && docker exec -i "$C" bash -c "cat > /tmp/$f" < "$D/$f" \
               && docker exec "$C" chmod 644 "/tmp/$f"
done

case "$1" in
  --probe)  shift; docker exec "$C" bash -c "$ENV cd /tmp && python3 /tmp/probe_math.py $*"; exit $?;;
  --dma)    shift; docker exec "$C" bash -c "$ENV bash /tmp/env_setup.sh >/dev/null 2>&1; cd /tmp && python3 /tmp/probe_dma.py $*"; exit $?;;
  --ub)     shift; docker exec "$C" bash -c "$ENV cd /tmp && python3 /tmp/probe_ubhandle.py $*"; exit $?;;
  --golden) shift; docker exec "$C" bash -c "cd /tmp && python3 /tmp/c_gdn_golden.py $*"; exit $?;;
  --matrix) shift
            for v in scalar ubstage; do for s in exp rational; do
              echo "########## variant=$v silu=$s ##########"
              docker exec "$C" bash -c "$ENV bash /tmp/env_setup.sh | tail -3; cd /tmp && \
                python3 /tmp/c_gdn_conv_910b.py --variant $v --silu $s --dump /tmp/gdn_${v}_${s}.asc $*"
            done; done
            echo "########## baseline e2e_cube (compat 注入无回归检查) ##########"
            docker exec "$C" bash -c "$ENV cd /tmp && python3 /tmp/e2e_cube.py 2>&1 | tail -4"
            exit $?;;
esac

docker exec "$C" bash -c "$ENV bash /tmp/env_setup.sh | tail -5; cd /tmp && python3 /tmp/c_gdn_conv_910b.py $*"
