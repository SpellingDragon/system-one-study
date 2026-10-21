#!/bin/bash
# attempts/E/run.sh — 宿主侧一键：同步 E 波件 → cann910b-e → 跑 compile 判决
# 用法: bash attempts/E/run.sh <子命令>
#   table    — rope 表加载件（主判决）
#   sinf     — rope 核内自算取证件（预期 FAIL：标量面 sinf/cosf 缺件反证）
#   rational — rope 泰勒兜底骨架（预期 PASS，数值非真值）
#   attn     — attn_sw 两变体（stage 主判决 + pure 保险）
#   sign     — rope 反向件 (sign=-1)
#   golden   — CPU 语义 golden（纯 CPU/numpy，无卡可跑）
#   matrix   — 全矩阵 + e2e_cube 基线（RESULT 凭据源，写 verdict_log.txt）
# 判据纪律：每次判决 TILELANG_CACHE_DIR=$(mktemp -d)（tilelang 按 kernel 源码哈希
# 命中缓存、不含模板内容 → 不换 cache 就是陈旧 PASS，LOCAL_RUN.md/B/C 实证）。
set -e
E=${TL_E_CONTAINER:-cann910b-e}
D=$(cd "$(dirname "$0")" && pwd)
P=$(cd "$D/../../patches" && pwd)
ENV='source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null 2>&1; export BISHENG_HOME=/usr/local/Ascend/cann-8.5.0/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201 TILELANG_CACHE_DIR=$(mktemp -d /tmp/tlc_cache_e.XXXXXX);'

for f in e_rope_910b.py e_attn_sw_910b.py e_golden.py; do
  docker exec -i "$E" bash -c "cat > /tmp/$f" < "$D/$f"
  docker exec "$E" chmod 644 "/tmp/$f"
done
docker exec -i "$E" bash -c "cat > /tmp/patch_bisheng.py" < "$P/patch_bisheng.py"

# 环境自证：compat v3 同步态（pip 副本 == 主仓真源）+ bisheng 注入（幂等）
docker exec "$E" bash -c 'cp -f /tilelang/src/tl_templates/ascend/*.h \
  /usr/local/lib/python3.10/dist-packages/tilelang/src/tl_templates/ascend/ 2>/dev/null || true; \
  cmp -s /tilelang/src/tl_templates/ascend/port910b_compat.h \
         /usr/local/lib/python3.10/dist-packages/tilelang/src/tl_templates/ascend/port910b_compat.h \
  && echo "ENV-COMPAT-IN-SYNC ($(wc -l < /tilelang/src/tl_templates/ascend/port910b_compat.h) lines)" \
  || echo "ENV-COMPAT-DRIFT"; python3 /tmp/patch_bisheng.py'

case "$1" in
  table)   docker exec "$E" bash -c "$ENV cd /tmp && python3 /tmp/e_rope_910b.py --mode table ${*:2}";;
  sinf)    docker exec "$E" bash -c "$ENV cd /tmp && python3 /tmp/e_rope_910b.py --mode sinf ${*:2}";;
  rational)docker exec "$E" bash -c "$ENV cd /tmp && python3 /tmp/e_rope_910b.py --mode rational ${*:2}";;
  sign)    docker exec "$E" bash -c "$ENV cd /tmp && python3 /tmp/e_rope_910b.py --mode table --sign -1 ${*:2}";;
  attn)    docker exec "$E" bash -c "$ENV cd /tmp && python3 /tmp/e_attn_sw_910b.py --variant stage ${*:2}; \
            $ENV cd /tmp && python3 /tmp/e_attn_sw_910b.py --variant pure ${*:2}";;
  golden)  docker exec "$E" bash -c "cd /tmp && python3 /tmp/e_golden.py ${*:2}";;
  matrix)
    for combo in "rope table" "rope rational" "rope sinf" "attn stage" "attn pure"; do
      echo "########## E-MATRIX $combo ##########"
      set -- $combo
      if [ "$1" = "rope" ]; then
        if [ "$2" = "sinf" ]; then
          # 取证件：预期 FAIL（标量面 sinf/cosf 缺件反证），不计入主判决矩阵
          docker exec "$E" bash -c "$ENV cd /tmp && python3 /tmp/e_rope_910b.py --mode sinf" \
            || echo "E-SINF-PROBE-FAIL-AS-EXPECTED（见上一行错误摘要）"
        else
          docker exec "$E" bash -c "$ENV cd /tmp && python3 /tmp/e_rope_910b.py --mode $2 --dump /tmp/gen_rope_$2.asc"
        fi
      else
        docker exec "$E" bash -c "$ENV cd /tmp && python3 /tmp/e_attn_sw_910b.py --variant $2 --dump /tmp/gen_attnsw_$2.asc"
      fi
    done
    echo "########## E-MATRIX rope sign=-1 (反向件) ##########"
    docker exec "$E" bash -c "$ENV cd /tmp && python3 /tmp/e_rope_910b.py --mode table --sign -1"
    echo "########## E-MATRIX golden ##########"
    docker exec "$E" bash -c "cd /tmp && python3 /tmp/e_golden.py --npz /tmp/e_pairs.npz"
    echo "########## baseline e2e_cube (compat 注入无回归检查) ##########"
    docker exec "$E" bash -c "$ENV cd /tmp && python3 /tmp/e2e_cube.py 2>&1 | tail -4"
    ;;
  *) echo "用法: bash run.sh {table|sinf|rational|sign|attn|golden|matrix}"; exit 2;;
esac
