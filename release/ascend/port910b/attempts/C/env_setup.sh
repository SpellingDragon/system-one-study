#!/bin/bash
# attempts/C/env_setup.sh — cann910b-c 一次性环境准备（幂等，对齐 bundle_ondemand.sh [0]-[3]）
#   [1] CANN env + 内层 ccec bisheng
#   [2] overlay 主仓模板树→pip 树（真源在 /tilelang，本波禁改主仓；每次重置为真源）
#   [2b] 追加 GAP-C compat 块到 pip 树 compat 副本：
#          块1 = 标量数学面（expf/fabsf），块2 = UB<->GM 搬运面（asc_store_l2_cache_mode
#                 + asc_copy_ub2gm_align 7 参重载）
#   [2c] 就地改写 pip 副本 compat §10b 的 asc_copy_gm2ub_align（槽位对齐 codegen + bytes->32B 块单位）
#   [3] bisheng.py 注入 -DTL_PORT910B_NATIVE_TYPES + asc include（见 patch_bisheng.py）
set +u
source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null 2>&1
export BISHENG_HOME=${BISHENG_HOME:-/usr/local/Ascend/cann-8.5.0/aarch64-linux/ccec_compiler}
export ASCEND_NPU_ARCH=dav-2201
echo "BISHENG_HOME=$BISHENG_HOME arch=$ASCEND_NPU_ARCH"

TPL=$(python3 -c "import tilelang,os;print(os.path.join(os.path.dirname(tilelang.__file__),'src','tl_templates','ascend'))")
cp -f /tilelang/src/tl_templates/ascend/*.h "$TPL"/
for d in c_api simt_api; do
  mkdir -p "$(dirname "$TPL")/../$d" 2>/dev/null
  cp -f /tilelang/src/tl_templates/ascend/$d/*.h "$(dirname "$TPL")/../$d"/ 2>/dev/null || true
done
echo "[2] overlay common.h TL_ASCEND_SIMT hits: $(grep -c TL_ASCEND_SIMT "$TPL/common.h")"
echo "[2] overlay compat size: $(wc -l < "$TPL/port910b_compat.h") lines"

if [ -f /tmp/compat_patch_C.h ]; then
  # [2] 的 cp 每次都把副本重置为真源，故这里必然重新追加；再按 marker 兜一层幂等
  grep -q "TL_PORT910B_COMPAT_GAP_C_H" "$TPL/port910b_compat.h" || cat /tmp/compat_patch_C.h >> "$TPL/port910b_compat.h"
  echo "[2b] GAP-C compat: $(wc -l < "$TPL/port910b_compat.h") lines (math=$(grep -c TL_PORT910B_COMPAT_GAP_C_H "$TPL/port910b_compat.h") dma=$(grep -c TL_PORT910B_COMPAT_GAP_C_DMA_H "$TPL/port910b_compat.h"))"
else
  echo "[2b] GAP-C compat 缺失（/tmp/compat_patch_C.h）→ 数学面/搬运面裸奔，SiLU 与 T.copy 必挂"
fi

if [ -f /tmp/patch_compat_10b.py ]; then
  python3 /tmp/patch_compat_10b.py || echo "[2c] WARN: §10b GM->UB 改写未生效（见上方输出）"
else
  echo "[2c] patch_compat_10b.py 缺失 → §10b GM->UB 槽位/单位错误未修（G-C5/G-C4 仍在）"
fi

python3 /tmp/patch_bisheng.py
echo "SETUP-DONE"
