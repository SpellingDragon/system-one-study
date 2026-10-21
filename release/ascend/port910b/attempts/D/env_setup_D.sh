#!/bin/bash
# attempts/D/env_setup_D.sh — cann910b-d 一次性环境装配（幂等；容器内执行）
#   [1] CANN env + bisheng + 910B arch
#   [2] overlay 主仓模板树（只读 /tilelang）→ pip 运行副本（真源一字不改）
#   [2b] GAP-D1  compat_patch_D.h：补 asc_copy_l12l0a_transpose（L1→L0A 转置装填件）
#   [2c] GAP-D1b patch_compat_D.py：就地改写 asc_copy_l12l0b_transpose 的 6 参原生调用 → 官方 8 参
#   [3] bisheng.py 注入 -DTL_PORT910B_NATIVE_TYPES + asc include（幂等自愈）
set +u
source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null 2>&1
export BISHENG_HOME=${BISHENG_HOME:-/usr/local/Ascend/cann-8.5.0/aarch64-linux/ccec_compiler}
export ASCEND_NPU_ARCH=dav-2201
echo "[1] BISHENG_HOME=$BISHENG_HOME arch=$ASCEND_NPU_ARCH"

TPL=$(python3 -c "import tilelang,os;print(os.path.join(os.path.dirname(tilelang.__file__),'src','tl_templates','ascend'))")
echo "[2] pip template tree: $TPL"
cp -f /tilelang/src/tl_templates/ascend/*.h "$TPL"/
for d in c_api simt_api; do
  mkdir -p "$(dirname "$TPL")/../$d" 2>/dev/null
  cp -f /tilelang/src/tl_templates/ascend/$d/*.h "$(dirname "$TPL")/../$d"/ 2>/dev/null || true
done
echo "[2] compat lines: $(wc -l < "$TPL/port910b_compat.h") (真源同步 md5=$(md5sum "$TPL/port910b_compat.h" | cut -c1-12))"

if [ -f /tmp/compat_patch_D.h ]; then
  grep -q "TL_PORT910B_COMPAT_GAP_D_H" "$TPL/port910b_compat.h" || cat /tmp/compat_patch_D.h >> "$TPL/port910b_compat.h"
  echo "[2b] GAP-D1 compat: $(wc -l < "$TPL/port910b_compat.h") lines (marker=$(grep -c TL_PORT910B_COMPAT_GAP_D_H "$TPL/port910b_compat.h"))"
else
  echo "[2b] GAP-D1 compat 缺件 → L1->L0A 转置装填裸奔（l0tr 变体必挂）"
fi

if [ -f /tmp/patch_compat_D.py ]; then
  python3 /tmp/patch_compat_D.py || echo "[2c] WARN: GAP-D1b（asc_copy_l12l0b_transpose 6参→8参）改写未生效"
else
  echo "[2c] patch_compat_D.py 缺失 → GAP-D1b 未修，L1->L0B 转置装填必挂"
fi

python3 /tmp/patch_bisheng.py
echo "SETUP-DONE"
