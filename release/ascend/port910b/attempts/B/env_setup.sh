#!/bin/bash
# attempts/B/env_setup.sh — cann910b-b 判决环境一次性准备（幂等，可反复重跑以回到干净态）
#   [1] CANN env + 内层 ccec bisheng + dav-2201
#   [2] overlay 主仓模板树 -> pip 运行副本（**会抹掉 GAP-B**，故其后必须再跑 apply_compat_gapB.py）
#   [3] bisheng.py 编译选项注入：交给 fix_bisheng.py（自带"撤销坏注入 + 正确注入 + ast 自证"）
#       —— 不再内联 bundle_ondemand.sh [3] 的锚替换：那个锚在 tilelang 0.1.15 上不是列表结尾，
#          注入会把 bisheng.py 改成 SyntaxError（详见 compat_gap_B.md / RESULT.md ① 段）。
set +u
source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null 2>&1
export BISHENG_HOME=/usr/local/Ascend/cann-8.5.0/aarch64-linux/ccec_compiler
export ASCEND_NPU_ARCH=dav-2201
echo "BISHENG_HOME=$BISHENG_HOME arch=$ASCEND_NPU_ARCH bisheng=$(command -v bisheng || echo $BISHENG_HOME/bin/bisheng)"

TPL=$(python3 -c "import tilelang,os;print(os.path.join(os.path.dirname(tilelang.__file__),'src','tl_templates','ascend'))" 2>/dev/null) \
  || { echo "FATAL: cannot import tilelang (bisheng.py 可能仍是坏注入态 → 先跑 fix_bisheng.py)"; exit 1; }
mkdir -p "$TPL"
cp -f /tilelang/src/tl_templates/ascend/*.h "$TPL"/
for d in c_api simt_api; do
  cp -rf /tilelang/src/tl_templates/ascend/$d "$(dirname "$TPL")/" 2>/dev/null || true
done
echo "overlay compat lines: $(wc -l < "$TPL/port910b_compat.h")  GAPB-present: $(grep -c TL910B_GAPB_APPLIED "$TPL/port910b_compat.h")"
echo "overlay common.h TL_ASCEND_SIMT hits: $(grep -c TL_ASCEND_SIMT "$TPL/common.h")"

python3 /tmp/agentB/fix_bisheng.py
echo "SETUP-DONE"
