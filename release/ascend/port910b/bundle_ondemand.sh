#!/bin/bash
# ============================================================================
# bundle_ondemand.sh — P1-1d/P1-1e/P1-3 上机一键（自包含·幂等·单文件，窗口预算 <=10min）
# 用法（910B 实例上）： bash bundle_ondemand.sh
# 输出：每 target 一行 NUM-* PASS/FAIL + VERDICT-SUMMARY
# 本窗使命：验 §12 九参修复是否消除 cube 面 aicore 507015（gemm_l1/dW）+ 补收 addln/readout/delta 真机数值
# ============================================================================
set -uo pipefail
# 缓存纪律：tilelang cache key 不含模板内容——若复用旧实例的 ~/.tilelang/cache 会命中旧 .o。
# 强制全新缓存目录，保证本窗 overlay 后的 compat §12 真被编译（R20/R-B 战报）。
export TILELANG_CACHE_DIR=$(mktemp -d)
echo "=== [0] env ==="
SRC="https://github.com/SpellingDragon/system-one-study.git"
SUB=/tmp/sys1 && rm -rf $SUB
git clone -q --depth 1 "$SRC" $SUB || { echo "FATAL: clone failed"; exit 1; }
P910B=$SUB/release/ascend/port910b/patches   # 模板真源（compat/debug/dcache/common/numeric_limits + stubs）
[ -d "$P910B" ] || { echo "FATAL: $P910B missing"; exit 1; }

# CANN env 自适应（官方 set_env.sh 引用未定义变量，source 期间放宽 -u）
set +u
for c in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/cann-*; do
  [ -f "$c/set_env.sh" ] && { source "$c/set_env.sh"; break; }
done
set -u
CCEC=$(find /usr/local/Ascend -path "*ccec_compiler/bin/bisheng" 2>/dev/null | head -1)
[ -n "$CCEC" ] && export BISHENG_HOME=$(dirname $(dirname "$CCEC"))
ASCEND_NPU_ARCH=dav-2201; export ASCEND_NPU_ARCH
echo "BISHENG_HOME=${BISHENG_HOME:-<default>}  arch=$ASCEND_NPU_ARCH"

echo "=== [1] tilelang ==="
python3 -c "import tilelang" 2>/dev/null || pip3 install -q -i https://pypi.tuna.tsinghua.edu.cn/simple tilelang
python3 -c "import tilelang; print('tilelang', tilelang.__version__)"

echo "=== [2] overlay 910B templates onto pip tree ==="
TPL=$(python3 -c "import tilelang,os;print(os.path.join(os.path.dirname(tilelang.__file__),'src','tl_templates','ascend'))")
cp -f "$P910B"/*.h "$TPL"/ 2>/dev/null
for d in c_api simt_api; do mkdir -p "$(dirname "$TPL")/../$d" 2>/dev/null && cp -f "$P910B"/$d/*.h "$(dirname "$TPL")/../$d"/ 2>/dev/null || true; done
grep -c TL_ASCEND_SIMT "$TPL/common.h" || { echo "FATAL: common.h not patched"; exit 1; }

echo "=== [3] bisheng options injection (idempotent, py_compile-verified) ==="
python3 "$P910B/../patches/patch_bisheng.py" || { echo "FATAL: bisheng inject failed"; exit 1; }

# 缓存纪律：tilelang 缓存 key 不含模板内容，改 compat 必须换 TILELANG_CACHE_DIR（见 run B/C 战报）
echo "=== [4] consolidated numerics (single window · P1-1d cube fix + P1-1e rig + P1-3 delta) ==="
cd "$SUB/release/ascend/port910b"
# 每件子进程隔离（aicore exception 不连坐，oncard-wave1 教训）。cube 修复复验 + vector 回归 + rig 补收。
for t in gemm_l1 dw addln readout gdn; do
  timeout 480 python3 run_numerics.py --only $t 2>&1 | grep -E "NUM-$t" | head -1
done
# P1-3 GDN delta rule 首次真机执行（F 波仅 CPU 编过，本件上卡跑+对拍）
timeout 480 python3 run_delta_card.py 2>&1 | grep -E "NUM-delta_fwd" | head -1

echo "=== VERDICT-SUMMARY (copy the lines above into run notes) ==="
