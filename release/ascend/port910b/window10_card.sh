#!/bin/bash
# window10_card.sh — 第十窗：#1307 workaround 判决 + 对齐版 gdn_bwd 全量验收 + 一旗腿补跑
# [1]=v14 A/B（[1]buf 对照 vs [8] 32B 对齐）  [2]=对齐版 ladder  [3]=对齐版 L0 全量  [4]=P1-1zi（autoload 修复配方）
set -u
TARPATH=/tmp/ta_pkg.tar.gz
UNP=/tmp/ta_unpack
export TA_ROOT=$UNP/ta_pkg
EXPECT_SHA=d5607ec6f9c321c9652fa7d25f234d2e17a074cf308938fa2c4b522830b02172

echo "=== [0] TA 解包 + env + 自证 ==="
BYTES=$(wc -c < "$TARPATH" 2>/dev/null || echo 0); SHA=$(sha256sum "$TARPATH" 2>/dev/null | cut -d' ' -f1)
echo "W10-TAR bytes=$BYTES sha256=${SHA:-none}"
[ "$SHA" = "$EXPECT_SHA" ] || { echo "W10-ABORT reason=tar-sha"; echo W10-DONE; exit 1; }
rm -rf "$UNP"; mkdir -p "$UNP"
tar xzf "$TARPATH" -C "$UNP" || { echo "W10-ABORT reason=untar"; echo W10-DONE; exit 1; }
[ -f "$TA_ROOT/tilelang/__init__.py" ] || { echo "W10-ABORT reason=layout"; echo W10-DONE; exit 1; }
echo "W10-UNPACK files=$(find "$TA_ROOT" -type f | wc -l)"
set +u
for envf in /usr/local/Ascend/cann-8.5.2/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh; do
  [ -f "$envf" ] && { source "$envf"; echo "W10-CANN-env $envf"; break; }
done
set -u
export ASCEND_NPU_ARCH=dav-2201
export BISHENG_HOME=$(dirname "$(dirname "$(command -v bisheng)")")
export PYTHONPATH="$TA_ROOT:$TA_ROOT/3rdparty/tvm/python"
export LD_LIBRARY_PATH="$TA_ROOT/build/tvm:$TA_ROOT/build:${LD_LIBRARY_PATH:-}"
CORE="Cython,cloudpickle,ml_dtypes,psutil,tqdm,typing_extensions,numpy,torch"
python3 -c "import $CORE" 2>/dev/null || pip install -q Cython cloudpickle ml_dtypes psutil tqdm typing_extensions numpy >/dev/null 2>&1 || true
python3 - <<'EOF' || { echo "W10-ABORT reason=tl-import"; echo W10-DONE; exit 1; }
import tilelang
assert "/ta_unpack/ta_pkg/" in tilelang.__file__, f"not TA: {tilelang.__file__}"
print("W10-TLPKG", tilelang.__file__)
EOF
echo "W10-SRC gdn_bwd=$(wc -c < /tmp/gdn_bwd.py)B test=$(wc -c < /tmp/test_gdn_bwd.py)B v14=$(wc -c < /tmp/w9_probe_v14_align.py)B"

echo "=== [1] v14：#1307 workaround 判决（A=[1]buf 对照 / B=[8] 32B 对齐）==="
export TILELANG_CACHE_DIR=$(mktemp -d)
timeout 600 python3 /tmp/w9_probe_v14_align.py > /tmp/w10_v14.log 2>&1
echo "W10-V14-EXIT=$?"; grep -E "^V14" /tmp/w10_v14.log

echo "=== [2] 对齐版 ladder（7 档）==="
export TILELANG_CACHE_DIR=$(mktemp -d)
timeout 900 python3 /tmp/w9_probe_v3_ladder.py > /tmp/w10_ladder.log 2>&1
echo "W10-LADDER-EXIT=$?"; grep -E "^LADDER" /tmp/w10_ladder.log

echo "=== [3] 对齐版 L0 全量（六梯度×6 case + special）==="
export TILELANG_CACHE_DIR=$(mktemp -d)
timeout 2400 python3 - <<'EOF' > /tmp/w10_l0.log 2>&1
import sys, runpy
sys.argv = ["test_gdn_bwd.py", "--level", "l0"]
try:
    import torch_npu  # noqa: F401  register torch.npu before tilelang's lazy torch import
except ImportError:
    print("W10-NPU-WARN no torch_npu")
runpy.run_path("/tmp/test_gdn_bwd.py", run_name="__main__")
EOF
echo "W10-L0-EXIT=$?"; grep -E "^ITEM |^\[PRECISION" /tmp/w10_l0.log | tail -50

echo "=== [4] 一旗腿 P1-1zi（autoload 修复配方：显式 torch_npu 预载）==="
T=/tmp/sys1; CR=/tmp/card_run.sh
if [ ! -d "$T/release" ]; then
  env -u PYTHONPATH -u LD_LIBRARY_PATH -u TA_ROOT CARDRUN_STOP_AFTER=2 bash "$CR" > /tmp/w10_boot.log 2>&1 || true
else
  env -u PYTHONPATH -u LD_LIBRARY_PATH -u TA_ROOT CARDRUN_SKIP_CLONE=1 CARDRUN_STOP_AFTER=2 bash "$CR" > /tmp/w10_boot.log 2>&1 || true
fi
if [ -d "$T/release" ]; then
  cd "$T" && git pull -q origin main 2>/dev/null
  echo "W10-FLAG-HEAD=$(git -C $T rev-parse --short HEAD)"
  cd "$T/release"
  AE=ascend/kernels/ascend_env.py
  cp -n "$AE" "$AE.bak" 2>/dev/null
  sed -i 's|return {"target": "ascend"}|return {"target": "ascend", "pass_configs": {"tl.device_compile_flags": "--cce-auto-sync"}}|' "$AE"
  echo "W10-FLAG-INJECTED=$(grep -c 'cce-auto-sync' "$AE")"
  export TILELANG_CACHE_DIR=$(mktemp -d)
  env -u PYTHONPATH -u LD_LIBRARY_PATH -u TA_ROOT timeout 900 python3 -c "import torch_npu; import sys, runpy; sys.argv=['run_delta_card.py','--target','ascend','--bwd']; runpy.run_path('ascend/port910b/run_delta_card.py', run_name='__main__')" > /tmp/w10_flag.log 2>&1
  echo "W10-FLAG-EXIT=$?"; grep -E "^ITEM |^NUM-delta_bwd|cce-auto-sync|unknown argument|Error" /tmp/w10_flag.log | head -10
  mv "$AE.bak" "$AE" 2>/dev/null && echo "W10-FLAG-REVERTED=$(grep -c 'cce-auto-sync' "$AE")"
else
  echo "W10-FLAG-SKIP no-clone"
fi

echo "=== [5] 现场保全 ==="
tar czf /tmp/w10_artifacts.tgz -C /tmp w10_v14.log w10_ladder.log w10_l0.log w10_boot.log w10_flag.log 2>/dev/null
echo "W10-TGZ size=$(wc -c < /tmp/w10_artifacts.tgz)B"; echo "W10-DONE"
