#!/bin/bash
# window9_card.sh — 第九窗：TA gdn_bwd 真机首验（主腿）+ P1-1zi 一旗（搭车腿）
# 主腿复用窗五 TA 通道（ta_pkg 解包即用；窗五已实证 8.5.0 容器包在 8.5.2 卡上六件 rel=0 ⇒ F1 有正面历史证据）
# 失败预案映射：[1]=F1 卡上编译差异  [2]=F2 L0 数值  [3]=P1-1zi 一旗（dbeta 二值判决）
set -u
TARPATH=/tmp/ta_pkg.tar.gz
UNP=/tmp/ta_unpack
export TA_ROOT=$UNP/ta_pkg
EXPECT_SHA=d5607ec6f9c321c9652fa7d25f234d2e17a074cf308938fa2c4b522830b02172

echo "=== [0] TA 解包 + env + 自证 ==="
BYTES=$(wc -c < "$TARPATH" 2>/dev/null || echo 0); SHA=$(sha256sum "$TARPATH" 2>/dev/null | cut -d' ' -f1)
echo "W9-TAR bytes=$BYTES sha256=${SHA:-none}"
[ "$SHA" = "$EXPECT_SHA" ] || { echo "W9-ABORT reason=tar-sha"; echo W9-DONE; exit 1; }
rm -rf "$UNP"; mkdir -p "$UNP"
tar xzf "$TARPATH" -C "$UNP" || { echo "W9-ABORT reason=untar"; echo W9-DONE; exit 1; }
[ -f "$TA_ROOT/tilelang/__init__.py" ] || { echo "W9-ABORT reason=layout"; echo W9-DONE; exit 1; }
echo "W9-UNPACK files=$(find "$TA_ROOT" -type f | wc -l)"
set +u
for envf in /usr/local/Ascend/cann-8.5.2/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh; do
  [ -f "$envf" ] && { source "$envf"; echo "W9-CANN-env $envf"; break; }
done
set -u
export ASCEND_NPU_ARCH=dav-2201
export BISHENG_HOME=$(dirname "$(dirname "$(command -v bisheng)")")
export PYTHONPATH="$TA_ROOT:$TA_ROOT/3rdparty/tvm/python"
export LD_LIBRARY_PATH="$TA_ROOT/build/tvm:$TA_ROOT/build:${LD_LIBRARY_PATH:-}"
CORE="Cython,cloudpickle,ml_dtypes,psutil,tqdm,typing_extensions,numpy,torch"
python3 -c "import $CORE" 2>/dev/null || pip install -q Cython cloudpickle ml_dtypes psutil tqdm typing_extensions numpy >/dev/null 2>&1 || true
python3 - <<'EOF' || { echo "W9-ABORT reason=tl-import"; echo W9-DONE; exit 1; }
import tilelang
assert "/ta_unpack/ta_pkg/" in tilelang.__file__, f"not TA: {tilelang.__file__}"
print("W9-TLPKG", tilelang.__file__)
EOF
echo "W9-SRC gdn_bwd=$(wc -c < /tmp/gdn_bwd.py)B test=$(wc -c < /tmp/test_gdn_bwd.py)B"

echo "=== [1] F1 卡上编译（bisheng 8.5.2，两 dtype 分判）==="
export TILELANG_CACHE_DIR=$(mktemp -d)
timeout 900 python3 - <<'EOF'
import sys; sys.path.insert(0, "/tmp")
for dt in ("float16", "float32"):
    try:
        from gdn_bwd import gdn_bwd
        k = gdn_bwd(1, 2, 64, 64, 64, dtype=dt, accum_dtype="float")
        print(f"W9-COMPILE-{dt} OK src={len(k.get_kernel_source())}")
    except Exception as e:
        print(f"W9-COMPILE-{dt} FAIL {type(e).__name__}: {e}")
EOF
echo "W9-COMPILE-EXIT=$?"

echo "=== [2] L0 真机数值（六梯度×6 case + special；runpy 注入 torch_npu 注册，不动 TA 冻结件）==="
timeout 2400 python3 - <<'EOF' > /tmp/w9_l0.log 2>&1
import sys, runpy
sys.argv = ["test_gdn_bwd.py", "--level", "l0"]
try:
    import torch_npu  # noqa: F401  注册 torch.npu 后端（test 件本体无此 import）
except ImportError:
    print("W9-NPU-WARN no torch_npu")
runpy.run_path("/tmp/test_gdn_bwd.py", run_name="__main__")
EOF
echo "W9-L0-EXIT=$?"; grep -E "^ITEM |^\[PRECISION" /tmp/w9_l0.log | tail -50

echo "=== [3] 一旗腿 P1-1zi：fork gdnb 加 --cce-auto-sync，判 dbeta（TA 环境出，回 fork 世界）==="
T=/tmp/sys1; CR=/tmp/card_run.sh
# 新实例引导：无 clone 则跑 card_run 全套（[1]clone+pip [2]overlay，STOP_AFTER=2）；有 clone 则沿用窗四配方
# ⚠ 必须 -u PYTHONPATH/-u LD_LIBRARY_PATH：主腿导出的 TA 路径会欺骗 card_run 的 tilelang 探测（窗九-b 实证：overlay 打进 TA 包、pip 被跳过）
if [ ! -d "$T/release" ]; then
  env -u PYTHONPATH -u LD_LIBRARY_PATH -u TA_ROOT CARDRUN_STOP_AFTER=2 bash "$CR" > /tmp/w9_boot.log 2>&1 || true
else
  env -u PYTHONPATH -u LD_LIBRARY_PATH -u TA_ROOT CARDRUN_SKIP_CLONE=1 CARDRUN_STOP_AFTER=2 bash "$CR" > /tmp/w9_boot.log 2>&1 || true
fi
if [ -d "$T/release" ]; then
  cd "$T" && git pull -q origin main 2>/dev/null
  echo "W9-FLAG-HEAD=$(git -C $T rev-parse --short HEAD)"
  cd "$T/release"
  AE=ascend/kernels/ascend_env.py
  cp -n "$AE" "$AE.bak" 2>/dev/null
  sed -i 's|return {"target": "ascend"}|return {"target": "ascend", "pass_configs": {"tl.device_compile_flags": "--cce-auto-sync"}}|' "$AE"
  echo "W9-FLAG-INJECTED=$(grep -c 'cce-auto-sync' "$AE")"
  export TILELANG_CACHE_DIR=$(mktemp -d)
  env -u PYTHONPATH -u LD_LIBRARY_PATH -u TA_ROOT timeout 900 python3 ascend/port910b/run_delta_card.py --target ascend --bwd > /tmp/w9_flag.log 2>&1
  echo "W9-FLAG-EXIT=$?"; grep -E "^ITEM |^NUM-delta_bwd|cce-auto-sync|unknown argument" /tmp/w9_flag.log | head -10
  mv "$AE.bak" "$AE" 2>/dev/null && echo "W9-FLAG-REVERTED=$(grep -c 'cce-auto-sync' "$AE")"
else
  echo "W9-FLAG-SKIP no-system-one-clone"
fi

echo "=== [4] 现场保全 ==="
tar czf /tmp/w9_artifacts.tgz -C /tmp w9_l0.log w9_boot.log w9_flag.log 2>/dev/null
echo "W9-TGZ size=$(wc -c < /tmp/w9_artifacts.tgz)B"; echo "W9-DONE"
