#!/bin/bash
# window11_card.sh — 第十一窗：B 线 v2 真机验收 + 真形档 workspace + D 线腿0 判决&dump + A 臂一枪
# [0] TA 解包+env 自证      [1] v2 L0 全量（7 case：六梯度+adjoint，no early exit）
# [2] 真形档 workspace 实测（B1 H16 L8192 DK128 DV128 fp16，307.5 MiB 声明）
# [3] D 线腿0：fork boot + 无改判决复采 + 生成码 dump（同 boot 同 cache）
# [3b] cpu 基线（A 臂门左：未改 c 后端六项）   [4] A 臂（W11_A2=1 才执行）
# [5] 现场保全
set -u
TARPATH=/tmp/ta_pkg.tar.gz
UNP=/tmp/ta_unpack
export TA_ROOT=$UNP/ta_pkg
EXPECT_SHA=d5607ec6f9c321c9652fa7d25f234d2e17a074cf308938fa2c4b522830b02172
W11_A2=${W11_A2:-0}
export W11_A2

echo "=== [0] TA 解包 + env + 自证 ==="
BYTES=$(wc -c < "$TARPATH" 2>/dev/null || echo 0); SHA=$(sha256sum "$TARPATH" 2>/dev/null | cut -d' ' -f1)
echo "W11-TAR bytes=$BYTES sha256=${SHA:-none}"
[ "$SHA" = "$EXPECT_SHA" ] || { echo "W11-ABORT reason=tar-sha"; echo W11-DONE; exit 1; }
rm -rf "$UNP"; mkdir -p "$UNP"
tar xzf "$TARPATH" -C "$UNP" || { echo "W11-ABORT reason=untar"; echo W11-DONE; exit 1; }
[ -f "$TA_ROOT/tilelang/__init__.py" ] || { echo "W11-ABORT reason=layout"; echo W11-DONE; exit 1; }
echo "W11-UNPACK files=$(find "$TA_ROOT" -type f | wc -l)"
set +u
for envf in /usr/local/Ascend/cann-8.5.2/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh; do
  [ -f "$envf" ] && { source "$envf"; echo "W11-CANN-env $envf"; break; }
done
set -u
export ASCEND_NPU_ARCH=dav-2201
export BISHENG_HOME=$(dirname "$(dirname "$(command -v bisheng)")")
export PYTHONPATH="$TA_ROOT:$TA_ROOT/3rdparty/tvm/python"
export LD_LIBRARY_PATH="$TA_ROOT/build/tvm:$TA_ROOT/build:${LD_LIBRARY_PATH:-}"
CORE="Cython,cloudpickle,ml_dtypes,psutil,tqdm,typing_extensions,numpy,torch"
python3 -c "import $CORE" 2>/dev/null || pip install -q Cython cloudpickle ml_dtypes psutil tqdm typing_extensions numpy >/dev/null 2>&1 || true
python3 - <<'EOF' || { echo "W11-ABORT reason=tl-import"; echo W11-DONE; exit 1; }
import tilelang
assert "/ta_unpack/ta_pkg/" in tilelang.__file__, f"not TA: {tilelang.__file__}"
print("W11-TLPKG", tilelang.__file__)
EOF
echo "W11-SRC v2=$(wc -c < /tmp/gdn_bwd_v2.py)B test=$(wc -c < /tmp/test_gdn_bwd_v2.py)B wsdrv=$(wc -c < /tmp/_w11_ws_true.py)B a2_patch=$(wc -c < /tmp/w11_a2_patch.py)B"

echo "=== [1] B 线 v2 L0 全量（7 case：六梯度+adjoint，no early exit）==="
export TILELANG_CACHE_DIR=$(mktemp -d)
timeout 5400 python3 - <<'EOF' > /tmp/w11_l0.log 2>&1
import sys, runpy
sys.argv = ["test_gdn_bwd_v2.py", "--level", "l0"]
sys.path.insert(0, "/tmp")
try:
    import torch_npu  # noqa: F401  register torch.npu before tilelang's lazy torch import
except ImportError:
    print("W11-NPU-WARN no torch_npu")
runpy.run_path("/tmp/test_gdn_bwd_v2.py", run_name="__main__")
EOF
echo "W11-L0-EXIT=$?"; grep -E "^ITEM |^\[PRECISION|^Test Passed" /tmp/w11_l0.log | tail -70

echo "=== [2] 真形档 workspace 实测（B1 H16 L8192 DK128 DV128 fp16）==="
WSCACHE=$(mktemp -d)
export TILELANG_CACHE_DIR=$WSCACHE
timeout 3600 python3 /tmp/_w11_ws_true.py > /tmp/w11_ws.log 2>&1
echo "W11-WS-EXIT=$?"; grep -E "^WS-" /tmp/w11_ws.log
# workspace 声明量级核对（生成件反推；仅存档，本窗判据=编译+跑通）
grep -rhoE "GetWorkspaceSize|workspace[A-Za-z_]*\[[0-9]+\]|workspace_size[^;]*" "$WSCACHE" 2>/dev/null | head -5 | sed 's/^/W11-WS-HOST /'
tar czf /tmp/w11_ws_cache.tgz -C "$WSCACHE" . 2>/dev/null
echo "W11-WS-CACHE-ENTRIES=$(ls "$WSCACHE" | wc -l) tgz=$(wc -c < /tmp/w11_ws_cache.tgz 2>/dev/null || echo 0)B"

echo "=== [3] D 线腿0：fork boot + 无改判决复采 + 生成码 dump ==="
T=/tmp/sys1; CR=/tmp/card_run.sh
if [ ! -d "$T/release" ]; then
  env -u PYTHONPATH -u TA_ROOT CARDRUN_STOP_AFTER=2 bash "$CR" > /tmp/w11_boot.log 2>&1 || true
fi
grep -c "OVERLAY-OK" /tmp/w11_boot.log 2>/dev/null | sed 's/^/W11-OVERLAY=/'
if [ -d "$T/release" ]; then
  cd "$T/release"
  export TILELANG_CACHE_DIR=/tmp/sys1_r6_dump; rm -rf "$TILELANG_CACHE_DIR"; mkdir -p "$TILELANG_CACHE_DIR"
  env -u PYTHONPATH -u TA_ROOT timeout 1500 python3 -c "import torch_npu; import sys, runpy; sys.argv=['run_delta_card.py','--target','ascend','--bwd']; runpy.run_path('ascend/port910b/run_delta_card.py', run_name='__main__')" > /tmp/w11_r6_bwd.log 2>&1
  echo "W11-R6-EXIT=$?"; grep -E "^ITEM |^NUM-delta_bwd" /tmp/w11_r6_bwd.log
  for d in "$TILELANG_CACHE_DIR"/*/; do
    grep -l 'void gdn_bwd_impl_kernel(' "$d/device_kernel.asc" 2>/dev/null | sed "s/^/W11-R6-KERNEL /"
  done
  tar czf /tmp/w11_sys1_r6_kernels.tgz -C "$TILELANG_CACHE_DIR" . 2>/dev/null
  echo "W11-R6-DUMP-ENTRIES=$(ls "$TILELANG_CACHE_DIR" | wc -l) tgz=$(wc -c < /tmp/w11_sys1_r6_kernels.tgz 2>/dev/null || echo 0)B"

  echo "=== [3b] cpu 基线（A 臂门左：未改 c 后端六项）==="
  export TILELANG_CACHE_DIR=$(mktemp -d)
  env -u PYTHONPATH -u TA_ROOT timeout 1500 python3 -c "import sys, runpy; sys.argv=['run_delta_card.py','--target','cpu','--bwd']; runpy.run_path('ascend/port910b/run_delta_card.py', run_name='__main__')" > /tmp/w11_r6_cpu_base.log 2>&1
  echo "W11-CPUBASE-EXIT=$?"; grep -E "^ITEM |^NUM-delta_bwd" /tmp/w11_r6_cpu_base.log

  if [ "$W11_A2" = "1" ]; then
    echo "=== [4] A 臂（候选 A：单点求值 (v-p)）==="
    AE=ascend/kernels/gdn_asc.py
    md5sum "$AE" | sed 's/^/W11-A2-PRE-MD5 /'
    cp "$AE" /tmp/w11_a2_gdn_asc.py.bak
    python3 /tmp/w11_a2_patch.py || echo "W11-A2-PATCH-FAIL"
    echo "W11-A2-PATCHED=$(grep -c 'diff = v_scr\[j\] - p' "$AE")"
    python3 -m py_compile "$AE" && echo "W11-A2-PYCOMPILE=OK"
    echo "--- [4a] cpu 门（改后 vs 基线，逐字）---"
    export TILELANG_CACHE_DIR=$(mktemp -d)
    env -u PYTHONPATH -u TA_ROOT timeout 1500 python3 -c "import sys, runpy; sys.argv=['run_delta_card.py','--target','cpu','--bwd']; runpy.run_path('ascend/port910b/run_delta_card.py', run_name='__main__')" > /tmp/w11_r6_cpu_arm.log 2>&1
    echo "W11-A2-CPU-EXIT=$?"; grep -E "^ITEM |^NUM-delta_bwd" /tmp/w11_r6_cpu_arm.log
    if diff <(grep -E "^ITEM |^NUM-delta_bwd" /tmp/w11_r6_cpu_base.log) <(grep -E "^ITEM |^NUM-delta_bwd" /tmp/w11_r6_cpu_arm.log) > /tmp/w11_a2_cpu_diff.txt 2>&1; then
      echo "W11-A2-CPUGATE=IDENTICAL"
      echo "--- [4b] pytest 门（本机 340 套件已在发射前预跑；此处仅记录 skip）---"
      echo "W11-A2-PYTEST=pre-run-on-host (see launcher log)"
      echo "--- [4c] A 臂上卡一枪 ---"
      export TILELANG_CACHE_DIR=/tmp/sys1_r6_a2; rm -rf "$TILELANG_CACHE_DIR"; mkdir -p "$TILELANG_CACHE_DIR"
      env -u PYTHONPATH -u TA_ROOT timeout 1500 python3 -c "import torch_npu; import sys, runpy; sys.argv=['run_delta_card.py','--target','ascend','--bwd']; runpy.run_path('ascend/port910b/run_delta_card.py', run_name='__main__')" > /tmp/w11_a2_ascend.log 2>&1
      echo "W11-A2-ASC-EXIT=$?"; grep -E "^ITEM |^NUM-delta_bwd" /tmp/w11_a2_ascend.log
    else
      echo "W11-A2-CPUGATE=DIFF (arm dropped; not going on-card)"
      head -20 /tmp/w11_a2_cpu_diff.txt | sed 's/^/W11-A2-DIFF /'
    fi
    echo "--- [4d] 回滚 ---"
    cp /tmp/w11_a2_gdn_asc.py.bak "$AE"
    echo "W11-A2-REVERTED=$(grep -c 'diff = v_scr\[j\] - p' "$AE")"
    md5sum "$AE" | sed 's/^/W11-A2-POST-MD5 /'
  else
    echo "=== [4] A 臂 SKIPPED（W11_A2=0，本机 pytest 门未过或未授权）==="
  fi
else
  echo "W11-R6-SKIP no-clone"
fi

echo "=== [5] 现场保全 ==="
tar czf /tmp/w11_artifacts.tgz -C /tmp w11_l0.log w11_ws.log w11_boot.log w11_r6_bwd.log w11_r6_cpu_base.log w11_r6_cpu_arm.log w11_a2_cpu_diff.txt w11_a2_ascend.log 2>/dev/null
echo "W11-TGZ size=$(wc -c < /tmp/w11_artifacts.tgz 2>/dev/null || echo 0)B"; echo "W11-DONE"
