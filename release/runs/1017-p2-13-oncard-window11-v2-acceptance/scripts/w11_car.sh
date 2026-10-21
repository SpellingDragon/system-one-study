#!/bin/bash
# w11_car.sh — 单锁 + 清单校验 + 分腿执行 + 判活轮询（第 52 节复盘的流程落地）
#
# 用法:
#   bash w11_car.sh <legs>            # legs: step one zero box nd dw 的任意组合
#   例: bash w11_car.sh "step box"
#
# 设计要点（对应台账 [52] 的"发车三律"）:
#   1) 发车前对 MANIFEST 内每个文件做本地/远端 md5 比对，不符立即退出（不跑旧码）
#   2) 卡上用 flock 单锁，禁并发车
#   3) 判活看 python 进程与 cache 目录增长，不看"日志静默"（曾两次误杀编译中的车）
#   4) 每腿独立 cache 目录；日志与判决统一落 artifacts/w11car/
set -u
K=$REPO_ROOT/release/.venv/${CARD_KEY}
H=${CARD_SSH}
SSH="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o ServerAliveInterval=20 -i $K -p 31955"
SRC=$WORKSPACE_ROOT/tilelang-ascend/examples/gdn_bwd_v2
RUNS=$REPO_ROOT/release/runs/1017-p2-13-oncard-window11-v2-acceptance
OUT="$RUNS/artifacts/w11car"
LEGS="${1:-step one box}"
chmod 400 "$K" 2>/dev/null

echo "=== [gate] 三档输入门（本地，缺档即拒跑）==="
GATE_OUT=$(PYTHONPATH=$WORKSPACE_ROOT/tilelang-ascend/examples/gdn_bwd_v2 \
  python3 $WORKSPACE_ROOT/tilelang-ascend/examples/gdn_bwd_v2/_process/_w11_gate.py 2>&1) \
  || { echo "$GATE_OUT" | tail -5; echo "ABORT input gate"; exit 4; }
echo "$GATE_OUT" | tail -2

# ---- 清单：改动必须同步这里（文件 -> /tmp 名） ----
MANIFEST=(
  "$SRC/gdn_bwd_v2.py:gdn_bwd_v2.py"
  "$SRC/test_gdn_bwd_v2.py:test_gdn_bwd_v2.py"
  "$SRC/_process/_w11p9_step.py:_w11p9_step.py"
  "$SRC/_process/_w11p9_one.py:_w11p9_one.py"
  "$SRC/_process/_w11_s2a_boxes.py:_w11_s2a_boxes.py"
  "$SRC/_process/_w11_s2b_zero.py:_w11_s2b_zero.py"
  "$SRC/_process/_w11_num_diag.py:_w11_num_diag.py"
)

echo "=== [0] liveness + TA ==="
ok=0
for i in $(seq 1 12); do
  ssh -T $SSH "$H" 'echo ALIVE' 2>/dev/null | grep -q ALIVE && { ok=1; break; }
  sleep 10
done
[ "$ok" = 1 ] || { echo "ABORT unreachable"; exit 2; }
T=$(ssh -T $SSH "$H" 'ls /tmp/ta_unpack/ta_pkg/build/tvm/libtvm.so >/dev/null 2>&1 && echo TA-OK || echo TA-MISSING' 2>/dev/null | tail -1)
echo "TA=$T"
if [ "$T" != "TA-OK" ]; then
  PKG=$REPO_ROOT/refs/ta_pkg/ta_pkg.tar.gz
  cat "$PKG" | ssh -T $SSH "$H" 'cat > /tmp/ta_pkg.tar.gz'
  ssh -T $SSH "$H" 'rm -rf /tmp/ta_unpack; mkdir -p /tmp/ta_unpack; tar xzf /tmp/ta_pkg.tar.gz -C /tmp/ta_unpack; echo UNPACK-OK'
fi

echo "=== [1] manifest md5 (拒跑旧码) ==="
fail=0
for item in "${MANIFEST[@]}"; do
  f="${item%%:*}"; b="${item##*:}"
  L=$(md5 -q "$f")
  R=$(cat "$f" | ssh -T $SSH "$H" "cat > /tmp/$b; md5sum /tmp/$b | cut -d' ' -f1" 2>/dev/null | tail -1)
  if [ "$L" = "$R" ]; then echo "OK   $b $L"; else echo "FAIL $b local=$L remote=$R"; fail=1; fi
done
[ "$fail" = 0 ] || { echo "ABORT manifest mismatch"; exit 3; }

echo "=== [2] launch (flock 单锁) ==="
ssh -T $SSH "$H" "rm -f /tmp/w11car_all.log /tmp/w11car.lock /tmp/w11dump_*.npy /tmp/w11carddv_*.npy /tmp/w11car_*.log; : > /tmp/w11car_all.log
cat > /tmp/w11car_run.sh <<'EOF'
#!/bin/bash
set -u
exec 9>/tmp/w11car.lock; flock -n 9 || { echo \"ABORT another car holds the lock\"; exit 9; }
export TA_ROOT=/tmp/ta_unpack/ta_pkg
source /usr/local/Ascend/cann-8.5.2/set_env.sh >/dev/null 2>&1
export ASCEND_NPU_ARCH=dav-2201
export BISHENG_HOME=\$(dirname \"\$(dirname \"\$(command -v bisheng)\")\")
export PYTHONPATH=\"\$TA_ROOT:\$TA_ROOT/3rdparty/tvm/python:/tmp\"
export LD_LIBRARY_PATH=\"\$TA_ROOT/build/tvm:\$TA_ROOT/build:\${LD_LIBRARY_PATH:-}\"
cd /tmp
export TILELANG_CACHE_DIR=/tmp/w11car_cache; rm -rf \"\$TILELANG_CACHE_DIR\"; mkdir -p \"\$TILELANG_CACHE_DIR\"
run_leg () {  # name cmd timeout
  echo \"=== LEG \$1 ===\"
  timeout \"\$3\" bash -c \"\$2\" > /tmp/w11car_\$1.log 2>&1
  echo \"\$1-EXITC=\$?\"
  grep -aE '^STEP |^ONE|^STAT-|^out[0-9]_|DUMP-|PROBE-|PV-|RuntimeError|Traceback|5070|NAMING|S2ABOX|W11DIAG|S2BZERO' /tmp/w11car_\$1.log | tail -20
}
for leg in $LEGS; do
  case \$leg in
    step)   run_leg step   'python3 -u /tmp/_w11p9_step.py' 900 ;;
    one)    run_leg one    'python3 -u /tmp/_w11p9_one.py float16 basic' 900 ;;
    onet)   run_leg onet   'python3 -u /tmp/_w11p9_one.py float16 typical' 1500 ;;
    h0)     run_leg h0     'python3 -u /tmp/_w11p9_one.py float32 h0' 1500 ;;
    onem)   run_leg onem   'python3 -u /tmp/_w11p9_one.py float16 multi' 1500 ;;
    one32)  run_leg one32  'python3 -u /tmp/_w11p9_one.py float32 basic' 900 ;;
    zero)   run_leg zero   'python3 -u /tmp/_w11_s2b_zero.py float16' 900 ;;
    box)    run_leg box    'python3 -u /tmp/_w11_s2a_boxes.py float16 normal on' 900 ;;
    boxb)   run_leg boxb   'python3 -u /tmp/_w11_s2a_boxes.py float16 normal zero' 900 ;;
    dw)     run_leg dw     'GDN_S2A_DW_DBG=1 python3 -u /tmp/_w11_s2a_boxes.py float16 normal on' 900 ;;
    dw2)    run_leg dw2    'GDN_S2A_DW_DBG=2 python3 -u /tmp/_w11_s2a_boxes.py float16 normal on' 900 ;;
    nd)     run_leg nd     'python3 -u /tmp/_w11_num_diag.py' 1200 ;;
    ndb)    run_leg ndb    'GDN_S2B_DBG=1 python3 -u /tmp/_w11_num_diag.py' 1200 ;;
    ndm)    run_leg ndm    'python3 -u /tmp/_w11_num_diag.py l0_multi_fp16' 1200 ;;
    ndv)    run_leg ndv    'GDN_S2A_VN_DBG=1 GDN_S2A_VN_CC=1 python3 -u /tmp/_w11_num_diag.py l0_multi_fp16' 1200 ;;
    *)      echo \"unknown leg \$leg\" ;;
  esac
done
echo W11CAR-DONE
EOF
chmod +x /tmp/w11car_run.sh
(setsid bash /tmp/w11car_run.sh > /tmp/w11car_all.log 2>&1 < /dev/null &)
echo CAR-LAUNCHED"

echo "=== [3] poll（进程/缓存活性，不看日志静默）==="
waited=0
while [ $waited -lt 240 ]; do
  sleep 20; waited=$((waited+1))
  body=$(ssh -T $SSH "$H" 'cat /tmp/w11car_all.log' 2>/dev/null)
  echo "$body" | grep -q "W11CAR-DONE" && { echo "[done @ $((waited*20))s]"; break; }
  if [ $((waited % 6)) -eq 0 ]; then
    P=$(ssh -T $SSH "$H" 'P=$(pgrep -f "python3 -u /tmp/_w11" | tail -1); [ -n "$P" ] && ps -o pcpu= -p $P | tr -d " " || echo idle' 2>/dev/null | tail -1)
    echo "[t=$((waited*20))s] cpu=$P cache=$(ssh -T $SSH "$H" 'ls /tmp/w11car_cache 2>/dev/null | wc -l' 2>/dev/null | tail -1)"
  fi
done

if ! ssh -T $SSH "$H" 'grep -q W11CAR-DONE /tmp/w11car_all.log' 2>/dev/null; then
  echo "PULL-EARLY: W11CAR-DONE not seen -- later legs may still be running"
fi
echo "=== [4] pull ==="
mkdir -p "$OUT"
# dump / output tensors (for the host-side P^T dO recomputation)
for f in /tmp/w11dump_1.npy /tmp/w11dump_2.npy /tmp/w11box_out2.npy; do
  b=$(basename "$f")
  ssh -T $SSH "$H" "cat $f" 2>/dev/null > "$OUT/$b" || true
  [ -s "$OUT/$b" ] && echo "PULLED $b $(wc -c < "$OUT/$b")B" || rm -f "$OUT/$b"
done
ssh -T $SSH "$H" 'cat /tmp/w11car_all.log; for f in /tmp/w11car_*.log; do echo "----- $f -----"; grep -aE "^STEP |^ONE|^STAT-|^out[0-9]_|DUMP-|PROBE-|PV-|RuntimeError|Traceback|5070|NAMING|S2ABOX|W11DIAG|S2BZERO|EXITC" $f | tail -40; done' 2>/dev/null > "$OUT/w11car_verdict.txt"
tail -60 "$OUT/w11car_verdict.txt"
echo "W11CAR-FIRE-DONE"
