#!/bin/bash
# run_window5_local.sh — 宿主机发射器（第五窗 / tilelang-ascend B 路线）
# ============================================================================
# 与 run_window4_local.sh 刻意保持"复制而非重构"：付费窗口前不动已验证过的发射链。
#
# 上传物（2 件）：
#   ① release/ascend/port910b/window5_ta_card.sh   （≈24 KB，自足：内联驱动）
#   ② refs/ta_pkg/ta_pkg.tar.gz                     （36,478,146 B = 34.8 MiB，sha256 d5607ec6…）
# 体积 / 上传时间（估）：ta_pkg 36.5 MB —— SSH ≥1 MB/s ⇒ ≤40s；≥0.3 MB/s ⇒ ≤2min。
#   上传后以 REMOTE_BYTES + sha256 前 16 位核对完整性，不整则 ABORT（不烧卡时）。
#
# 监控纪律（吸取窗二教训）：**不单看外层**——按内层日志行数 + 段落标记判进度：
#   outer /tmp/ta_all.log 的 "=== [N] " 段标记 + inner /tmp/ta_smoke.log、/tmp/ta_run.log 行数。
#   内层行数连续 10 轮（7.5 分钟）不增且无 python3 ⇒ STALLED 如实停报。
#
# 回读判决行：TA0-*（解包/env/快检）、TA-ABORT、TA-GDN-*、TA-GEMM、TA-KERNELPROOF、
#   TA-SUMMARY、TA1-EXIT、TA-TGZ、TA-DONE。性能行只标 track=ta-ascendc，不出结论。
# ============================================================================
set -u
K=/Users/pengweiye/Documents/codes/system-one/release/.venv/KeyPair-19e0.pem
D=/Users/pengweiye/Documents/codes/system-one/release/ascend/port910b
PKG=/Users/pengweiye/Documents/codes/system-one/refs/ta_pkg/ta_pkg.tar.gz
EXPECT_SHA=d5607ec6f9c321c9652fa7d25f234d2e17a074cf308938fa2c4b522830b02172
H=ma-user@dev-modelarts-cnsouth1.huaweicloud.com
S="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ServerAliveInterval=20 -i $K -p 31955"
chmod 400 "$K" 2>/dev/null
[ -f "$K" ] || { echo "LOCAL-ABORT key-not-found $K"; exit 1; }
[ -f "$PKG" ] || { echo "LOCAL-ABORT pkg-not-found $PKG"; exit 1; }

# ---- 上传卡脚本 ----
L=$(wc -c < "$D/window5_ta_card.sh"); echo "LOCAL window5_ta_card.sh=$L"
[ "$L" -lt 3000 ] && { echo "TRUNCATED-ABORT"; exit 1; }
cat "$D/window5_ta_card.sh" | ssh -T $S $H "cat > /tmp/window5_ta_card.sh; echo REMOTE_SCRIPT=\$(wc -c < /tmp/window5_ta_card.sh)" 2>/dev/null

# ---- 上传 ta_pkg ----
PB=$(wc -c < "$PKG"); LSHA=$(shasum -a 256 "$PKG" | cut -c1-16); echo "LOCAL ta_pkg.tar.gz=$PB sha16=$LSHA"
[ "$LSHA" = "${EXPECT_SHA:0:16}" ] || { echo "LOCAL-ABORT pkg-sha-mismatch local=$LSHA expect=${EXPECT_SHA:0:16}"; exit 1; }
cat "$PKG" | ssh -T $S $H "cat > /tmp/ta_pkg.tar.gz; echo REMOTE_TAR=\$(wc -c < /tmp/ta_pkg.tar.gz)" 2>/dev/null
RSHA=$(ssh -T $S $H 'sha256sum /tmp/ta_pkg.tar.gz 2>/dev/null | cut -c1-16' 2>/dev/null)
echo "REMOTE ta_pkg sha16=$RSHA"
[ "$RSHA" = "$LSHA" ] || { echo "LOCAL-ABORT tar-sha-mismatch local=$LSHA remote=$RSHA"; exit 1; }

# ---- 环境探针 + 起跑 ----
ssh -T $S $H 'echo "python3=$(pgrep -fc python3 2>/dev/null || echo 0)"; npu-smi info 2>/dev/null | grep -m1 "910B" || echo NPU-NOT-FOUND; python3 -c "import torch_npu;print(\"torch_npu=\",torch_npu.__version__)" 2>/dev/null || echo torch_npu-MISSING' 2>/dev/null
ssh -T $S $H 'rm -f /tmp/ta_smoke.log /tmp/ta_run.log /tmp/ta_w5_logs.tgz; : > /tmp/ta_all.log
(setsid bash /tmp/window5_ta_card.sh > /tmp/ta_all.log 2>&1 < /dev/null &)
echo LAUNCHED; sleep 20; head -8 /tmp/ta_all.log' 2>/dev/null

echo "--- 同步等待（上限 ~45 分钟；45s 一轮，按内层日志行数 + 段标记判进度）---"
QUIET=0; LAST=""
for i in $(seq 1 60); do
  R=$(ssh -T $S $H 'A=$(grep -c TA-DONE /tmp/ta_all.log 2>/dev/null || echo 0)
SEG=$(grep -oE "^=== \[[0-9]\] " /tmp/ta_all.log 2>/dev/null | tail -1 | tr -d "=[] ")
P=$(pgrep -fc python3 2>/dev/null || echo 0)
S1=$(wc -l < /tmp/ta_smoke.log 2>/dev/null || echo 0); S2=$(wc -l < /tmp/ta_run.log 2>/dev/null || echo 0)
echo "$A|$SEG|$P|$S1/$S2"' 2>/dev/null)
  DONE=$(echo "$R" | cut -d"|" -f1); SEG=$(echo "$R" | cut -d"|" -f2); P=$(echo "$R" | cut -d"|" -f3); CS=$(echo "$R" | cut -d"|" -f4)
  echo "[t=$((i*45))s] done=$DONE 段=${SEG:-起} py=$P 内层行(smoke/run)=$CS"
  [ "$DONE" = "1" ] && { echo COMPLETED; break; }
  if [ "$P" = "0" ] && [ "$CS" = "$LAST" ] && [ -n "$LAST" ]; then QUIET=$((QUIET+1)); else QUIET=0; fi
  LAST=$CS
  [ "$QUIET" -ge 10 ] && { echo "STALLED-7.5分钟全静默——如实停报"; break; }
  sleep 45
done

echo "--- 判决回读（TA-*）---"
ssh -T $S $H 'grep -nE "TA0-TAR|TA0-UNPACK|TA0-ENV|TA0-DEPS|TA0-SMOKE|TA-ABORT|TA-DEVICE|TA-MODE|TA-TILELANG|TA-REFCHOICE|^TA-GDN|^TA-GEMM|TA-KERNELPROOF|TA-SUMMARY|TA1-EXIT|TA-TGZ|TA-DONE|Traceback|Fatal" /tmp/ta_all.log | tail -70
echo "──── 末 10 行 ────"; tail -10 /tmp/ta_all.log' 2>/dev/null

# ---- 拉回现场日志（best-effort）----
ssh -T $S $H 'cat /tmp/ta_w5_logs.tgz 2>/dev/null' > "$(dirname "$PKG")/w5_logs_remote.tgz" 2>/dev/null
ls -l "$(dirname "$PKG")/w5_logs_remote.tgz" 2>/dev/null || echo "LOGS-FETCH-SKIP"
