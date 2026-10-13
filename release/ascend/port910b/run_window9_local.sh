#!/bin/bash
# run_window9_local.sh — 第九窗发射器：上传 ta_pkg+TA 两件+card_run+窗脚本 → setsid → 轮询 → 回读+捞档
K=/Users/pengweiye/Documents/codes/system-one/release/.venv/KeyPair-19e0.pem
D=/Users/pengweiye/Documents/codes/system-one/release/ascend/port910b
TA=/Users/pengweiye/Documents/codes/tilelang-ascend/examples/gdn_bwd
PKG=/Users/pengweiye/Documents/codes/system-one/refs/ta_pkg/ta_pkg.tar.gz
RUNS=/Users/pengweiye/Documents/codes/system-one/release/runs/1013-p2-13-oncard-window9-ta-bwd
H=ma-user@dev-modelarts-cnsouth1.huaweicloud.com
S="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ServerAliveInterval=20 -i $K -p 31955"
chmod 400 "$K" 2>/dev/null
mkdir -p "$RUNS/artifacts"

echo "--- 上传小件 ---"
for f in card_run.sh window9_card.sh; do
  L=$(wc -c < "$D/$f"); echo "LOCAL $f=$L"
  cat "$D/$f" | ssh -T $S $H "cat > /tmp/$f; echo REMOTE=\$(wc -c < /tmp/$f)" 2>/dev/null
done
for f in gdn_bwd.py test_gdn_bwd.py; do
  L=$(wc -c < "$TA/$f"); echo "LOCAL $f=$L"
  cat "$TA/$f" | ssh -T $S $H "cat > /tmp/$f; echo REMOTE=\$(wc -c < /tmp/$f)" 2>/dev/null
done

echo "--- 上传 ta_pkg（34.8MB，sha 必核）---"
LOCAL_SHA=$(shasum -a 256 "$PKG" | cut -d' ' -f1)
cat "$PKG" | ssh -T $S $H "cat > /tmp/ta_pkg.tar.gz; echo PKG_BYTES=\$(wc -c < /tmp/ta_pkg.tar.gz); echo PKG_SHA=\$(sha256sum /tmp/ta_pkg.tar.gz | cut -d' ' -f1)" 2>/dev/null
echo "LOCAL_SHA=$LOCAL_SHA"

ssh -T $S $H "R=\$(sha256sum /tmp/ta_pkg.tar.gz | cut -d' ' -f1); [ \"\$R\" = \"$LOCAL_SHA\" ] || { echo PKG-SHA-MISMATCH; exit 1; }; echo PKG-OK" 2>/dev/null || { echo "ABORT pkg-sha"; exit 1; }

echo "--- 发射 ---"
ssh -T $S $H 'rm -f /tmp/w9_*.log /tmp/w9_artifacts.tgz; : > /tmp/w9_all.log
(setsid bash /tmp/window9_card.sh > /tmp/w9_all.log 2>&1 < /dev/null &)
echo LAUNCHED; sleep 20; head -6 /tmp/w9_all.log' 2>/dev/null

echo "--- 轮询（间隔 60s，上限 50 分钟；超时不杀卡上进程，可续查）---"
for i in $(seq 1 50); do
  R=$(ssh -T $S $H 'echo "$(grep -c W9-DONE /tmp/w9_all.log 2>/dev/null)|$(grep -oE "^=== \[[0-4]\]" /tmp/w9_all.log 2>/dev/null | tail -1)|py=$(pgrep -fc python3 2>/dev/null || echo 0)"' 2>/dev/null)
  echo "[t=$((i*60))s] $R"
  echo "$R" | grep -q "^1|" && { echo COMPLETED; break; }
  sleep 60
done

echo "--- 窗九判决回读 ---"
ssh -T $S $H 'grep -nE "W9-|^ITEM |^\[PRECISION|^NUM-delta_bwd|Traceback|FATAL|ABORT" /tmp/w9_all.log | tail -70' 2>/dev/null

echo "--- 捞现场档 ---"
ssh -T $S $H 'cat /tmp/w9_artifacts.tgz' > "$RUNS/artifacts/w9_artifacts.tgz" 2>/dev/null
tar xzf "$RUNS/artifacts/w9_artifacts.tgz" -C "$RUNS/artifacts/" 2>/dev/null
ssh -T $S $H 'cat /tmp/w9_all.log' > "$RUNS/artifacts/w9_all.log" 2>/dev/null
ls -la "$RUNS/artifacts/" | head -10
echo "LAUNCHER-DONE"
