#!/bin/bash
# m4_hostcheck.sh <tag> — 跑 host 四腿判决并把输出落 /tmp/m4host_<tag>.*
set -u
TAG="$1"
REL=$REPO_ROOT/release
PY=$REL/.venv/bin/python
( cd "$REL/ascend/port910b" && $PY run_delta_card.py --target cpu --bwd ) > /tmp/m4host_$TAG.delta 2>&1
echo "delta rc=$?"
( cd "$REL" && $PY ascend/gradcheck_card.py --target cpu ) > /tmp/m4host_$TAG.grad 2>&1
echo "grad rc=$?"
( cd "$REL" && $PY ascend/train_step.py --target cpu ) > /tmp/m4host_$TAG.step 2>&1
echo "step rc=$?"
( cd "$REL" && $PY ascend/train_step_real.py --target cpu --dry ) > /tmp/m4host_$TAG.real 2>&1
echo "real rc=$?"
echo "=== [$TAG] delta ==="; grep -E '^(ITEM|NUM-delta)' /tmp/m4host_$TAG.delta
echo "=== [$TAG] gradcheck tail ==="; grep -E '^GRADCHECK|PASS|FAIL|TOTAL|结论' /tmp/m4host_$TAG.grad | tail -14
echo "=== [$TAG] train_step ==="; grep -E '^P2-STEP|^KERNEL-USED|^DTYPE-TRACE' /tmp/m4host_$TAG.step
echo "=== [$TAG] train_step_real ==="; grep -E '^P2-REAL|VERDICT|^COMPILE-KEY|^COUNTS|结论' /tmp/m4host_$TAG.real | tail -20
