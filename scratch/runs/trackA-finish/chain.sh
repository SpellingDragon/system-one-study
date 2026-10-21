#!/usr/bin/env bash
# 轨A 收官链：dc8c 正式预训练模型 → S2 决策 SFT → S3 调温 → 四轴评测（P1 正式手动档口径）
set -euo pipefail
cd "$(dirname "$0")/../.."   # scratch 根
V=.venv/bin/python; PY=$V
W=runs/trackA-finish; L=$W/logs
S1_MODEL=runs/1005-s1-mps-main-dc8c/model
TOK=runs/1003-s0-bpe-16k-realedu-zh-en/tokenizer
[ -d "$S1_MODEL" ] || { echo "缺 S1 模型 $S1_MODEL"; exit 2; }

echo "== S2 决策 SFT（正式 600 题口径 epochs=1 lr=0.03）=="
$PY learning/s2_decision_sft.py --ckpt "$S1_MODEL" --tokenizer "$TOK" \
    --data bench/p1-09/sft_train.jsonl --epochs 1 --lr 0.03 --holdout 60 --seed 0 \
    --batch-size 16 > "$L/s2.log" 2>&1
S2_MODEL=$(grep -oE '\[s2\] model dir *: *[^ ]+' "$L/s2.log" | awk '{print $NF}')
S2_RUN=$(grep -oE '\[s2\] run-id *: *[^ ]+' "$L/s2.log" | awk '{print $NF}')
[ -d "$S2_MODEL" ] || { tail -20 "$L/s2.log"; echo "S2 无模型目录"; exit 3; }
echo "   S2 run=$S2_RUN model=$S2_MODEL"

echo "== 切 calibrate/eval_holdout（与 s2 同种子互斥，复现 repro 口径）=="
$PY - "$S2_MODEL" <<'PYX'
import sys
sys.path.insert(0, "learning"); sys.path.insert(0, ".")
from learning.s2_decision_sft import load_records, split_holdout
import json, pathlib
recs = load_records("bench/p1-09/sft_train.jsonl")
train, hold = split_holdout(recs, holdout=60, seed=0)
out = pathlib.Path("runs/trackA-finish"); 
for name, rows in (("calibrate.jsonl", hold), ("eval_holdout.jsonl", hold)):
    (out/name).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows)+"\n", encoding="utf-8")
print(f"holdout={len(hold)} -> calibrate/eval_holdout")
PYX

echo "== S3 调温 =="
$PY learning/s3_calibrate.py --model-dir "$S2_MODEL" --tokenizer "$TOK" \
    --data "$W/calibrate.jsonl" --min-samples 10 > "$L/s3.log" 2>&1 || { tail -20 "$L/s3.log"; exit 3; }
grep -E '\[s3\] ' "$L/s3.log" | head -6
S3_RUN=$(grep -oE '\[s3\] run-id *: *[^ ]+' "$L/s3.log" | awk '{print $NF}')

echo "== 四轴评测（typed test 正式集，device mps，limit 150）=="
$PY -m sys1.eval.report --model "$S2_MODEL" --tokenizer "$TOK" \
    --data bench/p1-09/typed_decisions/test.jsonl \
    --device mps --limit 150 --speed-samples 30 --parity-samples 20 \
    --record-root runs > "$L/eval.log" 2>&1; EVAL_RC=$?
tail -30 "$L/eval.log"
EVAL_RUN=$(grep -oE 'run record -> [^ ]+' "$L/eval.log" | tail -1 | sed 's/run record -> //')
{ echo "结论（轨A 收官链）：S1 dc8c(2000步, loss 9.85->3.43) → S2 $S2_RUN → S3 $S3_RUN → EVAL ${EVAL_RUN:-见 $L/eval.log}(rc=$EVAL_RC)；链日志 runs/trackA-finish/logs/。"; } > "$W/CHAIN_NOTES.txt"
echo "== 链完成 rc=$EVAL_RC =="; exit $EVAL_RC
