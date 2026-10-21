#!/usr/bin/env bash
# examples/repro_p1.sh — 一阶段最小闭环一键复现：S0 词表 → S1 预训练 → S2 决策 SFT → S3 调温 → 四轴评测
#
# 职责（p1-10 B2）：把"从原始文字到评测数字"这条链一次性跑通，并在终端连排交出四轴摘要表
#   （决策质量 / 把握 / 耗时 / 跨后端一致）。**退出码即变更级 DoD 判定入口**：
#     0 = 链路跑通且评测判定通过（溯源不是豁免来的、被点名的轴都出了数、一致轴名次没换人）
#     1 = 评测判定不通过（四轴数字照打，红在 verdict 那一行）
#     2 = 前置产物缺失，一步都没跑（错误消息里给出补齐命令，绝不拿半截输入假装成功）
#     3 = 中间某一阶段（S0/S1/S2/S3）失败，日志留在 runs/repro-p1/logs/
#
# 用法：
#   bash examples/repro_p1.sh                     # 默认极小档 + CPU 口径
#   DEVICE=mps bash examples/repro_p1.sh          # 有 Metal 的机器（一致轴会真开模具）
#   EVAL_LIMIT=100 bash examples/repro_p1.sh      # 放大评测规模
#   可用变量：PYTHON / DEVICE / CORPUS_DIR / TOKENIZER_DIR / SFT_DATA / EVAL_DATA /
#             S1_STEPS / S2_EPOCHS / S2_HOLDOUT / EVAL_LIMIT / SPEED_SAMPLES / PARITY_SAMPLES
#
# 规模口径：全部走 tiny/smoke 档（S1 20 步、S2 一轮、评测 60 条），参考耗时"本机 CPU 约 2 分钟"，
#   设计目标为干净机器 <20 分钟。**慢机不判负**：时长只作参考，不做退出码。
#
# 前置产物（一阶段前序域的真实产出，本脚本不下载任何数据集）：
#   1) S0 词表目录        默认 runs/1003-s0-bpe-16k-realedu-zh-en/tokenizer
#        重建：python learning/s0_tokenizer.py --corpus mini --vocab-size 16000
#   2) tiny 编号流语料    默认 runs/corpus_tiny（S1 的输入）
#        重建：python -m sys1.data.pretrain_corpus --config learning/configs/corpus.yaml
#   3) 转写训练集 jsonl    默认 bench/p1-09/sft_train.jsonl（S2 的输入）
#        重建：python -m sys1.data.transcribe（XNLI/MASSIVE/rating 三条转写线，需本地数据集）
#   4) 评测集 jsonl        默认 bench/p1-09/typed_decisions/test.jsonl（typed-decisions@f7a2487e）
#        取不到时可退回 (3) 的留出子集：EVAL_DATA=auto 会自动改用留出子集并在摘要里标注来源
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 2
PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

DEVICE="${DEVICE:-cpu}"
TOKENIZER_DIR="${TOKENIZER_DIR:-runs/1003-s0-bpe-16k-realedu-zh-en/tokenizer}"
CORPUS_DIR="${CORPUS_DIR:-runs/corpus_tiny}"
SFT_DATA="${SFT_DATA:-bench/p1-09/sft_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-bench/p1-09/typed_decisions/test.jsonl}"
S1_STEPS="${S1_STEPS:-20}"
S2_EPOCHS="${S2_EPOCHS:-1}"
S2_HOLDOUT="${S2_HOLDOUT:-60}"
S2_SEED="${S2_SEED:-0}"                       # 与 s2 CLI 默认一致，留出切分靠它复现
EVAL_LIMIT="${EVAL_LIMIT:-60}"
SPEED_SAMPLES="${SPEED_SAMPLES:-30}"
PARITY_SAMPLES="${PARITY_SAMPLES:-8}"

WORK="runs/repro-p1"
LOGS="$WORK/logs"
mkdir -p "$LOGS"

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit "${2:-2}"; }
grab() { grep -oE "$2" "$1" | tail -1 | sed -E 's/.*: //'; }   # grab <log> <pattern>

START_TS=$SECONDS

# ── 阶段 0：前置体检（缺东西就在开工前说清楚，绝不跑半截冒充成功）──────────────────
step "阶段 0/5 前置体检"
[ -d "$TOKENIZER_DIR" ] || die "缺 S0 词表目录 $TOKENIZER_DIR；先跑：$PY learning/s0_tokenizer.py --corpus mini --vocab-size 16000"
[ -d "$CORPUS_DIR" ] || die "缺 tiny 语料目录 $CORPUS_DIR；先跑：$PY -m sys1.data.pretrain_corpus --config learning/configs/corpus.yaml"
[ -f "$SFT_DATA" ] || die "缺转写训练集 $SFT_DATA；先跑：$PY -m sys1.data.transcribe"
EVAL_SOURCE="typed-decisions"
if [ "$EVAL_DATA" = "auto" ] || [ ! -f "$EVAL_DATA" ]; then
    printf '⚠ 评测集 %s 不可用，改用转写训练集的留出子集（来源会写进摘要）\n' "$EVAL_DATA"
    EVAL_SOURCE="holdout-split"
    EVAL_DATA="$WORK/eval_holdout.jsonl"
fi
printf '   tokenizer=%s\n   corpus=%s\n   sft=%s\n   eval=%s (source=%s)\n' \
    "$TOKENIZER_DIR" "$CORPUS_DIR" "$SFT_DATA" "$EVAL_DATA" "$EVAL_SOURCE"

# ── 阶段 1：S0 词表（已就位则复用；REBUILD_TOKENIZER=1 时用内置 mini 语料重建）──────
step "阶段 1/5 S0 词表"
if [ "${REBUILD_TOKENIZER:-0}" = "1" ]; then
    S0_LOG="$LOGS/s0.log"
    $PY learning/s0_tokenizer.py --corpus mini --vocab-size 4000 --out "$WORK/tokenizer" > "$S0_LOG" 2>&1 \
        || { tail -20 "$S0_LOG"; die "S0 失败，日志见 $S0_LOG" 3; }
    TOKENIZER_DIR="$WORK/tokenizer"
    printf '   重建词表 -> %s\n' "$TOKENIZER_DIR"
else
    printf '   复用已有 S0 产物：%s（要重建请设 REBUILD_TOKENIZER=1）\n' "$TOKENIZER_DIR"
fi

# ── 阶段 2：S1 预训练（smoke_tiny 档，步数压到极小）────────────────────────────────
step "阶段 2/5 S1 预训练（smoke_tiny, max_steps=$S1_STEPS）"
S1_LOG="$LOGS/s1.log"
$PY learning/s1_pretrain_gpt.py --config smoke_tiny --corpus "$CORPUS_DIR" \
    --max-steps "$S1_STEPS" --device cpu --sample 0 --quiet > "$S1_LOG" 2>&1 \
    || { tail -20 "$S1_LOG"; die "S1 失败，日志见 $S1_LOG" 3; }
S1_RUN="$(grab "$S1_LOG" '\[s1\] run *: *[^ ]+')"
[ -n "$S1_RUN" ] || die "没从 $S1_LOG 抓到 S1 的 run 目录" 3
S1_MODEL="$S1_RUN/model"
[ -d "$S1_MODEL" ] || die "S1 没有交出模型目录 $S1_MODEL" 3
printf '   S1 run=%s\n   S1 model=%s\n' "$S1_RUN" "$S1_MODEL"
grep -E '\[s1\] (loss|steps/tokens)' "$S1_LOG" || true

# ── 阶段 3：S2 决策 SFT（吃 S1 起点 + 转写训练集，切出留出集）──────────────────────
step "阶段 3/5 S2 决策 SFT（epochs=$S2_EPOCHS, holdout=$S2_HOLDOUT）"
S2_LOG="$LOGS/s2.log"
$PY learning/s2_decision_sft.py --ckpt "$S1_MODEL" --tokenizer "$TOKENIZER_DIR" \
    --data "$SFT_DATA" --epochs "$S2_EPOCHS" --holdout "$S2_HOLDOUT" --seed "$S2_SEED" \
    --batch-size 16 > "$S2_LOG" 2>&1 \
    || { tail -20 "$S2_LOG"; die "S2 失败，日志见 $S2_LOG" 3; }
S2_MODEL="$(grab "$S2_LOG" '\[s2\] model dir *: *[^ ]+')"
S2_RUN_ID="$(grab "$S2_LOG" '\[s2\] run-id *: *[^ ]+')"
[ -n "$S2_MODEL" ] || die "没从 $S2_LOG 抓到 S2 的模型目录" 3
printf '   S2 run-id=%s\n   S2 model=%s\n' "$S2_RUN_ID" "$S2_MODEL"
grep -E '\[s2\] acc ' "$S2_LOG" || true

# ── 阶段 4：留出子集（供 S3 调温 + 评测集缺失时兜底；与训练集按同一种子互斥切分）────
step "阶段 4/5 复现留出子集（S3 校准输入）"
HOLD_PY="$WORK/holdout_split.py"
cat > "$HOLD_PY" <<'PYEOF'
"""按 s2 的同一套切分复现留出集，写 calibrate.jsonl（评测集缺失时兼作 eval_holdout.jsonl）。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from learning.s2_decision_sft import load_records, split_holdout

src, holdout, seed, outdir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
records = load_records(Path(src))
_, held = split_holdout(records, holdout, seed)
outdir.mkdir(parents=True, exist_ok=True)
for name in ("calibrate.jsonl", "eval_holdout.jsonl"):
    with (outdir / name).open("w", encoding="utf-8") as fh:
        for row in held:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"[holdout] seed={seed} 抽出 {len(held)} 条 -> {outdir}/calibrate.jsonl")
PYEOF
HOLD_LOG="$LOGS/holdout.log"
$PY "$HOLD_PY" "$SFT_DATA" "$S2_HOLDOUT" "$S2_SEED" "$WORK" > "$HOLD_LOG" 2>&1 \
    || { cat "$HOLD_LOG"; die "留出子集生成失败" 3; }
printf '   %s\n' "$(tail -1 "$HOLD_LOG")"
if [ "$EVAL_SOURCE" = "holdout-split" ]; then
    printf '   评测输入改用留出子集：%s（typed-decisions 未拉取，来源如实标注）\n' "$EVAL_DATA"
fi

# ── 阶段 5：S3 调温 + 四轴评测（退出码由评测判定决定）──────────────────────────────
step "阶段 5/5 S3 调温"
S3_LOG="$LOGS/s3.log"
$PY learning/s3_calibrate.py --model-dir "$S2_MODEL" --tokenizer "$TOKENIZER_DIR" \
    --data "$WORK/calibrate.jsonl" --min-samples 10 > "$S3_LOG" 2>&1 \
    || { tail -20 "$S3_LOG"; die "S3 失败，日志见 $S3_LOG" 3; }
grep -E '\[s3\] ' "$S3_LOG" | head -8
S3_RUN_ID="$(grab "$S3_LOG" '\[s3\] run-id *: *[^ ]+')"

step "四轴评测摘要（$EVAL_DATA, source=$EVAL_SOURCE, device=$DEVICE）"
EVAL_LOG="$LOGS/eval.log"
$PY -m sys1.eval.report --model "$S2_MODEL" --tokenizer "$TOKENIZER_DIR" --data "$EVAL_DATA" \
    --device "$DEVICE" --limit "$EVAL_LIMIT" --speed-samples "$SPEED_SAMPLES" \
    --parity-samples "$PARITY_SAMPLES" --record-root "$WORK/runs" 2>&1 | tee "$EVAL_LOG"
EVAL_RC="${PIPESTATUS[0]}"
EVAL_RUN="$(grep -oE 'run record -> [^ ]+' "$EVAL_LOG" | tail -1 | sed 's/run record -> //')"
printf '\n复现产物目录：%s（各阶段日志在 %s）\n' "$WORK" "$LOGS"
printf 'run-id 链：S1 %s → S2 %s → S3 %s → 评测 %s\n' \
    "$(basename "$S1_RUN")" "$S2_RUN_ID" "${S3_RUN_ID:-(见 $S3_LOG)}" "${EVAL_RUN:-(见 $EVAL_LOG)}"
printf '评测数据来源：%s（%s）\n' "$EVAL_SOURCE" "$EVAL_DATA"
printf '本次耗时：%d 秒（参考上限 1200 秒，仅参考、不参与判定）\n' "$((SECONDS - START_TS))"
if [ "$EVAL_RC" -eq 0 ]; then
    printf '\033[32m✓ 一阶段闭环复现通过（退出码 0）\033[0m\n'
else
    printf '\033[31m✗ 评测判定未通过（退出码 %s，四轴数字见 %s）\033[0m\n' "$EVAL_RC" "$EVAL_LOG"
fi
exit "$EVAL_RC"
