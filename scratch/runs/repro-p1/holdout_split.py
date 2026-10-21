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
