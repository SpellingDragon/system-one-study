"""A2 实测召回曲线 run：needle envelopes × Qwen3.5-0.8B on MPS（设备与上限如实标注）。"""
import sys; sys.path.insert(0, ".")
import json, pathlib, time, torch, glob
from sys1.runs import new_run

from production.assets import load_backbone
from sys1.decision import LETTERS, from_systemone, readout

ROWS = [json.loads(l) for l in open("bench/eval_data/assembled/needle-synthetic/needle-envelopes.jsonl")]
ctx = new_run("p2-07-a2-needle-curve-v3", config={
    "run": {"name": "p2-07-a2-needle-curve-v4"},
    "model": "Qwen3.5-0.8B 零样本基座（p2-01 载重口 load_backbone，无 SFT）",
    "device": "cpu（A2 设备边界取证：MPS 在 GDN conv1d 硬伤不支持——Qwen3.5 混合架构 Mac/MPS 前向不可行的实证，归 notes）", "dtype": "float32",
    "note": "v1 空跑/v2 render 形误均留档；v3 依 test_assets._decide 已验路执行",
})
bb = load_backbone("qwen3.5-0.8b", device="cpu", dtype=torch.float32)  # MPS 实测硬伤：GDN causal_conv1d conv 通道>65536 不支持（v3 取证），CPU fp32 口径
tok_ids = list(bb.letter_ids)
step = 0
by_bucket = {}; skipped = []
for r in ROWS:
    if int(str(r["id"]).split(":")[1]) >= 262144:
        skipped.append(r["id"]); continue   # 256K 超本地时间预算，归云端 C 波（如实）
    smp = r["sample"]
    qid, spec = "q", list(smp["questions"].values())[0]
    row = from_systemone(smp["state"], spec, qid=qid)
    ids, order = bb.encode_prompt(row)
    hidden = bb.forward(torch.tensor([ids], dtype=torch.long))
    out = readout(hidden, bb.model.get_output_embeddings().weight,
                  tok_ids[:len(order)], lengths=[len(ids)], qtypes=[row["type"]])
    probs = out.probs[0].detach().cpu().tolist()
    n = len(ids)
    b = int(str(r["id"]).split(":")[1])
    pred_code = order[int(probs and __import__("numpy").array(probs).argmax())]
    gold_code = str(r["needle"]["answer_key"])
    ok = pred_code == gold_code
    by_bucket.setdefault(b, []).append(ok)
    step += 1
    import resource; peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    ctx.log_metrics(step=step, bucket=b, ctx_tokens=n, pred=pred_code, gold=gold_code, ok=ok, peak_gb=round(peak, 2))
    print(f"[{step}/{len(ROWS)}] bucket={b} ids={n} ok={ok} peak={peak:.2f}GB", flush=True)
accs = {b: round(sum(v) / len(v), 3) for b, v in sorted(by_bucket.items())}
ctx.conclude(f"A2 召回曲线（零样本基座码位口径，device=mps）：" + " / ".join(f"{b}→{a}" for b, a in accs.items()) + f"；峰值内存见逐行 metrics（上限档如实），256K/1M 归云端 C 波。")
ctx.finish()
print("✓ run:", ctx.run_id, "| accs:", accs)
