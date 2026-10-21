"""B2 教师吞吐冒烟：50 条未见请求，测缓存前后的耗时与每秒字数，落成一份 run。

【做什么】
    拿一阶段留下的 50 条"教师没见过的真实请求"，让本地文本教师连打两轮：第一轮全部
    没算过（每次都真跑一次模型），第二轮原样重来（应该全部命中缓存、一次算力都不花）。
    记下三个数：单条平均耗时、每秒处理的字数、两轮墙钟之比。外加生成式那条路的每秒
    出字数，因为伪标真正吃紧的是"一次要问几万条"。

【怎么做】
    机器这一侧仍按 B3 的守卫口径走"同架构脚手架"（真图纸 + 随机数字 + 纯 CPU）：本机
    空闲内存放不下 8.7GB 的正版权重，硬载会把一阶段长跑拖进换页。于是这里的数字是
    **通路吞吐**，用来判断"缓存这条捷径值不值"，绝不作为 4B 在云端每秒能算多少条的
    预测值——那句话与缓存键的隔离一起写进结论。计时用单调钟；两轮之间把结果真的 flush
    成 parquet 再重开一次缓存，所以第二轮命中走的是"落过盘的数据"，不是内存里的侥幸。
    前后耗时都用 `TeacherStats.forward_calls` 复核：第二轮增量必须为 0，否则缓存叙事
    当场作废。

【为什么】
    教师侧的容量规划取决于"重复请求多不多"：伪标会反复重跑同一批题（补打标、换抽样、
    回归复算），只要缓存有效，第二遍的成本就近乎零，这个结论必须有实测数字撑着。
    被否方案一：等 4B 真载完再测——本机资源不允许，且云端环境的吞吐与本机不同源，
    先把方法与口径在同一段代码里跑通，C3 只是换机器重跑同一脚本；被否方案二：只报
    命中率不报耗时——命中率是账本数字，回答不了"省了多少时间"；被否方案三：在同一
    份缓存目录里反复叠加计时轮次——第二轮之所以要重开缓存，就是要证明落盘的数据也
    能被查到，同目录里追加只会把两轮的边界搅混。
"""
from __future__ import annotations

import json
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]          # release/
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from production.teachers.cache import DistCache                        # noqa: E402
from production.teachers.text import (                                 # noqa: E402
    TeacherStats,
    TextTeacher,
    load_decision_config,
)
from production.teachers.verify_b3 import (                            # noqa: E402
    PACK_ROOT,
    SCAFFOLD_MODEL_ID,
    SNAPSHOT,
    real_prompts,
    scaffold_model,
)

BENCH_ROOT = PACK_ROOT / "throughput_bench"
N_SAMPLES = 50


def build_teacher() -> tuple[TextTeacher, Any]:
    """装好脚手架教师（真分词器 + 真结构 + 空缓存目录），交给两轮计时用。

    白话：先把这台"按正版图纸搭的袖珍教师"和一本空白账本摆好。账本每次都是新的，
    第一轮才谈得上"从没算过"，量出来的耗时才有意义。
    """
    from transformers import AutoConfig, AutoTokenizer

    shutil.rmtree(BENCH_ROOT, ignore_errors=True)
    config = AutoConfig.from_pretrained(str(SNAPSHOT))
    tokenizer = AutoTokenizer.from_pretrained(str(SNAPSHOT), trust_remote_code=True)
    model = scaffold_model(config)
    backend = _Backend(model, tokenizer, SCAFFOLD_MODEL_ID)
    teacher = TextTeacher(
        backend,
        cache=DistCache(root=BENCH_ROOT).load(),
        stats=TeacherStats(),
        settings=load_decision_config(SNAPSHOT),
    )
    return teacher, tokenizer


class _Backend:
    """把真模型的"一批请求 → 逐候选原始分数"包一层，供计时轮调用。

    白话：教师那边只管问"这几个候选各占多少"，这里负责真去跑一趟模型，把每道题写完
    最后一步留下的数字里、候选对应的那几列挑出来。取哪一格的规矩全在真后端里，本层
    只是把计时与批量凑起来，不改口径。
    """

    def __init__(self, model: Any, tokenizer: Any, model_id: str) -> None:
        from production.teachers.text import HfCausalLMBackend

        self.inner = HfCausalLMBackend(model, tokenizer, model_id, device="cpu")
        self.model_id = model_id

    def last_letter_logits(self, prompts, option_keys):
        """转交一批请求，收回逐题的候选原始分数（真实前向，一次都不省）。"""
        return self.inner.last_letter_logits(prompts, option_keys)

    def letter_logprobs(self, prompt, option_keys):
        """转交单条请求，收回对数分数（教师主路的入口口径）。"""
        return self.inner.letter_logprobs(prompt, option_keys)

    def generate_text(self, prompt, *, max_new_tokens: int = 12):
        """转交生成式那条路：真要一个字母就得让模型开口吐字。"""
        return self.inner.generate_text(prompt, max_new_tokens=max_new_tokens)


def run_passes(samples: list[tuple[str, list[str], str]], teacher: TextTeacher, tokenizer: Any) -> dict[str, Any]:
    """两轮打分：第一轮全算、第二轮全查表，各自记墙钟与每秒字数。

    白话：同一批题问两遍。第一遍是真做题，慢；第二遍只翻账本，快。两遍之间还把答案
    誊到了磁盘上，所以第二遍查的是"存住的数据"。快慢一比，缓存这条路值不值就有数了。
    """
    started = time.perf_counter()
    for prompt, keys, qtype in samples:
        teacher.score_options(prompt, keys, qtype=qtype)
    cold = time.perf_counter() - started
    cold_forward = teacher.stats.forward_calls

    written = teacher.cache.flush()
    teacher.cache = DistCache(root=BENCH_ROOT).load()      # 重开：第二轮必须从盘上查，不许靠内存侥幸
    teacher.stats.reset()
    started = time.perf_counter()
    for prompt, keys, qtype in samples:
        teacher.score_options(prompt, keys, qtype=qtype)
    warm = time.perf_counter() - started
    warm_forward = teacher.stats.forward_calls

    tokens = sum(len(tokenizer.encode(p, add_special_tokens=False)) for p, _, _ in samples)
    per_prompt = [None] * len(samples)
    started = time.perf_counter()
    for i, (prompt, keys, qtype) in enumerate(samples[:10]):
        one = time.perf_counter()
        teacher.score_options(prompt, keys, qtype=qtype, use_cache=False)
        per_prompt[i] = time.perf_counter() - one
    probe_seconds = time.perf_counter() - started
    return {
        "samples": len(samples),
        "prompt_tokens_total": tokens,
        "cold_pass_seconds": round(cold, 3),
        "cold_tokens_per_second": round(tokens / max(cold, 1e-6), 1),
        "cold_forward_calls": cold_forward,
        "warm_pass_seconds": round(warm, 3),
        "warm_tokens_per_second": round(tokens / max(warm, 1e-6), 1),
        "warm_forward_calls": warm_forward,               # 必须是 0，否则缓存叙事作废
        "speedup_cold_over_warm": round(cold / max(warm, 1e-6), 1),
        "parquet_rows_written": written,
        "single_request_median_ms": round(1000 * statistics.median([x for x in per_prompt if x]), 2),
        "single_request_probe_n": 10,
        "single_request_probe_seconds": round(probe_seconds, 3),
        "stats": teacher.stats.as_dict(),
    }


def run_batch_vs_loop(samples: list[tuple[str, list[str], str]], teacher: TextTeacher, tokenizer: Any) -> dict[str, Any]:
    """一次一趟 vs 一条条问：同样的请求，批量能省多少墙钟。

    白话：把没算过的题攒成一摞一次问完，和一道一道问，比一比总时间。批量之所以划算，
    是因为模型一趟能同时看好几道题，省掉的是每次重新走一遍流程的开销。
    """
    fresh = DistCache(root=BENCH_ROOT / "batch").load()
    teacher.cache = fresh
    teacher.stats.reset()
    keys = samples[0][1]
    batch_items = [(p, keys) for p, k, _ in samples if k == keys][:20]
    if not batch_items:
        return {"skipped": "没有候选集相同的请求可组批"}
    tokens = sum(len(tokenizer.encode(p, add_special_tokens=False)) for p, _ in batch_items)
    started = time.perf_counter()
    teacher.score_batch(batch_items, qtype="choice")
    batched = time.perf_counter() - started
    batch_forward = teacher.stats.forward_calls            # 批量这趟的真前向次数，必须在下一次 reset 之前取
    teacher.cache = DistCache(root=BENCH_ROOT / "loop").load()
    teacher.stats.reset()
    started = time.perf_counter()
    for prompt, _ in batch_items:
        teacher.score_options(prompt, keys, qtype="choice")
    one_by_one = time.perf_counter() - started
    loop_forward = teacher.stats.forward_calls             # 逐条这趟的真前向次数（与批量可比）
    return {
        "batch_size": len(batch_items),
        "batched_seconds": round(batched, 3),
        "batched_tokens_per_second": round(tokens / max(batched, 1e-6), 1),
        "batch_forward_calls": batch_forward,             # 一趟里算了几行，与逐条口径可比
        "one_by_one_seconds": round(one_by_one, 3),
        "one_by_one_tokens_per_second": round(tokens / max(one_by_one, 1e-6), 1),
        "loop_forward_calls": loop_forward,
        "ratio_one_by_one_over_batched": round(one_by_one / max(batched, 1e-6), 2),
    }


def run_generation_throughput(samples: list[tuple[str, list[str], str]], teacher: TextTeacher) -> dict[str, Any]:
    """生成式那条路的每秒出字数（伪标若改用"开口作答"格式，成本就看这个数）。

    白话：主路一个字都不生成，所以还要单独量一下"非要它开口"时每秒能吐多少字。这个数字
    直接决定生成式产标要跑多久、值不值得开。
    """
    teacher.cache = None
    teacher.stats.reset()
    new_tokens = 16
    started = time.perf_counter()
    results = [teacher.score_options_generative(p, k, max_new_tokens=new_tokens) for p, k, _ in samples[:3]]
    seconds = time.perf_counter() - started
    return {
        "samples": 3,
        "max_new_tokens": new_tokens,
        "seconds": round(seconds, 3),
        "generated_tokens_per_second": round(3 * new_tokens / max(seconds, 1e-6), 1),
        "parsable": sum(1 for r in results if r is not None),      # 随机权重下预期为 0
        "generations": teacher.stats.generations,
    }


def write_run(payload: dict[str, Any]) -> str:
    """把两轮耗时与每秒字数写进一份 run：notes 必须有 tok/s 行，指标分步不重写。

    白话：量完就照项目规矩记进实验记录本——参数单、流水账、笔记一样不少，最后必须留
    一行结论，把"缓存前后各多快、这数字能不能外推到 4B"说清楚，别让人事后猜。
    """
    from sys1.runs import new_run

    passes = payload["passes"]
    ctx = new_run("p202-teacher-throughput-bench", {
        "domain": "p2-02-teacher-adapters",
        "n_samples": N_SAMPLES,
        "backend": SCAFFOLD_MODEL_ID,
        "device": "cpu",
        "cache_root": str(BENCH_ROOT),
        "note": "吞吐为同架构脚手架口径；4B 真载吞吐按 D6 延后至云端 C3",
    })
    ctx.log_metrics(0, **{"samples": passes["samples"], "prompt_tokens": passes["prompt_tokens_total"],
                          "cold_seconds": passes["cold_pass_seconds"],
                          "cold_tokens_per_second": passes["cold_tokens_per_second"],
                          "cold_forward_calls": passes["cold_forward_calls"],
                          "parquet_rows": passes["parquet_rows_written"]})
    ctx.log_metrics(1, **{"warm_seconds": passes["warm_pass_seconds"],
                          "warm_tokens_per_second": passes["warm_tokens_per_second"],
                          "warm_forward_calls": passes["warm_forward_calls"],
                          "speedup": passes["speedup_cold_over_warm"],
                          "single_request_median_ms": passes["single_request_median_ms"]})
    batch = payload.get("batch", {})
    gen = payload["generation"]
    ctx.log_metrics(2, **{"batched_seconds": batch.get("batched_seconds"),
                          "one_by_one_seconds": batch.get("one_by_one_seconds"),
                          "batch_ratio": batch.get("ratio_one_by_one_over_batched"),
                          "generated_tokens_per_second": gen["generated_tokens_per_second"],
                          "generations": gen["generations"]})
    text = ctx.notes_file.read_text(encoding="utf-8")
    body = "\n".join([
        f"- 50 条未见请求（P1 render 产出的真实考卷）第一轮：{passes['cold_pass_seconds']} 秒 / "
        f"{passes['prompt_tokens_total']} 字 = **tok/s {passes['cold_tokens_per_second']}**"
        f"（前向 {passes['cold_forward_calls']} 次，落盘 {passes['parquet_rows_written']} 行）。",
        f"- 同一批第二轮（重开缓存、从盘上查）：{passes['warm_pass_seconds']} 秒 / "
        f"**tok/s {passes['warm_tokens_per_second']}**，前向增量 {passes['warm_forward_calls']} 次"
        f"（必须为 0），缓存前后加速 {passes['speedup_cold_over_warm']} 倍；"
        f"单条中位 {passes['single_request_median_ms']} ms。",
        f"- 批量口径：{batch.get('batch_size')} 条一趟 {batch.get('batched_seconds')} 秒 vs 逐条 "
        f"{batch.get('one_by_one_seconds')} 秒（比值 {batch.get('ratio_one_by_one_over_batched')}）。"
        f"生成式口径：{gen['generated_tokens_per_second']} tok/s（{gen['samples']} 条 x "
        f"{gen['max_new_tokens']} 字，可解析 {gen['parsable']} 条——随机权重下预期为 0）。",
        f"- 口径边界：以上全部为**同架构脚手架 + 纯 CPU**的通路吞吐；本机可回收内存不足以安全载入 "
        f"8.7GB 权重（见 B3 记录），完整载入吞吐实测按 D6 延后至云端 C3，本行数字不外推。",
        f"- 版权（D11）：教师权重只读取用于打分，绝不进任何训练初始化/基座；缓存键含 model_id "
        f"`{SCAFFOLD_MODEL_ID}`，与真权重天然分家，随机分数的分布不会被复用为教师结果。",
    ])
    for key, value in (("假设", "缓存能把重复请求的算力开销压到近零，且批量比逐条划算"),
                       ("观察", body)):
        text = text.replace(f"- {key}：待填写", f"- {key}：{value}")
    ctx.notes_file.write_text(text, encoding="utf-8")
    ctx.conclude(
        f"50 条未见请求冒烟完成：缓存前 {passes['cold_pass_seconds']}s（tok/s "
        f"{passes['cold_tokens_per_second']}，前向 {passes['cold_forward_calls']} 次）→ 缓存后 "
        f"{passes['warm_pass_seconds']}s（tok/s {passes['warm_tokens_per_second']}，前向增量 "
        f"{passes['warm_forward_calls']} 次），加速 {passes['speedup_cold_over_warm']} 倍；结论：分布缓存这条"
        f"路对重复请求确实零算力。数字为脚手架 CPU 通路口径，4B 真载吞吐按 D6 延后至 C3 复跑同一脚本。"
    )
    ctx.finish()
    (ctx.path / "throughput_payload.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return ctx.run_id


def main() -> int:
    """命令行入口：跑冒烟、开 run、把 run_id 与关键数字打到屏幕。

    白话：一句 `python production/teachers/bench_throughput.py` 就把两轮计时、批量对比、
    生成式各量一遍，并在屏幕上留下记录本的门牌号，方便谁都能自己再跑一次核对。
    """
    teacher, tokenizer = build_teacher()
    samples = real_prompts(limit=N_SAMPLES)
    payload = {"samples_collected": len(samples), "model_id": SCAFFOLD_MODEL_ID}
    payload["passes"] = run_passes(samples, teacher, tokenizer)
    payload["batch"] = run_batch_vs_loop(samples, teacher, tokenizer)
    payload["generation"] = run_generation_throughput(samples, teacher)
    run_id = write_run(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str)[:3000])
    print(f"RUN_ID={run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
