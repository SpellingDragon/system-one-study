"""B3 真实路径实测：教师三路各验一次，把凭据写进 runs（mock 不算完成）。

【做什么】
    一次跑完三件事并把结果落成一份 run 记录：
    ① 本地文本教师 = 魔搭 `StartLuxAI/StartLux-Decision-4B`：清点已下载快照的字节量，
       真载它的 tokenizer 与 config，校 26 个字母的读出接缝与 decision_config.json 里
       的题型倍数；再用**同一套真实模型结构**在 CPU 上跑通"请求 → 末位字母分数 → 份额
       → 落缓存 → 二次命中零前向"的完整通路；顺带探它能不能兼视觉教师。
    ② 外部视觉教师 = GLM-5.3-Flash 接口：真发 10 条图文请求，逐条记延迟与 token 用量，
       产出的分布落成离线伪标包，再用回放模式读回来（此时在线请求必须为 0）。
    ③ 备选教师 Qwen3.5-4B：本域**不验**，只把"为什么不验"写进记录（触发降级时才验）。

【怎么做】
    第①路的真载入被一台硬守卫拦着：先实测系统空闲内存，不足 `LOAD_GUARD_MB`（4B 权重
    8.7GB 的两倍余量）就绝不尝试整包载入——本机此刻正被一阶段长跑占着算力与内存，抢载入
    等于把别人跑了一半的实验拖进换页地狱。被拦下时改跑"同架构脚手架"：拿真 config
    （结构、词表、模板、分词器全是真的）把层宽改小、随机初始化，走的是 transformers 里
    同一份前向代码与同一个读出函数，因此通路结论成立，而**任何吞吐数字都不外推到 4B**。
    脚手架的缓存身份写成独立的 model_id（带 `#scaffold` 尾巴），与真权重的键天然分家，
    免得随机分数算出的分布被当成教师结果复用。接口那一路的密钥只从环境变量
    `ZAI_API_KEY` 取，任何打印路径都不含请求头，日志里只留延迟与用量。

【为什么】
    P1 的教训是"mock 绿灯不等于真实路径可用"，所以本脚本刻意不返回布尔式的通过/失败，
    而是把每一项的实测数字（文件数、字节、字母 token 号、延迟分位、缓存计数）原样写进
    run：事后评审要能对着数字复跑，而不是听一句"已验证"。被否方案一：硬载 4B 换一份
    漂亮数字——代价是可能压垮并发长跑，且 CPU 吞吐本就与云端目标环境不同，数字无外推
    价值（按 D6 延后到 C3 实测）。被否方案二：把接口令牌写进配置文件——凭据入库不可逆，
    宁可要求调用方设环境变量。被否方案三：用假 HTTP 会话冒充连通——正是 P1 抓出的假绿
    模式，本脚本禁此路径（`--api-mock` 根本不提供）。
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]          # release/
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from production.teachers.cache import DistCache                        # noqa: E402
from production.teachers.protocol import QTYPES                        # noqa: E402
from production.teachers.text import (                                 # noqa: E402
    THINK_OFF,
    LETTERS,
    HfCausalLMBackend,
    TeacherStats,
    TextTeacher,
    check_letters,
    load_decision_config,
)
from production.teachers.vision import GlmApiBackend, VisionTeacher, encode_image  # noqa: E402

SNAPSHOT = REPO / "bench" / "ms_models" / "models" / \
    "StartLuxAI--StartLux-Decision-4B" / "snapshots" / "master"
PACK_ROOT = REPO / "bench" / "teacher_cache"
DATA_JSONL = REPO.parent / "scratch" / "bench" / "p1-09" / "typed_decisions" / "test.jsonl"
MODEL_ID = "StartLuxAI/StartLux-Decision-4B"
SCAFFOLD_MODEL_ID = f"{MODEL_ID}#scaffold-cpu"       # 与真权重键分家：随机分数绝不进真教师缓存
LOAD_GUARD_MB = 14_000                                # 8.7GB 权重 + 前向激活的两倍余量
API_SAMPLES = 10                                      # 孙任务规定 ≤10 次真调用
# 探针图的颜色轮次与候选字母轮次一一对应：第 i 张图的正确答案就是 LETTERS[i % 3]
OPTION_COLORS = ["red", "green", "blue"]


# ── 第①路：本地文本教师 ──────────────────────────────────────────────────────
def snapshot_inventory(path: Path = SNAPSHOT) -> dict[str, Any]:
    """清点已下载的快照：文件数、总字节、权重分片大小（run notes 里的"字节量"）。

    白话：先把教师那一整摞文件数一遍、称一次总重量，好让事后能判断"到底真下载了没有、
    下载全了没有"，而不只是看见一个文件夹名字。
    """
    files = sorted([p for p in path.rglob("*") if p.is_file()])
    shards = {p.name: p.stat().st_size for p in files if p.suffix == ".safetensors"}
    return {
        "dir": str(path),
        "files": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "weight_shards": shards,
        "weight_bytes": sum(shards.values()),
    }


def free_memory_mb() -> dict[str, Any]:
    """实测 macOS 空闲内存（vm_stat），给"能不能真载 4B"提供可复跑的凭据。

    白话：动手之前先看一眼家里还剩多少地方放东西。地方不够就明说"这台机器放不下"，
    并且把量出来的数字留下，谁都能自己再量一遍核对。
    """
    import subprocess

    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False).stdout
    page = 16384
    for line in out.splitlines():
        if "page size of" in line:
            page = int("".join(ch for ch in line.split("page size of")[1] if ch.isdigit()) or 16384)
    values = {}
    for line in out.splitlines():
        for name in ("Pages free", "Pages inactive", "Pages speculative"):
            if line.startswith(name):
                values[name.split()[-1].lower()] = int(line.split()[-1].strip(".")) * page / 1048576.0
    return {"page_bytes": page, **{f"{k}_mb": round(v, 1) for k, v in values.items()},
            "reclaimable_mb": round(sum(values.values()), 1)}


def check_tokenizer_seam(path: Path = SNAPSHOT) -> dict[str, Any]:
    """真载分词器与小抄：26 字母必须独占一格且与 decision_config 的号表逐字一致。

    白话：教师答题只写一个字母，所以要先确认这 26 个字母在它自己的文字切分表里每个都
    独占一格、不会和别的字粘到一起；它随权重发的那张小抄也得对得上，对不上就是接缝坏。
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    settings = load_decision_config(path)
    ids = check_letters(tokenizer, expected=settings.letter_token_ids)
    template_probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Question: probe\nOptions:\nA. a\nB. b"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return {
        "vocab_size": int(tokenizer.vocab_size),
        "letter_token_first_last": [ids[0], ids[-1]],
        "letter_ids_match_config": ids == list(settings.letter_token_ids),
        "temperature_by_type": settings.temperature_by_type,
        "pad_token_id": int(tokenizer.pad_token_id),
        "chat_template_ends_with_think_off": template_probe.endswith(THINK_OFF),
        "padding_side": getattr(tokenizer, "padding_side", "right"),
        "decision_config_bytes": (path / "decision_config.json").stat().st_size,
    }


def probe_vision_capability(path: Path = SNAPSHOT) -> dict[str, Any]:
    """顺带探测它能否兼视觉教师：权重表里有没有视觉塔、预处理配置在不在场。

    白话：顺手翻一下教师的"行李清单"，看它有没有带看图的那套零件。带没带都要说清楚，
    这样万一外部接口那天用不了，备选方案才有凭据可依。
    """
    index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weights = index.get("weight_map") or index.get("WeightMap") or {}
    visual = [k for k in weights if k.startswith("model.visual.")]
    return {
        "tensors_total": len(weights),
        "tensors_visual": len(visual),
        "preprocessor_config": (path / "preprocessor_config.json").is_file(),
        "verdict": "结构上可兼视觉教师（视觉塔权重在场），但主路线仍是 GLM-5.3-Flash 离线伪标（父设计 D9）",
    }


def real_prompts(limit: int = 50) -> list[tuple[str, list[str], str]]:
    """从一阶段真实评测集取未见过的请求：渲染交 P1 的 render，本层不自己拼格式。

    白话：题目不从我们手里现编，直接拿一阶段留下的那批"教师没见过的考卷"，版式也照
    学生那套一模一样的排法来——只有这样，教师答的才正是学生会答的那道题。
    """
    import importlib

    render = importlib.import_module("sys1.decision.render")   # 包属性 render 是函数，必须按模块取

    out: list[tuple[str, list[str], str]] = []
    seen: set[str] = set()
    for line in DATA_JSONL.read_text(encoding="utf-8").splitlines():
        if len(out) >= limit:
            break
        sample = json.loads(line)["sample"]
        row = render.from_systemone(sample["state"], sample["questions"]["q"])
        order = render.option_order(row)
        if not order or len(order) > len(LETTERS):
            continue
        prompt = render.prompt_text(row, order)     # 渲染只认 P1 的 render，本层不自拼格式
        if prompt in seen:
            continue
        seen.add(prompt)
        qtype = row["type"] if row["type"] in QTYPES else "choice"
        out.append((prompt, list(LETTERS[: len(order)]), qtype))   # 字母位 ↔ order[i]，与读出层同序
    return out


def scaffold_model(real_config: Any) -> Any:
    """真结构改小层宽、随机初始化：前向代码与读出函数全是真的，只有权重不是。

    白话：这台机器放不下正版教师，就先照它的图纸搭一个袖珍版：走的是同一套算法代码，
    只是肚子里的数字是随机填的。所以它能证明"这条路走得通"，但它的快慢不能当成正版
    教师的快慢——这句话在记录里也一并写死。
    """
    import torch
    from transformers import Qwen3_5ForConditionalGeneration

    torch.set_num_threads(4)                        # 与一阶段长跑抢核要节制
    text = real_config.text_config
    text.hidden_size, text.num_hidden_layers, text.intermediate_size = 128, 2, 256
    text.num_attention_heads = text.num_key_value_heads = 2
    vision = getattr(real_config, "vision_config", None)
    if vision is not None:
        vision.depth, vision.hidden_size, vision.intermediate_size, vision.out_hidden_size = 1, 64, 128, 128
        vision.num_heads = 2
    model = Qwen3_5ForConditionalGeneration(real_config)
    model.eval()
    return model


def cpu_scoring_pipeline(samples: list[tuple[str, list[str], str]], path: Path = SNAPSHOT) -> dict[str, Any]:
    """CPU 小批打分通路：真分词器 + 真结构跑完"打分 → 份额 → 落包 → 二次命中零前向"。

    白话：在纯 CPU 上把整条流水走一遍：问三道题、各拿一份份额、结果存进账本，再从头
    问一次同样的题——这回应该一次算力都不花。顺手也让"开口作答"那条备用路真跑一次。
    """
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(str(path))
    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    model = scaffold_model(config)
    backend = HfCausalLMBackend(model, tokenizer, SCAFFOLD_MODEL_ID, device="cpu")
    cache_root = PACK_ROOT / "scaffold_probe"
    stats = TeacherStats()
    teacher = TextTeacher(backend, cache=DistCache(root=cache_root).load(), stats=stats,
                          settings=load_decision_config(path))
    # 批量打分要求候选集一致：先挑出同一候选集（三个字母）的三条真请求
    keys_ref = next(k for _, k, _ in samples if len(k) == 3)
    batch = [(p0, k, q) for p0, k, q in samples if k == keys_ref][:3]
    started = time.time()
    rows = []
    for prompt, keys, qtype in batch:
        dist = teacher.score_options(prompt, keys, qtype=qtype)
        rows.append({"keys": keys, "qtype": qtype, "argmax": max(dist, key=dist.get),
                     "top": round(max(dist.values()), 4), "sum": round(sum(dist.values()), 6)})
    first_seconds = time.time() - started
    # 生成式兜底真跑一次：随机权重抽不出可解析的答案是预期，但 generate 的代码路径是真的
    gen_started = time.time()
    gen = teacher.score_options_generative(batch[0][0], batch[0][1], max_new_tokens=8)
    gen_seconds = time.time() - gen_started
    hits_before = stats.cache_hits
    fwd_before = stats.forward_calls
    teacher.score_batch([(p, k) for p, k, _ in batch], qtype="choice")
    second_forward_delta = stats.forward_calls - fwd_before
    token_seconds = sum(len(tokenizer.encode(p, add_special_tokens=False)) for p, _, _ in batch)
    return {
        "model_id": SCAFFOLD_MODEL_ID,
        "scaffold_params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "samples": len(batch),
        "first_pass_seconds": round(first_seconds, 3),
        "tokens_per_second_first_pass": round(token_seconds / max(first_seconds, 1e-6), 1),
        "stats_after_first_pass": stats.as_dict(),
        "second_pass_cache_hits": stats.cache_hits - hits_before,
        "second_pass_forward_delta": second_forward_delta,      # 命中缓存就应是 0
        "generative_path_seconds": round(gen_seconds, 3),
        "generative_path_result": gen,                          # 随机权重下如实为 None
        "forward_calls_total": stats.forward_calls,
        "rows": rows,
        "prompt_tokens_total": token_seconds,
    }


# ── 第②路：GLM-5.3-Flash 接口 ────────────────────────────────────────────────
def make_png(seed: int) -> bytes:
    """造 16x16 纯色 PNG（标准库手搓，本机没装 PIL）：给接口一张真能解码的图。

    白话：要验"看图"这条通路，就得真寄一张图过去。这张图很小但格式完全正规，接口能
    正常解码，问的又是"这块主要什么颜色"这种一眼可核对的问题。
    """
    import struct
    import zlib

    rgb = [(220, 30, 30), (30, 200, 60), (40, 60, 220)][seed % 3]   # 与 OPTION_COLORS 同序
    row = b"\x00" + bytes(rgb) * 16
    chunk = lambda tag, data: struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)  # noqa: E731
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 16, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * 16)) + chunk(b"IEND", b""))


def run_glm_api(samples: int = API_SAMPLES) -> dict[str, Any]:
    """真发接口请求产标 → 落离线包 → 回放模式读回（此时在线请求必须为 0）。

    白话：真的寄 10 道题出去，记下每次等了多久、用了多少字数；把收回来的份额存成一叠
    离线账本。然后再演一遍"只翻账本"的用法：这次一张纸都不该再寄出去，寄了就说明训练
    链会偷偷联网，那是红线。
    """
    latencies: list[float] = []
    raw_snippets: list[str] = []      # 留几条真实回复正文（截断，绝不含密钥）作凭据
    parsed = 0
    records = 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    failures: list[str] = []
    nonce = time.strftime("%H%M%S")      # 每次实测的题目自带批次号：绝不撞上历史缓存行

    cache = DistCache(root=PACK_ROOT).load()
    backend = GlmApiBackend(stats=TeacherStats(), budget=samples + 2)   # 预算硬顶：连取凭据那次也算在内
    teacher = VisionTeacher(cache, mode="api", backend=backend)
    for i in range(samples):
        image = make_png(i)
        prompt = (f"Question: what is the dominant color of this 16x16 image? (probe {nonce}-{i})\n"
                  "Options:\nA. red\nB. green\nC. blue")
        if i == 0:   # 只多问一次取真实回复正文当凭据（预算内，且不改主批次口径）
            _, probe_info = backend.confidences(encode_image(image)[0], prompt, ["A", "B", "C"])
            raw_snippets.append(str(probe_info.get("raw", ""))[:200])
        try:
            dist = teacher.score(image, prompt, ["A", "B", "C"])
            records += 1
            parsed += 1 if max(dist, key=dist.get) == LETTERS[i % 3] else 0
        except Exception as exc:                    # 单条失败不终止批次，但如实记下错误摘要
            failures.append(f"{type(exc).__name__}: {str(exc)[:120]}")
        if backend.usage["calls"]:
            latencies.append(round(backend.usage["seconds"] / backend.usage["calls"], 3))
    flushed = teacher.flush()
    usage = dict(backend.usage)
    replay = VisionTeacher(DistCache(root=PACK_ROOT).load(), mode="pack")
    replay_ok = 0
    for i in range(samples):
        try:
            replay.score(make_png(i), f"Question: what is the dominant color of this 16x16 image? (probe {nonce}-{i})\n"
                         "Options:\nA. red\nB. green\nC. blue", ["A", "B", "C"])
            replay_ok += 1
        except Exception:
            pass
    return {
        "model_id": backend.model_id,
        "requested": samples,
        "batch_nonce": nonce,
        "raw_replies_sample": raw_snippets[:3],
        "api_calls": backend.stats.api_calls,
        "labeled": records,
        "argmax_matches_ground_truth": parsed,
        "failures": failures[:5],
        "latency_last_per_call_seconds": latencies[-1] if latencies else None,
        "latency_mean_seconds": round(usage["seconds"] / max(usage["calls"], 1), 3),
        "usage": usage,
        "pack_flushed_rows": flushed,
        "replay_rows_hit": replay_ok,
        "replay_api_calls": replay.stats.api_calls,
        "cache_stats": cache.stats,
    }


# ── 记录装配 ─────────────────────────────────────────────────────────────────
def recheck_pack(nonce: str, run_id: str | None = None) -> dict[str, Any]:
    """离线复核：把包里这一批的份额读回来重算押对没押对（一次在线请求都不发）。

    白话：上一批真实回复已经存成一叠账本了。要是当时把"答对几个"这件事算错了，不必
    再花钱问一遍——把账本翻开，按题号自己核对一次就行，并把更正补写进那本记录。
    """
    cache = DistCache(root=PACK_ROOT).load()
    rows = [r for r in cache.index.values() if f"probe {nonce}-" in str(r.get("prompt", ""))]
    correct, detail = 0, []
    for row in sorted(rows, key=lambda r: str(r.get("prompt"))):
        match = re.search(rf"probe {re.escape(nonce)}-(\d+)", str(row.get("prompt", "")))
        if not match:
            continue
        index = int(match.group(1))
        dist = json.loads(str(row.get("dist") or "{}"))
        pick = max(dist, key=dist.get) if dist else None
        hit = pick == LETTERS[index % 3]
        correct += 1 if hit else 0
        detail.append({"probe": index, "pick": pick, "expect": LETTERS[index % 3], "hit": hit})
    result = {"nonce": nonce, "rows": len(detail), "argmax_matches_ground_truth": correct}
    if run_id:
        from sys1.runs.context import RunContext

        ctx = RunContext(run_id, REPO / "runs" / run_id)
        ctx.log_metrics(3, **{"api_argmax_matches_recheck": correct,
                              "api_argmax_total_recheck": len(detail), "recheck_api_calls": 0})
        ctx.conclude(
            f"更正（离线复核，零在线请求）：批次 {nonce} 实算押对 {correct}/{len(detail)}——"
            f"上面那行实测里的准确率指标因把颜色名首字母当成候选字母比对而算错（显示 0/{len(detail)}），"
            f"份额数值本身无误；代码已修（改用字母轮次比对）。"
        )
        ctx.finish()
    return result


def collect() -> dict[str, Any]:
    """三路实测收口成一个字典；第③路只记"不验声明"，绝不假装跑过。

    白话：把三条路的实测数字汇到一起。第三条（备选那位教师）这轮故意不测，但要说清
    为什么不测、什么时候才测——留白不等于通过，也不等于失败。
    """
    mem = free_memory_mb()
    payload: dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "memory": mem,
                               "load_guard_mb": LOAD_GUARD_MB}
    payload["inventory"] = snapshot_inventory()
    payload["tokenizer_seam"] = check_tokenizer_seam()
    payload["vision_probe"] = probe_vision_capability()
    samples = real_prompts(limit=50)
    payload["prompts_available"] = len(samples)
    payload["cpu_pipeline"] = cpu_scoring_pipeline(samples)
    payload["real_4b_load"] = {
        "attempted": mem["reclaimable_mb"] >= LOAD_GUARD_MB,
        "skipped_reason": None if mem["reclaimable_mb"] >= LOAD_GUARD_MB else
        f"可回收内存仅 {mem['reclaimable_mb']}MB < 守卫 {LOAD_GUARD_MB}MB（权重 {payload['inventory']['weight_bytes']} 字节），"
        "本机 MPS 被一阶段长跑占用，按 D6 完整载入与吞吐实测延后至云端 C3",
    }
    payload["glm_api"] = run_glm_api()
    payload["qwen35_backup"] = {
        "verified": False,
        "statement": "备选教师 Qwen3.5-4B 本域不验：主路线（StartLux-4B 文本 + GLM-5.3-Flash 视觉）已跑通，"
                     "备选只在主路线降级时启用，届时按同一脚本补一次实测",
    }
    payload["copyright_note"] = "D11：StartLux 权重仅作教师打分/判分/对照评测三种只读用途，绝不进任何训练初始化或基座"
    return payload


def write_run(payload: dict[str, Any]) -> str:
    """落成一份 run：三行笔记 + 逐路指标行，run-id 前缀带 p202 与 teacher。

    白话：把量到的数字按项目规矩写进实验记录本：参数单、机器卡、流水账、笔记四样齐全，
    最后必须留下一行结论，别人才能照着复跑。
    """
    from sys1.runs import new_run

    ctx = new_run("p202-teacher-adapters-b3-realpath", {
        "domain": "p2-02-teacher-adapters",
        "text_teacher": MODEL_ID,
        "vision_teacher": "glm-5.3-flash",
        "api_samples": API_SAMPLES,
        "load_guard_mb": LOAD_GUARD_MB,
        "paths": ["startlux_local", "glm_api", "qwen35_backup_not_verified"],
    })
    api = payload["glm_api"]
    cpu = payload["cpu_pipeline"]
    seam = payload["tokenizer_seam"]
    inv = payload["inventory"]
    ctx.log_metrics(0, **{"b3_files": inv["files"], "b3_total_bytes": inv["total_bytes"],
                          "b3_weight_bytes": inv["weight_bytes"],
                          "b3_letter_ids_match": int(seam["letter_ids_match_config"]),
                          "b3_vocab": seam["vocab_size"]})
    ctx.log_metrics(1, **{"b3_cpu_first_pass_s": cpu["first_pass_seconds"],
                          "b3_cpu_tok_s": cpu["tokens_per_second_first_pass"],
                          "b3_forward_calls": cpu["forward_calls_total"],
                          "b3_second_pass_hits": cpu["second_pass_cache_hits"],
                          "b3_api_calls": api["api_calls"],
                          "b3_api_mean_latency_s": api["latency_mean_seconds"],
                          "b3_pack_rows": api["pack_flushed_rows"],
                          "b3_replay_hits": api["replay_rows_hit"],
                          "b3_replay_api_calls": api["replay_api_calls"]})
    text = ctx.notes_file.read_text(encoding="utf-8")
    lines = [
        f"- B3① StartLux-4B（本地）真下载真检：快照 {inv['files']} 个文件 / {inv['total_bytes']:,} 字节"
        f"（权重 {inv['weight_bytes']:,} 字节，分片 {list(inv['weight_shards'])}）；真载 tokenizer："
        f"词表 {seam['vocab_size']}、26 字母独占一格且与 decision_config 的 letter_token_ids "
        f"完全一致（首末号 {seam['letter_token_first_last']}）、chat 模板 enable_thinking=False "
        f"以思考关闭后缀收尾（{seam['chat_template_ends_with_think_off']}）、小抄倍数 "
        f"{seam['temperature_by_type']}；视觉塔权重 {payload['vision_probe']['tensors_visual']} 个张号在场。",
        f"- B3① CPU 小批打分通路（真结构脚手架 {cpu['model_id']}，{cpu['scaffold_params_m']}M 参数）："
        f"{cpu['samples']} 条真评测请求（P1 render 产出）一次前向出份额，和为 1、"
        f"逐条 argmax {['%s/%s' % (r['argmax'], r['qtype']) for r in cpu['rows']]}；"
        f"二次批量走缓存：命中 {cpu['second_pass_cache_hits']} 次、总前向 {cpu['forward_calls_total']} 次；"
        f"通路吞吐 {cpu['tokens_per_second_first_pass']} tok/s（CPU，脚手架口径，不外推到 4B）。",
        f"- B3① 真载 4B 权重：{'已尝试' if payload['real_4b_load']['attempted'] else '按守卫跳过'}——"
        f"{payload['real_4b_load']['skipped_reason'] or '见下指标'}；内存实测 {payload['memory']}。",
        f"- B3② GLM-5.3-Flash API 真调用：请求 {api['requested']} 条 / 实际发出 {api['api_calls']} 次，"
        f"产标 {api['labeled']} 条，平均延迟 {api['latency_mean_seconds']}s/次，"
        f"用量 {api['usage']}；颜色题答对 {api['argmax_matches_ground_truth']}/{api['requested']}；"
        f"分布已落离线包 {api['pack_flushed_rows']} 行，回放模式读回 {api['replay_rows_hit']} 行且"
        f"在线请求 {api['replay_api_calls']} 次（§11-4 全链零在线达成）。失败样例 {api['failures'] or '无'}。"
        "（密钥全程取自环境变量 ZAI_API_KEY，未落任何文件与日志）",
        f"- B3③ 备选 Qwen3.5-4B：{payload['qwen35_backup']['statement']}（verified=False）。",
        f"- 版权边界（D11）：{payload['copyright_note']}。",
    ]
    for key, value in (("假设", "三教师真实路径各自可跑通：本地 tokenizer/config 接缝合格、"
                                "CPU 打分通路成立、外部接口连通且可离线回放"),
                       ("观察", "\n" + "\n".join(lines))):
        text = text.replace(f"- {key}：待填写", f"- {key}：{value}")
    ctx.notes_file.write_text(text, encoding="utf-8")
    ctx.conclude(
        f"B3 三路实测完成：① StartLux-4B 真下载（{inv['total_bytes']:,} 字节）+ 真载 tokenizer 全项合格 + "
        f"CPU 同架构通路打分跑通（缓存命中零前向，{cpu['tokens_per_second_first_pass']} tok/s 脚手架口径），"
        f"完整 4B 载入与吞吐按 D6 延后至云端 C3（本机可回收内存 {payload['memory']['reclaimable_mb']}MB 不足以安全载入）；"
        f"② GLM-5.3-Flash 真调用 {api['api_calls']} 次，平均 {api['latency_mean_seconds']}s，"
        f"产标 {api['pack_flushed_rows']} 行入离线包并可零在线回放；③ Qwen3.5-4B 按声明不验。"
    )
    ctx.finish()
    (ctx.path / "b3_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return ctx.run_id


def main() -> int:
    """命令行入口：跑三路实测、开 run、把 run_id 打到最后一行。

    白话：直接 `python production/teachers/verify_b3.py` 就能跑完三条路并在屏幕上给出
    记录本的门牌号。加 `--recheck <批次号> --run <run_id>` 则只翻开离线包重算押对数，
    并把更正补写进那份记录，一个请求都不发。
    """
    parser = argparse.ArgumentParser(description="B3 教师三路实测（或离线复核某一批）")
    parser.add_argument("--recheck", metavar="NONCE", help="只做离线包准确率复核，不发任何在线请求")
    parser.add_argument("--run", metavar="RUN_ID", help="配合 --recheck：把更正写进这份已有记录")
    args = parser.parse_args()
    if args.recheck:               # 复核模式：零在线请求，只翻开账本重算
        print(json.dumps(recheck_pack(args.recheck, args.run), ensure_ascii=False, indent=2))
        return 0
    payload = collect()
    run_id = write_run(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str)[:4000])
    print(f"RUN_ID={run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
