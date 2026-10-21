"""learning/s1_pretrain_gpt.py - S1 从零预训练（随机初始化起步，学下一词，留曲线与存档，末尾采样冒烟）。

目的（PRODUCTION §8.1 口径）：证明"不载任何第三方权重、只从随机数字起步也能学出语言信号"，
并把这条学习曲线连同可续训的存档一起留在 runs/ 里当证据（GUIDE §7-0 红线只约束本目录）。
输入：--config 档位名或 yaml（learning/configs/*.yaml，全部超参都在里面，代码不硬编码）；
      语料来自 p1-08 装配好的分片编号流（--corpus 可覆盖配置里的目录）。
产出 run-id：默认 runs/<MMDD>-<run.name>-<参数散列4位>（sys1.runs.new_run 建四件套），
      目录含 metrics.jsonl（loss/lr/tok_per_s 逐步追加、同 step 同指标防重）、ckpt/step-N/
      （模型目录 + 优化器状态 + 指纹）、model/（最终模型）、notes.md（采样原文 + 判定 + 结论）。
预计耗时：smoke_tiny 档 CPU 约 1–2 分钟（300 步 × 32×128 ≈ 1.2M 编号）；mps_main 档 MPS 数小时。

【做什么】把一堆中英混排文本切成窗口，喂给一个随机初始化的小模型，让它一遍遍猜"这一位的下一个
片段是哪个"，猜错的代价越学越小；中途随时把学到的数字存档，断了能接着练，练完再让它自己往外
蹦几句话看看像不像人写的。
【怎么做】a)  读配置：全部尺寸与超参取自 yaml，代码里只有一份默认值都不留；b)  开跑前先建 run
目录（四件套由 sys1.runs 负责），曲线只经 log_metrics 追加，同一步同一指标写第二次会被拒；
c)  取数：语料是 uint32 分片编号流，按内存映射随机切窗口（答案就是输入整体左移一位，装配侧已
保证窗口不跨片）；d)  每一步：模型前向得到逐位置的数字串，再乘独立输出层拿到每个候选的分，按交叉熵算猜错的代价，
反向传播后按累积步数更新一次；学习率走 warmup + cosine；e)  每 log_interval 步写一行曲线（loss 取窗口
均值），每 save_interval 步存一份存档（模型目录 + 优化器状态 + 配置指纹 + 随机状态）；f)  一发现
这一位的平均值不是有限数就地中断，把情况如实写进 notes 再抛错，绝不带着一堆坏数字继续练；
g)  --resume 走"读回最近存档  再  校验指纹一致  再  从断点步继续"，续训后的曲线不会出现重复 step；
h)  收尾按配置采样若干短句写进 notes，并按"至少一条能读"的口径如实下结论。
【为什么】训练循环用 torch 原生手写，不用框架的 Trainer -- 本仓一阶段的目标是让读者能逐行看懂
"猜下一个片段"这件事是怎么一步步算出来的，框架把循环藏进回调后，答辩时指不出哪一行负责什么
（design 技术要点）。被否方案一：载一个现成的小模型再微调 -- 省下的正是本域要证明的东西，直接
违反 §7-0。被否方案二：每步都写一行曲线 -- 300 步的档还行，正式档几百万步会把流水账写成噪声，
且写盘节奏干扰吞吐测量，故按窗口均值定时写。被否方案三：续训时只认权重不校验配置 -- 改过模型
宽度或学习率还照续，会得到一条"前半段一个模型、后半段另一个模型"的假曲线，比不续训更难查，
故指纹不一致一律拒续（spec 断点续训场景）。被否方案四：存档只留最新一份 -- 中途崩在半步上会
整份作废，按步号留多份并配轮转上限（keep_last），既能在任意一份上续，也不把磁盘写满。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# 仓库根引导：脚本要支持 `python learning/s1_pretrain_gpt.py` 从任意 cwd 直跑，
# 而 editable 安装在部分环境没把仓库根挂上 sys.path（兄弟脚本 s2/s3 同法兜底）。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sys1.data import pretrain_corpus as corpus_mod
from sys1.lang import bpe
from sys1.model import Decoder, ModelConfig
from sys1.runs import RunContext, new_run

# ── 目录锚点与落盘名字（p1-09 按"模型目录"消费 model/，不感知权重来路）────────────────
CONFIG_DIR = REPO_ROOT / "learning" / "configs"
CKPT_DIRNAME = "ckpt"
LATEST_FILE = "latest.json"
STATE_FILE = "train_state.pt"
META_FILE = "meta.json"
MODEL_DIRNAME = "model"
# 续训一致性只看这三段：改设备、改日志节奏、改采样条数都不该挡续训，改尺寸和学习率必须挡住
FINGERPRINT_SECTIONS = ("model", "data", "train")
# train 段里这几项是"怎么收手、怎么写盘"的量，不进指纹：max_steps 续训本来就是要往后调的，
# 把它算进代号会让每一次正当续训都被自己拒掉（写盘节奏与设备同理）。
VOLATILE_TRAIN_KEYS = ("max_steps", "log_interval", "save_interval", "keep_last", "device", "threads")
CJK = re.compile(r"[\u4e00-\u9fff]")
LATIN_WORD = re.compile(r"[A-Za-z]{3,}")
# 机械初筛的可读性判据：有连续两个汉字，或有连续一个 3 字母以上的英文单词，才算"像话"
MIN_CJK_RUN = 2
MIN_LATIN_WORD = 3


class ResumeError(RuntimeError):
    """续训前提不成立：没有可用存档、或配置指纹对不上。宁可不续，也不接一条假曲线。"""


class DivergedError(RuntimeError):
    """这一位的平均值不再是有限数（NaN/Inf）：当场中断，把已经观察到的情况原样交代清楚。"""


class _Quiet:
    """进度打印开关（默认打）：--quiet 只关掉逐步进度行，最终摘要照打。"""

    steps = False


QUIET = _Quiet()


# ══ B1a-1 配置：尺寸与超参一律取自 yaml，代码里不留第二处默认值 ════════════════════════

def load_config(name_or_path: str | Path) -> dict[str, Any]:
    """读训练配置：可以是档位名（查 learning/configs/<名>.yaml），也可以是完整 yaml 路径。

    白话：把这次要用的那份设置单取回来。单子要么按名字去固定那个抽屉里找，要么人家直接
    给了完整路径。拿回来先看要紧的几段（模型尺寸、取数、怎么练）齐不齐，缺一段就停下问，
    绝不"缺了就按我猜的补"——猜出来的尺寸和单子上写的不是一个东西，跑完的数字就没法交代。
    """
    path = Path(str(name_or_path))
    if path.suffix not in {".yaml", ".yml"}:
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"找不到配置 {path}；learning/configs/ 下现有档位: "
                                f"{sorted(p.stem for p in CONFIG_DIR.glob('*.yaml'))}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = [k for k in ("model", "data", "train") if k not in cfg]
    if missing:
        raise ValueError(f"配置 {path.name} 缺段落 {missing}；model/data/train 三段都必须写全")
    cfg.setdefault("run", {"name": path.stem})
    cfg["_config_path"] = str(path)
    return cfg


def fingerprint(cfg: dict[str, Any]) -> str:
    """对 model/data/train 三段算一个短指纹，续训时用它判断"还是不是同一件事在继续"。

    白话：把这次设置单里最要紧的三段抄下来算一小串代号。续训的时候先比这串代号：对得上，
    说明接的还是同一个模型、同一套练法，可以往后写；对不上，说明中间动过尺寸或学习率，
    硬接会得到一条前半截和后半截不是一个模型画出来的曲线——这种曲线看着像进步，其实什么都
    不是，所以一律拒绝。改设备、改写盘节奏、改"练到第几步收手"、改采样条数都不在比对范围
    内——那些不动"学的是哪个模型"；其中收手步数续训时本来就要往后调，把它算进代号会把每次
    正当的接续都拒之门外，故见 VOLATILE_TRAIN_KEYS。
    """
    parts = {}
    for key in FINGERPRINT_SECTIONS:
        section = cfg.get(key) or {}
        if key == "train":
            section = {k: v for k, v in section.items() if k not in VOLATILE_TRAIN_KEYS}
        parts[key] = section
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def resolve_device(want: str) -> torch.device:
    """按配置要的设备实取一个可用设备（auto 依次试 mps、cuda、cpu），并如实回一个名字串。

    白话：单子上写着想在哪台机器上跑，就先去确认那台真在；写"随便"就按苹果显卡、英伟达显卡、
    中央处理器的顺序往下试。试到哪个用哪个，并把最后用的是谁回出去，好让记录里留下证据。
    """
    if want == "auto":
        for cand in ("mps", "cuda", "cpu"):
            if _device_available(cand):
                return torch.device(cand)
        return torch.device("cpu")
    if not _device_available(want):
        raise RuntimeError(f"配置要 device={want}，但这台机器上不可用（auto 会自动回退）")
    return torch.device(want)


def _device_available(name: str) -> bool:
    """设备是否可用：mps/cuda 问 torch，cpu 恒真。"""
    if name == "cpu":
        return True
    if name == "mps":
        return torch.backends.mps.is_available()
    return torch.cuda.is_available()


def build_model(cfg: dict[str, Any]) -> Decoder:
    """按配置里的尺寸随机初始化一个解码器——这是全局唯一的模型创建入口（红线自查点）。

    白话：照单子上写好的宽窄深浅空造一个模型，里面每个数字都是摇出来的初始值，不从任何
    别人练好的文件里取。整份脚本只有这一处把模型造出来，所以"有没有偷偷接现成的"只要看
    这一个函数就能回答，审计不用翻全文。
    """
    model = cfg["model"]
    backend = str(cfg["train"].get("kernel_backend", "off"))
    return Decoder(ModelConfig(**{k: v for k, v in model.items() if k in
                                  {"d", "L", "heads", "ctx", "vocab", "rope_theta", "seed",
                                   "ffn_mult", "norm_eps", "pad_token_id"}}),
                 kernel_backend=backend)


def lr_at(step: int, train: dict[str, Any]) -> float:
    """warmup 线性爬升 + cosine 退火到 min_lr_frac 的学习率曲线（每一步都按公式现算）。

    白话：一开始别把步子迈太大（数字全是摇来的，一上来猛踩容易飞出去），先按固定步数一点点
    加大；到顶之后按一条逐渐平缓的弧线收小，收尾时留一点点余量而不是一路踩到零——留的那点
    余量让最后几段还在动，不至于停在半路。
    """
    base = float(train["lr"])
    warmup = int(train.get("warmup_steps", 0))
    total = int(train["max_steps"])
    floor = base * float(train.get("min_lr_frac", 0.1))
    if warmup > 0 and step < warmup:
        return base * (step + 1) / warmup
    progress = max(0.0, min(1.0, (step - warmup) / max(1, total - warmup)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + (base - floor) * cosine


def next_batch(reader: corpus_mod.CorpusReader, rng: np.random.Generator, seq_len: int,
               batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """从编号流里取一小批窗口，搬去目标设备；返回（题面, 答案）两摞，答案就是题面左移一位。

    白话：从几本账里随手抽几本、各挑一处开头，往下数出一段来；把这段的前一半当题面、后一半
    当答案——因为要练的就是"看着前面猜下一个"。抽的时候保证起点不会顶到本子末尾，免得数出
    半截还要跨本去接（一跨本，这一段就不再是连着的那句话了）。
    """
    xs, ys = [], []
    for _ in range(batch_size):
        x, y = reader.window(rng, seq_len)
        xs.append(x)
        ys.append(y)
    def _to_device(parts):
        return torch.from_numpy(np.stack(parts)).to(device)

    return _to_device(xs), _to_device(ys)


# ══ B1a-2 训练循环主体：随机初始化 下一词 CE 定时写曲线，发现坏数字就地停 ══════════════

def train(model: Decoder, reader: corpus_mod.CorpusReader, run: RunContext, cfg: dict[str, Any],
          *, device: torch.device, start_step: int = 0, state: dict[str, Any] | None = None
          ) -> dict[str, Any]:
    """跑完（或续完）训练循环，返回这次观察到的摘要；曲线只经 run.log_metrics 追加。

    每一步做的事：取一小批窗口 让模型按位置猜下一个 算猜错的平均代价 把这份代价往前传，
    攒够配置的累积步数才真更新一次；每 log_interval 步把窗口均值、学习率、吞吐写进流水账；
    每 save_interval 步留一份存档；任何时候平均值不是有限数就当场抛 DivergedError 停下，
    并把"停在哪一步、前一次正常值是多少"如实写进 notes。

    start_step / state 是给续训用的：从存档读回的数字直接接上，不从头再算一遍。
    返回的摘要里 first_loss / last_loss 是验收"曲线确实在往下走"的两个端点。

    白话：像教一个人认字：每次给他看一小段话的前半，让他说下一个字，说错了记下错多少。
    错得厉害就先小步改、别改太猛（开头步子小，慢慢加大，后段再一点点收）。练一阵就把平均
    错量抄到账上，练到规定点就把当前状态本抄一份存起来，明天能从这一页接着练。要是某一轮
    错量变成了"没法看的数"（既不是大也不是小，是算不出来），立刻收手并把话说清楚：练到第
    几步、上一次正常是多少——绝不能把一堆算不出来的数继续往账上抄。
    """
    train_cfg = cfg["train"]
    seq_len = int(cfg["data"]["seq_len"])
    batch_size = int(cfg["data"]["batch_size"])
    accum = max(1, int(train_cfg.get("grad_accum", 1)))
    log_every = max(1, int(train_cfg.get("log_interval", 10)))
    save_every = int(train_cfg.get("save_interval", 0))
    total = int(train_cfg["max_steps"])
    clip = float(train_cfg.get("grad_clip", 1.0))

    model.to(device).train()
    optim = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]),
                              betas=tuple(train_cfg.get("betas", (0.9, 0.95))),
                              weight_decay=float(train_cfg.get("weight_decay", 0.1)))
    rng = np.random.default_rng(int(cfg["data"].get("seed", 0)))
    if state:
        optim.load_state_dict(state["optimizer"])
        saved_seed = state.get("numpy_rng_seed")          # 存档里记的是抽下一处窗口的那颗种
        rng = np.random.default_rng(int(saved_seed) if saved_seed is not None
                                    else int(cfg["data"].get("seed", 0)))
        torch_state = state.get("torch_rng_state")
        if torch_state is not None:                       # 采样也要接着上次的随机流
            torch.random.set_rng_state(torch_state)
        tokens_seen = int(state.get("tokens_seen", 0))
    else:
        tokens_seen = int(start_step) * batch_size * seq_len

    if start_step == 0:
        _log_baseline(model, reader, run, cfg, device, seq_len, batch_size, rng)

    step = start_step
    window: list[float] = []
    window_start = time.time()
    while step < total:
        step += 1
        lr = lr_at(step, train_cfg)
        for group in optim.param_groups:
            group["lr"] = lr
        optim.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro in range(accum):
            xs, ys = next_batch(reader, rng, seq_len, batch_size, device)
            logits = model.lm_head(model(xs))
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ys.reshape(-1))
            (loss / accum).backward()
            loss_sum += float(loss.detach())
            if not math.isfinite(loss_sum / (micro + 1)):
                _abort_diverged(run, step, window, loss_sum, tokens_seen)
        if clip > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
        else:
            grad_norm = 0.0
        optim.step()
        tokens_seen += batch_size * seq_len * accum
        mean = loss_sum / accum
        window.append(mean)

        if step % log_every == 0 or step == total:
            elapsed = time.time() - window_start
            rows = len(window) or 1
            run.log_metrics(step, loss=round(sum(window) / rows, 4), lr=round(lr, 8),
                            tok_per_s=int(batch_size * seq_len * accum * rows / max(1e-6, elapsed)),
                            grad_norm=round(grad_norm, 4), tokens_seen=tokens_seen,
                            sec=round(elapsed, 2))
            if not QUIET.steps:
                print(f"[s1] step={step:5d}/{total} loss={sum(window) / rows:.4f} lr={lr:.2e} "
                      f"tok/s={batch_size * seq_len * accum * rows / max(1e-6, elapsed):,.0f}")
            window, window_start = [], time.time()
        if save_every and (step % save_every == 0 or step == total):
            save_ckpt(model, optim, step, fingerprint(cfg), tokens_seen, run.path / CKPT_DIRNAME,
                      int(train_cfg.get("keep_last", 3)), rng)

    first = float(_metric_series(run.metrics_file, "loss")[0])
    last = float(_metric_series(run.metrics_file, "loss")[-1])
    return {"steps": step, "tokens_seen": tokens_seen, "first_loss": first, "last_loss": last,
            "improved": last < first, "device": str(device)}


def _log_baseline(model: Decoder, reader: corpus_mod.CorpusReader, run: RunContext,
                  cfg: dict[str, Any], device: torch.device, seq_len: int, batch_size: int,
                  rng: np.random.Generator) -> None:
    """把"还没学过"的起点代价记成 step=0 的一行，首末对比才有真正的首（否则首是第 20 步）。"""
    with torch.no_grad():
        xs, ys = next_batch(reader, rng, seq_len, batch_size, device)
        logits = model.lm_head(model(xs))
        loss = float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), ys.reshape(-1)))
    run.log_metrics(0, loss=round(loss, 4), lr=0.0, tok_per_s=0,
                    tokens_seen=0, note="step0 baseline（随机初始化，未学过）")


def _abort_diverged(run: RunContext, step: int, window: list[float], loss_sum: float,
                   tokens_seen: int) -> None:
    """发现坏数字：把现场如实写进 notes 再抛错，绝不带病继续往账上抄。"""
    last_ok = f"{window[-1]:.4f}" if window else "无（本窗口第一个点就坏了）"
    _note(run, f"step={step} 平均值非有限（loss_sum={loss_sum}），上一次正常值 {last_ok}；"
               f"已练到 tokens_seen={tokens_seen}，训练按设计中断。")
    raise DivergedError(f"step={step} 出现非有限的代价读数，训练中断（详见 notes.md）")


def _metric_series(path: Path, name: str) -> list[float]:
    """从流水账里按写入顺序取出某个指标的全部数值（跳过残行，与 runs 侧读法一致）。"""
    out: list[float] = []
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and name in row and isinstance(row[name], (int, float)):
            out.append(float(row[name]))
    return out


def _note(run: RunContext, text: str) -> None:
    """往 notes.md 追加一行"观察："（曲线之外的定性交代，失败实验也必须留痕）。"""
    with run.notes_file.open("a", encoding="utf-8") as handle:
        handle.write(f"- 观察：{text}\n")


# ══ B1b 存档分片 + 续训：按 step 号留多份、轮转限量，续之前先验指纹 ════════════════════

def save_ckpt(model: Decoder, optim: torch.optim.Optimizer, step: int, cfg_fp: str,
              tokens_seen: int, ckpt_root: Path, keep_last: int = 3,
              rng: np.random.Generator | None = None) -> Path:
    """把当前状态按步号存成一份独立存档，并更新 latest 指针；只保留最近 keep_last 份。

    一份存档里有三样：模型目录（config.json + weights.safetensors，p1-09 直接按模型目录消费）、
    优化器与进度状态（train_state.pt：步号、累计读入量、随机状态、指纹、时间）、一份 meta.json
    说明这份是谁、由哪条配置指纹生成。latest.json 永远指向步号最大的那一份。

    白话：练到整点就把当前状态本抄一份，封皮上写清抄于第几页、按的哪份设置单。抄完只留最近
    的几本，旧的合上收走，免得桌子被自己挤满。明天来上班的人翻开那行"最近一本"就知道从哪儿
    接着抄；他要是发现设置单换了（封皮上的代号不对），就得先问清楚，不能闷头接上继续写。
    """
    target = ckpt_root / f"step-{step:06d}"
    target.mkdir(parents=True, exist_ok=True)
    model.save(target)
    torch.save({"step": int(step), "optimizer": optim.state_dict(), "tokens_seen": int(tokens_seen),
                "config_fingerprint": cfg_fp, "torch_rng_state": torch.random.get_rng_state(),
                "numpy_rng_seed": int(rng.integers(0, 2**31 - 1)) if rng is not None else None,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
               target / STATE_FILE)
    meta = {"step": int(step), "config_fingerprint": cfg_fp, "tokens_seen": int(tokens_seen),
            "files": [MODEL_DIRNAME + "/*", STATE_FILE, META_FILE], "keep_last": int(keep_last)}
    (target / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ckpt_root / LATEST_FILE).write_text(
        json.dumps({"step": int(step), "dir": target.name, "config_fingerprint": cfg_fp},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _rotate(ckpt_root, keep_last)
    return target


def load_ckpt(ckpt_dir: str | Path, cfg_fp: str | None = None) -> dict[str, Any]:
    """读回一份存档；给了期望指纹就当场比对，不一致直接拒（宁可另开新 run，也不接假曲线）。

    白话：把状态本翻开之前，先看封皮上的代号跟手上这张设置单算出来的一不一样。一样才继续；
    不一样说明中间动过模型宽窄或练法，硬接会让账上出现"一条曲线两个模型"的事后极难查的错，
    所以这里直接拒绝，并说清楚存档里记的是哪个代号。
    """
    src = Path(ckpt_dir)
    meta_file = src / META_FILE
    if not meta_file.exists():
        raise ResumeError(f"存档 {src} 缺 {META_FILE}，不认这份存档")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if cfg_fp is not None and meta.get("config_fingerprint") != cfg_fp:
        raise ResumeError(f"配置指纹不一致：存档 {meta.get('config_fingerprint')!r} vs 本次 {cfg_fp!r}；"
                          f"改过 model/data/train 三段就不能续，请另开新 run")
    state = torch.load(src / STATE_FILE, map_location="cpu", weights_only=True)
    state["ckpt_dir"] = src
    state["meta"] = meta
    return state


def find_latest_ckpt(run_dir: str | Path) -> Path | None:
    """找一份 run 目录里"最近的一份存档"：先看 latest.json，再退回按步号挑最大。

    白话：想知道该从哪一本状态本接着练，先翻桌角那张"最近一本写在这儿"的条子；条子丢了或
    者指的是一个已经不存在的文件夹，就把这一摞按封皮上的页码排一遍，挑页码最大的那本。两条
    路都找不着才说没有——不猜、不凑合用一本旧的。
    """
    root = Path(run_dir) / CKPT_DIRNAME
    pointer = root / LATEST_FILE
    if pointer.exists():
        name = json.loads(pointer.read_text(encoding="utf-8")).get("dir")
        if name and (root / name).exists():
            return root / name
    candidates = sorted(p for p in root.glob("step-*") if (p / META_FILE).exists()) if root.exists() else []
    return candidates[-1] if candidates else None


def _rotate(ckpt_root: Path, keep_last: int) -> None:
    """轮转：只留最近 keep_last 份存档（keep_last<=0 表示全留），别让磁盘被自己挤满。"""
    if keep_last <= 0:
        return
    dirs = sorted(p for p in ckpt_root.glob("step-*") if p.is_dir())
    for stale in dirs[:-keep_last]:
        for file in sorted(stale.rglob("*"), reverse=True):
            if file.is_file():
                file.unlink()
        for leftover in sorted(stale.rglob("*"), reverse=True):
            if leftover.is_dir():
                leftover.rmdir()
        stale.rmdir()


def open_resume(run_dir: str | Path, cfg: dict[str, Any]) -> tuple[RunContext, Path, dict[str, Any]]:
    """续训入口：认出原 run 目录、找回最近存档、验完指纹后把三者交出去。

    白话：先确认这个文件夹确实是本记录本里的一页（四件套齐在），再翻到"最近一本状态"那页，
    把封皮代号对一遍。都对了才交回原账本句柄、状态本和存档位置——第几页往后写由账本句柄说
    了算，同一步的同一个数写第二遍会被账本自己挡下来，所以续训天然不会出现重复页码。
    """
    path = Path(run_dir)
    if not (path / "config.yaml").exists() or not (path / "metrics.jsonl").exists():
        raise ResumeError(f"{path} 不像一个 run 目录（缺 config.yaml 或 metrics.jsonl）")
    ckpt = find_latest_ckpt(path)
    if ckpt is None:
        raise ResumeError(f"{path} 里没有可用存档（{CKPT_DIRNAME}/step-* 为空），无法续训")
    state = load_ckpt(ckpt, fingerprint(cfg))
    run = RunContext(path.name, path)
    return run, ckpt, state


# ══ C1 采样冒烟：按温度往外蹦短句，原文与判定一起进 notes（可读性是过程信号不是 gate）══

def sample_texts(model: Decoder, tok: bpe.Tokenizer, cfg: dict[str, Any], count: int,
                 device: torch.device) -> list[dict[str, Any]]:
    """从配置给的每个起头往下续写，采出 count 条短句；返回 [{prompt, text, tokens, stop}]。

    每一步只看"当前这一段"算出各候选的分，除以温度再截前 top_k 个，按概率抽一个接着写；
    写到 max_new_tokens 个或碰上整篇结束位就收。全程不参考任何外部现成模型，读的就是这次
    练出来的那份数字。

    白话：先给它一句话的开头，让它自己往下蹦字：每蹦一个之前，把所有可能的字各打一个分，
    分数按一个"敢不敢赌"的旋钮抹平一点（旋钮越大越敢乱来），只在前几名里按名次的高低随机
    挑一个。蹦到规定个数，或者自己蹦出"这说完了"的记号就停。蹦出来的每一句都连同原文一起
    抄进笔记，好叫后来看的人能自己判断这像不像话。
    """
    sample_cfg = cfg.get("sample") or {}
    temperature = float(sample_cfg.get("temperature", 0.8))
    top_k = int(sample_cfg.get("top_k", 40))
    max_new = int(sample_cfg.get("max_new_tokens", 40))
    prompts = list(sample_cfg.get("prompts") or [""])
    eos_id = int(tok.token_to_id(sample_cfg.get("stop_token", bpe.ENDOFTEXT_TOKEN)))
    model.to(device).eval()
    records: list[dict[str, Any]] = []
    for index in range(count):
        prompt = prompts[index % len(prompts)]
        ids = tok.encode(prompt, add_special_tokens=False).ids if prompt else []
        ids = ids[-int(cfg["model"]["ctx"]) - max_new:]
        generated: list[int] = []
        stop = "max_new_tokens"
        while len(generated) < max_new:
            window = (ids + generated)[-int(cfg["model"]["ctx"]):]
            feed = torch.tensor([window], dtype=torch.long, device=device)
            with torch.no_grad():
                scores = model.lm_head(model(feed))[0, -1].to(torch.float32)
            nxt = _pick_candidate(scores, temperature, top_k)
            if nxt == eos_id:
                stop = "stop_token"
                break
            generated.append(int(nxt))
        text = tok.decode(ids[-8:] + generated, skip_special_tokens=True)
        records.append({"prompt": prompt, "text": text, "tokens": len(generated), "stop": stop,
                        "temperature": temperature, "top_k": top_k})
    return records


def _pick_candidate(scores: torch.Tensor, temperature: float, top_k: int) -> int:
    """温度抹平 截前 top_k 归一成概率 抽一个：返回被抽中的候选号。"""
    scaled = scores / max(1e-6, temperature)
    k = max(1, min(top_k, int(scaled.numel())))
    values, indices = torch.topk(scaled, k)
    probs = torch.softmax(values, dim=-1)
    draw = torch.multinomial(probs.cpu(), num_samples=1)
    return int(indices[draw[0]])


def judge_readability(records: list[dict[str, Any]]) -> dict[str, Any]:
    """机械初筛：每条标"有没有连续汉字 / 有没有 3 字母以上的连续英文词"，汇总成一份可核对的账。

    白话：不假装自己能读懂，只按两条最笨的形状标准各看一眼：这句话里是不是连着出了两个
    汉字，或者是不是连着拼出了一个像样的英文词。都没出现的句子，多半只是一串碰巧的符号。
    这份初筛只是给人工判定省事的辅助线索，最终"像不像话"由人看完原文说了算。
    """
    flags: list[dict[str, Any]] = []
    for record in records:
        text = record["text"]
        run = max((len(m.group(0)) for m in re.finditer(r"[\u4e00-\u9fff]{2,}", text)), default=0)
        words = [w.group(0) for w in LATIN_WORD.finditer(text)]
        flags.append({"cjk_run": run, "latin_words": words[:4],
                      "ok": run >= MIN_CJK_RUN or any(len(w) >= MIN_LATIN_WORD for w in words)})
    readable = sum(1 for f in flags if f["ok"])
    return {"per_sample": flags, "readable": readable, "total": len(flags),
            "criteria": f"连续汉字>={MIN_CJK_RUN} 或 连续英文词>={MIN_LATIN_WORD} 字母"}


def write_samples_to_notes(run: RunContext, records: list[dict[str, Any]], verdict: dict[str, Any],
                           summary: dict[str, Any]) -> None:
    """把采样原文逐条 + 机械初筛结果写进 notes.md（spec「可读性抽检」要求的证据形态）。

    白话：把蹦出来的每一句原文一字不改地抄进笔记本，前头标上它是从哪句话起头的、蹦了几个、
    怎么停的，后头跟一句形状初筛的结论和曲线两端。抄原文是为了让后面来的人能自己动手判断，
    而不是只信我们给的分数。
    """
    lines = ["", "## 采样冒烟（温度采样，逐条原文）", ""]
    for i, (record, flag) in enumerate(zip(records, verdict["per_sample"]), start=1):
        head = f"（起头 {record['prompt']!r}）" if record["prompt"] else "（无起头，自由续写）"
        lines.append(f"{i}. {head} {record['tokens']} 个片段，止于 {record['stop']}："
                     f"{'有' if flag['ok'] else '无'}像话片段（汉字连串 {flag['cjk_run']}、"
                     f"英文词 {flag['latin_words']}）")
        lines.append(f"   > {record['text']}")
    readable_line = (f"初筛：{verdict['readable']}/{verdict['total']} 条形状上像话"
                     f"（判据：{verdict['criteria']}）；学习曲线 "
                     f"loss {summary['first_loss']:.4f} -> {summary['last_loss']:.4f}。")
    review_line = "人工判定：待复核（教师版按 design 口径，可读短句是过程信号不是 gate，不达标如实记）。"
    lines += ["", readable_line, review_line, ""]
    with run.notes_file.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _run_record(cfg: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """拼 run 的参数单：配置原文 + 指纹 + 语料统计 + 设备，全量留档（报告里每个数字能指回来）。"""
    record = {"stage": "s1-pretrain", "config_path": cfg.get("_config_path"),
              "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
              "fingerprint": fingerprint(cfg)}
    record.update(extra)
    return record


# ══ CLI：python learning/s1_pretrain_gpt.py --config smoke_tiny [--sample 8] [--resume ...] ══

def build_parser() -> argparse.ArgumentParser:
    """搭出参数表：--config/--resume/--sample/--corpus/--device/--max-steps/--seed/--quiet。

    白话：把能敲的选项列清楚。用哪份设置单、接哪个旧 run 继续、蹦几条句子看看、语料换到哪
    个文件夹、在哪台机器上跑、最多练到第几步（测试用来掐表）、随机起点换哪个数。敲 --help
    就能看见这份说明，不用去翻代码。
    """
    parser = argparse.ArgumentParser(
        prog="s1_pretrain_gpt.py",
        description="S1 从零预训练（随机初始化，禁载第三方权重；GUIDE §7-0）：练曲线、留存档、"
                    "可续训、末尾采样冒烟。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="smoke_tiny", help="档位名（查 learning/configs/）或 yaml 路径")
    parser.add_argument("--resume", default=None, metavar="RUN_DIR",
                        help="续训：指向已有 run 目录（验配置指纹后从最近存档的步号接着练）")
    parser.add_argument("--sample", type=int, default=None,
                        help="采样条数；不给就用配置里的 sample.n（0=不采样）")
    parser.add_argument("--corpus", default=None, help="覆盖配置里的语料目录")
    parser.add_argument("--kernel-backend", default=None, choices=["off", "tilelang"],
                        help="前向/反向是否走 TileLang 可微内核链（不给=配置值，默认 off=纯 torch 基线）")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "mps", "cuda"],
                        help="覆盖配置里的设备")
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖练到第几步（测试与短时冒烟用）")
    parser.add_argument("--seed", type=int, default=None, help="覆盖模型与取数的随机起点")
    parser.add_argument("--quiet", action="store_true", help="不打印每 log_interval 步的进度")
    return parser


def _apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """把命令行覆盖项落到配置上（只动这几处，其余仍以 yaml 为唯一真源）。"""
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = int(args.max_steps)
    if args.device:
        cfg["train"]["device"] = args.device
    if args.seed is not None:
        cfg["model"]["seed"] = int(args.seed)
        cfg["data"]["seed"] = int(args.seed)
    if args.corpus:
        cfg["data"]["corpus_dir"] = args.corpus
    if args.kernel_backend and args.kernel_backend != "off":
        # 只在偏离默认时落盘：既有 run 的配置指纹（不含此键）不因新键而变，续训校验向后兼容
        cfg["train"]["kernel_backend"] = args.kernel_backend
    sample_cfg = cfg.setdefault("sample", {})
    sample_cfg["n"] = (int(args.sample) if args.sample is not None
                       else int(sample_cfg.get("n", 0)))
    return cfg


def main(argv: list[str] | None = None) -> int:
    """CLI 主体：认配置 认语料 建/续 run 练曲线 存最终模型 采样 写笔记 下结论。

    白话：一条命令把"照着设置单起一个空模型、翻语料、一路猜下一个、按点抄账、按点存状态本、"
    "练完自己蹦几句话、把看到的东西写进笔记并下一句结论"全走完。中途要是发现账本接不上
    （续训时配置改了）或者数字坏了，就如实把情况写进笔记再非零退出——失败也要留下交代。
    """
    args = build_parser().parse_args(argv)
    cfg = _apply_overrides(load_config(args.config), args)
    threads = int(cfg["train"].get("threads", 0))
    if threads > 0:
        torch.set_num_threads(threads)

    corpus_dir = str(REPO_ROOT / cfg["data"]["corpus_dir"]) if not Path(cfg["data"]["corpus_dir"]).is_absolute() \
        else cfg["data"]["corpus_dir"]
    reader = corpus_mod.open_corpus(corpus_dir)
    if reader.vocab != int(cfg["model"]["vocab"]):
        raise ValueError(f"语料表规模 {reader.vocab} 与模型 vocab {cfg['model']['vocab']} 不一致，"
                         f"练出来的是两个世界的东西")
    if min(reader.lengths) <= int(cfg["data"]["seq_len"]):
        raise ValueError(f"语料最短片 {min(reader.lengths)} 不超过窗口 {cfg['data']['seq_len']}，"
                         f"装配时把 shard_tokens 调大")
    device = resolve_device(cfg["train"].get("device", "auto"))
    if args.resume:
        run, ckpt, state = open_resume(args.resume, cfg)
        model = Decoder.load(ckpt)
        start_step = int(state["step"])
        print(f"[s1] resume run={run.run_id} from step={start_step} (ckpt={ckpt.name})")
    else:
        run = new_run(cfg["run"]["name"], _run_record(cfg, {"corpus": reader.stats(),
                                                            "device": str(device), "resumed_from": None}))
        model = build_model(cfg)
        start_step, state = 0, None
        torch.manual_seed(int(cfg["model"]["seed"]))

    QUIET.steps = bool(args.quiet)
    try:
        summary = train(model, reader, run, cfg, device=device, start_step=start_step, state=state)
        model.save(run.path / MODEL_DIRNAME)
        records = sample_texts(model, bpe.load(run_tok_dir(cfg, reader)), cfg,
                               int(cfg["sample"]["n"]), device) if int(cfg["sample"]["n"]) > 0 else []
        verdict = judge_readability(records) if records else {"per_sample": [], "readable": 0, "total": 0,
                                                              "criteria": "未采样"}
        if records:
            write_samples_to_notes(run, records, verdict, summary)
        conclusion = (f"随机初始化起步练到 step={summary['steps']}（读入 {summary['tokens_seen']:,} 个片段，"
                      f"设备 {summary['device']}）：loss {summary['first_loss']:.4f} -> "
                      f"{summary['last_loss']:.4f}，{'确实往下走' if summary['improved'] else '没往下走（如实记）'}；"
                      f"采样 {verdict['total']} 条，形状初筛像话 {verdict['readable']} 条"
                      f"（判据 {verdict['criteria']}）。最终模型在 {MODEL_DIRNAME}/，"
                      f"续训起点在 {CKPT_DIRNAME}/（latest.json 指向），p1-09 可直接按模型目录消费。")
        run.conclude(conclusion)
        run.finish()
        print(f"[s1] run            : {run.path}")
        print(f"[s1] loss           : {summary['first_loss']:.4f} -> {summary['last_loss']:.4f} "
              f"(improved={summary['improved']})")
        print(f"[s1] steps/tokens   : {summary['steps']} / {summary['tokens_seen']:,}")
        print(f"[s1] samples        : {verdict['readable']}/{verdict['total']} 形状像话，原文见 notes.md")
        print(f"[s1] conclusion     : {conclusion}")
        return 0
    except DivergedError as exc:
        run.conclude(f"数字坏了，本次如实记为失败：{exc}；已写出的曲线保留在 metrics.jsonl，"
                     f"存档保留在 {CKPT_DIRNAME}/，不拿坏曲线冒充进步。")
        run.finish()
        print(f"[s1] DIVERGED: {exc}", file=sys.stderr)
        return 2


def run_tok_dir(cfg: dict[str, Any], reader: corpus_mod.CorpusReader) -> str:
    """采样要用的对照表目录：优先配置显式给的，否则回用语料索引里记的那份（保持同源）。

    白话：练的时候用的是哪张对照表，往外蹦句子时就必须还是那张。配置里写了就用配置里的，
    没写就照语料本上记的那一条去找，免得"练用一张表、看结果用另一张表"对不上号。
    """
    explicit = (cfg.get("sample") or {}).get("tokenizer_dir") or cfg.get("data", {}).get("tokenizer_dir")
    return str(explicit) if explicit else str(reader.index["tokenizer_dir"])


if __name__ == "__main__":
    sys.exit(main())
