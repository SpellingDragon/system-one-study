"""p1-08 工作项 B/C 训练侧验收单测（`-k` 关键字即 tasks.md 条目）。

  B1a `-k loop`              训练循环主体：随机初始化起步、下一词代价定时成账、坏数字当场收手
  B1b `-k ckpt_resume_impl`  存档分片落盘、轮转、latest 指针、指纹守卫、存档可被载回
  B3  `-k resume`            中断→续训：曲线不出现重复页码；配置改动/没存档都拒续
  C2  `-k e2e`（`-m slow`）  smoke_tiny 端到端：退出码 0、首末 loss 下降、8 条采样入 notes、
                             checkpoint 可载，CPU 用时 < 10 分钟

离线口径：前三组一律用 learning/configs/corpus.yaml 的 `unit` 档（配置里内联的十几篇小文档），
秒级、零网络；产物全部写进 tmp（DMLAYA_RUNS_DIR），不碰仓库的 runs/。
e2e 用 A2 的真产物 runs/corpus_tiny + learning/configs/smoke_tiny.yaml：默认也写 tmp（CI 可反复跑），
只有显式 DMLAYA_KEEP_ARTIFACTS=1 时才落进真实 runs/（教师版留真实证据用）。
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from sys1.data import pretrain_corpus as pc
from sys1.model import Decoder
from sys1.runs import MetricsConflictError, new_run

REPO_ROOT = Path(__file__).resolve().parents[1]
S1_PATH = REPO_ROOT / "learning" / "s1_pretrain_gpt.py"
CORPUS_TINY = REPO_ROOT / "runs" / "corpus_tiny"
RUNS_ROOT = REPO_ROOT / "runs"
E2E_BUDGET_SEC = 600                # C2 的 CPU 时长硬上限
VOCAB = 16000                       # 与自训产物一致（unit 语料也是这张表装出来的）


def _load_s1():
    """按文件路径加载训练脚本（与 test_s2_sft 同法，避开命名空间包的导入歧义）。"""
    spec = importlib.util.spec_from_file_location("s1_pretrain_gpt", S1_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def s1():
    return _load_s1()


@pytest.fixture
def unit_corpus(tmp_path):
    """离线小语料：corpus.yaml 的 unit 档，装到 tmp 下再打开（分片 64 个编号/片）。"""
    src = tmp_path / "corpus"
    pc.build(pc.load_spec("unit"), src, verbose=False)
    return pc.open_corpus(src)


def _mini_cfg(corpus_dir: Path, *, name: str = "s1-unit", max_steps: int = 30,
              log_interval: int = 5, save_interval: int = 10, keep_last: int = 2) -> dict:
    """极小训练配置：d=32/L=1/heads=2/ctx=64，窗口 32、每次 8 段，够把逻辑走全而不耗时。"""
    return {
        "run": {"name": name},
        "model": {"d": 32, "L": 1, "heads": 2, "ctx": 64, "vocab": VOCAB, "rope_theta": 10000.0,
                  "seed": 0, "ffn_mult": 4, "norm_eps": 1e-5, "pad_token_id": 0},
        "data": {"corpus_dir": str(corpus_dir), "seq_len": 32, "batch_size": 8, "seed": 0},
        "train": {"device": "cpu", "threads": 1, "max_steps": max_steps, "lr": 3e-4,
                  "warmup_steps": 5, "min_lr_frac": 0.1, "betas": [0.9, 0.95],
                  "weight_decay": 0.1, "grad_clip": 1.0, "grad_accum": 1,
                  "log_interval": log_interval, "save_interval": save_interval,
                  "keep_last": keep_last},
        "sample": {"n": 0, "temperature": 0.8, "top_k": 40, "max_new_tokens": 16,
                   "prompts": ["今天天气", "The capital of France is"]},
    }


def _rows(metrics_file: Path) -> list[dict]:
    """把流水账读成字典列表（按写入顺序）。"""
    return [json.loads(line) for line in
            metrics_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def _open_run(tmp_path, monkeypatch, name: str = "s1-unit"):
    """把记录本根目录挪进 tmp，再开一个 run（训练侧测试不碰仓库 runs/）。"""
    monkeypatch.setenv("DMLAYA_RUNS_DIR", str(tmp_path / "runs"))
    return new_run(name, {"purpose": "unit"})


# ── B1a `-k loop`：训练循环主体 ───────────────────────────────────────────────────────

def test_loop_starts_from_random_init_not_third_party(s1):
    """红线自查（GUIDE §7-0）：唯一的建模型入口只看 seed，没有任何"载外部权重"的调用。"""
    import inspect

    src = inspect.getsource(s1.build_model)
    forbidden = ("load_state_dict", "load_file", "safetensors", "from_pretrained",
                 "torch.load", "http")
    for word in forbidden:
        assert word not in src, f"build_model 里出现了 {word!r}——起步必须只有摇出来的随机数字"
    cfg = _mini_cfg(Path("."))
    a = s1.build_model(cfg)
    b = s1.build_model(cfg)
    same = all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert same, "同 seed 两次建出来的初始数字必须逐位相同，否则复现无从谈起"
    finite = all(torch.isfinite(t).all() for t in a.state_dict().values())
    assert finite, "随机初始化的数字得是有限数"


def test_loop_writes_curve_and_loss_declines(s1, unit_corpus, tmp_path, monkeypatch):
    """spec「下一词预训练循环」：定时写 loss/lr/tok_per_s，且这条曲线真在往下走。"""
    cfg = _mini_cfg(tmp_path / "corpus")
    run = _open_run(tmp_path, monkeypatch)
    model = s1.build_model(cfg)
    summary = s1.train(model, unit_corpus, run, cfg, device=torch.device("cpu"))

    rows = _rows(run.metrics_file)
    steps = [r["step"] for r in rows]
    assert steps[0] == 0, 'step=0 要有一行「还没学过」的起点，首末对比才有真正的头'   
    assert steps == [0, 5, 10, 15, 20, 25, 30], f"按 log_interval 定时成账，收到 {steps}"
    for row in rows[1:]:
        assert {"loss", "lr", "tok_per_s", "grad_norm", "tokens_seen"} <= set(row)
    assert summary["steps"] == 30
    assert summary["tokens_seen"] == 30 * 8 * 32
    assert summary["first_loss"] > summary["last_loss"], "30 步随机数据也够看出往下走"
    assert summary["improved"] is True
    assert rows[-1]["loss"] == pytest.approx(summary["last_loss"])


def test_loop_metrics_is_the_single_source_with_dedup(s1, unit_corpus, tmp_path, monkeypatch):
    """曲线唯一真源是 metrics.jsonl，且同一步同一指标写第二遍会被记录本自己挡下。"""
    cfg = _mini_cfg(tmp_path / "corpus")
    run = _open_run(tmp_path, monkeypatch)
    s1.train(s1.build_model(cfg), unit_corpus, run, cfg, device=torch.device("cpu"))
    with pytest.raises(MetricsConflictError):
        run.log_metrics(5, loss=0.0)          # 第 5 步已经写过 loss，再写必须被拒
    steps = [r["step"] for r in _rows(run.metrics_file)]
    assert len(steps) == len(set(steps)) or all(
        len([s for s in steps if s == 5]) == 1 for s in steps), "被拒的那次不该落进账里"


def test_loop_nan_interrupts_and_records_the_scene(s1, unit_corpus, tmp_path, monkeypatch):
    """坏数字（NaN）检测：当场中断、把现场写进 notes，绝不带病继续往账上抄。"""
    cfg = _mini_cfg(tmp_path / "corpus", max_steps=30, log_interval=5)
    run = _open_run(tmp_path, monkeypatch)
    real = s1.F.cross_entropy
    calls = {"n": 0}

    def poisoned(logits, target, *args, **kwargs):
        calls["n"] += 1
        out = real(logits, target, *args, **kwargs)
        return out * float("nan") if calls["n"] == 3 else out

    monkeypatch.setattr(s1.F, "cross_entropy", poisoned)
    with pytest.raises(s1.DivergedError):
        s1.train(s1.build_model(cfg), unit_corpus, run, cfg, device=torch.device("cpu"))

    rows = _rows(run.metrics_file)
    assert [r["step"] for r in rows] == [0], "起点行已写，坏掉那一步之后一条都不许再写"
    notes = run.notes_file.read_text(encoding="utf-8")
    assert "观察：" in notes and "step=2" in notes, "停在哪一步、上一次正常值多少要如实交代"


def test_loop_lr_shape_warms_up_then_decays_to_floor(s1):
    """warmup 线性爬升 + cosine 退火到 min_lr_frac（不是踩到零）。"""
    train_cfg = {"lr": 1e-3, "warmup_steps": 10, "max_steps": 100, "min_lr_frac": 0.1}
    ramp = [s1.lr_at(step, train_cfg) for step in range(1, 11)]
    assert ramp == sorted(ramp) and abs(ramp[-1] - 1e-3) < 1e-12, "前 10 步只许往上加"
    tail = [s1.lr_at(step, train_cfg) for step in range(10, 101)]
    assert tail == sorted(tail, reverse=True), "到顶之后按弧线收小"
    assert tail[-1] == pytest.approx(1e-4), "收尾留 10% 余量，不是一路踩到零"


# ── B1b `-k ckpt_resume_impl`：存档分片落盘 + 守卫 ────────────────────────────────────

def test_ckpt_resume_impl_writes_a_loadable_bundle(s1, unit_corpus, tmp_path):
    """一份存档=模型目录三件 + 状态本 + meta.json，且 latest.json 指向步号最大的那份。"""
    cfg = _mini_cfg(tmp_path / "corpus")
    model = s1.build_model(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    fp = s1.fingerprint(cfg)
    rng = np.random.default_rng(1)
    root = tmp_path / "ckpt"
    first = s1.save_ckpt(model, optim, 10, fp, 1000, root, keep_last=3, rng=rng)
    second = s1.save_ckpt(model, optim, 20, fp, 2000, root, keep_last=3, rng=rng)

    assert first.name == "step-000010" and second.name == "step-000020"
    for name in ("config.json", "weights.safetensors", s1.STATE_FILE, s1.META_FILE):
        assert (second / name).exists(), f"存档缺 {name}"
    pointer = json.loads((root / s1.LATEST_FILE).read_text(encoding="utf-8"))
    assert pointer["dir"] == "step-000020" and pointer["config_fingerprint"] == fp

    state = s1.load_ckpt(second, fp)
    assert state["step"] == 20 and state["tokens_seen"] == 2000
    assert state["numpy_rng_seed"] is not None and state["torch_rng_state"] is not None
    loaded = Decoder.load(second)                       # spec：checkpoint 可被 load
    out = loaded.lm_head(loaded(torch.randint(2, VOCAB, (2, 32))))
    assert torch.isfinite(out).all(), "载回的存档得能正向跑出有限数"


def test_ckpt_resume_impl_rotation_keeps_only_last_n(s1, unit_corpus, tmp_path):
    """keep_last 轮转：只留最近几份，latest 仍指向最新，被丢的那份目录真消失。"""
    model = s1.build_model(_mini_cfg(tmp_path / "corpus"))
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    root = tmp_path / "ckpt"
    for step in (10, 20, 30, 40):
        s1.save_ckpt(model, optim, step, "fp", step * 100, root, keep_last=2)
    left = sorted(p.name for p in root.glob("step-*"))
    assert left == ["step-000030", "step-000040"], f"该只留两份，收到 {left}"
    assert json.loads((root / s1.LATEST_FILE).read_text(encoding="utf-8"))["step"] == 40


def test_ckpt_resume_impl_find_latest_falls_back_without_pointer(s1, unit_corpus, tmp_path):
    """latest.json 丢了/指空也要能按步号挑到最大那份，续训不因一条指针而瘫痪。"""
    model = s1.build_model(_mini_cfg(tmp_path / "corpus"))
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    root = tmp_path / "ckpt"
    s1.save_ckpt(model, optim, 10, "fp", 100, root, keep_last=5)
    s1.save_ckpt(model, optim, 20, "fp", 200, root, keep_last=5)
    (root / s1.LATEST_FILE).unlink()
    assert s1.find_latest_ckpt(tmp_path).name == "step-000020"


def test_ckpt_resume_impl_fingerprint_guard_rejects_mismatch(s1, unit_corpus, tmp_path):
    """改过尺寸/学习率就必须拒续（否则会接出一条"两截不是一个模型"的假曲线）。"""
    cfg = _mini_cfg(tmp_path / "corpus")
    model = s1.build_model(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    target = s1.save_ckpt(model, optim, 10, s1.fingerprint(cfg), 100, tmp_path / "ckpt", 3)

    wider = _mini_cfg(tmp_path / "corpus")
    wider["model"]["d"] = 64
    with pytest.raises(s1.ResumeError, match="指纹不一致"):
        s1.load_ckpt(target, s1.fingerprint(wider))

    longer = _mini_cfg(tmp_path / "corpus", max_steps=999)
    assert s1.fingerprint(longer) == s1.fingerprint(cfg), "只调收手步数属于正当续训，不该被拒"


def test_ckpt_resume_impl_rejects_missing_meta(s1, tmp_path):
    """半截存档（缺 meta.json）不认：宁可报错，也不拿一份说不清来历的状态继续写。"""
    with pytest.raises(s1.ResumeError, match="缺"):
        s1.load_ckpt(tmp_path / "not_a_ckpt")


# ── B3 `-k resume`：中断→续训，曲线无重复页码 ─────────────────────────────────────────

def test_resume_after_interrupt_continues_without_duplicate_steps(s1, unit_corpus, tmp_path,
                                                                  monkeypatch):
    """spec「断点续训」：从断点 step 继续，metrics 里不出现重复 step。"""
    monkeypatch.setenv("DMLAYA_RUNS_DIR", str(tmp_path / "runs"))
    cfg = _mini_cfg(tmp_path / "corpus", max_steps=20, log_interval=5, save_interval=10)
    run = new_run("s1-resume-unit", {"purpose": "resume"})
    model = s1.build_model(cfg)
    s1.train(model, unit_corpus, run, cfg, device=torch.device("cpu"))
    after_first = _rows(run.metrics_file)
    assert [r["step"] for r in after_first] == [0, 5, 10, 15, 20]
    assert (run.path / "ckpt" / "step-000020").exists(), "每 save_interval 步得留下面包屑"

    # 模拟"中断后重开"：新配置只把收手步数往后挪，其余一字不动，走 --resume 的同一条路
    longer = _mini_cfg(tmp_path / "corpus", max_steps=40, log_interval=5, save_interval=10,
                       name="s1-resume-unit")
    run2, ckpt, state = s1.open_resume(run.path, longer)
    assert ckpt.name == "step-000020" and state["step"] == 20
    resumed = s1.train(Decoder.load(ckpt), unit_corpus, run2, longer, device=torch.device("cpu"),
                       start_step=int(state["step"]), state=state)

    steps = [r["step"] for r in _rows(run2.metrics_file)]
    assert steps == sorted(steps), "续训后的页码必须单调"
    assert len(steps) == len(set(steps)), f"出现重复 step：{steps}"
    assert steps == [0, 5, 10, 15, 20, 25, 30, 35, 40], "接着第 20 页往后写，不回头重写也不跳页"
    assert resumed["steps"] == 40 and resumed["tokens_seen"] == 40 * 8 * 32
    assert (run2.path / "ckpt" / "step-000040").exists()


def test_resume_refuses_when_config_changed(s1, unit_corpus, tmp_path, monkeypatch):
    """改学习率/尺寸再 --resume → 拒续，并说清是哪一段变了。"""
    monkeypatch.setenv("DMLAYA_RUNS_DIR", str(tmp_path / "runs"))
    cfg = _mini_cfg(tmp_path / "corpus", max_steps=10, save_interval=10)
    run = new_run("s1-resume-refuse", {"purpose": "resume"})
    s1.train(s1.build_model(cfg), unit_corpus, run, cfg, device=torch.device("cpu"))

    tampered = _mini_cfg(tmp_path / "corpus", max_steps=20, save_interval=10)
    tampered["train"]["lr"] = 1e-2
    with pytest.raises(s1.ResumeError, match="指纹不一致"):
        s1.open_resume(run.path, tampered)


@pytest.mark.parametrize("breakage", ["not_a_run", "no_ckpt"])
def test_resume_refuses_when_precondition_missing(s1, tmp_path, monkeypatch, breakage):
    """目录不像 run / 有 run 但没存档：都在开练之前就说清楚，不静默从头再练一遍。"""
    monkeypatch.setenv("DMLAYA_RUNS_DIR", str(tmp_path / "runs"))
    if breakage == "not_a_run":
        with pytest.raises(s1.ResumeError, match="不像一个 run 目录"):
            s1.open_resume(tmp_path / "random_dir", _mini_cfg(tmp_path / "corpus"))
    else:
        run = new_run("s1-resume-nockpt", {"purpose": "resume"})
        (run.path / "ckpt").mkdir()                  # 有账本、有目录，但里面是空的
        with pytest.raises(s1.ResumeError, match="没有可用存档"):
            s1.open_resume(run.path, _mini_cfg(tmp_path / "corpus"))


def test_resume_state_restores_random_streams(s1, unit_corpus, tmp_path):
    """续训交回来的状态里带着两条随机流（抽窗口的、采样的），接得上才谈得上可复现。"""
    model = s1.build_model(_mini_cfg(tmp_path / "corpus"))
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    rng = np.random.default_rng(7)
    target = s1.save_ckpt(model, optim, 30, "fp", 300, tmp_path / "ckpt", 3, rng)
    state = s1.load_ckpt(target)
    needed = {"step", "optimizer", "tokens_seen", "numpy_rng_seed", "torch_rng_state"}
    assert needed <= set(state), "续训要用的状态一样都不能少"
    assert state["meta"]["step"] == 30


# ── C2 `-k e2e`（@slow）：smoke_tiny 端到端 ───────────────────────────────────────────

SAMPLE_HEAD = re.compile(r"^\s*\d+\.\s")


def _sample_lines(notes_text: str) -> list[str]:
    """采样小节里那几条编号原文行（`1. ` 起头）；小节之外的一律不计。"""
    section = notes_text.split("## 采样冒烟", 1)
    if len(section) == 1:
        return []
    return [line for line in section[1].splitlines() if SAMPLE_HEAD.match(line)]


@pytest.mark.slow
def test_e2e_smoke_tiny_end_to_end(s1, tmp_path, monkeypatch):
    """C2：一条命令跑完 smoke_tiny，退出码 0、曲线降、8 条采样、存档可载，CPU < 10 分钟。"""
    if not (CORPUS_TINY / "index.json").exists():
        pytest.skip("runs/corpus_tiny 不在；先跑 A2（python -m sys1.data.pretrain_corpus "
                    "--config tiny --stream-limit 200MB --out runs/corpus_tiny）")
    keep = os.environ.get("DMLAYA_KEEP_ARTIFACTS") == "1"
    if not keep:
        monkeypatch.setenv("DMLAYA_RUNS_DIR", str(tmp_path / "runs"))
    root = RUNS_ROOT if keep else Path(str(tmp_path / "runs"))
    root.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in root.glob("*")}

    started = time.time()
    code = s1.main(["--config", "smoke_tiny", "--sample", "8", "--quiet"])
    elapsed = time.time() - started

    assert code == 0, f"端到端必须正常退出，收到 {code}"
    assert elapsed < E2E_BUDGET_SEC, f"CPU 用时 {elapsed:.0f}s 超过 10 分钟预算"

    created = sorted({p.name for p in root.glob("*")} - before)
    assert len(created) == 1, f"应只新开一个 run 目录，收到 {created}"
    run_dir = root / created[0]
    rows = _rows(run_dir / "metrics.jsonl")
    assert len(rows) >= 5, f"spec 要至少 5 行曲线，收到 {len(rows)}"
    steps = [r["step"] for r in rows]
    assert steps == sorted(steps) and len(steps) == len(set(steps)) == 16  # step 0 + 每 20 步
    first, last = rows[0]["loss"], rows[-1]["loss"]
    assert abs(rows[0]["loss"] - math.log(VOCAB)) < 2.0, \
        "随机初始化时，下一词的代价本该贴近 ln(表规模)——差太远说明起点不是空模型"
    assert last < first, f"首末 loss 没下降：{first} -> {last}"
    assert last < first - 0.05, f"下降幅度不足以说明真在学：{first} -> {last}"
    print(f"[e2e] run={run_dir.name} loss {first:.4f} -> {last:.4f} 用时 {elapsed:.1f}s")

    ckpt = s1.find_latest_ckpt(run_dir)
    assert ckpt is not None, "spec 要求产出 checkpoint"
    loaded = Decoder.load(ckpt)
    out = loaded.lm_head(loaded(torch.randint(2, VOCAB, (1, 128))))
    assert torch.isfinite(out).all()
    assert (run_dir / "model" / "config.json").exists(), "最终模型目录留给 p1-09 消费"

    notes = (run_dir / "notes.md").read_text(encoding="utf-8")
    assert len(_sample_lines(notes)) == 8, "C1 口径：8 条采样原文全部入 notes"
    assert "人工判定" in notes
    assert "结论：" in notes and "结论：待填写" not in notes.splitlines()[-1]
    recorded = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    flat = json.dumps(recorded, ensure_ascii=False)
    assert "s1-pretrain" in flat, "run 的参数单里要能查出这是 S1 这一档（报告里每个数字指得回来）"
    assert "smoke_tiny" in flat, "配置路径也一并留档，改没改超参一眼可见"
