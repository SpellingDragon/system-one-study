"""p1-09 工作项 B/C 验收单测（SFT 读出位 CE + 防泄漏自检）。

`-k` 过滤名即 tasks.md 条目：
  B1 `-k mask`         读出位 CE：非读出位置逐位数字梯度恒为零、读出位非零（手工梯度验证）
     `-k no_parallel`  grep 门：s2 复用 decision.option_scores，无平行"取末位/点字母行/整表 CE"实现
     `-k matches`      读出位 CE 与"手摊软标签交叉熵"逐值一致
     `-k mismatch`     targets 形状不符当场拒绝
  B3 `-k e2e` (`-m slow`) smoke SFT 端到端：微型起点 + 500 条转写集，留出集 choice/noul 超分桶基线，
                        产物含 decision_config.json（且带 temperatures 字段）
  C1 `-k leak`         check_leak：训练 id ∩ test id = ∅ 放行；植入重叠 id 即判泄漏

桩件/起点全部离线生成，不下载任何数据集，测试产物一律重定向到 tmp（DMLAYA_RUNS_DIR）。
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
from argparse import Namespace
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
S2_PATH = REPO_ROOT / "learning" / "s2_decision_sft.py"
LEAK_PATH = REPO_ROOT / "tools" / "check_leak.py"
TOKENIZER_DIR = REPO_ROOT / "runs" / "1003-s0-bpe-16k-realedu-zh-en" / "tokenizer"


def _load_by_path(module_name: str, path: Path):
    """按文件路径加载模块（避开命名空间包/坏 editable 安装的导入歧义）。"""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def s2():
    return _load_by_path("s2_decision_sft", S2_PATH)


@pytest.fixture(scope="module")
def leak():
    return _load_by_path("check_leak", LEAK_PATH)


# ---------------------------------------------------------------- 离线数据生成
CHOICE_OPTIONS = ["晴", "雨", "阴"]          # 固定候选集 → 字母序恒定 → 多数答案落在固定字母


def gen_records(n_noul: int, n_choice: int, seed: int, id_prefix: str = "sft-train") -> list[dict]:
    """造 n_noul 条 noul + n_choice 条 choice 的转写记录，id 互不重叠且带明确前缀。

    标签刻意带偏置（noul 70% 判假、choice 60% 落在"晴"），使留出集的多数类份额本就高于
    分桶随机基线——smoke 的意义是验证"读出位 CE 能把这份先验学进末位字母分"，而非考难题。
    """
    from sys1.data.transcribe import classification_to_choice, xnli_to_noul

    rng = random.Random(seed)
    records: list[dict] = []
    for i in range(n_noul):
        false_heavy = rng.random() < 0.70                      # 70% 判假（neutral/contradiction）
        label = rng.choice(["neutral", "contradiction"]) if false_heavy else "entailment"
        records.append(xnli_to_noul({
            "id": f"{id_prefix}-noul-{i:04d}",
            "premise": f"情境{i}：有人在窗边看着外面。",
            "hypothesis": f"推断{i}：外面正在下雨。",
            "label": label,
        }))
    for i in range(n_choice):
        roll = rng.random()
        label = "晴" if roll < 0.60 else ("雨" if roll < 0.80 else "阴")   # 60% 多数落"晴"
        records.append(classification_to_choice({
            "id": f"{id_prefix}-choice-{i:04d}",
            "text": f"报道{i}：天空的状况记录。",
            "options": CHOICE_OPTIONS,
            "label": label,
        }))
    rng.shuffle(records)
    return records


def write_jsonl(records: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return path


def make_tiny_ckpt(dir_: Path, vocab: int) -> Path:
    """随机初始化一个微型 Decoder 存成模型目录，充当 smoke 的可训练起点（p1-08 s1 未产出时的兜底）。"""
    from sys1.model import Decoder, ModelConfig

    cfg = ModelConfig(d=64, L=2, heads=2, ctx=256, vocab=vocab, rope_theta=10_000.0, seed=0)
    return Decoder(cfg).save(dir_)


# ---------------------------------------------------------------- B1：读出位 CE 手工梯度
def _leaf_batch(B: int, T: int, d: int, k: int, lengths: list[int], seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    last_hidden = torch.randn(B, T, d, generator=gen).requires_grad_(True)   # 叶子，直接考验屏蔽
    head_weight = torch.randn(64, d, generator=gen)
    letter_ids = list(range(1, k + 1))
    targets = torch.softmax(torch.randn(B, k, generator=gen), dim=-1)
    return last_hidden, head_weight, letter_ids, targets, lengths


def test_mask_non_readout_positions_get_zero_gradient(s2):
    """只在读出位计损：损失对非读出位置的逐位数字梯度恒为零，对读出位非零。"""
    B, T, d, k = 3, 7, 8, 4
    lengths = [2, 5, 7]                                                       # 读出位 = length-1
    last_hidden, head_weight, letter_ids, targets, lengths = _leaf_batch(B, T, d, k, lengths, seed=1)

    loss = s2.readout_ce_loss(last_hidden, head_weight, letter_ids, targets, lengths)
    loss.backward()

    grad = last_hidden.grad
    assert grad is not None
    for b, length in enumerate(lengths):
        readout_pos = length - 1
        # 阳性对照：读出位有梯度（否则整条读出链路根本没接到损失上）
        assert grad[b, readout_pos].abs().sum().item() > 0.0, f"行 {b} 读出位梯度不应为零"
        for t in range(T):
            if t != readout_pos:
                assert torch.allclose(grad[b, t], torch.zeros(d)), f"行 {b} 非读出位 t={t} 梯度应恰为零"


def test_mask_no_lengths_reads_tensor_end(s2):
    """lengths=None 时按整行最右一格读：梯度只在末列非零（其余列恒为零）。"""
    B, T, d, k = 2, 6, 8, 3
    last_hidden, head_weight, letter_ids, targets, _ = _leaf_batch(B, T, d, k, [T] * B, seed=2)

    loss = s2.readout_ce_loss(last_hidden, head_weight, letter_ids, targets, lengths=None)
    loss.backward()
    grad = last_hidden.grad
    for b in range(B):
        assert grad[b, T - 1].abs().sum().item() > 0.0
        assert torch.allclose(grad[b, : T - 1], torch.zeros(T - 1, d))


def test_readout_ce_matches_manual_soft_ce(s2):
    """读出位 CE 与"逐行 −Σ p_i·log softmax(z_i)"逐值一致（soft target 整体监督的定义式）。"""
    B, T, d, k = 4, 5, 8, 3
    lengths = [3, 5, 2, 4]
    last_hidden, head_weight, letter_ids, targets, lengths = _leaf_batch(B, T, d, k, lengths, seed=3)

    loss = s2.readout_ce_loss(last_hidden.detach(), head_weight, letter_ids, targets, lengths)

    from sys1.decision import option_scores

    scores = option_scores(last_hidden.detach(), head_weight, letter_ids, lengths)
    manual = -(targets * torch.log_softmax(scores, dim=-1)).sum(dim=-1).mean()
    assert torch.allclose(loss, manual, atol=1e-6)


def test_targets_shape_mismatch_rejected(s2):
    B, T, d, k = 2, 4, 6, 3
    last_hidden, head_weight, letter_ids, _targets, lengths = _leaf_batch(B, T, d, k, [4, 4], seed=4)
    bad = torch.zeros(B, k + 1)                                                # 列数与字母列不符
    with pytest.raises(ValueError):
        s2.readout_ce_loss(last_hidden.detach(), head_weight, letter_ids, bad, lengths)


# ---------------------------------------------------------------- grep 门：无平行读出实现
def test_no_parallel_imports_option_scores():
    """s2 必须复用 decision 的 option_scores，而不是自造一套末位读出。"""
    src = S2_PATH.read_text(encoding="utf-8")
    assert "option_scores" in src, "读出位 CE 应复用 sys1.decision.option_scores"
    assert "from sys1.decision import" in src


def test_no_parallel_full_vocab_ce_and_tensor_end_gather():
    """禁止在 s2 里铺整表交叉熵或用"张量末列"取末位（那是与读出层漂移的平行实现）。"""
    src = S2_PATH.read_text(encoding="utf-8")
    assert "cross_entropy" not in src, "读出位 CE 不该用整表 cross_entropy（那是逐 token 自回归监督）"
    assert "[:, -1]" not in src, "不该用 [:, -1] 当末位（右补空洞时正是空洞位，见 readout 设计红线）"


# ---------------------------------------------------------------- C1：防泄漏自检
def test_leak_disjoint_ids_pass(tmp_path, leak):
    train = write_jsonl(gen_records(10, 8, seed=5, id_prefix="sft-train"), tmp_path / "train.jsonl")
    test = write_jsonl(gen_records(4, 3, seed=9, id_prefix="typed-test"), tmp_path / "test.jsonl")
    report = leak.check(train, test)
    assert report["ok"] and report["overlap_count"] == 0
    assert leak.main(["--train", str(train), "--test", str(test)]) == 0


def test_leak_detects_overlap(tmp_path, leak):
    shared = {"id": "typed-test-LEAK", "premise": "p", "hypothesis": "h", "label": "neutral"}
    from sys1.data.transcribe import xnli_to_noul

    train = write_jsonl([xnli_to_noul(shared)], tmp_path / "train.jsonl")
    test = write_jsonl([xnli_to_noul(shared)], tmp_path / "test.jsonl")
    report = leak.check(train, test)
    assert not report["ok"] and report["overlap_count"] == 1
    assert leak.main(["--train", str(train), "--test", str(test)]) == 1


def test_leak_dir_input(tmp_path, leak):
    """--test 给目录也应收下其中所有 *.jsonl。"""
    (tmp_path / "typed_decisions").mkdir()
    write_jsonl(gen_records(3, 0, seed=11, id_prefix="td-a"), tmp_path / "typed_decisions" / "a.jsonl")
    write_jsonl(gen_records(2, 0, seed=12, id_prefix="td-b"), tmp_path / "typed_decisions" / "b.jsonl")
    ids, _, _ = leak.collect_ids(tmp_path / "typed_decisions")
    assert len(ids) == 5


# ---------------------------------------------------------------- B3：smoke SFT 端到端（slow）
@pytest.mark.slow
def test_e2e_smoke_sft_beats_bucket_baseline(tmp_path, s2, monkeypatch):
    """微型起点 + 500 条转写集跑读出位 SFT：留出集 choice/noul 各桶准确率 > 对应分桶随机基线。"""
    if not TOKENIZER_DIR.is_dir():
        pytest.skip("缺 S0 真实 tokenizer 产物，无法跑端到端编码")
    from sys1.lang import bpe

    vocab = bpe.load(TOKENIZER_DIR).get_vocab_size()
    ckpt = make_tiny_ckpt(tmp_path / "pretrain", vocab=vocab)
    train_file = write_jsonl(gen_records(300, 200, seed=0), tmp_path / "sft_train.jsonl")

    monkeypatch.setenv("DMLAYA_RUNS_DIR", str(tmp_path / "runs"))          # 记录本重定向到 tmp
    args = Namespace(
        ckpt=str(ckpt), tokenizer=str(TOKENIZER_DIR), data=str(train_file),
        epochs=10, lr=0.05, batch_size=32, holdout=100, seed=0,
    )
    result = s2.run_sft(args)

    # 产物：模型目录 + decision_config.json（含 temperatures 字段，s3 之后再填数）
    decision_cfg = Path(result["model_dir"]) / "decision_config.json"
    assert decision_cfg.is_file(), "SFT 产物必须含 decision_config.json"
    cfg = json.loads(decision_cfg.read_text(encoding="utf-8"))
    assert "temperatures" in cfg

    summary = result["summary"]
    assert os.fspath(result["run_id"])                                     # run-id 正常产出
    # 逐桶：留出准确率必须严格高于该桶随机基线（弱模型靠先验亦须可见信号）
    per = summary["per_bucket"]
    assert "noul" in per and "choice" in per, f"留出集应含 noul/choice 两桶，实得 {list(per)}"
    for qtype in ("noul", "choice"):
        for k, row in per[qtype].items():
            assert row["accuracy"] > row["baseline"], f"{qtype}/k={k} 未超基线：{row}"
    assert summary["all_beat_baseline"] is True
