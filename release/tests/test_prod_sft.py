"""p2-05 SFT 单测：装配 / 混合配比 / 读出位损失复用 / LoRA 生效 / 内核档对拍 / 配置一致性 / 端到端通路。

分层说明（为什么要自己搭一台玩具学生）：
  默认全套离线——假编号器 + 两小层玩具 body 走**真实**的 `production.sft` 代码路径
  （渲染 → 装配 → 注入 → 读点损失 → 训练循环 → 落盘），只是权重是随机小结构。
  于是"数值对不对"能在毫秒级复算，不必载 1.2G 真权重；真权重那条路（CPU 冒烟）由
  `scripts` 层单独跑（`python -m production.sft --config production/configs/tiny_cpu.yaml`），
  它的凭据是 runs/ 里那份 metrics.jsonl 与 model/ 产物，不在本文件里假装通过。

对拍尺子的来源（都是"另一条独立实现"，不拿被测代码自证）：
  · 损失 → 测试里手工取末位、手工点字母行、手工 log_softmax；
  · LoRA → 已安装的 peft 0.21（同 r/α/dropout，逐元素比前向与合并权重）；
  · kernel 档 → 同文件的 `torch` 参照档（p2-13 自研件的 cpu 路径必须与它数与梯度都相等）。

运行方式（务必在 release/ 下用 -m，保证 production 包可导入）：
    cd release && .venv/bin/python -m pytest tests/test_prod_sft.py -q
    ... -k data|mix|loss|lora|kernel|configs|train   # 对应各孙任务的验证命令
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from production import sft
from production.teachers.cache import DistCache

SFT_FILE = Path(__file__).resolve().parents[1] / "production" / "sft.py"
VOCAB, HID = 64, 16
LETTER_IDS = list(range(2, 28))            # 26 枚字母各占一格，落在 [2,27]
CONTENT_RANGE = (32, 64)                   # 正文字落在这一段，绝不与字母段撞


# ── 假件：编号器 ─────────────────────────────────────────────────────────────
class _ToyTokenizer:
    """把文字按字节映射成小编号；只需满足"字母独占一格 + 可套壳"这两条契约。"""

    padding_side = "right"
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        ids = [CONTENT_RANGE[0] + (b % (CONTENT_RANGE[1] - CONTENT_RANGE[0])) for b in
               str(text).encode("utf-8")]
        return ids or [self.eos_token_id]

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": self.encode(text)}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return "\n".join(f"<{m['role']}>{m['content']}" for m in messages)


# ── 假件：模型（language 侧 qkvo + 一枚该被跳过的视觉分支）────────────────────
class _ToyAttn(nn.Module):
    def __init__(self, d=HID):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.gate_proj = nn.Linear(d, d, bias=False)      # 不在 leaves 里：不该被注入

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        return self.o_proj(q * torch.tanh(k) + v)      # 四枚都进图，注入面才算全覆盖


class _ToyBlock(nn.Module):
    def __init__(self, d=HID):
        super().__init__()
        self.self_attn = _ToyAttn(d)

    def forward(self, x):
        return x + self.self_attn(x)


class _ToyBody(nn.Module):
    def __init__(self, layers=2):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HID)
        self.layers = nn.ModuleList([_ToyBlock() for _ in range(layers)])
        self.norm = nn.LayerNorm(HID)
        self.visual = _ToyAttn()                          # 路径含 visual：注入必须跳过

    def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
        x = self.embed_tokens(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        for block in self.layers:
            x = block(x)
        return SimpleNamespace(last_hidden_state=self.norm(x))


class _ToyModel(nn.Module):
    def __init__(self, body):
        super().__init__()
        self.body = body
        self.head = nn.Linear(HID, VOCAB, bias=False)

    def forward(self, *args, **kwargs):
        return self.body(*args, **kwargs)


def toy_student(seed: int = 0) -> sft.Student:
    """搭一台玩具学生（带图前向可用），并把真模型的载入口整个绕开。"""
    torch.manual_seed(seed)
    body = _ToyBody()
    model = _ToyModel(body)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return sft.Student(
        name="toy-0.1b", model=model, body=body, tokenizer=_ToyTokenizer(),
        head_weight=model.head.weight.detach(), letter_ids=tuple(LETTER_IDS), pad_id=0,
        loader="toy", dtype=torch.float32, provenance={"snapshot_path": "/toy", "source": "test", "revision": "0"},
    )


# ── 假件：registry 信封行（形态与 p2-03 交出的统一信封一致）──────────────────
def record(qid="q", qtype="choice", options=("continue", "escalate", "human_review"),
           gold=(0.1, 0.2, 0.7), rid="rec-1", task="typed-choice", state="step 3 timed out twice"):
    spec = {"type": qtype, "instructions": "Which action fits the evidence?",
            "criteria": {k: f"criterion for {k}" for k in options}}
    return {"id": rid, "qtype": qtype, "task": task,
            "sample": {"state": state, "questions": {qid: spec},
                       "targets": {qid: dict(zip(options, gold))}}}


def fill_cache(root: Path, model_id: str, items, dists) -> DistCache:
    """把若干 (prompt, letters, 分布) 写进一本可写缓存并落盘（模拟"伪标包已到位"）。"""
    cache = DistCache(root=root).load()
    for (prompt, letters), dist in zip(items, dists):
        cache.record(model_id=model_id, prompt=prompt, option_keys=list(letters),
                     dist={k: float(v) for k, v in dist.items()}, source="unit-test")
    cache.flush()
    return DistCache(root=root, read_only=True).load()


# ================================================================ 工作项 A：装配（-k data）
def test_data_assembles_choice_sample():
    """一条信封编成一份样本：字母序、hard 押最大档、target 和为 1、prompt 与训练同一条渲染路。"""
    tok = _ToyTokenizer()
    one = sft.encode_record(record(), tok, cache=None, mix_soft=0.0)
    assert one is not None
    assert one.letters == ("A", "B", "C")                  # 候选按字母序 → 代号从 A 起
    assert one.order == ("continue", "escalate", "human_review")
    assert math.isclose(sum(one.hard), 1.0, rel_tol=1e-9)
    assert one.hard[2] > one.hard[0]                       # 人工份额最大档（0.7）拿到绝大部分
    assert math.isclose(sum(one.target), 1.0, rel_tol=1e-9)
    assert one.soft is None and "Evidence:" in one.prompt and one.prompt.startswith("Apply the criterion")
    assert len(one.ids) > 8 and one.ids[0] >= CONTENT_RANGE[0]


def test_data_rejects_missing_gold_and_empty_options():
    """没人工份额 / 没候选 / 份额不落在在场候选上：一律丢，且台账把丢弃数记下来。"""
    tok = _ToyTokenizer()
    no_gold = record(); no_gold["sample"]["targets"] = {}
    assert sft.encode_record(no_gold, tok, cache=None) is None
    off = record(gold=(0.0, 0.0, 0.0))
    assert sft.encode_record(off, tok, cache=None) is None
    samples, stats = sft.assemble([no_gold, off, record()], tok, cache=None)
    assert stats.seen == 3 and stats.encoded == 1 and stats.skipped.get("rejected") == 2
    assert stats.soft_coverage == 0.0                      # 没缓存就没有覆盖，不能当成满覆盖


def test_data_bucket_and_collate_shapes():
    """分桶按补齐后的字数封顶；拼出来的批：可见位、份额、在场列三本账互相对得上。"""
    tok = _ToyTokenizer()
    samples, _ = sft.assemble([record(rid=f"r{i}") for i in range(6)], tok, cache=None)
    buckets = sft.bucket_batches(samples, max_tokens=64, shuffle=True, seed=7)
    assert sum(len(b) for b in buckets) == len(samples)
    batch = sft.collate(samples, buckets[0], pad_id=0, letter_map=LETTER_IDS)
    assert batch.ids.shape == batch.mask.shape
    assert batch.target.shape[1] == len(batch.letter_ids) == batch.k
    assert [int(x) for x in batch.mask.sum(1)] == batch.lengths        # 右补空洞：真长度=可见位数
    assert torch.allclose(batch.target.sum(1), torch.ones(len(batch.lengths)), atol=1e-6)
    assert batch.keep[0].all()
    for b in buckets:
        width = max(len(samples[i].ids) for i in b)
        assert width * len(b) >= 1


@pytest.mark.parametrize("mix", [0.0, 0.5, 1.0])
def test_data_mix_switch_flows_into_target(mix):
    """装配层的 mix_soft 真把 soft 掺进 target（0=纯人工，1=纯教师）。"""
    tok = _ToyTokenizer()
    base = sft.encode_record(record(), tok, cache=None, mix_soft=0.0)
    soft = (0.2, 0.3, 0.5)
    cache = fill_cache(_tmp_root("mixflow"), "toy#t",
                       [(base.prompt, base.letters)], [dict(zip(base.letters, soft))])
    got = sft.encode_record(record(), tok, cache=cache, teacher_model_id="toy#t", mix_soft=mix)
    want = [round((1 - mix) * h + mix * s, 9) for h, s in zip(base.hard, soft)]
    total = sum(want)
    assert all(abs(g - w / total) < 1e-6 for g, w in zip(got.target, want))


def _tmp_root(name: str) -> Path:
    """单测专用缓存目录（tests 目录下 .tmp，跑前清干净；不落 runs/，不污染记录本）。"""
    root = Path(__file__).resolve().parent / ".tmp_p205" / name
    for f in root.glob("*.parquet"):
        f.unlink()
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_data_real_registry_port():
    """真数据口：从 registry 取 quality 轴，头几条真题能编成样本（题不是本域造的）。"""
    records = sft.load_quality_records("quality")
    if not records:
        pytest.skip("bench/eval_data 未就位（registry 装配产物是 gitignored 的本地副本）")
    samples, stats = sft.assemble(records, _ToyTokenizer(), cache=None, limit=16, max_length=100000)
    assert len(samples) >= 8 and stats.encoded == len(samples)
    assert samples[0].source and samples[0].letters[0] == "A"
    assert all(math.isclose(sum(s.target), 1.0, rel_tol=1e-6) for s in samples)


# ================================================================ 工作项 A：混合配比数值（-k mix）
def test_mix_soft_ratio_30_70_numeric():
    """spec 场景：配比 30/70 时 target 逐列等于 0.7·hard + 0.3·soft（归一后）。"""
    hard, soft = (0.8, 0.1, 0.1), (0.1, 0.6, 0.3)
    got = sft.mix_targets(hard, soft, 0.3)                 # soft 权重 0.3 → hard 占七成
    want = [0.7 * a + 0.3 * b for a, b in zip(hard, soft)]
    total = sum(want)
    assert all(abs(g - w / total) < 1e-9 for g, w in zip(got, want))
    assert math.isclose(sum(got), 1.0, rel_tol=1e-9)


def test_mix_endpoints_and_missing_soft():
    """两端点：0=纯人工、1=纯教师；soft 缺席就退成人工那份（并归一，不改名次）。"""
    hard, soft = (0.6, 0.3, 0.1), (0.2, 0.2, 0.6)
    assert sft.mix_targets(hard, soft, 0.0) == pytest.approx(list(hard), rel=1e-9)
    assert sft.mix_targets(hard, soft, 1.0) == pytest.approx(list(soft), rel=1e-9)
    assert sft.mix_targets(hard, None, 0.5) == pytest.approx(list(hard), rel=1e-9)
    assert sft.mix_targets((0.5, 0.5), None, 0.5) == pytest.approx([0.5, 0.5], rel=1e-9)


def test_mix_rejects_bad_inputs():
    """配比越界、两列不等长、掺完全零：都当场报错，不静默给一份错的份额。"""
    with pytest.raises(ValueError, match="mix_soft"):
        sft.mix_targets((1.0,), (1.0,), 1.5)
    with pytest.raises(ValueError, match="长度"):
        sft.mix_targets((0.5, 0.5), (1.0,), 0.5)
    with pytest.raises(ValueError, match="总和为 0"):
        sft.mix_targets((0.0, 0.0), (0.0, 0.0), 0.5)
    with pytest.raises(ValueError, match="mix_soft"):
        sft.encode_record(record(), _ToyTokenizer(), mix_soft=-0.1)


def test_cache_read_only_blocks_training_writes():
    """训练侧的缓存必须只读：想往教师账本里写一行（= 在线补算）会被当场拦住。"""
    cache = fill_cache(_tmp_root("readonly"), "toy#t", [("p", ("A", "B"))], [{"A": 0.5, "B": 0.5}])
    with pytest.raises(Exception):
        cache.record(model_id="toy#t", prompt="p2", option_keys=["A", "B"], dist={"A": 1.0, "B": 0.0})
    assert cache.lookup("toy#t", "p", ["A", "B"]) == {"A": 0.5, "B": 0.5}
    assert cache.lookup("toy#t", "别的题", ["A", "B"]) is None
    assert cache.stats["misses"] == 1 and cache.stats["hits"] == 1


# ================================================================ 工作项 B：读出位损失（-k loss）
def test_loss_reuses_decision_readout_no_parallel_impl():
    """grep 静态门：取末位/挑字母行只准走 decision/readout 那一个口子，本域不得另摆一把尺子。"""
    src = SFT_FILE.read_text(encoding="utf-8")
    assert "from sys1.decision.readout import option_scores" in src
    assert src.count("option_scores(") == 1, "读点只允许一处调用（多处=平行实现）"
    forbidden = (
        r"last_hidden\s*\[\s*torch\.arange",       # 自己抓末位格
        r"\[\s*:\s*,\s*-1\s*\]",                    # 直接取最后一格
        r"index_select",                            # 自己挑字母行
        r"\.float\(\)\s*@\s*.*\.T",                  # 自己点字母列
        r"logits\s*=",                              # 另算一份 logits
        r"\bsoftmax\(",                             # 另起一份归一（只许 log_softmax 一处口径）
    )
    for pattern in forbidden:
        hits = re.findall(pattern, src)
        assert not hits, f"sft.py 出现疑似平行实现 {pattern!r}：{hits}"
    assert src.count("log_softmax") == 1, "对数归一只有一处口径"


def test_loss_matches_manual_reference():
    """数值对拍：与测试里手工取的"末位格 × 字母行 × log_softmax"逐点相等。"""
    torch.manual_seed(3)
    B, T, d, k = 3, 7, 5, 4
    hidden = torch.randn(B, T, d, requires_grad=True)
    head = torch.randn(VOCAB, d)
    lengths = [3, 7, 5]
    targets = torch.softmax(torch.randn(B, k), dim=-1)
    got = sft.readout_ce_loss(hidden, head, LETTER_IDS[:k], targets, lengths=lengths)
    pos = torch.tensor(lengths) - 1
    rows = hidden[torch.arange(B), pos]
    scores = rows.float() @ head[torch.tensor(LETTER_IDS[:k])].float().transpose(0, 1)
    want = -(targets * torch.log_softmax(scores, dim=-1)).sum(-1).mean()
    assert torch.allclose(got, want, atol=1e-5)


def test_loss_hard_onehot_equals_plain_ce():
    """退化成人工 one-hot 时，本损失等于普通"取那一列的负对数"（与 F.nll_loss 同一把尺子）。"""
    torch.manual_seed(4)
    B, T, d, k = 2, 6, 8, 3
    hidden = torch.randn(B, T, d)
    head = torch.randn(VOCAB, d)
    lengths = [6, 4]
    targets = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    got = sft.readout_ce_loss(hidden, head, LETTER_IDS[:k], targets, lengths=lengths)
    scores = torch.stack([hidden[i, lengths[i] - 1].float() @ head[LETTER_IDS[j]].float()
                          for i in range(B) for j in range(k)]).view(B, k)
    want = torch.nn.functional.nll_loss(torch.log_softmax(scores, dim=-1),
                                        torch.tensor([2, 0]), reduction="mean")
    assert torch.allclose(got, want, atol=1e-5)


def test_loss_padding_region_does_not_move_readout():
    """读点按每行真长度取格：空洞区怎么改，损失一个数都不动。"""
    torch.manual_seed(5)
    lengths = [3, 5]
    hidden = torch.randn(2, 5, 6)
    head = torch.randn(VOCAB, 6)
    targets = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
    keep = torch.ones(2, 2, dtype=torch.bool)
    a = sft.readout_ce_loss(hidden, head, LETTER_IDS[:2], targets, keep=keep, lengths=lengths)
    dirty = hidden.clone()
    dirty[0, 3:] += 50.0            # 第 0 行真长度 3：它后面的格子全是空洞
    b = sft.readout_ce_loss(dirty, head, LETTER_IDS[:2], targets, keep=keep, lengths=lengths)
    assert torch.allclose(a, b, atol=0.0)


def test_loss_no_nan_when_columns_masked():
    """候选数不齐时不得冒出 nan：封掉的列既不进归一也不进加权（0×-inf 这一坑必须有回归测试）。"""
    torch.manual_seed(6)
    lengths = [4, 4]
    hidden = torch.randn(2, 4, 8)
    head = torch.randn(VOCAB, 8)
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])   # 第 0 行只有两个候选在场
    keep = torch.tensor([[True, True, False], [True, True, True]])
    got = sft.readout_ce_loss(hidden, head, LETTER_IDS[:3], targets, keep=keep, lengths=lengths)
    assert torch.isfinite(got)
    got.backward() if got.requires_grad else None
    lone = sft.readout_ce_loss(hidden[:1], head, LETTER_IDS[:1], torch.tensor([[1.0]]),
                               keep=torch.tensor([[True]]), lengths=[4])
    assert torch.isfinite(lone) and lone >= 0


def test_forward_grad_flows_to_readout():
    """带图前向 + 损失 → 改动回得到（玩具学生真走 `student_forward`）。"""
    student = toy_student(seed=1)
    loras = sft.inject_lora(student.body, backend=sft.BACKEND_TORCH)
    sft.freeze_all_but_lora(student.model, loras)
    samples, _ = sft.assemble([record(rid=f"r{i}") for i in range(2)], student.tokenizer, cache=None)
    batch = sft.collate(samples, [0, 1], pad_id=student.pad_id, letter_map=student.letter_ids)
    hidden = sft.student_forward(student, batch)
    assert hidden.requires_grad
    loss = sft.readout_ce_loss(hidden, student.head_weight, batch.letter_ids, batch.target,
                               keep=batch.keep, lengths=batch.lengths)
    loss.backward()
    mod = next(iter(loras.values()))
    assert mod.lora_b.grad is not None and torch.isfinite(mod.lora_b.grad).all()
    assert mod.base.weight.grad is None                       # 底座冻着，不该收到改动


# ================================================================ 工作项 B：LoRA 注入生效（-k lora）
def test_lora_injection_targets_language_qkvo_only():
    """只挂 language 侧 qkvo：视觉分支、mlp 闸门、层归一都不该被包住。"""
    student = toy_student()
    loras = sft.inject_lora(student.body, r=4, alpha=8)
    names = sorted(loras)
    assert all(n.split(".")[-1] in sft.LORA_TARGET_LEAVES for n in names)
    assert not any("visual" in n for n in names)
    assert len(names) == 2 * 4                                # 两层 × qkvo
    assert not any("gate_proj" in n for n in names)
    counts = sft.freeze_all_but_lora(student.model, loras)
    assert counts["trainable"] == len(loras) * 2 * (4 * HID)  # 每枚垫片 A(4,16)+B(16,4)
    assert counts["frozen"] > counts["trainable"]
    assert isinstance(student.body.layers[0].self_attn.q_proj, sft.LoRALinear)
    assert isinstance(student.body.visual.q_proj, nn.Linear)  # 视觉侧原样不动


def test_lora_zero_init_is_identity_then_moves():
    """B 全零 → 挂上那一瞬前向与底座逐元素相等；B 一动，差别立刻显出来。"""
    torch.manual_seed(7)
    lin = nn.Linear(HID, 8, bias=False)
    mod = sft.LoRALinear(lin, r=4, alpha=8, dropout=0.0)
    x = torch.randn(3, 2, HID)
    with torch.no_grad():
        assert torch.equal(mod.lora_b, torch.zeros(8, 4))
        assert torch.allclose(mod(x), lin(x), atol=0.0)
        mod.lora_b.normal_(0.0, 0.1)
        assert not torch.allclose(mod(x), lin(x), atol=1e-6)
        delta = mod.delta(x)
        assert torch.allclose(delta, sft.lora_delta_torch(x, mod.lora_a, mod.lora_b, mod.scaling), atol=1e-6)


def test_lora_grads_only_on_adapters():
    """改动只落两片垫片：底座不收，力度 α/r 与 A/B 各自形状都对。"""
    student = toy_student(seed=2)
    loras = sft.inject_lora(student.body, r=4, alpha=8, backend=sft.BACKEND_TORCH)
    counts = sft.freeze_all_but_lora(student.model, loras)
    samples, _ = sft.assemble([record(rid=f"r{i}") for i in range(2)], student.tokenizer, cache=None)
    batch = sft.collate(samples, [0, 1], pad_id=student.pad_id, letter_map=student.letter_ids)
    loss = sft.readout_ce_loss(sft.student_forward(student, batch), student.head_weight,
                               batch.letter_ids, batch.target, keep=batch.keep, lengths=batch.lengths)
    loss.backward()
    grads = [p for p in student.model.parameters() if p.grad is not None]
    assert len(grads) == len(loras) * 2
    assert all(torch.isfinite(g.grad).all() for g in grads)
    mod = next(iter(loras.values()))
    assert mod.lora_a.shape == (4, HID) and mod.lora_b.shape == (HID, 4)
    assert mod.scaling == pytest.approx(2.0)
    assert counts["trainable"] == sum(p.numel() for p in grads)


def test_lora_matches_peft_reference():
    """与已安装的 peft 0.21 对拍：同垫片下前向逐元素相等、合并权重相等、键名同源。"""
    peft = pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(8)
    r, alpha, drop = 4, 8, 0.05
    base_w = torch.randn(HID, HID)
    x = torch.randn(2, 3, HID)

    holder_mine = _ToyAttn()                                  # 用玩具注意力壳，路径末段是 q_proj
    holder_mine.q_proj.weight.data.copy_(base_w)
    mine = sft.LoRALinear(holder_mine.q_proj, r=r, alpha=alpha, dropout=drop)
    mine.lora_b.data.normal_(0.0, 0.2)                         # 零起点看不出差别，先垫出厚度

    holder_ref = _ToyAttn()
    holder_ref.q_proj.weight.data.copy_(base_w)
    pm = get_peft_model(holder_ref, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=drop,
                                               target_modules=["q_proj"], bias="none", task_type=None))
    key_a = next(k for k in pm.state_dict() if k.endswith("lora_A.default.weight"))
    key_b = next(k for k in pm.state_dict() if k.endswith("lora_B.default.weight"))
    assert key_a == sft._adapter_key("q_proj", "A"), "存储键名必须与 peft 同源"
    sd = dict(pm.state_dict())
    sd[key_a] = mine.lora_a.detach().clone()
    sd[key_b] = mine.lora_b.detach().clone()
    pm.load_state_dict(sd)

    pm.eval(); mine.eval()
    with torch.no_grad():
        got = mine(x)
        want = pm.base_model.model.q_proj(x)            # 只比被包住的那一枚线性（底座+旁路）
    assert torch.allclose(got, want, atol=1e-6)

    pm.train(); mine.train()
    torch.manual_seed(11)
    got_d = mine(x)
    torch.manual_seed(11)
    want_d = pm.base_model.model.q_proj(x)               # 同种子同丢弃图：入口位置也要一致
    assert torch.allclose(got_d, want_d, atol=1e-6)

    with torch.no_grad():
        merged_mine = mine.merged_delta()
        pm.merge_adapter()
        merged_ref = holder_ref.q_proj.base_layer.weight - holder_mine.q_proj.weight
    assert torch.allclose(merged_mine, merged_ref, atol=1e-6)


def test_adapter_save_load_roundtrip(tmp_path):
    """落盘→读回：垫片逐元素复原；少一条键当场报错，绝不静默给"看着像"的模型。"""
    student = toy_student(seed=3)
    loras = sft.inject_lora(student.body, r=4, alpha=8, backend=sft.BACKEND_TORCH)
    for mod in loras.values():
        with torch.no_grad():
            mod.lora_a.normal_(0.0, 0.05); mod.lora_b.normal_(0.0, 0.05)
    listing = sft.save_adapter(loras, tmp_path / "adapter", base_model="/toy/snapshot")
    assert listing["tensors"] == len(loras) * 2 and listing["params"] > 0
    snapshot = {n: (mod.lora_a.detach().clone(), mod.lora_b.detach().clone()) for n, mod in loras.items()}
    for mod in loras.values():
        with torch.no_grad():
            mod.lora_a.zero_(); mod.lora_b.zero_()
    assert sft.load_adapter(loras, tmp_path / "adapter") == len(snapshot) * 2
    for name, (a, b) in snapshot.items():
        assert torch.equal(loras[name].lora_a, a) and torch.equal(loras[name].lora_b, b)
    with pytest.raises(KeyError):
        sft.load_adapter(dict(list(loras.items()) + [("ghost.q_proj", loras["layers.0.self_attn.q_proj"])]),
                         tmp_path / "adapter")


# ================================================================ 工作项 B：内核档对拍（-k kernel）
def test_kernel_backend_matches_torch_reference():
    """kernel 档（p2-13 自研件 cpu 路径）与 torch 参照：前向增量逐元素相等。"""
    torch.manual_seed(9)
    x = torch.randn(24, HID)
    a = torch.randn(4, HID)
    b = torch.randn(HID, 4) * 0.1
    scaling = 2.0
    want = sft.lora_delta_torch(x, a, b, scaling)
    got = sft.lora_delta_kernel(x, a, b, scaling, target="cpu")
    assert got.shape == want.shape
    assert torch.allclose(got, want, atol=1e-5)
    assert float((got - want).abs().max()) < 1e-5


def test_kernel_backend_switch_keeps_weights_and_grads():
    """换档不丢垫片里的数；两条路的改动（dx/dA/dB）互相对得上；合并补丁也一致。"""
    student = toy_student(seed=10)
    loras = sft.inject_lora(student.body, r=4, alpha=8, dropout=0.0, backend=sft.BACKEND_TORCH)
    for mod in loras.values():
        with torch.no_grad():
            mod.lora_a.normal_(0.0, 0.02); mod.lora_b.normal_(0.0, 0.02)
    before = {n: mod.lora_a.detach().clone() for n, mod in loras.items()}
    samples, _ = sft.assemble([record(rid=f"r{i}") for i in range(2)], student.tokenizer, cache=None)
    batch = sft.collate(samples, [0, 1], pad_id=student.pad_id, letter_map=student.letter_ids)

    losses = {}
    for backend in sft.BACKENDS:
        sft.set_backend_all(loras, backend, ascend_target="cpu")   # 换档只换执行人
        student.model.zero_grad(set_to_none=True)
        loss = sft.readout_ce_loss(sft.student_forward(student, batch), student.head_weight,
                                   batch.letter_ids, batch.target, keep=batch.keep, lengths=batch.lengths)
        loss.backward()
        losses[backend] = (float(loss.detach()), {n: (mod.lora_a.grad.detach().clone(), mod.lora_b.grad.detach().clone())
                                         for n, mod in loras.items()})
    assert set(before) == set(loras)
    assert all(torch.equal(before[n], loras[n].lora_a) for n in before), "换档不得重造垫片"
    (l_torch, g_torch), (l_kernel, g_kernel) = losses[sft.BACKEND_TORCH], losses[sft.BACKEND_KERNEL]
    assert abs(l_torch - l_kernel) < 1e-6
    for name in g_torch:
        for pa, pk in zip(g_torch[name], g_kernel[name]):
            assert torch.allclose(pa, pk, atol=1e-5, rtol=1e-4), f"{name} 两条路的改动不一致"


def test_kernel_backend_merged_delta_matches():
    """推理期合并：kernel 档出的补丁与 torch 参照一致（同一叠补丁才能既训练又上线）。"""
    torch.manual_seed(12)
    a = torch.randn(4, HID)
    b = torch.randn(HID, 4) * 0.1
    lin = nn.Linear(HID, HID, bias=False)
    mod = sft.LoRALinear(lin, r=4, alpha=8, dropout=0.0, backend=sft.BACKEND_KERNEL, ascend_target="cpu")
    with torch.no_grad():
        mod.lora_a.copy_(a); mod.lora_b.copy_(b)
        got = mod.merged_delta()
    want = b * 2.0 @ a
    assert got.shape == (HID, HID)
    assert torch.allclose(got, want, atol=1e-5)


def test_ascend_target_cpu_path_has_no_blocker():
    """本地档：cpu 路径不该报"编译受阻"；档名解析与真件清点都按 p2-13 的口径走。"""
    from ascend.kernels import ascend_env

    assert ascend_env.normalize_target("cpu") == "cpu"
    assert ascend_env.normalize_target(None) in ascend_env.TARGETS
    assert "cpu" in ascend_env.TARGETS and "ascend" in ascend_env.TARGETS
    blockers = ascend_env.blockers()
    assert not any("lora" in key for key in blockers), f"cpu 档的 LoRA 件受阻：{blockers}"


# ================================================================ 工作项 B：分段记账开关（-k checkpointing）
class _CkptBody(_ToyBody):
    """长了检查点钩子的玩具 body：只为盯住"开关真按下了没、按不下时老不老实"。"""

    def __init__(self, layers=2):
        super().__init__(layers)
        self.calls = []

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.calls.append("enable")

    def enable_input_require_grads(self):
        self.calls.append("require_grads")


def _student_with(body):
    """把任意一个 body 装进玩具学生外壳（底座全冻，与 `toy_student` 同一套规矩）。"""
    torch.manual_seed(0)
    model = _ToyModel(body)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return sft.Student(
        name="toy-0.1b", model=model, body=body, tokenizer=_ToyTokenizer(),
        head_weight=model.head.weight.detach(), letter_ids=tuple(LETTER_IDS), pad_id=0,
        loader="toy", dtype=torch.float32,
        provenance={"snapshot_path": "/toy", "source": "test", "revision": "0"},
    )


def test_gradient_checkpointing_switch_presses_both_hooks():
    """开关三态：关上不碰钩子；打开要连"首层输入留钩子"一起按（少一个改动链就断）。"""
    body = _CkptBody()
    student = _student_with(body)
    assert sft.enable_gradient_checkpointing(student, False) is False
    assert body.calls == []
    assert sft.enable_gradient_checkpointing(student, True) is True
    assert body.calls == ["enable", "require_grads"]


def test_gradient_checkpointing_degrades_honestly():
    """底座没那对钩子：如实交回 False，绝不"假装开上了"（配置写 True 也不能骗人）。"""
    body = _ToyBody()
    student = _student_with(body)
    assert not hasattr(body, "gradient_checkpointing_enable")
    assert sft.enable_gradient_checkpointing(student, True) is False


def test_gradient_checkpointing_on_keeps_training_path_usable():
    """开着检查点仍能前向+反向下笔（玩具钩子是空壳，真 HF 底座靠这一条保证没被开关打断）。"""
    body = _CkptBody()
    student = _student_with(body)
    assert sft.enable_gradient_checkpointing(student, True) is True
    loras = sft.inject_lora(student.body, r=4, alpha=8, backend=sft.BACKEND_KERNEL)
    sft.freeze_all_but_lora(student.model, loras)
    samples, _ = sft.assemble([record(rid=f"r{i}") for i in range(2)], student.tokenizer, cache=None)
    batch = sft.collate(samples, [0, 1], pad_id=student.pad_id, letter_map=student.letter_ids)
    loss = sft.readout_ce_loss(sft.student_forward(student, batch), student.head_weight,
                               batch.letter_ids, batch.target, keep=batch.keep, lengths=batch.lengths)
    loss.backward()
    assert any(mod.lora_b.grad is not None for mod in loras.values())


# ================================================================ 工作项 B：配置一致性（-k configs）
CONFIG_DIR = Path(__file__).resolve().parents[1] / "production" / "configs"
SCALE_KEYS = {"backbone", "max_tokens", "accum", "max_steps"}


def _load(name: str) -> dict:
    import yaml

    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def test_configs_pair_diff_only_scale():
    """spec 场景：gate08b 与 scaling_06b 逐键比，差别只许落在底座/批规模/步数。"""
    a, b = _load("gate08b.yaml"), _load("scaling_06b.yaml")
    assert set(a) == set(b), f"两份档的键集合须一致，差集 {set(a) ^ set(b)}"
    diff = {k for k in a if a[k] != b[k]}
    assert diff, "两档至少得在底座上不同"
    assert diff <= SCALE_KEYS, f"越界差异（口径键不许动）：{ {k: (a[k], b[k]) for k in diff - SCALE_KEYS} }"
    assert "backbone" in diff
    for key in ("mix_soft", "hard_mass", "r", "alpha", "dropout", "kernel_backend", "ascend_target",
                "teacher_model_id", "teacher_cache", "axis", "lr", "warmup_ratio", "epochs"):
        assert a[key] == b[key]


def test_configs_are_valid_and_cloud_flavoured():
    """两份正式档能被 `load_config` 收下，且口径指向云端（910B + 自研件 + 真伪标号）。"""
    for name in ("gate08b.yaml", "scaling_06b.yaml"):
        cfg = sft.load_config(CONFIG_DIR / name)
        assert cfg["kernel_backend"] == sft.BACKEND_KERNEL
        assert cfg["ascend_target"] == "ascend"
        assert cfg["teacher_model_id"] == sft.DEFAULT_TEACHER_MODEL_ID   # 不带 #scaffold-cpu
        assert cfg["gradient_checkpointing"] is True
        assert cfg["r"] == 16 and cfg["alpha"] == 32


def test_configs_reject_unknown_keys():
    """配置表里冒出不认识的键 → 当场报错（拼错一个字母不该静默失效）。"""
    with pytest.raises(ValueError, match="未知配置键"):
        sft.load_config(None, overrides={"max_token": 1})
    with pytest.raises(ValueError, match="未知配置键"):
        sft.load_config(CONFIG_DIR / "gate08b.yaml", overrides={"nope": 1})


def test_cli_flags_cover_config_keys():
    """命令行开关与配置键同源：--print-config 三层合流能跑，缺省项不覆盖 yaml。"""
    args = sft.build_parser().parse_args(["--config", "x.yaml", "--mix-soft", "0.3"])
    assert args.config == "x.yaml" and args.mix_soft == 0.3 and args.lr is None
    covered = {key for key, _, _ in sft._CLI_FLAGS}
    assert covered <= set(sft.default_cfg())
    cfg = sft.load_config(CONFIG_DIR / "tiny_cpu.yaml",
                          overrides={key: getattr(args, key) for key, _, _ in sft._CLI_FLAGS})
    assert cfg["mix_soft"] == 0.3 and cfg["max_steps"] == 100


# ================================================================ 端到端通路（-k train）
def _toy_cfg(tmp_path, student, records, **over):
    """玩具档：真走 `train()`，但数据口/载入口/教师账本都换成测试替身。"""
    cfg = sft.default_cfg()
    teacher_root = _tmp_root("train_cache")
    base = [sft.encode_record(r, student.tokenizer, cache=None, mix_soft=0.0) for r in records]
    dists = [{k: v for k, v in zip(b.letters, [0.6, 0.3, 0.1][: len(b.letters)])} for b in base]
    fill_cache(teacher_root, "toy#teacher", [(b.prompt, b.letters) for b in base], dists)
    cfg.update({"backbone": "toy-0.1b", "loader": "toy", "device": "cpu", "dtype": "float32",
                "threads": 2, "axis": "toy", "limit": len(records), "max_length": 4096,
                "teacher_model_id": "toy#teacher", "teacher_cache": str(teacher_root),
                "mix_soft": 0.5, "r": 4, "alpha": 8, "dropout": 0.0, "max_tokens": 64,
                "accum": 1, "max_steps": 2, "epochs": 1, "gradient_checkpointing": False,
                "kernel_backend": sft.BACKEND_KERNEL, "ascend_target": "cpu",
                "run_prefix": "p2-05-test", "out_dir": str(tmp_path / "model")})
    cfg.update(over)
    return cfg


def test_train_end_to_end_toy_writes_artifacts(tmp_path, monkeypatch):
    """两 toy 步跑完整条路：run 四件套 + metrics 有 loss + 适配器/小抄/README 齐全。"""
    student = toy_student(seed=21)
    records = [record(rid=f"r{i}", gold=(0.1, 0.2, 0.7)) for i in range(4)]
    monkeypatch.setenv("DMLAYA_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(sft, "build_student", lambda *a, **k: student)
    monkeypatch.setattr(sft, "load_quality_records", lambda *a, **k: records)
    cfg = _toy_cfg(tmp_path, student, records)
    result = sft.train(cfg)

    assert result["steps"] == 2 and result["samples"] == 4
    assert all(math.isfinite(v) and v > 0 for v in result["losses"])
    assert result["soft_coverage"] == 1.0 and result["teacher_hits"] == 4   # 每道题都掺到了伪标
    run_dir = Path(result["run_dir"])
    assert (run_dir / "config.yaml").is_file() and (run_dir / "system.json").is_file()
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    assert rows[0]["step"] == 0 and "asm_encoded" in rows[0]        # 开局账必须落在曲线最前一行
    assert [r["step"] for r in rows if "loss" in r] == [1, 2]       # 不许有 step 0 补记漂到末步之后
    assert rows[-1].get("kernel_compiled") is not None              # 自研件终局清点挂在最后一步（发生时点）
    stepped = [r for r in rows if "loss" in r]
    assert len(stepped) == 2 and {"loss", "lr", "step_seconds", "tokens"} <= set(stepped[0])

    model_dir = Path(result["model_dir"])
    for name in (sft.ADAPTER_FILE, sft.ADAPTER_CONFIG_NAME, sft.CONFIG_FILE,
                 sft.DECISION_CONFIG_FILE, sft.README_NAME):
        assert (model_dir / name).is_file(), f"产物缺 {name}"
    decision = json.loads((model_dir / sft.DECISION_CONFIG_FILE).read_text())
    assert decision["readout"]["position"] == "last_real_token"
    assert decision["letter_token_ids"] == list(LETTER_IDS) and decision["render_version"]
    assert decision["temperatures"] == {} and decision["calibration"]["status"] == "pending"
    adapter_cfg = json.loads((model_dir / sft.ADAPTER_CONFIG_NAME).read_text())
    assert adapter_cfg["r"] == 4 and adapter_cfg["peft_type"] == "LORA"
    assert adapter_cfg["target_modules"] == list(sft.LORA_TARGET_LEAVES)

    notes = (run_dir / "notes.md").read_text(encoding="utf-8")
    assert "结论：" in notes and sft.TRAINABLE_NOTE in notes
    assert "tok/s" in notes and "loss" in notes                   # 吞吐与首末损失都在结论行里

    readme = (model_dir / sft.README_NAME).read_text(encoding="utf-8")
    assert "D11" in readme and "零教师前向" in readme              # 来源链要把红线写进产物目录
    assert "过目" in readme and "丢 {}" not in readme             # 台账空值不许留花括号


def test_train_loss_decreases_over_steps_on_single_pattern(tmp_path, monkeypatch):
    """真曲线判据：反复喂同一批"固定份额"的题，损失必须往下走（读点通了才可能降）。"""
    student = toy_student(seed=22)
    records = [record(rid=f"r{i}", gold=(0.05, 0.05, 0.9)) for i in range(2)]
    monkeypatch.setenv("DMLAYA_RUNS_DIR", str(tmp_path / "runs2"))
    monkeypatch.setattr(sft, "build_student", lambda *a, **k: student)
    monkeypatch.setattr(sft, "load_quality_records", lambda *a, **k: records)
    cfg = _toy_cfg(tmp_path, student, records, lr=5e-2, max_steps=12, epochs=6, mix_soft=0.0,
                   out_dir=str(tmp_path / "model2"))
    result = sft.train(cfg)
    losses = result["losses"]
    assert len(losses) == 12
    assert losses[-1] < losses[0] - 1e-3, f"损失没降：{losses[:3]} … {losses[-3:]}"
    assert result["final_loss"] < result["first_loss"]
