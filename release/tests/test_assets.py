"""p2-01 载重接缝测试：装载路径与四道接缝校验、接缝破坏必须硬拦、换 backbone 冒烟。

口径：
  * 名字带 mock 的用例只验"分支逻辑/必须拒载"，用假对象，不需要网络与权重；
  * 名字带 real 的用例走 bench/ms_models 里的真快照（缺则 skip，不假绿）；
  * 资源纪律：前向一律 CPU（MPS 归一阶段训练长跑）；MPS 用例需 SYS1_ALLOW_MPS=1 显式放行。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from production import assets
from production.assets import (
    DEFAULT_CACHE_DIR,
    SEAM_LETTERS,
    SEAM_LETTER_ROWS,
    SEAM_MM_IDS,
    SEAM_THINK_OFF,
    SeamCheckError,
    canonical_prompt,
    check_letter_boundary,
    check_letter_rows,
    check_letters,
    check_mm_token_ids,
    check_think_off,
    fetch_snapshot,
    letter_ids_from_mapping,
    resolve_source,
)
from sys1.decision.render import RENDER_VERSION, THINK_OFF_SUFFIX

SNAP = DEFAULT_CACHE_DIR / "models" / "Qwen--Qwen3.5-0.8B" / "snapshots" / "master"
HAS_SNAP = SNAP.is_dir() and bool(list(SNAP.glob("*.safetensors")))
real_only = pytest.mark.skipif(not HAS_SNAP, reason="真快照不在 bench/ms_models：B2 要真下载，mock 不算完成")
need_mps = pytest.mark.skipif(
    os.environ.get("SYS1_ALLOW_MPS") != "1",
    reason="MPS 归一阶段训练长跑占用；显式 SYS1_ALLOW_MPS=1 才放行（B1 设备迁移用例）",
)


class _Ids:
    def __init__(self, ids):
        self.ids = list(ids)


class FakeBackend:
    """假 tokenizers.Tokenizer：按给定词表试编，可故意让某个字母不达标。"""

    def __init__(self, vocab=None, split_for=()):
        self.vocab = dict(vocab or {})
        self.split_for = set(split_for)

    def encode(self, text, add_special_tokens=False):
        if text in self.split_for:
            return _Ids([0, 0])                      # 故意编成两个符号：字母不达标
        if text in self.vocab:
            return _Ids([self.vocab[text]])
        return _Ids([self.vocab.get("<unk>", 0)])

    def id_to_token(self, i):
        for k, v in self.vocab.items():
            if v == i:
                return k
        return "<unk>"

class FakeTok:
    """假 HF 分词器：只交出接缝校验真正用到的四样（试编、套壳、查特殊符编号）。"""

    def __init__(self, backend=None, off="", on=None, special=None):
        self.backend_tokenizer = backend or FakeBackend()
        self._off = off
        self._on = off if on is None else on
        self.special = special or {}

    def encode(self, text, add_special_tokens=False):
        return self.backend_tokenizer.encode(text, add_special_tokens).ids

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False):
        return self._on if enable_thinking else self._off

    def convert_tokens_to_ids(self, token):
        return self.special.get(token, -1)


def good_vocab(split=()):
    """造一份"52 个字母都单 token"的词表；split 里那些字母故意编成两个符号。"""
    letters = assets.UPPERCASE + assets.LOWERCASE
    return {c: 32 + i for i, c in enumerate(letters) if c not in split}


@pytest.fixture(scope="session")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(SNAP))


@pytest.fixture(scope="session")
def bb():
    """真载一次（CPU/fp16）：本模块所有"换脑"用例共用同一个身板。"""
    return assets.load_backbone("qwen3.5-0.8b", device="cpu")


def test_load_aliases_and_registry():
    assert resolve_source("Qwen3.5_0.8B").key == "qwen3.5-0.8b"
    assert resolve_source("0.6b").modelscope_repo == "Qwen/Qwen3-0.6B"
    with pytest.raises(KeyError):
        resolve_source("llama-3-8b")


def test_load_prefers_modelscope_and_records_provenance(monkeypatch, tmp_path):
    import huggingface_hub
    import modelscope
    calls = []

    def fake_ms(repo, **kw):
        calls.append(("modelscope", repo, kw.get("revision"), kw.get("allow_patterns")))
        d = tmp_path / "ms"
        d.mkdir(exist_ok=True)
        (d / "config.json").write_text("{}")
        return str(d)

    monkeypatch.setattr(modelscope, "snapshot_download", fake_ms)
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda repo, **kw: (_ for _ in ()).throw(AssertionError("不该走兜底")))
    snap = fetch_snapshot("qwen3.5-0.8b", cache_dir=tmp_path, allow_patterns=["*.json"])
    assert calls[0][0] == "modelscope" and calls[0][2] == "master" and calls[0][3] == ["*.json"]
    cfg = snap.as_config()
    assert cfg["backbone_repo"] == "Qwen/Qwen3.5-0.8B" and cfg["backbone_revision"] == "master"
    assert {"backbone_snapshot", "backbone_download_bytes", "backbone_download_seconds",
            "backbone_endpoint", "backbone_files"} <= set(cfg)


def test_load_falls_back_to_huggingface(monkeypatch, tmp_path):
    import huggingface_hub
    import modelscope
    seen = {}

    def boom(repo, **kw):
        raise ConnectionError("modelscope 不可达")

    def fake_hf(repo, **kw):
        seen["repo"], seen["revision"] = repo, kw.get("revision")
        d = tmp_path / "hf"
        d.mkdir(exist_ok=True)
        (d / "config.json").write_text("{}")
        return str(d)

    monkeypatch.setattr(modelscope, "snapshot_download", boom)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_hf)
    snap = fetch_snapshot("0.6b", cache_dir=tmp_path)
    assert snap.source == "huggingface" and seen["repo"] == "Qwen/Qwen3-0.6B"
    assert seen["revision"] == "main"          # master 是 ModelScope 的写法，HF 侧要换名


def test_load_both_remotes_fail_reports_each(monkeypatch, tmp_path):
    import huggingface_hub
    import modelscope
    monkeypatch.setattr(modelscope, "snapshot_download",
                        lambda repo, **kw: (_ for _ in ()).throw(RuntimeError("ms 挂了")))
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda repo, **kw: (_ for _ in ()).throw(RuntimeError("hf 挂了")))
    with pytest.raises(FileNotFoundError) as exc:
        fetch_snapshot("qwen3.5-0.8b", cache_dir=tmp_path)
    msg = str(exc.value)
    assert "modelscope" in msg and "huggingface" in msg          # 两条路的原始报错都要留痕


@real_only
def test_load_backbone_real_cpu(bb):
    """真载：fp16 + CPU 装成窄接口对象，四道接缝在载入路径里当场过一遍（B2 的真实路径）。"""
    assert bb.dtype is torch.float16 and bb.device == "cpu"
    assert bb.hidden_size == 1024 and bb.seam.head_shape == (248320, 1024)
    assert bb.letter_rows.shape == (26, 1024)
    cfg = bb.provenance_config()
    assert cfg["backbone_repo"] == "Qwen/Qwen3.5-0.8B" and cfg["backbone_source"] == "modelscope"
    assert cfg["render_version"] == RENDER_VERSION
    assert "model" not in sys.modules or "sys1.model" not in sys.modules


@real_only
def test_seams_letters_single_token_real(tok):
    """接缝①（全计划最大单点风险）：A–Z 每个字母在真分词器里必须是一个符号。"""
    mapping = check_letters(tok)
    ids = letter_ids_from_mapping(mapping)
    assert ids == tuple(range(32, 58)), f"字母编号不是 32..57：{ids}"
    assert len(mapping) == 52 and set(mapping) == set(assets.UPPERCASE + assets.LOWERCASE)
    ref = DEFAULT_CACHE_DIR / "models" / "StartLuxAI--StartLux-Decision-0.8B"
    files = list(ref.glob("**/decision_config.json"))
    if files:                                                               # 同系参照：StartLux 也是 32 起
        import json
        letter_ids = json.loads(files[0].read_text())["letter_token_ids"]
        assert list(ids) == list(letter_ids)


@real_only
def test_seams_letter_rows_rebuild_real(tok):
    """接缝②：字母行随隐藏宽度重建——换个身板（宽度不同）取出的行就跟着变宽。"""
    ids = letter_ids_from_mapping(check_letters(tok))
    for hidden in (512, 1024):
        rows = check_letter_rows(torch.randn(2000, hidden), ids, hidden)
        assert rows.shape == (26, hidden) and rows.dtype is torch.float32


@real_only
def test_seams_think_off_snapshot_real(tok):
    """接缝③：模板尾部逐字节等于 decision 的关闭思考常量，并把实测指纹固化成快照。"""
    row = assets.from_systemone(assets.CANONICAL_STATE, dict(assets.CANONICAL_ROW_SPEC))
    messages, _ = assets.render_messages(row)
    text, suffix = check_think_off(tok, messages)
    assert suffix == THINK_OFF_SUFFIX and text.endswith(THINK_OFF_SUFFIX)
    prompt, ntok, sha = canonical_prompt(tok)
    assert ntok == 80 and sha == "a1891aca6c002e48", f"排版/模板动过：tokens={ntok} sha={sha}"
    assert prompt.endswith(THINK_OFF_SUFFIX)


@real_only
def test_seams_letter_boundary_tail_real(tok):
    """接缝①的边界半边：渲染文字尾部接 26 个字母任一，只多出那一格，不被前文吞并。"""
    ids = letter_ids_from_mapping(check_letters(tok))
    prompt, _, _ = canonical_prompt(tok)
    check_letter_boundary(tok, prompt, ids)


@real_only
def test_seams_mm_token_ids_real(tok):
    """第四校验：多模态四个编号存在、与分词器对得上、且不与字母抢号（p2-08 视觉接口）。"""
    import transformers
    cfg = transformers.AutoConfig.from_pretrained(str(SNAP))
    ids = letter_ids_from_mapping(check_letters(tok))
    found = check_mm_token_ids(cfg, tok, ids)
    assert found == {"image_token_id": 248056, "video_token_id": 248057,
                     "vision_start_token_id": 248053, "vision_end_token_id": 248054}



# ================================================================ A3 接缝破坏必须硬拦（mock）
# 口径：这一组只用假对象验"分支必须拒载 + 消息里写着是哪道缝"，不碰网络也不碰权重。

class _Cfg:
    """假 config：顶层有什么字段全由用例点名给（"缺字段"本身也是要被拦住的一种破坏）。"""

    def __init__(self, **fields):
        self.__dict__.update(fields)


class _MergingTok:
    """假分词器：文字尾部贴一个字母时，把末格与字母搅成一格（BPE 前缀合并的破坏形态）。"""

    def __init__(self, base=(1, 2, 3), merged=(1, 2, 99)):
        self.base = list(base)
        self.merged = list(merged)

    def encode(self, text, add_special_tokens=False):
        return list(self.merged) if text.endswith("A") else list(self.base)


#: 四个多模态记号在分词器里的在册编号（真检查点的实证值；破坏用例就在他身上动手）
MM_GOOD = {"<image_pad>": 248056, "<video_pad>": 248057,
           "<vision_start>": 248053, "<vision_end>": 248054}


def test_seam_fail_letters_split_mock():
    """接缝①破坏：'A' 被编成两个符号——必须抛 SeamCheckError 且消息含"字母单 token"。"""
    tok = FakeTok(backend=FakeBackend(good_vocab(split="A"), split_for={"A"}), off=THINK_OFF_SUFFIX)
    with pytest.raises(SeamCheckError) as exc:
        check_letters(tok)
    assert SEAM_LETTERS in str(exc.value)


def test_seam_fail_letters_symbol_not_self_mock():
    """接缝①破坏之二：只出一个符号、但符号原文不是字母本身（带空白前缀的变体）——同样拒载。"""
    vocab = {c: 32 + i for i, c in enumerate(assets.UPPERCASE + assets.LOWERCASE)}
    b_id = vocab["B"]
    backend = FakeBackend(vocab)
    origin = backend.id_to_token
    backend.id_to_token = lambda i: " b" if i == b_id else origin(i)   # 故意让 B 的原文对不上
    with pytest.raises(SeamCheckError) as exc:
        check_letters(FakeTok(backend=backend))
    assert SEAM_LETTERS in str(exc.value)


def test_seam_fail_letter_boundary_merge_mock():
    """接缝①的边界破坏：渲染文字尾部贴字母被并成一格，读点位就读不到纯字母了。"""
    ids = tuple(range(32, 58))
    with pytest.raises(SeamCheckError) as exc:
        check_letter_boundary(_MergingTok(), "已经排好的那段文字", ids)
    assert SEAM_LETTERS in str(exc.value)
    assert "A" in str(exc.value)                      # 消息要点名是哪个字母出的事


def test_seam_fail_letter_rows_width_mock():
    """接缝②破坏：输出层宽度与这副身板不符（挑出来的行配不上钥匙）。"""
    with pytest.raises(SeamCheckError) as exc:
        check_letter_rows(torch.zeros(200, 8), list(range(32, 58)), 1024)
    assert SEAM_LETTER_ROWS in str(exc.value)


def test_seam_fail_letter_rows_dim_mock():
    """接缝②破坏之二：输出层不是一张两维的表（拿错了张量）。"""
    with pytest.raises(SeamCheckError) as exc:
        check_letter_rows(torch.zeros(200, 16, 1), list(range(32, 58)), 16)
    assert SEAM_LETTER_ROWS in str(exc.value)


def test_seam_fail_letter_rows_out_of_range_mock():
    """接缝②破坏之三：字母编号越出词表行数（编号与权重对不上号）。"""
    with pytest.raises(SeamCheckError) as exc:
        check_letter_rows(torch.zeros(40, 16), list(range(32, 58)), 16)
    assert SEAM_LETTER_ROWS in str(exc.value)




# 接缝③的两型破坏（模板尾部不符 / 思考开关扳不动）。尾部字符串一律用常量派生，别手打标记。
_THINK_NOT_CLOSED = THINK_OFF_SUFFIX.replace("think", "thought")      # 看着像关闭记号但不是
_THINK_REOPENED = _THINK_NOT_CLOSED + "<" + "|think|" + ">" + chr(10)  # 关闭记号后面又开了个头


def test_seam_fail_think_off_suffix_mock():
    """接缝③破坏：模板尾部对不上 decision 钉死的那截关闭记号（这里换成了思考开场）。"""
    tok = FakeTok(off=_THINK_NOT_CLOSED, on=_THINK_REOPENED)
    with pytest.raises(SeamCheckError) as exc:
        check_think_off(tok, [])
    assert SEAM_THINK_OFF in str(exc.value)
    assert "尾部" in str(exc.value)                 # 消息要说清是尾部对不上


def test_seam_fail_think_switch_noop_mock():
    """接缝③破坏之二：enable_thinking 扳过去一个字节都不变——无法确认思考真被关掉。"""
    same = _TPL_OFF                          # 关闭态尾部本就正确，罪状只剩"开关扳不动"
    tok = FakeTok(off=same, on=same)                # 开启态与关闭态同文：开关是摆设
    with pytest.raises(SeamCheckError) as exc:
        check_think_off(tok, [])
    assert SEAM_THINK_OFF in str(exc.value)
    assert "enable_thinking" in str(exc.value)


def test_seam_fail_mm_id_clash_mock():
    """第四校验破坏：图片记号与字母抢同一个编号（图文装配会把占位当成选项字母）。"""
    ids = list(range(32, 58))
    cfg = _Cfg(image_token_id=ids[0], video_token_id=248057,
               vision_start_token_id=248053, vision_end_token_id=248054)
    special = dict(MM_GOOD)
    special["<image_pad>"] = ids[0]
    with pytest.raises(SeamCheckError) as exc:
        check_mm_token_ids(cfg, FakeTok(special=special), ids)
    assert SEAM_MM_IDS in str(exc.value)
    assert "image_token_id" in str(exc.value)              # 消息要点名是哪个字段抢的号


def test_seam_fail_mm_id_missing_mock():
    """第四校验破坏之二：config 顶层缺多模态字段（视觉塔接口无从登记，p2-08 会撞墙）。"""
    cfg = _Cfg(image_token_id=248056, video_token_id=248057, vision_start_token_id=248053)
    with pytest.raises(SeamCheckError) as exc:
        check_mm_token_ids(cfg, FakeTok(special=dict(MM_GOOD)), list(range(32, 58)))
    assert SEAM_MM_IDS in str(exc.value)
    assert "vision_end_token_id" in str(exc.value)


def test_seam_fail_mm_id_mismatch_mock():
    """第四校验破坏之三：config 里的编号与分词器在册编号不是同一个（两本账对不上）。"""
    cfg = _Cfg(image_token_id=248056, video_token_id=248057,
               vision_start_token_id=248053, vision_end_token_id=248054)
    special = dict(MM_GOOD)
    special["<image_pad>"] = 777
    with pytest.raises(SeamCheckError) as exc:
        check_mm_token_ids(cfg, FakeTok(special=special), list(range(32, 58)))
    assert SEAM_MM_IDS in str(exc.value)
    assert "不一致" in str(exc.value)


def test_seam_fail_mm_id_dup_mock():
    """第四校验破坏之四：两个多模态字段填了同一个编号（分词器侧不再逐一对得上）。"""
    cfg = _Cfg(image_token_id=248056, video_token_id=248056,
               vision_start_token_id=248053, vision_end_token_id=248053)
    with pytest.raises(SeamCheckError) as exc:                      # 在册值与 config 互相矛盾
        check_mm_token_ids(cfg, FakeTok(special=dict(MM_GOOD)), list(range(32, 58)))
    assert SEAM_MM_IDS in str(exc.value)


#: 假模板用的小尾巴：关闭态就是 decision 钉死的那截；开启态在它后面又开了个头
_TPL_OFF = THINK_OFF_SUFFIX
_TPL_ON = THINK_OFF_SUFFIX + "<" + "|think|" + ">" + chr(10)


class _FakeHead:
    """假输出层：只有一张权重重表，供载入路径按行挑字母。"""

    def __init__(self, weight):
        self.weight = weight


class _FakeModel:
    """假模型壳：载入路径只用到 eval / to / 取输出层 / 取文本栈这四样，其余一概不碰。"""

    def __init__(self, head_weight):
        self._head = _FakeHead(head_weight)
        self.model = _Cfg()

    def eval(self):
        return self

    def to(self, device):
        return self

    def get_output_embeddings(self):
        return self._head


def _fake_arch(head_weight):
    """造一个"检查点自述的类名"对应的假类，from_pretrained 直接交出指定输出层的壳。"""
    class _Arch:
        @classmethod
        def from_pretrained(cls, path, **kw):
            return _FakeModel(head_weight)
    return _Arch


class _FakeConfig:
    """假出厂说明单：交出类名与文本栈宽度；多模态编号按需挂在顶层（缺就是缺）。"""

    def __init__(self, hidden, arch="FakeSeamArch"):
        self.architectures = [arch]
        self._hidden = hidden

    def get_text_config(self):
        return _Cfg(hidden_size=self._hidden)


@pytest.mark.parametrize("kind", ["letters", "letter_rows", "think_off"])
def test_seam_fail_load_backbone_rejects_mock(monkeypatch, tmp_path, kind):
    """破坏必须拦在载入路径上：三种接缝各坏一次，load_backbone 一律拒载且不交出窄接口对象。"""
    import transformers
    if kind == "letters":
        tok = FakeTok(backend=FakeBackend(good_vocab(split="Z"), split_for={"Z"}),
                      off=_TPL_OFF, on=_TPL_ON)
        head_weight, seam = torch.zeros(300, 16), SEAM_LETTERS
    elif kind == "letter_rows":
        tok = FakeTok(backend=FakeBackend(good_vocab()), off=_TPL_OFF, on=_TPL_ON)
        head_weight, seam = torch.zeros(300, 8), SEAM_LETTER_ROWS      # 层宽 8 与身板 16 不符
    else:
        tok = FakeTok(backend=FakeBackend(good_vocab()), off=_THINK_NOT_CLOSED, on=_THINK_REOPENED)
        head_weight, seam = torch.zeros(300, 16), SEAM_THINK_OFF

    snap = assets.Snapshot(path=tmp_path, repo="Qwen/Qwen3.5-0.8B", revision="master",
                           source="modelscope")
    monkeypatch.setattr(assets, "fetch_snapshot", lambda name, **kw: snap)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        classmethod(lambda cls, *a, **k: tok))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained",
                        classmethod(lambda cls, *a, **k: _FakeConfig(16)))
    monkeypatch.setattr(transformers, "FakeSeamArch", _fake_arch(head_weight), raising=False)

    with pytest.raises(SeamCheckError) as exc:
        assets.load_backbone("qwen3.5-0.8b")
    assert seam in str(exc.value)


# ==================================================== A4 换脑冒烟（真权重 + 一阶段 decision 程序）
# 三个 typed 问题对应 spec 场景"3 个 typed 问题"；题面用最小题干，跨 run 可比。
QUESTIONS = [
    ("q1", "weather: clear sky at 07:00, humidity 40%",
     {"type": "choice", "instructions": "Which option fits the evidence?",
      "criteria": {"a": "sunny", "b": "rain", "c": "cloudy"}}),
    ("q2", "server status: cpu 12%, memory 30%, disk 45%",
     {"type": "choice", "instructions": "Is the server healthy?",
      "criteria": {"a": "healthy", "b": "degraded", "c": "down"}}),
    ("q3", "review: the code passes all tests and has no lint errors",
     {"type": "choice", "instructions": "What is the review verdict?",
      "criteria": {"a": "approve", "b": "request changes", "c": "comment only", "d": "reject"}}),
]


def _decide(bb, qid, state, spec):
    """走一遍一阶段冻结的决策程序：渲染 → 套壳 → 前向 → 末位字母读出（decision 一个字都不改）。"""
    from sys1.decision import LETTERS, from_systemone, readout
    row = from_systemone(state, spec, qid=qid)
    ids, order = bb.encode_prompt(row)
    hidden = bb.forward(torch.tensor([ids], dtype=torch.long))
    out = readout(hidden, bb.model.get_output_embeddings().weight,
                  list(bb.letter_ids[:len(order)]), lengths=[len(ids)], qtypes=[row["type"]])
    probs = out.probs[0].detach()          # 只取值，别把自动求值的链条拖进测试
    pick = int(probs.argmax())
    return {"qid": qid, "tokens": len(ids), "order": order, "letter": LETTERS[pick],
            "code": order[pick], "probs": [float(v) for v in probs]}


@real_only
def test_swap_brain_three_questions_cpu(bb):
    """换脑冒烟（CPU 实测）：载重后以一阶段程序决策 3 问，份额合法且全程不 import 一阶段 model.py。"""
    before = set(sys.modules)          # 快照差集：断"换脑不新引入一阶段 model"，对执行顺序免疫（全盘跑他域已 import 属合法）
    outs = [_decide(bb, *q) for q in QUESTIONS]
    assert len(outs) == 3
    for o in outs:
        assert len(o["probs"]) == len(o["order"]) >= 2
        assert abs(sum(o["probs"]) - 1.0) < 1e-5, f"{o['qid']} 份额没摊成一锅：{o['probs']}"
        assert o["letter"] in assets.UPPERCASE and o["code"] == o["letter"].lower()
        assert o["tokens"] > 0
    assert "sys1.model" not in set(sys.modules) - before  # 窄接口之外，换脑过程不可新见一阶段 model


@real_only
@pytest.mark.mps
@need_mps
def test_swap_brain_mps_smoke():
    """换脑冒烟（MPS 变体）：同一套程序落 MPS 再走一遍；MPS 被他用占用时按资源纪律跳过。"""
    bb_mps = assets.load_backbone("qwen3.5-0.8b", device="mps")
    try:
        assert bb_mps.device == "mps" and bb_mps.dtype is torch.float16
        out = _decide(bb_mps, *QUESTIONS[0])
        assert abs(sum(out["probs"]) - 1.0) < 1e-5
    finally:
        del bb_mps
        torch.mps.empty_cache()


@real_only
def test_render_alignment_frozen_decision(tok, bb):
    """渲染对齐：编号出自冻结的 decision/render（只读引用）——逐字节同文，版号随 run 走。"""
    from sys1.decision import from_systemone, render
    assert assets.render_messages is render                 # 同一个函数对象，不是第二套排版
    row = from_systemone(assets.CANONICAL_STATE, dict(assets.CANONICAL_ROW_SPEC))
    messages, order = render(row)
    direct = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    prompt, ntok, sha = canonical_prompt(tok)
    assert prompt == direct                                  # 载入路径没另起炉灶改排版
    assert ntok == len(tok.encode(prompt, add_special_tokens=False))
    assert (ntok, sha) == (80, "a1891aca6c002e48")           # 与接缝③的固化快照同值
    ids, order2 = bb.encode_prompt(row)
    assert order2 == order and ids == tok.encode(prompt, add_special_tokens=False)
    assert bb.provenance_config()["render_version"] == RENDER_VERSION


@real_only
def test_narrow_interface_right_pad_causal_invariance(bb):
    """因果不变性（右垫空位对照）：同三行内容，垫位批量送与逐条单行送，读点位上的候选分必须同值同名次。

    这是"窄接口能替一阶段扛批次"的硬指标：垫出来的空位要是漏进了因果，读点位上拿到的
    就不是该行自己的末位，26 行读出全废；所以拿真权重两种送法各算一遍再逐位比。
    """
    from sys1.decision import from_systemone, lengths_from_mask, option_scores
    rows = []
    for qid, state, spec in QUESTIONS:
        ids, _ = bb.encode_prompt(from_systemone(state, spec, qid=qid))
        rows.append(ids)
    pad = bb.tokenizer.pad_token_id
    pad = int(pad) if pad is not None else 0
    width = max(len(r) for r in rows)
    ids_b = torch.full((len(rows), width), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        ids_b[i, :len(r)] = torch.tensor(r)
        mask[i, :len(r)] = 1
    lengths = [int(v) for v in lengths_from_mask(mask)]
    assert lengths == [len(r) for r in rows], f"右垫位下的行长读歪了：{lengths}"

    head = bb.model.get_output_embeddings().weight
    ids_letters = list(bb.letter_ids[:3])
    padded = option_scores(bb.forward(ids_b, attn_mask=mask), head, ids_letters, lengths=lengths)
    single = torch.stack([
        option_scores(bb.forward(torch.tensor([r], dtype=torch.long)), head, ids_letters,
                      lengths=[len(r)])[0]
        for r in rows
    ])
    padded, single = padded.detach(), single.detach()      # 只取数比对，不把自动求导的链条拖进断言
    max_abs = float((padded - single).abs().max())
    rel = max_abs / max(float(single.abs().mean()), 1e-6)
    assert rel < 1e-3, f"垫位漏进因果：最大差 {max_abs:.3e}，相对差 {rel:.3e}"
    assert bool(torch.eq(padded.argmax(dim=1), single.argmax(dim=1)).all()), "两种送法名次不一致"
    assert tuple(bb.forward(ids_b, attn_mask=mask).shape) == (len(rows), width, bb.hidden_size)
