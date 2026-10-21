"""p1-07 决策程序域验收单测（八条孙任务；`-k` 过滤名即 tasks.md 的条目）。

【做什么】
    把"渲染 + 末位字母读出 + 类型温度 + 分组票选"这条两阶段同构桥梁钉死：黄金快照逐
    字节、右补空洞不改概率、任何生成都没有、温度不翻答案、五十选一的分布合法。

【怎么做】
    `-k parse` A1 请求解析与契约接入；`-k render` A2 排版与字母序；`-k snapshot` A3
    逐字节快照 + 确定性；`-k readout` B1 末位读出；`-k pad_invariance` B2 右补空洞
    不变性（以因果桩为主，另配一条真 tiny Decoder 的端到端复核）；`-k no_generate`
    B3 静态门（本域源码禁生成调用，并自证门抓得住真违例）；`-k temperature` B4 温度；
    `-k wide` C1 分组票选。桩件：`_StubTokenizer` 把 26 个大写字母钉在固定编号 1..26
    上（其余字符走确定性散列），`_stub_hidden` 用"本行到该位置为止的符号向量累加"
    当一个纯因果前向——它只考验一件事：读出层取的是不是每行真正的末位。

【为什么】
    被否方案一：全部用例都挂真模型——tiny 档前向确实跑得起，但概率对拍要的是"因果性
    被破坏时测试必须变红"的可控现场，桩件能把这一条做成阳性对照；真模型只留一条端到端
    复核，避免整个域被模型侧的进度卡住（开发期本域与 p1-06 并行）。
    被否方案二：快照写成外部 fixture 文件——多一个需要"记得重新生成"的产物，逐字节
    期望写在测试里，改格式时 diff 直接指着人问"要不要升 RENDER_VERSION"。
    被否方案三：静态门只扫不证——抓不到违规的门等于没设，故每条 grep 门都配构造样本。
"""
from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

import pytest
import torch

from sys1.data.schema import SchemaError
from sys1.decision import (
    GROUP_SIZE,
    KEEP_TOP,
    LETTERS,
    MAX_OPTIONS,
    OTHER_KEY,
    OTHER_TEXT,
    RENDER_VERSION,
    RESIDUAL_SHARE,
    SYSTEM_LINE,
    TEMP_MAX,
    TEMP_MIN,
    UNCALIBRATED,
    Readout,
    RenderError,
    WideError,
    apply_temperature,
    clamp_temperature,
    from_systemone,
    lengths_from_mask,
    option_lines,
    option_order,
    option_scores,
    plan_groups,
    prompt_text,
    readout,
    render,
    subrow,
    temperature_for,
    wide_vote,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DECISION_DIR = REPO_ROOT / "sys1" / "decision"

VOCAB = 256          # 桩词表规模：只需容下字母 + 散列出来的杂字符
PAD_ID = 0           # 与 p1-03 契约一致：<|pad|> 恒为 0 号
WIDTH = 16           # 桩模型的"每位置数字串"宽度


# ---------------------------------------------------------------- 桩件（stub 并行开发的替身）
class _StubTokenizer:
    """字符级桩词表：26 个大写字母钉死在编号 1..26，其余字符按确定性散列落进 [27, VOCAB)。

    只为满足本域对 tokenizer 的两点依赖——能编文本、字母各有唯一编号。真词表由 p1-03 提供
    （`sys1.lang.bpe.check_tokenizer` 保证字母单编号），集成期把这里换成它即可。
    """

    def __init__(self, vocab: int = VOCAB) -> None:
        self.vocab = vocab
        self.letter_ids = list(range(1, 27))

    def encode(self, text: str) -> list[int]:
        ids = [self._one(ch) for ch in text]
        assert all(0 <= i < self.vocab for i in ids), "桩编号越出词表行数"
        return ids

    def _one(self, ch: str) -> int:
        if "A" <= ch <= "Z":
            return ord(ch) - ord("A") + 1
        span = self.vocab - 27
        return 27 + (ord(ch) * 31 % span)


def _table(rows: int, cols: int, seed: int) -> torch.Tensor:
    """定种子造一张固定的"编号 → 向量"表（同 seed 必同值，跨用例可复现）。"""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=gen)


def _stub_hidden(ids: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
    """因果桩前向：位置 t 的数字 = 本行 0..t 的符号向量累加——只看得见左边。

    这正是"右补空洞不该影响读数"的因果模型最小组合；读出层若取错位置（取了最右列），
    在这套桩上立刻会产生偏差，比挂真模型更能把门钉住。
    """
    return torch.cumsum(emb[ids], dim=1)


def _row_choice():
    return from_systemone(
        "今天多云，气温 18 度，体感偏凉。",
        {
            "type": "choice",
            "instructions": "根据证据，今天天气属于哪一类？",
            "criteria": {"A": "晴", "B": "多云", "C": "雨"},
        },
        qid="weather",
    )


def _row_noul():
    return from_systemone(
        "小明说：今天没下雨。",
        {
            "type": "noul",
            "instructions": "这句话与今天下了雨是否矛盾？",
            "criteria": {"false": "不矛盾", "true": "矛盾"},
        },
        qid="noul1",
    )


def _row_score():
    return from_systemone(
        "客服回复很快，但问题没解决。",
        {
            "type": "score",
            "instructions": "这次服务的满意度打几档（0 最低）？",
            "criteria": {"3": "很好", "0": "很差", "1": "一般", "2": "不错"},
        },
        qid="score1",
    )


# ==================== A1 from_systemone：请求解析 + schema 接入（-k parse） ====================


def test_parse_choice_row_shape():
    """choice 请求解析成渲染行：代号沿用 criteria 原键，说明文字一并带过来。"""
    row = _row_choice()
    assert row["id"] == "weather"
    assert row["type"] == "choice"
    assert row["state"] == "今天多云，气温 18 度，体感偏凉。"
    assert row["instructions"] == "根据证据，今天天气属于哪一类？"
    assert [o["id"] for o in row["options"]] == ["A", "B", "C"]
    assert [o["criterion"] for o in row["options"]] == ["晴", "多云", "雨"]
    assert "ordered" not in row, "只有 score 需要声明档位有序"


def test_parse_noul_key_convention():
    """键约定：noul 的候选恒为 {false, true} 两枚，请求没给说明也照样补齐。"""
    row = from_systemone("证据在此", {"type": "noul", "instructions": "是否为真？"})
    assert [o["id"] for o in row["options"]] == ["false", "true"]
    assert option_order(row) == ["false", "true"]
    assert all(o["criterion"] for o in row["options"]), "缺说明时回落成代号本身，渲染不许留空"


def test_parse_noul_accepts_bool_alias_and_criteria():
    """上游偶见 type=bool 的写法：语义同 noul；criteria 里的怪键（True/False）也照收。"""
    row = from_systemone("证据", {"type": "bool", "instructions": "是否为真？", "criteria": {True: "是", False: "否"}})
    assert row["type"] == "noul"
    assert [o["id"] for o in row["options"]] == ["false", "true"]
    assert [o["criterion"] for o in row["options"]] == ["否", "是"]


def test_parse_score_keys_are_zero_based_and_low_to_high():
    """键约定：score 的代号一律重写成 0..n-1，说明文字按原档位数值低→高对齐。

    样本契约只认 "0".."k-1" 这套连续键，请求里写成 2/9/10 这类跳号档位时，本层必须先
    理顺成规范键再交给校验，否则合法请求也会在校验层被判非法。
    """
    row = from_systemone(
        "评价文本",
        {"type": "score", "instructions": "打几档？", "criteria": {"10": "极高", "2": "低", "9": "较高"}},
    )
    assert [o["id"] for o in row["options"]] == ["0", "1", "2"], "代号一律重写成 0..n-1"
    assert [o["criterion"] for o in row["options"]] == ["低", "较高", "极高"], "按原档位数值升序对齐"
    assert row["ordered"] is True
    assert option_order(row) == ["0", "1", "2"]
    # 本来就是 0..n-1 的请求：原样通过，不二次改写
    row2 = from_systemone("评价", {"type": "score", "instructions": "打几档？", "criteria": ["很差", "一般", "很好"]})
    assert [o["id"] for o in row2["options"]] == ["0", "1", "2"]
    assert [o["criterion"] for o in row2["options"]] == ["很差", "一般", "很好"]


def test_parse_criteria_list_form_gets_index_keys():
    """choice 的候选写成裸列表时按下标造代号（0..k-1），说明就是元素本身。"""
    row = from_systemone("s", {"type": "choice", "instructions": "q", "criteria": ["Paris", "London"]})
    assert [o["id"] for o in row["options"]] == ["0", "1"]
    assert [o["criterion"] for o in row["options"]] == ["Paris", "London"]


def test_parse_rejects_illegal_qtype_with_schema_path():
    """非法 qtype 由样本契约拒绝：错误消息必须带字段路径与合法枚举（不另造口径）。"""
    with pytest.raises(SchemaError) as exc:
        from_systemone("s", {"type": "rank", "instructions": "q", "criteria": {"a": "x", "b": "y"}})
    message = str(exc.value)
    assert "questions.q1.type" in message or "questions.q.type" in message, message
    assert "choice" in message and "noul" in message and "score" in message


def test_parse_rejects_single_option_choice():
    """choice 至少要两个候选：契约层就拒掉，本域不放松。"""
    with pytest.raises(SchemaError, match="至少 2 个选项"):
        from_systemone("s", {"type": "choice", "instructions": "q", "criteria": {"only": "一个"}})


def test_parse_requires_criteria_and_instructions():
    """缺候选清单 / 缺问题正文：本层直接拒绝并说清缺的是哪一项。"""
    with pytest.raises(RenderError, match="criteria"):
        from_systemone("s", {"type": "choice", "instructions": "q"})
    with pytest.raises(RenderError, match="instructions"):
        from_systemone("s", {"type": "choice", "criteria": {"a": "x", "b": "y"}})
    with pytest.raises(RenderError, match="dict"):
        from_systemone("s", {"type": "choice", "instructions": "q", "criteria": 3})


def test_parse_empty_state_becomes_placeholder():
    """空证据补成占位串：契约不接受空白 state，渲染侧也不留空行。"""
    for blank in ("", "   ", None):
        row = from_systemone(blank, {"type": "noul", "instructions": "是否为真？"})
        assert row["state"] == "(none)"
        assert "\nEvidence:\n(none)\n" in prompt_text(row)


def test_parse_question_alias_and_qid():
    """兼容 question 写法与自定义 qid：行 id 就是 qid（评测按 qid 对齐打分）。"""
    row = from_systemone("s", {"type": "noul", "question": "是否为真？"}, qid="q7")
    assert row["id"] == "q7"
    assert row["instructions"] == "是否为真？"
    assert from_systemone("s", {"type": "noul", "instructions": "q"}, qid="q7")["id"] == "q7"


# ==================== A2 渲染函数（-k render） ====================


def test_render_block_order_and_lettering():
    """排版骨架：固定系统行 + Evidence + 空行 + Question + Options（字母代号按序发）。"""
    text = prompt_text(_row_choice())
    assert text.startswith(SYSTEM_LINE + "\nEvidence:\n")
    assert "\n\nQuestion: 根据证据，今天天气属于哪一类？\nOptions:\nA) 晴\nB) 多云\nC) 雨" in text
    assert not text.endswith("\n"), "末尾不留换行：读点位就是最后一个字符"
    lines = text.splitlines()
    assert lines[0] == SYSTEM_LINE
    assert [ln[:3] for ln in lines if re.match(r"^[A-Z]\) ", ln)] == ["A) ", "B) ", "C) "]


def test_render_messages_roles():
    """render() 交回 messages 与代号列：role 只有 system/user，order 与字母位一一对应。"""
    row = _row_choice()
    messages, order = render(row)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_LINE
    assert order == ["A", "B", "C"]
    assert order[0] == LETTERS[0], "第 0 个候选永远拿 A"
    with pytest.raises(RenderError, match="排列"):
        render(row, order=["A", "B"])
    with pytest.raises(RenderError, match="排列"):
        render(row, order=["A", "B", "D"])


def test_render_noul_shows_yes_no():
    """noul 的正文用 yes/no 出人话，字母位仍按 false→true 的键序发。"""
    text = prompt_text(_row_noul())
    assert "A) no: 不矛盾" in text and "B) yes: 矛盾" in text
    assert option_order(_row_noul()) == ["false", "true"]


def test_render_bare_choice_ids_hide_the_id():
    """代号本身就是光杆字母时隐去它：不许出现 A) A: 晴 这种让答案字母含糊的写法。"""
    text = prompt_text(_row_choice())
    assert "A) A" not in text
    loose = from_systemone("s", {"type": "choice", "instructions": "q", "criteria": {"A": None, "B": "伦敦"}})
    assert option_lines(loose, option_order(loose)) == ["A) A", "B) 伦敦"], "缺说明的项回落成代号本身"


def test_render_refuses_over_26_options():
    """超过 26 个候选拒绝整轮渲染，并指路 wide 分组票选。"""
    big = from_systemone("s", {"type": "choice", "instructions": "q", "criteria": {f"k{i}": f"说明{i}" for i in range(27)}})
    with pytest.raises(RenderError, match="26"):
        prompt_text(big)
    assert len(big["options"]) == 27


def test_render_subrow_appends_other_slot_last():
    """分组裁出的子行把残差槽排在最后一个字母位，正文用固定话术出人。"""
    row = _row_choice()
    group = subrow(row, ["A", "C"], other=True)
    assert option_order(group) == ["A", "C", OTHER_KEY]
    lines = option_lines(group, option_order(group))
    assert lines[-1] == f"C) {OTHER_TEXT}"
    with pytest.raises(RenderError, match="不存在"):
        subrow(row, ["A", "ZZ"])


def test_render_options_count_matches_letters():
    """候选数 = 字母行数 = 渲染行数：三处任何一处对不上，读出的列就挂错代号。"""
    for row in (_row_choice(), _row_noul(), _row_score()):
        _, order = render(row)
        lines = option_lines(row, order)
        assert len(lines) == len(row["options"]) == len(order) <= MAX_OPTIONS
        assert [ln[0] for ln in lines] == list(LETTERS[: len(lines)])


def test_render_score_levels_are_low_to_high_in_letters():
    """score 档位在字母序里保持低→高：A 是最低档，末位字母是最高档。"""
    _, order = render(_row_score())
    assert order == ["0", "1", "2", "3"]
    assert "A) 0: 很差" in prompt_text(_row_score())


# ==================== A3 黄金快照（逐字节）+ 确定性（-k snapshot） ====================

SNAPSHOT_CHOICE = (
    "Apply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only.\n"
    "Evidence:\n"
    "今天多云，气温 18 度，体感偏凉。\n"
    "\n"
    "Question: 根据证据，今天天气属于哪一类？\n"
    "Options:\n"
    "A) 晴\n"
    "B) 多云\n"
    "C) 雨"
)

SNAPSHOT_NOUL = (
    "Apply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only.\n"
    "Evidence:\n"
    "小明说：今天没下雨。\n"
    "\n"
    "Question: 这句话与今天下了雨是否矛盾？\n"
    "Options:\n"
    "A) no: 不矛盾\n"
    "B) yes: 矛盾"
)

SNAPSHOT_SCORE = (
    "Apply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only.\n"
    "Evidence:\n"
    "客服回复很快，但问题没解决。\n"
    "\n"
    "Question: 这次服务的满意度打几档（0 最低）？\n"
    "Options:\n"
    "A) 0: 很差\n"
    "B) 1: 一般\n"
    "C) 2: 不错\n"
    "D) 3: 很好"
)

SNAPSHOTS = {
    "choice(3 选项)": (SNAPSHOT_CHOICE, _row_choice),
    "noul": (SNAPSHOT_NOUL, _row_noul),
    "score(4 档)": (SNAPSHOT_SCORE, _row_score),
}


@pytest.mark.parametrize("label", list(SNAPSHOTS))
def test_snapshot_is_byte_exact(label: str):
    """黄金快照逐字节对拍：改一个字也要显式升 RENDER_VERSION 并同步改这里。"""
    expected, make = SNAPSHOTS[label]
    got = prompt_text(make())
    assert got == expected, f"渲染漂移（{label}）\n--- 期望 ---\n{expected}\n--- 实得 ---\n{got}"
    assert got.encode("utf-8") == expected.encode("utf-8"), "快照以 utf-8 字节为准"


def test_snapshot_render_version_constant():
    """RENDER_VERSION 是常量且随快照一起报：格式改动必须升版，老数字不许复用。"""
    assert isinstance(RENDER_VERSION, str) and RENDER_VERSION.strip()
    assert RENDER_VERSION.startswith("dmlaya_render_")
    assert prompt_text(_row_choice()) == SNAPSHOT_CHOICE, "当前版号对应的就是这份快照"


def test_snapshot_is_deterministic_across_calls_and_dict_order():
    """确定性：同一请求渲染两次逐字相同；候选在请求里的书写顺序不同也不影响输出。"""
    first = prompt_text(_row_choice())
    second = prompt_text(_row_choice())
    assert first == second
    shuffled = from_systemone(
        "今天多云，气温 18 度，体感偏凉。",
        {"type": "choice", "instructions": "根据证据，今天天气属于哪一类？", "criteria": {"C": "雨", "A": "晴", "B": "多云"}},
        qid="weather",
    )
    assert prompt_text(shuffled) == first, "字母序由渲染层决定，不由请求里的书写顺序决定"


def test_snapshot_no_timestamp_or_random_source_in_render():
    """渲染源码不许引入时间/随机源：那是确定性的头号杀手，静态钉住。"""
    banned = ("random", "datetime", "time.", "uuid")
    for path in sorted(DECISION_DIR.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        text = _code_only(src)
        for token in banned:
            assert token not in text, f"{path.name} 出现 {token!r}（渲染必须与时间/随机无关）"
        assert "import random" not in text and "import time" not in text
        assert isinstance(tree, ast.Module)  # 只为让"能解析"这条显式成立


# ==================== B1 末位字母读出（-k readout） ====================


def _stub_pair(texts: list[str], emb_seed: int = 11, head_seed: int = 12):
    """把若干段文本编成右补空洞的一批，返回 (编号, 掩码, 每行长度, 桩前向数字, 输出层权重)。"""
    tok = _StubTokenizer()
    rows = [tok.encode(t) for t in texts]
    longest = max(len(r) for r in rows)
    ids = torch.full((len(rows), longest), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(rows), longest), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        mask[i, : len(r)] = 1
    hidden = _stub_hidden(ids, _table(VOCAB, WIDTH, emb_seed))
    weight = _table(VOCAB, WIDTH, head_seed)
    lengths = [len(r) for r in rows]
    return ids, mask, lengths, hidden, weight, tok


def test_readout_scores_match_manual_dot_product():
    """分数 = 每行末位数字 × 字母行：与手算逐行对拍，形状 (B, k)。"""
    texts = [prompt_text(_row_choice()), prompt_text(_row_noul())]
    _, _, lengths, hidden, weight, tok = _stub_pair(texts)
    scores = option_scores(hidden, weight, tok.letter_ids, lengths)
    assert scores.shape == (2, 26) and scores.dtype == torch.float32
    for b, n in enumerate(lengths):
        h = hidden[b, n - 1].float()
        manual = [float(h @ weight[i].float()) for i in tok.letter_ids]
        assert torch.allclose(scores[b], torch.tensor(manual), atol=1e-6), f"第 {b} 行末位取错了"


def test_readout_probs_sum_to_one_and_pick_max_letter():
    """份额每行和为 1；答案就是分到最多的那个字母（无生成、无第二个候选源）。"""
    _, _, lengths, hidden, weight, tok = _stub_pair([prompt_text(_row_choice())])
    out: Readout = readout(hidden, weight, tok.letter_ids, lengths, qtypes="choice")
    assert out.probs.shape == (1, 26)
    assert abs(float(out.probs.sum()) - 1.0) <= 1e-6
    assert torch.equal(out.probs.argmax(dim=-1), out.scores.argmax(dim=-1)), "折份额不许改变名次"
    assert out.temperatures == (1.0,) and out.uncalibrated == (True,)
    assert isinstance(out.to_list(), list) and len(out.to_list()[0]) == 26


def test_readout_uses_last_real_position_not_tensor_end():
    """读点位是 lengths-1 而不是张量最右列：单独一行与批内补空洞后的同一行必须同值。"""
    solo_text = prompt_text(_row_choice())
    _, _, lengths, hidden, weight, tok = _stub_pair([solo_text, solo_text + "更长的尾巴" * 4])
    picked = option_scores(hidden, weight, tok.letter_ids, lengths)
    wrong = option_scores(hidden, weight, tok.letter_ids)  # lengths=None → 取最右列
    assert not torch.allclose(picked[0], wrong[0], atol=1e-9), "桩件没造出差异，这组对照无效"
    solo_hidden = _stub_hidden(torch.tensor([tok.encode(solo_text)]), _table(VOCAB, WIDTH, 11))
    alone = option_scores(solo_hidden, weight, tok.letter_ids, [len(tok.encode(solo_text))])
    assert torch.allclose(picked[0], alone[0], atol=1e-6)


def test_readout_length_and_id_validation():
    """入参体检：长度出格、编号越界/重复/超 26、形状不匹配一律当场报错。"""
    _, _, lengths, hidden, weight, tok = _stub_pair([prompt_text(_row_choice())])
    with pytest.raises(ValueError, match="长度"):
        option_scores(hidden, weight, tok.letter_ids, [0])
    with pytest.raises(ValueError, match="出格"):
        option_scores(hidden, weight, tok.letter_ids, [lengths[0] + 5])
    with pytest.raises(ValueError, match="批大小"):
        option_scores(hidden, weight, tok.letter_ids, [lengths[0], lengths[0]])
    with pytest.raises(ValueError, match="越出"):
        option_scores(hidden, weight, [VOCAB, 1], lengths)
    with pytest.raises(ValueError, match="两两不同"):
        option_scores(hidden, weight, [1, 1, 2], lengths)
    with pytest.raises(ValueError, match="为空"):
        option_scores(hidden, weight, [], lengths)
    with pytest.raises(ValueError, match="最多"):
        option_scores(hidden, weight, list(range(1, 28)), lengths)
    with pytest.raises(ValueError, match="三维"):
        option_scores(torch.zeros(2, 3), weight, tok.letter_ids, [2])       # 少了 T 维
    with pytest.raises(ValueError, match="两维"):
        option_scores(hidden, weight.unsqueeze(0), tok.letter_ids, lengths)  # 权重多出一批维
    with pytest.raises(TypeError, match="head_weight"):
        option_scores(hidden, [1.0, 2.0], tok.letter_ids, lengths)           # 不是张量当场拦
    with pytest.raises(ValueError, match="不符"):
        option_scores(hidden, weight.T, tok.letter_ids, lengths)            # 权重方向摆反
    with pytest.raises(ValueError, match="不符"):
        option_scores(hidden[..., :8], weight, tok.letter_ids, lengths)      # 逐位数字的宽度对不上


def test_readout_touches_only_hidden_and_head_weight():
    """窄接口自证：读出层只吃这两样，本域源码不 import 模型内部（父 design D1）。"""
    hidden = torch.zeros(1, 4, WIDTH)
    weight = torch.zeros(8, WIDTH)
    out = readout(hidden, weight, [1, 2, 3], [4], qtypes="choice")
    assert out.scores.shape == (1, 3) and torch.equal(out.probs, torch.full((1, 3), 1 / 3))
    for path in sorted(DECISION_DIR.glob("*.py")):
        code = _code_only(path.read_text(encoding="utf-8"))
        assert "sys1.model" not in code, f"{path.name} 引到了模型内部"
        assert not re.search(r"\bfrom sys1\.model\b", code), f"{path.name} 导入了 sys1.model"
        assert "Decoder(" not in code and "Decoder." not in code, f"{path.name} 直接构造/使用了模型对象"


def test_readout_lengths_from_mask():
    """掩码换算长度：1 的个数就是该行真符号数；全零行直接报错（没有可读位置）。"""
    _, mask, lengths, _, _, _ = _stub_pair([prompt_text(_row_choice()), prompt_text(_row_noul())])
    assert lengths_from_mask(mask) == lengths
    with pytest.raises(ValueError, match="整行"):
        lengths_from_mask(torch.zeros((2, 5), dtype=torch.long))
    with pytest.raises(ValueError, match="两维"):
        lengths_from_mask(torch.ones((3, 4, 5)))


# ==================== B2 右 padding 因果不变性（-k pad_invariance） ====================


def test_pad_invariance_stub_causal_batch_matches_solo():
    """spec「右 padding 不变性」：同一样本单独前向 vs 批内右补空洞前向，份额最大偏差 ≤1e-5。"""
    texts = [prompt_text(_row_choice()), prompt_text(_row_noul()), prompt_text(_row_score())]
    emb = _table(VOCAB, WIDTH, 11)
    weight = _table(VOCAB, WIDTH, 12)
    tok = _StubTokenizer()
    qtypes = ["choice", "noul", "score"]

    batch_out = {}
    ids, mask, _, _, _, _ = _stub_pair(texts)
    hidden = _stub_hidden(ids, emb)
    batched = readout(hidden, weight, tok.letter_ids, lengths_from_mask(mask), qtypes=qtypes)
    for i, t in enumerate(texts):
        solo_ids = torch.tensor([tok.encode(t)], dtype=torch.long)
        solo = readout(_stub_hidden(solo_ids, emb), weight, tok.letter_ids, [solo_ids.shape[1]], qtypes=qtypes[i])
        diff = float((solo.probs[0] - batched.probs[i]).abs().max())
        assert diff <= 1e-5, f"第 {i} 行批内与单算差 {diff}：末位取错或桩件非因果"
        batch_out[t] = solo.probs[0]
    assert len(batch_out) == 3


def test_pad_invariance_positive_control_catches_wrong_slot():
    """阳性对照：把读点位换成张量最右列（补空洞位），偏差必须显著——门不是假门。"""
    emb = _table(VOCAB, WIDTH, 11)
    weight = _table(VOCAB, WIDTH, 12)
    tok = _StubTokenizer()
    # 第 0 行取最短的 noul：它在批内一定被补了空洞，读错点位才会露馅
    texts = [prompt_text(_row_noul()), prompt_text(_row_choice()), prompt_text(_row_score())]
    ids, mask, lengths, hidden, _, _ = _stub_pair(texts)
    assert lengths[0] < ids.shape[1], "第 0 行没被补空洞时这组对照无效"
    assert float((mask[0] - 1).abs().sum()) > 0
    right = readout(hidden, weight, tok.letter_ids, lengths, qtypes="noul")
    wrong = readout(hidden, weight, tok.letter_ids, None, qtypes="noul")   # 等价于读最右列
    drift = float((right.probs[0] - wrong.probs[0]).abs().max())
    assert drift > 1e-3, f"读错位与读对只差 {drift}：桩件没造出差异，阳性对照失效"
    # 反向自查：按 lengths 取位时，批内结果与单独前向必须咬合（这才叫不变性）
    solo_ids = torch.tensor([tok.encode(texts[0])], dtype=torch.long)
    solo = readout(_stub_hidden(solo_ids, emb), weight, tok.letter_ids, [solo_ids.shape[1]], qtypes="noul")
    assert float((solo.probs[0] - right.probs[0]).abs().max()) <= 1e-5


def test_pad_invariance_real_decoder_tiny():
    """端到端复核（真模型 tiny 档）：p1-06 的因果 decoder 与本域读出接起来同样不变。"""
    from sys1.model import Decoder, ModelConfig

    torch.manual_seed(0)
    cfg = ModelConfig(d=64, L=2, heads=4, ctx=512, vocab=VOCAB, seed=5)
    model = Decoder(cfg).eval()
    weight = model.lm_head.weight
    tok = _StubTokenizer()
    texts = [prompt_text(_row_choice()), prompt_text(_row_score()), prompt_text(_row_noul())]
    lengths = [len(tok.encode(t)) for t in texts]
    longest = max(lengths)
    ids = torch.full((len(texts), longest), PAD_ID, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, t in enumerate(texts):
        row = torch.tensor(tok.encode(t), dtype=torch.long)
        ids[i, : row.numel()] = row
        mask[i, : row.numel()] = 1
    assert longest + 8 <= cfg.ctx
    padded = torch.nn.functional.pad(ids, (0, 8))            # 再多补 8 个洞，长度不一致也要扛住
    padded_mask = torch.nn.functional.pad(mask, (0, 8))
    qtypes = ["choice", "score", "noul"]
    with torch.no_grad():                                    # 读出是推理路径，不该拖着一张计算图
        batch_hidden = model(padded, padded_mask)
        for i, t in enumerate(texts):
            solo_ids = torch.tensor([tok.encode(t)], dtype=torch.long)
            solo_hidden = model(solo_ids, torch.ones_like(solo_ids))
            solo = readout(solo_hidden, weight, tok.letter_ids, [solo_ids.shape[1]], qtypes=qtypes[i])
            batch = readout(batch_hidden, weight, tok.letter_ids,
                            lengths_from_mask(padded_mask), qtypes=qtypes[i])
            diff = float((solo.probs[0] - batch.probs[i]).abs().max())
            assert diff <= 1e-5, f"真模型第 {i} 行批内补洞后份额漂移 {diff}"


def test_pad_invariance_probs_stay_finite_with_long_padding():
    """补得很长也不许出 NaN/Inf：份额行行和为 1。"""
    emb = _table(VOCAB, WIDTH, 21)
    weight = _table(VOCAB, WIDTH, 22)
    tok = _StubTokenizer()
    short = tok.encode(prompt_text(_row_noul()))
    ids = torch.tensor([short], dtype=torch.long)
    for pad in (1, 7, 64, 257):
        padded = torch.full((1, len(short) + pad), PAD_ID, dtype=torch.long)
        padded[0, : len(short)] = ids[0]
        out = readout(_stub_hidden(padded, emb), weight, tok.letter_ids, [len(short)], qtypes="noul")
        assert torch.isfinite(out.probs).all() and torch.isfinite(out.scores).all()
        assert abs(float(out.probs.sum()) - 1.0) <= 1e-6


# ==================== B3 无生成循环静态断言（-k no_generate） ====================

_GENERATE_CALL = re.compile(r"\.\s*generate\s*\(|\bgenerate\s*\(")


def _code_only(src: str) -> str:
    """抹掉注释、docstring 与字符串字面量，只留代码骨架（文档里举例不算违例）。"""
    out: list[str] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError:
        tokens = []
    for tok in tokens:
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            out.append("\n" * (tok.end[0] - tok.start[0]))
            continue
        out.append(tok.string)
    return " ".join(out)


def _scan_no_generate(src: str) -> list[str]:
    """扫一份源码里的"生成/解码循环"调用：`.generate(` 与裸 `generate(` 都算违例。"""
    code = _code_only(src)
    return [f"出现生成调用 {m.group(0)!r}" for m in _GENERATE_CALL.finditer(code)]


def test_no_generate_absent_from_decision_sources():
    """spec「无生成断言」：decision/ 全部源码不含任何生成调用（output_tokens = 0 红线）。"""
    files = sorted(DECISION_DIR.glob("*.py"))
    assert files, "本域源码没落地，门无从校验"
    violations: list[str] = []
    for path in files:
        violations += [f"{path.name}: {v}" for v in _scan_no_generate(path.read_text(encoding="utf-8"))]
    assert not violations, "存在生成循环：\n  " + "\n  ".join(violations)


def test_no_generate_gate_catches_planted_violation():
    """自证门不是假门：代码里真写一句生成调用必须被抓到，注释/docstring 里举例不许被抓。"""
    bad = "def predict(body):\n    return body.generate(x)\n"
    assert _scan_no_generate(bad), "门抓不到真违例"
    bad2 = "def predict(body):\n    return body . generate (x)\n"
    assert _scan_no_generate(bad2), "带空格的写法也要抓到"
    prose = 'def predict(body):\n    """这里不许调 generate(body.generate(...))，只是说明。"""\n    return body.readout()\n'
    assert _scan_no_generate(prose) == [], f"文档里的举例被误判：{_scan_no_generate(prose)}"


def test_no_generate_readout_is_one_pass_forward():
    """读出的取值面：只走一次前向的产物（逐位置数字 × 字母行），结果里没有第二候选源。"""
    _, _, lengths, hidden, weight, tok = _stub_pair([prompt_text(_row_choice())])
    out = readout(hidden, weight, tok.letter_ids, lengths, qtypes="choice")
    assert out.probs.shape == (1, len(tok.letter_ids))
    assert not hasattr(out, "text") and not hasattr(out, "tokens"), "读出结果不许含生成文本"
    again = readout(hidden, weight, tok.letter_ids, lengths, qtypes="choice")
    assert torch.equal(out.probs, again.probs), "同输入必须逐位相同（无随机采样）"


# ==================== B4 类型温度（-k temperature） ====================


def test_temperature_clamp_bounds():
    """合法倍数夹进 [0.2, 5]：越界贴边界，界内原样，边界本身与 spec 一致。"""
    assert (TEMP_MIN, TEMP_MAX) == (0.2, 5.0)
    assert clamp_temperature(1.0).value == 1.0
    assert clamp_temperature(0.05).value == TEMP_MIN
    assert clamp_temperature(99.0).value == TEMP_MAX
    assert clamp_temperature(TEMP_MIN).calibrated and clamp_temperature(TEMP_MAX).calibrated
    assert clamp_temperature("2.5").value == 2.5, "json 读回来的字符串倍数也要能用"


def test_temperature_rejects_non_numeric_and_bool():
    """非数字 / 布尔 / None / NaN / Inf / 负数：退成 1.0 并标注未标定，绝不静默生效。"""
    import math

    for bad in (True, False, None, "abc", float("nan"), float("inf"), -1.0, 0.0, [2.0]):
        decided = clamp_temperature(bad)
        assert decided.value == 1.0 and decided.uncalibrated, f"{bad!r} 应退成 1.0 未标定"
    assert not math.isnan(clamp_temperature(float("nan")).value)


def test_temperature_missing_table_is_1_and_uncalibrated():
    """缺表 / 缺该 qtype / 表不是映射：一律 T=1.0 且 label = uncalibrated（写进配置如实报）。"""
    assert temperature_for("choice", None).label == UNCALIBRATED
    assert temperature_for("choice", {}).uncalibrated
    assert temperature_for("choice", {"noul": 2.0}).uncalibrated
    got = temperature_for("choice", {"choice": 3.0})
    assert not got.uncalibrated and got.value == 3.0
    assert temperature_for("choice", {"choice": "bad"}).uncalibrated, "表里写坏值也算没标定"
    assert temperature_for(None, {"choice": 2.0}).uncalibrated


def _tiers(vec: list[float]) -> list[list[int]]:
    """把一列分数折成"从高到低的档位表"，同分的下标并在一档里。

    白话：先按数值把人分成几条线（同分的算同一条线），再按线的高低排好；名次与并列
    关系就全写在这一张表里，缩放前后一比就知道有没有走样。
    """
    buckets: dict[float, list[int]] = {}
    for i, v in enumerate(vec):
        buckets.setdefault(round(float(v), 9), []).append(i)
    return [buckets[k] for k in sorted(buckets, reverse=True)]


def test_temperature_preserves_argmax_and_ties():
    """spec「温度不改 argmax」：遍历整条网格，名次与并列结构与 T=1 时完全一致。"""
    grid = [TEMP_MIN + i * 0.05 for i in range(97)] + [1.0, TEMP_MAX]
    cases = [
        [1.0, 3.0, 2.0],
        [0.0, 0.0, 0.0],                      # 全体并列
        [2.0, -1.0, 2.0, 0.5],                # 含并列头名
        [-7.0, -3.0, -5.0, -3.0],             # 负分区间
    ]
    for case in cases:
        base = torch.tensor([case])
        tiers_ref = _tiers(case)
        for t in grid:
            scaled = apply_temperature(base, "choice", {"choice": t})
            assert _tiers(scaled[0].tolist()) == tiers_ref, f"T={t} 名次或并列被改动"
            assert torch.allclose(scaled[0] * t, base[0], atol=1e-6), f"T={t} 只是同除一个正数"
            probs = torch.softmax(scaled.float(), dim=-1)
            assert abs(float(probs.sum()) - 1.0) <= 1e-6
            assert int(probs.argmax()) == int(base[0].argmax()), f"T={t} 翻了头名"


def test_temperature_applies_per_row_and_keeps_dtype():
    """逐行不同 qtype 各按各的倍数；1-D 视为单行；形状与 dtype 原样保留。"""
    scores = torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.float64)
    table = {"choice": 2.0, "noul": 4.0}
    scaled = apply_temperature(scores, ["choice", "noul"], table)
    assert scaled.shape == scores.shape and scaled.dtype == torch.float64
    assert torch.allclose(scaled[0], scores[0] / 2.0) and torch.allclose(scaled[1], scores[1] / 4.0)
    one_d = apply_temperature(torch.tensor([1.0, 2.0], dtype=torch.float32), "choice", table)
    assert one_d.dim() == 1 and torch.allclose(one_d, torch.tensor([0.5, 1.0], dtype=torch.float32))
    with pytest.raises(ValueError, match="行数"):
        apply_temperature(scores, ["choice"], table)
    with pytest.raises(TypeError):
        apply_temperature([[1.0, 2.0]], "choice", table)


def test_temperature_readout_reports_uncalibrated_rows():
    """readout 把"这行按了几倍、是不是调过"一并交回，报告不许把未标定混进已标定。"""
    _, _, lengths, hidden, weight, tok = _stub_pair([prompt_text(_row_choice()), prompt_text(_row_noul())])
    out = readout(hidden, weight, tok.letter_ids, lengths, qtypes=["choice", "noul"],
                  table={"choice": 2.0, "noul": 0.1})
    assert out.temperatures == (2.0, TEMP_MIN), "0.1 越界 → 贴下界 0.2"
    assert out.uncalibrated == (False, False)
    plain = readout(hidden, weight, tok.letter_ids, lengths, qtypes=["choice", "noul"], table={"choice": 2.0})
    assert plain.uncalibrated == (False, True) and plain.temperatures == (2.0, 1.0)
    assert torch.allclose(plain.probs[0], torch.softmax(out.scores[0] / 2.0, dim=-1), atol=1e-7)
    assert torch.allclose(plain.probs[1], torch.softmax(out.scores[1], dim=-1), atol=1e-7)


def test_temperature_bounds_match_calibrate_grid():
    """契约同源：本域的夹界区间 MUST 等于 p1-05 拟合网格的边界（漂移就有一头先红）。"""
    from sys1.calibrate import GRID_HI, GRID_LO

    assert TEMP_MIN == GRID_LO and TEMP_MAX == GRID_HI, (TEMP_MIN, TEMP_MAX, GRID_LO, GRID_HI)


# ==================== C1 >26 分组票选（-k wide） ====================


def _biased_vote(strength: dict[str, float], *, other_mass: float = 0.05):
    """造一个"一轮问答"：份额正比于预设强度，残差槽固定吃掉 other_mass 的质量。"""

    def vote(candidates: list[str]) -> list[float]:
        weights = []
        for name in candidates:
            weights.append(other_mass if name == OTHER_KEY else max(strength.get(name, 0.0), 0.0))
        total = sum(weights)
        if total <= 0.0:
            return [1.0 / len(candidates)] * len(candidates)
        return [w / total for w in weights]

    return vote


def test_wide_groups_are_near_equal_and_within_limit():
    """分组：组数最少、组间长度差不超 1，且每组真实候选 ≤ 25（第 26 个字母位留给残差槽）。"""
    for n in (26, 27, 49, 50, 51, 101):
        keys = [f"k{i:03d}" for i in range(n)]
        groups = plan_groups(keys)
        assert all(len(g) <= GROUP_SIZE for g in groups), keys
        assert sum(len(g) for g in groups) == n
        assert max(len(g) for g in groups) - min(len(g) for g in groups) <= 1
        assert [k for g in groups for k in g] == keys, "分组不许打乱原序、不许丢候选"


def test_wide_k50_distribution_is_legal():
    """spec「五十选一分布合法」：恰好 50 个概率、和为 1（容差 1e-6）、项项非负。"""
    keys = [f"k{i:02d}" for i in range(50)]
    strength = {k: (i % 7) + 1 for i, k in enumerate(keys)}
    out = _run_wide(keys, strength)
    assert len(out) == 50 and list(out) == keys, "键数与键序都要跟候选列一致"
    assert abs(sum(out.values()) - 1.0) <= 1e-6
    assert all(v >= 0.0 for v in out.values())
    top = max(out, key=lambda k: out[k])
    assert out[top] > min(out.values()), "分布不许摊平成一锅粥"


def test_wide_final_round_follows_letter_order():
    """决赛轮收到的候选列必须按字母序（与首轮裁出前的原序一致），组内序不许由名次决定。"""
    keys = [f"k{i:02d}" for i in range(50)]
    strength = {k: (i % 7) + 1 for i, k in enumerate(keys)}
    seen: list[list[str]] = []

    def spy(candidates: list[str]) -> list[float]:
        seen.append(list(candidates))
        return _biased_vote(strength)(candidates)

    wide_vote(keys, spy)
    finals = seen[-1]
    assert finals == sorted(finals, key=keys.index), f"决赛组内序不是字母序：{finals}"
    assert len(finals) == len(set(finals)) and len(finals) <= MAX_OPTIONS
    first_rounds = seen[: len(plan_groups(keys))]
    assert all(g[-1] == OTHER_KEY and len(g) <= MAX_OPTIONS for g in first_rounds)


def test_wide_losers_keep_residual_share():
    """落选者不为零：决赛拿 1-residual，落选者按首轮份额比例分 residual。"""
    keys = [f"k{i:02d}" for i in range(50)]
    strength = {k: (i % 7) + 1 for i, k in enumerate(keys)}
    out = _run_wide(keys, strength)
    finalists = _finalist_set(keys, strength)
    assert all(out[k] > 0.0 for k in keys if k not in finalists), "落选者被抹成零是路线错误"
    losers = [out[k] for k in keys if k not in finalists]
    assert max(losers) < min(out[k] for k in finalists), "留量不该盖过决赛份额"
    scale = RESIDUAL_SHARE * KEEP_TOP * 2  # 量级上界，仅用来确认"小而不零"
    assert max(losers) <= scale


def test_wide_other_slot_mass_is_spread_back():
    """残差槽吃掉的质量按比例摊回组内候选：组内相对结构不变，全局和恒为 1。"""
    keys = [f"k{i:02d}" for i in range(30)]
    strength = {k: 1.0 for k in keys}          # 组内谁都不分高下
    out = _run_wide(keys, strength, other_mass=0.4)
    assert abs(sum(out.values()) - 1.0) <= 1e-9, "残差槽的份额被吞掉就会小于 1"
    assert all(v > 0.0 for v in out.values())
    finalists = _finalist_set(keys, strength)
    fin = {round(out[k], 12) for k in finalists}
    los = {round(out[k], 12) for k in keys if k not in finalists}
    assert len(fin) == 1 and len(los) == 1, "组内等强时摊回不许打破对称"
    assert next(iter(fin)) > next(iter(los)), "决赛份额必须盖过落选留量"

    # 整组质量都落进残差槽（模型完全拒绝表态）：退化成组内平均，既不除零也不留空组
    all_other = wide_vote(keys, _biased_vote({}, other_mass=1.0))
    assert abs(sum(all_other.values()) - 1.0) <= 1e-9
    assert all(v > 0.0 for v in all_other.values())
    assert len({round(v, 12) for v in all_other.values()}) == 2, "只该有决赛/落选两档份额"


def test_wide_recurses_when_finalists_exceed_group():
    """决赛仍超一组上限时递归票选：结果依旧 50 项、和为 1、非负。"""
    keys = [f"k{i:02d}" for i in range(50)]
    strength = {k: (i % 11) + 1 for i, k in enumerate(keys)}
    seen: list[list[str]] = []

    def spy(candidates: list[str]) -> list[float]:
        seen.append(list(candidates))
        return _biased_vote(strength)(candidates)

    out = wide_vote(keys, spy, keep=KEEP_TOP, group_size=4)
    assert len(out) == 50 and abs(sum(out.values()) - 1.0) <= 1e-6
    assert all(v >= 0.0 for v in out.values())
    assert len(seen) > len(plan_groups(keys, 4)) + 1, "没有多轮就说明递归没发生"


def test_wide_rejects_bad_votes_and_keys():
    """票选前提不成立要当场拒绝：候选重复、回值长度不符、份额非法、决赛无法收敛。"""
    with pytest.raises(WideError, match="重复"):
        wide_vote(["a", "a", "b", "c"], _biased_vote({}))
    with pytest.raises(WideError, match="为空"):
        wide_vote([], _biased_vote({}))
    with pytest.raises(WideError, match="对不上"):
        wide_vote([f"k{i}" for i in range(30)], lambda cands: [1.0])
    with pytest.raises(WideError, match="非法"):
        wide_vote([f"k{i}" for i in range(30)], lambda cands: [-1.0] * len(cands))
    with pytest.raises(WideError, match="非数字|不是数字"):
        wide_vote([f"k{i}" for i in range(30)], lambda cands: ["x"] * len(cands))
    with pytest.raises(WideError, match="无法收敛"):
        wide_vote([f"k{i}" for i in range(30)], _biased_vote({}), group_size=1, keep=3)
    with pytest.raises(WideError, match="residual"):
        wide_vote([f"k{i}" for i in range(30)], _biased_vote({}), residual=1.0)


def test_wide_end_to_end_with_stub_readout():
    """链路自证：渲染子行 → 桩前向 → 字母读出 → 票选，30 个候选一趟跑通且和为 1。"""
    row = from_systemone(
        "某城市今天的情况。",
        {"type": "choice", "instructions": "这座城市最可能是哪一座？",
         "criteria": {f"city{i:02d}": f"城市{i}" for i in range(30)}},
        qid="city",
    )
    keys = option_order(row)
    emb = _table(VOCAB, WIDTH, 31)
    weight = _table(VOCAB, WIDTH, 32)
    tok = _StubTokenizer()
    seen: list[list[str]] = []

    def spy(candidates: list[str]) -> list[float]:
        """一轮问答：按字母位裁出子行渲染，走一次桩前向读出，回一列份额。"""
        seen.append(list(candidates))
        sub = subrow(row, [c for c in candidates if c != OTHER_KEY], other=OTHER_KEY in candidates)
        ids = torch.tensor([tok.encode(prompt_text(sub))], dtype=torch.long)
        out = readout(_stub_hidden(ids, emb), weight, tok.letter_ids[: len(candidates)],
                      [ids.shape[1]], qtypes="choice")
        return out.probs[0].tolist()

    out = wide_vote(keys, spy)
    assert len(out) == 30 and list(out) == keys, "票选结果必须一一对应回候选原序"
    assert abs(sum(out.values()) - 1.0) <= 1e-6
    assert all(v >= 0.0 for v in out.values())
    assert len(seen) == len(plan_groups(keys)) + 1, "两轮制：每组一轮 + 决赛一轮"
    assert seen[-1] == sorted(seen[-1], key=keys.index), "决赛轮收到的候选列必须按字母序"
    # 桩前向的分差极大，决赛内部某格下溢成 0 属模型侧输出；本层保底的是落选者不为零
    for k in set(keys) - set(seen[-1]):
        assert out[k] > 0.0, f"落选者 {k} 被抹成零，soft target 会学到假的绝不可能"


def _run_wide(keys: list[str], strength: dict[str, float], *, other_mass: float = 0.05) -> dict[str, float]:
    """跑一次票选（默认参数）并把结果交回，便于各用例只关心断言。"""
    return wide_vote(keys, _biased_vote(strength, other_mass=other_mass))


def _finalist_set(keys: list[str], strength: dict[str, float]) -> set[str]:
    """按同一套分组与取头名规则复算决赛名单，用于校验留量只给落选者。"""
    vote = _biased_vote(strength)
    finalists: set[str] = set()
    for group in plan_groups(keys):
        shares = vote(list(group) + [OTHER_KEY])
        real = shares[:-1]
        ranked = sorted(zip(group, real), key=lambda x: (-x[1], keys.index(x[0])))
        finalists.update(name for name, _ in ranked[:KEEP_TOP])
    return finalists
