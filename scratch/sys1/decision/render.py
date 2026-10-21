"""systemone 渲染层：把"就某个状态问一道题"的请求排成模型要读的定死文字。

【做什么】
    输入是请求里的 state（发生了什么）加一道题（类型、问法、候选清单），输出是一段
    逐字节锁死的文字：一句固定系统指令 + `Evidence:` 证据 + `Question:` 问题 +
    `Options:` 带字母代号的候选。同一条请求渲染两次，结果必须一模一样。

【怎么做】
    分两步。第一步 `from_systemone()` 做"解析 + 校验"：三种问法（choice / score / noul）
    统一成内部行 `{"id", "type", "state", "instructions", "options": [{"id", "criterion"}]}`；
    候选缺说明时用代号本身补位（样本契约要求说明是非空文字），空 state 补成 `(none)`，
    随后把行还原成样本形态交给 `sys1.data.schema.validate_sample` 过一遍，非法请求
    当场弹回，错误消息自带字段路径。
    第二步 `render()` 做排版：固定系统行，再接正文三段——`Evidence:` 证据行、空行后的
    `Question:` 问题行、`Options:` 起的字母代号候选行（`A) ...`、`B) ...`）；代号一律按
    字母序排（数字代号按数值排，"2" 不会排在 "10" 后），
    超过 26 个候选直接拒绝并提示改走 `wide.py` 分组。
    `RENDER_VERSION` 是这段文字的版号：动过一个字节就升一次版本、黄金快照同批改；
    `THINK_OFF_SUFFIX` 只是留给二阶段对话模板接缝的常量，本阶段的读点位是"最后一个真
    token 的位置"，不依赖任何模板后缀。

【为什么】
    渲染是模型唯一的输入面：漂移一个空格，选项分布就跟着变，而它既不进损失也不进梯度，
    出了问题最难归因。所以做成"纯函数 + 版号常量 + 逐字节快照"三件套，把格式钉成契约。
    被否方案一：在服务层就地拼字符串——二阶段换 backbone 仍要复用同一套文字，散着拼
    必然漂移，跨阶段对齐（PRODUCTION G7）也无从下手。
    被否方案二：为"多样性"给渲染加时间戳或随机序——spec 明确要"渲染确定性"，一旦带上
    随机，快照测试与跨端对比同时失效。
    被否方案三：用 jinja 模板渲染（照抄参考实现的 chat template）——本阶段词表里根本没有
    模板控制符，模板反而把"读点位在第几个 token"这件事藏进不可控的字符串里。
"""
from __future__ import annotations

import json
import re
import string
from typing import Any

from sys1.data.schema import validate_sample

# ---------------------------------------------------------------- 契约常量（改动 = 升版本）
# 版号写进 run config，跨报告引用时随数字一起报（decision/README 硬约定）。
RENDER_VERSION = "dmlaya_render_v1"

# 固定系统行：一句"照准则看证据、只答一个代号"。措辞进快照，所以钉在这里而非散落各处。
SYSTEM_LINE = "Apply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only."

# 只留给二阶段对话模板接缝；本阶段读出用不到（读点位 = 最后一个真 token）。
THINK_OFF_SUFFIX = "<think>\n\n</think>\n\n"

LETTERS = string.ascii_uppercase          # 26 枚字母代号，A 起头按序发
MAX_OPTIONS = len(LETTERS)                # 一轮渲染的候选上限；超出必须走 wide 分组
EMPTY_STATE = "(none)"                    # 空证据的占位串：样本契约不接受空白 state
OTHER_KEY = "__other__"                   # 分组票选的"残差槽"代号（不是真候选，不对用户暴露）
OTHER_TEXT = "其他（以上都不是）"          # 残差槽在文字面上的样子
_NOUL_QUESTION_FALLBACK = "Which answer fits the evidence?"  # noul 请求没写问法时的兜底
_NOUL_LABEL = {"false": "no", "true": "yes"}                 # noul 代号 → 人读的答法


class RenderError(ValueError):
    """渲染层自己的拒绝；契约级违规仍抛 `SchemaError`（消息带字段路径）。"""


# ---------------------------------------------------------------- 内部小工具（私有，不对外）
def _norm(text: Any) -> str:
    """把"是否等价"的比较拉平：折叠空白/下划线/连字符并转小写（判断说明是不是在复述代号）。"""
    return re.sub(r"[\s_\-]+", " ", str(text)).strip().lower()


# "光一个代号样"的写法：A / (a) / B. / 3 / option_12 之类——这类代号在字母行里是歧义源。
_BARE_ID = re.compile(r"^(?:\(?[A-Za-z][).]?|\(?\d{1,3}[).]?|opt(?:ion)?[_ ]?\d{1,3})$", re.IGNORECASE)


def _num_sort_key(s: str) -> tuple[int, int, str]:
    """排序键：纯数字串按数值大小、其余按字典序——"2" 才不会被排在 "10" 后面。"""
    return (0, int(s), "") if s.isdigit() else (1, 0, s)


def _text_or_none(value: Any) -> str | None:
    """把请求里的说明字段收成"有内容的字符串"或 None；非字符串先 json 化，空串视作没给。"""
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = text.strip()
    return text or None


def _state_text(state: Any) -> str:
    """证据面：字符串直接用，结构体压成一行 json，空的补占位——契约不接受空白 state。"""
    if isinstance(state, str):
        return state.strip() or EMPTY_STATE
    if state is None or state == {} or state == []:
        return EMPTY_STATE
    return json.dumps(state, ensure_ascii=False)


def _criteria_pairs(qtype: str, criteria: Any) -> list[tuple[str, str]]:
    """把三型问法的候选清单拉平成 (代号, 说明) 列表，并把非法形态当场拒绝。"""
    if qtype == "noul":
        table = criteria if isinstance(criteria, dict) else {}
        # 键约定：noul 恒为 {"false","true"}；请求写成 bool/True 之类的怪键也照收（json 里键会被 str 化）。
        lookup = {_norm(k): v for k, v in table.items()}
        return [(key, _text_or_none(lookup.get(key)) or _NOUL_LABEL[key]) for key in ("false", "true")]

    if qtype == "score":
        # 档位顺序按数值升序（低→高）；dict 或 list 都接，最终代号一律重写成 "0".."n-1"。
        values = _score_values(criteria)
        return [(str(i), _text_or_none(v) or str(i)) for i, v in enumerate(values)]

    # choice：dict 保原键，list 按下标造键；两者都要求每项是个能给人读的说明（缺则回落到代号）。
    if isinstance(criteria, dict):
        items = list(criteria.items())
    elif isinstance(criteria, (list, tuple)):
        items = [(str(i), v) for i, v in enumerate(criteria)]
    else:
        raise RenderError(f"choice 的 criteria 必须是 dict 或 list，实际是 {type(criteria).__name__}")
    pairs: list[tuple[str, str]] = []
    for key, value in items:
        if not isinstance(key, str) or not key.strip():
            raise RenderError(f"choice 的选项代号必须是非空字符串，出现 {key!r}")
        pairs.append((key, _text_or_none(value) or key))
    return pairs


def _score_values(criteria: Any) -> list[Any]:
    """score 档位按数值升序取出值序列；键不是数字时按给定顺序取（兼容星级/文字档名）。"""
    if isinstance(criteria, dict):
        try:
            ordered = sorted(criteria, key=lambda k: float(k))
        except (TypeError, ValueError):
            ordered = list(criteria)
        return [criteria[k] for k in ordered]
    if isinstance(criteria, (list, tuple)):
        return list(criteria)
    raise RenderError(f"score 的 criteria 必须是 dict 或 list，实际是 {type(criteria).__name__}")


# ---------------------------------------------------------------- A1：请求解析 + schema 接入
def from_systemone(state: Any, spec: dict[str, Any], qid: str = "q") -> dict[str, Any]:
    """把一个 `/v1/systemone` 的单题请求解析成"待渲染行"，并过一遍样本契约。

    【做什么】吃 `spec = {"type", "instructions"/"question", "criteria"/"options"}`，吐
    `{"id", "type", "state", "instructions", "options": [{"id", "criterion"}]}`；score 额外带
    `ordered=True` 声明"这列档位有低到高的顺序"。

    【怎么做】先按 qtype 把候选清单拉平成 (代号, 说明) 对，缺说明的项用代号本身补上（样本
    契约的选项说明不接受空串）；再把这份对偶还原成 `{"state", "questions": {qid: {...}}}`
    交给 `validate_sample`，用它归一化后的结果回造行——于是下游拿到的每一行都已经在契约内。

    白话：这一步像前台收件：来人递上的表格格式五花八门，先统一抄成同一张单子，缺字的
    按规矩补上默认话术，空着的一栏也不许留着不填；抄完再对着验收标准逐条核对一遍，
    不合格的当场退回并说清是哪一格不对，合格才发出。

    :raises SchemaError: 请求不合样本契约（消息含字段路径，如 `questions.q1.type`）。
    :raises RenderError: 候选清单的形态本身认不出来（既不是 dict 也不是 list）。
    """
    if not isinstance(spec, dict):
        raise RenderError(f"题目请求必须是 dict，实际是 {type(spec).__name__}")
    qtype = spec.get("type", "choice")
    if qtype == "bool":  # 上游偶见 "bool" 写法，语义就是非真即假
        qtype = "noul"
    # 未知 qtype 不在这里判：交给样本契约拒，错误消息自带字段路径与合法枚举。

    criteria = spec.get("criteria")
    if criteria is None:
        criteria = spec.get("options")
    if qtype in ("choice", "score") and criteria is None:
        raise RenderError(f"qtype={qtype} 必须给出 criteria（候选清单）")
    pairs = _criteria_pairs(qtype, criteria)

    instructions = _text_or_none(spec.get("instructions")) or _text_or_none(spec.get("question"))
    if instructions is None:
        if qtype == "noul":
            instructions = _NOUL_QUESTION_FALLBACK
        else:
            raise RenderError(f"qtype={qtype} 必须给出 instructions（问题正文）")

    sample = {
        "state": _state_text(state),
        "questions": {qid: {"type": qtype, "instructions": instructions, "criteria": dict(pairs)}},
    }
    normalized = validate_sample(sample)  # SchemaError 原样上抛：路径式消息是下游调试的抓手
    question = normalized["questions"][qid]
    row: dict[str, Any] = {
        "id": qid,
        "type": question["type"],
        "state": normalized["state"],
        "instructions": question["instructions"],
        "options": [{"id": key, "criterion": value} for key, value in question["criteria"].items()],
    }
    if question["type"] == "score":
        row["ordered"] = True
    return row


# ---------------------------------------------------------------- A2：排版（字母序 + 版本常量）
def option_order(row: dict[str, Any]) -> list[str]:
    """给一行的候选定出字母序：残差槽恒排最后，其余按代号（数字按数值）升序。

    白话：先把这些候选的名字排成一列队——纯数字的按大小排，其余按字母表顺序，
    那个"以上都不是"的垫底名额永远站最后；排好的列子决定谁拿到 A、谁拿到 B。
    """
    ids = [opt["id"] for opt in _options_of(row)]
    rest = [i for i in ids if i != OTHER_KEY]
    rest.sort(key=_num_sort_key)
    return rest + [OTHER_KEY] if OTHER_KEY in ids else rest


def option_lines(row: dict[str, Any], order: list[str]) -> list[str]:
    """按给定顺序把候选排成 `A) ...` 一行的形态，行数与字母数一一对应。

    白话：手里有一列已经排好队的候选，这里给第一个人发 A、第二个人发 B，再把它的
    说明写在代号后面拼成一行；说明跟代号本身是同一句话时就只写代号，免得屏幕上出现
    "A) A: 晴" 这种让答案字母含糊的写法。
    """
    table = {opt["id"]: opt.get("criterion") for opt in _options_of(row)}
    hide = _hide_ids(row, order, table)
    lines: list[str] = []
    for slot, key in enumerate(order):
        if slot >= MAX_OPTIONS:
            raise RenderError(
                f"一行渲染最多 {MAX_OPTIONS} 个候选（字母代号用尽），实得 {len(order)}；请走 wide.py 分组票选"
            )
        body = OTHER_TEXT if key == OTHER_KEY else _option_body(key, table.get(key), row, hide)
        lines.append(f"{LETTERS[slot]}) {body}")
    return lines


def render(row: dict[str, Any], order: list[str] | None = None) -> tuple[list[dict[str, str]], list[str]]:
    """把一行渲染成 messages（system 固定行 + user 正文），并回字母序对应的代号列。

    白话：把证据、问题、候选拼成一封短信，信纸开头永远是那句一动不动的规矩，正文按
    "证据—问题—候选"三段排；同时把"第几号字母对应哪个候选"一并交回去，读答案的人
    只认这份对照，不用再去猜字母是从哪来。

    返回值是 `(messages, order)`：`messages` 为 `[{"role": "system", ...}, {"role": "user", ...}]`
    （二阶段直接交给对话模板即可），`order[i]` 是字母 `LETTERS[i]` 代表的候选代号。
    """
    expect = option_order(row)
    chosen = expect if order is None else list(order)
    if sorted(chosen) != sorted(expect):
        raise RenderError(f"order 必须是本行代号的排列；期望 {expect}，实得 {chosen}")
    return (
        [{"role": "system", "content": SYSTEM_LINE}, {"role": "user", "content": user_content(row, chosen)}],
        chosen,
    )


def prompt_text(row: dict[str, Any], order: list[str] | None = None) -> str:
    """把渲染结果压成一整段文字（一阶段没有对话模板，编号器直接编这一段）。

    白话：把系统规矩和正文首尾接成一条长文字，中间只隔一个换行；同一条请求两次给出
    的文字必须逐字节相同，黄金快照锁的就是它。
    """
    messages, _ = render(row, order)
    return _join(messages)


def user_content(row: dict[str, Any], order: list[str]) -> str:
    """正文骨架：`Evidence:` 证据行 → 空行 → `Question:` 问题行 → `Options:` 加各候选行。

    白话：正文就三段——先摆证据，再问问题，最后一行列出带字母的候选；每一段的接缝
    都是固定的几个换行，写死在这里，任何一处改动都会立刻让快照对不上。
    """
    state = row["state"] if str(row["state"]).strip() else EMPTY_STATE
    body = "\n".join(option_lines(row, order))
    return (
        "Evidence:\n" + state + "\n\nQuestion: " + row["instructions"] + "\nOptions:\n" + body
    )


def subrow(row: dict[str, Any], keys: list[str], *, other: bool = False) -> dict[str, Any]:
    """从整行里裁出子候选集（wide 分组用），可选追加一枚残差槽。

    白话：把一长串候选按组切开，每组照样带上说明文字；末尾再塞一个"以上都不是"的
    垫底名额，让被硬按着头选的局面有个出口。
    """
    table = {opt["id"]: opt.get("criterion") for opt in _options_of(row)}
    missing = [k for k in keys if k not in table]
    if missing:
        raise RenderError(f"分组裁出的代号在本行里不存在：{missing}")
    options = [{"id": k, "criterion": table[k]} for k in keys]
    if other:
        options.append({"id": OTHER_KEY, "criterion": None})
    return {**row, "options": options, "ordered": False}


# ---------------------------------------------------------------- 私有排版细节
def _options_of(row: dict[str, Any]) -> list[dict[str, Any]]:
    """取一行的候选列表并做最低限度体检（必须是带 id 的 dict 列表）。"""
    options = row.get("options")
    if not isinstance(options, list) or not options:
        raise RenderError("渲染行的 options 必须是非空 list（由 from_systemone 产出）")
    for opt in options:
        if not isinstance(opt, dict) or not isinstance(opt.get("id"), str) or not opt["id"]:
            raise RenderError(f"候选项必须是含非空 id 的 dict，出现 {opt!r}")
    return options


def _hide_ids(row: dict[str, Any], order: list[str], table: dict[str, Any]) -> bool:
    """判断是否该把代号从正文里隐去：choice 且每个代号都像"光杆字母/编号"且都有说明。"""
    if row.get("type") != "choice":
        return False
    keys = [k for k in order if k != OTHER_KEY]
    return bool(keys) and all(_BARE_ID.match(k) for k in keys) and all(table.get(k) for k in keys)


def _option_body(key: str, criterion: Any, row: dict[str, Any], hide: bool) -> str:
    """单行正文：noul 翻成 yes/no；说明缺席或只是复述代号时只留代号；否则"代号: 说明"。"""
    name = _NOUL_LABEL.get(key, key) if row.get("type") == "noul" else key
    text = _text_or_none(criterion)
    if hide and text is not None:
        return text
    if text is None or _norm(text) == _norm(name):
        return name
    return f"{name}: {text}"


def _join(messages: list[dict[str, str]]) -> str:
    """把两条 messages 拼成一段文字（system 固定行在前，正文紧随其下）。"""
    return "\n".join(msg["content"] for msg in messages)


__all__ = [
    "EMPTY_STATE",
    "LETTERS",
    "MAX_OPTIONS",
    "OTHER_KEY",
    "OTHER_TEXT",
    "RENDER_VERSION",
    "SYSTEM_LINE",
    "THINK_OFF_SUFFIX",
    "RenderError",
    "from_systemone",
    "option_lines",
    "option_order",
    "prompt_text",
    "render",
    "subrow",
    "user_content",
]
