"""教师作答协议：把自由生成钉进 `<answer>X</answer>`，再按问法分域抽取与确定性判分。

【做什么】
    三种问法（choice 选字母 / noul 判真假 / score 报数值）各有各的答案形状。本模块
    负责两件事：给教师一段"必须用 <answer>…</answer> 收尾"的约束话术，以及把回复里
    的 `<answer>` 内容按问法抽成规范值——字母、"true"/"false"、或一个纯数字串。
    抽不出来（标签套标签、内容空、字母不在候选里、数字读不出）就返回 None，由调用方
    重试一轮。

【怎么做】
    先正则取 `<answer>` 段（对齐 GLM-V 官方 verifier 的写法：段内再出现 think/answer
    标签即判非法，直接 None），再按 qtype 走三条分域小函数：`_domain_choice` 在候选
    字母集合内做精确匹配（先去 "(A)"、"A)"、"A." 这类装饰，出现两个不同字母判歧义
    返 None）；`_domain_noul` 把中英文的是/否说法折成 "true"/"false"；`_domain_score`
    剥掉单位与千分位后按浮点收，规约成十进制字符串。判分同样分域：choice 比字母、
    noul 比布尔、score 比"差值落进容差带"。重试话术由 `retry_prompt` 追加一句更硬的
    格式提醒，只追加不改写，保证同一请求两次之间只差这一句。

【为什么】
    生成式兜底必须结构化：自由生成再通用解析会把"我倾向于 A，因为……"这种长句判成
    不可预测，而 `<answer>` 一贴就把抽取面收窄到几个字符，抽取器能写死分域规则；
    max_tokens 也因此能压到十几，省钱省时。被否方案一：一个正则通吃三域——score 域里
    "3.5/10" 会被字母规则误读，noul 域里 "no" 会被数字规则漏掉，分域是唯一稳的写法。
    被否方案二：抽取失败就上外部裁判兜底——按 §11-4 红线，判定链不能依赖第三方在线
    接口，且确定性 verifier 才可复现；判不出就如实返 None，重试一轮后仍失败交给调用方
    按缺失处理。被否方案三：把容差写死成常量不分域——score 的合理容差跟量纲走，
    调用方（p2-11 RL 双通道）本来就要按 qtype 配权重，容差同理交给调用方。
"""
from __future__ import annotations

import re
from typing import Any, Sequence

__all__ = [
    "ANSWER_RE",
    "QTYPES",
    "build_answer_prompt",
    "extract_answer",
    "hard_distribution",
    "judge",
    "retry_prompt",
]

QTYPES = ("choice", "noul", "score")

# GLM-V 官方 verifier 同款：`<answer>` 段 + 段内不许再出现成对标签（嵌套即判非法）
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_NESTED = ("<answer>", "</answer>", "<think>", "</think>", "<thinking>", "</thinking>")
# 只认"前后都不是字母"的独立字母：单词里的首字母（Answer 的 A）绝不当候选
_LETTER = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])")
_LETTER_LOWER = re.compile(r"(?<![A-Za-z])([a-z])(?![A-Za-z])")
_NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
# noul 说法表：命中即真/假；只看是否含下列子串，顺序无所谓
_TRUTHY = ("true", "yes", "是", "对", "正确", "属实", "肯定")
_FALSY = ("false", "no", "否", "不是", "不对", "不错误", "错误", "不属实", "非", "假")
# 约束话术：只加一句硬要求，不动原请求正文（原正文由 P1 渲染层负责，本层不越界）
_ANSWER_RULE = "Answer strictly as <answer>X</answer> where X is {placeholder}. Output nothing after the closing tag."
_RETRY_HINT = "Your previous reply had no parsable <answer>...</answer> value. Reply with ONLY <answer>{placeholder}</answer>."


def build_answer_prompt(prompt: str, option_keys: Sequence[str], *, qtype: str = "choice") -> str:
    """给原请求附上一行作答约束，返回可直接发给教师的新请求（原请求逐字保留）。

    白话：在题目底下加一句"答案必须用尖括号包住、除此之外别的话都别写"，
    选字母的题就说填字母，判真假的题就说填 true 或 false，打分的题就说填数字。
    """
    if qtype not in QTYPES:
        raise ValueError(f"未知 qtype {qtype!r}（合法：{QTYPES}）")
    placeholder = _placeholder(qtype, option_keys)
    return f"{prompt}\n\n{_ANSWER_RULE.format(placeholder=placeholder)}"


def retry_prompt(prompt: str, option_keys: Sequence[str], *, qtype: str = "choice") -> str:
    """重试一轮的话术：把格式要求再压一遍，除此之外不新增任何信息。

    白话：上一回没读懂它的答复，这次只补一句"照这个格式重说"，不提示答案也不改题目，
    免得第二次请求掺进新信息，让两次结果没法公平比较。
    """
    placeholder = _placeholder(qtype, option_keys)
    return f"{prompt}\n\n{_RETRY_HINT.format(placeholder=placeholder)}"


def extract_answer(
    text: str,
    *,
    qtype: str = "choice",
    option_keys: Sequence[str] | None = None,
    strict: bool = True,
) -> str | None:
    """从教师回复里抽 `<answer>` 值并按问法规约；不可解析返回 None（由调用方重试）。

    白话：先在回复里找到尖括号包住的那一小段。找不到、尖括号套尖括号、里头是空的，
    都算"没答"。找到了再按题型收拾：字母题把 "(A)" 之类装饰剥掉、判断题折成
    true/false、分数题去掉单位留下纯数字。收拾不成也算"没答"。
    """
    if qtype not in QTYPES:
        raise ValueError(f"未知 qtype {qtype!r}（合法：{QTYPES}）")
    if not isinstance(text, str) or not text.strip():
        return None
    matches = ANSWER_RE.findall(text)
    if not matches:
        # 宽松模式（strict=False）才允许退到"整段回复就是答案"；默认关，防误抽长句里的字母
        if strict:
            return None
        body = text.strip()
    else:
        body = matches[-1].strip()  # 取最后一段：模型改主意时的终答才算数
    if not body:
        return None
    if any(tag in body.lower() for tag in _NESTED):
        return None  # 段内仍有成对标签 → 协议违规，判 None（GLM-V verifier 同款口径）
    if qtype == "choice":
        return _domain_choice(body, option_keys)
    if qtype == "noul":
        return _domain_noul(body)
    return _domain_score(body)


def judge(pred: Any, gold: Any, *, qtype: str = "choice", rel_tol: float = 0.05, abs_tol: float = 0.0) -> bool:
    """确定性判分（分域三规则），返回对错——不给分、不解释、外部裁判一律不用。

    白话：字母题就比两个字母是否一样；判断题比是不是同一个真假；分数题看两个数相差
    够不够近，近的标准是"相对误差不超过约定比例，或者绝对差不超过约定底线"。
    """
    if pred is None or gold is None:
        return False
    if qtype == "choice":
        return str(pred).strip().upper() == str(gold).strip().upper()
    if qtype == "noul":
        left = _domain_noul(str(pred))
        right = _domain_noul(str(gold))
        return left is not None and left == right
    left, right = _to_float(pred), _to_float(gold)
    if left is None or right is None:
        return False
    slack = max(abs_tol, rel_tol * abs(right))
    return abs(left - right) <= slack


def hard_distribution(answer: str, option_keys: Sequence[str], *, mass: float = 0.9) -> dict[str, float]:
    """把抽到的确定答案折成"绝大多数押它、余下一点均摊"的分布（answer 兜底路径出口）。

    白话：模型只说了一个字母，可训练那边要一份份额表。于是给它压上绝大部分，剩下
    一小撮在别的候选里摊平；这样既保留"它确实只答了一个"的信息，也不出现纯粹的 0。
    """
    keys = [str(k) for k in option_keys]
    if not keys:
        raise ValueError("option_keys 不能为空")
    if not 0.0 < mass < 1.0:
        raise ValueError(f"mass 必须落在 (0,1)，实得 {mass}")
    if len(keys) == 1:
        return {keys[0]: 1.0}
    target = str(answer).strip().upper() if _looks_like_letter(keys) else str(answer).strip().lower()
    if target not in keys:
        raise ValueError(f"答案 {answer!r} 不在候选 {keys} 内，无法折分布")
    rest = (1.0 - mass) / (len(keys) - 1)
    return {k: (mass if k == target else rest) for k in keys}


# ── 分域小函数（只在本文件内用，规则各自独立，便于逐域加测试）────────────────
def _domain_choice(body: str, option_keys: Sequence[str] | None) -> str | None:
    """字母域：剥装饰后在候选集合内精确匹配；出现两个不同字母判歧义返 None。"""
    allowed = [str(k).strip().upper() for k in (option_keys or [])] or list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    stripped = body.strip().strip("。.！!？?()[]{}:：、 \"'`\t\r\n")
    if len(stripped) == 1 and stripped.isalpha():
        found = [stripped.upper()]                        # "b" / "(B)" 这类整段就是一个字母
    else:
        found = _LETTER.findall(stripped) or _LETTER_LOWER.findall(stripped)
    unique = sorted(set(found))
    if len(unique) != 1:
        return None  # 0 个字母或不止 1 个字母：都算没给出唯一候选
    letter = unique[0]
    return letter if letter in allowed else None


def _domain_noul(body: str) -> str | None:
    """真假域：中英文是/否说法折成 "true"/"false"；真假两可（同现两类词）判 None。"""
    low = body.strip().lower().strip("。.！!？? \t\r\n")
    if not low:
        return None
    hit_false = any(w in low for w in _FALSY)
    residual = low
    for word in _FALSY:          # 先把否定短语整块摘掉，剩下的才算肯定证据（"不对"≠"对"）
        residual = residual.replace(word, "")
    hit_true = any(w in residual for w in _TRUTHY)
    if hit_true and hit_false:   # 又说是又说不是 → 不可判定
        return None
    if hit_true:
        return "true"
    return "false" if hit_false else None


def _domain_score(body: str) -> str | None:
    """数值域：剥掉单位/千分位/百分号，收第一个可转浮点的数并规约成十进制字符串。"""
    cleaned = body.replace(",", "").replace("，", "").replace("%", "").strip()
    for token in _NUMBER.findall(cleaned):
        value = _to_float(token)
        if value is not None:
            return repr(value)  # repr 而非 str：往返无损，且与 judge 的解析口径一致
    return None


def _to_float(value: Any) -> float | None:
    """尽力把单个数值候选转 float；转不动返回 None，绝不抛错打断抽取。"""
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _looks_like_letter(keys: Sequence[str]) -> bool:
    """候选是否都是单个字母（决定 hard_distribution 归一化时该压大小写还是压小写）。"""
    return all(len(str(k)) == 1 and str(k).isalpha() for k in keys)


def _placeholder(qtype: str, option_keys: Sequence[str] | None) -> str:
    """按问法生成作答占位说明（字母集合 / true|false / 数字）。"""
    if qtype == "choice":
        letters = "/".join(str(k).strip().upper() for k in (option_keys or [])) or "the option letter"
        return f"one of {{{letters}}}"
    if qtype == "noul":
        return "true or false"
    return "a single number"
