"""三类源数据（XNLI / 分类 / 星级）→ typed 决策样本的转写层。

【做什么】
    把外部任务数据翻译成 sys1 样本契约认得的那张单子：一道题、一份证据、以及
    "每个候选分到多少信心"。三件活——NLI 三分类折成非真即假（noul）、意图/类别
    标签折成多选（choice）、星级评分折成分档（score）。转写只做字段搬运与键约定
    对齐，不发明标签、不臆测缺失项，源头没写清楚的一律当场退回。

【怎么做】
    每条公开函数吃一个"来源记录"（带 id 与该任务原生字段），吐一条"数据记录"
    `{"id", "task", "sample"}`：sample 是过完 `validate_sample` 的 typed 样本，id
    原样带上做溯源（防泄漏比对靠它，见 tools/check_leak.py）。转写规则钉在模块
    常量里而非散进 if 分支——XNLI→NOUL_MAP 把三分类压成二值（只有蕴含算 true，
    中立与矛盾统统归 false），星级→档位用固定换算，分类标签直接把标签名当候选代号。
    产出前一律走一遍样本契约：概率和、键约定、qtype 枚举由契约把关，转写层只保证
    "搬运不错、键对得上"。score 特殊：契约要求档位键必须 0..k-1 全档齐活，故实得
    那一档押 1、其余档如实写 0，绝不留空档。

【为什么】
    把"外部格式→内部契约"的翻译收敛到一个模块，是为了让 s2 训练只管吃契约内的
    样本流，不必各自认得 XNLI/MASSIVE/星级的千面格式；id 留在 sample 之外的信封里，
    是因为样本契约的顶层字段是封闭集合（多一个键即拒），溯源信息必须住在信封上。
    被否方案一：把 label→分布的规则写成 if/elif 内联在 s2 里——转写与训练耦死，
    换一批源数据就得改训练脚本，且规则无法被独立单测钉住（本模块 A2 的存在理由）。
    被否方案二：让 id 混进 sample 顶层——validate_sample 会因"未知字段"直接拒收，
    等于把契约当橡皮泥乱捏，得不偿失；于是 id 与 sample 分家，各归各位。
"""
from __future__ import annotations

from typing import Any

from sys1.data.schema import SchemaError, validate_sample

# 转写版号：动过任一条映射规则就升版本，与黄金快照同批改（跨报告引用时随数字报出）。
TRANSCRIBE_VERSION = "dmlaya_transcribe_v1"

# XNLI 三分类 → noul 二值的显式映射表（spec A2 锁定：只有蕴含记 true，中立/矛盾归 false）。
# 键统一小写存放，查表前对来源 label 先折叠大小写与首尾空白，容错上游书写差异。
XNLI_TO_NOUL: dict[str, str] = {
    "entailment": "true",
    "neutral": "false",
    "contradiction": "false",
}

# 星级下限恒为 1、上限等于 scale：换算成 score 档位下标用这条常量锁死，单测逐值核对。
STAR_MIN = 1

# noul / choice / score 三型的固定 qid：转写产物单题单问，qid 写死省去下游猜题号。
_QID = "q"


class TranscribeError(ValueError):
    """来源记录不满足转写前提时抛出；继承 ValueError 便于调用方与 SchemaError 同族兜底。"""


# ---------------------------------------------------------------- 内部小工具
def _norm_label(value: Any) -> str:
    """把来源标签折成可查表的小写键（去首尾空白、转小写）；非字符串先转字符串再折。"""
    return str(value).strip().lower()


def _require(record: dict[str, Any], key: str, task: str) -> Any:
    """从记录里取必填字段；缺失即抛带任务名与字段名的转写错误，绝不静默用默认顶替。"""
    if not isinstance(record, dict):
        raise TranscribeError(f"{task} 记录必须是 dict，实际是 {type(record).__name__}")
    if key not in record or record[key] is None:
        raise TranscribeError(f"{task} 记录缺少必填字段 {key!r}")
    return record[key]


def _record(record: Any, task: str, sample: dict[str, Any]) -> dict[str, Any]:
    """把 typed 样本装进带溯源 id 的信封：先校验样本，再回 {id, task, sample}。"""
    sid = record.get("id") if isinstance(record, dict) else None
    if not isinstance(sid, str) or not sid.strip():
        raise TranscribeError(f"{task} 记录缺少非空字符串 id（防泄漏比对要靠它，见 tools/check_leak.py）")
    validated = validate_sample(sample)  # SchemaError 原样上抛：转写产物必须先过契约
    return {"id": sid, "task": task, "sample": validated}


# ---------------------------------------------------------------- A1：XNLI-zh → noul
def xnli_to_noul(record: dict[str, Any]) -> dict[str, Any]:
    """把一条 XNLI 记录（premise/hypothesis/label）转成单题 noul 样本信封。

    白话：给两句中文和一个关系标签——若是"前者成立则后者必成立"（蕴含）就答"真"，
    其余两种关系（说不上、互相打架）都答"假"。证据栏放首句，问题栏带上次句，
    分数只押在判出的那一头上。

    :param record: {"id", "premise", "hypothesis", "label"}，label 支持大小写与空白差异。
    :returns: {"id", "task": "noul", "sample": {state, questions, targets}}。
    """
    premise = _require(record, "premise", "xnli")
    hypothesis = _require(record, "hypothesis", "xnli")
    raw_label = _require(record, "label", "xnli")
    key = _norm_label(raw_label)
    if key not in XNLI_TO_NOUL:
        raise TranscribeError(f"xnli 未知 label {raw_label!r}；合法取值 {sorted(XNLI_TO_NOUL)}")
    verdict = XNLI_TO_NOUL[key]
    sample = {
        "state": str(premise),
        "questions": {_QID: {"type": "noul", "instructions": f"据此判断：{hypothesis}"}},
        "targets": {_QID: {verdict: 1.0}},
    }
    return _record(record, "noul", sample)


# ---------------------------------------------------------------- A1：分类 → choice
def classification_to_choice(record: dict[str, Any]) -> dict[str, Any]:
    """把一条分类记录（文本 + 候选标签 + 正解标签）转成单题 choice 样本信封。

    白话：给一段话、一串可能的类别名、以及这句话真正属于哪一类。候选名各自占一格，
    分数全押在正确那一类上，其余为 0——相当于"多选题只勾一个标准答案"。

    :param record: {"id", "text", "options", "label"}；options 可为标签列表或 {代号: 说明}。
    :returns: {"id", "task": "choice", "sample": {...}}。
    """
    text = _require(record, "text", "classification")
    options = _require(record, "options", "classification")
    label = _require(record, "label", "classification")
    criteria = _choice_criteria(options)
    answer = str(label)
    if answer not in criteria:
        raise TranscribeError(f"classification 正解 {label!r} 不在候选集 {sorted(criteria)} 内")
    sample = {
        "state": str(text),
        "questions": {_QID: {"type": "choice", "instructions": "这段话属于哪一类？", "criteria": criteria}},
        "targets": {_QID: {answer: 1.0}},
    }
    return _record(record, "choice", sample)


def _choice_criteria(options: Any) -> dict[str, str]:
    """把候选标签归一成 {代号: 说明}：列表用标签自身当代号，字典原样收下。"""
    if isinstance(options, dict):
        return {str(k): str(v) for k, v in options.items()}
    if isinstance(options, (list, tuple)):
        names = [str(o) for o in options]
        if not names:
            raise TranscribeError("classification options 为空列表，无法构造候选集")
        return {name: name for name in names}
    raise TranscribeError(f"classification options 需为 list 或 dict，实际是 {type(options).__name__}")


# ---------------------------------------------------------------- A1：星级 → score
def rating_to_score(record: dict[str, Any]) -> dict[str, Any]:
    """把一条评分记录（文本 + 星级 + 档数）转成单题 score 样本信封。

    白话：给一段评语和它打出的星数（比如五星里的三星），把 1 到上限的每一档铺成
    从低到高的候选格子，实得那一档押满分、其余档写零。档位从 0 起编号，低到高排好队。

    :param record: {"id", "text", "rating", "scale"}；scale 缺省 5，rating ∈ [1, scale]。
    :returns: {"id", "task": "score", "sample": {...}}。
    """
    text = _require(record, "text", "rating")
    rating = _require(record, "rating", "rating")
    scale = int(record.get("scale", 5))
    if scale < 2:
        raise TranscribeError(f"rating scale 至少 2 档，实得 {scale}")
    level = int(rating)
    if level < STAR_MIN or level > scale:
        raise TranscribeError(f"rating {level} 越出区间 [{STAR_MIN}, {scale}]")
    index = level - STAR_MIN  # 星级从 1 起、档位下标从 0 起：实得星数减下限即落在哪一档
    criteria = {str(i): f"{i + STAR_MIN} 档" for i in range(scale)}
    # score 契约要求键**完整覆盖** 0..k-1（缺档即拒），故 one-hot 也要把其余档位显式写 0
    dist = {str(i): (1.0 if i == index else 0.0) for i in range(scale)}
    sample = {
        "state": str(text),
        "questions": {_QID: {"type": "score", "instructions": "这条评价属于哪一档？", "criteria": criteria}},
        "targets": {_QID: dist},
    }
    return _record(record, "score", sample)


# ---------------------------------------------------------------- 分派入口
_TASK_DISPATCH = {
    "noul": xnli_to_noul,
    "choice": classification_to_choice,
    "score": rating_to_score,
}


def transcribe(record: dict[str, Any]) -> dict[str, Any]:
    """按记录里的 task 字段分派到对应转写器；缺 task 时按字段特征猜，猜不出即拒。

    白话：来一张来源单子，先看它标了哪种任务（非真即假、多选、分档），交给会办这种的
    那一双手去转；要是没标，就按它带了哪几栏来推断（有 hypothesis 判 NLI、有 options
    判多选、有 rating 判评分），三条都对不上就退回，绝不硬猜成别的模样。

    :raises TranscribeError: 无法判定任务类型，或分派后来源字段不合规。
    """
    if not isinstance(record, dict):
        raise TranscribeError(f"transcribe 需要 dict 记录，实际是 {type(record).__name__}")
    task = record.get("task")
    resolved = _norm_label(task) if task is not None else _infer_task(record)
    handler = _TASK_DISPATCH.get(resolved)
    if handler is None:
        raise TranscribeError(
            f"未知 task {task!r}；支持 {sorted(_TASK_DISPATCH)}（或不带 task 由字段特征推断）"
        )
    return handler(record)


def _infer_task(record: dict[str, Any]) -> str | None:
    """没标 task 时按字段特征猜类型：hypothesis→noul、options→choice、rating→score。"""
    if "hypothesis" in record and "label" in record:
        return "noul"
    if "options" in record and "label" in record:
        return "choice"
    if "rating" in record:
        return "score"
    return None


__all__ = [
    "TRANSCRIBE_VERSION",
    "XNLI_TO_NOUL",
    "SchemaError",
    "TranscribeError",
    "classification_to_choice",
    "rating_to_score",
    "transcribe",
    "xnli_to_noul",
]
