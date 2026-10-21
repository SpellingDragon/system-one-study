"""样本契约：校验、归一化、jsonl 序列化与 score 键序工具。

【做什么】
    给 sys1 训练/评测管线提供**唯一合法的样本入口**：一个 dict 想被当作样本用，
    必须先过这里的 `validate_sample`；样本想落盘/回读，必须走 `dump` / `load`。

【怎么做】
    样本形态是 `/v1/systemone` 的超集：
        {state: str,
         questions: {qid: {type, instructions, criteria?}},
         targets?:  {qid: {opt_key: p}},
         ep_group?, ep_step?}                       # 二阶段 RL 多轮用，透传不校验语义
    校验沿字段路径逐层展开，违规抛出 `SchemaError`，消息里始终带上路径（如
    `questions.q3.type`）——这条约定是后续域调试的生命线，不可省。
    targets 概率和按 1e-6 容差判定，训练侧默认摊平到严格 1；评测侧可传
    `allow_unnormalized=True` 豁免。score 类 targets 的键必须是 `"0".."k-1"`
    字符串（低→高），noul 是 `"false"/"true"`，choice 沿用 criteria 原键。
    序列化是 jsonl 行式：一行一个样本，`dump` 前逐字段断言所有 dict 键均为 str
    （json 会把非字符串键静默改成字符串，类型丢了就查不回来）。

【为什么】
    把契约立在最上游，后续各域只管生产样本流、不必各自重造校验；`ep_*` 字段
    透传不校验，是为了让二阶段 RL 无需回改 schema。
    **被否方案 1：pydantic / dataclass**——引入外部依赖或大量样板，违背本域
    "零依赖 stdlib"的前提约束（schema.py 会被 p1-07/08/09/10 反向依赖，任何
    第三方绑定都会顺带传染到那些域）。
    **被否方案 2：jsonschema 库**——现成但错误消息形如 `$.questions.q3.type`
    仍需自拼路径；且它不会替我们做"概率和容差 1e-6 → 摊平"这类领域归一，
    收益不抵成本。
    因此选择纯 stdlib + 手写校验，把"路径式错误消息"和"targets 摊平"这两件
    本域独有的事做实。
"""
from __future__ import annotations

import json
from typing import Any

# 合法 qtype 的三枚取值；spec 明确 list 顺序即"低→高"，错误消息按此顺序展示。
QTYPES: tuple[str, ...] = ("choice", "noul", "score")

# targets 概率和的容差：多人投票份额落到 float 后允许的最小误差。
# 越界即视为分布拼错，训练侧拒绝；评测侧才被 allow_unnormalized 放行。
PROB_TOL: float = 1e-6

# 样本顶层允许的键。其他键视为"未知字段"直接拒（防"字段名拼错却静默通过"）。
_ALLOWED_TOP = {"state", "questions", "targets", "ep_group", "ep_step"}
# question 层允许的键。criteria 对 noul 可缺省，对 choice/score 必存在（下方校验）。
_ALLOWED_Q = {"type", "instructions", "criteria"}


class SchemaError(ValueError):
    """样本不符合契约时抛出；消息**必须**包含字段路径。

    继承 `ValueError` 是刻意的：调用方拿到的仍然是标准异常族，`except ValueError`
    兜底不会漏；同时又比裸 `ValueError` 多了一层可识别类型，管线在采集脏样本
    时可以精确只捕这一类、把非契约错误留给上层。
    """


def _fmt_path(parts: list[str]) -> str:
    """把字段名列表拼成 `a.b.c` 形式的路径字符串（内部工具，不对外）。"""
    return ".".join(parts) if parts else "<root>"


def _require_str(value: Any, path: list[str], what: str) -> str:
    """校验 value 是 str 且非空白，否则抛 SchemaError 并带上路径。"""
    # 白话：这里检查一个格子是否"写着人类可读的一句话"，不是 None、不是数字、
    # 也不是空串；一旦不合规就把出错的格子位置（比如 state 或 questions.q3.instructions）
    # 拼回消息里，方便下游调试时一眼定位到"是哪个问题的哪一行写歪了"。
    if not isinstance(value, str):
        raise SchemaError(f"{_fmt_path(path)} 必须是字符串（{what}），实际是 {type(value).__name__}")
    if not value.strip():
        raise SchemaError(f"{_fmt_path(path)} 不能是空白字符串（{what}）")
    return value


def _validate_criteria(choice_like: dict[str, Any], path: list[str]) -> list[str]:
    """要求 criteria 是 dict[str, str]，返回其键列表；违规抛 SchemaError。"""
    # 白话：criteria 相当于"选项清单"——左边是代号、右边是给模型看的说明文字。
    # 这一步只做三件事：确认它确实是一张清单、代号必须是给人读的字符串、说明文字
    # 也不能缺。任何一项对不上就顺手把"是哪道题的哪一格"报出来，不让人对着整段样本猜。
    if not isinstance(choice_like, dict):
        raise SchemaError(f"{_fmt_path(path)} 必须是 dict（选项清单），实际是 {type(choice_like).__name__}")
    if not choice_like:
        raise SchemaError(f"{_fmt_path(path)} 不能是空 dict（选项清单至少要有 1 项）")
    keys: list[str] = []
    for k, v in choice_like.items():
        if not isinstance(k, str):
            raise SchemaError(f"{_fmt_path(path + [str(k)])} 的键必须是字符串（json 里非 str 键会静默变形）")
        if not isinstance(v, str) or not v.strip():
            raise SchemaError(f"{_fmt_path(path + [k])} 必须是非空字符串（选项说明）")
        keys.append(k)
    return keys


def _score_expected_keys(k: int) -> set[str]:
    """按约定生成 score 类应有的键集合 {"0","1",...,"k-1"}（低→高）。"""
    return {str(i) for i in range(k)}


def _num_sort_key(s):
    """错误消息里排序键：数字串按数值大小、其他按字典序——避免 int("low") 这种二次崩。

    错误路径本身必须比正常路径更稳：如果打印非法键的提示时又抛 ValueError，真正的
    SchemaError 就被掩盖，后续域拿到的是完全无关的 traceback，直接失去调试抓手。
    """
    return (0, int(s), "") if isinstance(s, str) and s.isdigit() else (1, 0, str(s))


def _normalize_targets(
    targets: dict[str, Any],
    questions: dict[str, Any],
    path: list[str],
    allow_unnormalized: bool,
) -> dict[str, dict[str, float]]:
    """逐 qid 校验 targets 键约定与概率和，返回摊平后的新 dict（不修改入参）。

    白话：targets 是"每道题各个候选分到多少信心"。我们要求这些信心全是非负的
    小数，加在一起等于 1（允许一点点浮点误差）；训练侧默认把它重新摊平到严格 1，
    评测侧可以关掉这一步保留原样。每题的候选代号还得跟问题里给出的清单对得上，
    对不上就报错并指出是哪道题。
    """
    # 白话段落已附在函数 docstring 上，这里只做实现级提示。
    if not isinstance(targets, dict):
        raise SchemaError(f"{_fmt_path(path)} 必须是 dict（每题的分布），实际是 {type(targets).__name__}")

    out: dict[str, dict[str, float]] = {}
    for qid, dist in targets.items():
        # qid 必须在 questions 里出现过——否则等于"给不存在的题打了分"，静默丢弃是灾难。
        qpath = path + [qid]
        if not isinstance(qid, str):
            raise SchemaError(f"{_fmt_path(path)} 的键必须是字符串（qid），出现 {type(qid).__name__}")
        if qid not in questions:
            raise SchemaError(f"{_fmt_path(qpath)} 在 questions 里找不到对应题目（分布挂到了孤儿 qid 上）")
        q = questions[qid]
        qtype = q["type"]
        if not isinstance(dist, dict):
            raise SchemaError(f"{_fmt_path(qpath)} 必须是 dict（qid→{qtype} 的分布）")

        # —— 键约定校验：按 qtype 分别处理
        key_path = _fmt_path(qpath)
        if qtype == "noul":
            legal = {"false", "true"}
        elif qtype == "score":
            # 风险条目：k 从 criteria 键集合为准；criteria 已在 _validate_question 里确保为 str→str
            k = len(q["criteria"])
            legal = _score_expected_keys(k)
        else:  # choice
            legal = set(q["criteria"].keys())
        for opt in dist:
            if not isinstance(opt, str):
                raise SchemaError(f"{_fmt_path(qpath)} 的选项键必须是字符串（json 里非 str 键会静默变形）")
            if opt not in legal:
                raise SchemaError(
                    f"{key_path} 出现非法选项键 '{opt}'；该 qtype({qtype}) 允许的键集合是 {sorted(legal)}"
                )
        if qtype == "score":
            # score 特严：键必须**完整覆盖** {"0".."k-1"}，缺项即拒（spec 场景"键集合与 k 不符"）。
            missing = legal - set(dist.keys())
            if missing:
                raise SchemaError(
                    f"{key_path} score 键集合与 k={len(legal)} 不符，缺少 {sorted(missing, key=_num_sort_key)}"
                )

        # —— 概率值校验（bool 是 int 子类，这里显式拒 bool 以免 True/False 被当成 1/0）
        probs: dict[str, float] = {}
        for opt, p in dist.items():
            vpath = qpath + [opt]
            if isinstance(p, bool) or not isinstance(p, (int, float)):
                raise SchemaError(f"{_fmt_path(vpath)} 必须是数字概率（float/int），实际是 {type(p).__name__}")
            pv = float(p)
            if pv < 0.0:
                raise SchemaError(f"{_fmt_path(vpath)} 概率不能为负：{pv}")
            probs[opt] = pv

        # —— 概率和：默认要求 =1（容差 PROB_TOL），允许通过开关跳过；训练侧再做摊平
        total = sum(probs.values())
        if not allow_unnormalized and abs(total - 1.0) > PROB_TOL:
            # 消息必须含 qid 路径与"概率和"字样（spec 硬要求，便于日志 grep）。
            raise SchemaError(
                f"{key_path} 概率和 = {total!r}，偏离 1 超过容差 {PROB_TOL}（qid={qid}）"
            )
        if allow_unnormalized:
            out[qid] = probs
        else:
            # 摊平到严格 1：total==0 时上面已经因为 |0-1|>tol 被拒；此处 total 必接近 1。
            out[qid] = {opt: pv / total for opt, pv in probs.items()}
    return out


def _validate_question(qid: str, q: Any, path: list[str]) -> dict[str, Any]:
    """校验单题（type / instructions / criteria），返回一份新 dict 供上层聚合。"""
    # 白话：一"题"由三块组成——题目属于哪种玩法（三选一）、人类可读的问题正文、
    # 以及一份候选清单。choice 和 score 两种玩法必须有候选清单，noul 则可省。
    # 任何一块不对，都把"是哪道题的哪一格"报回去，别让调用方猜。
    qpath = path + [qid]
    if not isinstance(q, dict):
        raise SchemaError(f"{_fmt_path(qpath)} 必须是 dict（题目对象），实际是 {type(q).__name__}")
    # 未知字段拒（防拼错，如把 criteria 写成 critria 会静默通过，是隐患）。
    unknown = set(q.keys()) - _ALLOWED_Q
    if unknown:
        raise SchemaError(f"{_fmt_path(qpath)} 出现未知字段 {sorted(unknown)}；只允许 {sorted(_ALLOWED_Q)}")

    # type 必须存在且在枚举里；错误消息同时带上"实际值"与"合法枚举"（spec Scenario 3）。
    if "type" not in q:
        raise SchemaError(f"{_fmt_path(qpath + ['type'])} 缺失；合法 qtype 枚举为 {list(QTYPES)}")
    qtype = q["type"]
    if qtype not in QTYPES:
        raise SchemaError(
            f"{_fmt_path(qpath + ['type'])} = {qtype!r} 不在合法 qtype 枚举 {list(QTYPES)} 内"
        )

    instructions = _require_str(q.get("instructions"), qpath + ["instructions"], "题目正文")

    # criteria 按 qtype 差异化：choice/score 必须有、noul 可选。
    criteria = q.get("criteria")
    if qtype in ("choice", "score"):
        if criteria is None:
            raise SchemaError(f"{_fmt_path(qpath + ['criteria'])} 缺失（qtype={qtype} 必须声明选项/档位）")
        keys = _validate_criteria(criteria, qpath + ["criteria"])
        if qtype == "score":
            # 风险条目：score 的 k 从 criteria 键集合为准；键必须严格是 {"0".."k-1"}。
            expected = _score_expected_keys(len(keys))
            if set(keys) != expected:
                raise SchemaError(
                    f"{_fmt_path(qpath + ['criteria'])} score 键必须是 \"0\"..\"k-1\" 连续字符串；"
                    f"实际 {sorted(keys, key=_num_sort_key)}，期望 {sorted(expected, key=_num_sort_key)}"
                )
            if len(keys) < 2:
                raise SchemaError(f"{_fmt_path(qpath + ['criteria'])} score 至少 2 档，实际 {len(keys)}")
        else:  # choice
            if len(keys) < 2:
                raise SchemaError(f"{_fmt_path(qpath + ['criteria'])} choice 至少 2 个选项，实际 {len(keys)}")
        norm_criteria: dict[str, str] = dict(criteria)
    elif criteria is not None:
        # noul 允许携带 criteria，但不校验语义；只要格式合法即可（透传，兼容上游产物）。
        norm_criteria = dict(criteria) if isinstance(criteria, dict) else None
    else:
        norm_criteria = None

    result: dict[str, Any] = {"type": qtype, "instructions": instructions}
    if norm_criteria is not None:
        result["criteria"] = norm_criteria
    return result


def validate_sample(sample: Any, *, allow_unnormalized: bool = False) -> dict[str, Any]:
    """校验单个样本并返回归一化后的新 dict；违规抛 `SchemaError`。

    白话：把一份"题目+答案分数"的样本从头到尾检查一遍——state 得是文字、
    每道题得说清是什么玩法、分数得挂到真实存在的题上且加在一起等于 1。默认
    把分数摊平到严格 1（对付浮点漂移），跑评测时可关掉这一开关保留原分布。
    """
    # —— 顶层
    if not isinstance(sample, dict):
        raise SchemaError(f"<root> 必须是 dict（样本），实际是 {type(sample).__name__}")
    unknown = set(sample.keys()) - _ALLOWED_TOP
    if unknown:
        raise SchemaError(f"<root> 出现未知字段 {sorted(unknown)}；只允许 {sorted(_ALLOWED_TOP)}")

    state = _require_str(sample.get("state"), ["state"], "state")

    # —— questions
    questions_raw = sample.get("questions")
    if not isinstance(questions_raw, dict) or not questions_raw:
        raise SchemaError(
            "questions 必须是非空 dict（题目集合）" + (f"，实际是 {type(questions_raw).__name__}" if questions_raw is not None else "（缺失）")
        )
    questions: dict[str, Any] = {}
    for qid, q in questions_raw.items():
        if not isinstance(qid, str) or not qid.strip():
            raise SchemaError(f"questions 的键必须是字符串 qid，出现 {qid!r}")
        questions[qid] = _validate_question(qid, q, ["questions"])

    # —— targets（可选）
    out: dict[str, Any] = {"state": state, "questions": questions}
    targets_raw = sample.get("targets")
    if targets_raw is not None:
        out["targets"] = _normalize_targets(
            targets_raw, questions, ["targets"], allow_unnormalized=allow_unnormalized
        )

    # —— ep_group / ep_step：透传不校验语义（二阶段 RL 多轮用；spec 明确"不校验语义"）
    for passthrough_key in ("ep_group", "ep_step"):
        if passthrough_key in sample:
            out[passthrough_key] = sample[passthrough_key]
    return out


def dump(sample: dict[str, Any]) -> str:
    """把样本序列化为一行 jsonl（末尾带 `\\n`）。要求所有 dict 键均为 str。

    白话：写文件时一行装一个样本，读回来也是按行切。落盘前再走一遍校验，
    免得手改出来的脏样本静默进盘——json 会把非字符串键悄悄变成字符串，
    那种漂移在这里直接拒。
    """
    # 先归一再落盘：dump 的输入必须已经过校验，validate 是幂等的，代价可接受。
    normalized = validate_sample(sample)
    _assert_str_keys(normalized, [])
    return json.dumps(normalized, ensure_ascii=False) + "\n"


def _assert_str_keys(obj: Any, path: list[str]) -> None:
    """递归确保所有 dict 键都是 str（json 会把非 str 键静默 str 化，导致类型漂移）。"""
    # 白话：json 的规矩是"格子名必须是字符串"，Python 却允许你把 3、True、
    # (1,2) 当键塞进去；写盘时它会偷偷换成字符串，读回来就对不上原文。
    # 这里提前扫一遍，把这类会漂移的键揪出来报错，别等文件写完才追悔。
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise SchemaError(f"{_fmt_path(path + [repr(k)])} dict 键非字符串，会被 json 静默 str 化")
            _assert_str_keys(v, path + [k])
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_str_keys(v, path + [f"[{i}]"])


def load(line: str) -> dict[str, Any]:
    """把一行 jsonl 还原成样本；读回后**重新走一遍校验**（对齐 design 风险 2）。

    白话：从文件里掏出一行文字，把它拆回 Python 世界里的样本对象。拆完还得
    再检查一次——磁盘那趟路可能踩到人工编辑、截断或字段漂移，不复查就等于
    没设卡。检查失败同样抛 SchemaError，跟 validate_sample 保持一致。
    """
    stripped = line.strip()
    if not stripped:
        raise SchemaError("load 传入空行（jsonl 一行必须是完整样本的 json）")
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as e:
        raise SchemaError(f"load 无法解析 json（line={stripped[:60]!r}…）: {e}") from e
    return validate_sample(parsed)


def to_rank_vector(targets: dict[str, float]) -> list[float]:
    """把 score 类 targets（键 `"0".."k-1"`）按档位顺序展平成数值向量。

    白话：给一列"每档分到多少信心"，输出一个列表：第 0 档的信心、第 1 档的信心……
    一直排到最后一档。顺序按数字走，不按字符串排（"10" 不该排在 "2" 前面）。
    这样下游算 MAE、within-1 之类的分档指标就能直接把向量喂进去。
    """
    if not isinstance(targets, dict):
        raise SchemaError(f"to_rank_vector 需要 dict（score targets），实际是 {type(targets).__name__}")
    if not targets:
        raise SchemaError("to_rank_vector 收到空 dict（score 至少 1 档；validate_sample 会确保 >=2 档）")
    # 键必须是 "0".."k-1"：先反解成整数索引、再按索引回填，保证顺序不靠 dict 迭代次序。
    idx_map: dict[int, float] = {}
    for k, v in targets.items():
        if not isinstance(k, str) or not k.isdigit():
            raise SchemaError(f"to_rank_vector 键必须是 \"0\"..\"k-1\" 数字字符串，出现 {k!r}")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise SchemaError(f"to_rank_vector 值必须是数字，键 {k!r} 处是 {type(v).__name__}")
        idx_map[int(k)] = float(v)
    k = len(idx_map)
    if set(idx_map.keys()) != set(range(k)):
        raise SchemaError(
            f"to_rank_vector 键不连续：共 {k} 档但索引集合是 {sorted(idx_map.keys())}，期望 0..{k - 1}"
        )
    return [idx_map[i] for i in range(k)]


__all__ = [
    "PROB_TOL",
    "QTYPES",
    "SchemaError",
    "dump",
    "load",
    "to_rank_vector",
    "validate_sample",
]
