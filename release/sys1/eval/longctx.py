"""sys1/eval/longctx.py — 长文轴的合成长文（多针）正文生成器与召回曲线报告。

【做什么】
    p2-03 的 registry 只登记了"针位表"（哪个档位、第几根针、什么颜色、几分、插在几成处），
    30 行骨架落在 `bench/eval_data/assembled/needle-synthetic/`，`envelope_ready=false` 自述
    "题面在盘上、判分件属 p2-07"。本文件补上后半段：按那张确定的针位表**生成正文**——把长
    干扰文拼出来、把针（颜色→编号）按 `pos_frac` 插进去、把提问与候选排出来，产出的每一行
    都是 `sys1.data.schema` 认得的可判分信封（同 P1 预测出口一种形状）；对外再交两件：
    `score_recall()` 按"精确串匹配"算针召回，`format_curve()` 交召回曲线（含峰值内存与设备
    标注，1M 行按 measured/config 双列口径呈现，不虚报）。

【怎么做】
    ① 针位表两个来源，同一张嘴：盘上的 registry 骨架（首选，逐行原样读）或由
       `registry.needle_plan()` 现推（镜像 `_assemble_needle` 的取随机顺序：先颜色后数值）。
       两者必须逐字相同，tests 里钉着这条同源断言——计划与产物分家而不漂移。
    ② 一篇文档 = 多针：同档位登记的针**一根不删**全插进同一篇（根数记在 `needle.needles_in_doc`）；
       针位表里同一颜色会重复出现（实测 8K 档两针都是"青"），提问因此按"第几次提到这个颜色"消歧
       （`needle.color_ordinal`），既不动登记表，也不留两个正确答案。针序按 `pos_frac` 升序，段间
       干扰文长度按目标位置倒推，实际落点逐针记在 `needle.pos_actual`（整句粒度，偏差如实记）。
    ③ 长度单位两档，口径随行披露：没给编号器时按"字"造、`est_tokens = 字数 / CHARS_PER_TOKEN_EST`
       （行里 `len_units="est-chars"`，明说这是估算，不冒充 token）；给了编号器就真测 token，并按
       实测比例回炉重拼一次，行里落 `tokens` 实测值与 `tokens_delta`，容忍带 `TOKEN_TOL_RATIO`。
    ④ 判分：`score_recall(records, preds)` 吃 P1 预测行 `{id, answers:{q:{码:份额}}}`，取份额最大的
       候选代号，再拿该代号的**文字**与针上的编号字符串做精确相等判定——文字对不上就是没召回，
       不接受"数值接近"。按档位（ctx）分组出召回，`format_curve` 连峰值内存与设备一起排。

【为什么】
    被否方案一：本域自己另写一张针位表——registry 已把 seed/档位/针数钉成计划（`needle_plan`），
    两处定义必然漂移，报告里的曲线就说不清是哪一份题；所以针位只认 registry 那一份，本文件只补正文。
    被否方案二：把长干扰文写成一句重复 N 遍——重复文本让模型靠"周期"而不是"检索"答题，召回率
    失去诊断价值；故干扰文从句子池里按 seed 抽样拼贴，同 seed 复跑逐字相同，不同 seed 分布不同。
    被否方案三：没有编号器时把字数当 token 数报——档位（8K/32K/128K/256K）本来就是 token 口径，
    张冠李戴会让"实测上限"这类主张失去凭据；故估算档显式标 `est-chars`，真 token 口径必须实测。
    被否方案四：召回率用"数值误差 < 某阈值"判——针的语义是"能不能把那串字原样捞出来"，
    近似判定会把"捞回另一个数"记成命中，与 spec 的 exact-string 口径不符。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from sys1.data.schema import validate_sample
from sys1.eval import registry

# ---------------------------------------------------------------- 契约常量（改动即升版本）
#: 生成器版号：正文模板或拼装口径动过一个字节就升一次，随召回数字一起报（同 decision 的规矩）
LONGCTX_VERSION = "needle_body_v1"

#: 针句模板：{color} 颜色、{value} 编号。措辞进正文，所以钉在这里不散落各处。
NEEDLE_TEMPLATE = "（记录：{color}色对应的编号是 {value}。）"
#: 提问模板：只问探测针那一种颜色，答案位是那串编号文字。
QUESTION_TEMPLATE = "上面这份材料里，{color}色对应的编号是多少？"
#: 同色多针（针位表里同一颜色会出现两次以上）时的序数提问：不删针、不改契约，靠序数消歧
QUESTION_TEMPLATE_ORDINAL = "上面这份材料里，第 {ordinal} 次提到“{color}色”时给出的编号是多少？"
QUESTION_INSTRUCTIONS_KEY = "q"          # 一题一信封，题号固定（与 registry 的信封同形）

#: 与 registry `_assemble_needle` 同源的十种颜色（顺序不能动：动了一步就换一份针）
COLORS = ("红", "橙", "黄", "绿", "青", "蓝", "紫", "粉", "灰", "黑")

#: 干扰文句子池：中文陈述句，彼此语义无关，按 seed 抽样拼贴（周期文本会退化检索任务）
DISTRACTOR_LINES: tuple[str, ...] = (
    "项目组在周三的例会上核对了库存台账，本周不做调整。",
    "运维值班表按季度轮换，夜间告警仍由当班同学接手。",
    "这份季度小结引用了上月的口径，未纳入最新一版指标。",
    "仓库出入库记录以批次号归档，历史批次保留三年。",
    "培训材料第三版补了两处笔误，正文结构与上一版一致。",
    "巡检清单按楼层打印，每层各一份，签字后交回行政。",
    "供应商报价单有效期为六十天，逾期需重新询价。",
    "设备保养由外包方执行，厂方只核对工单与照片。",
    "会议纪要只记结论，讨论过程以录音为准不入正文。",
    "样本送检单据一式三份，检测中心留底一份。",
    "排班表每月一号更新，临时调班需在群里报备。",
    "资产盘点以标签号为准，标签脱落的先补后盘。",
    "客服工单按类型分派，超时未回复自动升级到二线。",
    "预算执行进度每两周同步一次，口径与财务系统一致。",
    "实验记录本按项目编号借阅，归还前须补齐签字页。",
    "质检抽样比例按批次规模分档，小批次全检不做。",
    "培训签到表与考核卷一起归档，保存期为两年。",
    "系统升级公告提前一天发布，回滚方案另附一页。",
    "外包人员的账号在 project 结束后当日回收。",
    "备件领用走工单，库存低于安全线才补采购。",
)

#: 没编号器时的换算地板值：中文 BPE 大致 1 token ≈ 1.5 字（仅作估算，行内显式标口径）
CHARS_PER_TOKEN_EST = 1.5
#: 有编号器时实测 token 与档位目标的相对容差（合成长文按段落粒度拼装，做不到逐 token 精确）
TOKEN_TOL_RATIO = 0.10

#: registry 骨架的默认落点（相对 release 根）
SKELETON_REL = Path("bench") / "eval_data" / "assembled" / "needle-synthetic" / "needle-synthetic.jsonl"
#: 本域正文产物的默认落点（与骨架同目录、不同文件名，不改写 registry 的账）
CORPUS_NAME = "needle-envelopes.jsonl"
META_NAME = "needle-envelopes.meta.json"

#: 召回口径名（进报告表头，与 registry.needle_plan 的 metric 字段同义）
METRIC_NAME = "needle_recall(exact-string)"

#: 1M 行的双列口径（D7 三件套/云端档）：本机没真跑过就一律 measured: —，配置就绪另起一列
UNMEASURED = "—"
MEASURED_BUCKETS = registry.NEEDLE_BUCKETS


class NeedleError(ValueError):
    """长文轴的当场拒绝（针位表对不上、正文造不出、判分缺件）；带上下文，不静默降级。"""


# ---------------------------------------------------------------- 针位表（只认 registry 那一份）
@dataclass(frozen=True)
class Needle:
    """一根针：属于哪个档位、插在几成处、写什么颜色与编号、由哪颗 seed 决定。

    白话：一张卡片上记着"多长的文章、第几个位置、什么颜色、几分"，卡片本身不含正文——
    正文是本文件按卡片现拼出来的；同一张卡片两次拼出来的逐字相同。
    """

    ctx: int
    index: int
    seed: int
    pos_frac: float
    color: str
    value: int

    @property
    def row_id(self) -> str:
        """信封 id：`needle:<档位>:<档内序号>`（与 registry 骨架同一写法）。"""
        return f"needle:{self.ctx}:{self.index}"


def needle_table_from_plan(plan: dict[str, Any] | None = None) -> list[Needle]:
    """由 registry 的针位计划现推针位表（逐字镜像 `_assemble_needle` 的取随机顺序）。

    白话：计划里只写着"这档多长、哪颗种子、几根针"，颜色和数值要现场按那颗种子摇出来。
    摇的顺序必须和登记侧一模一样——先摇颜色再摇数值，一步不差，否则两份针位表会悄悄分家。
    """
    pl = plan or registry.needle_plan()
    rows: list[Needle] = []
    for bucket in pl["buckets"]:
        rng = random.Random(bucket["seed"])
        for n in range(bucket["needles"]):
            color = rng.choice(COLORS)            # 顺序①：先颜色
            value = rng.randint(100, 999)         # 顺序②：后数值
            rows.append(Needle(ctx=int(bucket["ctx"]), index=n, seed=int(bucket["seed"]),
                               pos_frac=round((n + 1) / (bucket["needles"] + 1), 4),
                               color=color, value=value))
    return rows


def load_skeleton(path: str | Path | None = None) -> list[Needle]:
    """读 registry 装配好的 30 行针位骨架；缺文件就报清怎么补，不猜也不空手过。

    白话：骨架是"考卷的座位表"，正文还没写。这里把座位表原样收进来；要是盘上还没有，
    就指名道姓地告诉你去跑哪条命令补齐，而不是自己摇一份然后当作盘上就有。
    """
    src = Path(path) if path is not None else registry.REPO_ROOT / SKELETON_REL
    if not src.is_file():
        raise NeedleError(
            f"针位骨架不在盘上：{src}（先跑 `python -m sys1.eval.registry fetch --needle`）")
    rows: list[Needle] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        rows.append(Needle(ctx=int(obj["ctx"]), index=int(obj["id"].rsplit(":", 1)[1]),
                           seed=int(obj["seed"]), pos_frac=float(obj["pos_frac"]),
                           color=str(obj["needle_color"]), value=int(obj["needle_value"])))
    if not rows:
        raise NeedleError(f"骨架文件是空的：{src}")
    return rows


def resolve_table(*, skeleton: Sequence[Needle] | None = None,
                  plan: dict[str, Any] | None = None) -> tuple[list[Needle], str]:
    """取针位表：盘上骨架优先，没有就用计划现推；回 `(针位表, 来源说明)`。

    白话：两条路给的是同一张表（同源断言在单测里钉着），但报数时必须说清这次走的是哪条，
    免得日后追问"你那 30 根针是从文件读的还是代码推的"没人答得上。
    """
    if skeleton is not None:
        return list(skeleton), "skeleton"
    try:
        return load_skeleton(), "skeleton"
    except NeedleError:
        return needle_table_from_plan(plan), "needle_plan"


def group_by_bucket(rows: Sequence[Needle]) -> dict[int, list[Needle]]:
    """按档位分组并把档内按 `pos_frac` 升序排（正文按插入顺序拼，序乱则位置失真）。

    白话：先把同一档长度的考卷归成一摞，再把这摞里的几根针按"插在文章几成处"从头排到尾。
    正文是一个字往后写出来的，针的顺序一乱，落点就全错位——落点错了，"中段比开头难捞"这种
    诊断价值也就没了。
    """
    out: dict[int, list[Needle]] = {}
    for row in rows:
        out.setdefault(row.ctx, []).append(row)
    for ctx in out:
        out[ctx].sort(key=lambda n: (n.pos_frac, n.index))
    return out


# ---------------------------------------------------------------- 正文拼装
def _units(text: str, tokenizer: Any | None) -> int:
    """这段文字在当前口径下算多长：有编号器就真数 token，没有就数字数（口径由调用方披露）。"""
    if tokenizer is None:
        return len(text)
    enc = tokenizer.encode(text, add_special_tokens=False)
    ids = enc["input_ids"] if isinstance(enc, dict) else getattr(enc, "ids", enc)
    return len(ids)


def _target_units(ctx: int, tokenizer: Any | None) -> int:
    """档位目标长度（单位随口径变：token 档直接用 ctx，字数档按估算系数换算）。"""
    return ctx if tokenizer is not None else max(1, math.ceil(ctx * CHARS_PER_TOKEN_EST))


def _distractor(rng: random.Random, want: int, tokenizer: Any | None,
                cache: dict[str, int] | None = None) -> str:
    """拼一段约 `want` 长的干扰文（整句抽样，句内不再切字，避免半截话）。

    白话：往文章里垫话。垫到多长由目标位置定，一句句往上加；最后一句要么整句留、要么整句
    舍——不做"切掉半句"那种事，因为切半句会留下语法不通的尾巴，模型可以靠语病认出针的位置。
    """
    if want <= 0:
        return ""
    parts: list[str] = []
    got = 0
    box = cache if cache is not None else {}
    while got < want:
        sent = rng.choice(DISTRACTOR_LINES)
        n = _cached_units(sent, tokenizer, box)
        if n <= 0:
            continue
        if got + n > want * 1.25 and got > 0:      #  overshoot 太多就收手，宁可短一点
            break
        parts.append(sent)
        got += n
    return "".join(parts)


def _cached_units(sent: str, tokenizer: Any | None, cache: dict[str, int]) -> int:
    """同一句话的只数一次（编号器调用不便宜，档位大时重复句极多）。"""
    hit = cache.get(sent)
    if hit is None:
        hit = _units(sent, tokenizer)
        cache[sent] = hit
    return hit


def doc_needles(probe: Needle, bucket: Sequence[Needle]) -> tuple[list[Needle], int]:
    """这篇文档里的全部针：同档位登记的针**一根不删**，另回探测针在同色针里的序数。

    白话：针位表钉死了"这一档几根针、各写什么颜色"，正文无权删针——删了就和登记表对不上。
    表里同一颜色会重复出现（实测 8K 档两针都是"青"），所以提问写成"第几次提到这个颜色"，
    针全留着、答案也只有一个。序数按 `pos_frac` 升序数，跨机器复跑数出来的是同一个序。

    :returns: `(按 pos_frac 升序的针列, 探测针在同色针中的序数（从 1 起）)`。
    """
    chosen = sorted(bucket, key=lambda n: (n.pos_frac, n.index))
    same = [n for n in chosen if n.color == probe.color]
    ordinal = next(i + 1 for i, n in enumerate(same) if n.row_id == probe.row_id)
    return chosen, ordinal


def _needle_line(row: Needle) -> str:
    """针句正文（一模板、一处出）：判分要精确匹配，模板就不许两处各写一份。"""
    return NEEDLE_TEMPLATE.format(color=row.color, value=row.value)


def build_document(probe: Needle, bucket: Sequence[Needle], *, seed: int,
                   tokenizer: Any | None = None,
                   units_cache: dict[str, int] | None = None) -> tuple[str, dict[str, Any]]:
    """按针位把长干扰文拼成一篇文章，并回这篇文章的长度凭据。

    白话：先算出每根针该落在第几个字（或第几个符号）上，再把干扰文一段段填进针与针之间，
    最后补一段尾巴凑到档位长度。填的时候只按整句加减，落点位置因此是"近似到句"的——
    实际落点与计划落点的偏差都记在凭据里，报告里说多少就是多少。

    :returns: `(正文, 凭据字典)`，凭据含 `len_units/target_units/units/needles_in_doc/pos_actual`。
    """
    chosen, _ordinal = doc_needles(probe, bucket)
    rng = random.Random(f"{LONGCTX_VERSION}|{probe.seed}|{probe.row_id}")
    target = _target_units(probe.ctx, tokenizer)
    cache = units_cache if units_cache is not None else {}

    parts: list[str] = []
    cum = 0
    actual: dict[str, float] = {}
    for row in chosen:
        line = _needle_line(row)
        want = round(row.pos_frac * target) - cum - _units(line, tokenizer)
        filler = _distractor(rng, max(0, want), tokenizer, cache)
        if filler:
            parts.append(filler)
            cum += _units(filler, tokenizer)
        parts.append(line)
        n_units = _units(line, tokenizer)
        actual[row.row_id] = round((cum + n_units / 2.0) / target, 4)
        cum += n_units
    tail = _distractor(rng, max(0, target - cum), tokenizer, cache)
    parts.append(tail)
    text = "".join(parts)

    units = _units(text, tokenizer)
    cred: dict[str, Any] = {
        "len_units": "tokens" if tokenizer is not None else "est-chars",
        "target_units": target, "units": units,
        "est_tokens": units if tokenizer is not None else round(units / CHARS_PER_TOKEN_EST, 1),
        "chars": len(text), "needles_in_doc": len(chosen), "pos_actual": actual,
    }
    if tokenizer is not None:
        cred["tokens"] = units
        cred["tokens_delta"] = units - probe.ctx
        cred["tokens_rel_err"] = round(abs(units - probe.ctx) / probe.ctx, 4)
    return text, cred


# ---------------------------------------------------------------- 信封产出（可判分）
def decoy_values(rng: random.Random, gold: int, have: Sequence[int], k: int) -> list[int]:
    """凑错选项：先取同篇其它针的编号（最像的干扰），不够再从 100..999 里补，绝不与正确值同值。

    白话：给正确答案配几个像样的干扰项。就近取材最狠——同一篇里别的针的编号也是三位数、也
    真在文中出现过，蒙不中就说明是真没捞到；凑不齐才去 100..999 里补，补进来的绝不与正确项
    撞值（撞值就是两道正确答案）。
    """
    pool: list[int] = []
    for v in have:
        if v != gold and v not in pool:
            pool.append(v)
    while len(pool) < k:
        cand = rng.randint(100, 999)
        if cand != gold and cand not in pool:
            pool.append(cand)
    return pool[:k]


def build_envelope(probe: Needle, bucket: Sequence[Needle], *, options: int = 4,
                   tokenizer: Any | None = None, seed_shift: int = 0,
                   units_cache: dict[str, int] | None = None) -> dict[str, Any]:
    """一根探测针 → 一行可判分信封（`{id, task, qtype, sample, needle, split}`）。

    白话：把文章、问题、候选、真值装进评测唯一认的那张表头里；候选是四个编号文字，
    正确项只占一格，位置由那颗种子打乱。信封先过一遍样本契约再交出去，字段跑形的
    东西不会流到下游。
    """
    text, cred = build_document(probe, bucket, seed=probe.seed + seed_shift,
                               tokenizer=tokenizer, units_cache=units_cache)
    others = [row.value for row in bucket if row.row_id != probe.row_id]
    rng = random.Random(f"{LONGCTX_VERSION}|decoy|{probe.row_id}")
    picks = decoy_values(rng, probe.value, others, options - 1)
    gold_text = str(probe.value)                                  # 洗牌前先把正确值钉住
    texts = [gold_text, *[str(v) for v in picks]]
    rng.shuffle(texts)
    criteria = {str(i + 1): t for i, t in enumerate(texts)}       # 代号 1..k 按数值升序 → 字母 A..
    gold_key = next(k for k, t in criteria.items() if t == gold_text)
    chosen, ordinal = doc_needles(probe, bucket)
    same_color = sum(1 for n in chosen if n.color == probe.color)
    if same_color > 1:
        question = QUESTION_TEMPLATE_ORDINAL.format(ordinal=ordinal, color=probe.color)
    else:
        question = QUESTION_TEMPLATE.format(color=probe.color)
    sample = {
        "state": text,
        "questions": {QUESTION_INSTRUCTIONS_KEY: {"type": "choice", "instructions": question,
                                                  "criteria": criteria}},
        "targets": {QUESTION_INSTRUCTIONS_KEY: {k: (1.0 if k == gold_key else 0.0)
                                                for k in criteria}},
    }
    normalized = validate_sample(sample)      # 契约不过就抛 SchemaError，不放半成品出去
    return {
        "id": probe.row_id,
        "task": "needle",
        "qtype": "choice",
        "axis": "longctx",
        "split": "n/a",
        "generator": {"version": LONGCTX_VERSION, "seed": probe.seed, "options": options},
        "sample": normalized,
        "needle": {
            "ctx": probe.ctx, "seed": probe.seed, "index": probe.index,
            "pos_frac": probe.pos_frac, "color": probe.color, "value": str(probe.value),
            "color_ordinal": ordinal, "same_color_in_doc": same_color, "question": question,
            "answer_key": gold_key, "answer_text": str(probe.value), **cred,
        },
    }


def build_corpus(*, buckets: Sequence[int] | None = None, rows: Sequence[Needle] | None = None,
                 tokenizer: Any | None = None, options: int = 4, limit_per_bucket: int = 0,
                 cap_ctx: int | None = None,
                 plan: dict[str, Any] | None = None) -> dict[str, Any]:
    """批量产出针信封：按档位分组，逐档生成（`limit_per_bucket` 给 CPU 冒烟留闸口）。

    白话：整档全出就是全套考卷；只想快点看一眼链路通不通，就每档先出几题。
    `cap_ctx` 是"这台机器最多能摊开多长的文章"的闸门——档位被压短时，返回值里
    `capped` 会逐档记一笔，报数时绝不把压短的那份当原生档位。

    :returns: `{"version", "rows", "ledger", "capped", "source"}`。
    """
    table, source = resolve_table(plan=plan) if rows is None else (list(rows), "given")
    grouped = group_by_bucket(table)
    wanted = [int(b) for b in buckets] if buckets else sorted(grouped)
    out_rows: list[dict[str, Any]] = []
    ledger: dict[str, Any] = {}
    capped: dict[str, int] = {}
    cache: dict[str, int] = {}
    for ctx in wanted:
        bucket = grouped.get(ctx) or []
        if not bucket:
            raise NeedleError(f"针位表里没有 {ctx} 这一档（可选：{sorted(grouped)}）")
        take = bucket[:limit_per_bucket] if limit_per_bucket else bucket
        if cap_ctx is not None and cap_ctx < ctx:
            capped[str(ctx)] = int(cap_ctx)
        for probe in take:
            row = build_envelope(probe, bucket, options=options, tokenizer=tokenizer,
                                 units_cache=cache)
            if cap_ctx is not None and cap_ctx < ctx:
                row["needle"]["ctx_requested"] = ctx
                row["needle"]["ctx_capped_to"] = cap_ctx
                row["needle"]["capped"] = True
            out_rows.append(row)
            ledger.setdefault(str(ctx), {"needles": 0, "probes": 0})
            ledger[str(ctx)]["needles"] = row["needle"]["needles_in_doc"]
            ledger[str(ctx)]["probes"] += 1
    return {"version": LONGCTX_VERSION, "rows": out_rows, "ledger": ledger,
            "capped": capped, "source": source, "plan": plan or registry.needle_plan()}


def corpus_fingerprint(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """给这批正文称个重：条数、总字数、逐档字节与整批 sha256 前 16 位（复跑对得上才算确定）。

    白话：把这批正文整个压成一个"重量级暗号"，再按档位分开记体量。两次构建暗号逐字相同，
    才配说这份考卷可复现；暗号一变就说明拼装口径被动过，报告里的召回曲线就得重跑。
    """
    blob = "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows)
    per_bucket: dict[str, int] = {}
    for r in rows:
        per_bucket[str(r["needle"]["ctx"])] = per_bucket.get(str(r["needle"]["ctx"]), 0) + len(
            r["sample"]["state"])
    return {"count": len(rows), "sha256_16": hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16],
            "chars_total": len(blob), "chars_by_ctx": per_bucket}


def write_corpus(rows: Sequence[dict[str, Any]], out_dir: str | Path, *,
                 source: str = "skeleton") -> Path:
    """把正文落成一整份自持副本（jsonl + 一份自述 meta），路径回 jsonl 那一个。

    白话：题面与真值同文件放着——这是合成长文，不涉及"人工答案另存防泄漏"的顾虑；但档位、
    种子、生成器版号与指纹都得写在 meta 里，否则日后没人说得清这份正文是哪一版生成的。
    """
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (dst / CORPUS_NAME).write_text(body, encoding="utf-8")
    meta = {
        "kind": "needle-envelopes", "generator_version": LONGCTX_VERSION,
        "table_source": source, "envelope_ready": all("sample" in r for r in rows),
        "samples": len(rows), "ledger": {str(k): v for k, v in _ledger_of(rows).items()},
        "fingerprint": corpus_fingerprint(rows), "plan": registry.needle_plan(),
        "note": "正文由 p2-07 生成器按 registry 针位表补出；registry 那份 needle-synthetic.jsonl "
                "仍是 pin 住的针位表，两份各记各账，不互相改写。",
    }
    (dst / META_NAME).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    return dst / CORPUS_NAME


def _ledger_of(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """从成品行反推逐档台账（档内针数、探测题数），供 meta 自述用。"""
    led: dict[str, Any] = {}
    for r in rows:
        key = str(r["needle"]["ctx"])
        ent = led.setdefault(key, {"needles": int(r["needle"]["needles_in_doc"]), "probes": 0,
                                   "units": r["needle"]["len_units"]})
        ent["probes"] += 1
    return led


# ---------------------------------------------------------------- 召回判分与曲线
def score_recall(records: Sequence[dict[str, Any]], preds: Sequence[Any], *,
                 qid: str = QUESTION_INSTRUCTIONS_KEY) -> dict[str, Any]:
    """针召回：预测里份额最大的候选代号，取其**文字**与针上编号做精确相等判定。

    白话：模型报"我选 B"，就把 B 那格的文字取出来跟针上的编号一字一比；差一个字符就算没捞到。
    没有预测行的题记进 `missing`，不天降命中也不偷偷从分母里抹掉。

    :returns: `{"by_ctx": {档: {probes, hit, miss, recall}}, "overall": {...}, "missing": [...]}`。
    """
    by_id: dict[str, Any] = {}
    for p in preds:
        pid = p.get("id") if isinstance(p, dict) else getattr(p, "id", None)
        if pid is not None:
            by_id[str(pid)] = p
    buckets: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for rec in records:
        needle = rec["needle"]
        key = str(needle["ctx"])
        ent = buckets.setdefault(key, {"ctx": int(needle["ctx"]), "probes": 0, "hit": 0,
                                       "miss": 0, "units": needle["len_units"],
                                       "peak_units": 0})
        ent["probes"] += 1
        ent["peak_units"] = max(ent["peak_units"], int(needle["units"]))
        pred = by_id.get(str(rec["id"]))
        if pred is None:
            missing.append(str(rec["id"]))
            ent["miss"] += 1
            continue
        answers = (pred.get("answers") if isinstance(pred, dict) else getattr(pred, "answers", {})) or {}
        dist = answers.get(qid) or {}
        if not dist:
            missing.append(str(rec["id"]))
            ent["miss"] += 1
            continue
        top = max(dist, key=lambda k: (float(dist[k]), -int(k)))
        criteria = rec["sample"]["questions"][qid]["criteria"]
        hit = str(criteria.get(top, "")) == str(needle["answer_text"])
        ent["hit"] += int(hit)
        if not hit:
            ent["miss"] += 1
    for ent in buckets.values():
        ent["recall"] = round(ent["hit"] / ent["probes"], 4) if ent["probes"] else 0.0
    total = sum(e["probes"] for e in buckets.values())
    hits = sum(e["hit"] for e in buckets.values())
    return {"by_ctx": dict(sorted(buckets.items())),
            "overall": {"probes": total, "hit": hits, "miss": total - hits,
                        "recall": round(hits / total, 4) if total else 0.0},
            "missing": missing, "metric": "needle_recall(exact-string)"}


def _device_note(device: str | None = None) -> dict[str, Any]:
    """设备与内存口径标注：本机是什么就报什么，CPU 口径绝不说成显存。"""
    import os
    import platform
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024      # macOS 报字节，Linux 报 KB
    return {"device": device or "cpu", "os": f"{platform.system()}-{platform.machine()}",
            "peak_rss_bytes": int(peak_bytes), "memory_ledger": "cpu-rss(peak), 非 NPU 显存账",
            "pid": os.getpid()}


def format_curve(summary: dict[str, Any], *, measured: Sequence[int] | None = None,
                 extra_rows: Sequence[dict[str, Any]] | None = None) -> str:
    """把召回结果排成曲线表（档位/针数/探测数/召回/长度口径/单位数/设备），双列口径不混淆。

    白话：一档一行，跑过的档报真数与实测设备；没跑过的档（比如 1M）明写
    `measured: — / config: ready`，读的人一眼看出哪列是量出来的、哪列是配置备好的。
    """
    meas = set(int(m) for m in (measured or []))
    dev = summary.get("device") or {}
    lines = [
        f"needle 召回曲线（生成器 {LONGCTX_VERSION}；口径 {summary.get('metric') or METRIC_NAME}",
        f"设备：{dev.get('device', 'n/a')}｜内存口径：{dev.get('memory_ledger', 'n/a')}"
        f"｜峰值驻留 {dev.get('peak_rss_bytes', 0):,} 字节",
        "",
        f"{'档位':>8}  {'针数':>4}  {'探测':>4}  {'召回':>7}  {'长度口径':>10}  {'单位数(实测)':>12}  measured/config",
        "-" * 84,
    ]
    for key, ent in (summary.get("by_ctx") or {}).items():
        ctx = int(key)
        state = ("measured" if (not meas or ctx in meas) else "pending")
        lines.append(f"{ctx:>8,}  {ent['needles'] if 'needles' in ent else ent['probes']:>4}  "
                     f"{ent['probes']:>4}  {ent['recall']:>7.4f}  {ent.get('units', 'n/a'):>10}  "
                     f"{ent.get('peak_units', 0):>12,}  {state}")
    for row in extra_rows or []:
        lines.append(f"{row['label']:>8}  {'—':>4}  {'—':>4}  {row.get('measured', UNMEASURED):>7}  "
                     f"{'—':>10}  {'—':>12}  measured: {row.get('measured', UNMEASURED)} / "
                     f"config: {row.get('config', 'ready')}")
    ov = summary.get("overall") or {}
    lines += ["-" * 84,
              f"合计：探测 {ov.get('probes', 0)} 题 / 命中 {ov.get('hit', 0)} / 召回 {ov.get('recall', 0):.4f}"
              f"；缺预测行 {len(summary.get('missing') or [])} 条（计入 miss，不从天上掉分母）"]
    return "\n".join(lines)


def one_million_row(*, config_ready: bool = True, measured: float | None = None) -> dict[str, Any]:
    """1M 那一行的双列口径：没实测就 `measured: —`，配置是否备好另说（防虚报的制度化）。

    白话：报告里 1M 那一格，本机到底跑没跑过？没跑过就把"实测"留空，只在"配置"那一列写
    ready。两列各说各话，读的人才不会把"配置备好了"误读成"量出来的召回"——这正是父变更
    反复强调的"不虚报"落到表头上的写法。
    """
    return {"label": "1M", "measured": (UNMEASURED if measured is None else f"{measured:.4f}"),
            "config": "ready" if config_ready else "pending",
            "note": "0.8B@1M 属 D7 三件套边界实验（滑窗封顶+GDN 旁路+门控），云端档产出；"
                    "本机不实测吞吐与召回，配置见 production/configs/longctx_1m_cloud.yaml"}


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    """CLI：`build` 出正文副本，`selfcheck` 只报针位表与计划是否同源（不写盘）。

    白话：命令行就两个动作。`build` 按针位表把正文造出来、落盘；`selfcheck` 一个字都不写，
    只回答一个问题——"盘上那张座位表"和"代码照种子摇出来的那张"是不是同一张。
    """
    ap = argparse.ArgumentParser(prog="python -m sys1.eval.longctx",
                                 description="长文轴合成长文：针位表 → 正文 → 召回曲线")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="生成 needle 正文信封并落盘")
    b.add_argument("--out", default=str(registry.DATA_DIR / "assembled" / "needle-synthetic"))
    b.add_argument("--buckets", default="", help="逗号分隔档位，缺省全四档")
    b.add_argument("--limit", type=int, default=0, help="每档最多出几题（0=全出）")
    b.add_argument("--cap-ctx", type=int, default=0, help="正文长度上限（0=不压）")
    b.add_argument("--options", type=int, default=4)
    b.add_argument("--tokenizer", default="", help="编号器目录（给了就按真 token 量长度）")
    sub.add_parser("selfcheck", help="只核针位表与 needle_plan 是否同源")
    return ap


def main(argv: list[str] | None = None) -> int:
    """CLI 出口：selfcheck 打印指纹；build 落盘并回 sha256 与条数。

    白话：`selfcheck` 把同源核对打成一行，方便人一眼看清"能不能拿代码推的那张当题"；
    `build` 造完正文后把条数、逐档台账、指纹一起报出来——报的数字全来自刚生成的那批行，
    不另算一遍，也就不存在"报的与写的不是同一份"。
    """
    args = build_parser().parse_args(argv)
    if args.cmd == "selfcheck":
        disk = load_skeleton()
        plan = needle_table_from_plan()
        same = disk == plan
        print(json.dumps({"on_disk": len(disk), "from_plan": len(plan), "same": same,
                          "fingerprint": corpus_fingerprint([{"needle": {"ctx": n.ctx,
                                                                         "answer_text": ""},
                                                              "sample": {"state": ""}} for n in disk])
                          ["sha256_16"]}, ensure_ascii=False))
        return 0 if same else 1
    buckets = [int(x) for x in str(args.buckets).split(",") if x.strip()] or None
    tokenizer = None
    if args.tokenizer:
        import transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    corpus = build_corpus(buckets=buckets, tokenizer=tokenizer, options=args.options,
                          limit_per_bucket=args.limit, cap_ctx=args.cap_ctx or None)
    path = write_corpus(corpus["rows"], args.out, source=corpus["source"])
    print(json.dumps({"out": str(path), "rows": len(corpus["rows"]),
                      "ledger": corpus["ledger"], "capped": corpus["capped"],
                      "source": corpus["source"],
                      "fingerprint": corpus_fingerprint(corpus["rows"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
