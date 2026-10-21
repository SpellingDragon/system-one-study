"""sys1/eval/chinese.py — 中文子集 → 决策信封的转写件（CMMLU / CLUE tnews / CLUE ocnli）。

【做什么】
    p2-03 的 registry 已经把 CMMLU 200 题与 CLUE 400 题的**原样题面副本**钉在盘上
    （`bench/eval_data/assembled/cmmlu-subset/`、`.../clue-subset/`，账上写着
    `envelope_ready=false`，登记话自述"决策化成信封由 p2-09 承接"）。本文件补后半段三件事：
    ① 逐条把那张原样副本誊成统一决策信封（choice / noul 二型），键约定与真值折算全部走
    `sys1/data/transcribe` 既有件，产出经 registry 装配口落盘并记账（条数 / 逐题型 / 真值覆盖 /
    sha256）；② G3 中文对照表骨架（laya 列与失败样本挂列就位，DML 列显式标"待 C5"）；
    ③ 配比两档（30% / 50%）的参数校验与卫生闸（训练配比不得引用评测集）。

【怎么做】
    ① 题型规则一律从盘上那份 pin 取，不发明标签：
       CMMLU 四选一 → choice，criteria = `{A,B,C,D: 选项文本}`，真值 = 源里的 answer 字母；
       CLUE tnews 十五类 → choice，候选代号与文字都取源 parquet 自带的 ClassLabel names
       （实测为 CLUE 官方类别码 `"100".."116"`），真值 = label 下标映到的那一枚码；
       CLUE ocnli 三分类 → noul，先用源 ClassLabel names（实测
       `["neutral","entailment","contradiction"]`）把下标折成关系名，再交
       `transcribe.XNLI_TO_NOUL` 折成二值（只有蕴含算 true，中立与矛盾统统 false）。
       这三条映射写死在本模块常量里（`TNEWS_LABEL_CODES` / `OCNLI_RELATIONS`），并有用例
       拿盘上 parquet 的 metadata 逐位对拍——上游改版即红，不靠记忆也不靠外部文献。
    ② 信封本体由转写既有件产出：`classification_to_choice`（options+label → criteria + one-hot
       真值）、`xnli_to_noul`（premise/hypothesis/label → false/true）。本层只补三样，且都
       只补在行级或问法级、不动真值：中文任务自己的问法（`CMMLU_INSTRUCTIONS`，改完重新过一遍
       样本契约）、noul 的 `criteria={"false","true"}` 两格说明（与 registry 的 noul 信封同形，
       否则打分层拆出 k=0 的假题）、行级 `qtype` 与 `task`（registry 的分桶与按集记账只认这两个
       字段；task 用 `zh-cmmlu`/`zh-tnews`/`zh-ocnli` 三个通道名，跨集不混算）。id 原样带着源
       副本的 `cmmlu:<学科>:<档位>/<行号>`、`clue:<任务>:<档位>:<行号>`，逐条回溯可查。
    ③ 注册即上轴、复跑即重建：两个派生集（`cmmlu-decision` / `clue-decision`）在 registry 走
       derived 路——零流量（不下载，按已 pin 的子集副本现推），assembler 指回本模块，于是
       `python -m sys1.eval.registry fetch --zh`（含在 `--all` 里）一键重装配。底账（样本数 /
       qtype_counts / gold_coverage / assembled_bytes / sha256 / envelope_ready）由 registry
       registry 那套唯一的记账代码写，本模块不自记一份，免得两处底账互相改写。转写口径变了就升
       `ZH_DECISION_VERSION`，并把版本号写进派生 pin 的 note——报告里的中文分据此能追到规则版本。
       D3 再补一路**训练档**：`clue-train-subset → clue-train-decision`（registry 的 train 轴，
       经 `load_train_records()` 消费）。折法一条不改，只把档位另登一格——同一条规则分别喂
       train 与 validation，训测隔离才成立；CMMLU 经探查上游确无 train 分割（取证见
       `registry.ZH_TRAIN_PROBE`），本模块不拿 dev 档改名凑数。
    ④ G3 骨架：`build_g3_md()` 按集（cmmlu / clue）各出一块表，指标行取 DML / laya / Δ 三列，
       laya 那一格复用 p2-04 的亲跑产物口径（`baselines/table.py::load_row` 读
       `<root>/laya/<split>.row.json`）；读到就填实测值并挂 run-id 与采样参数，读不到一律
       写 `—`/`待 C5`，绝不拿别处的数顶。失败样本挂列由 `pick_failure_samples()` 出（拆题走
       打分层同一个 `rows_from_envelopes`，口径不分叉），没预测时交三行显式占位。
    ⑤ 配比两档：`load_mix_config()` 读 yaml 并把键分成两类——`production/sft.py` 现在就认的
       键、与本项目待接入的 `zh_ratio` / `zh_corpus` 两键；`assert_mix_hygiene()` 拦住
       "把评测集当训练语料"（D1/p2-05 教训：训测同集出假分），`compare_mix_configs()` 钉住
       "两档除配比外逐键相同"。C5 真要跑时，配置里待接入的两键由 p2-05 侧消费口接上。

【为什么】
    被否方案一：就在 p2-03 那份 `cmmlu-subset.jsonl` 上原地改成信封——同一条目既是原始题面
       凭据又是消费产物，p2-03 的 `cn_sets_carry_human_gold` 用例按原字段（answer/label）核对
       真值覆盖，信封行没这两字段，改一场就砸一场；且原样副本是"源头长什么样"的证据，被覆盖
       就再也拿不回来。故另立派生 id，原样副本与决策信封各占一格。
    被否方案二：把 tnews 的类别码翻成"科技/体育/财经"这类中文类别名——盘上 pin 的源头只给了
       码，没给"码↔类别名"对照表；凭记忆补一套名字等于把臆测写进真值口径，一旦与上游不符，
       整列中文 acc 失去意义。故 criteria 的文字就用源头给的码本身，并把这条局限写进 pin note
       与本报告脚注（宁可难看，不可含糊）。
    被否方案三：本模块自己算 sha256、自己往 manifest 里塞一条账——registry 已有唯一的记账口
       （`_store_assembled`），两处各记必然漂移，报告数字就说不清是哪本账。
    被否方案四：现在就把 laya 拉起来出中文分——laya 亲跑通路 p2-04 已打通，但 G3 的 DML 行
       = 配比消融的优者，须待正式 SFT 产物（C5）；本波摆半张有数的表比摆一张全占位的表更容
       易被误读成"已经对照过了"。故只做骨架与挂列，实测格一律显式标注来源与缺口。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sys1.data.schema import validate_sample
from sys1.data.transcribe import TranscribeError, classification_to_choice, xnli_to_noul
from sys1.eval import registry
from sys1.eval.baselines.table import load_row as load_baseline_row

#: 转写版号：动过任一条映射规则（候选口径、问法、真假折法）就升一次，随中文分一起报出。
ZH_DECISION_VERSION = "zh_decision_v1"

#: CMMLU 的候选代号（源 csv 就是 A/B/C/D 四枚；上限 4，渲染层一轮装得下）。
CMMLU_OPTION_KEYS = ("A", "B", "C", "D")
#: CMMLU 是知识选择题不是"归类"，故问法自己写；只改 instructions 一栏，候选与真值一律不动。
CMMLU_INSTRUCTIONS = "这道题的正确选项是哪一个？"
# tnews 是主题归类，转写既有件的默认问法（"这段话属于哪一类？"）本就对味：故 `tnews_to_envelope`
# 不传 instructions，这里也不留一个恒为 None 的假常量（看着像开关，实则没人读它）。

#: CLUE tnews 十五类的类别码，逐字取自盘上 pin 的 parquet ClassLabel names（对拍用例见
#: `tests/test_chinese.py::test_transcribe_label_tables_match_source`）。**源头只给码不给名**，
#: 故候选文字沿用码本身——见【为什么】被否方案二。
TNEWS_LABEL_CODES: tuple[str, ...] = (
    "100", "101", "102", "103", "104", "106", "107", "108", "109", "110",
    "112", "113", "114", "115", "116",
)
#: CLUE ocnli 三分类的关系名，同样取自盘上那份 parquet 的 metadata（下标顺序即源头顺序）。
#: 折成 noul 时只有 entailment 记 true，其余两关系记 false——规则本身住在 transcribe 里。
OCNLI_RELATIONS: tuple[str, ...] = ("neutral", "entailment", "contradiction")
#: noul 的两格说明文字：与 registry 的 noul 信封同形（打分层按 criteria 拆 k，缺格会拆出 k=0）。
NOUL_CRITERIA = {"false": "不成立", "true": "成立"}

#: 源副本集 id → 决策信封集 id（registry 里成对登记；派生集零流量，按左边那份现推）。
SOURCE_TO_DECISION = {"cmmlu-subset": "cmmlu-decision", "clue-subset": "clue-decision"}
DECISION_TO_SOURCE = {v: k for k, v in SOURCE_TO_DECISION.items()}
#: 中文 **train 语料**那一格（D3）：与上面那对考卷分开设表——`--zh` 按钮、quality 轴的账都
#: 按考卷那对走，混进同一张表会把评测面一起改了；两张表合并成 `ALL_SOURCE_TO_DECISION` 供
#: 名字换算与转写分派用（换算是一张表，记账是两张表，各管各的）。
TRAIN_SOURCE_TO_DECISION = {"clue-train-subset": "clue-train-decision"}
ALL_SOURCE_TO_DECISION = {**SOURCE_TO_DECISION, **TRAIN_SOURCE_TO_DECISION}
ALL_DECISION_TO_SOURCE = {v: k for k, v in ALL_SOURCE_TO_DECISION.items()}
#: CLUE 系的两个档位（考卷 validation / 语料 train）：转写折法完全同一条，只是原件来自哪一档。
CLUE_SOURCE_PIDS = ("clue-subset", "clue-train-subset")
#: 各源副本缺件时的补救命令（把"先敲哪条"写进报错里，别让下游拿空表算出 0 分当实测）。
FETCH_HINT = {"cmmlu-subset": "fetch --cn", "clue-subset": "fetch --cn",
              "clue-train-subset": "fetch --zh-train"}
#: 按集记账的通道名（`scoring.score_rows` 拿行的 task 当分账口，跨集不混算靠它）。
ZH_CHANNEL = {"cmmlu": "zh-cmmlu", "tnews": "zh-tnews", "ocnli": "zh-ocnli"}

QID = "q"                                   # 一题一信封，题号固定（与 registry 的信封同形）
CLUE_CHOICE_TASKS = ("tnews",)              # CLUE 里当多选题的那几档
CLUE_NOUL_TASKS = ("ocnli",)                # CLUE 里当是非题的那几档

#: G3 的切分名：一集一切分，C5 由 p2-04 的亲跑设施按同名写出 `<split>.row.json`。
G3_SPLITS = {"cmmlu-decision": "zh-cmmlu", "clue-decision": "zh-clue"}
G3_SETS = ("cmmlu-decision", "clue-decision")
#: 表上的指标行（键名与 p2-04 row.json 的 metrics 同源，禁止另立第二套口径）。
G3_METRIC_ROWS = (("acc", "准确率", 4), ("ece_after", "ECE(调温后)", 4),
                  ("ms_p50", "逐题耗时中位(ms)", 1), ("throughput_qps", "吞吐(题/秒)", 3))
G3_DML_LABEL = "DML(配比优者)"
PENDING_LABEL = "待 C5"                     # 本波做不了的那几格统一挂这块牌子
FAILURE_SAMPLE_SLOTS = 3                    # 崩溃域证据的挂列位数（spec：失败样本 3 例）
FAILURE_STATE_CHARS = 80                    # 题面摘录长度（截断展示，不整段外抄）

#: 配比两档里待 p2-05 侧接入的配置键（其余键 `production/sft.py` 今天就认）。
MIX_PENDING_KEYS = ("zh_ratio", "zh_corpus")
MIX_RATIOS = (0.30, 0.50)
#: `zh_corpus` 本波仍挂显式占位符：registry 的 train 轴自 D3 起已有中文语料档
#: （`clue-train-subset`/`clue-train-decision`，2026-10-06 登记并真拉装配），但把这一格从 PENDING
#: 切到真集属 p2-09 B1 的接入动作（配比口径由训练侧定），本模块不越权改。占位符能被卫生闸
#: 读出来是"缺口在案"，随手写成某个评测集当语料才是假绿——不复用任何考卷凑数。
MIX_CORPUS_PENDING = f"PENDING@{PENDING_LABEL}:zh-train"


class ChineseTranscribeError(ValueError):
    """源副本不满足转写前提时抛出（沿用 ValueError 族，与 TranscribeError/SchemaError 同处置）。"""


# ---------------------------------------------------------------- 源副本读取
def decision_id(source_pid: str) -> str:
    """源副本集 id → 决策信封集 id（没登记过的名字当场报错，不静默回一个怪 id）。

    白话：中文轨有两张面孔——p2-03 那份"题面长什么样"的原样副本，和 p2-09 这份"能直接判分"的
    信封。这个函数只干一件事：把前者的名字翻成后者的名字。翻不出来说明这条集根本没登过记，
    当场停下来问，比硬拼一个不存在的路径让人去猜要省心得多。
    """
    if source_pid not in ALL_SOURCE_TO_DECISION:
        raise ChineseTranscribeError(
            f"没有为 {source_pid!r} 登记决策化去向；已登记的源副本：{sorted(ALL_SOURCE_TO_DECISION)}")
    return ALL_SOURCE_TO_DECISION[source_pid]


def source_id(decision_pid: str) -> str:
    """决策信封集 id → 它的源副本集 id（回溯"这批题面是从哪份 pin 誊来的"）。

    白话：分数要能追问出处，就得能从"考卷"倒回"题面原件"：这一集是从 p2-03 钉过版本的
    那份副本誊出来的，源副本的 revision、取的是哪一档、总共几条，都记在原件那一格账上。
    名字没登记过就当场报错，绝不猜一个看着顺眼的原件。
    """
    if decision_pid not in ALL_DECISION_TO_SOURCE:
        raise ChineseTranscribeError(
            f"{decision_pid!r} 不是本模块登记的中文决策集；可选：{sorted(ALL_DECISION_TO_SOURCE)}")
    return ALL_DECISION_TO_SOURCE[decision_pid]


def subset_path(source_pid: str) -> Path:
    """p2-03 那份自持子集副本的落盘路径（`assembled/<id>/<id>.jsonl`，与 registry 同口径）。"""
    return (registry.DATA_DIR / registry.ASSEMBLED_DIR_NAME / source_pid
            / f"{source_pid}.jsonl")


def load_subset(source_pid: str) -> list[dict[str, Any]]:
    """读回原样子集副本的每一行；文件不在就抛带补救命令的错，不交一张空表冒充"读过了"。

    白话：把中文题面那一摞卡片一张张读进内存。读不到不等于"这套题是空的"——那说明还没 fetch
    过，p2-03 的副本才是唯一的题源；所以这里把该敲的命令写进错误消息里，让人照着做就能续上，
    而不是让下游拿着空列表算出 0% 的中文准确率还当成实测。
    """
    path = subset_path(source_pid)
    if not path.is_file():
        raise ChineseTranscribeError(
            f"中文子集副本不在盘上：{path}（先跑 `python -m sys1.eval.registry "
            f"{FETCH_HINT.get(source_pid, 'fetch --cn')}`）")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ChineseTranscribeError(f"中文子集副本是空文件：{path}")
    return rows


# ---------------------------------------------------------------- 信封小件
def _q(sample: dict[str, Any]) -> dict[str, Any]:
    """取信封里那唯一一题（本模块一题一行，题号恒为 `q`）。"""
    return sample["questions"][QID]


def _revalidate(env: dict[str, Any]) -> dict[str, Any]:
    """改过问法/候选说明之后重新过一遍样本契约，把成品钉回"合法信封"。"""
    # 白话：转写件已经把样本校验过一遍了，但本层还会动两处文字（问法、是非题的两格说明）。
    # 动过就得再查一遍——契约是"候选说明不许空白、概率和必须为一"这类毛病的唯一捕手，
    # 省掉这一步等于把改坏的样本直接端给打分层，出问题时报在哪儿都费劲。
    env["sample"] = validate_sample(env["sample"])
    return env


def _finish(env: dict[str, Any], *, channel: str, instructions: str | None = None,
            criteria: dict[str, str] | None = None, **extra: Any) -> dict[str, Any]:
    """把转写出的裸信封补成 registry 口径的行：行级 qtype/task、问法与候选说明、溯源字段。"""
    # 白话：转写件交回的是"样本 + 一个 id"，而评测那边要认的是带题型、带分账名字的一行。
    # 这一步就是把那几样外衣穿上：题型抄进 `qtype`（registry 数逐题型条数只认这个口），
    # 分账名字写进 `task`（按集记账靠它，跨集不许混算），要换问法或补候选说明就在这里换，
    # 换完立刻重过一遍契约。真值与候选代号一个字节都不动——动那些就改分数了。
    q = _q(env["sample"])
    if instructions:
        q["instructions"] = instructions
    if criteria is not None:
        q["criteria"] = dict(criteria)
    sample = _revalidate(env)["sample"]             # 动过文字，重新过一遍样本契约
    row = {"id": env["id"], "task": channel, "qtype": q["type"], "sample": sample}
    row.update(extra)                               # subject / task_name 之类的溯源字段原样带上
    return row


# ---------------------------------------------------------------- 三型转写
def cmmlu_to_envelope(row: dict[str, Any]) -> dict[str, Any]:
    """CMMLU 一行（题干 + 四候选 + 正确字母）→ choice 信封（k=4，criteria=选项文本）。

    白话：一道中文知识选择题，题干当证据、四个选项铺成候选、正确那个字母押满分其余写零。
    折法完全走转写既有件（它顺手管两件事：候选少于两个不收、正解不在候选里不收），
    本层只把问法换成"这道题的正确选项是哪一个？"——因为既有件默认那句是"归类"问法，
    用在知识选择题上会变成答非所问。
    """
    question = str(row.get("question") or "").strip()
    options = row.get("options")
    if not isinstance(options, dict) or not options:
        raise ChineseTranscribeError(f"CMMLU 行缺候选清单：{row.get('id')}")
    picked = {str(k): str(v).strip() for k, v in options.items() if str(v).strip()}
    missing = [k for k in CMMLU_OPTION_KEYS if k not in picked]
    if missing:
        raise ChineseTranscribeError(f"CMMLU 行 {row.get('id')} 缺候选 {missing}（四选一不该缺项）")
    criteria = {k: picked[k] for k in CMMLU_OPTION_KEYS}
    env = classification_to_choice({"id": row["id"], "text": question, "options": criteria,
                                    "label": row.get("answer")})
    return _finish(env, channel=ZH_CHANNEL["cmmlu"], instructions=CMMLU_INSTRUCTIONS,
                   subject=str(row.get("subject") or "unknown"),
                   zh_source="cmmlu-subset")


def tnews_to_envelope(row: dict[str, Any]) -> dict[str, Any]:
    """CLUE tnews 一行（标题 + 类别下标）→ choice 信封（k=15，候选=源头给的类别码）。

    白话：一条新闻标题配十五个主题类别码，问它属于哪一类。下标先按源头给的顺序换成码本身
    （第 8 号下标 → "109" 这类），再交转写既有件铺成候选与 one-hot 真值。这里最克制的一点是
    **不替类别码起中文名**：源头没给对照表，起了就是猜，猜错整列分都废。
    """
    index = _label_index(row)
    if index >= len(TNEWS_LABEL_CODES):
        raise ChineseTranscribeError(
            f"tnews 标签下标 {index} 越出本模块登记的类别码表（{len(TNEWS_LABEL_CODES)} 枚）"
            f"——上游改版了，先对拍 metadata 再改常量：{row.get('id')}")
    codes = {code: code for code in TNEWS_LABEL_CODES}
    env = classification_to_choice({"id": row["id"], "text": str(row.get("sentence") or "").strip(),
                                    "options": codes, "label": TNEWS_LABEL_CODES[index]})
    return _finish(env, channel=ZH_CHANNEL["tnews"], task_name="tnews",
                   label_code=TNEWS_LABEL_CODES[index])


def ocnli_to_envelope(row: dict[str, Any]) -> dict[str, Any]:
    """CLUE ocnli 一行（两句 + 三分类下标）→ noul 信封（蕴含=true，其余=false）。

    白话：给两句话和它们的关系（中立 / 蕴含 / 矛盾），折成"前句成立时后句成不成立"的判断题：
    只有蕴含记真，中立与矛盾都记假——这条折法住在 transcribe 的映射表里，本层只负责把源头
    的下标换成关系名，再补上 false/true 两格说明（缺那两格，打分层会把这题拆成零候选的假题）。
    """
    index = _label_index(row)
    if index >= len(OCNLI_RELATIONS):
        raise ChineseTranscribeError(
            f"ocnli 标签下标 {index} 越出本模块登记的关系名表（{len(OCNLI_RELATIONS)} 枚）"
            f"——上游改版了，先对拍 metadata 再改常量：{row.get('id')}")
    env = xnli_to_noul({"id": row["id"], "premise": str(row.get("sentence1") or "").strip(),
                        "hypothesis": str(row.get("sentence2") or "").strip(),
                        "label": OCNLI_RELATIONS[index]})
    return _finish(env, channel=ZH_CHANNEL["ocnli"], task_name="ocnli",
                   relation=OCNLI_RELATIONS[index], criteria=dict(NOUL_CRITERIA))


def _label_index(row: dict[str, Any]) -> int:
    """把源副本里的真值读成整数下标（`gold` 优先，退回 `label`）；读不出下标即拒。"""
    # 白话：p2-03 那份副本把答案原样抄成了字符串（"8"、"2"），也可能留着数字。这里只认
    # "能当行号用的那种写法"：整数就整数，能转成整数的字符串也收下；其余（空、"-1" 那种
    # 官方藏着答案的档、看着像文字的）一律当场退回——宁可少一条题，不肯多一条猜出来的题。
    raw = row.get("gold")
    if raw is None:
        raw = row.get("label")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise ChineseTranscribeError(f"{row.get('id')} 没有真值下标（gold/label 皆空）")
    try:
        index = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ChineseTranscribeError(f"{row.get('id')} 真值读不成下标：{raw!r}") from exc
    if index < 0:
        raise ChineseTranscribeError(f"{row.get('id')} 真值下标为负（{index}）——无真值的档一律不收")
    space = [str(s) for s in (row.get("label_space") or [])]
    if space and str(index) not in space:
        raise ChineseTranscribeError(
            f"{row.get('id')} 真值下标 {index} 不在源副本的标签空间 {space} 内（来源形态变了？）")
    return index


def clue_to_envelope(row: dict[str, Any]) -> dict[str, Any]:
    """CLUE 一行按它自带的 `task_name` 分派（tnews→choice / ocnli→noul）。

    白话：CLUE 那份副本是几个任务混在一起的一摞卡，每张卡上写着自己是哪个任务。这里照着
    任务名交给会办的那种折法；任务名对不上（上游新加了一个任务）就停下来问，绝不因为
    "看着像多选题"就顺手硬折——折错了没人能从分数上看出来，只能等报告被人质疑才发现。
    """
    name = str(row.get("task_name") or "").strip()
    if name in CLUE_CHOICE_TASKS:
        return tnews_to_envelope(row)
    if name in CLUE_NOUL_TASKS:
        return ocnli_to_envelope(row)
    raise ChineseTranscribeError(
        f"CLUE 行 {row.get('id')} 的任务名 {name!r} 未登记决策化规则；"
        f"已登记：choice={list(CLUE_CHOICE_TASKS)} noul={list(CLUE_NOUL_TASKS)}")


def transcribe_subset(source_pid: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """一份原样子集副本 → 一摞决策信封（逐条转写，一条折不动就整批停下来说清是哪条）。

    白话：整套中文题面逐条誊成能判分的样子。p2-03 的副本自己标着"本域不决策化"，
    题型提示（qtype）只是给人看的；这里不看提示猜折法，只看任务名与字段——提示与事实
    不一致时（比如 ocnli 三分类被提示成 choice），以规则的登记方为准，走不到的那条会
    当场抛错而不是将就出一条假题。任何一条转写失败都把 id 带进消息里，好定位到那一行。
    """
    if source_pid == "cmmlu-subset":
        return [cmmlu_to_envelope(r) for r in rows]
    if source_pid in CLUE_SOURCE_PIDS:          # validation 考卷与 train 语料共用这一条折法
        return [clue_to_envelope(r) for r in rows]
    raise ChineseTranscribeError(
        f"未登记的中文源副本集 {source_pid!r}；可选：{sorted(ALL_SOURCE_TO_DECISION)}")


# ---------------------------------------------------------------- registry 装配钩子
def assemble_cmmlu_decision(raw: Path) -> list[dict[str, Any]]:
    """registry 装配口（cmmlu-decision）：读 p2-03 那份 CMMLU 副本，交转写层出信封。

    :param raw: registry 传下来的原始检出目录（派生集零流量，这里不读它，只读兄弟集的副本）。
    :raises registry.AssembleError: 源副本不在盘上或某条转不动时抛出（消息带可执行的补救命令）。

    白话：这个名字挂在 registry 的装配表上，作用是"复跑 fetch 就能把中文信封重出来一遍"。
    派生集没有要下载的东西，所以 raw 那个目录里压根没有文件，题面从 cmmlu-subset 那份已
    pin 的副本里读——版本凭证在那儿，不在这儿再钉一份，免得两处 pin 各说各话。
    """
    del raw
    try:
        rows = load_subset("cmmlu-subset")
    except ChineseTranscribeError as exc:
        raise registry.AssembleError(str(exc)) from exc
    return transcribe_subset("cmmlu-subset", rows)


def assemble_clue_decision(raw: Path) -> list[dict[str, Any]]:
    """registry 装配口（clue-decision）：读 p2-03 那份 CLUE 副本，tnews/ocnli 各按各的折法。

    白话：同上，只是这一份里面两种题型混着（十五类主题 + 三分类蕴含），分派在
    `clue_to_envelope` 里按任务名走；装配失败一律报成 registry 的装配错，让 fetch 把那
    个集记进 failed 栏而不是静默出一半。
    """
    del raw
    try:
        rows = load_subset("clue-subset")
    except ChineseTranscribeError as exc:
        raise registry.AssembleError(str(exc)) from exc
    return transcribe_subset("clue-subset", rows)


def assemble_clue_train_decision(raw: Path) -> list[dict[str, Any]]:
    """registry 装配口（clue-train-decision，D3）：读 train 档原件，交同一条折法出训练信封。

    白话：和 `assemble_clue_decision` 是同一门手艺，只是题目从 train 那份副本里读。这样
    "练的题"与"考的题"由同一条规则各写一遍，差别只在档位——若两份信封由两套规则写出来，
    训测之间的可比性当场没了，而分数上看不出这种破口。装配失败仍报成 registry 的装配错，
    让 fetch 把这个集记进 failed 栏，绝不静默出一半。
    """
    del raw
    try:
        rows = load_subset("clue-train-subset")
    except ChineseTranscribeError as exc:
        raise registry.AssembleError(str(exc)) from exc
    return transcribe_subset("clue-train-subset", rows)


def load_decision_records(decision_pid: str, *, limit: int = 0) -> list[dict[str, Any]]:
    """读回已落盘的中文决策信封（同 harness 同打分口的消费入口）。

    白话：中文轨的成品怎么被用？还是走 p2-03 那套唯一的读题口——一行一题、字段一个不加工。
    这里只负责"按集名把那份文件读进来"，读不到就明说还没装配，好让上层知道自己该先跑哪条命令。
    """
    path = (registry.DATA_DIR / registry.ASSEMBLED_DIR_NAME / decision_pid
            / f"{decision_pid}.jsonl")
    if not path.is_file():
        raise ChineseTranscribeError(
            f"中文决策信封不在盘上：{path}（先跑 `python -m sys1.eval.registry fetch --zh`）")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return rows[:limit] if limit else rows


# ---------------------------------------------------------------- G3 中文对照表（骨架 + 挂列）
def collect_g3_rows(*, root: Path | None = None, splits: dict[str, str] | None = None,
                    baseline: str = "laya") -> dict[str, dict[str, Any] | None]:
    """按集取基线亲跑行（laya 走 p2-04 的 row.json 口径）；没跑过的集如实回 None。

    白话：G3 表上 laya 那一列的数不是拍脑袋来的，是从 p2-04 已经打通的那条亲跑通路上
    落的记录卡里取的（同 harness、同采样、带 run-id）。这张卡现在还没有——中文集是这一波
    才决策化出来的，还没人在 C5 拿它跑过一遍，于是这里取到的是"没有"，表上就写短横。
    取不到不丢人，取不到还填一个数才丢人。
    """
    root = root or (Path(__file__).resolve().parents[2] / "sys1" / "eval" / "baselines")
    splits = splits or G3_SPLITS
    return {pid: load_baseline_row(root, baseline, splits[pid]) for pid in splits}


def fmt(value: Any, digits: int, *, pending: str = PENDING_LABEL) -> str:
    """把一个指标摊成表里的字符串；没实测的一律成占位牌，不当 0 也不当空。

    白话：G3 这张表上，"还没跑"与"跑了是 0"必须一眼分得开。所以不是数字的东西（没值、
    空着、压根是个字符串）统统换成那块写着缺口的牌子，而不是 0——0 会被读成一个真分数。
    """
    if value is None or isinstance(value, bool):
        return pending
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return pending


def pick_failure_samples(records: list[dict[str, Any]], preds: dict[str, dict[str, float]],
                         *, limit: int = FAILURE_SAMPLE_SLOTS,
                         state_chars: int = FAILURE_STATE_CHARS) -> list[dict[str, Any]]:
    """挑失败样本（模型名次 ≠ 人工名次）至多 `limit` 例，题面按字数截断展示。

    :param records: 决策信封行（一题一行，本模块的产物形状）。
    :param preds: 预测出口那种形状 `{ "<id>/<qid>": {候选代号: 份额} }`。
    :returns: 逐例 `{id, qtype, k, gold, pred, state}`；`state` 已按 `state_chars` 截断。

    白话："laya 中文会崩"这句结论要有凭据，凭据就是几道答错题的原样摘录：题面、人工押哪一格、
    模型押哪一格。拆题面与真值这一步用的是打分层同一个摊平函数（`rows_from_envelopes`），
    这样"被判成错的那一格"跟分数表里的口径完全一致，不存在两套判定。没给预测就一条也挑不出——
    挑不出就交空表，由上层写成显式占位，绝不编三条看着像崩掉的例子。
    """
    from sys1.eval import scoring

    flat = scoring.rows_from_envelopes(records)
    out: list[dict[str, Any]] = []
    for r in flat:
        pred = (preds or {}).get(r["id"])
        if not pred or not r["keys"]:
            continue
        keys = r["keys"]
        gold = max(keys, key=lambda k: (float(r["gold"].get(k, 0.0)), -keys.index(k)))
        got = max(keys, key=lambda k: (float(pred.get(k, 0.0)), -keys.index(k)))
        if gold == got:
            continue
        state = ""
        for rec in records:
            if rec.get("id") == r["id"].rsplit("/", 1)[0]:
                state = str((rec.get("sample") or {}).get("state") or "")
                break
        out.append({"id": r["id"], "qtype": r["qtype"], "k": r["k"], "gold": gold, "pred": got,
                    "state": state[:state_chars] + ("…" if len(state) > state_chars else "")})
        if len(out) >= limit:
            break
    return out


def _failure_lines(samples: list[dict[str, Any]]) -> list[str]:
    """失败样本挂列（三格；没数就写占位，格子本身先摆好）。"""
    lines = ["| # | 题号 | 题型/k | 人工 | 模型 | 题面摘录（截断） |", "|---|---|---|---|---|---|"]
    for i in range(FAILURE_SAMPLE_SLOTS):
        if i < len(samples):
            s = samples[i]
            lines.append(f"| {i + 1} | `{s['id']}` | {s['qtype']}/k={s['k']} | {s['gold']} | "
                         f"{s['pred']} | {s['state'] or '—'} |")
        else:
            lines.append(f"| {i + 1} | {PENDING_LABEL} | {PENDING_LABEL} | {PENDING_LABEL} | "
                         f"{PENDING_LABEL} | 待 laya 中文亲跑落预测 jsonl 后挂载 |")
    return lines


def build_g3_md(*, rows: dict[str, dict[str, Any] | None] | None = None,
                failure_samples: dict[str, list[dict[str, Any]]] | None = None,
                ledger: dict[str, Any] | None = None,
                pending: str = PENDING_LABEL) -> str:
    """生成 G3 中文对照表 markdown 骨架（DML 列 + laya 列 + Δ 列 + 出处 + 失败样本挂列）。

    白话：这张表要摆的是"同一把尺下，自训的 DML 与 laya 在中文题上各得几分"。本波只有骨架：
    中文集刚从题面副本决策化出来，正式 SFT 产物（配比消融的优者）要等 C5，laya 也还没在这
    两集上亲跑过，于是所有实测格都挂着 `待 C5` 或短横，而列名、指标行、出处行、失败样本的
    三格挂位一个不少——到 C5 只往格子里填数，不必再改表的结构。ledger 传 manifest 的中文条目
    是为了把"这套题到底几道、哪来的"写在脚注里；拿不到就写"账上无名"，不猜条数。
    """
    rows = rows if rows is not None else collect_g3_rows()
    failure_samples = failure_samples or {}
    parts = [
        "# G3 中文对照表（DML vs laya · 同 harness 亲跑）",
        "",
        f"> 波次边界：本表由 p2-09 本地半场出**骨架**；DML 行取自中文配比消融的优者（30%/50% 两档，"
        f"须待正式 SFT 产物），laya 行走 p2-04 已打通的亲跑通路，两者实测格均标 `{pending}`，"
        "不以替身数字或卡面值顶替。",
        f"> 转写版号 `{ZH_DECISION_VERSION}`；题型 choice/noul 二型，分桶口径 (qtype, k)，"
        "禁止跨 k 合并。",
        "",
    ]
    for pid in G3_SETS:
        row = rows.get(pid)
        metrics = (row or {}).get("metrics") or {}
        pin = registry.REGISTRY.get(pid)
        parts += [f"## {pid}（split=`{pin.split if pin else '—'}`，切分名 `{G3_SPLITS[pid]}`）", "",
                  f"| 指标 | {G3_DML_LABEL} | laya | Δ(laya−DML) |", "|---|---|---|---|"]
        for _key, label, digits in G3_METRIC_ROWS:
            parts.append(f"| {label} | {pending} | {fmt(metrics.get(_key), digits, pending='—')} | "
                         f"{pending} |")
        parts += ["", "### 运行出处", "",
                  "| 基线 | 数据集/切分 | 样本数 | 采样参数 | 设备 | run-id | source | gate |",
                  "|---|---|---|---|---|---|---|---|"]
        parts.append(f"| {G3_DML_LABEL} | {pid}/{G3_SPLITS[pid]} | {pending} | {pending} | "
                     f"{pending} | {pending} | 待 C5（配比消融优者） | false |")
        if row:
            parts.append(
                "| laya | {ds}/{sp} | {n} | {smp} | {dev} | {rid} | {src} | {gate} |".format(
                    ds=row.get("dataset", pid), sp=row.get("split", G3_SPLITS[pid]),
                    n=row.get("n_samples", "—"), smp=row.get("sampling_params", "—"),
                    dev=row.get("on_device", "—"), rid=row.get("run_id", "—"),
                    src=row.get("source", "—"), gate="true" if row.get("gate") else "false"))
        else:
            parts.append(f"| laya | {pid}/{G3_SPLITS[pid]} | — | — | — | — | 未跑（{pending}） "
                         f"| false |")
        parts += ["", "### 崩溃域证据：失败样本（至多 3 例，题面截断展示）", ""]
        parts += _failure_lines(failure_samples.get(pid) or [])
        parts.append("")
    parts += ["## 脚注", ""]
    for pid in G3_SETS:
        entry = ((ledger or {}).get("sets") or {}).get(pid) or {}
        if entry:
            counts = entry.get("qtype_counts") or {}
            parts.append(f"- `{pid}`：{entry.get('samples', '—')} 题（choice={counts.get('choice', 0)} "
                         f"noul={counts.get('noul', 0)}），真值覆盖 "
                         f"{(entry.get('gold_coverage') or {}).get('with_gold', '—')}，"
                         f"sha256 `{str(entry.get('assembled_sha256'))[:12]}…`，"
                         f"源副本 `{DECISION_TO_SOURCE[pid]}`")
        else:
            parts.append(f"- `{pid}`：账上无名（还没装配——先跑 `python -m sys1.eval.registry fetch --zh`）")
    parts += ["- tnews 的候选文字沿用源头给的类别码（`\"100\"..\"116\"`），盘上 pin 的 parquet 没带"
              "\"码↔类别名\"对照表，本域不凭记忆补名字——这条局限随中文 acc 一起披露。",
              "- CLUE 取 validation 不取 test：实测 test 档 label 整列为 -1（官方隐藏答案），"
              "没有真值的题面只能当语料不能当考卷（口径由 p2-03 裁定并登记）。",
              f"- D11 边界：laya 行仅作对照评测推理，同 harness 同采样参数（`{PENDING_LABEL}` 的格子"
              "不许拿卡面值填）。", ""]
    return "\n".join(parts)


# ---------------------------------------------------------------- 配比两档（参数化 + 卫生闸）
def load_mix_config(path: str | Path) -> dict[str, Any]:
    """读一份中文配比档，拆成"现在就合法的配置键"与"待 p2-05 接入的配比键"两栏。

    白话：配比档是给人看的、也是给训练脚本读的。本波训练脚本还不认那两个配比键（消费口在 C5
    接），所以这里把键分成两堆：一堆现在就过 `production/sft.py` 的白名单校验（证明这份配置
    除了配比以外全是真的），另一堆是本域新增、显式点名的。混成一堆最坏：有人把整份配置喂给
    sft，它报错，于是有人把键名改成脚本认得的假键——那才是把消融做歪。
    """
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ChineseTranscribeError(f"配比档须是键值表：{path}")
    from production.sft import default_cfg as sft_default_cfg

    known = set(sft_default_cfg())
    pending = {k: data[k] for k in MIX_PENDING_KEYS if k in data}
    legal = {k: v for k, v in data.items() if k not in pending}
    unknown = sorted(set(legal) - known)
    if unknown:
        raise ChineseTranscribeError(
            f"{path} 含 `production/sft.py` 与 p2-09 都没登记的键 {unknown}（拼错一个字母就会静默失效）")
    if set(pending) != set(MIX_PENDING_KEYS):
        raise ChineseTranscribeError(f"{path} 必须齐备配比键 {list(MIX_PENDING_KEYS)}，实得 {sorted(pending)}")
    ratio = float(pending["zh_ratio"])
    if round(ratio * 100) not in {int(round(x * 100)) for x in MIX_RATIOS}:
        raise ChineseTranscribeError(f"{path} 配比 {ratio} 不在设计给定的两档 {list(MIX_RATIOS)} 内")
    return {"path": str(path), "config": legal, "mix": pending, "zh_ratio": ratio}


def assert_mix_hygiene(doc: dict[str, Any]) -> None:
    """配比档卫生闸：训练语料不得引用任何评测轴上的集（训测同集会出假分）。

    白话：中文配比说的是"往 SFT 语料里掺几成中文"，掺进去的东西必须来自训练档；要是图省事
    把刚决策化出来的中文**考卷**当训练语料，训练就见过考题，中文 acc 会漂亮得离谱。这一闸
    专门拦这一类：点名到评测轴上的集一律拒绝；还没注册的中文 train 档只能挂显式占位符。
    """
    corpus = doc["mix"].get("zh_corpus")
    items = [corpus] if isinstance(corpus, str) else list(corpus or [])
    eval_ids = {pid for pid, pin in registry.REGISTRY.items() if pin.axis != registry.TRAIN_AXIS
                and pin.kind != "synthetic"}
    for name in items:
        if str(name).startswith("PENDING@"):
            continue                                  # 显式缺口：在案可查，不算违规
        if name in eval_ids:
            raise ChineseTranscribeError(
                f"配比档把评测集 {name!r} 当训练语料（训测同集=假分）；"
                f"中文 train 档须另登记，未登记就挂 {MIX_CORPUS_PENDING}")
        if name not in registry.REGISTRY:
            raise ChineseTranscribeError(
                f"配比档引用的 {name!r} 没在 registry 登记；未就绪就挂显式占位 {MIX_CORPUS_PENDING}")


def compare_mix_configs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """两档对照：除 `zh_ratio` 外必须逐键相同（消融只许动一个因子）。

    白话：配比消融的全部说服力都在"其余都一样"这句上——同 seed、同超参、同底座、同题量，
    只有中文占比是 30% 与 50% 的差别。这个函数把两份配置摊平比一遍，把差在其他键上的情况
    逐条点名（哪怕只差一个 threads），免得日后有人顺手改了 lr 还当成配比差异带来的涨分。
    """
    diffs = {k: (a["config"].get(k), b["config"].get(k))
             for k in set(a["config"]) | set(b["config"]) if a["config"].get(k) != b["config"].get(k)}
    return {"ratio_a": a["zh_ratio"], "ratio_b": b["zh_ratio"],
            "identical": not diffs, "unexpected_diffs": diffs}


# ---------------------------------------------------------------- CLI
def build_ledger() -> dict[str, Any]:
    """装配中文决策信封（走 registry 唯一的下载+装配+记账口），回账本与逐集底账。

    白话：一句话把中文决策化跑完：派生集点名交给 registry.fetch，它按登记的装配钩子读题面、
    转信封、落盘、算散列、数条数，再把结果写进 manifest。这里只把两份集的结果拆成"到手"与
    "没到手"两栏交回，失败的原因原样带着——装配失败比拉不到数据更该停下来（说明源形态变了）。
    """
    result = registry.fetch(tuple(SOURCE_TO_DECISION.values()))
    return {"ok": result["ok"], "failed": result["failed"], "manifest": result["manifest"]}


def build_parser() -> argparse.ArgumentParser:
    """搭出中文轨 CLI 的参数表（build / g3 / check-mix 三个子命令）。

    白话：把这张表上能敲的三条路摆清楚：要重建中文信封就 build（零流量，本地现推）；要出
    G3 骨架就 g3 并留一个输出路径；配比两档写得对不对就 check-mix 一比。三条都不碰网络，
    也不自带题量或配比的参数——那些都在登记与配置里，改那儿才算改口径。
    """
    parser = argparse.ArgumentParser(prog="sys1/eval/chinese.py",
                                     description="中文专项：决策化转写 / G3 骨架 / 配比两档自检")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="按 registry 装配中文决策信封并记账")
    b.add_argument("--force", action="store_true", help="忽略盘上已有副本，重新装配一遍")
    g = sub.add_parser("g3", help="生成 G3 中文对照表骨架")
    g.add_argument("--out", default=str(registry.REPO_ROOT / "production" / "baselines"
                                        / "g3_chinese_table.md"), help="骨架落盘路径")
    m = sub.add_parser("check-mix", help="校验配比两档（除 zh_ratio 外逐键相同 + 卫生闸）")
    m.add_argument("paths", nargs="*", help="配比档 yaml（默认取 production/configs/zh_mix_*.yaml）")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：build / g3 / check-mix 三条路，失败一律非零退出并列出原因。

    白话：三条路各自把结果打在标准输出上，成没成、写到哪、几条题各是什么型，一次说清。
    任一个集装配失败就退出非零——中文信封少一个集，G3 表与配比消融的分母就不齐，
    这种事不能靠"看起来跑完了"混过去。
    """
    args = build_parser().parse_args(argv)
    if args.cmd == "build":
        led = build_ledger()
        for pid, entry in led["ok"].items():
            print(f"[zh] {pid}: samples={entry.get('samples')} "
                  f"qtypes={entry.get('qtype_counts')} gold={(entry.get('gold_coverage') or {}).get('with_gold')} "
                  f"bytes={entry.get('assembled_bytes')} sha={str(entry.get('assembled_sha256'))[:16]}…")
        for pid, msg in led["failed"].items():
            print(f"[zh] FAILED {pid}: {msg}", file=sys.stderr)
        return 1 if led["failed"] else 0
    if args.cmd == "g3":
        md = build_g3_md(ledger=registry.load_manifest())
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"[g3] 写入 {out}（{len(md.splitlines())} 行，实测格标 {PENDING_LABEL}）")
        return 0
    paths = args.paths or sorted((registry.REPO_ROOT / "production" / "configs").glob("zh_mix_*.yaml"))
    docs = [load_mix_config(p) for p in paths]
    for doc in docs:
        assert_mix_hygiene(doc)
        print(f"[mix] {doc['path']}: zh_ratio={doc['zh_ratio']} 合法键 {len(doc['config'])} 项 "
              f"待接入键 {sorted(doc['mix'])}")
    if len(docs) == 2:
        cmp = compare_mix_configs(docs[0], docs[1])
        print(f"[mix] 两档对照 ratio {cmp['ratio_a']} vs {cmp['ratio_b']} "
              f"除配比外相同={cmp['identical']} 意外差异={cmp['unexpected_diffs']}")
        if not cmp["identical"]:
            return 1
    return 0


__all__ = [
    "CLUE_CHOICE_TASKS",
    "CLUE_NOUL_TASKS",
    "CMMLU_INSTRUCTIONS",
    "CMMLU_OPTION_KEYS",
    "ChineseTranscribeError",
    "DECISION_TO_SOURCE",
    "FAILURE_SAMPLE_SLOTS",
    "G3_SETS",
    "G3_SPLITS",
    "MIX_CORPUS_PENDING",
    "MIX_PENDING_KEYS",
    "MIX_RATIOS",
    "NOUL_CRITERIA",
    "OCNLI_RELATIONS",
    "PENDING_LABEL",
    "SOURCE_TO_DECISION",
    "TNEWS_LABEL_CODES",
    "ZH_CHANNEL",
    "ZH_DECISION_VERSION",
    "ALL_DECISION_TO_SOURCE",
    "ALL_SOURCE_TO_DECISION",
    "CLUE_SOURCE_PIDS",
    "TRAIN_SOURCE_TO_DECISION",
    "assemble_clue_decision",
    "assemble_clue_train_decision",
    "assemble_cmmlu_decision",
    "build_g3_md",
    "build_ledger",
    "collect_g3_rows",
    "decision_id",
    "fmt",
    "cmmlu_to_envelope",
    "load_decision_records",
    "load_mix_config",
    "load_subset",
    "assert_mix_hygiene",
    "compare_mix_configs",
    "ocnli_to_envelope",
    "pick_failure_samples",
    "source_id",
    "subset_path",
    "tnews_to_envelope",
    "clue_to_envelope",
    "transcribe_subset",
]


if __name__ == "__main__":
    sys.exit(main())
