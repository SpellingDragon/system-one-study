"""sys1/eval/predict.py — 评测唯一预测出口：本地模型目录或远端服务，两种模式交同一种行。

【做什么】
    把"一批决策样本 + 一个被测对象"变成一份逐样本的预测行：每行记着样本号、每个候选
    各分到几成把握、这一行花了多少毫秒。被测对象可以是仓库里的一个模型目录（自己载权重
    现场读末位），也可以是一个在线的 `/v1/systemone` 服务地址（发请求拿回同样的行）。
    顺带干第二件不许绕过去的事：查清这个模型目录是谁生成的（来源 run 号与代码版本），
    查不到就拒绝出数——报告里的数字必须先能指回一次真实的实验。

【怎么做】
    本地路走四步。① 溯源体检 `resolve_provenance()`：依次翻模型目录的 `config.json` 来源
    字段、随附的 `decision_config.json`（含 s3 写回的 `calibration.run_id`）、上一层 run
    账本 `config.yaml`/`system.json` 的 run_id 与 commit，三处都空即抛
    `MissingProvenanceError`；`--allow-missing-run-id` 只在冒烟时放行并打上豁免标记。
    ② 编码 `encode_record()`：样本经 decision 渲染层排成定死文字、编号器换成编号串，同时
    记下"第几个字母代表哪个候选"与人工份额（与字母列同序、和为 1）。③ 读数：按候选数分桶
    组批右补空洞，一次前向交逐位置数字，`option_scores` 收真末位点字母行，同一份分数折两次
    份额——按类型倍数除过的（对外答案）与没除的（给校准轴当"调整前"）。④ 落地：对外只交
    `{id, answers, ms}` 三键的 jsonl，内部另留一份对齐好的富样本供四轴复用。
    远端路只做协议转换：把样本切成 `{id, state, questions}` 的请求体发出去，把回包按本地
    候选序对齐成同一种行；网络调用点可注入替换，单测用假回包验协议，不依赖真服务。

【为什么】
    预测行是所有轴的唯一分母：质量、把握、耗时、跨后端一致率都从同一批行走出来，换个
    被测对象（自训小模型 / 二阶段大模型 / 别人的服务）只换入口，不换口径。溯源检查放在
    取数之前而不是写报告之后，是因为"无 run-id 视为捏造"这条红线要在数字产生前拦住——
    数字一旦落盘就会被复制粘贴到别处，事后追认等于没追。
    被否方案一：让各轴各自载模型各自读数——同一份前向写五遍，读数口径必然漂移，跨轴数字
    加起来对不上，且换 backbone 时五处同改。
    被否方案二：预测行里顺带把人工标准答案也写进去——评测集与预测集混成一份文件，泄漏
    风险从"约定"降级为"没人会注意"，故标准答案只留在内部富样本里，不进 jsonl。
    被否方案三：溯源缺失时给个默认 run-id 继续跑——等于把红线变成建议；宁可退出码非零，
    并把"怎么补上来源"写进错误消息。
"""
from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from sys1.calibrate import DEFAULT_BINS
from sys1.data.schema import validate_sample
from sys1.decision import (
    LETTERS,
    RENDER_VERSION,
    from_systemone,
    option_order,
    option_scores,
    prompt_text,
    render,
    temperatures_for,
    to_probs,
)
from sys1.kernels import backends
from sys1.lang import bpe
from sys1.model import CONFIG_FILE, DECISION_CONFIG_FILE, Decoder

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZER_DIR = REPO_ROOT / "runs" / "1003-s0-bpe-16k-realedu-zh-en" / "tokenizer"
PAD_TOKEN = bpe.PAD_TOKEN
ROW_KEYS = ("id", "answers", "ms")           # 预测行的三键（decision-program 的键约定，不可增删）
#: 模型目录里可能承载"我来自哪次实验"的字段名（三处来源任一命中即算有溯源）
PROVENANCE_KEYS = ("run_id", "provenance", "source_run", "origin_run", "parent_run")
EXEMPT_RUN_ID = "untracked-smoke"           # --allow-missing-run-id 时的显式标记，报告里如实带着
SYSTEMONE_PATH = "/v1/systemone"


class MissingProvenanceError(RuntimeError):
    """模型目录查不到任何来源信息时抛出：评测当场拒绝出数（spec"无溯源被拒"场景）。"""


@dataclass(frozen=True)
class Provenance:
    """一次溯源的结论：来源 run 号、代码版本、命中了哪几处字段，以及是否被豁免放行。"""

    run_id: str
    commit: str | None = None
    sources: dict[str, str] = field(default_factory=dict)
    chain: dict[str, str] = field(default_factory=dict)   # 上游链（如 s3 校准 run、s1 起点）
    exempt: bool = False

    def as_dict(self) -> dict[str, Any]:
        """摊成可写进报告与 run 记录的普通字典。

        白话：把"这模型打哪儿来、哪次实验、哪个代码版本、中间有没有人破例放行"这几句话
        整理成一张小纸条，报告开头附一张，读者不用翻代码就知道数字的出处。
        """
        return {
            "run_id": self.run_id,
            "commit": self.commit,
            "sources": dict(self.sources),
            "chain": dict(self.chain),
            "exempt": self.exempt,
        }


# ---------------------------------------------------------------- 溯源（B1：无来源即拒评）
def _load_json(path: Path) -> dict[str, Any]:
    """读一个 json 文件，读不到或不是对象都回空字典（溯源是"多处找一遍"，单处缺失不算错）。"""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_yaml(path: Path) -> dict[str, Any]:
    """读一份 yaml 账本（run 的 config.yaml）；没装 pyyaml 或文件缺失都回空字典。"""
    if not path.is_file():
        return {}
    try:
        import yaml  # 依赖已在 pyproject（run 记录本身就是 yaml 落的）

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001  账本读不动不阻断溯源判定，继续找别处
        return {}
    return data if isinstance(data, dict) else {}


def _pick_provenance(obj: dict[str, Any]) -> tuple[str | None, dict[str, str]]:
    """从一份配置里挑来源：命中字段是字符串直接取，是嵌套对象则取其 run_id。"""
    found: dict[str, str] = {}
    run_id: str | None = None
    for key in PROVENANCE_KEYS:
        if key not in obj:
            continue
        value = obj[key]
        if isinstance(value, str) and value.strip():
            found[key] = value
            run_id = run_id or value
        elif isinstance(value, dict):
            inner = value.get("run_id") or value.get("id")
            if isinstance(inner, str) and inner.strip():
                found[f"{key}.run_id"] = inner
                run_id = run_id or inner
    return run_id, found


def _git_commit() -> str | None:
    """取当前代码版本（只读操作）；不在 git 仓库里或命令缺失都回 None。"""
    try:
        out = subprocess.run(          # 固定参数、无用户输入，只问一句 HEAD 的散列，故不 check
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True,
            timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (out.stdout or "").strip()
    return text or None


def resolve_provenance(model_dir: str | Path, *, allow_missing: bool = False) -> Provenance:
    """查清模型目录的实验来源；三处都查不到则抛 `MissingProvenanceError`（豁免除外）。

    查找顺序即"就近原则"：目录内 `config.json` 的来源字段 → 目录内 `decision_config.json`
    （含 s3 写回的 `calibration.run_id`）→ 上一层 run 账本 `config.yaml`/`system.json`。
    代码版本优先取账本里记的那次（当时的 HEAD），账本没有才用当下仓库的 HEAD——因为被测
    模型可能是在别的 commit 上产出的。

    白话：想知道这份权重是谁做出来的，就翻它随身的三样东西：设置单、说明书、外面那层
    实验档案袋上的编号；只要有一处写着编号就认，全都没写就不给做评测，并告诉该怎么补。
    """
    src = Path(model_dir)
    if not src.is_dir():
        raise NotADirectoryError(f"模型目录不存在：{src}")

    cfg = _load_json(src / CONFIG_FILE)
    decision = _load_json(src / DECISION_CONFIG_FILE)
    run_record = _load_yaml(src.parent / "config.yaml")
    system = _load_json(src.parent / "system.json")

    run_id: str | None = None
    sources: dict[str, str] = {}
    for label, obj in ((f"{CONFIG_FILE}", cfg), (f"{DECISION_CONFIG_FILE}", decision),
                       ("../config.yaml", run_record), ("../system.json", system)):
        hit, found = _pick_provenance(obj)
        if hit:
            run_id = run_id or hit
            sources[label] = found[next(iter(found))]

    chain: dict[str, str] = {}
    calib = decision.get("calibration")
    if isinstance(calib, dict) and calib.get("run_id"):
        chain["calibration_run_id"] = str(calib["run_id"])
        chain["calibration_status"] = str(calib.get("status", "unknown"))
    sft = decision.get("sft")
    if isinstance(sft, dict) and sft.get("run_id"):
        chain["sft_run_id"] = str(sft["run_id"])
    if isinstance(run_record, dict) and run_record.get("ckpt"):
        chain["upstream_ckpt"] = str(run_record["ckpt"])

    commit = run_record.get("commit") if isinstance(run_record.get("commit"), str) else None
    commit = commit or _git_commit()

    if run_id is None:
        if not allow_missing:
            keys = "、".join(PROVENANCE_KEYS)
            where = f"{CONFIG_FILE}/{DECISION_CONFIG_FILE} 与上一层 config.yaml/system.json"
            raise MissingProvenanceError(
                f"模型目录 {src} 查不到任何 run 来源（已在 {where} 试过 {keys}）。"
                f"评测拒绝出数：请用 producing pipeline 重跑，或把来源 run 号写进 "
                f"{CONFIG_FILE} 的 run_id 字段；仅冒烟可加 --allow-missing-run-id。"
            )
        return Provenance(run_id=EXEMPT_RUN_ID, commit=commit, sources={}, chain=chain, exempt=True)
    return Provenance(run_id=run_id, commit=commit, sources=sources, chain=chain, exempt=False)


# ---------------------------------------------------------------- 样本编码（复用 decision 渲染）
@dataclass
class SamplePrediction:
    """一个样本一次读数的全部对齐信息：对外只露三键，四轴要吃的是其余字段。"""

    id: str
    qid: str
    qtype: str
    k: int
    options: list[str]                  # 与字母列同序的候选代号
    scores: list[float]                 # 末位原始分（未同除倍数）
    probs: list[float]                  # 按 decision_config 倍数折出的份额（对外答案）
    probs_uncalibrated: list[float]     # 同除之前的份额（校准轴的"调整前"）
    target: list[float]                 # 人工份额，与 options 同序、和为 1
    temperature: float
    uncalibrated: bool
    ms: float
    from_endpoint: bool = False
    seq: int = 0                        # 在评测集里的行号（组批按 k 分桶会打乱，出口还原）

    @property
    def gold(self) -> int:
        """人工答案的位置（份额最大那一格）。

        白话：人工给的那份份额里占得最多的一格就是标准答案；并列时取更靠左的，跟读答案
        的规矩保持一致，免得多解的题目每次都换个参照。
        """
        return max(range(self.k), key=lambda i: self.target[i])

    @property
    def pred(self) -> int:
        """模型答案的位置（分数最大那一格；同分取更靠左的候选，与读出层一致）。

        白话：模型自己选的那一格——看原始胆量分不看折算后的份额，且同分时永远取更靠左
        的候选，这条规矩与读数层一模一样，免得同一份分数在这里换个名次。
        """
        return max(range(self.k), key=lambda i: (self.scores[i], -i))

    @property
    def conf(self) -> float:
        """模型对自己答案的把握（份额最大那一项）。

        白话：模型答完之后有多笃定，就看它给胜出那一格分了几成把握。把握轴量的正是
        "这份笃定跟最终对错差多远"，所以先要有一个说得出口的数。
        """
        return max(self.probs) if self.probs else 0.0

    def to_row(self) -> dict[str, Any]:
        """压成对外预测行 `{id, answers: {qid: {opt: p}}, ms}`（只三键，别的都不外泄）。

        白话：把一道题的结果写成一条流水：题号、每个候选各占几成、花了多少毫秒。人工
        给的标准答案不在这一行里——评测用的答案与交出去的答案得分开摆，免得日后被人
        当成模型自己说过的话。
        """
        return {
            "id": self.id,
            "answers": {self.qid: {self.options[i]: self.probs[i] for i in range(self.k)}},
            "ms": self.ms,
        }


@dataclass
class EncodedSample:
    """编码后的可组批样本（编号串 + 字母列 + 人工份额），还没进过模型。"""

    id: str
    qid: str
    qtype: str
    k: int
    options: list[str]
    letter_ids: list[int]
    input_ids: list[int]
    length: int
    target: list[float]
    seq: int = 0


def _questions_of(record: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """从信封里取样本与唯一题号（转写/typed 产物一行一题，多题请拆成多行）。"""
    sample = validate_sample(record["sample"])
    qid = next(iter(sample["questions"]))
    return sample, qid


def encode_record(record: dict[str, Any], tokenizer: Any, letter_map: dict[str, int]) -> EncodedSample:
    """把一条 `{id, sample}` 记录编成可组批样本：定死文字 → 编号串 + 字母列 + 人工份额。

    白话：拿一道题先照固定格式抄成一封短信（证据、问题、候选各带一个字母），再把这封信
    逐段换成一串数字编号；同时记下"第一个字母代表哪个候选"和"人工觉得各候选该占几成"，
    后面读答案时按这份对照把字母分数换回候选名字。
    """
    sample, qid = _questions_of(record)
    question = sample["questions"][qid]
    row = from_systemone(sample["state"], question, qid=qid)
    _, order = render(row)                              # order[i] = LETTERS[i] 代表的候选代号
    text = prompt_text(row, order)
    ids = list(tokenizer.encode(text, add_special_tokens=False).ids)
    if not ids:
        raise ValueError(f"样本 {record.get('id')} 换完编号是空的，没有可读的位置")
    k = len(order)
    tmap = sample.get("targets", {}).get(qid, {})
    raw = [float(tmap.get(order[i], 0.0)) for i in range(k)]
    total = sum(raw)
    if total <= 0.0:
        raise ValueError(f"样本 {record.get('id')} 人工份额全为零，没法对照")
    return EncodedSample(
        id=str(record.get("id") or qid), qid=qid, qtype=question["type"], k=k, options=list(order),
        letter_ids=[int(letter_map[LETTERS[i]]) for i in range(k)],
        input_ids=ids, length=len(ids), target=[v / total for v in raw],
    )


def _resequence(encs: list[EncodedSample]) -> None:
    """把评测集行号写回样本（组批按候选数分桶会乱序，出口据此还原）。"""
    for i, ex in enumerate(encs):
        ex.seq = i


def _make_batch(rows: list[EncodedSample], pad_id: int) -> dict[str, Any]:
    """同一候选数的一堆样本右补空洞成一个批：编号矩阵 + 可见位 + 每行真长度。"""
    width = max(ex.length for ex in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, ex in enumerate(rows):
        input_ids[i, : ex.length] = torch.tensor(ex.input_ids, dtype=torch.long)
        attn_mask[i, : ex.length] = 1
    return {
        "input_ids": input_ids,
        "attn_mask": attn_mask,
        "lengths": [ex.length for ex in rows],
        "letter_ids": rows[0].letter_ids,
        "qtypes": [ex.qtype for ex in rows],
    }


def group_by_k(rows: list[EncodedSample]) -> dict[int, list[EncodedSample]]:
    """按候选数分堆（同堆才能并排组批）：`{候选个数: [样本...]}`。

    白话：有的题三个候选、有的二十个，格子数不一样就没法并排坐进同一批。这里先按
    "几个候选"把它们分堆，同一堆里再打捆送进模型，读出来的分数列才对得上。
    """
    buckets: dict[int, list[EncodedSample]] = {}
    for ex in rows:
        buckets.setdefault(ex.k, []).append(ex)
    return buckets


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """读一份评测集 jsonl（一行一个 `{id, task, sample}` 信封），空行跳过。

    白话：把盘里"一行一道题"的档案逐行拆开读回来，坏行不猜、空行略过，每行还原成
    一道带来源号的题，交给后面的抄信与读数。
    """
    records: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------- 本地模式（--model）
class LocalPredictor:
    """本地模型目录预测器：载权重 + 载词表 + 载决策配置（倍数表），一次前向出预测行。"""

    def __init__(self, model_dir: str | Path, *, tokenizer_dir: str | Path | None = None,
                 device: str | torch.device | None = None, allow_missing_run_id: bool = False,
                 bins: int = DEFAULT_BINS) -> None:
        """把三件套读进来并做溯源体检；溯源不过直接抛错，权重根本不用载。"""
        self.model_dir = Path(model_dir)
        self.provenance = resolve_provenance(self.model_dir, allow_missing=allow_missing_run_id)
        self.bins = bins
        self.tokenizer_dir = self._resolve_tokenizer_dir(tokenizer_dir)
        self.tokenizer = bpe.load(self.tokenizer_dir)
        self.letter_map = bpe.check_tokenizer(self.tokenizer)
        self.pad_id = int(self.tokenizer.token_to_id(PAD_TOKEN) or 0)
        self.decision_config = _load_json(self.model_dir / DECISION_CONFIG_FILE)
        self.temperatures: dict[str, float] = dict(self.decision_config.get("temperatures") or {})
        self.render_version = self.decision_config.get("render_version", RENDER_VERSION)
        self.device = backends.resolve_device(device)
        self.model = Decoder.load(self.model_dir).to(self.device).eval()
        # 读数落 CPU：p1-07 的 option_scores 内部按 CPU 建字母下标列子（跨设备会报"Passed CPU
        # tensor to MPS op"），而"末位 × 26 行"本就是几微秒的小矩阵；固定在 CPU 读数还有一个
        # 好处——同一份权重在 mps/cpu 上出来的份额逐位一致，质量与把握两轴的数不随设备漂移。
        self.head_weight_cpu = self.model.lm_head.weight.detach().cpu()

    def _resolve_tokenizer_dir(self, given: str | Path | None) -> Path:
        """词表目录三级定位：显式参数 → 模型目录内 tokenizer/ → 上一层 run 账本记的路径 → 仓库默认。"""
        if given is not None:
            return Path(given)
        inside = self.model_dir / "tokenizer"
        if inside.is_dir():
            return inside
        for candidate in (self.model_dir.parent / "config.yaml",):
            recorded = _load_yaml(candidate).get("tokenizer")
            if isinstance(recorded, str) and Path(recorded).is_dir():
                return Path(recorded)
        return DEFAULT_TOKENIZER_DIR

    def encode(self, records: list[dict[str, Any]]) -> list[EncodedSample]:
        """把记录编成可组批样本（与 `predict` 内部同一条编码路，供耗时轴数入参长度）。

        白话：单独把"抄题+换编号"这一步露出来：不碰模型只出对照表，谁想知道一道题
        变成多少个编号、哪个字母代表哪个候选，就用这个入口。与答题走的是同一条抄法，
        所以量出来的长度和真正送进模型的那一份绝不会两个版本。
        """
        return [encode_record(r, self.tokenizer, self.letter_map) for r in records]

    @torch.no_grad()
    def predict(self, records: list[dict[str, Any]], *, batch_size: int = 8) -> list[SamplePrediction]:
        """逐批读数并交富样本列表；对外三键行由 `to_row()` 现取，别在中间层里改口径。

        白话：把题目按候选数分堆打捆，一捆送进模型走一遍，只在每行写完的最后那一格收
        各候选的胆量分；同一把分数折两次份额——按类型倍数除过的拿去答题，没除的那份
        留着比较"调之前、调之后各说了几成"。顺带记一捆平均花了几毫秒。
        """
        encs = self.encode(records)
        _resequence(encs)
        out: list[SamplePrediction] = []
        for rows in group_by_k(encs).values():
            for start in range(0, len(rows), max(1, batch_size)):
                chunk = rows[start : start + batch_size]
                batch = _make_batch(chunk, self.pad_id)
                ids = batch["input_ids"].to(self.device)
                mask = batch["attn_mask"].to(self.device)
                t0 = time.perf_counter()
                last_hidden = self.model(ids, mask)
                scores = option_scores(last_hidden.detach().cpu(), self.head_weight_cpu,
                                       batch["letter_ids"], batch["lengths"])
                ms = (time.perf_counter() - t0) * 1000.0 / len(chunk)
                values, calibrated = temperatures_for(batch["qtypes"], self.temperatures, len(chunk))
                # 同除一个正数不改名次（p1-05 的单调保证），故份额折两次、名次只有一份
                scaled = scores / torch.tensor(values, dtype=scores.dtype, device=scores.device).unsqueeze(1)
                probs_cal = to_probs(scaled, 1.0)
                probs_raw = to_probs(scores, 1.0)
                for i, ex in enumerate(chunk):
                    out.append(SamplePrediction(
                        id=ex.id, qid=ex.qid, qtype=ex.qtype, k=ex.k, options=ex.options,
                        scores=scores[i].float().cpu().tolist(),
                        probs=probs_cal[i].float().cpu().tolist(),
                        probs_uncalibrated=probs_raw[i].float().cpu().tolist(),
                        target=ex.target, temperature=values[i],
                        uncalibrated=not calibrated[i], ms=float(ms), seq=ex.seq,
                    ))
        out.sort(key=lambda item: item.seq)      # 与评测集逐行同序，方便 diff 与按行打分
        return out

    def describe(self) -> dict[str, Any]:
        """入口摘要（供报告头与 run 记录直接抄）：来源、设备、后端、倍数表、渲染版号。

        白话：把"这次评测用的是哪份权重、在什么机器上、按哪份说明书读数、各类型除几"
        一句句列清楚，报告第一屏就能看见，出问题时知道该回去翻哪个目录。
        """
        return {
            "mode": "model",
            "model_dir": str(self.model_dir),
            "tokenizer_dir": str(self.tokenizer_dir),
            "device": str(self.device),
            "backend": backends.active_backend(self.device),
            "dtype": "float32",
            "render_version": self.render_version,
            "temperatures": dict(self.temperatures),
            "provenance": self.provenance.as_dict(),
        }


# ---------------------------------------------------------------- 远端模式（--endpoint）
def _http_post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """默认的 HTTP 发送件：json 进、json 出（单测不用它，走注入的假发送件）。"""
    req = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class EndpointPredictor:
    """/v1/systemone 服务预测器：把样本切成请求、把回包对齐成本地同一种行。

    真实服务联动归二阶段（p2-10），本域只保证协议转换成立：发送件可注入，单测用假回包
    验"请求体字段齐、回包份额原样落地、耗时字段优先服务端"三件事。
    """

    def __init__(self, endpoint: str, *, tokenizer_dir: str | Path | None = None,
                 timeout: float = 30.0,
                 post: Callable[[str, dict[str, Any], float], dict[str, Any]] | None = None,
                 allow_missing_run_id: bool = True) -> None:
        """记下地址与超时，并备好候选序对照表（回包的键要按本地字母序落位）。"""
        self.url = endpoint if endpoint.endswith(SYSTEMONE_PATH) else endpoint.rstrip("/") + SYSTEMONE_PATH
        self.timeout = timeout
        self.post = post or _http_post_json
        self.allow_missing_run_id = allow_missing_run_id
        # 注入的假发送件可以只收 (url, payload)，故先探一下它收不收超时参数
        try:
            self._post_wants_timeout = len(inspect.signature(self.post).parameters) >= 3
        except (TypeError, ValueError):
            self._post_wants_timeout = True

    def predict(self, records: list[dict[str, Any]], *, batch_size: int = 1) -> list[SamplePrediction]:
        """逐样本发一次请求（服务侧自批处理，本处不并批），把回包对齐成富样本。

        白话：一道题发一封问信，收回来照着"第几个字母代表哪个候选"把各候选分到几成填回
        原位；对方没答的候选记零成，人工标准答案仍然只留在自己这边，不混进流水。
        """
        out: list[SamplePrediction] = []
        for record in records:
            sample, qid = _questions_of(record)
            question = sample["questions"][qid]
            row = from_systemone(sample["state"], question, qid=qid)
            order = option_order(row)
            payload = {"id": record.get("id"), "state": sample["state"], "questions": {qid: question}}
            t0 = time.perf_counter()
            reply = self.post(self.url, payload, self.timeout) if self._post_wants_timeout else self.post(self.url, payload)
            ms = reply.get("ms")
            ms = float(ms) if isinstance(ms, (int, float)) else (time.perf_counter() - t0) * 1000.0
            answers = (reply.get("answers") or {}).get(qid, {})
            probs = [float(answers.get(order[i], 0.0)) for i in range(len(order))]
            total = sum(probs)
            probs = [p / total for p in probs] if total > 0 else [1.0 / len(order)] * len(order)
            tmap = sample.get("targets", {}).get(qid, {})
            raw = [float(tmap.get(order[i], 0.0)) for i in range(len(order))]
            mass = sum(raw)
            out.append(SamplePrediction(
                id=str(record.get("id") or qid), qid=qid, qtype=question["type"], k=len(order),
                options=list(order), scores=probs, probs=probs, probs_uncalibrated=probs,
                target=[v / mass for v in raw] if mass > 0 else [1.0 / len(order)] * len(order),
                temperature=1.0, uncalibrated=True, ms=ms, from_endpoint=True,
            ))
        return out

    def describe(self) -> dict[str, Any]:
        """入口摘要：远端模式没有本地权重可溯源，故如实记 `mode=endpoint` 与豁免标记。

        白话：交一份"这次是打远程服务测的"说明，里面写清地址、每次最多等多久、以及
        因为没有本地目录所以来源那一栏是按豁免处理的实话。
        """
        return {
            "mode": "endpoint",
            "endpoint": self.url,
            "timeout": self.timeout,
            "device": "remote",
            "dtype": "n/a",
            "provenance": {"run_id": EXEMPT_RUN_ID if self.allow_missing_run_id else None,
                           "exempt": True, "source": "endpoint",
                           "note": "远端服务无本地模型目录，来源由服务侧自证"},
        }


# ---------------------------------------------------------------- 行收发与 CLI
def write_rows(path: str | Path, preds: list[SamplePrediction]) -> Path:
    """把预测行写成 jsonl（一行一个 `{id, answers, ms}`），返回落盘路径。

    白话：把每道题的结果一条条抄进一个流水文件，一行一道，谁答了什么、各占几成、花了
    多少毫秒都在里面；标准答案不抄进来，免得日后被人当成模型说的话。
    """
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fh:
        for pred in preds:
            fh.write(json.dumps(pred.to_row(), ensure_ascii=False) + "\n")
    return dst


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """读回预测行 jsonl，并卡住三键契约（缺键即报错，不做"大概是预测文件"的猜测）。

    白话：把流水文件一行行收回内存，同时挨行验一遍该有的三样东西（题号、各候选几成、
    花了几毫秒）一样不缺。缺了就当场报错停下，绝不"看着像预测文件"就硬往下算——
    拿错文件算出来的数照样会被当成跑分。
    """
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [key for key in ROW_KEYS if key not in row]
        if missing:
            raise KeyError(f"预测行缺键 {missing}，契约要求恰为 {list(ROW_KEYS)}：{line[:120]}")
        rows.append(row)
    return rows


def build_predictor(args: argparse.Namespace) -> Any:
    """按参数挑本地或远端预测器（两者必须且只能给一个）。

    白话：命令行要么给一个本机模型目录，要么给一个远程服务地址，二者恰好选其一；
    两个都给或都不给都直接停下来问清楚，因为这两种模式的耗时口径与出处记录方式
    完全不同，猜一个来用等于把两次实验混成一次。
    """
    if bool(args.model) == bool(args.endpoint):
        raise SystemExit("--model 与 --endpoint 二选一（且必须给一个）")
    if args.model:
        return LocalPredictor(args.model, tokenizer_dir=args.tokenizer, device=args.device,
                              allow_missing_run_id=args.allow_missing_run_id)
    return EndpointPredictor(args.endpoint, tokenizer_dir=args.tokenizer, timeout=args.timeout)


def build_parser() -> argparse.ArgumentParser:
    """搭出预测出口的 CLI 参数表（被测对象、评测集、落盘位置、规模与设备）。

    白话：把命令行能敲的选项和默认值一次摆清楚，敲一下 --help 就知道该给哪个模型目录
    或哪个服务地址、题目从哪份文件读、结果写到哪、一捆多大、在什么设备上算。
    """
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description="评测唯一预测出口：--model 本地模型目录 / --endpoint /v1/systemone 服务，交同一种预测行。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=None, help="被测模型目录（runs/<id>/model，含 config.json）")
    parser.add_argument("--endpoint", default=None, help="被测服务地址（…/v1/systemone）")
    parser.add_argument("--data", required=True, help="评测集 jsonl（一行一个 {id,task,sample}）")
    parser.add_argument("--out", default=None, help="预测行落盘路径（不给则只统计不出文件）")
    parser.add_argument("--tokenizer", default=None, help="词表目录（默认按模型目录/tokenizer→run 账本→仓库 S0 产物定位）")
    parser.add_argument("--device", default="auto", help="计算设备（auto/cpu/mps）")
    parser.add_argument("--batch-size", type=int, default=8, help="本地模式每捆样本数（同候选数才能并批）")
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 条（0=全取）")
    parser.add_argument("--timeout", type=float, default=30.0, help="远端模式单请求超时秒数")
    parser.add_argument("--allow-missing-run-id", action="store_true",
                        help="无溯源也放行（仅冒烟；报告里会带 exempt 标记）")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：解析 → 溯源体检（不过即退出码非零）→ 出预测行。

    白话：命令行跑起来第一件事是查这份权重来路正不正——翻遍随身的设置单、说明书和外面
    那层档案袋都没编号，就一句"补上再来说不清"，直接非零退出，一道题都不测。来路清楚
    才把题目读一遍、按行抄进流水文件，末了报一句共多少行、写在哪儿。
    """
    args = build_parser().parse_args(argv)
    try:
        predictor = build_predictor(args)        # 溯源体检在载权重之前，不过就一道题都不测
    except MissingProvenanceError as exc:
        print(f"[predict] 拒绝出数：{exc}", file=sys.stderr)
        return 2
    records = load_records(args.data)
    if args.limit:
        records = records[: args.limit]
    preds = predictor.predict(records, batch_size=args.batch_size)
    if args.out:
        path = write_rows(args.out, preds)
        print(f"[predict] rows -> {path}")
    desc = predictor.describe()
    prov = desc.get("provenance") or {}
    print(f"[predict] mode={desc['mode']} run-id={prov.get('run_id')} commit={prov.get('commit')} "
          f"exempt={prov.get('exempt')} device={desc.get('device')} n={len(preds)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
