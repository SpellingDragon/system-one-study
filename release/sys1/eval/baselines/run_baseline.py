"""sys1/eval/baselines/run_baseline.py — 双基线同 harness 亲跑：laya 与 StartLux-0.8B 各出一条对照行。

【做什么】
    把同一批决策题，分别喂给两条现成参照基线，各产出一行"这套系统同口径能得几分"的记录：
    laya（起不了服务时走它本机等价的 predict 通路）与 StartLux-0.8B（本地权重目录，按字母
    读出末位）。两条都落在同一 split、同一采样参数下，分数能直接并排比。顺带各开一个 run
    记录，把机器、代码版本、样本数、逐题耗时、以及"这是亲跑还是降级抄卡面"写清楚。

【怎么做】
    两条基线共用一套骨架：① 选题 `pick_records`——按题型分堆轮流取满 limit 条，保证 choice
    与 noul 都到场，且两次基线取到的是同一串样本号。② 渲染——一律走 sys1.decision 的
    `from_systemone/render/prompt_text`，拿到"第几个字母代表哪个候选"的字母序，这是与自训
    小模型完全相同的那把尺。③ 各表读法：laya 直接问它本机的 predict，收每个候选各占几成，
    按字母序对齐；StartLux 把渲染文字喂进前向，取写完最后一个字那一格在 26 枚字母上的分数，
    再照它自带 decision_config 里各题型的倍数同除一遍摊成份额。④ 落地：读数拼成与预测出口
    同一种富样本，质量与把握两轴直接复用 sys1.eval 的现成算法，耗时逐题掐表；一份对外三键
    流水、一份内部对照行 json、外加 run 记录四件套。⑤ 权重拿不到又允许降级时，如实写
    `source: card, gate: false` 并把指标留空，绝不冒充实测。

【为什么】
    基线表的红线是"同 harness 亲跑、不接受转抄卡面"，所以读数口径必须与自训模型逐字相同：
    同一份渲染文字、同一套字母读出、同一把分桶与把握的尺。换个引擎只换取数那一步，换不掉
    打分那一步，否则三条线各自算各自的，并排出来的差就是假差。
    被否方案一：给 StartLux 现编一套提示词——它训练时用的话术我们无从确证，编一套只能让
    数字好看，却与自训模型不同尺，红线当场破；故宁可沿用统一渲染，读出好不好如实反映。
    被否方案二：MPS 空不出来就停摆——一阶段占着 MPS，本波验收只看"通路真跑通 + 小样本出分"，
    全量留给云端；故一律走 CPU，慢就慢，出的是真数，不是等不来。
    被否方案三：把降级抄来的卡面数字也填进指标格——那是把"没测"伪装成"测过"，宁可空着标
    `—`，也不给一个没有 run 背书的数。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from sys1.calibrate import DEFAULT_BINS
from sys1.decision import (
    from_systemone,
    prompt_text,
    render,
    temperatures_for,
    to_probs,
)
from sys1.eval.calibration import score_calibration
from sys1.eval.predict import SamplePrediction, load_records, write_rows
from sys1.eval.quality import score_quality
from sys1.model import DECISION_CONFIG_FILE
from sys1.runs import new_run

REPO_ROOT = Path(__file__).resolve().parents[3]
# 一阶段已 pin 的 typed-decisions 测试集；p2-03 registry fetch 就绪后改指 bench/ 路径。
DEFAULT_DATA = REPO_ROOT.parent / "scratch" / "bench" / "p1-09" / "typed_decisions" / "test.jsonl"
STARTLUX_MS_ID = "StartLuxAI/StartLux-Decision-0.8B"
STARTLUX_CACHE = "bench/ms_models"
DEFAULT_STARTLUX_DIR = (
    REPO_ROOT / STARTLUX_CACHE / "models" / "StartLuxAI--StartLux-Decision-0.8B" / "snapshots" / "master"
)
LAYA_ALIAS = "typed-decisions"
#: 采样参数：读出制无解码采样，统一记 T=1/top_p=1（与自训模型同口径，非生成式采样）
SAMPLING_PARAMS = "T=1/top_p=1(readout)"
#: 本波 CPU 小样本冒烟的默认题数（≥20 且 choice/noul 到场）；全量属云端 C6。
SMOKE_LIMIT = 24
ROW_KIND = "baseline-row"


# ---------------------------------------------------------------- 选题（两基线共用同一 split）
def pick_records(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """按题型分堆轮流取满 limit 条，choice 与 noul 都保证到场且顺序确定。

    白话：一堆题里若前二十道恰好全是同一种类，另两条基线就都在没见过的题型上比了。这里
    先把题目按种类分开，再像发牌一样一轮一类地取，取满要的条数为止；两类都齐，比的是
    同一串样本，两次跑分才对得上。
    """
    if limit <= 0 or limit >= len(records):
        return list(records)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        question = next(iter(rec["sample"]["questions"].values()))
        buckets.setdefault(str(question.get("type", "choice")), []).append(rec)
    keys = sorted(buckets)
    picked: list[dict[str, Any]] = []
    idx = 0
    while len(picked) < limit:
        exhausted = True
        for key in keys:
            if idx < len(buckets[key]):
                picked.append(buckets[key][idx])
                exhausted = False
                if len(picked) >= limit:
                    break
        if exhausted:
            break
        idx += 1
    return picked


def _record_parts(record: dict[str, Any]) -> tuple[str, dict[str, Any], str, list[str]]:
    """拆一条记录：题号、原始题目、题号键、以及字母序（候选代号列表）。

    白话：流水里一道题身上挂着几样东西——它自己的号、题面正文、还有"第一个字母代表哪个
    候选、第二个又代表哪个"的对照。后面所有读法都从这里出发，两条基线共用同一份对照。
    """
    sample = record["sample"]
    qid = next(iter(sample["questions"]))
    question = sample["questions"][qid]
    row = from_systemone(sample["state"], question, qid=qid)
    _, order = render(row)
    return str(record.get("id") or qid), question, qid, order


def _target_from_sample(record: dict[str, Any], qid: str, order: list[str]) -> list[float]:
    """人工份额按字母序摆齐（缺项补零、全零回均匀）。"""
    tmap = record["sample"].get("targets", {}).get(qid, {})
    raw = [float(tmap.get(order[i], 0.0)) for i in range(len(order))]
    total = sum(raw)
    if total <= 0.0:
        return [1.0 / len(order)] * len(order)
    return [v / total for v in raw]


def assemble_prediction(record: dict[str, Any], *, scores: list[float], probs: list[float],
                        temperature: float, uncalibrated: bool, ms: float) -> SamplePrediction:
    """把一条读数拼成与预测出口同一种富样本：对外只三键，四轴吃其余字段。

    白话：无论分数是从 laya 本机来的、还是从 StartLux 末位字母读出来的，到这里都归成同一种
    行——题号、每个候选各分几成、原始胆量分、人工该占几成、这一格花了多少毫秒。归成同一种，
    后面质量轴与把握轴才不必认出这道题是哪家引擎算的，一把尺量两条线。
    """
    _rid, question, qid, order = _record_parts(record)
    k = len(order)
    return SamplePrediction(
        id=str(record.get("id") or qid), qid=qid, qtype=str(question.get("type", "choice")), k=k,
        options=list(order), scores=list(scores), probs=list(probs),
        probs_uncalibrated=list(probs), target=_target_from_sample(record, qid, order),
        temperature=float(temperature), uncalibrated=bool(uncalibrated), ms=float(ms),
    )


# ---------------------------------------------------------------- laya 通路（本机 predict 等价服务）
class LayaRunner:
    """laya 参照基线：用其本机 `predict`（`/v1/systemone` 的等价调用）逐题读数。

    laya 是非自回归判别式引擎，一次前向直接交每个候选各占几成；本类只把它的答案字典按
    sys1.decision 的字母序对齐成本地同一种富样本，绝不改它的读数，也不套本地倍数表。
    """

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.describe_meta = {"backend": "laya", "model": LAYA_ALIAS}

    @classmethod
    def load(cls, device: str = "cpu", *, alias: str = LAYA_ALIAS) -> "LayaRunner":
        """载 laya 判别式引擎；权重不可得会原样抛错，交由上层判降级。

        白话：去把 laya 那套现成的判别引擎请出来站在这儿等提问。请得动就继续，请不动
        （比如断网、比如它那份权重搬不到这台机器）就直接把话撂下，别装着已经就位。
        """
        import laya  # 权重经 hf-mirror 拉取；直连 HF 不通时调用方须置 HF_ENDPOINT

        agent = laya.load(alias, device=device)
        return cls(agent)

    def _prob_map(self, answer: dict[str, Any], order: list[str]) -> list[float]:
        """把 laya 一题的答案按字母序折成份额列（choice/score 看 probabilities，noul 看 noul）。

        白话：laya 报回来的东西按候选名字各记了一成，可它排的顺序未必和我们的字母序一致。
        这里照着字母序一格一格去讨：点名的来路有就抄过来，没有就记零；判断题只给了"真的
        有几分"，那"假的"就补足剩下的，两格一凑才算完整的一份份额。
        """
        k = len(order)
        if answer.get("type") == "noul" or "noul" in answer:
            p_true = float(answer.get("noul", 0.5))
            lookup = {"true": p_true, "false": 1.0 - p_true}
            return [lookup.get(opt, 0.0) for opt in order]
        raw = answer.get("probabilities", {}) or {}
        probs = [float(raw.get(opt, 0.0)) for opt in order]
        total = sum(probs)
        return [p / total for p in probs] if total > 0 else [1.0 / k] * k

    def predict(self, records: list[dict[str, Any]]) -> list[SamplePrediction]:
        """逐题问 laya 并拼成富样本；耗时取 laya 单次 predict 的墙钟。

        白话：一题一题去问 laya 现成的判别引擎，把每题耗时掐在表里；它答回来的份额照字母序一格一格摆齐，再拼成和自训模型一模一样的那种富样本，好让质量与把握两把尺直接量，不必认出这数是谁家算的。
        """
        preds: list[SamplePrediction] = []
        for rec in records:
            _rid, question, qid, order = _record_parts(rec)
            t0 = time.perf_counter()
            result = self.agent.predict(rec["sample"]["state"], {qid: question})
            ms = (time.perf_counter() - t0) * 1000.0
            answer = result["answers"][qid]
            probs = self._prob_map(answer, order)
            preds.append(assemble_prediction(
                rec, scores=probs, probs=probs, temperature=1.0, uncalibrated=True, ms=ms,
            ))
        return preds


# ---------------------------------------------------------------- StartLux 通路（本地权重 letter-readout）
def fold_letter_scores(scores: list[float], temperature: float) -> tuple[list[float], list[float]]:
    """把一行的字母胆量分折两次份额：按倍数同除过的（对外）与没除的（未调温）。

    白话：几个字母各报一个胆量分，先照这类题自带的倍数把分头一起摊平一点（除的是同一个正数，
    谁排第一不会变，变的只是各占几成的陡与缓），再各自摊成一锅汤看几成；没摊平前那份也留着，
    好让把握轴比较"调之前、调之后各说了几成"。
    """
    import torch

    raw = torch.tensor(scores, dtype=torch.float64)
    t = max(float(temperature), 1e-12)
    probs_cal = to_probs(raw / t, 1.0).tolist()
    probs_raw = to_probs(raw, 1.0).tolist()
    return [float(v) for v in probs_cal], [float(v) for v in probs_raw]


class StartLuxRunner:
    """StartLux-0.8B 参照基线：本地权重 CPU 前向，末位字母读出 + 自带倍数直读。

    D11 边界：本类只做对照评测推理（白名单③）——绝不训练、绝不当基座、绝不二次发布其权重
    或衍生；读数一律复用 sys1.decision 渲染与字母读出，与自训模型同 harness。
    """

    def __init__(self, model_dir: Path, model: Any, tokenizer: Any, letter_ids: list[int],
                 temp_table: dict[str, float], device: str = "cpu") -> None:
        self.model_dir = Path(model_dir)
        self.model = model
        self.tokenizer = tokenizer
        self.letter_ids = list(letter_ids)
        self.temp_table = dict(temp_table)
        self.device = device
        self.describe_meta = {"backend": "startlux", "model_dir": str(self.model_dir),
                              "temperature_by_type": dict(temp_table), "d11": "仅对照评测推理"}

    @classmethod
    def load(cls, model_dir: str | Path = DEFAULT_STARTLUX_DIR, device: str = "cpu") -> "StartLuxRunner":
        """从本地权重目录载 StartLux（float32 走 CPU），读其 decision_config 的字母表与倍数。

        白话：把盘里那份 StartLux 权重连同它的说明书一起翻开——说明书上写着哪二十六个符号
        当字母答案、每一类题该把胆量分同除几。目录不在、文件残缺都会当场抛错，交给上层去
        判"降级抄卡面还是停下"，不在这里假装读到了东西。
        """
        src = Path(model_dir)
        if not (src / DECISION_CONFIG_FILE).is_file():
            raise FileNotFoundError(f"StartLux 权重目录缺少 {DECISION_CONFIG_FILE}：{src}")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        decision = json.loads((src / DECISION_CONFIG_FILE).read_text(encoding="utf-8"))
        letter_ids = [int(i) for i in decision["letter_token_ids"]]
        temp_table = dict(decision.get("temperature_by_type") or {})
        model = AutoModelForCausalLM.from_pretrained(str(src), dtype=torch.float32,
                                                     trust_remote_code=True)
        model.to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(str(src), trust_remote_code=True)
        return cls(src, model, tokenizer, letter_ids, temp_table, device=device)

    def _record_with_text(self, record: dict[str, Any]) -> tuple[dict[str, Any], list[str], str]:
        """取一条记录并渲成定死文字：把字母序与正文一并备好，供组批与读数复用。"""
        _rid, _q, _qid, order = _record_parts(record)
        sample = record["sample"]
        qid = next(iter(sample["questions"]))
        row = from_systemone(sample["state"], sample["questions"][qid], qid=qid)
        return record, order, prompt_text(row, order)

    def _scores_for_letters(self, input_ids: Any, attention_mask: Any, k: int) -> list[list[float]]:
        """一次前向，取每行末位在字母表前 k 列的胆量分（末位即"该写答案字母"那一格）。

        白话：一整批题并排送进模型走一遍，每行只在它"该报字母"的那一格收分；批里各行长短
        不一，右补了空格，所以按每行真长度找回它自己的末位，别把补出来的空位末当成答案位。
        """
        with self._no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        lengths = attention_mask.sum(dim=1)
        out: list[list[float]] = []
        for i in range(input_ids.shape[0]):
            pos = int(lengths[i]) - 1
            row = logits[i, pos, self.letter_ids[:k]]
            out.append([float(v) for v in row.tolist()])
        return out

    def _no_grad(self) -> Any:
        """包一层 torch.no_grad（推理不需要求斜率，省显存也防误触训练路径）。"""
        import torch

        return torch.no_grad()

    def predict(self, records: list[dict[str, Any]], *, batch_size: int = 4) -> list[SamplePrediction]:
        """逐批把渲染文字喂进前向、末位读字母、按题型倍数折份额，拼成富样本。

        白话：一道题先照统一格式抄成一段定死文字，换成一串符号送进模型走一遍；只在它"该
        报字母"的那一格收二十六个字母各自的胆量分，取前几个候选那几格，再按这类题自带的
        倍数同除一遍摊成份额。同除一个正数不改谁排第一，改的只是各占几成的陡与缓。
        """
        parts = [self._record_with_text(rec) for rec in records]
        by_k: dict[int, list[tuple[dict[str, Any], list[str], str]]] = {}
        for rec, order, text in parts:
            by_k.setdefault(len(order), []).append((rec, order, text))
        preds: list[SamplePrediction] = []
        for k, rows in by_k.items():
            for start in range(0, len(rows), max(1, batch_size)):
                chunk = rows[start:start + batch_size]
                texts = [t for _rec, _order, t in chunk]
                enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                                     add_special_tokens=False)
                enc = {kk: vv.to(self.device) for kk, vv in enc.items()}
                t0 = time.perf_counter()
                score_rows = self._scores_for_letters(enc["input_ids"], enc["attention_mask"], k)
                per_ms = (time.perf_counter() - t0) * 1000.0 / len(chunk)
                qtypes = [str(_record_parts(rec)[1].get("type", "choice")) for rec, _o, _t in chunk]
                values, calibrated = temperatures_for(qtypes, self.temp_table, len(chunk))
                for i, (rec, _order, _t) in enumerate(chunk):
                    probs_cal, _probs_raw = fold_letter_scores(score_rows[i], values[i])
                    preds.append(assemble_prediction(
                        rec, scores=score_rows[i], probs=probs_cal,
                        temperature=values[i], uncalibrated=not calibrated[i], ms=per_ms,
                    ))
        preds.sort(key=lambda p: p.id)     # 稳定序：组批按候选数分桶会乱，出口还原
        return preds


# ---------------------------------------------------------------- 打分与对照行
def score_predictions(preds: list[SamplePrediction], *, temp_table: dict[str, float] | None,
                      bins: int = DEFAULT_BINS) -> dict[str, Any]:
    """复用质量轴与把握轴，汇出 acc/ECE 前后与逐题耗时中位，作为对照行的数字来源。

    白话：两条基线跑出的那种富样本，直接送进现成的两把尺——一把量"名次对不对"（按题型和
    候选数分堆，各堆旁边摆一条闭眼乱选的底线），一把量"嘴上几成、结果几成对"的差距（调温
    前后各量一遍）。再顺手报个每道题平均花了几毫秒、一秒钟能出几道，凑成对照行要的几样数。
    """
    if not preds:
        return {"n": 0, "acc": None, "ece": None, "ece_after": None, "ece_before": None,
                "ms_p50": None, "ms_mean": None, "throughput_qps": None, "buckets": [], "by_type": []}
    quality = score_quality(preds)
    calib = score_calibration(preds, bins=bins, table=temp_table)
    ms_values = [p.ms for p in preds]
    total_hits = sum(b["hits"] for b in quality["buckets"])
    total_n = sum(b["n"] for b in quality["buckets"])
    overall = (calib.get("overall") or {}).get("after") or {}
    return {
        "n": len(preds),
        "acc": round(total_hits / total_n, 6) if total_n else None,
        "ece": overall.get("ece"),
        "ece_after": overall.get("ece"),
        "ece_before": ((calib.get("overall") or {}).get("before") or {}).get("ece"),
        "ms_p50": round(statistics.median(ms_values), 3),
        "ms_mean": round(statistics.fmean(ms_values), 3),
        "throughput_qps": round(len(preds) / (sum(ms_values) / 1000.0), 4) if sum(ms_values) > 0 else None,
        "buckets": quality["buckets"],
        "by_type": calib["by_type"],
    }


def build_row(*, baseline: str, dataset: str, split: str, n: int, device: str,
              sampling: str, metrics: dict[str, Any], run_id: str, commit: str | None,
              source: str, gate: bool, note: str) -> dict[str, Any]:
    """组一行对照记录（亲跑或降级都走这里），字段与 spec 六字段/表列口径对齐。

    白话：把"哪条基线、在哪个数据集哪一份切分、多少道题、什么设备上、按什么采样、跑出了
    几分、来自哪次实验、这数是实测还是抄的卡面"这些事实钉成一行。是实测就把数填满、把门开
    着；抄来的就 source 写 card、gate 关掉、指标留空，读者一眼分得清哪格可信哪格只是参照。
    """
    return {
        "kind": ROW_KIND, "baseline": baseline, "dataset": dataset, "split": split,
        "n_samples": n, "device": device, "sampling_params": sampling,
        "metrics": {key: metrics.get(key) for key in ("acc", "ece", "ece_after", "ece_before",
                                                      "ms_p50", "ms_mean", "throughput_qps")},
        "run_id": run_id, "commit": commit, "source": source, "gate": bool(gate),
        "on_device": "端侧(CPU通路)" if device == "cpu" else device,
        "note": note,
    }


def _downgrade_row(baseline: str, dataset: str, split: str, run_id: str, commit: str | None,
                   card_note: str) -> dict[str, Any]:
    """权重不可得时的降级行：`source: card, gate: false`，指标全空，绝不冒充实测。"""
    return build_row(
        baseline=baseline, dataset=dataset, split=split, n=0, device="—",
        sampling=SAMPLING_PARAMS, metrics={}, run_id=run_id, commit=commit,
        source="card", gate=False, note=card_note,
    )


# ---------------------------------------------------------------- 亲跑编排
def run(baseline: str, *, data: Path, limit: int, device: str, split: str,
        out_root: Path, startlux_dir: Path | None, allow_degrade: bool,
        card_note: str = "权重不可得，降级为卡面参考") -> dict[str, Any]:
    """跑一条基线并落产物：预测流水 + 对照行 json + run 记录四件套；返回对照行。

    白话：这是总装线。先把题目选好，再去把引擎请出来（请不动且允许降级，就走抄卡面那条路，
    把话写实在记录里）；请得动就逐题读数、拼成统一富样本、送进两把尺出分，然后开一个 run
    目录把参数、机器身份、流水账与结论都记下，最后把对照行写成一个 json，交给对照表那边取用。
    """
    records = pick_records(load_records(data), limit)
    out_root.mkdir(parents=True, exist_ok=True)
    dataset = data.stem

    temp_table: dict[str, float] | None = None
    runner: Any = None
    degrade_reason: str | None = None
    try:
        if baseline == "laya":
            runner = LayaRunner.load(device=device)
        elif baseline == "startlux":
            runner = StartLuxRunner.load(model_dir=startlux_dir or DEFAULT_STARTLUX_DIR, device=device)
            temp_table = runner.temp_table
        else:
            raise ValueError(f"未知基线 {baseline}（可选 laya / startlux）")
    except Exception as exc:  # noqa: BLE001  载入失败统一交由降级判定，不在这里伪装成功
        if not allow_degrade:
            raise
        degrade_reason = f"{type(exc).__name__}: {str(exc)[:180]}"

    cfg = {"baseline": baseline, "dataset": str(data), "split": split, "limit": limit,
           "device": device, "sampling_params": SAMPLING_PARAMS,
           "model": LAYA_ALIAS if baseline == "laya" else str(startlux_dir or DEFAULT_STARTLUX_DIR)}
    ctx = new_run(f"p2-04-{baseline}-cpu", cfg, root=out_root / "runs")

    if degrade_reason is not None:
        row = _downgrade_row(baseline, dataset, split, ctx.run_id, _commit(ctx), card_note)
        row["load_error"] = degrade_reason
        ctx.log_metrics(0, source="card", gate=False, n_samples=0)
        ctx.conclude(f"降级——权重不可得（{degrade_reason}），该行标 source=card/gate=false，不冒充实测。")
        ctx.finish()
        _write_row(out_root / baseline / f"{split}.row.json", row)
        return row

    preds = runner.predict(records)
    metrics = score_predictions(preds, temp_table=temp_table)
    write_rows(out_root / baseline / f"{split}.preds.jsonl", preds)
    ctx.log_metrics(0, n_samples=metrics["n"], acc=metrics["acc"], ece=metrics["ece"],
                    ms_p50=metrics["ms_p50"], device=device)
    row = build_row(baseline=baseline, dataset=dataset, split=split, n=metrics["n"], device=device,
                    sampling=SAMPLING_PARAMS, metrics=metrics, run_id=ctx.run_id, commit=_commit(ctx),
                    source="harness", gate=True,
                    note=f"同 harness 亲跑（CPU 小样本通路；全量延后 C6）。meta={getattr(runner, 'describe_meta', {})}")
    ctx.conclude(f"{baseline} 于 {dataset}/{split} 亲跑 {metrics['n']} 题，"
                 f"acc={metrics['acc']} ECE={metrics['ece']} ms_p50={metrics['ms_p50']}（CPU 通路）。")
    ctx.finish()
    _write_row(out_root / baseline / f"{split}.row.json", row)
    row["buckets"] = metrics["buckets"]
    return row


def _commit(ctx: Any) -> str | None:
    """从 run 账本里取代码版本（new_run 已写入 config.yaml 的 commit 字段）。"""
    try:
        import yaml

        data = yaml.safe_load((ctx.path / "config.yaml").read_text(encoding="utf-8")) or {}
        commit = data.get("commit")
        return str(commit) if commit else None
    except Exception:  # noqa: BLE001  取版本失败不影响出数，记 None 让报告如实呈现
        return None


def _write_row(path: Path, row: dict[str, Any]) -> Path:
    """把对照行写成 json（供 table.py 读取汇总）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    """搭出亲跑 CLI 的参数表（选基线、选数据、切分名、设备、规模与降级开关）。

    白话：先把命令行上能填的空都排好格——跑哪条基线、用哪份题、这刀切分叫什么、取多大规模、落在哪个设备、权重拿不到要不要降级抄卡面，一屏看全，省得来回试错撞墙。
    """
    parser = argparse.ArgumentParser(
        prog="run_baseline.py",
        description="双基线同 harness 亲跑：laya / StartLux-0.8B 各出一条对照行并落 run 记录。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline", required=True, choices=["laya", "startlux"], help="要亲跑的基线")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="typed-decisions 评测集 jsonl")
    parser.add_argument("--split", default="smoke24", help="切分名（写入产物与对照表行标签）")
    parser.add_argument("--limit", type=int, default=SMOKE_LIMIT, help="题数上限（0=全量，全量属 C6）")
    parser.add_argument("--device", default="cpu", help="计算设备（本波固定 cpu）")
    parser.add_argument("--out-root", default=str(REPO_ROOT / "sys1" / "eval" / "baselines"),
                        help="产物根目录（preds/row/runs 落此处）")
    parser.add_argument("--startlux-dir", default=None, help="StartLux 本地权重目录（默认 bench/ms_models 缓存）")
    parser.add_argument("--allow-degrade", action="store_true",
                        help="权重不可得时降级为 source=card/gate=false 行（不冒充实测）")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：解析参数 → 亲跑或降级 → 打印对照行摘要。

    白话：一趟流程的总指挥：收齐参数就开工，把那条基线跑成（或降级成）一行记录，再顺手把几分、几毫秒、出自哪个 run 号打印出来，让人一眼确认这条线到底真跑通了没。
    """
    args = build_parser().parse_args(argv)
    row = run(
        args.baseline, data=Path(args.data), limit=args.limit, device=args.device,
        split=args.split, out_root=Path(args.out_root),
        startlux_dir=Path(args.startlux_dir) if args.startlux_dir else None,
        allow_degrade=args.allow_degrade,
    )
    m = row["metrics"]
    print(f"[baseline] {row['baseline']} split={row['split']} n={row['n_samples']} "
          f"acc={m.get('acc')} ece={m.get('ece')} ms_p50={m.get('ms_p50')} "
          f"source={row['source']} gate={row['gate']} run-id={row['run_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
