"""learning/s2_decision_sft.py — S2 决策化微调 CLI：预训练起点 → 读出位 CE → 模型目录。

【做什么】
    载一个已预训练好的 GPT 目录，在决策样本上做"读出位交叉熵"微调：只让模型在每个
    样本最后一个真符号那一格，学会把 26 个字母候选里对的那一个的"胆量分"顶上去；
    训练完把权重、配置、以及一份决策配置（读点位口径 + 留给 s3 的温度槽）一起落盘。

【怎么做】
    三段。① 编码：每个样本先走 decision 渲染层排成定死文字，编成编号串；同时记下
    "第几个字母代表哪个候选"，把 typed 目标折成与字母列等长、和为 1 的软标签向量。
    ② 计损（唯一硬约束）：loss 只在读出位算——复用 `sys1.decision.option_scores`
    取末位那串数字 × 字母行得到候选分，再对整列软标签做交叉熵；其余位置的逐位置数字
    压根不进这条式子（`option_scores` 按行 gather 真末位，非末位的输出对本步无贡献，
    故它们的梯度恒为零，tests 里 -k mask 手工验证）。③ 训练/评测：按候选数分桶组批、
    右补空洞（读出侧靠 lengths 收口，补洞不改读数），Adam 全参微调若干轮；留出集上
    以"读出 argmax == 目标 argmax"记分，逐桶对照 `calibrate.bucket_baselines`。
    超参全进 configs，loss/acc 走 run-notebook，产出目录含 decision_config.json。

【为什么】
    读出位监督而非逐 token 自回归监督，是因为本引擎根本不往外生成字母——答案从末位
    数字里读出来。若在整段文字上做逐位预测，等于逼模型去续写提示语，与"零生成"红线冲突。
    被否方案一：把 CE 铺满所有位置——多算 vocab−1 倍、还把监督信号稀释到与决策无关的
    续写上；读出位 CE 把力气全花在"该选哪个字母"这一件事上。
    被否方案二：s2 里自造一套"取末位、点积字母行"——与 p1-07 读出层两份真源必然漂移，
    换 backbone 时两处要同改；故强制复用 `option_scores`（tests -k mask 附带 grep 门）。
    被否方案三：一阶段就上 LoRA——模型才 40M 级，全参微调更简单且无额外框架依赖；LoRA
    留给二阶段 0.8B（父 design 已裁）。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

# 仓库根引导：脚本需支持 `python learning/s2_decision_sft.py` 从任意 cwd 直接跑，
# 而 editable 安装在部分环境未把仓库根挂上 sys.path，这里显式补齐（仅本域 CLI 兜底）。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sys1.calibrate import bucket_baselines
from sys1.data.schema import validate_sample
from sys1.decision import (
    LETTERS,
    RENDER_VERSION,
    from_systemone,
    option_scores,
    prompt_text,
    render,
)
from sys1.lang import bpe
from sys1.model import DECISION_CONFIG_FILE, Decoder
from sys1.runs import new_run

REPO_ROOT = _REPO_ROOT
DEFAULT_TOKENIZER_DIR = REPO_ROOT / "runs" / "1003-s0-bpe-16k-realedu-zh-en" / "tokenizer"
PAD_TOKEN = bpe.PAD_TOKEN
SFT_VERSION = "dmlaya_s2_sft_v1"


# ---------------------------------------------------------------- B1：读出位交叉熵（复用 readout）
def readout_ce_loss(
    last_hidden: torch.Tensor,
    head_weight: torch.Tensor,
    letter_ids: list[int],
    targets: torch.Tensor,
    lengths: list[int] | None = None,
) -> torch.Tensor:
    """只在读出位对整列软标签计交叉熵；非读出位置的逐位置数字不参与，梯度自然为零。

    白话：一批样本各写到末尾留下一串可学的数字，把这几个末位数字分别跟 A、B、C…
    二十六个字母的"方向"配一配钥匙，谁配得最响就最像答案；再把人工给的那份"各候选
    该占几成"整列摊开比对，越贴得近罚得越轻。中间那些没写到末尾的位置一律不看，
    于是它们对罚分既不出力也不担责。

    :param last_hidden: `(B,T,d)`——模型前向交出的逐位置数字。
    :param head_weight: `(vocab,d)`——输出层权重（decision 读的就是 26 个字母那几行）。
    :param letter_ids:  本轮候选字母的编号，长度 = k，顺序即字母序（A 在第一个）。
    :param targets:     `(B,k)` 软标签，每行和为 1（多人投票份额照录不硬化）。
    :param lengths:     每行真符号数；None 表示按整行最右一格读。
    :returns:           标量 loss（对批内样本取均值）。
    """
    # 唯一取末位的入口交给读出层：option_scores 已按行 gather 真末位并只点字母行。
    scores = option_scores(last_hidden, head_weight, letter_ids, lengths)   # (B,k)
    log_prob = torch.log_softmax(scores.float(), dim=-1)
    tgt = targets.to(scores.dtype)
    if tgt.shape != scores.shape:
        raise ValueError(f"targets 形状 {tuple(tgt.shape)} 与读出分数 {tuple(scores.shape)} 不符")
    # soft target 整体监督：CE = −Σ_i p_i · log q_i，再对批内样本取均值。
    return -(tgt * log_prob).sum(dim=-1).mean()


# ---------------------------------------------------------------- 样本编码
def encode_example(tokenizer: Any, letter_map: dict[str, int], record: dict[str, Any]) -> dict[str, Any]:
    """把一条转写记录（{id, sample}）编成训练/评测可用的读出样本：编号串 + 字母列 + 软标签。

    白话：拿一条"题目+标准答案"的单子，先把证据、问题、候选排成模型要读的那段死文字，
    再照表把它换成一串编号；同时按排好的候选顺序，给每个字母记下"这格代表哪个候选"，
    并把答案折成一列"各字母该占几成"的软标签，跟字母列一样长。

    :returns: {"id","qtype","k","letter_ids","input_ids","length","target"}（target 和为 1）。
    """
    sample = validate_sample(record["sample"])
    qid = next(iter(sample["questions"]))           # 转写产物单题单问，取唯一题号
    question = sample["questions"][qid]
    row = from_systemone(sample["state"], question, qid=qid)
    _, order = render(row)                          # order[i] = 字母 LETTERS[i] 代表的候选代号
    prompt = prompt_text(row, order)
    ids = list(tokenizer.encode(prompt, add_special_tokens=False).ids)
    if not ids:
        raise ValueError(f"样本 {record.get('id')} 编码后为空，没有可读位置")
    k = len(order)
    target_map = sample.get("targets", {}).get(qid, {})
    raw = [float(target_map.get(order[i], 0.0)) for i in range(k)]
    total = sum(raw)
    if total <= 0.0:
        raise ValueError(f"样本 {record.get('id')} 目标分布全为零，无监督信号")
    target = [v / total for v in raw]               # 与字母列对齐、和严格为 1 的软标签
    return {
        "id": record.get("id"),
        "qtype": question["type"],
        "k": k,
        "letter_ids": [letter_map[LETTERS[i]] for i in range(k)],
        "input_ids": ids,
        "length": len(ids),
        "target": target,
    }


def group_by_k(examples: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """把样本按候选数分堆，返回 {k: [样本...]}。

    白话：选择题有的三个候选、有的五个，字母列长度就不一样，没法并排坐进一批。这里先按
    "有几个候选"把它们分堆，同一堆里候选数相同，后面组批补齐才对齐得上。
    """
    buckets: dict[int, list[dict[str, Any]]] = {}
    for ex in examples:
        buckets.setdefault(ex["k"], []).append(ex)
    return buckets


def make_batch(bucket: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    """把一个 k 桶组成一个右补空洞的批：编号矩阵 + 可见掩码 + 每行长度 + 软标签矩阵。

    白话：同一桶里的样本候选数一样，就能并排坐进一批。长短不齐就在每行右边垫空洞补齐，
    并记住每行真正写到了第几格——读答案时只在那一格收，后面的垫洞一概不算数。

    :returns: {"input_ids","attn_mask","lengths","letter_ids","targets"}（均为张量/列表）。
    """
    batch = len(bucket)
    width = max(ex["length"] for ex in bucket)
    input_ids = torch.full((batch, width), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((batch, width), dtype=torch.long)
    targets = torch.zeros((batch, bucket[0]["k"]), dtype=torch.float32)
    lengths: list[int] = []
    for i, ex in enumerate(bucket):
        ids = ex["input_ids"]
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn_mask[i, : len(ids)] = 1
        targets[i] = torch.tensor(ex["target"], dtype=torch.float32)
        lengths.append(ex["length"])
    # 同桶 letter_ids 只由 k 决定（恒为 A..第 k 个字母），取桶内任一行即可。
    return {
        "input_ids": input_ids,
        "attn_mask": attn_mask,
        "lengths": lengths,
        "letter_ids": bucket[0]["letter_ids"],
        "targets": targets,
    }


# ---------------------------------------------------------------- 训练与评测
def train_one_epoch(model: Decoder, examples: list[dict[str, Any]], optimizer: torch.optim.Optimizer,
                    pad_id: int, batch_size: int) -> float:
    """在一个个 k 分桶上小批遍历整轮，返回平均读出位罚分。

    白话：把同一类（候选数相同）的样本打乱分成小捆，一捆一捆地过：让模型读一遍、按末位
    那格算一次罚分、调一次旋钮；一轮下来把各捆的罚分平均起来报出去。

    :returns: 本轮内各小捆罚分的均值。
    """
    model.train()
    buckets = group_by_k(examples)
    losses: list[float] = []
    for k in sorted(buckets):
        rows = buckets[k]
        order = torch.randperm(len(rows))
        for start in range(0, len(rows), batch_size):
            chunk = [rows[int(i)] for i in order[start : start + batch_size]]
            batch = make_batch(chunk, pad_id)
            optimizer.zero_grad(set_to_none=True)
            last_hidden = model(batch["input_ids"], batch["attn_mask"])
            loss = readout_ce_loss(
                last_hidden, model.lm_head.weight, batch["letter_ids"], batch["targets"], batch["lengths"]
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
    return sum(losses) / len(losses) if losses else float("nan")


@torch.no_grad()
def evaluate(model: Decoder, examples: list[dict[str, Any]], pad_id: int, batch_size: int) -> dict[str, dict[int, float]]:
    """留出集读出打分：按 `qtype×k` 分桶统计"argmax 读出 == argmax 目标"的准确率。

    白话：把评测样本也按候选数分好类、补洞组批，让模型只读一次末位那格；哪个字母配得
    最响就当它是答案，再跟标准答案比，一样就算蒙对。按题型和候选数分开数命中率，免得
    把三选一的成绩拿去冒充二十选一的合格线。

    :returns: {qtype: {k: accuracy}}。
    """
    model.eval()
    hits: dict[tuple[str, int], list[int]] = {}
    for k, rows in group_by_k(examples).items():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            batch = make_batch(chunk, pad_id)
            last_hidden = model(batch["input_ids"], batch["attn_mask"])
            scores = option_scores(last_hidden, model.lm_head.weight, batch["letter_ids"], batch["lengths"])
            pred = scores.argmax(dim=-1)
            gold = batch["targets"].argmax(dim=-1)
            ok = (pred == gold).to(torch.float32)
            for ex, flag in zip(chunk, ok.tolist()):
                hits.setdefault((ex["qtype"], k), []).append(int(flag))
    out: dict[str, dict[int, float]] = {}
    for (qtype, kk), vals in hits.items():
        out.setdefault(qtype, {})[kk] = sum(vals) / len(vals)
    return out


def write_decision_config(model_dir: Path, meta: dict[str, Any]) -> Path:
    """把 SFT 产物的决策配置落盘（读点位口径 + 留给 s3 的空温度槽），返回文件路径。

    白话：随模型附一张说明书，写清答案是从末位那格按二十六字母读的、排版用的是哪版、
    以及这次微调的成绩单。温度那一栏先留个空槽，等下一步校准拟合好了再把数填进去。
    """
    path = model_dir / DECISION_CONFIG_FILE
    config = {
        "version": SFT_VERSION,
        "owner": "p1-09-sft-pipeline",
        "render_version": RENDER_VERSION,
        "readout": {"position": "last_real_token", "letters": list(LETTERS), "max_options": len(LETTERS)},
        "temperatures": {},                 # 占位空表：真实温度由 s3_calibrate 拟合后写入
        "calibration": {"status": "pending"},
        "sft": meta,
    }
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- CLI
def load_records(path: str | Path) -> list[dict[str, Any]]:
    """读一个转写数据文件（jsonl，一行一个 {id, task, sample} 信封），原样返回列表。

    白话：把盘里那份"一行一道题"的数据档逐行拆开读回来，空行跳过、坏行不猜——每行还原成
    一条带来源号与 typed 样本的信封，交给后面编码与训练去用。
    """
    records: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def split_holdout(records: list[dict[str, Any]], holdout: int, seed: int) -> tuple[list, list]:
    """按固定种子打乱后切出留出集，返回 (训练集, 留出集)；留出规模取 holdout 条。

    白话：先把整叠题照一个定死的洗牌法打乱（同样的种子必洗出同样的顺序，好复现），再从
    顶上抽出指定数量的题留作"考试专用"，这些题绝不参与调旋钮；剩下的拿去训练。
    """
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    n_hold = max(0, min(holdout, len(records) - 1))
    hold = set(idx[:n_hold])
    train = [r for i, r in enumerate(records) if i not in hold]
    heldout = [r for i, r in enumerate(records) if i in hold]
    return train, heldout


def run_sft(args: argparse.Namespace) -> dict[str, Any]:
    """SFT 主体：载入起点 → 编码转写集 → 读出位 CE 微调 → 留出打分 → 落模型目录与决策配置。

    白话：先接过一个已预训练好的起点模型，把转写来的题目一道道排成死文字、编成编号，
    然后只按每题末尾那一格反复调旋钮，让正确字母一次比一次更响；调完拿没见过的题考它，
    分题型数一数命中率，最后把调好的模型、配置、成绩单一起收进一个目录。

    :returns: 摘要字典（含各桶准确率、对应分桶随机基线、产出路径与 run-id）。
    """
    torch.manual_seed(args.seed)
    tokenizer = bpe.load(args.tokenizer)
    letter_map = bpe.check_tokenizer(tokenizer)
    pad_id = int(tokenizer.token_to_id(PAD_TOKEN) or 0)

    model = Decoder.load(args.ckpt)
    model.train()

    records = load_records(args.data)
    train_recs, held_recs = split_holdout(records, args.holdout, args.seed)
    train_ex = [encode_example(tokenizer, letter_map, r) for r in train_recs]
    held_ex = [encode_example(tokenizer, letter_map, r) for r in held_recs]

    run = new_run("s2-decision-sft", _run_config(args, train_recs, held_recs))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for step in range(args.epochs):
        loss = train_one_epoch(model, train_ex, optimizer, pad_id, args.batch_size)
        run.log_metrics(step, train_loss=loss)

    acc = evaluate(model, held_ex, pad_id, args.batch_size)
    baselines = bucket_baselines()
    summary = _summarize(acc, baselines, args, len(train_ex), len(held_ex))
    acc_row = {f"acc_{qt}_{kk}": v for qt, d in acc.items() for kk, v in d.items()}
    if acc_row:                                  # 留出集为空时不记指标（log_metrics 拒绝空记录）
        run.log_metrics(args.epochs, **acc_row)

    model_out = run.path / "model"
    model.eval().save(model_out)
    write_decision_config(model_out, summary)
    run.conclude(summary["conclusion"])
    run.finish()

    _print_summary(run.run_id, str(model_out), summary)
    return {"run_id": run.run_id, "model_dir": str(model_out), "summary": summary, "accuracy": acc}


def _run_config(args: argparse.Namespace, train_recs: list, held_recs: list) -> dict[str, Any]:
    """把这次 SFT 的可复现要素攒成 run config（含数据配方 data_revision）。"""
    return {
        "stage": "s2-decision-sft",
        "ckpt": str(args.ckpt),
        "tokenizer": str(args.tokenizer),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "holdout": args.holdout,
        "n_train": len(train_recs),
        "n_holdout": len(held_recs),
        # 数据配方（父 design D8 防泄漏）：来源与切分如实入 data_revision
        "data_revision": {
            "train": "transcribed(XNLI-zh->noul, MASSIVE-zh-subset->choice, rating->score) + Intern-Decision-train",
            "holdout": "in-domain transcribed split (NOT typed-decisions test)",
            "leak_policy": "typed-decisions test set is eval-only, excluded from SFT train (tools/check_leak.py)",
        },
    }


def _summarize(acc: dict, baselines: dict, args: argparse.Namespace, n_train: int, n_held: int) -> dict[str, Any]:
    """把各桶准确率与对应分桶随机基线并排，如实给出"是否超基线"的结论行。"""
    table: dict[str, Any] = {}
    all_beat = True
    for qtype, buckets in acc.items():
        for k, value in buckets.items():
            base = baselines.get(qtype, {}).get(k)
            base_val = float(base.value) if base else 0.0
            beat = value > base_val
            all_beat = all_beat and beat
            table.setdefault(qtype, {})[k] = {"accuracy": round(value, 4), "baseline": base_val, "beats_baseline": beat}
    conclusion = (
        f"s2 smoke 完成：train={n_train} holdout={n_held} epochs={args.epochs} lr={args.lr}；"
        f"留出集各桶准确率 vs 分桶随机基线：{table}；"
        f"{'全部桶超基线' if all_beat else '存在未超基线桶（弱模型，如实记录）'}。"
    )
    return {"per_bucket": table, "all_beat_baseline": all_beat, "conclusion": conclusion,
            "n_train": n_train, "n_holdout": n_held, "epochs": args.epochs, "lr": args.lr}


def _print_summary(run_id: str, model_dir: str, summary: dict[str, Any]) -> None:
    """把关键数字打到 stdout（smoke 日志一眼可读，也供 e2e 测试抓取）。"""
    print(f"[s2] run-id      : {run_id}")
    print(f"[s2] model dir   : {model_dir}")
    print(f"[s2] all_beat    : {summary['all_beat_baseline']}")
    for qtype, buckets in summary["per_bucket"].items():
        for k, row in buckets.items():
            print(f"[s2] acc {qtype}/k={k}: {row['accuracy']:.4f} (baseline {row['baseline']:.4f}) "
                  f"beats={row['beats_baseline']}")


def build_parser() -> argparse.ArgumentParser:
    """搭出 s2 CLI 参数表（起点/词表/数据/超参/留出规模），供 --help 与 main 共用。

    白话：把能敲的选项和默认值一次列清楚，让人敲一下 --help 就看到一份说明书：接哪个起点
    模型、用哪张表、喂哪份转写数据、调几轮、每轮一捆多大、留多少题专门用来考试。
    """
    parser = argparse.ArgumentParser(
        prog="s2_decision_sft.py",
        description="读出位 SFT：载预训练起点 → 决策样本 CE（soft target，仅末位计损）→ 产出模型目录。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", default=None, help="S1 预训练起点模型目录（含 config.json/weights.safetensors）")
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER_DIR), help="S0 自训 BPE 产物目录")
    parser.add_argument("--data", default=None, help="转写训练集 jsonl（一行一个 {id,task,sample}）")
    parser.add_argument("--epochs", type=int, default=6, help="全参微调轮数")
    parser.add_argument("--lr", type=float, default=0.03, help="Adam 学习率")
    parser.add_argument("--batch-size", type=int, default=32, help="每个 k 分桶内的小批规模")
    parser.add_argument("--holdout", type=int, default=100, help="留出集条数（不参与训练，专供评测）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（切分与初始化可复现）")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：解析参数 → 校验起点/数据存在 → 跑 run_sft → 退出码 0。

    白话：命令行跑起来先核对两样东西在不在——起点模型目录、转写数据文件；缺任何一样就
    停下报错，绝不拿半截输入硬跑。都在就交给主体走完"调旋钮—考试—归档"，最后正常收工；
    smoke 档不设"必须超基线才放行"的硬闸门，超没超都在摘要与 run 记录里如实报出。
    """
    args = build_parser().parse_args(argv)
    if args.ckpt is None or args.data is None:
        raise SystemExit("--ckpt 与 --data 必填（起点模型目录 + 转写训练集 jsonl）")
    if not Path(args.ckpt).is_dir():
        raise SystemExit(f"起点模型目录不存在：{args.ckpt}")
    if not Path(args.data).is_file():
        raise SystemExit(f"转写训练集不存在：{args.data}")
    run_sft(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
