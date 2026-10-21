"""双基线对照表生成器（把两条亲跑行拼成一张可并排的表）。

【做什么】
    读取 laya 与 StartLux-0.8B 各自跑完落下的对照行（run_baseline 写的 row.json），把它们
    与"我们自己那版还没跑的教师模型（DML）"、"外部卡面参照（JevBench）"并成一张 markdown
    表：一是按指标逐行对照（准确率、把握误差前后、逐题耗时中位、吞吐），DML 未跑的格子一律
    显式写 `—` 绝不留空；二是每家的运行元信息（数据集/样本数/毫秒/采样参数/端侧标记/run-id）；
    三是全量那一档，如实标 `待 C6`，不拿小样本的数字冒充全量。

【怎么做】
    ① `load_row` 按基线名去找它在某切分下的 row.json，读不到就当作"这格没有实测"；② `fmt`
    把一个数按位数摊成字符串，读不到的一律成 `—`；③ `delta` 算 laya 减 StartLux 的差，任一
    缺数则差也是 `—`；④ `build_compare_md` 把三块表按行拼齐、每行都挂上来源 run-id，末尾补
    一段卡面参照的脚注说明它跟本地不是一把尺；⑤ CLI 只认目录与切分名，产物写成 compare_table.md。

【为什么】
    红线是"同一把尺并排比"，所以表里凡是本地亲跑的格子都有 run-id 背书，凡是抄来的（JevBench）
    或还没跑的（DML、全量）都必须显式标出来，宁可写 `—` 与 `待 C6`，也不给一个没有出处的数——
    否则读者会把"参照上限"当成"我们跑出来的"，那张表就变成误导。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "sys1" / "eval" / "baselines"
DEFAULT_SPLIT = "smoke24"
#: 参与对照的两条参照基线（顺序即表列顺序）；DML 由后续域自填，此处固定留空。
BASELINE_ORDER = ("laya", "startlux")
DML_LABEL = "DML(自训)"
#: 指标行：本地亲跑能对齐的四把尺；kev 是 row.json 里 metrics 的键名。
METRIC_ROWS = (
    ("acc", "准确率", "acc", 4),
    ("ece_after", "ECE(调温后)", "ece_after", 4),
    ("ece_before", "ECE(调温前)", "ece_before", 4),
    ("ms_p50", "逐题耗时中位(ms)", "ms_p50", 1),
    ("throughput_qps", "吞吐(题/秒)", "throughput_qps", 3),
)
#: JevBench 卡面参照（口径与本地 typed-decisions 不同，仅作上限参照，非 gate、非同一 split）。
JEV_REF_NOTE = (
    "卡面 JevBench(/230)：laya 130 · StartLux-0.8B 179 · Jev1.13 199（上限）；"
    "延迟：StartLux 12.2ms(H200)、Jev 64.0ms(API)、laya 仅 CUDA 有效。"
)


def load_row(root: Path, baseline: str, split: str) -> dict[str, Any] | None:
    """取某条基线在某切分下的对照行；文件缺失或格式不符一律返回 None（当没测）。

    白话：去 laya 或 startlux 的小文件夹里翻那张"这次跑出几分"的记录卡。翻到了就把卡交给
    上面拼表；翻不到（还没跑、跑砸了没落盘）就老老实实回一个"没有"，让表格里那一格变成 `—`，
    绝不拿别家的数或凭空的数来填。
    """
    path = root / baseline / f"{split}.row.json"
    if not path.is_file():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return row if isinstance(row, dict) else None


def fmt(value: Any, digits: int) -> str:
    """把一个数按位数摊成字符串；空值/非数一律成占位符 `—`。

    白话：表里只放能对齐版面的东西——是个正经数字就照要的小数位写出来，别的（None、字符串、
    压根没这格）统统换成一条短横。这条短横是有意为之的"这里没实测"，不是排版漏了。
    """
    if value is None or isinstance(value, bool):
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return "—"


def delta(a: Any, b: Any, digits: int) -> str:
    """算 a 减 b 的差；任一不是数，差也如实成 `—`（不硬凑一个假差值）。

    白话：把 laya 和 StartLux 同一格相减看谁高谁低。可只要有一格本身是空的（没测），那这个
    差就没有意义，不能把"空"当 0 去减出一个看着像真的的数——那种差最会骗人。
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return "—"
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return "—"
    return f"{a - b:+.{digits}f}"


def _metrics(row: dict[str, Any] | None) -> dict[str, Any]:
    """从对照行里稳妥地取出 metrics 子字典（缺行给空表，让上层自然落 `—`）。"""
    if not row:
        return {}
    m = row.get("metrics")
    return m if isinstance(m, dict) else {}


def build_metric_table(rows: dict[str, dict[str, Any] | None]) -> list[str]:
    """拼第一块表：按指标逐行对照，六字段（指标 / DML / laya / startlux / Δ / jev-ref）。

    白话：这是给"同一把尺量出来的三条线"排座次的表——最左是量的是什么，中间留着我们自训模型
    那一列（现在还没跑，全填短横），再往后是 laya、StartLux 各自实测的数，然后是它俩之差，
    最后一列只放外部卡面当参照。DML 没跑就明写短横，读者一眼看出哪列还空着。
    """
    laya = _metrics(rows.get("laya"))
    startlux = _metrics(rows.get("startlux"))
    lines = [
        f"| 指标 | {DML_LABEL} | laya | startlux-0.8b | Δ(laya−startlux) | jev-ref(卡面·异口径) |",
        "|---|---|---|---|---|---|",
    ]
    for _key, label, mkey, digits in METRIC_ROWS:
        lv = fmt(laya.get(mkey), digits)
        sv = fmt(startlux.get(mkey), digits)
        dv = delta(laya.get(mkey), startlux.get(mkey), digits)
        lines.append(f"| {label} | — | {lv} | {sv} | {dv} | 见下方脚注 |")
    return lines


def build_meta_table(rows: dict[str, dict[str, Any] | None], split: str) -> list[str]:
    """拼第二块表：每条基线的运行元信息（样本数/毫秒/采样/端侧/run-id/来源/gate）。

    白话：指标只说"几分"，这张表说的是"这几分是怎么来的"——在哪份题上、多少道、单题多慢、
    按什么采样、跑在端侧还是云上、哪次实验、是亲测还是抄卡面、进不进最终判定。有了这排出处，
    上面那张分数表才站得住，不然就是无根的数字。
    """
    header = [
        "| 基线 | 数据集/切分 | 样本数 | 逐题耗时中位(ms) | 吞吐(题/秒) | 采样参数 | 端侧标记 | run-id | source | gate |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for baseline in BASELINE_ORDER:
        row = rows.get(baseline)
        if not row:
            header.append(f"| {baseline} | {split} | — | — | — | — | — | — | 未跑 | false |")
            continue
        m = _metrics(row)
        header.append(
            "| {b} | {ds}/{sp} | {n} | {ms} | {tps} | {smp} | {dev} | {rid} | {src} | {gate} |".format(
                b=baseline, ds=row.get("dataset", "—"), sp=row.get("split", split),
                n=row.get("n_samples", "—"), ms=fmt(m.get("ms_p50"), 1), tps=fmt(m.get("throughput_qps"), 3),
                smp=row.get("sampling_params", "—"), dev=row.get("on_device", "—"),
                rid=row.get("run_id", "—"), src=row.get("source", "—"),
                gate="true" if row.get("gate") else "false",
            )
        )
    return header


def build_full_pending_table(split_full: str) -> list[str]:
    """拼第三块表：全量那一档，所有实测格一律显式标 `待 C6`（不拿小样本冒充全量）。

    白话：本波只在 Mac 的 CPU 上跑了个小样本把管线走通，真正的整卷评测要等云上的 NPU 机器。
    这张表就把"全量"那一行先占好位，每个该填数的格子都写明"待 C6"，谁看都知道这不是偷懒漏填、
    而是资源排期使然——避免有人把 24 题的分数当成全量成绩拿去汇报。
    """
    lines = [
        f"| 指标 | {DML_LABEL} | laya | startlux-0.8b | Δ(laya−startlux) | jev-ref(卡面·异口径) |",
        "|---|---|---|---|---|---|",
    ]
    for _key, label, _mkey, _digits in METRIC_ROWS:
        lines.append(f"| {label} | 待 {split_full} | 待 {split_full} | 待 {split_full} | 待 {split_full} | 见下方脚注 |")
    return lines


def build_compare_md(root: Path, split: str, split_full: str) -> str:
    """把三块表与脚注组装成完整 markdown 文本（对照表正文一次生成完毕）。

    白话：先取三块料——指标对照、运行出处、全量占位——各拼成一段表，再照顺序叠起来，顶上写清
    这是哪个切分、脚下补一段卡面参照的说明和一句"卡面跟本地不是一把尺"的提醒。拼出来就是一页
    能直接贴进报告、每一格都能追问出处的对照表。
    """
    rows = {b: load_row(root, b, split) for b in BASELINE_ORDER}
    ran = sum(1 for b in BASELINE_ORDER if rows.get(b))
    parts: list[str] = [
        "# 双基线对照表（p2-04 · 同 harness 亲跑）",
        "",
        f"> 本波切分 `{split}`（CPU 小样本通路，≥20 题真实出分）；全量 `{split_full}` 延后至云端 C6（NPU）。",
        f"> 已落实测的基线：{ran}/{len(BASELINE_ORDER)}（未落者以 `—`/`未跑` 显式标注，不冒充）。",
        "",
        "## 一、指标对照（同 split、同采样口径）",
        "",
    ]
    parts += build_metric_table(rows)
    parts += ["", "## 二、运行元信息（出处与设备）", ""]
    parts += build_meta_table(rows, split)
    parts += ["", f"## 三、全量档（`{split_full}`，待 C6 云端 NPU 跑）", ""]
    parts += build_full_pending_table(split_full)
    parts += [
        "",
        "## 脚注：外部卡面参照（非本地亲跑，仅上限参照）",
        "",
        f"- {JEV_REF_NOTE}",
        "- JevBench 与本地 typed-decisions 是不同数据集、不同口径，卡面值只作量级参照，"
        "不参与 Δ 计算，也不作 gate。",
        "- D11 边界：StartLux-0.8B 仅作对照评测推理（白名单③），绝不训练、绝不当基座、"
        "绝不二次发布其权重或衍生。",
        "",
    ]
    return "\n".join(parts)


def build_parser() -> argparse.ArgumentParser:
    """搭出对照表生成器的参数表（产物根目录、小样本切分名、全量切分名、输出路径）。

    白话：把拼这张表要交代的来源都留好默认——去哪儿读两条基线的记录卡、这轮小样本切分叫什么名、全量那一档又标成什么、最后的表落到哪个文件，一处不填也有兜底值。
    """
    parser = argparse.ArgumentParser(
        prog="table.py",
        description="读取双基线 row.json，生成并排对照表 compare_table.md（六字段 + 出处 + 全量占位）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="基线产物根目录（含 laya/、startlux/）")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="小样本切分名（读 row.json 用）")
    parser.add_argument("--split-full", default="full", help="全量档标签（本波一律标待跑）")
    parser.add_argument("--out", default=str(REPO_ROOT / "production" / "baselines" / "compare_table.md"),
                        help="对照表输出路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：读三块表所需数据 → 生成 markdown → 落盘并回显一行摘要。

    白话：命令一敲就把三块料按顺序拼齐、写成文件，再回一行告诉人写到了哪儿、总共多少行、这轮用的哪个切分，成没成、写在哪，当场分明不必翻目录猜。
    """
    args = build_parser().parse_args(argv)
    md = build_compare_md(Path(args.root), args.split, args.split_full)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"[compare_table] 写入 {out}（{len(md.splitlines())} 行，切分 {args.split}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
