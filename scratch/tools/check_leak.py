"""tools/check_leak.py — SFT 训练集与评测测试集的 id 交集自检（父 design D8 防泄漏）。

【做什么】
    比对了不起：训练侧喂进去的样本 id，绝不能和评测用的 typed-decisions test 集有
    任何重叠。给两个路径（训练集、测试集），把各自的 id 收齐，算交集；交集为空即
    放行（退出码 0），只要冒出一个共同 id 就判泄漏（退出码 1）并把肇事 id 列出来。

【怎么做】
    路径既给单个 jsonl 文件、也给目录（目录收下其中所有 *.jsonl）。每行还原成一个
    dict，按 "先取顶层 id、再退到信封里的 id" 的顺序找一个字符串 id；找不到 id 的行
    记为 "无 id 行"，逐文件计数如实报出——宁可提示"这批数据没带 id、无从比对"，也不
    假装它们互不重叠。真正判定只看一件事：两边的 id 集合有没有交集。

【为什么】
    把 id 与 typed 样本分家存放（样本契约顶层字段封闭、不许塞 id），溯源信息只能住在
    记录信封上；于是防泄漏必须在"记录"这一层比 id，而不是钻进样本内部。这个工具就是
    那条边界的机器看门狗。

用法：
    python tools/check_leak.py --train <sft_data> --test bench/typed_decisions
退出码：0=无泄漏；1=检出交集（或显式 --fail-on-missing-id 时某侧完全无 id）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def iter_jsonl_files(path: str | Path) -> list[Path]:
    """把一个路径摊平成 jsonl 文件清单：单文件原样返回，目录收其下所有 *.jsonl。"""
    target = Path(path)
    if target.is_dir():
        return sorted(p for p in target.rglob("*.jsonl") if p.is_file())
    if target.is_file():
        return [target]
    raise FileNotFoundError(f"路径不存在：{target}")


def _extract_id(record: Any) -> str | None:
    """从一条记录里取字符串 id：先看顶层 id，再退到信封 {id, sample} 的顶层；都不是则 None。"""
    if isinstance(record, dict):
        candidate = record.get("id")
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


def collect_ids(path: str | Path) -> tuple[set[str], int, int]:
    """收集一个路径下所有 jsonl 的 id，返回 (id 集合, 总行数, 无 id 行数)。

    坏行（无法 json 解析）按"无 id"计数、不中断——数据里夹一行的抖动不该让整个自检崩掉，
    但计数会如实反映，最终由调用方看总账判断这批数据是否可用。
    """
    ids: set[str] = set()
    total = 0
    missing = 0
    for file in iter_jsonl_files(path):
        for raw in file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                missing += 1
                continue
            found = _extract_id(record)
            if found is None:
                missing += 1
            else:
                ids.add(found)
    return ids, total, missing


def check(train_path: str | Path, test_path: str | Path) -> dict[str, Any]:
    """跑一次泄漏比对，返回结构化结果（交集、计数、放行与否）。"""
    train_ids, train_total, train_missing = collect_ids(train_path)
    test_ids, test_total, test_missing = collect_ids(test_path)
    overlap = train_ids & test_ids
    return {
        "ok": not overlap,
        "train_ids": len(train_ids),
        "train_lines": train_total,
        "train_lines_without_id": train_missing,
        "test_ids": len(test_ids),
        "test_lines": test_total,
        "test_lines_without_id": test_missing,
        "overlap_count": len(overlap),
        "overlap_sample": sorted(overlap)[:20],
    }


def build_parser() -> argparse.ArgumentParser:
    """搭出命令行参数表（--train / --test 两个路径，可文件或目录）。"""
    parser = argparse.ArgumentParser(
        prog="check_leak.py",
        description="SFT 训练集 id ∩ typed-decisions test id 必须为空；有交集即判泄漏。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train", required=True, help="训练集：jsonl 文件或含 *.jsonl 的目录")
    parser.add_argument("--test", required=True, help="测试集（typed-decisions）：jsonl 文件或目录")
    parser.add_argument("--fail-on-missing-id", action="store_true",
                        help="任一侧完全取不到 id 时也判失败（默认只提示不失败）")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 主体：读两边 id → 算交集 → 打印结论；无泄漏退出 0，检出泄漏退出 1。"""
    args = build_parser().parse_args(argv)
    report = check(args.train, args.test)
    print(f"[check_leak] train: {report['train_ids']} ids / {report['train_lines']} lines "
          f"(无 id {report['train_lines_without_id']})")
    print(f"[check_leak] test : {report['test_ids']} ids / {report['test_lines']} lines "
          f"(无 id {report['test_lines_without_id']})")
    print(f"[check_leak] overlap: {report['overlap_count']}")

    if report["overlap_count"]:
        print(f"❌ 检出泄漏：训练集与测试集共有 {report['overlap_count']} 个 id，样例 {report['overlap_sample']}")
        return 1
    empty_side = report["train_ids"] == 0 or report["test_ids"] == 0
    if empty_side and args.fail_on_missing_id:
        print("❌ 有一侧完全取不到 id（--fail-on-missing-id 已置位，判失败）")
        return 1
    if empty_side:
        print("⚠️ 有一侧取不到任何 id，交集自然为空——请确认数据是否带 id，否则本自检无意义")
    print("✅ 无泄漏：训练样本 id ∩ test 集 id = ∅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
