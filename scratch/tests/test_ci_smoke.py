"""课程 CI 冒烟测试（stdlib-only，无 torch 依赖，任何环境可跑）。

【做什么】CI 自己也要被测：检查注释工具判得对不对、两份手册评分表是否合计 100、目录指引是否齐全。
【怎么做】直接 import tools/check_comments.py 用内存样例过 R1/R2；正则解析 markdown 评分表求和；枚举必需 README。
【为什么】rubric 权重曾被改错（92≠100）靠人眼没抓住——用测试锁死；曾考虑只靠 review——否，文档也会被改。

运行: pytest tests 或 python3 tests/test_ci_smoke.py（无 pytest 时手动模式）。
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("check_comments", ROOT / "tools" / "check_comments.py")
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

BAD_SRC = textwrap.dedent('''
    def pick_best(scores):
        """对 scores 做 softmax 归一化后取 argmax，返回最佳 embedding 索引。"""
        total = sum(scores)
        probs = [s / total for s in scores]
        idx = probs.index(max(probs))
        return idx
''').lstrip()

GOOD_SRC = textwrap.dedent('''
    """【做什么】把分数折算成份额。
    【怎么做】求和后逐个除，得到总和 1 的列表。
    【为什么】保原始分会被极端值支配；曾考虑取 max 不折算——否，拿不到确信度。"""


    def share_scores(raw):
        """折算份额。

        白话：把每个候选的胆量分倒进同一个盆里，看各能舀回几成——
        舀到最多的就是答案，那几成同时就是我们对它的确信程度。"""
        total = sum(raw) or 1
        out = []
        for x in raw:
            out.append(x / total)
        return out
''').lstrip()


def _check(src: str, tmp_dir: str) -> list[str]:
    p = Path(tmp_dir) / "sample.py"
    p.write_text(src, encoding="utf-8")
    return cc.check_file(p, ROOT)


def test_comment_gate_flags_jargon_only():
    """坏样例必须同时报 R1 与 R2。"""
    with tempfile.TemporaryDirectory() as td:
        errs = _check(BAD_SRC, td)
    assert any("R1 文件头" in e for e in errs), errs      # 错误串带文件路径前缀，按包含判定
    assert any("R2" in e and "白话" in e for e in errs), errs


def test_comment_gate_passes_plain_sample():
    """好样例（白话段无黑话）必须零违例。"""
    with tempfile.TemporaryDirectory() as td:
        errs = _check(GOOD_SRC, td)
    assert errs == [], errs


def _table_weights(md: str, anchor: str) -> list[int]:
    i = md.index(anchor)
    block = md[i:].split("\n---", 1)[0]
    ws = []
    for ln in block.splitlines():
        if not ln.strip().startswith("|") or set(ln) <= set("|-: "):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        nums = [c for c in cells if c.isdigit()]   # 评分表恰有一个纯数字权重列（加分行 +N 不匹配）
        if len(nums) == 1:
            ws.append(int(nums[0]))
    return ws


def test_rubrics_sum_to_100():
    """两份手册的评分表合计必须恰好 100（加分项除外）。"""
    g = _table_weights((ROOT / "GUIDE.md").read_text("utf-8"), "**评分（一阶段，满分 100")
    assert sum(g) == 100, g
    p = _table_weights((ROOT / "PRODUCTION.md").read_text("utf-8"), "## 10. 评分标准（正式版 Rubric，总分 100）")
    assert sum(p) == 100, p


def test_required_readmes_exist():
    for d in ["", "sys1", "sys1/decision", "sys1/kernels", "sys1/layers",
              "sys1/data", "sys1/eval", "learning", "production", "serving", "runs", "report"]:
        assert (ROOT / d / "README.md").is_file(), d


def test_ci_entry_and_hooks_present():
    assert (ROOT / "tools" / "ci.sh").is_file()
    assert (ROOT / ".githooks" / "pre-commit").is_file()


if __name__ == "__main__":  # 无 pytest 时的手动模式
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except AssertionError as e:
                fails += 1
                print(f" FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
