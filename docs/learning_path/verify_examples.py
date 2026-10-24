"""验证本教材的 Python 演示；不运行项目训练、下载或设备测试。

在已装 NumPy/PyTorch 的 CPU 环境运行。只执行本目录内由维护者确认过的
Markdown Python 代码块；这不是可安全执行任意外来文档的沙箱。
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
from pathlib import Path
import re
from urllib.parse import unquote


def main() -> None:
    parser = argparse.ArgumentParser(description="复跑教材代码块并检查本地文件链接")
    parser.add_argument("files", nargs="*", help="教材 Markdown 文件名，或 ../teaching_gdn_bwd_verification.md")
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("repeat 应为正数")
    root = Path(__file__).resolve().parent
    legacy = root.parent / "teaching_gdn_bwd_verification.md"
    files = [root / name for name in args.files] if args.files else sorted(root.glob("*.md")) + [legacy]
    link_files = files + sorted((root.parent / "reassessment").glob("*.md"))
    count = 0
    for path in link_files:
        text = path.read_text(encoding="utf-8")
        for link in re.findall(r"\]\(([^)]+)\)", text):
            if "://" in link or link.startswith("#"):
                continue
            target = unquote(link.split("#", 1)[0])
            if target and not (path.parent / target).exists():
                raise AssertionError(f"链接目标不存在：{path.name} -> {target}")
    for path in files:
        path = path.resolve()
        if (path.parent != root and path != legacy) or path.suffix != ".md":
            parser.error("只接受本教材目录及指定 GDN 教材")
        text = path.read_text(encoding="utf-8")
        for index, match in enumerate(re.finditer(r"^```python\n(.*?)^```\s*$", text, re.M | re.S), 1):
            source = match.group(1)
            code = compile(source, f"{path.name}:block-{index}", "exec")
            digest = hashlib.sha256(source.encode()).hexdigest()[:12]
            first_output = None
            for iteration in range(args.repeat):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exec(code, {"__name__": "__main__"})
                current_output = output.getvalue()
                if iteration == 0:
                    first_output = current_output
                elif current_output != first_output:
                    raise AssertionError(f"输出不稳定：{path.name} block={index}")
            count += 1
            print(f"PASS {path.name} block={index} sha256={digest} repeat={args.repeat}")
            if first_output:
                print(first_output.strip())
    if count == 0:
        raise AssertionError("没有执行任何 Python 代码块")
    print(f"EXAMPLES-PASS blocks={count} executions={count * args.repeat}")


if __name__ == "__main__":
    main()
