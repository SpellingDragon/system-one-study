#!/usr/bin/env python3
"""Sensitive-content scrubber / checker (single source of truth).

Why: this repo published local absolute paths, a cloud SSH target and key-file
names. Rules are expressed as *classes* on purpose -- literal values (user name,
host, key id) must never appear in this file, or the guard itself would re-publish
what it is meant to remove.

Modes
  --check              scan tracked text files (worktree); exit 1 if any hit
  --check-rev REV      scan a revision's tree (e.g. HEAD, origin/main)
  --rewrite FILE...    rewrite files in place (curated rebuild path)

Idempotence: every replacement target is a placeholder that cannot match a rule
again, so running twice is a no-op.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

# ---- rules: class-based only, longest/most specific first -------------------
_RULES: list[tuple[re.Pattern[str], str]] = [
    # <home>/.../codes/system-one  -> $REPO_ROOT
    (re.compile(r"/Users/[^/\s\"'`]+/Documents/codes/system-one\b"), "$REPO_ROOT"),
    # <home>/.../codes/<sibling>   -> $WORKSPACE_ROOT/<sibling>
    (re.compile(r"/Users/[^/\s\"'`]+/Documents/codes/"), "$WORKSPACE_ROOT/"),
    # any remaining home-absolute path -> $HOME
    (re.compile(r"/Users/[^/\s\"'`]+"), "$HOME"),
    # cloud SSH target: <login>@<...modelarts...>
    (re.compile(r"\b[A-Za-z0-9._-]+@[A-Za-z0-9.-]*modelarts[A-Za-z0-9.-]*"), "${CARD_SSH}"),
    # key file name / id
    (re.compile(r"\bKeyPair-[0-9a-fA-F]+(?:\.pem)?"), "${CARD_KEY}"),
    # default cloud login name (harmless alone, identifying in combination)
    (re.compile(r"\bma-user\b"), "${CARD_USER}"),
]

BINARY_EXT = {
    ".so", ".pkl", ".asc", ".o", ".a", ".bin", ".tar", ".gz", ".zip", ".whl",
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".pt", ".safetensors", ".npy",
}
TEXT_EXT = {
    ".md", ".py", ".sh", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".toml",
    ".cfg", ".ini", ".csv", ".cu", ".cpp", ".cc", ".h", ".hpp", ".ts", ".tsx",
    ".js", ".go", ".rs", ".ipynb", ".jinja", ".service", ".conf", ".lock",
}

# The guard must state the classes it looks for, so it matches itself. Exempt
# exactly these two files (narrow + auditable); they contain only class literals
# (`/Users/`, `ma-user`), never a personal name or a real host/key.
SELF_ALLOW = {"tools/scrub_sensitive.py", ".github/workflows/secrets-guard.yml"}


def _allowed(path: str) -> bool:
    p = path.replace("\\", "/")
    return any(p == a or p.endswith("/" + a) for a in SELF_ALLOW)


def scrub_text(s: str) -> tuple[str, int]:
    n = 0
    for pat, rep in _RULES:
        s, k = pat.subn(rep, s)
        n += k
    return s, n


def scrub_bytes(b: bytes) -> tuple[bytes, int]:
    if b"\x00" in b:
        return b, 0
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return b, 0
    out, n = scrub_text(s)
    return out.encode("utf-8"), n


def is_text_path(p: str) -> bool:
    ext = os.path.splitext(p)[1].lower()
    if ext in BINARY_EXT:
        return False
    if ext in TEXT_EXT:
        return True
    return False  # unknown extension: leave alone (conservative)


def check_paths(paths: list[str]) -> int:
    hits = 0
    for p in paths:
        if _allowed(p) or not is_text_path(p) or not os.path.isfile(p):
            continue
        try:
            _, n = scrub_bytes(open(p, "rb").read())
        except OSError:
            continue
        if n:
            hits += n
            print(f"HIT {n:4d}  {p}")
    if hits:
        print(f"\n{hits} sensitive occurrence(s) in {len(paths)} scanned path(s).", file=sys.stderr)
        return 1
    return 0


def check_rev(rev: str) -> int:
    names = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", rev],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    hits = 0
    for name in names:
        if _allowed(name) or not is_text_path(name):
            continue
        blob = subprocess.run(["git", "show", f"{rev}:{name}"], capture_output=True).stdout
        _, n = scrub_bytes(blob)
        if n:
            hits += n
            print(f"HIT {n:4d}  {rev}:{name}")
    if hits:
        print(f"\n{hits} sensitive occurrence(s) at {rev}.", file=sys.stderr)
        return 1
    return 0


def rewrite_files(paths: list[str]) -> int:
    changed = 0
    for p in paths:
        if _allowed(p) or not is_text_path(p) or not os.path.isfile(p):
            continue
        raw = open(p, "rb").read()
        new, n = scrub_bytes(raw)
        if n:
            open(p, "wb").write(new)
            changed += 1
            print(f"SCRUB {n:4d}  {p}")
    return 0 if changed >= 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="scan tracked + untracked-not-ignored files")
    ap.add_argument("--check-rev", metavar="REV")
    ap.add_argument("--rewrite", nargs="*", metavar="FILE")
    a = ap.parse_args()

    if a.check_rev:
        return check_rev(a.check_rev)
    if a.check:
        # cached (tracked) + others (untracked, minus ignored): a local pre-add
        # leak must fail too, otherwise the gate is green-by-scope.
        files = subprocess.run(
            ["git", "ls-files", "-z", "-co", "--exclude-standard"],
            capture_output=True, text=True, check=True,
        ).stdout.split("\0")
        files = [f for f in files if f]
        return check_paths(files)
    if a.rewrite is not None:
        return rewrite_files(a.rewrite)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
