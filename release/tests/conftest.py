"""tests/conftest.py — 把 release 根进 sys.path，并放公共的假分词器（接缝破坏用）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
