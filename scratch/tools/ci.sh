#!/usr/bin/env bash
# 课程 CI 唯一入口 —— 本地 / pre-commit / GitHub Actions 跑同一套门（标准单一真源）
#
# 用法:
#   bash tools/ci.sh                # CPU 全门: ruff + 注释 + compile + pytest(非设备用例)
#   bash tools/ci.sh --fast         # 轻量: 注释 + compile（pre-commit hook 档，秒级）
#   bash tools/ci.sh --device cuda  # 追加设备用例（cuda|mps|npu，需装依赖 pip install -e '.[dev]'）
#
# 环境变量: GATE_STRICT=1 时缺 ruff/pytest 直接判负（CI 用）；本地缺依赖仅告警。
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
MODE="${1:-cpu}"
DEV="${2:-}"
fail=0

run() {  # run <描述> <命令...>
  local desc="$1"; shift
  if "$@"; then printf '   ✓ %s\n' "$desc"
  else printf '   ✗ %s\n' "$desc"; fail=1; fi
}

if command -v ruff >/dev/null 2>&1; then RUFF=(ruff)
elif $PY -m ruff --version >/dev/null 2>&1; then RUFF=("$PY" -m ruff)
else RUFF=(); fi

if [ "${#RUFF[@]}" -gt 0 ]; then
  PYFILES=()
  while IFS= read -r f; do PYFILES+=("$f"); done < <(git ls-files '*.py' 2>/dev/null || find sys1 learning production serving tools tests -name '*.py' 2>/dev/null)
  if [ "${#PYFILES[@]}" -gt 0 ]; then run "ruff check" "${RUFF[@]}" check "${PYFILES[@]}"; else echo "== ruff: 无 py 文件"; fi
else
  echo "⚠️  ruff 未安装（pip install -e '.[dev]'）"; [ "${GATE_STRICT:-0}" = 1 ] && fail=1
fi

printf '\n== 注释质量门 (GUIDE §6.1)\n'
run "check_comments" $PY tools/check_comments.py

printf '\n== 编译检查\n'
run "compileall" $PY -m compileall -q sys1 learning production serving tools tests

if [ "$MODE" = "--fast" ]; then
  [ $fail -eq 0 ] && echo && echo "✅ fast gates 通过" || echo "❌ 有门禁未过"
  exit $fail
fi

printf '\n== pytest（CPU 门）\n'
if $PY -m pytest --version >/dev/null 2>&1; then
  run "pytest cpu" $PY -m pytest -q -m "not cuda and not mps and not npu" tests
else
  echo "⚠️  pytest 未安装（pip install -e '.[dev]'）"; [ "${GATE_STRICT:-0}" = 1 ] && fail=1
fi

if [ "$MODE" = "--device" ]; then
  printf '\n== pytest（设备: %s）\n' "$DEV"
  run "pytest -m $DEV" $PY -m pytest -q -m "$DEV" tests
fi

echo
[ $fail -eq 0 ] && echo "✅ CI 全绿" || echo "❌ 有门禁未过（见上）"
exit $fail
