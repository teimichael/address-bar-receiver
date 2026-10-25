#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
if [ -n "${OB_PY:-}" ]; then
  exec "$OB_PY" tools/reproduce.py "$@"
elif [ -x .venv/bin/python ]; then
  exec .venv/bin/python tools/reproduce.py "$@"
elif command -v poetry >/dev/null 2>&1; then
  exec poetry run python tools/reproduce.py "$@"
else
  exec python3 tools/reproduce.py "$@"
fi
