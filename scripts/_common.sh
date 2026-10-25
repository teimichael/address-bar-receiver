# Shared helpers for the origin-blind scripts (sourced, never executed).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
RESULTS="${OB_RESULTS:-$REPO_ROOT/results}"
mkdir -p "$RESULTS"
if [ -n "${OB_PY:-}" ]; then
  PY="$OB_PY"
elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  PY="$REPO_ROOT/.venv/bin/python"
else
  PY="poetry run python"
fi
OB="${OB_CLI:-$PY -m originblind.cli}"  # OB_CLI: test seam (tests/e2e/test_scripts.py stubs serving/runs)
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stamp() { date -u +%Y%m%dT%H%M%SZ; }
log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }
die() { echo "error: $*" >&2; exit 1; }
