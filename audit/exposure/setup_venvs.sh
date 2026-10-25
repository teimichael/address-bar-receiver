#!/usr/bin/env bash
# Install recorded framework versions without resolving a new dependency closure.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$REPO/audit/exposure"
PY="${OB_AUDIT_PYTHON:-python3.12}"
UV="${UV:-$(command -v uv || true)}"
FORCE=0
WHICH=()
for a in "$@"; do
  case "$a" in
    --force) FORCE=1 ;;
    bgym|bu|mcp) WHICH+=("$a") ;;
    *) echo "usage: $0 [--force] [bgym|bu|mcp ...]" >&2; exit 2 ;;
  esac
done
[ ${#WHICH[@]} -gt 0 ] || WHICH=(bgym bu mcp)
[ -n "$UV" ] || { echo "uv is required (set UV=/path/to/uv)" >&2; exit 1; }
command -v "$PY" >/dev/null || { echo "Python 3.12 is required (set OB_AUDIT_PYTHON)" >&2; exit 1; }

install_recorded() {
  local tag="$1" envdir="$REPO/.venv-$1"
  if [ "$FORCE" = 1 ] && [ -d "$envdir" ]; then rm -rf "$envdir"; fi
  if [ ! -x "$envdir/bin/python" ]; then "$UV" venv --python "$PY" "$envdir"; fi
  "$UV" pip install --python "$envdir/bin/python" --no-deps -r "$HERE/versions_$tag.txt"
  "$envdir/bin/python" - "$HERE/versions_$tag.txt" <<'PY'
import importlib.metadata as metadata
import sys

for line in open(sys.argv[1]):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    name, version = line.split("==", 1)
    actual = metadata.version(name)
    if actual != version:
        raise SystemExit(f"{name}: expected {version}, installed {actual}")
print("Recorded framework versions verified")
PY
}

for w in "${WHICH[@]}"; do
  case "$w" in
    bgym)
      install_recorded bgym
      PLAYWRIGHT_BROWSERS_PATH="$REPO/.venv-bgym/ms-playwright" \
        "$REPO/.venv-bgym/bin/python" -m playwright install chromium
      ;;
    bu) install_recorded bu ;;
    mcp)
      command -v npm >/dev/null || { echo "npm is required for the MCP exposure audit" >&2; exit 1; }
      [ -x "$REPO/.venv-bu/bin/python" ] || install_recorded bu
      mkdir -p "$REPO/.venv-bu/pwmcp"
      cp "$HERE/npm/package.json" "$HERE/npm/package-lock.json" "$REPO/.venv-bu/pwmcp/"
      npm ci --prefix "$REPO/.venv-bu/pwmcp" --no-audit --no-fund
      ;;
  esac
done
