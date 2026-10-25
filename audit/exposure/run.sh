#!/usr/bin/env bash
# RQ1 exposure audit (GPU-free): what do real agent frameworks' LLM-facing observations reveal about
# the top URL, the A2 sink (frame origin / form action) and the IDN form, on our twins?
#
#   1. setup the isolated venvs if needed (setup_venvs.sh is idempotent)
#   2. start `originblind lab serve --environment $OB_AUDIT_ENV` (main venv) -> <out>/ports.json
#   3. dump each pipeline's first observation of the landing page, per flow x origin class:
#        dump_ours.py  (main venv)        ours_T0/T1/T1u/T2, pw_aria, pw_aria_ai
#        dump_bgym.py  (.venv-bgym)       bgym_axtree, bgym_axtree_1tab, bgym_dom
#        dump_bu.py    (.venv-bu)         bu_default, bu_coi_off
#        dump_mcp.py   (.venv-bu + node)  pw_mcp   (skipped, and reported, if node/@playwright/mcp is missing)
#      -> <out>/raw/<pipeline>/<flow>_<class>.txt (+ .json sidecar; a failed dump leaves
#         <flow>_<class>.error.json and no .txt)
#   4. score (main venv) -> <out>/exposure.{json,csv}, exposure_errors.json, exposure_summary.txt,
#      manifest.json
#   5. stop the lab server (also on error/interrupt)
# <out> = results/<ns>/exposure for the realistic environment (the confirmatory stimuli; the file
#         src/originblind/analysis/audits.py reads), results/<ns>/exposure_<env> for any other.
#
# Usage: audit/exposure/run.sh <namespace> [--flows "login otp checkout"] [--classes "..."] [--skip-setup]
# Env:   OB_AUDIT_ENV  stimulus environment of the twin (run/episode.ENVIRONMENTS): realistic (default)
#                      or lab; recorded in ports.json, every sidecar, every scored row and the manifest
#        OB_AUDIT_TMP  parent of the private work dir (default /tmp). Keep it SHORT: Chromium puts its
#                      SingletonSocket under $TMPDIR and unix socket paths are limited to 108 bytes.
#        OB_AUDIT_OUT  override <out> (e.g. a scratch dir for a trial run)
# Exit:  0 every pipeline dumped every condition and was scored; 1 otherwise (what succeeded is still
#        scored); 2 usage.
set -uo pipefail

NS="${1:?usage: $0 <namespace> [--flows \"...\"] [--classes \"...\"] [--skip-setup]}"; shift
FLOWS="login otp checkout"
CLASSES="exact typosquat combosquat idn subdomain raw_ip a2_iframe a2_action"
SETUP=1
while [ $# -gt 0 ]; do
  case "$1" in
    --flows) FLOWS="$2"; shift 2 ;;
    --classes) CLASSES="$2"; shift 2 ;;
    --skip-setup) SETUP=0; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

AUDIT_ENV="${OB_AUDIT_ENV:-realistic}"
case "$AUDIT_ENV" in lab|realistic) ;; *) echo "OB_AUDIT_ENV must be lab|realistic, got '$AUDIT_ENV'" >&2; exit 2 ;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$REPO/audit/exposure"
OUT="${OB_AUDIT_OUT:-$REPO/results/$NS/exposure$([ "$AUDIT_ENV" = realistic ] || echo "_$AUDIT_ENV")}"
RAW="$OUT/raw"
LOG="$OUT/logs"
PY="${OB_PY:-$REPO/.venv/bin/python}"
mkdir -p "$OUT" "$LOG"

WORK="$(mktemp -d "${OB_AUDIT_TMP:-/tmp}/ob-exposure.XXXXXX")" || { echo "[exposure] cannot create a work dir" >&2; exit 1; }
if [ ${#WORK} -gt 48 ]; then
  echo "[exposure] work dir path too long for Chromium's socket (${#WORK} > 48 chars): $WORK; set OB_AUDIT_TMP" >&2
  rm -rf "$WORK"; exit 1
fi
LAB_PID=""
cleanup() {
  if [ -n "$LAB_PID" ] && kill -0 "$LAB_PID" 2>/dev/null; then
    kill -TERM "$LAB_PID" 2>/dev/null; wait "$LAB_PID" 2>/dev/null
  fi
  rm -rf "$WORK"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
# code identity at the start (audit/ and src/originblind/): the manifest compares it with the end state, so
# a file edited while the audit runs is listed instead of silently hashed in its new version
code_hashes() {
  PYTHONDONTWRITEBYTECODE=1 "$PY" - "$REPO" <<'EOF'
import hashlib, json, sys
from pathlib import Path
repo = Path(sys.argv[1])
files = [*(repo / "audit").rglob("*.py"), *(repo / "audit").rglob("*.sh"), *(repo / "src" / "originblind").rglob("*.py")]
print(json.dumps({str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}))
EOF
}
code_hashes > "$WORK/code_start.json" || { echo "[exposure] cannot hash the code" >&2; exit 1; }

if [ "$SETUP" = 1 ]; then
  echo "[exposure] setup venvs"
  "$HERE/setup_venvs.sh" > "$LOG/setup.log" 2>&1 || { echo "[exposure] setup failed (see $LOG/setup.log)" >&2; exit 1; }
fi

echo "[exposure] lab serve (environment $AUDIT_ENV) -> $OUT"
rm -f "$OUT/ports.json"
# shellcheck disable=SC2086
PYTHONDONTWRITEBYTECODE=1 "$PY" -m originblind.cli lab serve --out "$OUT/ports.json" --pki-dir "$WORK/pki" \
  --environment "$AUDIT_ENV" --flows $FLOWS > "$LOG/lab_serve.log" 2>&1 &
LAB_PID=$!
for _ in $(seq 1 300); do
  [ -f "$OUT/ports.json" ] && break
  kill -0 "$LAB_PID" 2>/dev/null || { echo "[exposure] lab serve died (see $LOG/lab_serve.log)" >&2; exit 1; }
  sleep 0.2
done
[ -f "$OUT/ports.json" ] || { echo "[exposure] lab serve not ready after 60 s" >&2; exit 1; }
cp "$OUT/ports.json" "$LOG/ports.json"
SERVED_ENV="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("environment", "lab"))' "$OUT/ports.json")"
[ "$SERVED_ENV" = "$AUDIT_ENV" ] || { echo "[exposure] lab serve runs environment '$SERVED_ENV', expected '$AUDIT_ENV'" >&2; exit 1; }

rm -rf "$RAW"; mkdir -p "$RAW"
FAILED=()
SKIPPED=()
run_dump() {  # run_dump <name> <cmd...>
  local name="$1"; shift
  echo "[exposure] dump $name"
  # shellcheck disable=SC2086
  if ! env TMPDIR="$WORK" PYTHONDONTWRITEBYTECODE=1 ANONYMIZED_TELEMETRY=false BROWSER_USE_CLOUD_SYNC=false \
       BROWSER_USE_CONFIG_DIR="$WORK/browseruse" "$@" --ports "$OUT/ports.json" --out "$RAW" \
       --flows $FLOWS --classes $CLASSES > "$LOG/dump_$name.log" 2>&1; then
    echo "[exposure]   $name FAILED (see $LOG/dump_$name.log)" >&2
    FAILED+=("$name")
  fi
  grep -h "^\[dump_" "$LOG/dump_$name.log" | tail -n 1
}
run_dump ours "$PY" "$HERE/dump_ours.py"
run_dump bgym env PLAYWRIGHT_BROWSERS_PATH="$REPO/.venv-bgym/ms-playwright" "$REPO/.venv-bgym/bin/python" "$HERE/dump_bgym.py"
run_dump bu "$REPO/.venv-bu/bin/python" "$HERE/dump_bu.py"
if command -v node >/dev/null 2>&1 && [ -f "$REPO/.venv-bu/pwmcp/node_modules/@playwright/mcp/cli.js" ]; then
  run_dump mcp "$REPO/.venv-bu/bin/python" "$HERE/dump_mcp.py"
else
  echo "[exposure] pw_mcp SKIPPED (node or .venv-bu/pwmcp missing)" >&2
  SKIPPED+=("mcp")
fi

kill -TERM "$LAB_PID" 2>/dev/null; wait "$LAB_PID" 2>/dev/null; LAB_PID=""

echo "[exposure] score"
PYTHONDONTWRITEBYTECODE=1 "$PY" "$HERE/score.py" --raw "$RAW" --out "$OUT" > "$LOG/score.log" 2>&1 \
  || FAILED+=("score")
tail -n 1 "$LOG/score.log"
cp "$HERE"/versions_*.txt "$OUT/" 2>/dev/null

# provenance: what ran, on which code and framework versions
code_hashes > "$WORK/code_end.json"
PYTHONDONTWRITEBYTECODE=1 "$PY" - "$OUT" "$NS" "$STARTED" "${FAILED[*]:-}" "${SKIPPED[*]:-}" "$FLOWS" "$CLASSES" \
  "$AUDIT_ENV" "$REPO" "$WORK/code_start.json" "$WORK/code_end.json" "$SETUP" <<'EOF'
import hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

out, ns, started, failed, skipped, flows, classes, environment, repo, code_start, code_end, setup = sys.argv[1:13]
out, repo = Path(out), Path(repo)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
def git(*a):
    try:
        # --no-optional-locks: a read-only query must not refresh (write) the index
        return subprocess.run(["git", "--no-optional-locks", "-C", str(repo), *a], capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ""
rows = json.loads((out / "exposure.json").read_text()) if (out / "exposure.json").exists() else []
# every module of audit/ and src/originblind/ (the stimuli: lab/pages.py, run/episode.py, ...; our arms:
# agent/observe.py) as it was when the run STARTED: git_commit does not pin it while the tree is dirty
start = json.loads(Path(code_start).read_text())
end = json.loads(Path(code_end).read_text())
changed = sorted(k for k in start.keys() | end.keys() if start.get(k) != end.get(k))
if changed:
    print(f"[exposure] WARNING: code changed while the audit ran: {changed}", file=sys.stderr)
manifest = {
    "audit": "exposure (RQ1/H1a)", "namespace": ns, "environment": environment, "started_utc": started,
    "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--", "audit", "src")),
    "flows": flows.split(), "classes": classes.split(),
    "pipelines": sorted({r["pipeline"] for r in rows}), "n_rows": len(rows),
    "row_environments": sorted({r.get("environment", "lab") for r in rows}),
    "failed_stages": failed.split(), "skipped_stages": skipped.split(),
    "venv_setup_ran": setup == "1",  # --skip-setup: versions_*.txt are immutable recorded installation inputs
    "ports": json.loads((out / "logs" / "ports.json").read_text()),
    "sha256": {p.name: sha(p) for p in [out / "exposure.json", out / "exposure.csv", out / "exposure_errors.json",
                                        *sorted(out.glob("versions_*.txt")),
                                        *sorted((repo / "audit" / "exposure").glob("*.py")),
                                        repo / "src" / "originblind" / "lab" / "serve.py"]},
    "code_changed_during_run": changed,
    "code_tree_sha256": hashlib.sha256("".join(f"{k} {v}\n" for k, v in start.items()).encode()).hexdigest(),
    "code_sha256": start,
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
EOF

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "[exposure] done with failures: ${FAILED[*]} -> $OUT" >&2
  exit 1
fi
echo "[exposure] done${SKIPPED:+ (skipped: ${SKIPPED[*]})} -> $OUT/exposure.csv"
