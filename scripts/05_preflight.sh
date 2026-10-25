#!/usr/bin/env bash
# Named execution blockers (config-aware).
# usage: scripts/05_preflight.sh [--configs "..."] [--full] [--smoke-only] [--host-checks]
#                                [--namespace NS] [--results DIR] [--out FILE]
# --full blocks on host hazards (B15-B21; OB_ACCEPT_HOST_RISK=1 accepts them, recorded); --host-checks
# reports them as warnings; --namespace adds the namespace codes (B22-B26: STOP alarm, resume identity,
# store integrity, serving stack). README.md lists every code.
source "$(dirname "$0")/_common.sh"
ARGS=(); OUT="$RESULTS/preflight.json"; CFG=()
while [ $# -gt 0 ]; do
  case "$1" in
    --configs) read -r -a CFG <<<"$2"; shift 2 ;;
    --full|--smoke-only|--host-checks) ARGS+=("$1"); shift ;;
    --namespace|--results) ARGS+=("$1" "$2"); shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) die "unknown argument $1" ;;
  esac
done
mkdir -p "$(dirname "$OUT")"
set +e
$OB preflight --repo-root "$REPO_ROOT" ${CFG:+--configs "${CFG[@]}"} "${ARGS[@]}" > "$OUT"
RC=$?
set -e
$PY - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for k, v in (d.get("blockers") or {}).items():
    if v.get("status") != "ok":
        print(f"  {v.get('status', '?'):7s} {k}: {v.get('detail', '')}")
ov = [k for k, v in (d.get("overrides") or {}).items() if v]
print(f"preflight ({d.get('mode', '?')}): {'READY' if d.get('ready') else 'BLOCKED'} "
      f"({d.get('n_blocked', '?')} blocked, {d.get('n_warn', '?')} warnings)" + (f"; overrides: {ov}" if ov else ""))
PY
exit $RC
