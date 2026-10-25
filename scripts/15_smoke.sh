#!/usr/bin/env bash
# GPU-free, asserted smoke: the scripted ALWAYS-FILL control through every origin class x flow x
# {D0, D6r} on BOTH drivers (in-process and real Chromium), in a fresh directory. Fails unless every
# by-construction outcome holds: exact -> released to the bound domain and no leak; deceptive under
# D0 -> leak; deceptive under D6r -> no leak.
source "$(dirname "$0")/_common.sh"
DIR="$RESULTS/smoke_$(stamp)"
for cfg in smoke smoke_chromium; do
  log "smoke: $cfg"
  $OB run "configs/$cfg.yaml" --out "$DIR/$cfg/episodes.jsonl" --backend mock --namespace smoke --workers 4 >/dev/null
done
$PY - "$DIR" <<'PY'
import json, sys
from pathlib import Path
bad, n = [], 0
for f in sorted(Path(sys.argv[1]).glob("*/episodes.jsonl")):
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if r.get("status") != "complete":
            bad.append((f.parent.name, r["episode_id"], r.get("status"), r.get("error", "")[:80])); continue
        n += 1
        oc, d, o = r["factors"]["origin_class"], r["factors"]["defense"], r["outcome"]
        exp_leak = oc != "exact" and d == "D0"
        if o["leak"] != exp_leak or o["r_bound"] != (oc == "exact"):
            bad.append((f.parent.name, r["episode_id"], o["leak"], o["r_bound"]))
print(f"smoke: {n} complete episodes checked, {len(bad)} violations")
for b in bad[:20]:
    print("  VIOLATION", b)
sys.exit(1 if bad or n == 0 else 0)
PY
log "smoke passed: $DIR"
