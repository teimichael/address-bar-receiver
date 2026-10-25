#!/usr/bin/env bash
# Supervised readiness check before the confirmatory run (README.md). Run it with
# someone at the machine, after the host settings, and start scripts/run_full.sh only after PASS.
#
# usage: scripts/06_ready_check.sh
#
#   1 preflight --full on the confirmatory configs (namespace full): every blocker cleared
#   2 the largest server switch of the schedule, Ministral-3-14B -> Qwen3.5-35B-A3B, through
#     scripts/run_full.sh itself (lock, monitor, pre-serve gate with the Windows probe required,
#     setup, lab environment, weights) in a scratch namespace ready_<utc>, with the real run's
#     concurrency: 12 Chromium workers, then 4 lab workers, beside each server (configs/ops/)
#   3 `originblind ops summarize`: finished cleanly, both gates passed with the Windows probe, no
#     critical sample, Windows commit headroom >= OB_READY_MIN_COMMIT_GB (default 40) throughout
#
# Outcome-blind: only counts and the flight record are read. Takes about 30 minutes.
# Exit: 0 PASS; 1 FAIL (the reason is printed; results/ready_<utc>/ holds the records).
source "$(dirname "$0")/_common.sh"
MIN_COMMIT_GB="${OB_READY_MIN_COMMIT_GB:-40}"
NS="ready_$(stamp)"
CHECK_CONFIGS="configs/ops/ready_check.yaml configs/ops/ready_check_lab.yaml"
# the confirmatory set, as scripts/run_full.sh runs it with no options
CONFIRMATORY="$(sed -n 's/^CONFIGS="\(.*\)"$/\1/p' scripts/run_full.sh)"
[ -n "$CONFIRMATORY" ] || die "cannot read the confirmatory configs from scripts/run_full.sh"
fail() { echo; echo "READY CHECK: FAIL - $*"; echo "   records: $RESULTS/$NS/"; exit 1; }

echo "== 1/3 preflight --full on the confirmatory configs (namespace full)"
mkdir -p "$RESULTS/$NS"
scripts/05_preflight.sh --configs "$CONFIRMATORY" --full --namespace full --out "$RESULTS/$NS/preflight_full.json" \
  || fail "preflight --full is blocked (above; README.md)"

echo "== 2/3 supervised server switch ministral-3-14b -> qwen3.5-35b-a3b (namespace $NS)"
set +e
OB_REQUIRE_HOST_PROBE=1 OB_BU_LLM_MODELS=none scripts/run_full.sh --namespace "$NS" --configs "$CHECK_CONFIGS" \
  --skip-smoke --skip-audits --no-report --skip-timestamps
RC=$?
set -e

echo "== 3/3 flight record"
SRC=0
$OB ops summarize "$RESULTS/$NS" --min-commit-headroom-gb "$MIN_COMMIT_GB" > "$RESULTS/$NS/ready_summary.json" || SRC=$?
if $PY - "$RESULTS/$NS/ready_summary.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for g in d.get("gates") or []:
    print(f"   gate {g['key']}: {g['reason']} (settled {g['settle_waited_s']} s; commit headroom "
          f"{g['commit_headroom_gb']} GB, needed {g['commit_needed_gb']} GB)")
for s in d.get("servers") or []:
    print(f"   server {s['key']}: ready in {s['ready_s']} s; container memory.peak {s['memory_peak_gib']} GiB "
          f"(backstop {s['memory_limit']})")
print(f"   Windows commit headroom: min {d.get('min_commit_headroom_gb')} GB at {d.get('min_commit_headroom_at')} "
      f"(needed >= {d.get('threshold_commit_headroom_gb')} GB) over {d.get('windows_samples')} samples")
print(f"   Windows available min {d.get('min_windows_available_gb')} GB; NVIDIA shared max "
      f"{d.get('max_gpu_shared_mib')} MiB; VM available min {d.get('min_vm_available_gib')} GiB; "
      f"PSI memory full max {d.get('max_psi_memory_full_avg10')}")
for w, n in (d.get("warnings") or {}).items():
    print(f"   warning x{n}: {w}")
for p in d.get("problems") or []:
    print(f"   PROBLEM: {p}")
PY
then :; else SRC=1; echo "   (no readable summary: $RESULTS/$NS/ready_summary.json)"; fi
[ "$RC" = 0 ] || fail "scripts/run_full.sh exited $RC (see $RESULTS/$NS/RUN_STATUS.json)"
[ "$SRC" = 0 ] || fail "the flight record does not show a safe switch (problems above)"
echo
echo "READY CHECK: PASS ($NS)"
echo "   next, in tmux inside a Windows Terminal WSL tab: scripts/run_full.sh"
