#!/usr/bin/env bash
# The single unattended entrypoint of the evaluation (README.md).
#
# usage: scripts/run_full.sh            # the confirmatory run (and its resume): no options needed
#        scripts/run_full.sh [--namespace NS] [--configs "c1.yaml c2.yaml ..."] [--models "k1 k2"]
#                            [--workers N] [--dry-run] [--allow-dirty] [--skip-setup] [--skip-fetch]
#                            [--skip-smoke] [--skip-audits] [--no-report] [--skip-timestamps]
# Defaults: namespace `full`, all seven experiment configurations and all rostered models.
# Native browser-use reproduction and proof upgrades are excluded from this public release.
#
# Steps (each recorded in results/<NS>/logs/run_full_<utc>.steps.jsonl and results/<NS>/RUN_STATUS.json,
# which is rewritten after every step; all output tee'd to results/<NS>/logs/):
#   0 host check     the host-safety subset of preflight and a pending STOP alarm, before anything else
#   1 setup          idempotent env + images pinned by digest (+ the lab image if a config needs it)
#     lab env        one lab environment per run: the lab image's packages are recorded on the first
#                    attempt and must match on every later one; then the lab image is frozen
#                    (OB_LAB_NO_BUILD=1: a mid-run source edit fails the lab instead of rebuilding it)
#   2 store check    integrity of the namespace's stores (a torn store needs a manual repair)
#   3 schedule, models (verified fetch + materialization into ob-models)
#   4 sweep          leftover containers of a crashed attempt (their logs are saved first)
#   5 preflight      config-aware blockers (--full for NS=full: clean tree, pinned weights, host safety)
#   6 smoke          GPU-free asserted controls on both drivers
#   7 model-major    serve -> identity -> host and screenshot configurations -> stop
#   8 audits         optional framework exposure audit
#   9 validate       per experiment (incomplete runs are recorded, never hidden)
#  10 report         tables, pre-registered claims, figures, SUMMARY.md -> results/<NS>/report/; for NS=full
#                    only when every confirmatory experiment is complete and no step failed (no interim looks)
#  11 timestamps     omitted from this public release
# Safety (README.md): one run at a time (flock on results/.ops/run_full.lock); a flight
# recorder (`originblind ops monitor`) writes logs/monitor_<utc>.jsonl and raises a STOP alarm on
# critical host/VM memory or any OOM kill; every child runs in the background so INT/TERM/HUP and the
# alarm are handled within OB_POLL_S; a stopped run stops its child gracefully and its vLLM server.
# Resumable: completed episodes are skipped; groups with nothing left to run never start a server.
# Exit: 0 everything complete; 2 some planned episodes still missing; 1 a pipeline step failed;
#       6 aborted by the resource guard or the pre-serve gate (fix the cause, move results/<NS>/STOP
#       aside, rerun); 128+n interrupted by signal n (rerun to resume).
source "$(dirname "$0")/_common.sh"

NS="full"
CONFIGS="configs/e1_default.yaml configs/e2_information.yaml configs/e2s_screenshot.yaml configs/e3_enforcement.yaml configs/e3_utility.yaml configs/e5_parity.yaml configs/e6_thinking.yaml"
DEFAULT_CONFIGS="$CONFIGS"   # the confirmatory set: validation of `full` always covers all of it
MODELS=""; WORKERS="${OB_WORKERS:-6}"; DRY=0; ALLOW_DIRTY=0; PARTIAL=0
SKIP_SETUP=0; SKIP_FETCH=0; SKIP_SMOKE=0; SKIP_AUDITS=0; NO_REPORT=0
AUDIT_DIR="${OB_AUDIT_DIR:-audit}"     # test seam (tests/e2e/test_scripts.py stubs the audits)
LAB_RUN="${OB_LAB_RUN:-scripts/lab_run.sh}"  # test seam
SETUP="${OB_SETUP:-scripts/00_setup.sh}"     # test seam
# OB_LAB_WORKERS: concurrent screenshot-arm episodes in the lab (default 4; one private display each)
# OB_MAX_WORKERS: optional cap on a pair's concurrency (default: the roster's `workers:`, throughput only)

while [ $# -gt 0 ]; do
  case "$1" in
    --namespace) NS="$2"; shift 2 ;; --namespace=*) NS="${1#*=}"; shift ;;
    --configs) CONFIGS="$2"; PARTIAL=1; shift 2 ;; --configs=*) CONFIGS="${1#*=}"; PARTIAL=1; shift ;;
    --models) MODELS="$2"; PARTIAL=1; shift 2 ;; --models=*) MODELS="${1#*=}"; PARTIAL=1; shift ;;
    --workers) WORKERS="$2"; shift 2 ;; --workers=*) WORKERS="${1#*=}"; shift ;;
    --dry-run) DRY=1; shift ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    --skip-setup) SKIP_SETUP=1; shift ;;
    --skip-fetch) SKIP_FETCH=1; shift ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    --skip-audits) SKIP_AUDITS=1; shift ;;
    --no-report) NO_REPORT=1; shift ;;
    --skip-timestamps) shift ;;
    -h|--help) sed -n '2,/^source /{/^source /!p}' "$0"; exit 0 ;;
    *) die "unknown argument $1" ;;
  esac
done
case "$NS" in ""|*/*|.*) die "invalid namespace '$NS'" ;; esac
# numeric knobs fail here, not as argparse's exit 2 (which the loop would read as "incomplete")
uint() { case "$2" in ''|*[!0-9]*|0) die "$1 must be a positive integer, not '$2'" ;; esac; }
uint "--workers/OB_WORKERS" "$WORKERS"
LAB_WORKERS="${OB_LAB_WORKERS:-4}"; uint OB_LAB_WORKERS "$LAB_WORKERS"
MAX_WORKERS="${OB_MAX_WORKERS:-}"; [ -z "$MAX_WORKERS" ] || uint OB_MAX_WORKERS "$MAX_WORKERS"
# the confirmatory run never starts a server the gate could not check against the Windows commit
# headroom (ops/gate.py); exported before the budget is read, so RUN_STATUS records it
if [ "$NS" = "full" ]; then export OB_REQUIRE_HOST_PROBE="${OB_REQUIRE_HOST_PROBE:-1}"; fi
BUDGET_SH="$($OB ops budget --shell)" || die "invalid resource budget (an OB_* variable; README.md)"
eval "$BUDGET_SH"
[ -n "${BUDGET_POLL_S:-}" ] || die "could not read the resource budget (originblind ops budget --shell)"
BUDGET_JSON="$($OB ops budget)"
POLL="${BUDGET_POLL_S%.*}"; [ "${POLL:-0}" -ge 1 ] 2>/dev/null || POLL=1
if [ "$NS" = "full" ] && [ "$PARTIAL" = "1" ] && [ "${OB_ALLOW_PARTIAL_FULL:-0}" != "1" ]; then
  die "the confirmatory namespace runs every confirmatory config and model; --configs/--models on 'full' need OB_ALLOW_PARTIAL_FULL=1 (recorded)"
fi
read -r -a CFG <<<"$CONFIGS"
for c in "${CFG[@]}"; do [ -f "$c" ] || die "config not found: $c"; done
MODEL_ARGS=(); [ -n "$MODELS" ] && { read -r -a _m <<<"$MODELS"; MODEL_ARGS=(--models "${_m[@]}"); }

RUNDIR="$RESULTS/$NS"; LOGDIR="$RUNDIR/logs"
RUNSTAMP="$(stamp)"
SCHED_JSON="$LOGDIR/schedule_$RUNSTAMP.json"; SCHED_TSV="$LOGDIR/schedule_$RUNSTAMP.tsv"

# ------------------------------------------------------------------ dry run (no Docker, no GPU)
if [ "$DRY" = "1" ]; then
  mkdir -p "$LOGDIR"
  echo "== dry run: namespace $NS (nothing is executed)"
  $OB serve schedule --configs "${CFG[@]}" "${MODEL_ARGS[@]}" --namespace "$NS" --results "$RESULTS" --format text
  echo "== per config (originblind plan --cost)"
  for c in "${CFG[@]}"; do
    $OB plan "$c" --cost | $PY -c "import json,sys; d=json.load(sys.stdin); print(f\"  {d['name']:<22} n={d['n_episodes']:>6,}  driver={d['driver']:<16} GPU-h~{d.get('projected_gpu_hours_total')}  by_model={d['by_model']}\")"
  done
  echo "== native browser-use reproduction excluded from public release"
  echo "== then: optional exposure audit, validate, report"
  echo "== pre-serve gate per server start: settle ${BUDGET_SETTLE_S} s after the previous server stopped"
  exit 0
fi

# ------------------------------------------------------------------ one run at a time
export OB_OPS_DIR="${OB_OPS_DIR:-$RESULTS/.ops}"
mkdir -p "$OB_OPS_DIR" "$LOGDIR"
export OB_RUN_LOCK="${OB_RUN_LOCK:-$OB_OPS_DIR/run_full.lock}"
exec {LOCK_FD}>>"$OB_RUN_LOCK"
if ! flock -n "$LOCK_FD"; then
  die "another scripts/run_full.sh is active (lock $OB_RUN_LOCK; find it with: fuser -v $OB_RUN_LOCK)"
fi
export OB_RUN_LOCK_HELD=1   # our children may serve; a manual `serve start` elsewhere is refused
if [ -f "$RUNDIR/STOP" ]; then
  die "a STOP alarm is pending: $(cat "$RUNDIR/STOP" | $PY -c 'import json,sys; print(json.load(sys.stdin).get("reason",""))' 2>/dev/null). Fix the cause (README.md), then: mv $RUNDIR/STOP $LOGDIR/STOP_$(stamp).json"
fi
export OB_RESULTS_DIR="$RESULTS"   # lab_run.sh exports into the same results root

LOG="$LOGDIR/run_full_$RUNSTAMP.log"
# tee ignores the signals meant for us (the cleanup output must still reach the log) and does not hold the lock
exec > >(trap '' INT TERM HUP PIPE; exec tee -p -a "$LOG" {LOCK_FD}>&-) 2>&1
STEPS="$LOGDIR/run_full_$RUNSTAMP.steps.jsonl"; : > "$STEPS"
MON_OUT="$LOGDIR/monitor_$RUNSTAMP.jsonl"
RUN_ID="$NS-$RUNSTAMP-$$"
STATE="running"; CURRENT=""; FINISHED=0; FINISHING=0
CHILD=""; CHILD_KIND=""; MON_PID=""; MON_RESTARTS=0
OVERRIDES="$( { [ "$ALLOW_DIRTY" = 1 ] && echo -n "allow_dirty "; [ "${OB_ACCEPT_HOST_RISK:-0}" = 1 ] && echo -n "OB_ACCEPT_HOST_RISK "; [ "$PARTIAL" = 1 ] && echo -n "partial_selection "; [ -n "$MAX_WORKERS" ] && echo -n "OB_MAX_WORKERS=$MAX_WORKERS "; } || true)"

status_json() {  # status_json [exit_code]: rewrite RUN_STATUS.json atomically (state, current step, steps)
  RS_STATE="$STATE" RS_CURRENT="$CURRENT" RS_EXIT="${1:-}" RS_OVERRIDES="$OVERRIDES" RS_BUDGET="$BUDGET_JSON" \
  $PY - "$STEPS" "$RUNDIR/RUN_STATUS.json" "$NS" "$RUNSTAMP" "$LOG" "$MON_OUT" "$OB_RUN_LOCK" "$$" <<'PY'
import json, os, sys
steps_p, out, ns, stamp, log, mon, lock, pid = sys.argv[1:9]
steps = [json.loads(l) for l in open(steps_p) if l.strip()]
e = os.environ
doc = {"namespace": ns, "run": stamp, "state": e.get("RS_STATE"), "exit_code": int(e["RS_EXIT"]) if e.get("RS_EXIT") else None,
       "pid": int(pid), "current_step": e.get("RS_CURRENT") or None, "log": log, "monitor": mon, "lock": lock,
       "overrides": e.get("RS_OVERRIDES", "").split(),
       "budget": json.loads(e.get("RS_BUDGET") or "{}"),
       "resume": "scripts/run_full.sh" + ("" if ns == "full" else f" --namespace {ns}"), "steps": steps}
tmp = out + ".tmp"
with open(tmp, "w") as fh:
    json.dump(doc, fh, indent=2)
    fh.flush(); os.fsync(fh.fileno())
os.replace(tmp, out)
PY
}
record() {  # record <step> <status> <rc> [detail]  (steps.jsonl line format unchanged; fsynced)
  $PY - "$STEPS" "$1" "$2" "$3" "${4:-}" "$(utc)" <<'PY'
import json, os, sys
p, step, status, rc, detail, t = sys.argv[1:7]
with open(p, "a") as fh:
    fh.write(json.dumps({"step": step, "status": status, "rc": int(rc), "detail": detail, "utc": t}) + "\n")
    fh.flush(); os.fsync(fh.fileno())
PY
  status_json || true
}
stop_monitor() {
  [ -n "$MON_PID" ] || return 0
  kill -TERM "$MON_PID" 2>/dev/null; wait "$MON_PID" 2>/dev/null; MON_PID=""
}
finish() {
  local rc="$1"
  set +e
  trap '' USR1   # a late monitor alarm must not restart the cleanup
  [ "$FINISHED" = "1" ] && exit "$rc"
  FINISHED=1
  stop_monitor
  if [ "$STATE" = "running" ]; then
    case "$rc" in 0) STATE=finished ;; 2) STATE=incomplete ;; 6) STATE=aborted_resource ;; *) STATE=failed ;; esac
  fi
  CURRENT=""
  $OB ops probe --host >"$LOGDIR/host_finish_$RUNSTAMP.json" 2>/dev/null </dev/null {LOCK_FD}>&-
  record finish "$STATE" "$rc" "exit $rc"   # an attempt without this record crashed
  status_json "$rc"
  $PY - "$STEPS" "$RUNDIR/RUN_STATUS.json" "$NS" "$rc" <<'PY'
import json, sys
steps_p, out, ns, rc = sys.argv[1:5]
steps = [json.loads(l) for l in open(steps_p) if l.strip()]
failed = [s for s in steps if s["status"] not in ("ok", "skipped") and s["step"] != "finish"]
print(f"\n== run_full {ns} finished (exit {rc}); {len(steps) - 1} steps, {len(failed)} not ok")
for s in failed:
    print(f"   {s['status']:10s} {s['step']} rc={s['rc']} {s['detail']}")
print(f"   status: {out}")
PY
  exit "$rc"
}
on_exit() {  # an unexpected exit (set -e) still leaves an accurate status
  local rc=$?
  [ "$FINISHED" = "1" ] && return
  STATE=failed; finish "$rc"
}

# ------------------------------------------------------------------ children, kill order, signals
child_alive() { [ -n "$CHILD" ] && kill -0 "$CHILD" 2>/dev/null; }
wait_gone() {  # wait_gone <pid> <seconds>
  local i=0
  while kill -0 "$1" 2>/dev/null && [ "$i" -lt "$2" ]; do sleep 1; i=$((i + 1)); done
}
group_alive() { kill -0 -- "-$1" 2>/dev/null; }
marker_pids() {  # processes carrying this run's OB_RUN_ID (children and their detached descendants)
  local p pid
  for p in /proc/[0-9]*; do
    pid="${p#/proc/}"
    [ "$pid" = "$$" ] && continue
    { tr '\0' '\n' <"$p/environ"; } 2>/dev/null | grep -qx "OB_RUN_ID=$RUN_ID" && echo "$pid"
  done
  return 0
}
sweep_marker() {  # processes of the child tree that left its process group (e.g. detached Chromium)
  local p pid
  [ -n "$(marker_pids)" ] || return 0
  log "sweeping leftover processes of this run: $(marker_pids | tr '\n' ' ')"
  for sig in TERM KILL; do
    for p in /proc/[0-9]*; do
      pid="${p#/proc/}"
      [ "$pid" = "$$" ] && continue
      { tr '\0' '\n' <"$p/environ"; } 2>/dev/null | grep -qx "OB_RUN_ID=$RUN_ID" || continue
      kill -"$sig" "$pid" 2>/dev/null
    done
    [ "$sig" = TERM ] && sleep 3
  done
}
stop_child() {  # graceful kill order per kind (README.md)
  [ -n "$CHILD" ] || return 0
  if child_alive; then
    log "stopping the running ${CHILD_KIND} child (pid $CHILD)"
    case "$CHILD_KIND" in
      host)  # the orchestrator is the only record writer: let it go first, then its workers
        kill -TERM "$CHILD" 2>/dev/null; wait_gone "$CHILD" 30
        kill -TERM -- "-$CHILD" 2>/dev/null; local i=0
        while group_alive "$CHILD" && [ "$i" -lt 15 ]; do sleep 1; i=$((i + 1)); done
        kill -KILL -- "-$CHILD" 2>/dev/null ;;
      lab)   # lab_run.sh stops its container and exports the volume copy
        kill -TERM "$CHILD" 2>/dev/null; wait_gone "$CHILD" 240
        kill -TERM -- "-$CHILD" 2>/dev/null; wait_gone "$CHILD" 10; kill -KILL -- "-$CHILD" 2>/dev/null ;;
      *)
        kill -TERM -- "-$CHILD" 2>/dev/null || kill -TERM "$CHILD" 2>/dev/null; wait_gone "$CHILD" 60
        kill -KILL -- "-$CHILD" 2>/dev/null ;;
    esac
  fi
  sweep_marker
  wait "$CHILD" 2>/dev/null
  CHILD=""; CHILD_KIND=""
}
stop_server_after_signal() {  # a killed `serve start` may still have a `docker run -d` in flight
  local i
  for i in 1 2; do
    $OB serve stop --logdir "$LOGDIR" </dev/null >/dev/null 2>&1 {LOCK_FD}>&-
    sleep "$POLL"
  done
}
on_signal() {  # on_signal <SIG> <exit code>
  local sig="$1" code="$2" why=""
  set +e
  if [ "$FINISHING" = "1" ]; then echo "== second $sig: skipping the rest of the cleanup"; exit "$code"; fi
  FINISHING=1
  if [ "$sig" = "USR1" ] || [ -f "$RUNDIR/STOP" ]; then
    STATE=aborted_resource; code=6
    why="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1])).get("reason",""))' "$RUNDIR/STOP" 2>/dev/null)"
  else
    STATE=interrupted
  fi
  log "== $sig: stopping (step ${CURRENT:-none}) ${why:+- $why}"
  stop_child
  stop_server_after_signal
  record "signal:$sig" "$( [ "$code" = 6 ] && echo aborted || echo interrupted )" "$code" "${why:-received $sig during ${CURRENT:-setup}}"
  finish "$code"
}
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap 'on_signal HUP 129' HUP
trap 'on_signal USR1 6' USR1
trap on_exit EXIT

dir_size() {  # total bytes of an experiment directory's stores (progress = growth)
  local f n=0 s
  for f in "$1"/episodes.jsonl "$1"/probes.jsonl "$1"/transcripts.jsonl.gz; do
    s=$(stat -c %s "$f" 2>/dev/null) && n=$((n + s))
  done
  echo "$n"
}
start_monitor() {
  $OB ops monitor --rundir "$RUNDIR" --pid "$$" --out "$MON_OUT" </dev/null >>"$LOGDIR/monitor_$RUNSTAMP.out" 2>&1 {LOCK_FD}>&- &
  MON_PID=$!
}
# run_child <kind host|lab|other> [PROGRESS_DIR=dir] <cmd...>: run in the background in its own session,
# poll every OB_POLL_S: STOP alarm, monitor alive, periodic fsync/sync, stall (host), wall limit (lab).
# Sets RC (124 = stopped by the stall or wall-limit watchdog).
run_child() {
  local kind="$1"; shift
  local pdir=""; case "${1:-}" in PROGRESS_DIR=*) pdir="${1#PROGRESS_DIR=}"; shift ;; esac
  set +e
  OB_RUN_ID="$RUN_ID" setsid "$@" </dev/null &
  CHILD=$!; CHILD_KIND="$kind"
  local t0=$SECONDS last_sync=$SECONDS last_change=$SECONDS size="" warned=0 sp fin st
  [ -n "$pdir" ] && size="$(dir_size "$pdir")"
  RC=""
  while :; do
    sleep "$POLL" & sp=$!
    fin=""; wait -n -p fin "$CHILD" "$sp"; st=$?
    if [ "$fin" = "$CHILD" ]; then RC=$st; kill "$sp" 2>/dev/null; wait "$sp" 2>/dev/null; break; fi
    [ -f "$RUNDIR/STOP" ] && on_signal USR1 6
    if [ -n "$MON_PID" ] && ! kill -0 "$MON_PID" 2>/dev/null; then
      if [ "$MON_RESTARTS" -lt 1 ]; then
        MON_RESTARTS=$((MON_RESTARTS + 1)); log "the monitor died; restarting it once"; start_monitor
      else
        log "WARNING: the monitor died again; continuing without the flight recorder"; MON_PID=""
        record monitor failed 1 "the monitor died twice"
      fi
    fi
    if [ -n "$pdir" ]; then
      local now_size; now_size="$(dir_size "$pdir")"
      [ "$now_size" != "$size" ] && { size="$now_size"; last_change=$SECONDS; warned=0; }
      sync "$pdir"/episodes.jsonl "$pdir"/probes.jsonl "$pdir"/transcripts.jsonl.gz 2>/dev/null
      local idle=$((SECONDS - last_change))
      if [ "$warned" = 0 ] && [ "$idle" -ge "${BUDGET_STALL_WARN_S%.*}" ]; then
        warned=1; log "WARNING: no progress in $pdir for ${idle} s"; record "stall_warning:$CURRENT" warn 0 "no progress for ${idle} s"
      fi
      if [ "${BUDGET_STALL_TIMEOUT_S%.*}" -gt 0 ] && [ "$idle" -ge "${BUDGET_STALL_TIMEOUT_S%.*}" ]; then
        log "no progress in $pdir for ${idle} s: stopping the pair (it is retried once)"
        stop_child; RC=124; break
      fi
    fi
    if [ "$kind" = "lab" ] && [ $((SECONDS - t0)) -ge "${BUDGET_LAB_MAX_RUN_S%.*}" ]; then
      log "the lab run exceeded ${BUDGET_LAB_MAX_RUN_S} s: stopping it"; stop_child; RC=124; break
    fi
    if [ $((SECONDS - last_sync)) -ge 60 ]; then sync; last_sync=$SECONDS; fi   # reaches the Docker volume too
  done
  [ -n "$CHILD" ] && wait "$CHILD" 2>/dev/null
  CHILD=""; CHILD_KIND=""
  [ "$kind" = "host" ] && sweep_marker   # an orphaned worker or Chromium must not outlive its pair
  set -e
  return 0
}
step() { CURRENT="$1"; status_json || true; }  # step <name>: shown as current_step in RUN_STATUS

log "run_full: namespace=$NS configs=${CFG[*]} models=${MODELS:-all} workers=$WORKERS lab_workers=$LAB_WORKERS max_workers=${MAX_WORKERS:-roster} overrides=${OVERRIDES:-none}"
status_json
start_monitor
$OB ops probe --host >"$LOGDIR/host_start_$RUNSTAMP.json" 2>/dev/null </dev/null {LOCK_FD}>&- || true
OOM_BASE="$($OB ops oom 2>/dev/null </dev/null || echo -1)"
GIT_DIRTY="$( [ -n "$(git status --porcelain 2>/dev/null)" ] && echo 1 || echo 0 )"
if [ "$NS" = "full" ] && [ "$GIT_DIRTY" = "1" ] && [ "$ALLOW_DIRTY" != "1" ]; then
  record "provenance" "failed" 1 "dirty git tree: commit before the confirmatory run (or --allow-dirty)"
  finish 1
fi
[ "$ALLOW_DIRTY" = "1" ] && export ORIGIN_BLIND_ALLOW_DIRTY=1
FULLFLAG=(); [ "$NS" = "full" ] && FULLFLAG=(--full)

# ------------------------------------------------------------------ 0 host check (fail fast, before any work)
step hostcheck
run_child other $OB ops hostcheck --namespace "$NS" --results "$RESULTS" "${FULLFLAG[@]}" --configs "${CFG[@]}"
if [ "$RC" = 0 ]; then record hostcheck ok 0; else
  record hostcheck failed "$RC" "host not safe for an unattended run (see the log; README.md; OB_ACCEPT_HOST_RISK=1 to accept, recorded)"; finish 1
fi

# ------------------------------------------------------------------ 1 setup
step setup
if [ "$SKIP_SETUP" = "1" ]; then record setup skipped 0; else
  run_child other "$SETUP" --configs "${CFG[*]}"
  if [ "$RC" = 0 ]; then record setup ok 0; else record setup failed "$RC"; finish 1; fi
fi

# ------------------------------------------------------------------ lab environment (one per run)
NEED_LAB=0
for c in "${CFG[@]}"; do grep -q "^driver: chromium-headful" "$c" && NEED_LAB=1 || true; done
if [ "$NEED_LAB" = "1" ]; then
  step lab_env
  run_child other $OB ops lab-env --image "${OB_LAB_IMAGE:-originblind-lab:latest}" \
    --record "$RUNDIR/lab_env.json" --out "$LOGDIR/lab_env_$RUNSTAMP.json"
  if [ "$RC" = 0 ]; then record lab_env ok 0 "$RUNDIR/lab_env.json"; else
    record lab_env failed "$RC" "the lab image is missing, or its system packages differ from this namespace's first attempt (see the log; README.md)"
    finish 1
  fi
  export OB_LAB_NO_BUILD=1   # scripts/lab_run.sh now refuses a missing or stale image instead of rebuilding
fi

# ------------------------------------------------------------------ 2 store integrity (before anything reads the stores)
step store_check
if [ -d "$RUNDIR" ]; then
  run_child other $OB ops store check "$RUNDIR"
  case "$RC" in
    0) record store_check ok 0 ;;
    2) record store_check failed 2 "a store needs a manual repair: originblind ops store check $RUNDIR; then ops store repair <dir> (README.md)"; finish 1 ;;
    *) record store_check failed "$RC"; finish 1 ;;
  esac
fi

# ------------------------------------------------------------------ schedule (also yields the weights keys)
step schedule
run_child other $OB serve schedule --configs "${CFG[@]}" "${MODEL_ARGS[@]}" --namespace "$NS" --results "$RESULTS" \
     --format text --json-out "$SCHED_JSON" --tsv-out "$SCHED_TSV"
if [ "$RC" != 0 ]; then record schedule failed 5 "invalid configuration"; finish 1; fi
record schedule ok 0 "$SCHED_TSV"
WEIGHTS_KEYS="$($PY -c "import json,sys; print(' '.join(g['weights_key'] for g in json.load(open(sys.argv[1]))['groups']))" "$SCHED_JSON")"

# ------------------------------------------------------------------ models
step models
if [ "$SKIP_FETCH" = "1" ] || [ -z "$WEIGHTS_KEYS" ]; then record models skipped 0 "$WEIGHTS_KEYS"; else
  # shellcheck disable=SC2086
  run_child other $OB models fetch --configs configs/roster.yaml --only $WEIGHTS_KEYS --report "$LOGDIR/models_fetch_$RUNSTAMP.json"
  if [ "$RC" = 0 ]; then
    # shellcheck disable=SC2086
    run_child other $OB models materialize --configs configs/roster.yaml --only $WEIGHTS_KEYS --report "$LOGDIR/models_materialize_$RUNSTAMP.json"
  fi
  if [ "$RC" = 0 ]; then record models ok 0 "$WEIGHTS_KEYS"; else
    record models failed "$RC" "fetch/materialize failed (see log); groups without ready weights will fail to serve"
  fi
fi

# ------------------------------------------------------------------ sweep leftovers of a crashed attempt
step sweep
run_child other $OB ops sweep --ns "$NS" --logdir "$LOGDIR"
if [ "$RC" = 0 ]; then record sweep ok 0; else record sweep failed "$RC" "leftover containers need attention (see the log)"; finish 1; fi

# ------------------------------------------------------------------ preflight
step preflight
PF=(--configs "${CFG[*]}" --out "$RUNDIR/preflight_$RUNSTAMP.json" --namespace "$NS" --results "$RESULTS" --host-checks)
[ "$NS" = "full" ] && PF+=(--full)
run_child other scripts/05_preflight.sh "${PF[@]}"
if [ "$RC" = 0 ]; then record preflight ok 0; else record preflight failed "$RC" "see $RUNDIR/preflight_$RUNSTAMP.json"; finish 1; fi

# ------------------------------------------------------------------ smoke
step smoke
if [ "$SKIP_SMOKE" = "1" ]; then record smoke skipped 0; else
  run_child other scripts/15_smoke.sh
  if [ "$RC" = 0 ]; then record smoke ok 0; else record smoke failed "$RC" "the instrument failed its by-construction controls"; finish 1; fi
fi

# ------------------------------------------------------------------ model-major loop
serve_up() {  # serve_up <weights_key> -> RC (6 = the pre-serve gate refused; nothing started)
  run_child other $OB serve start --key "$1" --logdir "$LOGDIR" --stop-file "$RUNDIR/STOP"
}
serve_down() { run_child other $OB serve stop --logdir "$LOGDIR"; }
gate_abort() {  # the gate refused (GPU not quiet, no memory headroom, STOP): stop the run, resumable
  record "serve:$1" aborted 6 "refused by the pre-serve gate (see $LOGDIR/gate_*.json; README.md)"
  STATE=aborted_resource; finish 6
}
oom_check() {  # oom_check <step>: any kernel OOM kill during a pair may have changed what episodes observed
  local now; now="$($OB ops oom 2>/dev/null </dev/null {LOCK_FD}>&- || echo -1)"
  if [ "$OOM_BASE" -ge 0 ] 2>/dev/null && [ "$now" -gt "$OOM_BASE" ] 2>/dev/null; then
    record "oom:$1" failed 6 "the kernel OOM killer fired $((now - OOM_BASE)) time(s) during $1; inspect before resuming (README.md)"
    $PY -c 'import json,sys; json.dump({"reason": sys.argv[2], "source": "run_full"}, open(sys.argv[1], "w"))' \
      "$RUNDIR/STOP" "kernel OOM kill during $1"
    $OB serve stop --logdir "$LOGDIR" </dev/null >/dev/null 2>&1 {LOCK_FD}>&- || true
    STATE=aborted_resource; finish 6
  fi
}
run_pair() {  # run_pair <runner> <config> <name> <model> -> RC of the experiment run
  local runner="$1" cfg="$2" name="$3" model="$4"
  step "run:$name:$model"
  if [ "$runner" = "lab" ]; then
    # screenshot arms: one private Xvfb display per worker (pixel-identical to the shared display;
    # tests/docker/lab_check.py display parity), so the lab runs OB_LAB_WORKERS episodes at a time
    run_child lab "$LAB_RUN" "$cfg" "$NS" --model "$model" --backend openai --workers "$LAB_WORKERS" --per-worker-display
  else
    # per-model concurrency from the roster (`workers:`), else --workers (throughput only); OB_MAX_WORKERS caps it
    local w
    w="$($PY -c "from originblind.config import load_spec; m=[m for m in load_spec('$cfg').models if m.key=='$model']; print((m[0].workers if m and m[0].workers else '') or '$WORKERS')")"
    [ -n "$MAX_WORKERS" ] && [ "$w" -gt "$MAX_WORKERS" ] && w="$MAX_WORKERS"
    PAIR_WORKERS="$w"
    run_child host PROGRESS_DIR="$RESULTS/$NS/$name" $OB run "$cfg" --model "$model" --namespace "$NS" \
      --out "$RESULTS/$NS/$name/episodes.jsonl" --workers "$w"
  fi
}
audit_env() {  # the audits default to <repo>/results; point them at relocated results (tests)
  [ "$RESULTS" = "$REPO_ROOT/results" ] && return 0
  echo "OB_AUDIT_OUT=$RESULTS/$NS/$1"
}
leave_group() {  # a server that cannot be stopped ends the run
  CURRENT="stop:$GROUP"
  serve_down
  if [ "$RC" != 0 ]; then
    record "stop:$GROUP" failed "$RC" "the server could not be stopped; refusing to start another (README.md)"
    finish 1
  fi
}
GROUP=""; GROUP_OK=0; GROUP_FAILED=0; FAILED_GROUPS=0
# The schedule is read on fd 3 and every child gets /dev/null as stdin: `docker run` (inside
# `serve start`) and other children read stdin and would otherwise swallow the remaining schedule
# rows, silently ending the loop after the first group.
while IFS=$'\t' read -r -a F <&3; do
  [ -f "$RUNDIR/STOP" ] && on_signal USR1 6
  case "${F[0]}" in
    G)
      if [ -n "$GROUP" ]; then
        [ "$GROUP_OK" = "1" ] && leave_group
        # breaker: consecutive failed groups stop the loop before another server switch
        if [ "$GROUP_FAILED" = "1" ]; then FAILED_GROUPS=$((FAILED_GROUPS + 1)); else FAILED_GROUPS=0; fi
        if [ "$FAILED_GROUPS" -ge "${BUDGET_BREAKER_GROUPS%.*}" ]; then
          record breaker failed 1 "$FAILED_GROUPS consecutive groups failed; stopping before another server switch"
          finish 1
        fi
      fi
      GROUP="${F[2]}"; GMISSING="${F[7]}"; GROUP_FAILED=0; GROUP_OK=0
      step "serve:$GROUP"
      if [ "$GMISSING" = "0" ]; then
        record "serve:$GROUP" skipped 0 "nothing left to run"; continue
      else
        log "== group ${F[1]}: weights $GROUP (served as ${F[3]}), ${F[7]} episodes to run (~${F[8]} GPU-h)"
      fi
      serve_up "$GROUP"
      if [ "$RC" = 0 ]; then GROUP_OK=1; record "serve:$GROUP" ok 0
      elif [ "$RC" = 6 ]; then gate_abort "$GROUP"
      else GROUP_FAILED=1; record "serve:$GROUP" failed "$RC" "server did not come up or identity mismatch"; fi
      ;;
    P)
      runner="${F[2]}"; cfg="${F[3]}"; name="${F[4]}"; model="${F[5]}"; missing="${F[9]}"
      if [ "$missing" = "0" ]; then record "run:$name:$model" skipped 0 "complete"; continue; fi
      if [ "$GROUP_OK" != "1" ]; then GROUP_FAILED=1; record "run:$name:$model" failed 1 "no server for $GROUP"; continue; fi
      log "-- $name x $model ($runner): $missing to run"
      PAIR_WORKERS="$LAB_WORKERS"
      run_pair "$runner" "$cfg" "$name" "$model"; rc="$RC"
      oom_check "$name:$model"
      if [ "$rc" = "2" ] || [ "$rc" = "124" ]; then   # incomplete or stalled: one retry if the server is healthy
        if $OB serve status >/dev/null 2>&1 </dev/null; then
          log "retrying $name x $model once (exit $rc)"; run_pair "$runner" "$cfg" "$name" "$model"; rc="$RC"; oom_check "$name:$model"
        fi
      elif [ "$rc" = "3" ]; then                     # infra abort: restart the server once, then retry
        log "infra abort: restarting the server for $GROUP"
        serve_down
        [ "$RC" = 0 ] || { record "stop:$GROUP" failed "$RC" "the server could not be stopped"; finish 1; }
        serve_up "$GROUP"
        if [ "$RC" = 0 ]; then run_pair "$runner" "$cfg" "$name" "$model"; rc="$RC"; oom_check "$name:$model"
        elif [ "$RC" = 6 ]; then gate_abort "$GROUP"
        else GROUP_OK=0; fi
      fi
      case "$rc" in 0) record "run:$name:$model" ok 0 "workers=$PAIR_WORKERS" ;;
                    2) record "run:$name:$model" incomplete 2 "workers=$PAIR_WORKERS" ;;
                    124) GROUP_FAILED=1; record "run:$name:$model" failed 124 "stalled or over the wall limit (workers=$PAIR_WORKERS)" ;;
                    *) GROUP_FAILED=1; record "run:$name:$model" failed "$rc" "workers=$PAIR_WORKERS" ;; esac
      ;;
    U) record "run:${F[1]}:${F[2]}" skipped 0 "${F[3]}" ;;
  esac
done 3< "$SCHED_TSV"
if [ -n "$GROUP" ] && [ "$GROUP_OK" = "1" ]; then leave_group; fi  # the last served group

# ------------------------------------------------------------------ audits (deterministic, no GPU)
step audits
if [ "$SKIP_AUDITS" = "1" ]; then record audits skipped 0; else
  for a in exposure; do
    if [ -x "$AUDIT_DIR/$a/run.sh" ]; then
      # shellcheck disable=SC2046
      run_child other env $(audit_env "$a") "$AUDIT_DIR/$a/run.sh" "$NS"
      record "audit:$a" "$( [ "$RC" = 0 ] && echo ok || echo failed )" "$RC"
    else
      record "audit:$a" skipped 0 "$AUDIT_DIR/$a/run.sh not present"
    fi
  done
fi

# ------------------------------------------------------------------ validate (for `full`: every confirmatory config)
step validate
VCFG=("${CFG[@]}"); [ "$NS" = "full" ] && read -r -a VCFG <<<"$DEFAULT_CONFIGS"
INCOMPLETE=0
for c in "${VCFG[@]}"; do
  name="$($PY -c "from originblind.config import load_spec; print(load_spec('$c').name)")"
  d="$RESULTS/$NS/$name"
  [ -f "$d/episodes.jsonl" ] || { record "validate:$name" skipped 0 "no results"; INCOMPLETE=1; continue; }
  run_child other $OB validate "$d" --config "$c"
  case "$RC" in 0) record "validate:$name" ok 0 ;; 2) record "validate:$name" incomplete 2; INCOMPLETE=1 ;;
                *) record "validate:$name" failed "$RC"; INCOMPLETE=1 ;; esac
done

# ------------------------------------------------------------------ report (for `full`: only when complete)
step report
if [ "$NO_REPORT" = "1" ]; then record report skipped 0
elif [ "$NS" = "full" ] && { [ "$INCOMPLETE" = "1" ] || grep -q '"status": "failed"' "$STEPS"; }; then
  record report deferred 0 "the confirmatory report runs only on a complete namespace with no failed step (no interim looks)"
else
  run_child other $OB report "$RESULTS/$NS" --out "$RESULTS/$NS/report"
  record report "$( [ "$RC" = 0 ] && echo ok || echo failed )" "$RC" "$RESULTS/$NS/report"
fi

# Historical timestamp proofs are not part of this public release.
record timestamps skipped 0 "proof upgrade utility excluded"

if grep -q '"status": "failed"' "$STEPS"; then finish 1; fi
[ "$INCOMPLETE" = "1" ] && finish 2
finish 0
