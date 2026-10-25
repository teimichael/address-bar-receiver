#!/usr/bin/env bash
# Run one experiment config inside the lab container: headful Chromium under Xvfb with the twin CA
# trusted through NSS. This is the only place the screenshot arms (driver: chromium-headful) run.
#
# usage: scripts/lab_run.sh <config> <namespace> [--model KEY] [--backend mock|openai] [--workers N]
#                           [--limit N] [--allow-code-change] [--per-worker-display]
#        scripts/lab_run.sh --ensure-image      build/refresh the lab image only, print its id
#
#  - Image: originblind-lab:latest. Rebuilt whenever its org.originblind.src_sha256 label differs
#    from the working tree (src/, pyproject.toml, poetry.lock, docker/, .dockerignore), so the
#    container always runs the current harness code. With OB_LAB_NO_BUILD=1 it refuses instead.
#  - Network: the user-defined bridge ob-net (created if missing). The vLLM container `ob-vllm`
#    (started elsewhere) is on the same network and is reached as http://ob-vllm:8000/v1.
#  - Files: nothing is bind-mounted. Docker Desktop bind mounts of WSL paths go stale, and the
#    docker CLI here can be the Windows binary, which cannot take Linux host paths. configs/ (plus,
#    for an out-of-repo config, the YAML files next to it) are streamed in as a tar. Results go to
#    the named volume ob-results (/app/results) and are exported to <repo>/results/<ns>/<exp>/
#    through a busybox tar pipe, where <exp> is the config's file stem.
#  - Concurrency: by default all headful episodes share the entrypoint's display, so the run is one
#    browser at a time (--workers is then ignored for chromium-headful). --per-worker-display (or
#    OB_LAB_PER_WORKER_DISPLAY=1) sets ORIGIN_BLIND_LAB_PER_WORKER_DISPLAY=1 in the container:
#    each of the --workers worker processes starts a private Xvfb with the entrypoint's geometry,
#    depth and DPI (browser/run_browser.py), so screenshot-arm episodes run concurrently with the
#    same pixels. The choice is recorded in the provenance sidecar.
#  - Resumable and idempotent: the volume keeps every episode, so re-running the same command
#    resumes. If the volume lacks the experiment but the host has it, the host copy seeds the
#    volume first. Each invocation writes lab_runs/<utc>.{json,log} next to the results
#    (image id, source hash, git state, argv, exit code; the container log).
#  - Safety (README.md): the container gets --memory/--memory-swap/--pids-limit backstops
#    (ops/budget.py: VM MemTotal minus a reserve; OB_LAB_MEMORY, OB_PIDS_LIMIT) and its OOMKilled state
#    is recorded in the sidecar. INT/TERM/HUP stop the container promptly (the attached client runs in
#    the background). Exited leftovers of this experiment are removed after saving their logs; a
#    running one is refused. The volume copy replaces the host copy only if the host copy is a
#    byte-identical prefix of it (`originblind ops store prefix-check`): the export is staged and
#    sha256-verified, the previous host copy is kept under results/.ops/backup/, files move into place
#    one by one (atomic renames) and host-only files (lab_runs/) stay. The same check runs before a
#    resume; on divergence the run is refused, never re-seeded automatically.
#  - Exit code: the CLI's (0 done, 2 incomplete, 3 infra abort, 4 served-model mismatch, 5 invalid
#    config or refused resume), 130 if interrupted, 1 for a failure of this script (usage, build,
#    export, refused resume).
#
# Environment overrides: OB_LAB_IMAGE, OB_NET, OB_RESULTS_VOLUME, OB_RESULTS_DIR (export target,
# default <repo>/results), OB_VLLM_URL, OB_BUSYBOX, OB_LAB_NO_BUILD, OB_LAB_PER_WORKER_DISPLAY, OB_PY.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${OB_LAB_IMAGE:-originblind-lab:latest}"
NET="${OB_NET:-ob-net}"
VOLUME="${OB_RESULTS_VOLUME:-ob-results}"
RESULTS_DIR="${OB_RESULTS_DIR:-$REPO_ROOT/results}"
VLLM_URL="${OB_VLLM_URL:-http://ob-vllm:8000/v1}"
BUSYBOX="${OB_BUSYBOX:-busybox:1.37.0@sha256:bdf57e528e45e4433820e045b29b4597825a1c9e38353532d90a01445013f82e}"
PY="${OB_PY:-$REPO_ROOT/.venv/bin/python}"
BASE_DIGEST="sha256:72bd171a9ffc2b4b59532aaa6210e21014d07093120dc25528870c0b840da1f0"

die() { echo "lab_run: $*" >&2; exit 1; }
usage() { sed -n '5,7p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 1; }
json_str() { local s="${1//\\/\\\\}"; s="${s//\"/\\\"}"; printf '"%s"' "$s"; }

command -v docker >/dev/null 2>&1 || die "docker CLI not found"
# The docker CLI may be the Windows binary (WSL interop): give it only paths relative to the repo
# root, and never let it read the caller's stdin unless a tar stream is meant for it.
dk() { docker "$@" </dev/null; }
cd "$REPO_ROOT"

src_hash() {
  { find src -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
    sha256sum pyproject.toml poetry.lock docker/lab.Dockerfile docker/entrypoint.sh docker/dpkg-packages.lock .dockerignore
  } | sha256sum | cut -c1-64
}

ensure_image() {
  local want have
  want="$(src_hash)"
  have="$(dk image inspect -f '{{index .Config.Labels "org.originblind.src_sha256"}}' "$IMAGE" 2>/dev/null || true)"
  if [ "$have" != "$want" ]; then
    [ "${OB_LAB_NO_BUILD:-0}" = "1" ] && die "image $IMAGE is missing or stale (label '${have:-none}' != $want) and OB_LAB_NO_BUILD=1"
    echo "lab_run: building $IMAGE for source hash ${want:0:12}" >&2
    dk build -t "$IMAGE" -f docker/lab.Dockerfile --build-arg "OB_SRC_SHA256=$want" . >&2 \
      || die "lab image build failed"
  fi
  SRC_SHA256="$want"
  IMAGE_ID="$(dk image inspect -f '{{.Id}}' "$IMAGE")"
  echo "lab_run: image $IMAGE id=$IMAGE_ID base=$BASE_DIGEST src=${want:0:12}" >&2
}

if [ "${1:-}" = "--ensure-image" ]; then
  ensure_image
  echo "$IMAGE_ID"
  exit 0
fi

[ $# -ge 2 ] || usage
CFG="$1"; NS="$2"; shift 2
CLI_EXTRA=()
PER_WORKER="${OB_LAB_PER_WORKER_DISPLAY:-0}"
while [ $# -gt 0 ]; do
  case "$1" in
    --model|--backend|--workers|--limit) [ $# -ge 2 ] || usage; CLI_EXTRA+=("$1" "$2"); shift 2 ;;
    --model=*|--backend=*|--workers=*|--limit=*|--allow-code-change) CLI_EXTRA+=("$1"); shift ;;
    --per-worker-display) PER_WORKER=1; shift ;;
    *) echo "lab_run: unknown argument $1" >&2; usage ;;
  esac
done
case "$PER_WORKER" in 0|1) ;; *) die "OB_LAB_PER_WORKER_DISPLAY must be 0 or 1, not '$PER_WORKER'" ;; esac
[ -f "$CFG" ] || die "config not found: $CFG"
case "$NS" in ""|*/*|.*) die "invalid namespace: '$NS'" ;; esac

CFG_ABS="$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")"
EXP="$(basename "$CFG_ABS")"; EXP="${EXP%.yaml}"; EXP="${EXP%.yml}"
OUT_REL="$NS/$EXP"

STAGE="$(mktemp -d)"
CID=""
cleanup() {
  [ -n "$CID" ] && dk rm -f "$CID" >/dev/null 2>&1 || true
  rm -rf "$STAGE"
}
trap cleanup EXIT

# configs: the repo's configs/ always (rosters are referenced relatively); an out-of-repo config is
# placed under configs/_external/ together with the YAML files next to it
cp -r configs "$STAGE/configs"
case "$CFG_ABS" in
  "$REPO_ROOT"/configs/*) CFG_IN="${CFG_ABS#"$REPO_ROOT"/}" ;;
  *)
    mkdir -p "$STAGE/configs/_external"
    find "$(dirname "$CFG_ABS")" -maxdepth 1 -type f \( -name '*.yaml' -o -name '*.yml' \) \
      -exec cp {} "$STAGE/configs/_external/" \;
    cp "$CFG_ABS" "$STAGE/configs/_external/"
    CFG_IN="configs/_external/$(basename "$CFG_ABS")"
    ;;
esac

ensure_image
dk network inspect "$NET" >/dev/null 2>&1 || dk network create "$NET" >/dev/null \
  || die "cannot create network $NET"
dk volume inspect "$VOLUME" >/dev/null 2>&1 || dk volume create "$VOLUME" >/dev/null \
  || die "cannot create volume $VOLUME"

# busybox on the results volume; only vol_in attaches stdin (never swallow the caller's stdin)
vol() { dk run --rm -v "$VOLUME:/r" "$BUSYBOX" "$@"; }
vol_in() { docker run --rm -i -v "$VOLUME:/r" "$BUSYBOX" "$@"; }

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
STAGING="$RESULTS_DIR/.ops/staging/$OUT_REL.$STAMP"
PREFIX="ob-lab-$(printf '%s' "$NS-$EXP" | tr -c 'a-zA-Z0-9_.-' '-')-"
# copy the volume's experiment into $STAGING and verify it against sha256 sums taken inside the volume
stage_volume() {
  rm -rf "$STAGING"; mkdir -p "$STAGING"
  vol tar -C /r -cf - "$OUT_REL" | tar -C "$STAGING" --no-same-owner -xf - || return 1
  local want have
  want="$(vol sh -c "cd /r && find '$OUT_REL' -type f -exec sha256sum {} + | sort -k2")" || return 1
  have="$(cd "$STAGING" && find "$OUT_REL" -type f -exec sha256sum {} + | sort -k2)" || return 1
  [ "$want" = "$have" ]
}
# the host copy is a byte-identical prefix of the staged volume copy (nothing on the host can be lost)
prefix_ok() {
  [ -e "$RESULTS_DIR/$OUT_REL/episodes.jsonl" ] || return 0
  "$PY" -m originblind.cli ops store prefix-check --host "$RESULTS_DIR/$OUT_REL" --candidate "$STAGING/$OUT_REL" >&2
}

# leftovers of this experiment from a crashed attempt: keep their logs, remove them; a running one
# is a second writer on the volume
while IFS=$'\t' read -r cname cstate; do
  case "$cname" in "$PREFIX"*) ;; *) continue ;; esac
  [ "$cstate" = "running" ] && die "a lab container of this experiment is still running ($cname); stop it first if no other run is active (docker stop $cname)"
  mkdir -p "$RESULTS_DIR/$OUT_REL/lab_runs"
  dk logs --timestamps "$cname" >"$RESULTS_DIR/$OUT_REL/lab_runs/orphan_${cname}_$STAMP.log" 2>&1 || true
  dk rm -f "$cname" >/dev/null 2>&1 || true
  echo "lab_run: removed the leftover container $cname ($cstate); its log is in lab_runs/" >&2
done < <(dk ps -a --filter "name=^$PREFIX" --format '{{.Names}}\t{{.State}}' 2>/dev/null || true)

# seed the volume from the host if only the host has this experiment (e.g. a recreated volume)
if [ -s "$RESULTS_DIR/$OUT_REL/episodes.jsonl" ] && ! vol test -e "/r/$OUT_REL/episodes.jsonl"; then
  echo "lab_run: seeding volume $VOLUME from $RESULTS_DIR/$OUT_REL" >&2
  tar -C "$RESULTS_DIR" -cf - "$OUT_REL" | vol_in tar -C /r -xf - || die "seeding the volume failed"
elif [ -s "$RESULTS_DIR/$OUT_REL/episodes.jsonl" ]; then
  # never resume from a volume copy that lost records the host holds
  stage_volume || die "staging the volume copy of $OUT_REL failed"
  prefix_ok || die "the volume copy of $OUT_REL does not continue the host copy (see above); reconcile them before resuming (README.md)"
  rm -rf "$STAGING"
fi

# container backstops (never bind before the VM itself would; README.md)
LIMITS=($("$PY" - <<'PYLIM'
from originblind.ops.budget import container_memory, load_budget
from originblind.ops.probes import vm_memory
b = load_budget()
m = container_memory(b.lab_memory, (vm_memory() or {}).get("total"), b.container_reserve_gib)
print(" ".join((["--memory", m, "--memory-swap", m] if m else []) + (["--pids-limit", str(b.pids_limit)] if b.pids_limit else [])))
PYLIM
)) || die "invalid resource budget (OB_LAB_MEMORY / OB_PIDS_LIMIT; README.md)"

STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
NAME="$PREFIX$$"
CMD=(python -m originblind.cli run "$CFG_IN" --out "results/$OUT_REL/episodes.jsonl" --namespace "$NS" "${CLI_EXTRA[@]}")
CID="$(dk create --name "$NAME" --init --ipc=host "${LIMITS[@]}" --network "$NET" \
  -v "$VOLUME:/app/results" \
  -e "ORIGIN_BLIND_VLLM_URL=$VLLM_URL" -e "ORIGIN_BLIND_LAB_IMAGE=$IMAGE@$IMAGE_ID" \
  -e "ORIGIN_BLIND_LAB_PER_WORKER_DISPLAY=$PER_WORKER" \
  "$IMAGE_ID" "${CMD[@]}")" || die "docker create failed"  # the ID recorded above, not the re-taggable tag
tar -C "$STAGE" -cf - configs | docker cp - "$CID:/app/" || die "copying configs into the container failed"

echo "lab_run: $NAME: ${CMD[*]} (per-worker display: $PER_WORKER)" >&2
INTERRUPTED=0
trap 'INTERRUPTED=1' INT TERM HUP
set +e
dk start -a "$CID" &   # in the background: bash runs a trap only between foreground commands
ATTACH=$!
while :; do
  wait "$ATTACH"; RC=$?
  [ "$INTERRUPTED" = "1" ] && break           # a trapped signal ended the wait early
  kill -0 "$ATTACH" 2>/dev/null || break      # the attached client exited: the run is over
done
if [ "$INTERRUPTED" = "1" ]; then
  dk stop -t 20 "$CID" >/dev/null 2>&1 || true
  wait "$ATTACH" 2>/dev/null
  RC=130
fi
OOM_KILLED="$(dk inspect -f '{{.State.OOMKilled}}' "$CID" 2>/dev/null || echo unknown)"
set -e
trap - INT TERM HUP
FINISHED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
[ "$OOM_KILLED" = "true" ] && echo "lab_run: WARNING: the lab container was OOM-killed (README.md)" >&2

# export the experiment (every episode the volume holds, also after a failure): staged, verified,
# only as a continuation of the host copy, the previous host copy kept, files renamed into place
EXPORT_OK=1
if vol test -d "/r/$OUT_REL"; then
  mkdir -p "$RESULTS_DIR"
  if ! stage_volume; then
    EXPORT_OK=0; echo "lab_run: staging the volume copy of $OUT_REL failed; the host copy is unchanged" >&2
  elif ! prefix_ok; then
    EXPORT_OK=0
    echo "lab_run: the volume copy would lose host records; the host copy is unchanged; the volume copy is in $STAGING" >&2
  else
    if [ -d "$RESULTS_DIR/$OUT_REL" ]; then
      mkdir -p "$(dirname "$RESULTS_DIR/.ops/backup/$OUT_REL")"
      cp -a "$RESULTS_DIR/$OUT_REL" "$RESULTS_DIR/.ops/backup/$OUT_REL.$STAMP"
    fi
    if ! (cd "$STAGING/$OUT_REL" && find . -type f) | while IFS= read -r f; do
           mkdir -p "$(dirname "$RESULTS_DIR/$OUT_REL/$f")" && mv -f "$STAGING/$OUT_REL/$f" "$RESULTS_DIR/$OUT_REL/$f"
         done; then
      EXPORT_OK=0; echo "lab_run: moving the export into place failed (previous copy: results/.ops/backup/)" >&2
    else
      rm -rf "$STAGING"
    fi
  fi
fi

# per-invocation provenance next to the results (the image cannot record its own id)
SIDE="$RESULTS_DIR/$OUT_REL/lab_runs"
mkdir -p "$SIDE"
dk logs "$CID" >"$SIDE/$STAMP.log" 2>&1 || true
GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$( [ -n "$(git status --porcelain 2>/dev/null)" ] && echo true || echo false )"
ARGV_JSON="$(for a in "${CMD[@]}"; do json_str "$a"; printf ','; done)"
cat >"$SIDE/$STAMP.json" <<EOF
{
  "image": $(json_str "$IMAGE"),
  "image_id": $(json_str "$IMAGE_ID"),
  "base_digest": $(json_str "$BASE_DIGEST"),
  "src_sha256": $(json_str "$SRC_SHA256"),
  "git_sha": $(json_str "$GIT_SHA"),
  "git_dirty": $GIT_DIRTY,
  "config": $(json_str "$CFG_IN"),
  "namespace": $(json_str "$NS"),
  "experiment": $(json_str "$EXP"),
  "argv": [${ARGV_JSON%,}],
  "network": $(json_str "$NET"),
  "volume": $(json_str "$VOLUME"),
  "vllm_url": $(json_str "$VLLM_URL"),
  "per_worker_display": $( [ "$PER_WORKER" = "1" ] && echo true || echo false ),
  "limits": $(json_str "${LIMITS[*]}"),
  "oom_killed": $(json_str "$OOM_KILLED"),
  "started_utc": $(json_str "$STARTED"),
  "finished_utc": $(json_str "$FINISHED"),
  "exit_code": $RC,
  "exported": $( [ "$EXPORT_OK" = "1" ] && echo true || echo false )
}
EOF
echo "lab_run: exit $RC; results in $RESULTS_DIR/$OUT_REL" >&2
if [ "$EXPORT_OK" != "1" ] && [ "$RC" = "0" ]; then
  exit 1
fi
exit "$RC"
