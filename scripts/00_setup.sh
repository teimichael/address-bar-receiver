#!/usr/bin/env bash
# Idempotent environment setup: Python env, host Chromium, container
# images pinned by digest, the docker network/volumes, and (when a selected config needs it) the lab image.
# usage: scripts/00_setup.sh [--configs "cfg1 cfg2 ..."]
source "$(dirname "$0")/_common.sh"
CONFIGS=""
while [ $# -gt 0 ]; do case "$1" in --configs) CONFIGS="$2"; shift 2 ;; --configs=*) CONFIGS="${1#*=}"; shift ;; *) die "unknown argument $1" ;; esac; done

if command -v poetry >/dev/null 2>&1; then
  # `poetry sync` (Poetry >= 2.0) replaces the deprecated `poetry install --sync`
  if poetry sync --help >/dev/null 2>&1; then
    log "poetry sync"; poetry sync --no-interaction
  else
    log "poetry install --sync"; poetry install --sync --no-interaction
  fi
elif [ -n "${OB_PY:-}" ] || [ -x .venv/bin/python ]; then
  log "poetry not on PATH; using the existing .venv"
else
  die "neither poetry nor .venv/bin/python found"
fi
log "playwright install chromium"; $PY -m playwright install chromium
# Pinned images: pulled by digest, then tagged under their OWN repository name. The run manifests
# record `RepoDigests[0]` of the tags (run/orchestrator.py), and a tag alone may since point to
# another build. The vLLM digest is serving/vllm.py's IMAGE_DIGEST; the lab base is the Dockerfile's FROM.
pin() {  # pin <repo:tag> <sha256:...>: idempotent
  local img="$1" dig="$2" repo="${1%:*}"
  if docker image inspect -f '{{.Id}} {{join .RepoDigests " "}}' "$img" 2>/dev/null </dev/null \
       | tr ' ' '\n' | grep -qx -e "$dig" -e "$repo@$dig"; then
    return 0
  fi
  log "docker pull $repo@$dig (as $img)"
  docker pull "$repo@$dig" </dev/null
  docker tag "$repo@$dig" "$img" </dev/null
}
read -r VLLM_IMG VLLM_DIGEST <<<"$($PY -c 'from originblind.serving.vllm import IMAGE, IMAGE_DIGEST; print(IMAGE, IMAGE_DIGEST)')"
BASE_REF="$(sed -n 's/^FROM \([^ ]*\).*/\1/p' docker/lab.Dockerfile | head -1)"   # repo:tag@sha256:...
case "$BASE_REF" in *:*@sha256:*) ;; *) die "docker/lab.Dockerfile: FROM must be repo:tag@sha256:digest, not '$BASE_REF'" ;; esac
pin "$VLLM_IMG" "$VLLM_DIGEST"
pin "${BASE_REF%@*}" "${BASE_REF#*@}"
docker image inspect busybox:1.37.0@sha256:bdf57e528e45e4433820e045b29b4597825a1c9e38353532d90a01445013f82e >/dev/null 2>&1 </dev/null || { log "docker pull busybox:1.37.0@sha256:bdf57e528e45e4433820e045b29b4597825a1c9e38353532d90a01445013f82e"; docker pull busybox:1.37.0@sha256:bdf57e528e45e4433820e045b29b4597825a1c9e38353532d90a01445013f82e </dev/null; }
docker network inspect ob-net >/dev/null 2>&1 || docker network create ob-net >/dev/null
for v in ob-models "${OB_RESULTS_VOLUME:-ob-results}"; do docker volume inspect "$v" >/dev/null 2>&1 || docker volume create "$v" >/dev/null; done
NEED_LAB=0
for c in $CONFIGS; do grep -q "^driver: chromium-headful" "$c" && NEED_LAB=1 || true; done
if [ "$NEED_LAB" = "1" ]; then log "lab image"; scripts/lab_run.sh --ensure-image; fi
log "setup done"
