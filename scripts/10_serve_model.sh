#!/usr/bin/env bash
# Serve one roster model from the ob-models volume and verify its identity.
# usage: scripts/10_serve_model.sh <model-key> [extra `originblind serve start` args]
source "$(dirname "$0")/_common.sh"
[ $# -ge 1 ] || die "usage: $0 <model-key>"
KEY="$1"; shift
exec $OB serve start --key "$KEY" --logdir "$RESULTS/logs" "$@"
