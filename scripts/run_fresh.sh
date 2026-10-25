#!/usr/bin/env bash
# Fresh evaluation, separate from the immutable retained evidence.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
./verify.sh --integrity-only
export OB_RESULTS="$REPO_ROOT/outputs/fresh"
export OB_RESULTS_VOLUME=address-bar-receiver-results
export OB_BU_LLM_MODELS=none
exec scripts/run_full.sh --allow-dirty --skip-audits --skip-timestamps "$@"
