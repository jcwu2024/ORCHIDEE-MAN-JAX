#!/bin/bash
# Independent zero-update probes; successful runtime checks precede heavy graphs.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:-$ROOT}
GPU_DEVICE=${GPU_DEVICE:?select an idle free GPU}
CHECK=${1:?checkpointed or directions}
PRINT_ONLY=${2:-}
[[ -z "$PRINT_ONLY" || "$PRINT_ONLY" == --dry-run ]] || exit 2
run_command() {
  if [[ "$PRINT_ONLY" == --dry-run ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}
case "$(hostname -s)" in gln01|ibgln01|ibc13b02n04) ;; *) exit 2;; esac
cd "$CODE_ROOT"
BASE=$ROOT/runtime/outputs/diagnostics/public-rollout-T2-v2
ARGS=(--horizon 2 --stage-directory "$BASE-stages")
case "$CHECK" in
  checkpointed)
    MODULE=scripts.hpc.probe_public_rollout_staged
    ARGS+=(--stage checkpointed)
    ;;
  directions)
    MODULE=scripts.hpc.diagnose_public_rollout_discrepancy
    ;;
  *) exit 2;;
esac
run_command env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="$GPU_DEVICE" \
  PYTHON_MODULE=scripts.hpc.check_daily_training_runtime RUN_TIMEOUT=3m \
  OUTPUT="$BASE-$CHECK-runtime.json" \
  bash scripts/hpc/run_fresh_start_one_day.sh --backend gpu
run_command env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="$GPU_DEVICE" \
  PYTHON_MODULE="$MODULE" RUN_TIMEOUT=90m \
  OUTPUT="$BASE-$CHECK-diagnosis.json" \
  bash scripts/hpc/run_fresh_start_one_day.sh \
    --protocol manifests/coarse_graining/fresh_start_one_day_probe_v3.json \
    "${ARGS[@]}"
