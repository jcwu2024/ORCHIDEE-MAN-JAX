#!/bin/bash
# Shared declared one-pool-pass entry for free metadata/integration and paid train.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?declare reviewed checkout}
MODE=${1:?metadata, integration, or train}
shift
export MIXED_PROTOCOL=$CODE_ROOT/manifests/coarse_graining/mixed_learning_epoch_v1.json
export MODEL_PROTOCOL=$CODE_ROOT/manifests/coarse_graining/mixed_learning_epoch_model_v1.json
export RECURRENT_SCHEDULE=$ROOT/runtime/outputs/diagnostics/mixed-learning-5415-recurrent-schedule-v1.json
case "$MODE" in
  metadata)
    export OUTPUT=${OUTPUT:-$ROOT/runtime/outputs/diagnostics/mixed-learning-epoch-v1-metadata.json}
    exec bash "$CODE_ROOT/scripts/hpc/run_mixed_learning.sh" metadata "$@" ;;
  integration|train)
    if [[ "$MODE" == integration ]]; then
      export OUTPUT=${OUTPUT:-$ROOT/runtime/outputs/diagnostics/mixed-learning-epoch-v1-integration.json}
      export RUN_TIMEOUT=${RUN_TIMEOUT:-90m}
    else
      export OUTPUT=${OUTPUT:-$ROOT/runtime/outputs/training/mixed-learning-epoch-v1/report.json}
      export RUN_TIMEOUT=${RUN_TIMEOUT:-1435m}
    fi
    # Preserve --dry-run as the first argument to the shared launcher.
    DRY=()
    if [[ "${1:-}" == --dry-run ]]; then DRY=(--dry-run); shift; fi
    exec bash "$CODE_ROOT/scripts/hpc/run_mixed_learning.sh" "$MODE" "${DRY[@]}" \
      --execution-policy "$CODE_ROOT/manifests/coarse_graining/mixed_learning_execution_fp32_checked_v1.json" "$@" ;;
  *) exit 2 ;;
esac
