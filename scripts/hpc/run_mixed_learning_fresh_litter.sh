#!/bin/bash
# One declared fresh method; integration checkpoints never initialize training.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?declare reviewed checkout}
MODE=${1:?metadata, integration, or train}
shift
export MIXED_PROTOCOL=$CODE_ROOT/manifests/coarse_graining/mixed_learning_fresh_litter_v1.json
export MODEL_PROTOCOL=$CODE_ROOT/manifests/coarse_graining/mixed_learning_fresh_litter_model_v1.json
export RECURRENT_SCHEDULE=$ROOT/runtime/outputs/diagnostics/mixed-learning-5415-recurrent-schedule-v1.json
case "$MODE" in
  metadata)
    export OUTPUT=${OUTPUT:-$ROOT/runtime/outputs/diagnostics/fresh-litter-v1-metadata.json}
    exec bash "$CODE_ROOT/scripts/hpc/run_mixed_learning.sh" metadata "$@" ;;
  integration|train)
    if [[ "$MODE" == integration ]]; then
      export OUTPUT=${OUTPUT:-$ROOT/runtime/outputs/diagnostics/fresh-litter-v1-integration.json}
      export INTEGRATION_UPDATES=2
      export RUN_TIMEOUT=${RUN_TIMEOUT:-90m}
    else
      export OUTPUT=${OUTPUT:-$ROOT/runtime/outputs/training/fresh-litter-v1/report.json}
      export RUN_TIMEOUT=${RUN_TIMEOUT:-295m}
    fi
    ARGS=("$MODE")
    if [[ "${1:-}" == --dry-run ]]; then ARGS+=(--dry-run); shift; fi
    exec bash "$CODE_ROOT/scripts/hpc/run_mixed_learning.sh" "${ARGS[@]}" \
      --execution-policy "$CODE_ROOT/manifests/coarse_graining/mixed_learning_execution_fp32_checked_v1.json" \
      --dual-device-training "$@" ;;
  *) exit 2 ;;
esac
