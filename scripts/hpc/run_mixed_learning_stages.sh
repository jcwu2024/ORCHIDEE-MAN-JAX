#!/bin/bash
# Same model/data/optimizer; declared daily-to-mixed schedule is the only change.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?declare reviewed checkout}
MODE=${1:?integration, joint, or staged}
shift
ARGS=("$MODE")
DRY=false
if [[ "${1:-}" == --dry-run ]]; then DRY=true; shift; fi
case "$MODE" in
  integration)
    ARGS=(integration)
    SCHEDULE=$CODE_ROOT/manifests/coarse_graining/mixed_learning_stages_integration_v1.json
    export INTEGRATION_UPDATES=2
    export OUTPUT=${OUTPUT:-$ROOT/runtime/outputs/diagnostics/mixed-learning-stages-integration-v1.json}
    export RUN_TIMEOUT=${RUN_TIMEOUT:-90m} ;;
  joint|staged)
    ARGS=(train)
    if [[ "$MODE" == joint ]]; then
      SCHEDULE=$CODE_ROOT/manifests/coarse_graining/mixed_learning_joint_pilot_v1.json
    else
      SCHEDULE=$CODE_ROOT/manifests/coarse_graining/mixed_learning_stages_pilot_v1.json
    fi
    export OUTPUT=${OUTPUT:-$ROOT/runtime/outputs/training/mixed-learning-stages-pilot-v1/$MODE/report.json}
    export RUN_TIMEOUT=${RUN_TIMEOUT:-220m} ;;
  *) exit 2 ;;
esac
if "$DRY"; then ARGS+=(--dry-run); fi
exec bash "$CODE_ROOT/scripts/hpc/run_mixed_learning_epoch.sh" "${ARGS[@]}" \
  --learning-schedule "$SCHEDULE" "$@"
