#!/bin/bash
# Free single-GPU fixed-checkpoint timing; every AdamW proposal is discarded.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?declare reviewed checkout}
DIAG=$ROOT/runtime/outputs/diagnostics
GPU_DEVICE=${GPU_DEVICE:-0}
PROFILE_REPEATS=${PROFILE_REPEATS:-3}
PRINT_ONLY=false
if [[ "${1:-}" == --dry-run ]]; then PRINT_ONLY=true; shift; fi
case "$PROFILE_REPEATS" in 3|5) ;; *) exit 2 ;; esac
if ! "$PRINT_ONLY"; then
  case "$(hostname -s)" in gln01|ibgln01|ibc13b02n04) ;; *) exit 2 ;; esac
fi
run_command() {
  if "$PRINT_ONLY"; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
run_command env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="$GPU_DEVICE" \
  GPU_MEMORY_FRACTION=0.85 PYTHON_MODULE=scripts.hpc.run_mixed_learning \
  RUN_TIMEOUT="${RUN_TIMEOUT:-90m}" \
  OUTPUT="${OUTPUT:-$DIAG/mixed-learning-budget-profile-v1.json}" \
  bash "$CODE_ROOT/scripts/hpc/run_fresh_start_one_day.sh" \
  --protocol "$CODE_ROOT/manifests/coarse_graining/mixed_learning_model_v1.json" \
  --mixed-protocol "$CODE_ROOT/manifests/coarse_graining/mixed_learning_v1.json" \
  --catalogue-report "$ROOT/runtime/outputs/training/one-day-expansion-15061225/merge_report.json" \
  --recurrent-schedule "$DIAG/mixed-learning-200-recurrent-schedule-v1.json" \
  --selection "$DIAG/trained-forward-selection-v1.json" \
  --execution-policy "$CODE_ROOT/manifests/coarse_graining/mixed_learning_execution_fp32_checked_v1.json" \
  --profile-checkpoint "$ROOT/runtime/outputs/training/mixed-learning-v1/report.update-000200.checkpoint.pkl" \
  --profile-training-report "$ROOT/runtime/outputs/training/mixed-learning-v1/report.json" \
  --profile-repeats "$PROFILE_REPEATS" --learning-budget-profile "$@"
