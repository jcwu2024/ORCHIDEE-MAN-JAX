#!/bin/bash
# Forward-only held-out validation for a declared fresh-method milestone.
set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?declare reviewed checkout}
GPU_DEVICE=${GPU_DEVICE:?select an idle or allocated GPU}
EXPECTED_UPDATE=${EXPECTED_UPDATE:-1000}
RUN_TIMEOUT=${RUN_TIMEOUT:-90m}
TRAINING_DIR=${TRAINING_DIR:-$ROOT/runtime/outputs/training/fresh-litter-v1}
TRAINING_REPORT=${TRAINING_REPORT:-$TRAINING_DIR/report.json}
VALIDATION_MANIFEST=${VALIDATION_MANIFEST:-$ROOT/runtime/outputs/training/fresh-method-validation-20260921/prepared_manifest.json}
printf -v UPDATE_PADDED '%06d' "$EXPECTED_UPDATE"
TRAINING_CHECKPOINT=${TRAINING_CHECKPOINT:-$TRAINING_DIR/report.update-$UPDATE_PADDED.checkpoint.pkl}
OUTPUT=${OUTPUT:-$ROOT/runtime/outputs/training/fresh-method-validation-20260921/evaluation-update-$UPDATE_PADDED.json}

PRINT_ONLY=false
if [[ "${1:-}" == --dry-run ]]; then PRINT_ONLY=true; shift; fi

ARGS=(env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="$GPU_DEVICE"
  GPU_MEMORY_FRACTION=0.85 PYTHON_MODULE=scripts.hpc.evaluate_fresh_method_validation
  RUN_TIMEOUT="$RUN_TIMEOUT" OUTPUT="$OUTPUT"
  bash "$CODE_ROOT/scripts/hpc/run_fresh_start_one_day.sh"
  --protocol "$CODE_ROOT/manifests/coarse_graining/mixed_learning_fresh_litter_model_v1.json"
  --mixed-protocol "$CODE_ROOT/manifests/coarse_graining/mixed_learning_fresh_litter_v1.json"
  --execution-policy "$CODE_ROOT/manifests/coarse_graining/mixed_learning_execution_fp32_checked_v1.json"
  --catalogue-report "$ROOT/runtime/outputs/training/one-day-expansion-15061225/merge_report.json"
  --recurrent-schedule "$ROOT/runtime/outputs/diagnostics/mixed-learning-5415-recurrent-schedule-v1.json"
  --selection "$ROOT/runtime/outputs/diagnostics/trained-forward-selection-v1.json"
  --training-report "$TRAINING_REPORT"
  --training-checkpoint "$TRAINING_CHECKPOINT"
  --expected-update "$EXPECTED_UPDATE"
  --validation-manifest "$VALIDATION_MANIFEST")

if "$PRINT_ONLY"; then
  printf '%q ' "${ARGS[@]}" "$@"
  printf '\n'
else
  exec "${ARGS[@]}" "$@"
fi
