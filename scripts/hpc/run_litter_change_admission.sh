#!/bin/bash
# Bounded free daily-loss admission, not a new paid learning run.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?declare diagnostic checkout}
GPU_DEVICE=${GPU_DEVICE:?select idle GPU}
PRINT_ONLY=false
if [[ "${1:-}" == --dry-run ]]; then PRINT_ONLY=true; shift; fi
ARGS=(env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="$GPU_DEVICE"
  GPU_MEMORY_FRACTION=0.80 PYTHON_MODULE=scripts.hpc.probe_litter_change_admission
  RUN_TIMEOUT="${RUN_TIMEOUT:-60m}"
  OUTPUT="${OUTPUT:-$ROOT/runtime/outputs/diagnostics/litter-change-admission-v1.json}"
  bash "$CODE_ROOT/scripts/hpc/run_fresh_start_one_day.sh"
  --protocol "$CODE_ROOT/manifests/coarse_graining/mixed_learning_epoch_model_v1.json"
  --mixed-protocol "$CODE_ROOT/manifests/coarse_graining/mixed_learning_epoch_v1.json"
  --admission-protocol "$CODE_ROOT/manifests/coarse_graining/litter_change_admission_v1.json"
  --execution-policy "$CODE_ROOT/manifests/coarse_graining/mixed_learning_execution_fp32_checked_v1.json"
  --catalogue-report "$ROOT/runtime/outputs/training/one-day-expansion-15061225/merge_report.json"
  --recurrent-schedule "$ROOT/runtime/outputs/diagnostics/mixed-learning-5415-recurrent-schedule-v1.json"
  --selection "$ROOT/runtime/outputs/diagnostics/trained-forward-selection-v1.json"
  --training-report "$ROOT/runtime/outputs/training/mixed-learning-dual-pool-v1/report.json"
  --checkpoint "$ROOT/runtime/outputs/training/mixed-learning-dual-pool-v1/report.update-004000.checkpoint.pkl"
  --expected-update 4000 --samples-per-panel 8)
if "$PRINT_ONLY"; then
  printf '%q ' "${ARGS[@]}" "$@"
  printf '\n'
else
  exec "${ARGS[@]}" "$@"
fi
