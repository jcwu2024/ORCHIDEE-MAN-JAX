#!/bin/bash
# Shared command for one approved parallel learning arm; also supports --dry-run.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?declare the parallel experiment checkout}
GPU_DEVICE=${GPU_DEVICE:?declare the allocated GPU}
ARM=${ARM:?select A, B, C or D}
case "$ARM" in A|B|C|D) ;; *) echo "Invalid arm: $ARM" >&2; exit 2;; esac
PRINT_ONLY=false
if [[ "${1:-}" == --dry-run ]]; then PRINT_ONLY=true; shift; fi
ARGS=(env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="$GPU_DEVICE"
  GPU_MEMORY_FRACTION=0.80 PYTHON_MODULE=scripts.hpc.probe_litter_mixed_admission
  RUN_TIMEOUT="${RUN_TIMEOUT:-175m}"
  OUTPUT="${OUTPUT:-$ROOT/runtime/outputs/training/litter-parallel-v1/$ARM/report.json}"
  bash "$CODE_ROOT/scripts/hpc/run_fresh_start_one_day.sh"
  --protocol "$CODE_ROOT/manifests/coarse_graining/mixed_learning_epoch_model_v1.json"
  --mixed-protocol "$CODE_ROOT/manifests/coarse_graining/mixed_learning_epoch_v1.json"
  --source-continuation-report "$ROOT/runtime/outputs/diagnostics/litter-mixed-continuation-v1.json"
  --parallel-protocol "$CODE_ROOT/manifests/coarse_graining/litter_parallel_training_v1.json"
  --parallel-arm "$ARM"
  --execution-policy "$CODE_ROOT/manifests/coarse_graining/mixed_learning_execution_fp32_checked_v1.json"
  --catalogue-report "$ROOT/runtime/outputs/training/one-day-expansion-15061225/merge_report.json"
  --recurrent-schedule "$ROOT/runtime/outputs/diagnostics/mixed-learning-5415-recurrent-schedule-v1.json"
  --selection "$ROOT/runtime/outputs/diagnostics/trained-forward-selection-v1.json"
  --training-report "$ROOT/runtime/outputs/training/mixed-learning-dual-pool-v1/report.json"
  --checkpoint "$ROOT/runtime/outputs/diagnostics/litter-mixed-continuation-v1.selected-continuation-update-000100.checkpoint.pkl"
  --expected-update 4000 --samples-per-panel 8)
if "$PRINT_ONLY"; then
  printf '%q ' "${ARGS[@]}" "$@"
  printf '\n'
else
  exec "${ARGS[@]}" "$@"
fi
