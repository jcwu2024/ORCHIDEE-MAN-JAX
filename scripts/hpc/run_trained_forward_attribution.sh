#!/bin/bash
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?declare the attribution checkout}
GPU_DEVICE=${GPU_DEVICE:?select an idle free GPU}
PRINT_ONLY=${1:-}
[[ -z "$PRINT_ONLY" || "$PRINT_ONLY" == --dry-run ]] || exit 2
cd "$CODE_ROOT"
ARGS=(--selection "$ROOT/runtime/outputs/diagnostics/trained-forward-selection-v1.json"
      --training-report "$ROOT/runtime/outputs/training/fresh-one-day-pool-15048786/report.json"
      --checkpoint "$ROOT/runtime/outputs/training/fresh-one-day-pool-15048786/report.checkpoint.pkl"
      --reference-report "$ROOT/runtime/outputs/diagnostics/trained-forward-ad-v1-forward.json")
run_command() {
  if [[ "$PRINT_ONLY" == --dry-run ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}
run_command env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="$GPU_DEVICE" \
  PYTHON_MODULE=scripts.hpc.check_daily_training_runtime RUN_TIMEOUT=3m \
  OUTPUT="$ROOT/runtime/outputs/diagnostics/trained-forward-attribution-v1-runtime.json" \
  bash "$CODE_ROOT/scripts/hpc/run_fresh_start_one_day.sh" --backend gpu
run_command env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="$GPU_DEVICE" \
  PYTHON_MODULE=scripts.hpc.diagnose_trained_forward RUN_TIMEOUT=60m \
  OUTPUT="$ROOT/runtime/outputs/diagnostics/trained-forward-attribution-v1.json" \
  bash "$CODE_ROOT/scripts/hpc/run_fresh_start_one_day.sh" "${ARGS[@]}"
