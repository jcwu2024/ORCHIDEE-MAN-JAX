#!/bin/bash
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?declare the diagnostic checkout}
GPU_DEVICE=${GPU_DEVICE:?select an idle free GPU}
PRINT_ONLY=${1:-}
[[ -z "$PRINT_ONLY" || "$PRINT_ONLY" == --dry-run ]] || exit 2
case "$CODE_ROOT" in "$ROOT"|"$ROOT"/runtime/checkouts/*) ;; *) exit 2;; esac
cd "$CODE_ROOT"
ARGS=(--selection "$ROOT/runtime/outputs/diagnostics/trained-forward-selection-v1.json"
      --training-report "$ROOT/runtime/outputs/training/fresh-one-day-pool-15048786/report.json"
      --checkpoint "$ROOT/runtime/outputs/training/fresh-one-day-pool-15048786/report.checkpoint.pkl"
      --attribution-report "$ROOT/runtime/outputs/diagnostics/trained-forward-attribution-v1.json"
      --archive-directory "$ROOT/runtime/outputs/diagnostics")
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
  OUTPUT="$ROOT/runtime/outputs/diagnostics/trained-gpp-state-interventions-v1-runtime.json" \
  bash "$CODE_ROOT/scripts/hpc/run_fresh_start_one_day.sh" --backend gpu
run_command env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="$GPU_DEVICE" \
  PYTHON_MODULE=scripts.hpc.diagnose_trained_gpp_state_interventions RUN_TIMEOUT=75m \
  OUTPUT="$ROOT/runtime/outputs/diagnostics/trained-gpp-state-interventions-v1.json" \
  bash "$CODE_ROOT/scripts/hpc/run_fresh_start_one_day.sh" "${ARGS[@]}"
