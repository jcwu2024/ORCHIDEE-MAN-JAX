#!/bin/bash
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:-$ROOT}
GPU_DEVICE=${GPU_DEVICE:?select an idle free GPU}
MODE=${1:?forward, ad-reverse or ad-verify}
PRINT_ONLY=${2:-}
[[ -z "$PRINT_ONLY" || "$PRINT_ONLY" == --dry-run ]] || exit 2
case "$(hostname -s)" in gln01|ibgln01|ibc13b02n04) ;; *) exit 2;; esac
cd "$CODE_ROOT"
BASE=$ROOT/runtime/outputs/diagnostics/trained-forward-ad-v1
ARGS=(--selection "$ROOT/runtime/outputs/diagnostics/trained-forward-selection-v1.json"
      --training-report "$ROOT/runtime/outputs/training/fresh-one-day-pool-15048786/report.json"
      --checkpoint "$ROOT/runtime/outputs/training/fresh-one-day-pool-15048786/report.checkpoint.pkl")
case "$MODE" in
  forward) MODULE=scripts.hpc.run_trained_forward_screen; LIMIT=75m;;
  ad-reverse|ad-verify)
    MODULE=scripts.hpc.probe_trained_checkpointed_grad
    ARGS+=(--phase "${MODE#ad-}" --stage-directory "$BASE-ad-stages")
    LIMIT=90m
    [[ "$MODE" != ad-verify ]] || LIMIT=60m
    ;;
  *) exit 2;;
esac
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
  OUTPUT="$BASE-$MODE-runtime.json" bash scripts/hpc/run_fresh_start_one_day.sh --backend gpu
run_command env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="$GPU_DEVICE" \
  PYTHON_MODULE="$MODULE" RUN_TIMEOUT="$LIMIT" OUTPUT="$BASE-$MODE.json" \
  bash scripts/hpc/run_fresh_start_one_day.sh "${ARGS[@]}"
