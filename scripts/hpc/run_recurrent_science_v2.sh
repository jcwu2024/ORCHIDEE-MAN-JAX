#!/bin/bash
# One TRAIN T7 native reverse/JVP; no optimizer update or paid submission.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?declare reviewed checkout}
GPU_DEVICE=${GPU_DEVICE:?select idle free GPU}
PRINT_ONLY=${1:-}
[[ -z "$PRINT_ONLY" || "$PRINT_ONLY" == --dry-run ]] || exit 2
cd "$CODE_ROOT"
DIAG=$ROOT/runtime/outputs/diagnostics
ARGS=(--selection "$DIAG/recurrent-training-probe-v2-selection.json"
  --training-report "$ROOT/runtime/outputs/training/fresh-feedback-pool-15061139/report.json"
  --checkpoint "$ROOT/runtime/outputs/training/fresh-feedback-pool-15061139/report.update-001998.checkpoint.pkl"
  --expected-update 1998
  --expected-pool-protocol "$CODE_ROOT/manifests/coarse_graining/fresh_start_feedback_pool_v1.json"
  --recurrent-protocol "$CODE_ROOT/manifests/coarse_graining/recurrent_science_probe_v2.json"
  --stage-directory "$DIAG/recurrent-science-v2-stages" --horizon 7)
run_command() {
  if [[ "$PRINT_ONLY" == --dry-run ]]; then
    printf '%q ' "$@"; printf '\n'
  else
    "$@"
  fi
}
# This probe uses the deployed container and its own changed recurrent scalar.
run_command env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="$GPU_DEVICE" \
  PYTHON_MODULE=scripts.hpc.check_daily_training_runtime RUN_TIMEOUT=3m \
  OUTPUT="$DIAG/recurrent-science-v2-runtime.json" \
  bash "$CODE_ROOT/scripts/hpc/run_fresh_start_one_day.sh" --backend gpu
run_command env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="$GPU_DEVICE" \
  PYTHON_MODULE=scripts.hpc.probe_trained_checkpointed_grad RUN_TIMEOUT=65m \
  OUTPUT="$DIAG/recurrent-science-v2-reverse.json" \
  bash "$CODE_ROOT/scripts/hpc/run_fresh_start_one_day.sh" "${ARGS[@]}" --phase reverse
if [[ "$PRINT_ONLY" != --dry-run ]]; then
  # Inspect the small result before beginning another compilation. The native
  # gradient is already durable; no repeated gradient or Teacher payload scan.
  "$ROOT/.venvs/orcjax_cpu/bin/python" -c \
    'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get("status")=="passed" else 1)' \
    "$DIAG/recurrent-science-v2-reverse.json"
fi
run_command env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="$GPU_DEVICE" \
  PYTHON_MODULE=scripts.hpc.probe_trained_checkpointed_grad RUN_TIMEOUT=35m \
  OUTPUT="$DIAG/recurrent-science-v2-verify.json" \
  bash "$CODE_ROOT/scripts/hpc/run_fresh_start_one_day.sh" "${ARGS[@]}" --phase verify
