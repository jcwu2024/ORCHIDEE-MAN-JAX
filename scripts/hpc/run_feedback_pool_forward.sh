#!/bin/bash
# Free forward evaluation of an explicitly declared retained training snapshot.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?declare the reviewed evaluation checkout}
TRAINING_REPORT=${TRAINING_REPORT:?path to the training report}
CHECKPOINT=${CHECKPOINT:?path to the retained checkpoint}
EXPECTED_UPDATE=${EXPECTED_UPDATE:?500 or 1998}
GPU_DEVICE=${GPU_DEVICE:?select an idle free GPU}
OUTPUT=${OUTPUT:?unique forward-report JSON under runtime/outputs}
PRINT_ONLY=${1:-}
[[ -z "$PRINT_ONLY" || "$PRINT_ONLY" == --dry-run ]] || exit 2
case "$EXPECTED_UPDATE" in 500|1998) ;; *) exit 2;; esac
case "$CODE_ROOT" in "$ROOT"/runtime/checkouts/*) ;; *) exit 2;; esac
cd "$CODE_ROOT"
COMMAND=(env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="$GPU_DEVICE"
  PYTHON_MODULE=scripts.hpc.run_trained_forward_screen RUN_TIMEOUT=75m OUTPUT="$OUTPUT"
  bash "$CODE_ROOT/scripts/hpc/run_fresh_start_one_day.sh"
  --selection "$ROOT/runtime/outputs/diagnostics/trained-forward-selection-v1.json"
  --training-report "$TRAINING_REPORT" --checkpoint "$CHECKPOINT"
  --expected-update "$EXPECTED_UPDATE"
  --expected-pool-protocol "$CODE_ROOT/manifests/coarse_graining/fresh_start_feedback_pool_v1.json")
if [[ "$PRINT_ONLY" == --dry-run ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
else
  "${COMMAND[@]}"
fi
