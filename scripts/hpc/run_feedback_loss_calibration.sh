#!/bin/bash
# New shared objective: real-array CPU shapes, then bounded free GPU calibration.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?declare the diagnostic checkout}
MODE=${1:?shape or gpu}
PRINT_ONLY=${2:-}
[[ -z "$PRINT_ONLY" || "$PRINT_ONLY" == --dry-run ]] || exit 2
case "$CODE_ROOT" in "$ROOT"/runtime/checkouts/*) ;; *) exit 2;; esac
cd "$CODE_ROOT"
DIAG=$ROOT/runtime/outputs/diagnostics
ARGS=(--catalogue-report "$DIAG/one-day-catalogue-v1.json"
      --pool-protocol "$CODE_ROOT/manifests/coarse_graining/fresh_start_one_day_pool_v1.json")
run_command() {
  if [[ "$PRINT_ONLY" == --dry-run ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}
case "$MODE" in
  shape)
    if [[ "$PRINT_ONLY" != --dry-run ]]; then
      case "$(hostname -s)" in test0[1-4]|ibtest0[1-4]) ;; *) exit 2;; esac
    fi
    export PYTHONPATH="$CODE_ROOT" JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export ORCHIDEE_REPO_ROOT="$CODE_ROOT" ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime"
    export ORCHIDEE_DATA_ROOT="$ROOT/runtime/data" ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets"
    export ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs" PYTHONDONTWRITEBYTECODE=1
    run_command timeout --signal=TERM 15m "$ROOT/.venvs/orcjax_cpu/bin/python" \
      -m scripts.hpc.probe_feedback_loss_calibration --shape-only \
      --dataset-manifest "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json" \
      --acceptance-report "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json" \
      --base-statistics "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json" \
      --process-statistics "$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json" \
      --fast-statistics "$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json" \
      --prepared-manifest "$ROOT/runtime/outputs/training/target-oriented-prepared-1000-a243029/prepared_manifest.json" \
      --output "$DIAG/feedback-loss-calibration-v1-cpu-shape.json" "${ARGS[@]}"
    ;;
  gpu)
    GPU_DEVICE=${GPU_DEVICE:?select an idle free GPU}
    run_command env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="$GPU_DEVICE" \
      PYTHON_MODULE=scripts.hpc.check_daily_training_runtime RUN_TIMEOUT=3m \
      OUTPUT="$DIAG/feedback-loss-calibration-v1-runtime.json" \
      bash "$CODE_ROOT/scripts/hpc/run_fresh_start_one_day.sh" --backend gpu
    run_command env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="$GPU_DEVICE" \
      PYTHON_MODULE=scripts.hpc.probe_feedback_loss_calibration RUN_TIMEOUT=60m \
      OUTPUT="$DIAG/feedback-loss-calibration-v1.json" \
      bash "$CODE_ROOT/scripts/hpc/run_fresh_start_one_day.sh" "${ARGS[@]}"
    ;;
  *) exit 2;;
esac
