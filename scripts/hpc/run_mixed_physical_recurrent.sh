#!/bin/bash
# Small free-node mixed daily/T7 or T30 admission, no paid submission path.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?declare reviewed checkout}
MODE=${1:?shape, forward, or gpu}
PRINT_ONLY=${2:-}
[[ -z "$PRINT_ONLY" || "$PRINT_ONLY" == --dry-run ]] || exit 2
HORIZON=${HORIZON:-7}
case "$HORIZON" in
  7) STEM=mixed-physical-recurrent-v1; MIXED_PROTOCOL=mixed_physical_recurrent_admission_v1.json ;;
  30) STEM=mixed-physical-recurrent-t30-v1; MIXED_PROTOCOL=mixed_physical_recurrent_t30_admission_v1.json ;;
  *) exit 2 ;;
esac
cd "$CODE_ROOT"
DIAG=$ROOT/runtime/outputs/diagnostics
ARGS=(--protocol "$CODE_ROOT/manifests/coarse_graining/physical_feedback_model_v2.json"
  --pool-protocol "$CODE_ROOT/manifests/coarse_graining/physical_feedback_probe_pool_v2.json"
  --admission-config "$CODE_ROOT/manifests/coarse_graining/physical_feedback_update_admission_v2.json"
  --mixed-protocol "$CODE_ROOT/manifests/coarse_graining/$MIXED_PROTOCOL"
  --catalogue-report "$ROOT/runtime/outputs/training/one-day-expansion-15061225/merge_report.json"
  --selection "$DIAG/recurrent-training-probe-v2-selection.json")
run_command() {
  if [[ "$PRINT_ONLY" == --dry-run ]]; then
    printf '%q ' "$@"; printf '\n'
  else
    "$@"
  fi
}
case "$MODE" in
  shape)
    if [[ "$PRINT_ONLY" != --dry-run ]]; then
      case "$(hostname -s)" in test0[1-4]|ibtest0[1-4]) ;; *) exit 2;; esac
    fi
    export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1
    export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export ORCHIDEE_REPO_ROOT="$CODE_ROOT" ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime"
    export ORCHIDEE_DATA_ROOT="$ROOT/runtime/data" ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets"
    export ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs"
    run_command timeout --signal=TERM 20m "$ROOT/.venvs/orcjax_cpu/bin/python" \
      -m scripts.hpc.probe_mixed_physical_recurrent --shape-only \
      --dataset-manifest "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json" \
      --acceptance-report "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json" \
      --base-statistics "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json" \
      --process-statistics "$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json" \
      --fast-statistics "$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json" \
      --prepared-manifest "$ROOT/runtime/outputs/training/target-oriented-prepared-1000-a243029/prepared_manifest.json" \
      --output "$DIAG/${STEM}-shape.json" "${ARGS[@]}"
    ;;
  forward|gpu)
    if [[ "$PRINT_ONLY" != --dry-run ]]; then
      case "$(hostname -s)" in gln01|ibgln01|ibc13b02n04) ;; *) exit 2;; esac
    fi
    if [[ "$MODE" == forward ]]; then ARGS+=(--forward-only); fi
    if [[ "$MODE" == gpu ]]; then
      run_command env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="${GPU_DEVICE:?select idle free GPU}" \
        PYTHON_MODULE=scripts.hpc.check_daily_training_runtime RUN_TIMEOUT=3m \
        OUTPUT="$DIAG/${STEM}-runtime.json" \
        bash "$CODE_ROOT/scripts/hpc/run_fresh_start_one_day.sh" --backend gpu
    fi
    run_command env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="${GPU_DEVICE:?select idle free GPU}" \
      GPU_MEMORY_FRACTION=0.85 PYTHON_MODULE=scripts.hpc.probe_mixed_physical_recurrent RUN_TIMEOUT=90m \
      OUTPUT="$DIAG/${STEM}-${MODE}.json" \
      bash "$CODE_ROOT/scripts/hpc/run_fresh_start_one_day.sh" "${ARGS[@]}"
    ;;
  *) exit 2 ;;
esac
