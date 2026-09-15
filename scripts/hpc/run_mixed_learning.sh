#!/bin/bash
# Shared metadata, CPU-shape, free integration/profile and production entry.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?declare reviewed checkout}
MODE=${1:?metadata, shape, integration, profile, checked-profile, precision-profile, or train}
shift
PRINT_ONLY=false
if [[ "${1:-}" == --dry-run ]]; then PRINT_ONLY=true; shift; fi
cd "$CODE_ROOT"
DIAG=$ROOT/runtime/outputs/diagnostics
MIXED_PROTOCOL=${MIXED_PROTOCOL:-$CODE_ROOT/manifests/coarse_graining/mixed_learning_v1.json}
MODEL_PROTOCOL=${MODEL_PROTOCOL:-$CODE_ROOT/manifests/coarse_graining/mixed_learning_model_v1.json}
RECURRENT_SCHEDULE=${RECURRENT_SCHEDULE:-$DIAG/mixed-learning-200-recurrent-schedule-v1.json}
ARGS=(--mixed-protocol "$MIXED_PROTOCOL"
  --catalogue-report "$ROOT/runtime/outputs/training/one-day-expansion-15061225/merge_report.json"
  --recurrent-schedule "$RECURRENT_SCHEDULE"
  --selection "$DIAG/trained-forward-selection-v1.json")
run_command() {
  if "$PRINT_ONLY"; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
case "$MODE" in
  metadata|shape)
    if ! "$PRINT_ONLY"; then
      case "$(hostname -s)" in test0[1-4]|ibtest0[1-4]) ;; *) exit 2;; esac
    fi
    export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1
    export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export ORCHIDEE_REPO_ROOT="$CODE_ROOT" ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime"
    export ORCHIDEE_DATA_ROOT="$ROOT/runtime/data" ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets"
    export ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs"
    if [[ "$MODE" == metadata ]]; then
      run_command timeout 10m "$ROOT/.venvs/orcjax_cpu/bin/python" -m scripts.hpc.mixed_learning_plan \
        "${ARGS[@]}" --output "${OUTPUT:-$DIAG/mixed-learning-v1-metadata.json}" "$@"
    else
      run_command timeout 20m "$ROOT/.venvs/orcjax_cpu/bin/python" -m scripts.hpc.run_mixed_learning \
        --shape-only "${ARGS[@]}" --protocol "$MODEL_PROTOCOL" \
        --dataset-manifest "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json" \
        --acceptance-report "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json" \
        --base-statistics "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json" \
        --process-statistics "$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json" \
        --fast-statistics "$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json" \
        --prepared-manifest "$ROOT/runtime/outputs/training/target-oriented-prepared-1000-a243029/prepared_manifest.json" \
        --output "${OUTPUT:-$DIAG/mixed-learning-v1-shape.json}" "$@"
    fi
    ;;
  integration|profile|checked-profile|precision-profile|train)
    if [[ "$MODE" == integration || "$MODE" == profile || "$MODE" == checked-profile || "$MODE" == precision-profile ]]; then
      if ! "$PRINT_ONLY"; then
        case "$(hostname -s)" in gln01|ibgln01|ibc13b02n04) ;; *) exit 2;; esac
      fi
      if [[ "$MODE" == profile || "$MODE" == checked-profile || "$MODE" == precision-profile ]]; then
        ARGS+=(--profile-checkpoint "$ROOT/runtime/outputs/training/mixed-learning-v1/report.update-000200.checkpoint.pkl"
          --profile-training-report "$ROOT/runtime/outputs/training/mixed-learning-v1/report.json"
          --profile-repeats 3)
        DEFAULT_OUTPUT=$DIAG/mixed-learning-profile-v1.json
        DEFAULT_TIMEOUT=40m
        if [[ "$MODE" == checked-profile ]]; then
          ARGS+=(--compare-checked-recurrent)
          DEFAULT_OUTPUT=$DIAG/mixed-learning-checked-profile-v1.json
          DEFAULT_TIMEOUT=55m
          if [[ -n "${CHECKED_REFERENCE_REPORT:-}" ]]; then
            ARGS+=(--checked-reference-report "$CHECKED_REFERENCE_REPORT")
          else
            run_command env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="${GPU_DEVICE:?select allocated GPU}" \
              PYTHON_MODULE=scripts.hpc.check_daily_training_runtime RUN_TIMEOUT=3m \
              OUTPUT="$DIAG/mixed-learning-checked-profile-v1-runtime.json" \
              bash "$CODE_ROOT/scripts/hpc/run_fresh_start_one_day.sh" --backend gpu
          fi
        elif [[ "$MODE" == precision-profile ]]; then
          : "${PRECISION_STAGE:?daily, T7, T30, or full}"
          ARGS+=(--precision-stage "$PRECISION_STAGE"
            --attention-score-dtype "${ATTENTION_SCORE_DTYPE:-legacy}")
          DEFAULT_OUTPUT=$DIAG/mixed-learning-attention-${ATTENTION_SCORE_DTYPE:-legacy}-${PRECISION_STAGE}-v1.json
          DEFAULT_TIMEOUT=50m
          if [[ -n "${PRECISION_REFERENCE_REPORT:-}" ]]; then
            ARGS+=(--precision-reference-report "$PRECISION_REFERENCE_REPORT")
          fi
          if [[ -n "${PRECISION_DAILY_REFERENCE:-}" ]]; then
            ARGS+=(--precision-daily-reference "$PRECISION_DAILY_REFERENCE")
          fi
        else
          ARGS+=(--profile-trace)
        fi
      else
        ARGS+=(--integration-updates 1)
        DEFAULT_OUTPUT=$DIAG/mixed-learning-v1-integration.json
        DEFAULT_TIMEOUT=90m
      fi
    else
      if ! "$PRINT_ONLY"; then
        [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_PARTITION:-}" == gnall ]] || exit 2
      fi
      DEFAULT_OUTPUT=$ROOT/runtime/outputs/training/mixed-learning-v1/report.json
      DEFAULT_TIMEOUT=355m
    fi
    run_command env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="${GPU_DEVICE:?select allocated GPU}" \
      GPU_MEMORY_FRACTION=0.85 PYTHON_MODULE=scripts.hpc.run_mixed_learning \
      RUN_TIMEOUT="${RUN_TIMEOUT:-$DEFAULT_TIMEOUT}" OUTPUT="${OUTPUT:-$DEFAULT_OUTPUT}" \
      bash "$CODE_ROOT/scripts/hpc/run_fresh_start_one_day.sh" \
      --protocol "$MODEL_PROTOCOL" "${ARGS[@]}" "$@"
    ;;
  *) exit 2 ;;
esac
