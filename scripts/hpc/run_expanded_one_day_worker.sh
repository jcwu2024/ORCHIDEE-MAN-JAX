#!/bin/bash
# One CPU worker: materialize assigned point-years from immutable v6 shards.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?declare reviewed expansion checkout}
OUTPUT_DIR=${OUTPUT_DIR:?declare this worker output}
PRINT_ONLY=${1:-}
[[ -z "$PRINT_ONLY" || "$PRINT_ONLY" == --dry-run ]] || exit 2
case "$CODE_ROOT" in "$ROOT"/runtime/checkouts/*) ;; *) exit 2;; esac
case "$OUTPUT_DIR" in "$ROOT"/runtime/outputs/training/*) ;; *) exit 2;; esac
if [[ "$PRINT_ONLY" != --dry-run ]]; then
  case "$(hostname -s)" in
    test0[1-4]|ibtest0[1-4]) ;;
    *) [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_PARTITION:-}" == cnall ]] || exit 2 ;;
  esac
fi
WORKER_INDEX=${SLURM_PROCID:-0}
WORKER_COUNT=${SLURM_NTASKS:-32}
FIRST=${START_ENTRY:-$((WORKER_INDEX * 17677 / WORKER_COUNT + 1))}
LAST=${END_ENTRY:-$(((WORKER_INDEX + 1) * 17677 / WORKER_COUNT))}
OUTPUT_DIR=${OUTPUT_DIR//WORKER_ID/$WORKER_INDEX}
cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ORCHIDEE_REPO_ROOT="$CODE_ROOT" ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime"
export ORCHIDEE_DATA_ROOT="$ROOT/runtime/data" ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets"
export ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs"
COMMAND=(timeout --signal=TERM "${RUN_TIMEOUT:-110m}" "$ROOT/.venvs/orcjax_cpu/bin/python"
  -m scripts.hpc.prepare_expanded_one_day
  --plan "$ROOT/runtime/outputs/diagnostics/one-day-expansion-plan-v1.json"
  --start-entry "$FIRST" --end-entry "$LAST"
  --dataset-manifest "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json"
  --acceptance-report "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json"
  --base-statistics "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json"
  --process-statistics "$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json"
  --fast-statistics "$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json"
  --output-dir "$OUTPUT_DIR")
if [[ "${RESUME:-0}" == 1 ]]; then
  COMMAND+=(--resume)
fi
if [[ "$PRINT_ONLY" == --dry-run ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
else
  "${COMMAND[@]}"
fi
