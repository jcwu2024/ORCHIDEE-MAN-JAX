#!/bin/bash
# Free CPU metadata selection or four-point-year data preparation; no model run.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?declare separate expansion checkout}
MODE=${1:?plan or pilot}
PRINT_ONLY=${2:-}
case "$CODE_ROOT" in "$ROOT"/runtime/checkouts/*) ;; *) exit 2;; esac
[[ -z "$PRINT_ONLY" || "$PRINT_ONLY" == --dry-run ]] || exit 2
if [[ "$PRINT_ONLY" != --dry-run ]]; then
  case "$(hostname -s)" in test0[1-4]|ibtest0[1-4]) ;; *) exit 2;; esac
fi
cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ORCHIDEE_REPO_ROOT="$CODE_ROOT" ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime"
export ORCHIDEE_DATA_ROOT="$ROOT/runtime/data" ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets"
export ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs"
PLAN="$ROOT/runtime/outputs/diagnostics/one-day-expansion-plan-v1.json"
COMMAND=(timeout --signal=TERM 15m "$ROOT/.venvs/orcjax_cpu/bin/python")
case "$MODE" in
  plan)
    COMMAND+=(-m scripts.hpc.prepare_one_day_expansion_plan
      --dataset-manifest "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json"
      --catalogue "$ROOT/runtime/outputs/diagnostics/one-day-catalogue-v1.catalogue.json.gz"
      --selection "$ROOT/runtime/outputs/diagnostics/trained-forward-selection-v1.json"
      --seed 20260913 --rows-per-point-year 16 --output "$PLAN") ;;
  pilot)
    COMMAND+=(-m scripts.hpc.prepare_expanded_one_day --plan "$PLAN"
      --entry-ids 1,5893,11785,17677
      --dataset-manifest "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json"
      --acceptance-report "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json"
      --base-statistics "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json"
      --process-statistics "$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json"
      --fast-statistics "$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json"
      --output-dir "$ROOT/runtime/outputs/training/one-day-expansion-pilot-v1") ;;
  *) exit 2 ;;
esac
if [[ "$PRINT_ONLY" == --dry-run ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
else
  "${COMMAND[@]}"
fi
